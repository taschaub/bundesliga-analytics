import pandas as pd
import requests
import os
from typing import List
from datetime import datetime

def download_bundesliga_data(seasons: List[str], output_path: str = 'data/bundesliga_matches.csv'):
    """
    Download and combine Bundesliga data from multiple seasons.
    Each season should have exactly 34 matchdays (9 matches each).
    """
    os.makedirs('data', exist_ok=True)
    base_url = "https://www.football-data.co.uk/mmz4281"
    dfs = []
    
    for season in seasons:
        url = f"{base_url}/{season}/D1.csv"
        print(f"Downloading season {season}...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Save raw file
            raw_file_path = f"data/D1_{season}.csv"
            with open(raw_file_path, "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded season {season}")
            
            # Read the CSV
            df = pd.read_csv(raw_file_path)
            
            # Add season information
            season_start_year = 2000 + int(season[:2])
            df['Season'] = f"{season_start_year}/{season_start_year+1}"
            
            # Convert date and ensure it's in the correct season
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            
            # Filter out matches that don't belong in this season
            season_start = pd.Timestamp(f"{season_start_year}-07-01")
            season_end = pd.Timestamp(f"{season_start_year+1}-06-30")
            df = df[
                (df['Date'] >= season_start) & 
                (df['Date'] <= season_end)
            ]
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Assign matchdays (every 9 matches = 1 matchday)
            df['Matchday'] = (df.index % len(df) // 9) + 1
            
            # Verify we have the expected number of matches
            n_matches = len(df)
            if n_matches != 306:  # 34 matchdays * 9 matches
                print(f"Warning: Found {n_matches} matches in season {season} (expected 306)")
            
            # Select relevant columns
            relevant_columns = [
                'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                'B365H', 'B365D', 'B365A',  # Betting odds
                'HS', 'AS',  # Shots
                'HST', 'AST',  # Shots on target
                'HC', 'AC',  # Corners
                'HF', 'AF',  # Fouls
                'HY', 'AY',  # Yellow cards
                'HR', 'AR',  # Red cards
                'Season', 'Matchday'
            ]
            
            # Only keep columns that exist in the dataset
            existing_columns = [col for col in relevant_columns if col in df.columns]
            df = df[existing_columns]
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error processing season {season}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(['Season', 'Date'])
        
        # Verify final dataset
        print("\nData verification:")
        for season in combined_df['Season'].unique():
            season_data = combined_df[combined_df['Season'] == season]
            print(f"\nSeason {season}:")
            print(f"Total matches: {len(season_data)}")
            print(f"Number of matchdays: {season_data['Matchday'].max()}")
            print(f"Date range: {season_data['Date'].min()} to {season_data['Date'].max()}")
            
            # Check matches per matchday
            matches_per_day = season_data.groupby('Matchday').size()
            if not all(count == 9 for count in matches_per_day):
                print("Warning: Irregular number of matches in matchdays:")
                print(matches_per_day[matches_per_day != 9])
        
        # Save both full and minimal versions
        combined_df.to_csv(output_path.replace('.csv', '_full.csv'), index=False)
        
        minimal_columns = ['Date', 'Season', 'Matchday', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        combined_df[minimal_columns].to_csv(output_path, index=False)
        
        print(f"\nData saved to {output_path}")
        print(f"Full data saved to {output_path.replace('.csv', '_full.csv')}")
        
    else:
        print("No data was downloaded!")

if __name__ == "__main__":
    seasons = ['1819', '1920', '2021', '2122', '2223']
    download_bundesliga_data(seasons)
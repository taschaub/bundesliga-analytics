import pandas as pd
import requests
import os
from typing import List
from datetime import datetime

def download_bundesliga_data(seasons: List[str]):
    """
    Download and combine Bundesliga data from multiple seasons.
    Each season should have exactly 34 matchdays (9 matches each).
    """
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    base_url = "https://www.football-data.co.uk/mmz4281"
    dfs = []
    
    for season in seasons:
        url = f"{base_url}/{season}/D1.csv"
        print(f"Downloading season {season}...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Save raw file
            raw_file_path = f"data/raw/D1_{season}.csv"
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
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error processing season {season}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(['Season', 'Date'])
        
        # Save full version
        full_path = 'data/processed/bundesliga_matches_full.csv'
        combined_df.to_csv(full_path, index=False)
        
        # Save minimal version
        minimal_columns = ['Date', 'Season', 'Matchday', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        minimal_path = 'data/processed/bundesliga_matches.csv'
        combined_df[minimal_columns].to_csv(minimal_path, index=False)
        
        print("\nData saved:")
        print(f"Full dataset: {full_path}")
        print(f"Minimal dataset: {minimal_path}")
        
        # Print verification info
        print("\nDataset verification:")
        for season in combined_df['Season'].unique():
            season_data = combined_df[combined_df['Season'] == season]
            print(f"\nSeason {season}:")
            print(f"Total matches: {len(season_data)}")
            print(f"Number of matchdays: {season_data['Matchday'].max()}")
            print(f"Date range: {season_data['Date'].min()} to {season_data['Date'].max()}")
    
    else:
        print("No data was downloaded!")

if __name__ == "__main__":
    # Download more seasons (10 years of data)
    seasons = [
        '1314', '1415', '1516', '1617', '1718',  # 2013-2018
        '1819', '1920', '2021', '2122', '2223'   # 2018-2023
    ]
    download_bundesliga_data(seasons)
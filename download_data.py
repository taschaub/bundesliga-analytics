import pandas as pd
import requests
import os
from typing import List
from datetime import datetime

def download_bundesliga_data(seasons: List[str], output_path: str = 'data/bundesliga_matches.csv'):
    """
    Download and combine Bundesliga data from multiple seasons.
    
    Args:
        seasons: List of seasons in format ['2122', '2223'] etc.
        output_path: Where to save the combined CSV
    """
    os.makedirs('data', exist_ok=True)
    base_url = "https://www.football-data.co.uk/mmz4281"
    dfs = []
    
    for season in seasons:
        url = f"{base_url}/{season}/D1.csv"
        print(f"Downloading season {season}...")
        
        try:
            df = pd.read_csv(url)
            
            # Add season information
            season_start_year = 2000 + int(season[:2])
            df['Season'] = f"{season_start_year}/{season_start_year+1}"
            
            # Select relevant columns (expanding from before)
            relevant_columns = [
                'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
                'B365H', 'B365D', 'B365A',  # Betting odds
                'HS', 'AS',  # Shots
                'HST', 'AST',  # Shots on target
                'HC', 'AC',  # Corners
                'HF', 'AF',  # Fouls
                'HY', 'AY',  # Yellow cards
                'HR', 'AR',  # Red cards
                'Season'
            ]
            
            # Only keep columns that exist in the dataset
            existing_columns = [col for col in relevant_columns if col in df.columns]
            df = df[existing_columns]
            
            # Add matchday information
            df = df.sort_values('Date')
            df['Matchday'] = ((df.groupby('Season').cumcount()) // 9) + 1  # 9 games per matchday
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error downloading season {season}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Convert date and sort
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True)
        combined_df = combined_df.sort_values(['Season', 'Matchday', 'Date'])
        
        # Save both full and minimal versions
        combined_df.to_csv(output_path.replace('.csv', '_full.csv'), index=False)
        
        # Save minimal version with essential columns
        minimal_columns = ['Date', 'Season', 'Matchday', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        combined_df[minimal_columns].to_csv(output_path, index=False)
        
        print(f"Data saved to {output_path}")
        print(f"Full data saved to {output_path.replace('.csv', '_full.csv')}")
        print(f"Total matches: {len(combined_df)}")
        
        # Print some basic statistics
        print("\nSeasons in dataset:")
        print(combined_df['Season'].value_counts().sort_index())
        print("\nTeams in dataset:")
        print(sorted(pd.concat([combined_df['HomeTeam'], combined_df['AwayTeam']]).unique()))
        
    else:
        print("No data was downloaded!")

if __name__ == "__main__":
    # Download last 5 seasons (adjust as needed)
    seasons = ['1819', '1920', '2021', '2122', '2223']
    download_bundesliga_data(seasons) 
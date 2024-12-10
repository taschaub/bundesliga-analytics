import pandas as pd
import requests
import os
from typing import List

def download_bundesliga_data(seasons: List[str], output_path: str = 'data/bundesliga_matches.csv'):
    """
    Download and combine Bundesliga data from multiple seasons.
    
    Args:
        seasons: List of seasons in format ['2122', '2223'] etc.
        output_path: Where to save the combined CSV
    """
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Base URL for football-data.co.uk
    base_url = "https://www.football-data.co.uk/mmz4281"
    
    # List to store dataframes
    dfs = []
    
    for season in seasons:
        url = f"{base_url}/{season}/D1.csv"
        print(f"Downloading season {season}...")
        
        try:
            df = pd.read_csv(url)
            
            # Select and rename only the columns we need
            df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error downloading season {season}: {e}")
    
    # Combine all seasons
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by date
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True)
        combined_df = combined_df.sort_values('Date')
        
        # Save to CSV
        combined_df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        print(f"Total matches: {len(combined_df)}")
    else:
        print("No data was downloaded!")

if __name__ == "__main__":
    # Download last 30 seasons
    seasons = [str(year)[-2:] + str(year + 1)[-2:] for year in range(1990, 2020)]
    download_bundesliga_data(seasons) 
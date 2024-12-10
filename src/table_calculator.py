import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

class TableCalculator:
    def __init__(self, df: pd.DataFrame):
        """Initialize with match data DataFrame."""
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
    
    def calculate_table(self, 
                       season: str,
                       date: pd.Timestamp) -> pd.DataFrame:
        """
        Calculate league table up to a specific date.
        
        Args:
            season: Season to calculate table for (e.g., "2022/2023")
            date: Include all matches up to and including this date
        """
        # Filter for season and date
        season_data = self.df[self.df['Season'] == season].copy()
        matches = season_data[season_data['Date'] <= date]
            
        # Initialize table with all teams from the season
        teams = pd.concat([season_data['HomeTeam'], season_data['AwayTeam']]).unique()
        table = pd.DataFrame(index=teams)
        table['MP'] = 0  # Matches played
        table['W'] = 0   # Wins
        table['D'] = 0   # Draws
        table['L'] = 0   # Losses
        table['GF'] = 0  # Goals for
        table['GA'] = 0  # Goals against
        table['GD'] = 0  # Goal difference
        table['Pts'] = 0 # Points
        
        # Calculate statistics
        for _, match in matches.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Update matches played
            table.loc[home_team, 'MP'] += 1
            table.loc[away_team, 'MP'] += 1
            
            # Update goals
            table.loc[home_team, 'GF'] += match['FTHG']
            table.loc[home_team, 'GA'] += match['FTAG']
            table.loc[away_team, 'GF'] += match['FTAG']
            table.loc[away_team, 'GA'] += match['FTHG']
            
            # Update results
            if match['FTHG'] > match['FTAG']:  # Home win
                table.loc[home_team, 'W'] += 1
                table.loc[away_team, 'L'] += 1
                table.loc[home_team, 'Pts'] += 3
            elif match['FTHG'] < match['FTAG']:  # Away win
                table.loc[away_team, 'W'] += 1
                table.loc[home_team, 'L'] += 1
                table.loc[away_team, 'Pts'] += 3
            else:  # Draw
                table.loc[home_team, 'D'] += 1
                table.loc[away_team, 'D'] += 1
                table.loc[home_team, 'Pts'] += 1
                table.loc[away_team, 'Pts'] += 1
        
        # Calculate goal difference
        table['GD'] = table['GF'] - table['GA']
        
        # Sort table
        table = table.sort_values(['Pts', 'GD', 'GF'], ascending=[False, False, False])
        table.index.name = 'Team'
        
        # Add position column
        table.insert(0, 'Pos', range(1, len(table) + 1))
        
        return table

    def get_matchday_matches(self, season: str, matchday: int) -> pd.DataFrame:
        """Get all matches for a specific matchday in a season."""
        return self.df[
            (self.df['Season'] == season) & 
            (self.df['Matchday'] == matchday)
        ].sort_values('Date').copy() 
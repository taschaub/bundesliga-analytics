import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize FeatureEngineer with DataFrame."""
        self.df = df.copy()
        self.features = []
        
    def calculate_league_table(self, date: pd.Timestamp) -> pd.DataFrame:
        """Calculate the league table up to a given date."""
        # Get all matches before the given date
        past_matches = self.df[self.df['Date'] < date].copy()
        
        # Debugging: Print information about the data
        print(f"\nCalculating table for date: {date}")
        print(f"Number of past matches: {len(past_matches)}")
        
        # If no past matches, return empty DataFrame with required columns
        if len(past_matches) == 0:
            empty_table = pd.DataFrame(columns=[
                'Points', 'GF', 'GA', 'GD', 'Matches', 'Wins', 
                'Draws', 'Losses', 'PPG', 'Position'
            ])
            # Add all teams with default values
            all_teams = pd.concat([self.df['HomeTeam'], self.df['AwayTeam']]).unique()
            empty_table = pd.DataFrame(
                {col: 0 for col in empty_table.columns},
                index=all_teams
            )
            empty_table['Position'] = range(1, len(all_teams) + 1)
            empty_table['PPG'] = 0
            return empty_table
        
        # Initialize table dictionary
        table = {}
        
        # Process each match
        for _, match in past_matches.iterrows():
            # Initialize teams if not in table
            for team in [match['HomeTeam'], match['AwayTeam']]:
                if team not in table:
                    table[team] = {
                        'Points': 0,
                        'GF': 0,  # Goals For
                        'GA': 0,  # Goals Against
                        'GD': 0,  # Goal Difference
                        'Matches': 0,
                        'Wins': 0,
                        'Draws': 0,
                        'Losses': 0
                    }
            
            # Update home team stats
            table[match['HomeTeam']]['GF'] += match['FTHG']
            table[match['HomeTeam']]['GA'] += match['FTAG']
            table[match['HomeTeam']]['Matches'] += 1
            
            # Update away team stats
            table[match['AwayTeam']]['GF'] += match['FTAG']
            table[match['AwayTeam']]['GA'] += match['FTHG']
            table[match['AwayTeam']]['Matches'] += 1
            
            # Update points and results
            if match['FTHG'] > match['FTAG']:  # Home win
                table[match['HomeTeam']]['Points'] += 3
                table[match['HomeTeam']]['Wins'] += 1
                table[match['AwayTeam']]['Losses'] += 1
            elif match['FTHG'] < match['FTAG']:  # Away win
                table[match['AwayTeam']]['Points'] += 3
                table[match['AwayTeam']]['Wins'] += 1
                table[match['HomeTeam']]['Losses'] += 1
            else:  # Draw
                table[match['HomeTeam']]['Points'] += 1
                table[match['AwayTeam']]['Points'] += 1
                table[match['HomeTeam']]['Draws'] += 1
                table[match['AwayTeam']]['Draws'] += 1
            
            # Update goal differences
            for team in table:
                table[team]['GD'] = table[team]['GF'] - table[team]['GA']
        
        # Convert to DataFrame and sort
        table_df = pd.DataFrame.from_dict(table, orient='index')
        
        # Add any missing teams with zero values
        all_teams = pd.concat([self.df['HomeTeam'], self.df['AwayTeam']]).unique()
        missing_teams = set(all_teams) - set(table_df.index)
        for team in missing_teams:
            table_df.loc[team] = {
                'Points': 0, 'GF': 0, 'GA': 0, 'GD': 0,
                'Matches': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0
            }
        
        table_df['PPG'] = table_df['Points'] / table_df['Matches'].clip(lower=1)
        table_df = table_df.sort_values(['Points', 'GD', 'GF'], ascending=[False, False, False])
        table_df['Position'] = range(1, len(table_df) + 1)
        
        return table_df
    
    def add_table_features(self) -> None:
        """Add features based on league table position and form."""
        for idx, match in self.df.iterrows():
            # Calculate table just before this match
            table = self.calculate_league_table(match['Date'])
            
            # Add position-based features
            self.df.loc[idx, 'HomeTeamPosition'] = table.loc[match['HomeTeam'], 'Position']
            self.df.loc[idx, 'AwayTeamPosition'] = table.loc[match['AwayTeam'], 'Position']
            self.df.loc[idx, 'PositionDiff'] = table.loc[match['HomeTeam'], 'Position'] - table.loc[match['AwayTeam'], 'Position']
            
            # Add form-based features
            self.df.loc[idx, 'HomeTeamPPG'] = table.loc[match['HomeTeam'], 'PPG']
            self.df.loc[idx, 'AwayTeamPPG'] = table.loc[match['AwayTeam'], 'PPG']
            
            # Add goal-based features
            home_matches = table.loc[match['HomeTeam'], 'Matches']
            away_matches = table.loc[match['AwayTeam'], 'Matches']
            
            # Ensure at least 1 match for division
            home_matches = max(1, home_matches)
            away_matches = max(1, away_matches)
            
            self.df.loc[idx, 'HomeTeamGDPerGame'] = table.loc[match['HomeTeam'], 'GD'] / home_matches
            self.df.loc[idx, 'AwayTeamGDPerGame'] = table.loc[match['AwayTeam'], 'GD'] / away_matches
        
        self.features.extend([
            'HomeTeamPosition', 'AwayTeamPosition', 'PositionDiff',
            'HomeTeamPPG', 'AwayTeamPPG',
            'HomeTeamGDPerGame', 'AwayTeamGDPerGame'
        ])

    def calculate_team_form(self, n_matches: int = 5) -> None:
        """Calculate rolling averages for goals scored and conceded."""
        teams = pd.concat([self.df['HomeTeam'], self.df['AwayTeam']]).unique()
        
        for team in teams:
            # Home form
            home_mask = self.df['HomeTeam'] == team
            self.df.loc[home_mask, 'HomeTeamFormGF'] = (
                self.df.loc[home_mask, 'FTHG'].rolling(n_matches, min_periods=1).mean()
            )
            self.df.loc[home_mask, 'HomeTeamFormGA'] = (
                self.df.loc[home_mask, 'FTAG'].rolling(n_matches, min_periods=1).mean()
            )
            
            # Away form
            away_mask = self.df['AwayTeam'] == team
            self.df.loc[away_mask, 'AwayTeamFormGF'] = (
                self.df.loc[away_mask, 'FTAG'].rolling(n_matches, min_periods=1).mean()
            )
            self.df.loc[away_mask, 'AwayTeamFormGA'] = (
                self.df.loc[away_mask, 'FTHG'].rolling(n_matches, min_periods=1).mean()
            )
        
        self.features.extend(['HomeTeamFormGF', 'HomeTeamFormGA', 
                            'AwayTeamFormGF', 'AwayTeamFormGA'])
    
    def get_feature_matrix(self) -> Tuple[pd.DataFrame, List[str]]:
        """Return feature matrix and list of feature names."""
        print("Calculating team form...")
        self.calculate_team_form()
        print("Adding table features...")
        self.add_table_features()
        return self.df[self.features], self.features
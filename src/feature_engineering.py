import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize with historical match data."""
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.team_ratings = {}
        self.league_patterns = {}
        
    def _calculate_team_ratings(self):
        """Calculate dynamic team ratings based on historical performance."""
        teams = pd.concat([self.df['HomeTeam'], self.df['AwayTeam']]).unique()
        self.team_ratings = {}
        
        # Process matches chronologically
        for idx, match in self.df.sort_values('Date').iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            match_date = match['Date']
            
            # Initialize ratings if not exists
            for team in [home_team, away_team]:
                if team not in self.team_ratings:
                    self.team_ratings[team] = {'overall': 1500, 'home': 1500, 'away': 1500}
            
            # Calculate ratings based on historical matches
            historical_matches = self.df[self.df['Date'] < match_date]
            if not historical_matches.empty:
                for team in [home_team, away_team]:
                    team_matches = historical_matches[
                        (historical_matches['HomeTeam'] == team) | 
                        (historical_matches['AwayTeam'] == team)
                    ]
                    
                    if not team_matches.empty:
                        # Calculate points
                        home_matches = team_matches[team_matches['HomeTeam'] == team]
                        away_matches = team_matches[team_matches['AwayTeam'] == team]
                        
                        home_points = (home_matches['FTHG'] > home_matches['FTAG']).sum() * 3 + \
                                    (home_matches['FTHG'] == home_matches['FTAG']).sum()
                        away_points = (away_matches['FTAG'] > away_matches['FTHG']).sum() * 3 + \
                                    (away_matches['FTHG'] == away_matches['FTAG']).sum()
                        
                        total_matches = len(home_matches) + len(away_matches)
                        if total_matches > 0:
                            ppg = (home_points + away_points) / total_matches
                            home_ppg = home_points / max(len(home_matches), 1)
                            away_ppg = away_points / max(len(away_matches), 1)
                            
                            # Update ratings (scale between 1000-2000)
                            self.team_ratings[team]['overall'] = 1000 + (ppg * 250)
                            self.team_ratings[team]['home'] = 1000 + (home_ppg * 250)
                            self.team_ratings[team]['away'] = 1000 + (away_ppg * 250)
            
            # Store ratings in dataframe
            self.df.at[idx, 'HomeTeamRating'] = self.team_ratings[home_team]['overall']
            self.df.at[idx, 'AwayTeamRating'] = self.team_ratings[away_team]['overall']
            self.df.at[idx, 'HomeTeamHomeRating'] = self.team_ratings[home_team]['home']
            self.df.at[idx, 'AwayTeamAwayRating'] = self.team_ratings[away_team]['away']
    
    def _calculate_league_patterns(self):
        """Calculate league-wide patterns using only historical data."""
        # Process matches chronologically
        for idx, match in self.df.sort_values('Date').iterrows():
            match_date = match['Date']
            historical_matches = self.df[self.df['Date'] < match_date]
            
            if not historical_matches.empty:
                # Calculate historical form for home team
                home_team = match['HomeTeam']
                home_historical = historical_matches[
                    (historical_matches['HomeTeam'] == home_team) | 
                    (historical_matches['AwayTeam'] == home_team)
                ].sort_values('Date')
                
                # Calculate historical form for away team
                away_team = match['AwayTeam']
                away_historical = historical_matches[
                    (historical_matches['HomeTeam'] == away_team) | 
                    (historical_matches['AwayTeam'] == away_team)
                ].sort_values('Date')
                
                # Calculate form features
                for n_matches in [3, 5, 10]:
                    # Home team form
                    if len(home_historical) >= n_matches:
                        recent_home = home_historical.tail(n_matches)
                        home_wins = sum(
                            ((recent_home['HomeTeam'] == home_team) & (recent_home['FTR'] == 'H')) |
                            ((recent_home['AwayTeam'] == home_team) & (recent_home['FTR'] == 'A'))
                        )
                        self.df.at[idx, f'HomeForm{n_matches}'] = home_wins / n_matches
                    else:
                        self.df.at[idx, f'HomeForm{n_matches}'] = 0.5
                    
                    # Away team form
                    if len(away_historical) >= n_matches:
                        recent_away = away_historical.tail(n_matches)
                        away_wins = sum(
                            ((recent_away['HomeTeam'] == away_team) & (recent_away['FTR'] == 'H')) |
                            ((recent_away['AwayTeam'] == away_team) & (recent_away['FTR'] == 'A'))
                        )
                        self.df.at[idx, f'AwayForm{n_matches}'] = away_wins / n_matches
                    else:
                        self.df.at[idx, f'AwayForm{n_matches}'] = 0.5
    
    def _add_team_specific_features(self):
        """Add features specific to team matchups using only historical data."""
        # Process matches chronologically
        for idx, match in self.df.sort_values('Date').iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            match_date = match['Date']
            
            # Get previous meetings
            h2h_matches = self.df[
                (
                    ((self.df['HomeTeam'] == home_team) & (self.df['AwayTeam'] == away_team)) |
                    ((self.df['HomeTeam'] == away_team) & (self.df['AwayTeam'] == home_team))
                ) &
                (self.df['Date'] < match_date)
            ].sort_values('Date', ascending=False)
            
            # Calculate H2H stats
            if len(h2h_matches) > 0:
                home_wins = sum((h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTR'] == 'H')) + \
                          sum((h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTR'] == 'A'))
                self.df.at[idx, 'H2HHomeWinRate'] = home_wins / len(h2h_matches)
            else:
                self.df.at[idx, 'H2HHomeWinRate'] = 0.5
    
    def _add_sequence_features(self):
        """Add sequential features using only historical data."""
        # Process matches chronologically
        for idx, match in self.df.sort_values('Date').iterrows():
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get historical matches for both teams
            for team, prefix in [(home_team, 'Home'), (away_team, 'Away')]:
                historical = self.df[
                    (
                        ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
                        (self.df['Date'] < match_date)
                    )
                ].sort_values('Date')
                
                if not historical.empty:
                    # Calculate recent performance metrics
                    recent_matches = historical.tail(5)
                    goals_scored = sum(
                        np.where(recent_matches['HomeTeam'] == team,
                                recent_matches['FTHG'],
                                recent_matches['FTAG'])
                    )
                    goals_conceded = sum(
                        np.where(recent_matches['HomeTeam'] == team,
                                recent_matches['FTAG'],
                                recent_matches['FTHG'])
                    )
                    
                    self.df.at[idx, f'{prefix}RecentGoalsScored'] = goals_scored / len(recent_matches)
                    self.df.at[idx, f'{prefix}RecentGoalsConceded'] = goals_conceded / len(recent_matches)
                else:
                    self.df.at[idx, f'{prefix}RecentGoalsScored'] = 0
                    self.df.at[idx, f'{prefix}RecentGoalsConceded'] = 0
    
    def engineer_features(self) -> pd.DataFrame:
        """Main method to engineer all features."""
        print("Calculating team ratings...")
        self._calculate_team_ratings()
        
        print("Calculating league patterns...")
        self._calculate_league_patterns()
        
        print("Adding team-specific features...")
        self._add_team_specific_features()
        
        print("Adding sequence features...")
        self._add_sequence_features()
        
        # Fill NaN values using forward fill
        self.df = self.df.ffill()
        
        # Create target variable (0: Away Win, 1: Draw, 2: Home Win)
        self.df['Target'] = np.where(self.df['FTHG'] > self.df['FTAG'], 2,
                                   np.where(self.df['FTHG'] == self.df['FTAG'], 1, 0))
        
        # Ensure all features are numeric
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df = self.df[numeric_columns]
        
        return self.df
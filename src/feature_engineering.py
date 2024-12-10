import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize FeatureEngineer with DataFrame."""
        self.df = df.copy()
        self.features = []
        
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
        self.calculate_team_form()
        return self.df[self.features], self.features 
import pandas as pd
import numpy as np
from typing import Tuple

class DataPreparator:
    def __init__(self, data_path: str):
        """Initialize the DataPreparator with path to data file."""
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the Bundesliga match data from CSV."""
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        return self.df
    
    def standardize_team_names(self) -> None:
        """Standardize team names to ensure consistency."""
        team_name_mapping = {
            'M\'gladbach': 'Borussia M.gladbach',
            'Bayern Munich': 'Bayern München',
            'Dusseldorf': 'Fortuna Düsseldorf',
            'Fortuna Dusseldorf': 'Fortuna Düsseldorf',
            'FC Koln': '1. FC Köln',
            'Koln': '1. FC Köln',
            'Ein Frankfurt': 'Eintracht Frankfurt',
            'Hertha': 'Hertha Berlin',
            'Greuther Furth': 'Greuther Fürth',
            'Nurnberg': 'Nürnberg'
        }
        self.df['HomeTeam'] = self.df['HomeTeam'].replace(team_name_mapping)
        self.df['AwayTeam'] = self.df['AwayTeam'].replace(team_name_mapping)
    
    def create_target_variable(self) -> None:
        """Create target variable (1: home win, 0: draw, -1: away win)."""
        self.df['Target'] = np.where(self.df['FTHG'] > self.df['FTAG'], 1,
                                   np.where(self.df['FTHG'] == self.df['FTAG'], 0, -1))
    
    def prepare_data(self) -> pd.DataFrame:
        """Main method to prepare the data."""
        self.load_data()
        self.standardize_team_names()
        self.create_target_variable()
        return self.df 
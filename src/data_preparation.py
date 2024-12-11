import pandas as pd
from typing import List

class DataPreparator:
    def __init__(self, data_path: str):
        """Initialize with path to data file."""
        self.data_path = data_path
    
    def prepare_data(self) -> pd.DataFrame:
        """Load and prepare data for analysis."""
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Add season progress features
        df['Season'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Calculate matches played in season
        df['MatchesInSeason'] = df.groupby('Season').cumcount() + 1
        
        return df
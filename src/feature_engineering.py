import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize FeatureEngineer with DataFrame."""
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
    def calculate_team_stats(self, team: str, before_date: pd.Timestamp, n_matches: int = 5) -> Dict:
        """Calculate recent statistics for a team."""
        team_matches = self.df[
            ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
            (self.df['Date'] < before_date)
        ].sort_values('Date', ascending=False).head(n_matches)
        
        stats = {
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'form_points': 0,
            'win_rate': 0,
            'matches_played': len(team_matches)
        }
        
        if stats['matches_played'] > 0:
            goals_scored = []
            goals_conceded = []
            points = []
            
            for _, match in team_matches.iterrows():
                if match['HomeTeam'] == team:
                    goals_scored.append(match['FTHG'])
                    goals_conceded.append(match['FTAG'])
                    if match['FTHG'] > match['FTAG']:
                        points.append(3)
                    elif match['FTHG'] == match['FTAG']:
                        points.append(1)
                    else:
                        points.append(0)
                else:
                    goals_scored.append(match['FTAG'])
                    goals_conceded.append(match['FTHG'])
                    if match['FTAG'] > match['FTHG']:
                        points.append(3)
                    elif match['FTAG'] == match['FTHG']:
                        points.append(1)
                    else:
                        points.append(0)
            
            stats['avg_goals_scored'] = np.mean(goals_scored)
            stats['avg_goals_conceded'] = np.mean(goals_conceded)
            stats['form_points'] = np.mean(points)
            stats['win_rate'] = sum(p == 3 for p in points) / len(points)
        
        return stats
    
    def get_h2h_stats(self, home_team: str, away_team: str, before_date: pd.Timestamp, n_matches: int = 5) -> Dict:
        """Calculate head-to-head statistics."""
        h2h_matches = self.df[
            (
                ((self.df['HomeTeam'] == home_team) & (self.df['AwayTeam'] == away_team)) |
                ((self.df['HomeTeam'] == away_team) & (self.df['AwayTeam'] == home_team))
            ) &
            (self.df['Date'] < before_date)
        ].sort_values('Date', ascending=False).head(n_matches)
        
        stats = {
            'home_wins': 0,
            'away_wins': 0,
            'draws': 0,
            'avg_goals': 0,
            'matches_played': len(h2h_matches)
        }
        
        if stats['matches_played'] > 0:
            total_goals = []
            for _, match in h2h_matches.iterrows():
                total_goals.append(match['FTHG'] + match['FTAG'])
                if match['FTHG'] > match['FTAG']:
                    if match['HomeTeam'] == home_team:
                        stats['home_wins'] += 1
                    else:
                        stats['away_wins'] += 1
                elif match['FTHG'] < match['FTAG']:
                    if match['HomeTeam'] == home_team:
                        stats['away_wins'] += 1
                    else:
                        stats['home_wins'] += 1
                else:
                    stats['draws'] += 1
            
            stats['avg_goals'] = np.mean(total_goals)
        
        return stats
    
    def get_league_position(self, team: str, before_date: pd.Timestamp) -> int:
        """Get team's league position before the match."""
        season = self.df[self.df['Date'] == before_date]['Season'].iloc[0]
        season_matches = self.df[
            (self.df['Season'] == season) &
            (self.df['Date'] < before_date)
        ]
        
        # Calculate points for each team
        team_points = {}
        for _, match in season_matches.iterrows():
            # Initialize teams if not in dict
            for t in [match['HomeTeam'], match['AwayTeam']]:
                if t not in team_points:
                    team_points[t] = {'points': 0, 'gd': 0}
            
            # Update points
            if match['FTHG'] > match['FTAG']:
                team_points[match['HomeTeam']]['points'] += 3
            elif match['FTHG'] < match['FTAG']:
                team_points[match['AwayTeam']]['points'] += 3
            else:
                team_points[match['HomeTeam']]['points'] += 1
                team_points[match['AwayTeam']]['points'] += 1
            
            # Update goal difference
            team_points[match['HomeTeam']]['gd'] += match['FTHG'] - match['FTAG']
            team_points[match['AwayTeam']]['gd'] += match['FTAG'] - match['FTHG']
        
        # Sort teams by points and goal difference
        sorted_teams = sorted(
            team_points.items(),
            key=lambda x: (x[1]['points'], x[1]['gd']),
            reverse=True
        )
        
        # Find position of team
        for i, (t, _) in enumerate(sorted_teams, 1):
            if t == team:
                return i
        return 0
    
    def create_match_features(self, home_team: str, away_team: str, match_date: pd.Timestamp) -> pd.DataFrame:
        """Create features for a specific match."""
        # Get team stats
        home_stats = self.calculate_team_stats(home_team, match_date)
        away_stats = self.calculate_team_stats(away_team, match_date)
        h2h_stats = self.get_h2h_stats(home_team, away_team, match_date)
        
        # Create feature dictionary
        features = {
            'home_avg_goals_scored': home_stats['avg_goals_scored'],
            'home_avg_goals_conceded': home_stats['avg_goals_conceded'],
            'home_form_points': home_stats['form_points'],
            'home_win_rate': home_stats['win_rate'],
            'away_avg_goals_scored': away_stats['avg_goals_scored'],
            'away_avg_goals_conceded': away_stats['avg_goals_conceded'],
            'away_form_points': away_stats['form_points'],
            'away_win_rate': away_stats['win_rate'],
            'h2h_home_win_rate': h2h_stats['home_wins'] / max(1, h2h_stats['matches_played']),
            'h2h_avg_goals': h2h_stats['avg_goals'],
            'home_position': self.get_league_position(home_team, match_date),
            'away_position': self.get_league_position(away_team, match_date),
            'position_diff': self.get_league_position(home_team, match_date) - 
                           self.get_league_position(away_team, match_date)
        }
        
        return pd.DataFrame([features])
    
    def get_feature_matrix(self) -> Tuple[pd.DataFrame, List[str]]:
        """Create feature matrix for all matches."""
        features_list = []
        
        for _, match in self.df.iterrows():
            features = self.create_match_features(
                match['HomeTeam'],
                match['AwayTeam'],
                match['Date']
            )
            features_list.append(features)
        
        feature_matrix = pd.concat(features_list, ignore_index=True)
        return feature_matrix, feature_matrix.columns.tolist()
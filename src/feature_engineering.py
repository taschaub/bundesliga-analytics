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
    
    def _add_advanced_features(self):
        """Add advanced features including expected goals, rest days, and momentum."""
        print("Adding advanced features...")
        
        for idx, match in self.df.sort_values('Date').iterrows():
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get historical matches before this match
            historical = self.df[self.df['Date'] < match_date]
            
            if not historical.empty:
                # Expected Goals (xG) based on historical scoring rates
                home_historical = historical[historical['HomeTeam'] == home_team]
                away_historical = historical[historical['AwayTeam'] == away_team]
                
                if not home_historical.empty:
                    home_scoring_rate = home_historical['FTHG'].mean()
                    home_conceding_rate = home_historical['FTAG'].mean()
                else:
                    home_scoring_rate = self.df['FTHG'].mean()
                    home_conceding_rate = self.df['FTAG'].mean()
                
                if not away_historical.empty:
                    away_scoring_rate = away_historical['FTAG'].mean()
                    away_conceding_rate = away_historical['FTHG'].mean()
                else:
                    away_scoring_rate = self.df['FTAG'].mean()
                    away_conceding_rate = self.df['FTHG'].mean()
                
                self.df.at[idx, 'ExpectedHomeGoals'] = (home_scoring_rate + away_conceding_rate) / 2
                self.df.at[idx, 'ExpectedAwayGoals'] = (away_scoring_rate + home_conceding_rate) / 2
                
                # Rest Days
                for team, prefix in [(home_team, 'Home'), (away_team, 'Away')]:
                    last_match = historical[
                        (historical['HomeTeam'] == team) | 
                        (historical['AwayTeam'] == team)
                    ].sort_values('Date').tail(1)
                    
                    if not last_match.empty:
                        rest_days = (match_date - last_match['Date'].iloc[0]).days
                        self.df.at[idx, f'{prefix}RestDays'] = rest_days
                    else:
                        self.df.at[idx, f'{prefix}RestDays'] = 7  # Default to a week if no previous match
                
                # Momentum Features
                for team, prefix in [(home_team, 'Home'), (away_team, 'Away')]:
                    recent = historical[
                        (historical['HomeTeam'] == team) | 
                        (historical['AwayTeam'] == team)
                    ].sort_values('Date').tail(5)
                    
                    if not recent.empty:
                        # Winning/Losing Streak
                        streak = 0
                        for _, game in recent.iterrows():
                            if ((game['HomeTeam'] == team and game['FTR'] == 'H') or 
                                (game['AwayTeam'] == team and game['FTR'] == 'A')):
                                streak += 1
                            else:
                                break
                        self.df.at[idx, f'{prefix}WinningStreak'] = streak
                        
                        # Recent Goals Trend
                        recent_goals_scored = []
                        recent_goals_conceded = []
                        for _, game in recent.iterrows():
                            if game['HomeTeam'] == team:
                                recent_goals_scored.append(game['FTHG'])
                                recent_goals_conceded.append(game['FTAG'])
                            else:
                                recent_goals_scored.append(game['FTAG'])
                                recent_goals_conceded.append(game['FTHG'])
                        
                        # Calculate trend (positive slope means improving)
                        if len(recent_goals_scored) >= 3:
                            scored_trend = np.polyfit(range(len(recent_goals_scored)), recent_goals_scored, 1)[0]
                            conceded_trend = np.polyfit(range(len(recent_goals_conceded)), recent_goals_conceded, 1)[0]
                            self.df.at[idx, f'{prefix}ScoringTrend'] = scored_trend
                            self.df.at[idx, f'{prefix}ConcedingTrend'] = conceded_trend
                        else:
                            self.df.at[idx, f'{prefix}ScoringTrend'] = 0
                            self.df.at[idx, f'{prefix}ConcedingTrend'] = 0
                    else:
                        self.df.at[idx, f'{prefix}WinningStreak'] = 0
                        self.df.at[idx, f'{prefix}ScoringTrend'] = 0
                        self.df.at[idx, f'{prefix}ConcedingTrend'] = 0

    def _add_league_position_features(self):
        """Add features based on league positions and points."""
        print("Adding league position features...")
        
        for season in self.df['Season'].unique():
            season_matches = self.df[self.df['Season'] == season].sort_values('Date')
            
            # Initialize points and stats dictionaries for the season
            points = {}
            matches_played = {}
            goals_scored = {}
            goals_conceded = {}
            
            for idx, match in season_matches.iterrows():
                home_team = match['HomeTeam']
                away_team = match['AwayTeam']
                
                # Initialize teams if not seen before
                for team in [home_team, away_team]:
                    if team not in points:
                        points[team] = 0
                        matches_played[team] = 0
                        goals_scored[team] = 0
                        goals_conceded[team] = 0
                
                # Calculate current points per game and goal differences
                for team in points:
                    if matches_played[team] > 0:
                        ppg = points[team] / matches_played[team]
                        gd = goals_scored[team] - goals_conceded[team]
                        gpg = goals_scored[team] / matches_played[team]
                        gcpg = goals_conceded[team] / matches_played[team]
                    else:
                        ppg = 0
                        gd = 0
                        gpg = 0
                        gcpg = 0
                    
                    if team == home_team:
                        self.df.at[idx, 'HomePPG'] = ppg
                        self.df.at[idx, 'HomeGD'] = gd
                        self.df.at[idx, 'HomeGPG'] = gpg
                        self.df.at[idx, 'HomeGCPG'] = gcpg
                    elif team == away_team:
                        self.df.at[idx, 'AwayPPG'] = ppg
                        self.df.at[idx, 'AwayGD'] = gd
                        self.df.at[idx, 'AwayGPG'] = gpg
                        self.df.at[idx, 'AwayGCPG'] = gcpg
                
                # Calculate and store current positions
                sorted_teams = sorted(
                    points.items(),
                    key=lambda x: (
                        -x[1],  # Points (descending)
                        -(goals_scored[x[0]] - goals_conceded[x[0]]),  # Goal difference (descending)
                        -goals_scored[x[0]]  # Goals scored (descending)
                    )
                )
                positions = {team: pos+1 for pos, (team, _) in enumerate(sorted_teams)}
                
                self.df.at[idx, 'HomePosition'] = positions.get(home_team, len(positions))
                self.df.at[idx, 'AwayPosition'] = positions.get(away_team, len(positions))
                self.df.at[idx, 'PositionDiff'] = positions.get(away_team, len(positions)) - positions.get(home_team, len(positions))
                
                # Calculate relative strength based on league position
                max_position = len(positions)
                home_rel_strength = 1 - ((positions.get(home_team, max_position) - 1) / (max_position - 1)) if max_position > 1 else 0.5
                away_rel_strength = 1 - ((positions.get(away_team, max_position) - 1) / (max_position - 1)) if max_position > 1 else 0.5
                self.df.at[idx, 'HomeRelativeStrength'] = home_rel_strength
                self.df.at[idx, 'AwayRelativeStrength'] = away_rel_strength
                
                # Update points and stats after match
                if match['FTR'] == 'H':
                    points[home_team] += 3
                elif match['FTR'] == 'A':
                    points[away_team] += 3
                else:
                    points[home_team] += 1
                    points[away_team] += 1
                
                # Update matches played and goals
                matches_played[home_team] += 1
                matches_played[away_team] += 1
                goals_scored[home_team] += match['FTHG']
                goals_scored[away_team] += match['FTAG']
                goals_conceded[home_team] += match['FTAG']
                goals_conceded[away_team] += match['FTHG']

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
        
        print("Adding advanced features...")
        self._add_advanced_features()
        
        print("Adding league position features...")
        self._add_league_position_features()
        
        # Fill NaN values using forward fill
        self.df = self.df.ffill()
        
        # Create target variable (0: Away Win, 1: Draw, 2: Home Win)
        self.df['Target'] = np.where(self.df['FTHG'] > self.df['FTAG'], 2,
                                   np.where(self.df['FTHG'] == self.df['FTAG'], 1, 0))
        
        # Ensure all numeric features are float
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Target', 'Season', 'Month', 'MatchesInSeason']:
                self.df[col] = self.df[col].astype(float)
        
        return self.df
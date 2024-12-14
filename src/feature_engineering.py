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
        
        # Initialize caches
        self.team_ratings_cache = {}  # (team, date) -> ratings dict
        self.form_cache = {}          # (team, date) -> form dict
        self.h2h_cache = {}           # (team1, team2, date) -> h2h stats
        self.goals_cache = {}         # (team, date) -> goals stats
        self.momentum_cache = {}      # (team, date) -> momentum features
        self.rest_cache = {}          # (team, date) -> rest days
        
    def _get_team_rating(self, team: str, date: pd.Timestamp) -> dict:
        """Get cached team rating or calculate if not exists."""
        cache_key = (team, date.date())
        if cache_key in self.team_ratings_cache:
            return self.team_ratings_cache[cache_key]
        
        # Calculate rating using historical matches
        historical = self.df[
            (
                ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
                (self.df['Date'] < date)
            )
        ]
        
        if historical.empty:
            rating = {'overall': 1500, 'home': 1500, 'away': 1500}
        else:
            # Calculate points
            home_matches = historical[historical['HomeTeam'] == team]
            away_matches = historical[historical['AwayTeam'] == team]
            
            home_points = (home_matches['FTHG'] > home_matches['FTAG']).sum() * 3 + \
                        (home_matches['FTHG'] == home_matches['FTAG']).sum()
            away_points = (away_matches['FTAG'] > away_matches['FTHG']).sum() * 3 + \
                        (away_matches['FTHG'] == away_matches['FTAG']).sum()
            
            total_matches = len(home_matches) + len(away_matches)
            if total_matches > 0:
                ppg = (home_points + away_points) / total_matches
                home_ppg = home_points / max(len(home_matches), 1)
                away_ppg = away_points / max(len(away_matches), 1)
                
                rating = {
                    'overall': 1000 + (ppg * 250),
                    'home': 1000 + (home_ppg * 250),
                    'away': 1000 + (away_ppg * 250)
                }
            else:
                rating = {'overall': 1500, 'home': 1500, 'away': 1500}
        
        self.team_ratings_cache[cache_key] = rating
        return rating
    
    def _get_team_form(self, team: str, date: pd.Timestamp, n_matches: int) -> float:
        """Get cached team form or calculate if not exists."""
        cache_key = (team, date.date(), n_matches)
        if cache_key in self.form_cache:
            return self.form_cache[cache_key]
        
        # Calculate form using historical matches
        historical = self.df[
            (
                ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
                (self.df['Date'] < date)
            )
        ].sort_values('Date').tail(n_matches)
        
        if len(historical) >= n_matches:
            wins = sum(
                ((historical['HomeTeam'] == team) & (historical['FTR'] == 'H')) |
                ((historical['AwayTeam'] == team) & (historical['FTR'] == 'A'))
            )
            form = wins / n_matches
        else:
            form = 0.5
        
        self.form_cache[cache_key] = form
        return form
    
    def _get_h2h_stats(self, home_team: str, away_team: str, date: pd.Timestamp) -> dict:
        """Get cached H2H stats or calculate if not exists."""
        cache_key = (home_team, away_team, date.date())
        if cache_key in self.h2h_cache:
            return self.h2h_cache[cache_key]
        
        # Get previous meetings
        h2h_matches = self.df[
            (
                ((self.df['HomeTeam'] == home_team) & (self.df['AwayTeam'] == away_team)) |
                ((self.df['HomeTeam'] == away_team) & (self.df['AwayTeam'] == home_team))
            ) &
            (self.df['Date'] < date)
        ]
        
        if not h2h_matches.empty:
            home_wins = sum(
                (h2h_matches['HomeTeam'] == home_team) & (h2h_matches['FTR'] == 'H')
            ) + sum(
                (h2h_matches['AwayTeam'] == home_team) & (h2h_matches['FTR'] == 'A')
            )
            stats = {'home_win_rate': home_wins / len(h2h_matches)}
        else:
            stats = {'home_win_rate': 0.5}
        
        self.h2h_cache[cache_key] = stats
        return stats
    
    def _get_goals_stats(self, team: str, date: pd.Timestamp, n_matches: int = 5) -> dict:
        """Get cached goals stats or calculate if not exists."""
        cache_key = (team, date.date(), n_matches)
        if cache_key in self.goals_cache:
            return self.goals_cache[cache_key]
        
        # Get recent matches
        recent = self.df[
            (
                ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
                (self.df['Date'] < date)
            )
        ].sort_values('Date').tail(n_matches)
        
        if not recent.empty:
            goals_scored = sum(
                np.where(recent['HomeTeam'] == team,
                        recent['FTHG'],
                        recent['FTAG'])
            )
            goals_conceded = sum(
                np.where(recent['HomeTeam'] == team,
                        recent['FTAG'],
                        recent['FTHG'])
            )
            stats = {
                'goals_scored_avg': goals_scored / len(recent),
                'goals_conceded_avg': goals_conceded / len(recent)
            }
        else:
            stats = {'goals_scored_avg': 0, 'goals_conceded_avg': 0}
        
        self.goals_cache[cache_key] = stats
        return stats
    
    def _get_momentum_features(self, team: str, date: pd.Timestamp) -> dict:
        """Get momentum-based features for a team."""
        cache_key = (team, date.date())
        if cache_key in self.momentum_cache:
            return self.momentum_cache[cache_key]
        
        # Get recent matches
        recent = self.df[
            (
                ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
                (self.df['Date'] < date)
            )
        ].sort_values('Date').tail(5)
        
        if not recent.empty:
            # Winning streak
            streak = 0
            for _, game in recent.iterrows():
                if ((game['HomeTeam'] == team and game['FTR'] == 'H') or 
                    (game['AwayTeam'] == team and game['FTR'] == 'A')):
                    streak += 1
                else:
                    break
            
            # Goals trend
            recent_goals_scored = []
            recent_goals_conceded = []
            for _, game in recent.iterrows():
                if game['HomeTeam'] == team:
                    recent_goals_scored.append(game['FTHG'])
                    recent_goals_conceded.append(game['FTAG'])
                else:
                    recent_goals_scored.append(game['FTAG'])
                    recent_goals_conceded.append(game['FTHG'])
            
            # Calculate trends
            if len(recent_goals_scored) >= 3:
                scoring_trend = np.polyfit(range(len(recent_goals_scored)), recent_goals_scored, 1)[0]
                conceding_trend = np.polyfit(range(len(recent_goals_conceded)), recent_goals_conceded, 1)[0]
            else:
                scoring_trend = 0
                conceding_trend = 0
            
            features = {
                'winning_streak': streak,
                'scoring_trend': scoring_trend,
                'conceding_trend': conceding_trend
            }
        else:
            features = {
                'winning_streak': 0,
                'scoring_trend': 0,
                'conceding_trend': 0
            }
        
        self.momentum_cache[cache_key] = features
        return features
    
    def _get_rest_days(self, team: str, date: pd.Timestamp) -> int:
        """Calculate rest days since last match."""
        cache_key = (team, date.date())
        if cache_key in self.rest_cache:
            return self.rest_cache[cache_key]
        
        # Find last match
        last_match = self.df[
            (
                ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
                (self.df['Date'] < date)
            )
        ].sort_values('Date').tail(1)
        
        if not last_match.empty:
            rest_days = (date - last_match['Date'].iloc[0]).days
        else:
            rest_days = 7  # Default to a week if no previous match
        
        self.rest_cache[cache_key] = rest_days
        return rest_days
    
    def engineer_features(self) -> pd.DataFrame:
        """Engineer all features using caching for efficiency."""
        print("Engineering features...")
        
        # Initialize additional caches
        self.momentum_cache = {}
        self.rest_cache = {}
        
        # Process matches chronologically
        for idx, match in self.df.sort_values('Date').iterrows():
            match_date = match['Date']
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get team ratings
            home_ratings = self._get_team_rating(home_team, match_date)
            away_ratings = self._get_team_rating(away_team, match_date)
            
            # Store ratings
            self.df.at[idx, 'HomeTeamRating'] = home_ratings['overall']
            self.df.at[idx, 'AwayTeamRating'] = away_ratings['overall']
            self.df.at[idx, 'HomeTeamHomeRating'] = home_ratings['home']
            self.df.at[idx, 'AwayTeamAwayRating'] = away_ratings['away']
            
            # Get form features
            for n in [3, 5, 10]:
                self.df.at[idx, f'HomeForm{n}'] = self._get_team_form(home_team, match_date, n)
                self.df.at[idx, f'AwayForm{n}'] = self._get_team_form(away_team, match_date, n)
            
            # Get H2H stats
            h2h_stats = self._get_h2h_stats(home_team, away_team, match_date)
            self.df.at[idx, 'H2HHomeWinRate'] = h2h_stats['home_win_rate']
            
            # Get goals stats
            home_goals = self._get_goals_stats(home_team, match_date)
            away_goals = self._get_goals_stats(away_team, match_date)
            
            self.df.at[idx, 'HomeRecentGoalsScored'] = home_goals['goals_scored_avg']
            self.df.at[idx, 'HomeRecentGoalsConceded'] = home_goals['goals_conceded_avg']
            self.df.at[idx, 'AwayRecentGoalsScored'] = away_goals['goals_scored_avg']
            self.df.at[idx, 'AwayRecentGoalsConceded'] = away_goals['goals_conceded_avg']
            
            # Calculate expected goals
            self.df.at[idx, 'ExpectedHomeGoals'] = (home_goals['goals_scored_avg'] + 
                                                   away_goals['goals_conceded_avg']) / 2
            self.df.at[idx, 'ExpectedAwayGoals'] = (away_goals['goals_scored_avg'] + 
                                                   home_goals['goals_conceded_avg']) / 2
            
            # Get momentum features
            home_momentum = self._get_momentum_features(home_team, match_date)
            away_momentum = self._get_momentum_features(away_team, match_date)
            
            self.df.at[idx, 'HomeWinningStreak'] = home_momentum['winning_streak']
            self.df.at[idx, 'HomeScoringTrend'] = home_momentum['scoring_trend']
            self.df.at[idx, 'HomeConcedingTrend'] = home_momentum['conceding_trend']
            self.df.at[idx, 'AwayWinningStreak'] = away_momentum['winning_streak']
            self.df.at[idx, 'AwayScoringTrend'] = away_momentum['scoring_trend']
            self.df.at[idx, 'AwayConcedingTrend'] = away_momentum['conceding_trend']
            
            # Get rest days
            self.df.at[idx, 'HomeRestDays'] = self._get_rest_days(home_team, match_date)
            self.df.at[idx, 'AwayRestDays'] = self._get_rest_days(away_team, match_date)
            
            # Add league position features
            if idx > 0:  # Skip first match of season
                season = self.df.at[idx, 'Season']
                season_matches = self.df[
                    (self.df['Season'] == season) & 
                    (self.df['Date'] < match_date)
                ]
                
                # Calculate points and positions
                points = {}
                goals_for = {}
                goals_against = {}
                matches_played = {}
                
                for _, m in season_matches.iterrows():
                    # Initialize if needed
                    for t in [m['HomeTeam'], m['AwayTeam']]:
                        if t not in points:
                            points[t] = 0
                            goals_for[t] = 0
                            goals_against[t] = 0
                            matches_played[t] = 0
                    
                    # Update points
                    if m['FTR'] == 'H':
                        points[m['HomeTeam']] += 3
                    elif m['FTR'] == 'A':
                        points[m['AwayTeam']] += 3
                    else:
                        points[m['HomeTeam']] += 1
                        points[m['AwayTeam']] += 1
                    
                    # Update goals and matches
                    goals_for[m['HomeTeam']] += m['FTHG']
                    goals_for[m['AwayTeam']] += m['FTAG']
                    goals_against[m['HomeTeam']] += m['FTAG']
                    goals_against[m['AwayTeam']] += m['FTHG']
                    matches_played[m['HomeTeam']] += 1
                    matches_played[m['AwayTeam']] += 1
                
                # Calculate positions
                teams_sorted = sorted(
                    points.keys(),
                    key=lambda t: (
                        points[t],
                        goals_for[t] - goals_against[t],
                        goals_for[t]
                    ),
                    reverse=True
                )
                positions = {t: i+1 for i, t in enumerate(teams_sorted)}
                max_position = len(positions)
                
                # Store positions and relative strength
                self.df.at[idx, 'HomePosition'] = positions.get(home_team, max_position)
                self.df.at[idx, 'AwayPosition'] = positions.get(away_team, max_position)
                self.df.at[idx, 'PositionDiff'] = (positions.get(away_team, max_position) - 
                                                  positions.get(home_team, max_position))
                
                # Calculate relative strength (0-1 scale, 1 being top position)
                self.df.at[idx, 'HomeRelativeStrength'] = 1 - ((positions.get(home_team, max_position) - 1) / 
                                                              (max_position - 1)) if max_position > 1 else 0.5
                self.df.at[idx, 'AwayRelativeStrength'] = 1 - ((positions.get(away_team, max_position) - 1) / 
                                                              (max_position - 1)) if max_position > 1 else 0.5
                
                # Store per-game stats
                for team, prefix in [(home_team, 'Home'), (away_team, 'Away')]:
                    if matches_played.get(team, 0) > 0:
                        self.df.at[idx, f'{prefix}PPG'] = points.get(team, 0) / matches_played[team]
                        self.df.at[idx, f'{prefix}GPG'] = goals_for.get(team, 0) / matches_played[team]
                        self.df.at[idx, f'{prefix}GCPG'] = goals_against.get(team, 0) / matches_played[team]
                        self.df.at[idx, f'{prefix}GD'] = ((goals_for.get(team, 0) - goals_against.get(team, 0)) / 
                                                         matches_played[team])
                    else:
                        self.df.at[idx, f'{prefix}PPG'] = 0
                        self.df.at[idx, f'{prefix}GPG'] = 0
                        self.df.at[idx, f'{prefix}GCPG'] = 0
                        self.df.at[idx, f'{prefix}GD'] = 0
        
        # Create target variable (0: Away Win, 1: Draw, 2: Home Win)
        self.df['Target'] = np.where(self.df['FTHG'] > self.df['FTAG'], 2,
                                    np.where(self.df['FTHG'] == self.df['FTAG'], 1, 0))
        
        # Ensure all numeric features are float
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Target', 'Season', 'Month', 'MatchesInSeason']:
                self.df[col] = self.df[col].astype(float)
        
        return self.df
    
    def engineer_features_for_match(self, match: pd.Series) -> pd.DataFrame:
        """Engineer features for a single match efficiently."""
        match_date = pd.to_datetime(match['Date'])
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Create a single-row DataFrame for the match
        match_df = pd.DataFrame([match])
        
        # Get team ratings
        home_ratings = self._get_team_rating(home_team, match_date)
        away_ratings = self._get_team_rating(away_team, match_date)
        
        # Store ratings
        match_df['HomeTeamRating'] = home_ratings['overall']
        match_df['AwayTeamRating'] = away_ratings['overall']
        match_df['HomeTeamHomeRating'] = home_ratings['home']
        match_df['AwayTeamAwayRating'] = away_ratings['away']
        
        # Get form features
        for n in [3, 5, 10]:
            match_df[f'HomeForm{n}'] = self._get_team_form(home_team, match_date, n)
            match_df[f'AwayForm{n}'] = self._get_team_form(away_team, match_date, n)
        
        # Get H2H stats
        h2h_stats = self._get_h2h_stats(home_team, away_team, match_date)
        match_df['H2HHomeWinRate'] = h2h_stats['home_win_rate']
        
        # Get goals stats
        home_goals = self._get_goals_stats(home_team, match_date)
        away_goals = self._get_goals_stats(away_team, match_date)
        
        match_df['HomeRecentGoalsScored'] = home_goals['goals_scored_avg']
        match_df['HomeRecentGoalsConceded'] = home_goals['goals_conceded_avg']
        match_df['AwayRecentGoalsScored'] = away_goals['goals_scored_avg']
        match_df['AwayRecentGoalsConceded'] = away_goals['goals_conceded_avg']
        
        # Calculate expected goals
        match_df['ExpectedHomeGoals'] = (home_goals['goals_scored_avg'] + 
                                        away_goals['goals_conceded_avg']) / 2
        match_df['ExpectedAwayGoals'] = (away_goals['goals_scored_avg'] + 
                                        home_goals['goals_conceded_avg']) / 2
        
        # Get momentum features
        home_momentum = self._get_momentum_features(home_team, match_date)
        away_momentum = self._get_momentum_features(away_team, match_date)
        
        match_df['HomeWinningStreak'] = home_momentum['winning_streak']
        match_df['HomeScoringTrend'] = home_momentum['scoring_trend']
        match_df['HomeConcedingTrend'] = home_momentum['conceding_trend']
        match_df['AwayWinningStreak'] = away_momentum['winning_streak']
        match_df['AwayScoringTrend'] = away_momentum['scoring_trend']
        match_df['AwayConcedingTrend'] = away_momentum['conceding_trend']
        
        # Get rest days
        match_df['HomeRestDays'] = self._get_rest_days(home_team, match_date)
        match_df['AwayRestDays'] = self._get_rest_days(away_team, match_date)
        
        # Add league position features
        season = match['Season']
        season_matches = self.df[
            (self.df['Season'] == season) & 
            (self.df['Date'] < match_date)
        ]
        
        if not season_matches.empty:
            # Calculate points and positions
            points = {}
            goals_for = {}
            goals_against = {}
            matches_played = {}
            
            for _, m in season_matches.iterrows():
                # Initialize if needed
                for t in [m['HomeTeam'], m['AwayTeam']]:
                    if t not in points:
                        points[t] = 0
                        goals_for[t] = 0
                        goals_against[t] = 0
                        matches_played[t] = 0
                
                # Update points
                if m['FTR'] == 'H':
                    points[m['HomeTeam']] += 3
                elif m['FTR'] == 'A':
                    points[m['AwayTeam']] += 3
                else:
                    points[m['HomeTeam']] += 1
                    points[m['AwayTeam']] += 1
                
                # Update goals and matches
                goals_for[m['HomeTeam']] += m['FTHG']
                goals_for[m['AwayTeam']] += m['FTAG']
                goals_against[m['HomeTeam']] += m['FTAG']
                goals_against[m['AwayTeam']] += m['FTHG']
                matches_played[m['HomeTeam']] += 1
                matches_played[m['AwayTeam']] += 1
            
            # Calculate positions
            teams_sorted = sorted(
                points.keys(),
                key=lambda t: (
                    points[t],
                    goals_for[t] - goals_against[t],
                    goals_for[t]
                ),
                reverse=True
            )
            positions = {t: i+1 for i, t in enumerate(teams_sorted)}
            max_position = len(positions)
            
            # Store positions and relative strength
            match_df['HomePosition'] = positions.get(home_team, max_position)
            match_df['AwayPosition'] = positions.get(away_team, max_position)
            match_df['PositionDiff'] = (positions.get(away_team, max_position) - 
                                       positions.get(home_team, max_position))
            
            # Calculate relative strength (0-1 scale, 1 being top position)
            match_df['HomeRelativeStrength'] = 1 - ((positions.get(home_team, max_position) - 1) / 
                                                   (max_position - 1)) if max_position > 1 else 0.5
            match_df['AwayRelativeStrength'] = 1 - ((positions.get(away_team, max_position) - 1) / 
                                                   (max_position - 1)) if max_position > 1 else 0.5
            
            # Store per-game stats
            for team, prefix in [(home_team, 'Home'), (away_team, 'Away')]:
                if matches_played.get(team, 0) > 0:
                    match_df[f'{prefix}PPG'] = points.get(team, 0) / matches_played[team]
                    match_df[f'{prefix}GPG'] = goals_for.get(team, 0) / matches_played[team]
                    match_df[f'{prefix}GCPG'] = goals_against.get(team, 0) / matches_played[team]
                    match_df[f'{prefix}GD'] = ((goals_for.get(team, 0) - goals_against.get(team, 0)) / 
                                               matches_played[team])
                else:
                    match_df[f'{prefix}PPG'] = 0
                    match_df[f'{prefix}GPG'] = 0
                    match_df[f'{prefix}GCPG'] = 0
                    match_df[f'{prefix}GD'] = 0
        else:
            # Set default values for first match of season
            for col in ['HomePosition', 'AwayPosition', 'PositionDiff',
                       'HomeRelativeStrength', 'AwayRelativeStrength',
                       'HomePPG', 'HomeGPG', 'HomeGCPG', 'HomeGD',
                       'AwayPPG', 'AwayGPG', 'AwayGCPG', 'AwayGD']:
                match_df[col] = 0
        
        # Create target variable (0: Away Win, 1: Draw, 2: Home Win)
        match_df['Target'] = np.where(match_df['FTHG'] > match_df['FTAG'], 2,
                                     np.where(match_df['FTHG'] == match_df['FTAG'], 1, 0))
        
        # Ensure all numeric features are float
        numeric_columns = match_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Target', 'Season', 'Month', 'MatchesInSeason']:
                match_df[col] = match_df[col].astype(float)
        
        return match_df
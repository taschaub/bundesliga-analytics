import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize FeatureEngineer with DataFrame."""
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
    def calculate_team_stats(self, team: str, before_date: pd.Timestamp, n_matches: int = 10) -> Dict:
        """Calculate recent statistics for a team with enhanced metrics."""
        team_matches = self.df[
            ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
            (self.df['Date'] < before_date)
        ].sort_values('Date', ascending=False).head(n_matches)
        
        stats = {
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'form_points': 0,
            'win_rate': 0,
            'home_win_rate': 0,
            'away_win_rate': 0,
            'clean_sheets': 0,
            'failed_to_score': 0,
            'comeback_wins': 0,
            'leads_lost': 0,
            'matches_played': len(team_matches),
            'goals_per_game_trend': 0,
            'points_trend': 0,
            'defensive_stability': 0,
            'offensive_efficiency': 0
        }
        
        if stats['matches_played'] > 0:
            home_games = 0
            away_games = 0
            goals_trend = []
            points_trend = []
            
            for idx, match in team_matches.iterrows():
                is_home = match['HomeTeam'] == team
                team_goals = match['FTHG'] if is_home else match['FTAG']
                opp_goals = match['FTAG'] if is_home else match['FTHG']
                
                # Basic stats
                stats['avg_goals_scored'] += team_goals
                stats['avg_goals_conceded'] += opp_goals
                
                # Clean sheets and scoring
                if opp_goals == 0:
                    stats['clean_sheets'] += 1
                if team_goals == 0:
                    stats['failed_to_score'] += 1
                
                # Win/loss stats
                points = 0
                if team_goals > opp_goals:
                    points = 3
                    if is_home:
                        home_games += 1
                        stats['home_win_rate'] += 1
                    else:
                        away_games += 1
                        stats['away_win_rate'] += 1
                elif team_goals == opp_goals:
                    points = 1
                    if is_home:
                        home_games += 1
                    else:
                        away_games += 1
                else:
                    if is_home:
                        home_games += 1
                    else:
                        away_games += 1
                
                stats['form_points'] += points
                points_trend.append(points)
                goals_trend.append(team_goals)
                
                # Comeback wins and leads lost
                if 'HT' in match.index:  # If we have half-time data
                    ht_team_goals = match['HTHG'] if is_home else match['HTAG']
                    ht_opp_goals = match['HTAG'] if is_home else match['HTHG']
                    
                    if ht_team_goals < ht_opp_goals and team_goals > opp_goals:
                        stats['comeback_wins'] += 1
                    elif ht_team_goals > ht_opp_goals and team_goals <= opp_goals:
                        stats['leads_lost'] += 1
            
            # Calculate averages and rates
            stats['avg_goals_scored'] /= stats['matches_played']
            stats['avg_goals_conceded'] /= stats['matches_played']
            stats['form_points'] /= stats['matches_played']
            stats['home_win_rate'] = stats['home_win_rate'] / max(1, home_games)
            stats['away_win_rate'] = stats['away_win_rate'] / max(1, away_games)
            stats['win_rate'] = (stats['home_win_rate'] * home_games + 
                               stats['away_win_rate'] * away_games) / stats['matches_played']
            
            # Calculate trends (using weighted average)
            weights = np.linspace(1, 0.1, len(goals_trend))
            stats['goals_per_game_trend'] = np.average(goals_trend, weights=weights)
            stats['points_trend'] = np.average(points_trend, weights=weights)
            
            # Calculate advanced metrics
            stats['defensive_stability'] = (stats['clean_sheets'] / stats['matches_played'] * 
                                         (1 / (stats['avg_goals_conceded'] + 0.1)))
            stats['offensive_efficiency'] = (stats['avg_goals_scored'] * 
                                         (1 - stats['failed_to_score'] / stats['matches_played']))
        
        return stats
    
    def get_h2h_stats(self, home_team: str, away_team: str, before_date: pd.Timestamp, n_matches: int = 10) -> Dict:
        """Calculate enhanced head-to-head statistics."""
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
            'home_team_dominance': 0,
            'recent_momentum': 0,
            'matches_played': len(h2h_matches),
            'avg_goal_diff': 0,
            'high_scoring_rate': 0,
            'low_scoring_rate': 0
        }
        
        if stats['matches_played'] > 0:
            total_goals = []
            goal_diffs = []
            momentum_points = []
            
            for idx, match in h2h_matches.iterrows():
                match_goals = match['FTHG'] + match['FTAG']
                total_goals.append(match_goals)
                
                if match['HomeTeam'] == home_team:
                    goal_diff = match['FTHG'] - match['FTAG']
                    if match['FTHG'] > match['FTAG']:
                        stats['home_wins'] += 1
                        momentum_points.append(1)
                    elif match['FTHG'] < match['FTAG']:
                        stats['away_wins'] += 1
                        momentum_points.append(-1)
                    else:
                        stats['draws'] += 1
                        momentum_points.append(0)
                else:
                    goal_diff = match['FTAG'] - match['FTHG']
                    if match['FTAG'] > match['FTHG']:
                        stats['home_wins'] += 1
                        momentum_points.append(1)
                    elif match['FTAG'] < match['FTHG']:
                        stats['away_wins'] += 1
                        momentum_points.append(-1)
                    else:
                        stats['draws'] += 1
                        momentum_points.append(0)
                
                goal_diffs.append(goal_diff)
                
                # High/low scoring matches
                if match_goals > 2.5:
                    stats['high_scoring_rate'] += 1
                if match_goals < 2.5:
                    stats['low_scoring_rate'] += 1
            
            stats['avg_goals'] = np.mean(total_goals)
            stats['avg_goal_diff'] = np.mean(goal_diffs)
            stats['high_scoring_rate'] /= stats['matches_played']
            stats['low_scoring_rate'] /= stats['matches_played']
            
            # Calculate home team dominance (weighted by recency)
            weights = np.linspace(1, 0.1, len(goal_diffs))
            stats['home_team_dominance'] = np.average(goal_diffs, weights=weights)
            
            # Calculate recent momentum (weighted by recency)
            stats['recent_momentum'] = np.average(momentum_points, weights=weights)
        
        return stats
    
    def get_league_position(self, team: str, before_date: pd.Timestamp) -> Dict:
        """Get enhanced team's league position and performance metrics before the match."""
        season = self.df[self.df['Date'] == before_date]['Season'].iloc[0]
        season_matches = self.df[
            (self.df['Season'] == season) &
            (self.df['Date'] < before_date)
        ]
        
        # If no matches have been played yet in the season, use previous season's final standings
        if len(season_matches) == 0:
            previous_season = str(int(season.split('/')[0]) - 1) + '/' + str(int(season.split('/')[1]) - 1)
            season_matches = self.df[self.df['Season'] == previous_season]
        
        # If still no matches (e.g., first season in dataset), return default stats
        if len(season_matches) == 0:
            return {
                'points': 0,
                'gd': 0,
                'goals_for': 0,
                'goals_against': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'home_points': 0,
                'away_points': 0,
                'matches_played': 0,
                'ppg': 0,
                'home_ppg': 0,
                'away_ppg': 0,
                'goals_per_game': 0,
                'goals_against_per_game': 0,
                'position': 1,  # Assume middle position
                'total_teams': 18,  # Bundesliga has 18 teams
                'relative_position': 0.5  # Middle position
            }
        
        # Calculate points and performance metrics for each team
        team_stats = {}
        for _, match in season_matches.iterrows():
            # Initialize teams if not in dict
            for t in [match['HomeTeam'], match['AwayTeam']]:
                if t not in team_stats:
                    team_stats[t] = {
                        'points': 0,
                        'gd': 0,
                        'goals_for': 0,
                        'goals_against': 0,
                        'wins': 0,
                        'draws': 0,
                        'losses': 0,
                        'home_points': 0,
                        'away_points': 0,
                        'matches_played': 0
                    }
            
            # Update stats for home team
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            home_goals = match['FTHG']
            away_goals = match['FTAG']
            
            team_stats[home_team]['matches_played'] += 1
            team_stats[away_team]['matches_played'] += 1
            
            team_stats[home_team]['goals_for'] += home_goals
            team_stats[home_team]['goals_against'] += away_goals
            team_stats[away_team]['goals_for'] += away_goals
            team_stats[away_team]['goals_against'] += home_goals
            
            if home_goals > away_goals:
                team_stats[home_team]['points'] += 3
                team_stats[home_team]['home_points'] += 3
                team_stats[home_team]['wins'] += 1
                team_stats[away_team]['losses'] += 1
            elif home_goals < away_goals:
                team_stats[away_team]['points'] += 3
                team_stats[away_team]['away_points'] += 3
                team_stats[away_team]['wins'] += 1
                team_stats[home_team]['losses'] += 1
            else:
                team_stats[home_team]['points'] += 1
                team_stats[away_team]['points'] += 1
                team_stats[home_team]['home_points'] += 1
                team_stats[away_team]['away_points'] += 1
                team_stats[home_team]['draws'] += 1
                team_stats[away_team]['draws'] += 1
            
            team_stats[home_team]['gd'] = (team_stats[home_team]['goals_for'] - 
                                         team_stats[home_team]['goals_against'])
            team_stats[away_team]['gd'] = (team_stats[away_team]['goals_for'] - 
                                         team_stats[away_team]['goals_against'])
        
        # If team is not in team_stats (new team), use default stats
        if team not in team_stats:
            stats = {
                'points': 0,
                'gd': 0,
                'goals_for': 0,
                'goals_against': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'home_points': 0,
                'away_points': 0,
                'matches_played': 0,
                'ppg': 0,
                'home_ppg': 0,
                'away_ppg': 0,
                'goals_per_game': 0,
                'goals_against_per_game': 0,
                'position': len(team_stats) + 1,  # Place new teams at the bottom
                'total_teams': max(18, len(team_stats) + 1),  # At least 18 teams
                'relative_position': 0.0  # Bottom position
            }
        else:
            stats = team_stats[team]
            matches_played = max(1, stats['matches_played'])
            
            # Calculate points per game and other averages
            stats['ppg'] = stats['points'] / matches_played
            stats['home_ppg'] = stats['home_points'] / max(1, matches_played/2)
            stats['away_ppg'] = stats['away_points'] / max(1, matches_played/2)
            stats['goals_per_game'] = stats['goals_for'] / matches_played
            stats['goals_against_per_game'] = stats['goals_against'] / matches_played
            
            # Calculate position
            sorted_teams = sorted(
                team_stats.items(),
                key=lambda x: (x[1]['points'], x[1]['gd'], x[1]['goals_for']),
                reverse=True
            )
            stats['position'] = next(i for i, (t, _) in enumerate(sorted_teams, 1) if t == team)
            stats['total_teams'] = len(team_stats)
            
            # Calculate relative position (0-1 scale)
            stats['relative_position'] = (stats['total_teams'] - stats['position']) / (stats['total_teams'] - 1)
        
        return stats
    
    def create_match_features(self, home_team: str, away_team: str, match_date: pd.Timestamp) -> pd.DataFrame:
        """Create enhanced features for a specific match."""
        # Get enhanced team stats
        home_stats = self.calculate_team_stats(home_team, match_date)
        away_stats = self.calculate_team_stats(away_team, match_date)
        h2h_stats = self.get_h2h_stats(home_team, away_team, match_date)
        home_league_stats = self.get_league_position(home_team, match_date)
        away_league_stats = self.get_league_position(away_team, match_date)
        
        # Create feature dictionary with enhanced features
        features = {
            # Basic offensive/defensive stats
            'home_avg_goals_scored': home_stats['avg_goals_scored'],
            'home_avg_goals_conceded': home_stats['avg_goals_conceded'],
            'away_avg_goals_scored': away_stats['avg_goals_scored'],
            'away_avg_goals_conceded': away_stats['avg_goals_conceded'],
            
            # Form and momentum
            'home_form_points': home_stats['form_points'],
            'away_form_points': away_stats['form_points'],
            'home_points_trend': home_stats['points_trend'],
            'away_points_trend': away_stats['points_trend'],
            'home_goals_trend': home_stats['goals_per_game_trend'],
            'away_goals_trend': away_stats['goals_per_game_trend'],
            
            # Win rates
            'home_overall_win_rate': home_stats['win_rate'],
            'home_home_win_rate': home_stats['home_win_rate'],
            'away_overall_win_rate': away_stats['win_rate'],
            'away_away_win_rate': away_stats['away_win_rate'],
            
            # Clean sheets and scoring
            'home_clean_sheet_rate': home_stats['clean_sheets'] / max(1, home_stats['matches_played']),
            'away_clean_sheet_rate': away_stats['clean_sheets'] / max(1, away_stats['matches_played']),
            'home_scoring_rate': 1 - (home_stats['failed_to_score'] / max(1, home_stats['matches_played'])),
            'away_scoring_rate': 1 - (away_stats['failed_to_score'] / max(1, away_stats['matches_played'])),
            
            # Performance stability
            'home_defensive_stability': home_stats['defensive_stability'],
            'away_defensive_stability': away_stats['defensive_stability'],
            'home_offensive_efficiency': home_stats['offensive_efficiency'],
            'away_offensive_efficiency': away_stats['offensive_efficiency'],
            
            # Head-to-head features
            'h2h_home_win_rate': h2h_stats['home_wins'] / max(1, h2h_stats['matches_played']),
            'h2h_draw_rate': h2h_stats['draws'] / max(1, h2h_stats['matches_played']),
            'h2h_avg_goals': h2h_stats['avg_goals'],
            'h2h_home_dominance': h2h_stats['home_team_dominance'],
            'h2h_recent_momentum': h2h_stats['recent_momentum'],
            'h2h_high_scoring_rate': h2h_stats['high_scoring_rate'],
            'h2h_low_scoring_rate': h2h_stats['low_scoring_rate'],
            
            # League position features
            'home_league_position': home_league_stats['position'],
            'away_league_position': away_league_stats['position'],
            'home_relative_position': home_league_stats['relative_position'],
            'away_relative_position': away_league_stats['relative_position'],
            'position_diff': home_league_stats['position'] - away_league_stats['position'],
            'relative_position_diff': home_league_stats['relative_position'] - away_league_stats['relative_position'],
            
            # Points per game
            'home_ppg': home_league_stats['ppg'],
            'away_ppg': away_league_stats['ppg'],
            'home_home_ppg': home_league_stats['home_ppg'],
            'away_away_ppg': away_league_stats['away_ppg'],
            'ppg_diff': home_league_stats['ppg'] - away_league_stats['ppg']
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
    
    def calculate_team_form(self, team: str, before_date: pd.Timestamp, n_matches: int = 5) -> Dict:
        """Calculate detailed form metrics for a team."""
        team_matches = self.df[
            ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
            (self.df['Date'] < before_date)
        ].sort_values('Date', ascending=False).head(n_matches)
        
        stats = {
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'form_points': 0,
            'win_rate': 0,
            'home_win_rate': 0,
            'away_win_rate': 0,
            'clean_sheets': 0,
            'failed_to_score': 0,
            'comeback_wins': 0,
            'leads_lost': 0,
            'avg_possession': 0,
            'avg_shots': 0,
            'shot_accuracy': 0,
            'matches_played': len(team_matches)
        }
        
        if stats['matches_played'] > 0:
            home_games = 0
            away_games = 0
            
            for _, match in team_matches.iterrows():
                is_home = match['HomeTeam'] == team
                
                # Get goals and result
                team_goals = match['FTHG'] if is_home else match['FTAG']
                opp_goals = match['FTAG'] if is_home else match['FTHG']
                
                # Update basic stats
                stats['avg_goals_scored'] += team_goals
                stats['avg_goals_conceded'] += opp_goals
                
                # Clean sheets and scoring
                if opp_goals == 0:
                    stats['clean_sheets'] += 1
                if team_goals == 0:
                    stats['failed_to_score'] += 1
                
                # Win/loss stats
                if is_home:
                    home_games += 1
                    if team_goals > opp_goals:
                        stats['home_win_rate'] += 1
                else:
                    away_games += 1
                    if team_goals > opp_goals:
                        stats['away_win_rate'] += 1
                
                # Points
                if team_goals > opp_goals:
                    stats['form_points'] += 3
                elif team_goals == opp_goals:
                    stats['form_points'] += 1
                
                # Shots and possession if available
                if 'HS' in match and 'AS' in match:
                    shots = match['HS'] if is_home else match['AS']
                    shots_target = match['HST'] if is_home else match['AST']
                    stats['avg_shots'] += shots
                    if shots > 0:
                        stats['shot_accuracy'] += shots_target / shots
                
                if 'HF' in match and 'AF' in match:
                    stats['avg_possession'] += match['HP'] if is_home else match['AP']
            
            # Calculate averages
            stats['avg_goals_scored'] /= stats['matches_played']
            stats['avg_goals_conceded'] /= stats['matches_played']
            stats['form_points'] /= stats['matches_played']
            stats['home_win_rate'] = stats['home_win_rate'] / max(1, home_games)
            stats['away_win_rate'] = stats['away_win_rate'] / max(1, away_games)
            stats['win_rate'] = (stats['home_win_rate'] * home_games + 
                               stats['away_win_rate'] * away_games) / stats['matches_played']
            stats['avg_shots'] /= stats['matches_played']
            stats['shot_accuracy'] /= stats['matches_played']
            stats['avg_possession'] /= stats['matches_played']
        
        return stats
    
    def get_streak_info(self, team: str, before_date: pd.Timestamp) -> Dict:
        """Calculate current streaks."""
        team_matches = self.df[
            ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
            (self.df['Date'] < before_date)
        ].sort_values('Date', ascending=False)
        
        streaks = {
            'unbeaten': 0,
            'winning': 0,
            'scoring': 0,
            'clean_sheet': 0
        }
        
        for _, match in team_matches.iterrows():
            is_home = match['HomeTeam'] == team
            team_goals = match['FTHG'] if is_home else match['FTAG']
            opp_goals = match['FTAG'] if is_home else match['FTHG']
            
            # Check streaks
            if team_goals > opp_goals:
                if streaks['winning'] == streaks['unbeaten']:
                    streaks['winning'] += 1
                streaks['unbeaten'] += 1
            elif team_goals == opp_goals:
                streaks['winning'] = 0
                streaks['unbeaten'] += 1
            else:
                break  # End of unbeaten streak
            
            if team_goals > 0:
                streaks['scoring'] += 1
            else:
                break
                
            if opp_goals == 0:
                streaks['clean_sheet'] += 1
            else:
                break
        
        return streaks

    def create_sequence_features(self, team: str, before_date: pd.Timestamp, n_matches: int = 5) -> np.ndarray:
        """Create sequence of match results for LSTM."""
        team_matches = self.df[
            ((self.df['HomeTeam'] == team) | (self.df['AwayTeam'] == team)) &
            (self.df['Date'] < before_date)
        ].sort_values('Date', ascending=False).head(n_matches)
        
        sequence = []
        for _, match in team_matches.iterrows():
            is_home = match['HomeTeam'] == team
            team_goals = match['FTHG'] if is_home else match['FTAG']
            opp_goals = match['FTAG'] if is_home else match['FTHG']
            
            features = [
                1 if is_home else 0,  # Home/Away
                team_goals,
                opp_goals,
                team_goals - opp_goals,  # Goal difference
                3 if team_goals > opp_goals else (1 if team_goals == opp_goals else 0)  # Points
            ]
            sequence.append(features)
        
        # Pad sequence if needed
        while len(sequence) < n_matches:
            sequence.append([0] * 5)
        
        return np.array(sequence)
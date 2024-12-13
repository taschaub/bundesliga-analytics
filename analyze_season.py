import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from src.feature_engineering import FeatureEngineer
from src.data_preparation import DataPreparator
from src.betting_strategy import BettingStrategy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime
import joblib

def analyze_season(season: int = 2023):
    """Analyze betting decisions for a specific season."""
    # Load and prepare data
    print("Loading data...")
    data_prep = DataPreparator("data/processed/bundesliga_matches_full.csv")
    df = data_prep.prepare_data()
    
    # Load pre-trained model and scaler
    print("Loading model...")
    model = joblib.load('models/model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    
    # Get matches for betting season
    betting_season = df[df['Season'] == season].copy()
    
    # Sort matches by date
    betting_season = betting_season.sort_values('Date')
    
    # Initialize betting strategy
    strategy = BettingStrategy(bankroll=1000)
    
    # Store betting decisions and results
    betting_history = []
    
    # For each match day
    for match_date in betting_season['Date'].unique():
        print(f"\nAnalyzing matches for {match_date.date()}")
        
        # Get matches for this day
        day_matches = betting_season[betting_season['Date'] == match_date]
        
        # Use only data available before this date for feature engineering
        available_data = df[df['Date'] < match_date]
        
        # Engineer features
        feature_eng = FeatureEngineer(available_data)
        features = feature_eng.engineer_features()
        
        # Get metadata columns
        metadata_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Target',
                          'Season', 'Month', 'MatchesInSeason', 'B365H', 'B365D', 'B365A']
        
        # Get numeric features only
        numeric_features = features.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_features if col not in metadata_columns]
        
        # For each match on this day
        for idx, match in day_matches.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            actual_result = match['FTR']
            
            # Get features for this match
            match_features = features[
                (features['HomeTeam'] == home_team) & 
                (features['AwayTeam'] == away_team) &
                (features['Date'] == match_date)
            ]
            
            if len(match_features) == 0:
                print(f"Warning: No features found for match {home_team} vs {away_team}")
                continue
                
            X_match = match_features[feature_columns]
            X_match_scaled = scaler.transform(X_match)
            
            # Get model predictions
            X_match_xgb = xgb.DMatrix(X_match_scaled)
            probs = model.predict(X_match_xgb)[0]
            predicted_probs = {'H': probs[2], 'D': probs[1], 'A': probs[0]}
            
            # Get odds
            odds = {
                'H': match['B365H'],
                'D': match['B365D'],
                'A': match['B365A']
            }
            
            # Analyze betting opportunities
            opportunities = strategy.analyze_betting_opportunity(
                match_date, home_team, away_team, predicted_probs, odds
            )
            
            # Record match information
            match_info = {
                'Date': match_date,
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'ActualResult': actual_result,
                'HomeOdds': odds['H'],
                'DrawOdds': odds['D'],
                'AwayOdds': odds['A'],
                'HomePred': predicted_probs['H'],
                'DrawPred': predicted_probs['D'],
                'AwayPred': predicted_probs['A'],
                'BetPlaced': len(opportunities) > 0,
                'BankrollBefore': strategy.bankroll
            }
            
            # Place bets if opportunities exist
            for bet_type, amount, edge in opportunities:
                strategy.place_bet(
                    match_date, home_team, away_team,
                    predicted_probs, actual_result, odds,
                    bet_type, amount, edge
                )
                match_info['BetType'] = bet_type
                match_info['BetAmount'] = amount
                match_info['Edge'] = edge
                match_info['BetResult'] = bet_type == actual_result
            
            match_info['BankrollAfter'] = strategy.bankroll
            betting_history.append(match_info)
    
    # Convert to DataFrame
    history_df = pd.DataFrame(betting_history)
    
    # Create visualizations
    plt.style.use('default')
    
    # 1. Bankroll Evolution
    plt.figure(figsize=(15, 6))
    plt.plot(history_df['Date'], history_df['BankrollAfter'], 
             marker='o', linestyle='-', linewidth=2)
    plt.title(f'Bankroll Evolution - Season {season}')
    plt.xlabel('Date')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('bankroll_evolution.png')
    plt.close()
    
    # 2. Betting Analysis
    bet_matches = history_df[history_df['BetPlaced']]
    total_matches = len(history_df)
    bets_placed = len(bet_matches)
    
    print(f"\nBetting Analysis for Season {season}")
    print(f"Total Matches: {total_matches}")
    print(f"Bets Placed: {bets_placed} ({bets_placed/total_matches*100:.1f}%)")
    
    if bets_placed > 0:
        print("\nBetting Performance:")
        print(f"Starting Bankroll: ${1000:.2f}")
        print(f"Final Bankroll: ${strategy.bankroll:.2f}")
        print(f"Total Return: {(strategy.bankroll/1000 - 1)*100:.1f}%")
        
        winning_bets = bet_matches[bet_matches['BetResult'] == True]
        print(f"Win Rate: {len(winning_bets)/bets_placed*100:.1f}%")
        
        print("\nBet Size Distribution:")
        print(bet_matches['BetAmount'].describe())
        
        # Save detailed betting history
        bet_matches.to_csv(f'betting_history_season_{season}.csv', index=False)
        
        # Print example bets
        print("\nExample Bets (First 5):")
        for _, bet in bet_matches.head().iterrows():
            print(f"\nMatch: {bet['HomeTeam']} vs {bet['AwayTeam']}")
            print(f"Bet Type: {bet['BetType']}")
            print(f"Bet Amount: ${bet['BetAmount']:.2f}")
            print(f"Edge: {bet['Edge']*100:.1f}%")
            print(f"Result: {'Won' if bet['BetResult'] else 'Lost'}")
        
        return history_df, strategy

if __name__ == "__main__":
    analyze_season(2023) 
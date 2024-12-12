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

def analyze_season(season: int = 2022):
    """Analyze betting decisions for a specific season."""
    # Load and prepare data
    print("Loading data...")
    data_prep = DataPreparator("data/processed/bundesliga_matches_full.csv")
    df = data_prep.prepare_data()
    
    # Split data into training (up to previous season) and test (current season)
    train_df = df[df['Season'] < season].copy()
    test_df = df[df['Season'] == season].copy()
    
    # Engineer features for training data
    print("Engineering features for training data...")
    feature_eng_train = FeatureEngineer(train_df)
    df_features_train = feature_eng_train.engineer_features()
    
    # Engineer features for test data
    print("Engineering features for test data...")
    # Include test data in feature engineering to maintain continuity
    full_df = pd.concat([train_df, test_df]).sort_values('Date')
    feature_eng_full = FeatureEngineer(full_df)
    df_features_full = feature_eng_full.engineer_features()
    
    # Split back into train and test
    df_features_train = df_features_full[df_features_full['Season'] < season]
    df_features_test = df_features_full[df_features_full['Season'] == season]
    
    # Separate features for training vs. metadata
    metadata_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Target',
                       'Season', 'Month', 'MatchesInSeason', 'B365H', 'B365D', 'B365A']
    
    # Get numeric features only
    numeric_features = df_features_train.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_features if col not in metadata_columns]
    
    X_train = df_features_train[feature_columns]
    y_train = df_features_train['Target']
    X_test = df_features_test[feature_columns]
    y_test = df_features_test['Target']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training model...")
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'learning_rate': 0.01,
        'max_depth': 4,
        'min_child_weight': 5,
        'gamma': 0.3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.5,
        'reg_lambda': 1.5,
        'tree_method': 'hist',
        'random_state': 42
    }
    
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=1000)
    
    # Initialize betting strategy
    strategy = BettingStrategy(bankroll=1000)
    
    # Store betting decisions and results
    betting_history = []
    
    # Analyze each match in the test season
    print("\nAnalyzing betting opportunities...")
    for idx, row in test_df.iterrows():
        match_date = pd.to_datetime(row['Date'])
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        actual_result = row['FTR']  # Full Time Result
        
        # Get model predictions
        test_idx = df_features_test.index.get_loc(idx)
        X_match = xgb.DMatrix(X_test_scaled[test_idx:test_idx+1])
        probs = model.predict(X_match)[0]
        predicted_probs = {'H': probs[2], 'D': probs[1], 'A': probs[0]}
        
        # Get odds
        odds = {
            'H': row['B365H'],
            'D': row['B365D'],
            'A': row['B365A']
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
    analyze_season(2022) 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from src.feature_engineering import FeatureEngineer
from src.data_preparation import DataPreparator
from src.betting_strategy import BettingStrategy
import matplotlib.pyplot as plt

def train_model(X_train, y_train, X_val, y_val):
    """Train model to output probabilities."""
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
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, 'eval'), (dtrain, 'train')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    return model

def evaluate_predictions(model, X_test, y_test, df_test, odds_test):
    """Evaluate predictions and simulate betting strategy."""
    print("\nEvaluating predictions and betting strategy...")
    
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Initialize betting strategy with more conservative parameters
    strategy = BettingStrategy(bankroll=1000, kelly_fraction=0.02)
    
    # Reset DataFrame index to make sure we can iterate properly
    df_test = df_test.reset_index(drop=True)
    
    # Simulate betting
    for i in range(len(y_test)):
        match_date = pd.to_datetime(df_test.loc[i, 'Date'])
        home_team = df_test.loc[i, 'HomeTeam']
        away_team = df_test.loc[i, 'AwayTeam']
        
        # Convert probabilities to dict
        probs = {
            'H': y_pred_proba[i, 2],  # Home win probability
            'D': y_pred_proba[i, 1],  # Draw probability
            'A': y_pred_proba[i, 0]   # Away win probability
        }
        
        # Get real Bet365 odds for this match
        match_odds = {
            'H': odds_test.iloc[i]['HomeOdds'],
            'D': odds_test.iloc[i]['DrawOdds'],
            'A': odds_test.iloc[i]['AwayOdds']
        }
        
        # Get actual result
        actual_result = 'H' if y_test.iloc[i] == 2 else ('D' if y_test.iloc[i] == 1 else 'A')
        
        # Analyze betting opportunities
        opportunities = strategy.analyze_betting_opportunity(
            match_date,
            home_team,
            away_team,
            probs,
            match_odds
        )
        
        # Place bets
        for bet_type, bet_amount, edge in opportunities:
            strategy.place_bet(
                match_date,
                home_team,
                away_team,
                probs,
                actual_result,
                match_odds,
                bet_type,
                bet_amount,
                edge
            )
    
    # Get betting summary
    summary = strategy.get_betting_summary()
    
    # Print betting performance
    print("\nBetting Performance Summary:")
    print(f"Total Bets: {summary['total_bets']}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Total Profit: ${summary['total_profit']:.2f}")
    print(f"ROI: {summary['roi']:.1f}%")
    print(f"Final Bankroll: ${summary['final_bankroll']:.2f}")
    
    print("\nMonthly Profits:")
    print(summary['monthly_profits'])
    
    print("\nWin Rates by Bet Type:")
    for bet_type, win_rate in summary['win_rates_by_type'].items():
        print(f"{bet_type}: {win_rate:.1f}%")
    
    # Plot performance
    strategy.plot_performance()
    
    # Calculate overall accuracy metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Performance:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, strategy

def main():
    # Load and prepare data
    print("Loading data...")
    data_prep = DataPreparator("data/processed/bundesliga_matches_full.csv")
    df = data_prep.prepare_data()
    
    # Engineer features
    print("Engineering features...")
    feature_eng = FeatureEngineer(df)
    df_features = feature_eng.engineer_features()
    
    # Separate features for training vs. metadata
    metadata_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Target',
                       'Season', 'Month', 'MatchesInSeason', 'B365H', 'B365D', 'B365A']
    
    # Get numeric features only
    numeric_features = df_features.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_features if col not in metadata_columns]
    
    X = df_features[feature_columns]
    y = df_features['Target']
    
    # Use the last season for testing (proper temporal separation)
    last_season = df_features['Season'].max()
    train_mask = df_features['Season'] < last_season
    test_mask = df_features['Season'] == last_season
    
    # Split data
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    df_test = df_features[test_mask].copy()
    
    # Create odds DataFrame from real Bet365 odds
    odds_test = pd.DataFrame({
        'HomeOdds': df_test['B365H'],
        'DrawOdds': df_test['B365D'],
        'AwayOdds': df_test['B365A']
    })
    
    print(f"\nTraining on seasons: {df_features[train_mask]['Season'].unique()}")
    print(f"Testing on season: {last_season}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Evaluate model and betting strategy
    accuracy, strategy = evaluate_predictions(model, X_test_scaled, y_test, df_test, odds_test)
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Plot monthly profit distribution
    plt.figure(figsize=(12, 6))
    monthly_profits = strategy.get_betting_summary()['monthly_profits']
    monthly_profits.plot(kind='bar')
    plt.title('Monthly Betting Profits')
    plt.xlabel('Month')
    plt.ylabel('Profit ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_profits.png')
    plt.close()
    
    return model, scaler, accuracy, strategy

if __name__ == "__main__":
    main()
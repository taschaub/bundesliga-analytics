import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from src.feature_engineering import FeatureEngineer
from src.data_preparation import DataPreparator

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

def evaluate_model(model, X_test, y_test):
    """Basic model evaluation."""
    print("\nEvaluating model...")
    
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred = np.argmax(model.predict(dtest), axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

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
    
    print(f"\nTraining on seasons: {df_features[train_mask]['Season'].unique()}")
    print(f"Testing on season: {last_season}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return model, scaler, accuracy

if __name__ == "__main__":
    main()
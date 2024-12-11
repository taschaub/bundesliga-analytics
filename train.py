import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from src.feature_engineering import FeatureEngineer
from src.data_preparation import DataPreparator

def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model with focus on learning league-wide patterns."""
    print("Training model...")
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'learning_rate': 0.005,
        'max_depth': 3,
        'min_child_weight': 5,
        'gamma': 0.5,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'tree_method': 'hist',
        'random_state': 42,
        'max_leaves': 8,
        'min_child_samples': 20
    }
    
    # Create DMatrix for faster training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set up evaluation metrics
    evallist = [(dval, 'eval'), (dtrain, 'train')]
    
    # Train model with early stopping
    num_round = 3000
    early_stopping = 100
    
    model = xgb.train(
        params,
        dtrain,
        num_round,
        evallist,
        early_stopping_rounds=early_stopping,
        verbose_eval=100
    )
    
    return model

def evaluate_predictions(model, X_test, y_test):
    """Evaluate model predictions."""
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Performance:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    
    # Calculate class-specific accuracies
    print("\nClass-specific Accuracies:")
    for i, result in enumerate(['Away Win', 'Draw', 'Home Win']):
        class_mask = y_test == i
        class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
        print(f"{result} ({i}): {class_accuracy:.3f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy, {i: accuracy_score(y_test[y_test == i], y_pred[y_test == i]) for i in range(3)}

def analyze_feature_importance(model, feature_names):
    """Analyze and print feature importance."""
    importance_scores = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame(
        [(k, v) for k, v in importance_scores.items()],
        columns=['feature', 'importance']
    )
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    for _, row in importance_df.head(15).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return importance_df

def main():
    # Load and prepare data
    print("Loading data...")
    data_prep = DataPreparator("data/processed/bundesliga_matches_full.csv")
    df = data_prep.prepare_data()
    
    # Engineer features
    print("Engineering features...")
    feature_eng = FeatureEngineer(df)
    df_features = feature_eng.engineer_features()
    
    # Prepare features and target
    feature_columns = [col for col in df_features.columns if col not in [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Target',
        'Season', 'Month', 'MatchesInSeason'
    ]]
    
    X = df_features[feature_columns]
    y = df_features['Target']
    
    # Create time series split (using 80-20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Evaluate model
    accuracy, class_accuracies = evaluate_predictions(model, X_test_scaled, y_test)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(model, feature_columns)
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return model, scaler, accuracy, class_accuracies

if __name__ == "__main__":
    main()
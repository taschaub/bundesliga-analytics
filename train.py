import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from src.data_preparation import DataPreparator
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import ModelEvaluator
import matplotlib.pyplot as plt

def train_model(model_type: str = 'random_forest'):
    """Train and evaluate the prediction model."""
    # Load and prepare data
    data_prep = DataPreparator('data/processed/bundesliga_matches_full.csv')
    df = data_prep.prepare_data()
    
    # Create evaluator
    evaluator = ModelEvaluator(df)
    
    # Split based on seasons (use last season for testing)
    seasons = sorted(df['Season'].unique())
    train_df, test_df = evaluator.split_seasons([seasons[-1]])
    
    print(f"Training on seasons: {', '.join(sorted(train_df['Season'].unique()))}")
    print(f"Testing on season: {test_df['Season'].unique()[0]}")
    
    # Engineer features
    feat_eng = FeatureEngineer(train_df)
    X_train, feature_names = feat_eng.get_feature_matrix()
    y_train = train_df['Target']
    
    feat_eng_test = FeatureEngineer(test_df)
    X_test, _ = feat_eng_test.get_feature_matrix()
    y_test = test_df['Target']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid for RandomForest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }
    
    # Initialize and train model with grid search
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    print("\nPerforming grid search...")
    grid_search.fit(X_train_scaled, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    # Evaluate
    results = evaluator.evaluate_model(y_test, y_pred, y_prob)
    
    # Print results
    print("\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Average Prediction Confidence: {results['avg_confidence']:.3f}")
    print(f"Average Confidence when Correct: {results['correct_confidence']:.3f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save everything needed for predictions
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save evaluation plots
    evaluator.plot_confusion_matrix(results['confusion_matrix']).savefig('models/confusion_matrix.png')
    evaluator.plot_prediction_distribution(y_prob).savefig('models/prediction_distribution.png')
    
    # Analyze feature importance
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importances.head(10))
    
    # Save feature importances plot
    plt.figure(figsize=(12, 6))
    plt.bar(importances['feature'][:10], importances['importance'][:10])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('models/feature_importances.png')

if __name__ == "__main__":
    train_model('random_forest')
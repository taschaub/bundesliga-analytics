import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.data_preparation import DataPreparator
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import ModelEvaluator

def train_model(model_type: str = 'random_forest'):
    """
    Train and evaluate the prediction model.
    
    Args:
        model_type: Either 'random_forest' or 'logistic'
    """
    # Load and prepare data
    data_prep = DataPreparator('data/bundesliga_matches_full.csv')
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
    
    # Initialize model
    if model_type == 'logistic':
        model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    # Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Evaluate
    results = evaluator.evaluate_model(y_test, y_pred, y_prob)
    
    # Print results
    print("\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Average Prediction Confidence: {results['avg_confidence']:.3f}")
    print(f"Average Confidence when Correct: {results['correct_confidence']:.3f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save evaluation plots
    evaluator.plot_confusion_matrix(results['confusion_matrix']).savefig('models/confusion_matrix.png')
    evaluator.plot_prediction_distribution(y_prob).savefig('models/prediction_distribution.png')
    
    # If using random forest, save feature importances
    if model_type == 'random_forest':
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 Most Important Features:")
        print(importances.head(10))

if __name__ == "__main__":
    train_model('random_forest')  # or 'logistic'
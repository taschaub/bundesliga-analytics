import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data_preparation import DataPreparator
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import ModelEvaluator

def train_model(model_type: str = 'logistic'):
    # Load and prepare data
    data_prep = DataPreparator('data/bundesliga_matches_full.csv')
    df = data_prep.prepare_data()
    
    # Create evaluator
    evaluator = ModelEvaluator(df)
    
    # Split based on seasons (use last season for testing)
    seasons = sorted(df['Season'].unique())
    train_df, test_df = evaluator.split_seasons([seasons[-1]])
    
    # Engineer features
    feat_eng = FeatureEngineer(train_df)
    X_train, feature_names = feat_eng.get_feature_matrix()
    y_train = train_df['Target']
    
    feat_eng_test = FeatureEngineer(test_df)
    X_test, _ = feat_eng_test.get_feature_matrix()
    y_test = test_df['Target']
    
    # Initialize model
    if model_type == 'logistic':
        model = LogisticRegression(multi_class='multinomial', random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    results = evaluator.evaluate_model(y_test, y_pred, y_prob)
    
    # Print results
    print("\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save evaluation plots
    evaluator.plot_confusion_matrix(results['confusion_matrix']).savefig('models/confusion_matrix.png')
    evaluator.plot_prediction_distribution(y_prob, y_test).savefig('models/prediction_distribution.png')

if __name__ == "__main__":
    train_model('random_forest')  # or 'logistic'
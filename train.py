import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.data_preparation import DataPreparator
from src.feature_engineering import FeatureEngineer
import os

def train_model():
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)  # Create the directory if it doesn't exist
    # Load and prepare data
    data_prep = DataPreparator('data/bundesliga_matches.csv')
    df = data_prep.prepare_data()
    
    # Engineer features
    feat_eng = FeatureEngineer(df)
    X, feature_names = feat_eng.get_feature_matrix()
    y = df['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression(multi_class='multinomial', random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and feature names
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Print basic evaluation
    print(f"Model accuracy: {model.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    train_model() 
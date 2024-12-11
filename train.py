import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation import DataPreparator
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import ModelEvaluator

def train_xgboost(X_train, y_train, param_grid, cv):
    """Train XGBoost model with custom cross-validation."""
    best_score = -np.inf
    best_params = None
    best_model = None
    
    total_combinations = (len(param_grid['n_estimators']) * 
                        len(param_grid['max_depth']) * 
                        len(param_grid['learning_rate']))
    current_combination = 0
    
    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                current_combination += 1
                print(f"\nTrying combination {current_combination}/{total_combinations}")
                print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
                
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'eval_metric': 'mlogloss',
                    'random_state': 42,
                    'tree_method': 'hist'  # For faster training
                }
                
                model = xgb.XGBClassifier(**params)
                scores = []
                
                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train), 1):
                    print(f"  Fold {fold}/{cv.n_splits}", end='\r')
                    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
                    
                    model.fit(
                        X_train_fold, 
                        y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        verbose=True
                    )
                    
                    pred = model.predict(X_val_fold)
                    score = np.mean(pred == y_val_fold)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                print(f"\nAverage score: {avg_score:.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params
                    best_model = model
                    print(f"New best score: {best_score:.3f}")
    
    return best_model, best_params, best_score

def train_model(model_type: str = 'ensemble'):
    """Train and evaluate the prediction model with enhanced features and training process."""
    # Load and prepare data
    print("Loading and preparing data...")
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
    print("Engineering features...")
    feat_eng = FeatureEngineer(train_df)
    X_train, feature_names = feat_eng.get_feature_matrix()
    y_train = train_df['Target']
    
    feat_eng_test = FeatureEngineer(test_df)
    X_test, _ = feat_eng_test.get_feature_matrix()
    y_test = test_df['Target']
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute class weights for balanced training
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Initialize models and parameter grids
    cv = TimeSeriesSplit(n_splits=5)
    best_models = {}
    
    if model_type == 'ensemble' or model_type == 'rf':
        print("\nTraining Random Forest model...")
        rf_model = RandomForestClassifier(
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        )
        
        rf_param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=rf_param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        rf_grid_search.fit(X_train_scaled, y_train)
        best_models['rf'] = rf_grid_search.best_estimator_
        print(f"Best RF parameters: {rf_grid_search.best_params_}")
        print(f"Best RF score: {rf_grid_search.best_score_:.3f}")
    
    if model_type == 'ensemble' or model_type == 'xgb':
        print("\nTraining XGBoost model...")
        xgb_param_grid = {
            'n_estimators': [200],
            'max_depth': [3, 5],
            'learning_rate': [0.1]
        }
        
        best_xgb, best_xgb_params, best_xgb_score = train_xgboost(
            X_train_scaled, y_train, xgb_param_grid, cv
        )
        best_models['xgb'] = best_xgb
        print(f"Best XGB parameters: {best_xgb_params}")
        print(f"Best XGB score: {best_xgb_score:.3f}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = {}
    probabilities = {}
    
    for name, model in best_models.items():
        predictions[name] = model.predict(X_test_scaled)
        probabilities[name] = model.predict_proba(X_test_scaled)
    
    # Combine predictions for ensemble
    if model_type == 'ensemble':
        ensemble_probs = np.mean([probs for probs in probabilities.values()], axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        final_predictions = ensemble_preds
        final_probabilities = ensemble_probs
    else:
        model_name = list(best_models.keys())[0]
        final_predictions = predictions[model_name]
        final_probabilities = probabilities[model_name]
    
    # Evaluate predictions
    print("\nModel Performance:")
    print(f"Accuracy: {evaluator.evaluate_model(y_test, final_predictions, final_probabilities)['accuracy']:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, final_predictions))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, final_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    # Feature importance analysis (for RF only)
    if 'rf' in best_models:
        rf_model = best_models['rf']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png')
        plt.close()
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    # Save models
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    
    if model_type == 'ensemble':
        for name, model in best_models.items():
            joblib.dump(model, f'models/{name}_model.pkl')
    else:
        model_name = list(best_models.keys())[0]
        joblib.dump(best_models[model_name], 'models/model.pkl')
    
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return best_models, scaler

if __name__ == "__main__":
    train_model('ensemble')
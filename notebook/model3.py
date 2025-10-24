import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor # <-- New model added
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# STATION 1: DATA PREPARATION
# ==============================================================================
def prepare_data(comp_path, orig_path, test_path):
    """Loads all data, performs feature engineering, and prepares it for modeling."""
    print("--- 1. Loading and Preparing Data ---")
    
    # Load data
    comp_df = pd.read_csv(comp_path)
    original_df = pd.read_csv(orig_path)
    test_df = pd.read_csv(test_path)
    test_ids = test_df['id']

    # Combine training data
    comp_df = comp_df.drop('id', axis=1)
    comp_df['is_original'] = 0
    original_df['is_original'] = 1
    full_train_df = pd.concat([comp_df, original_df], ignore_index=True)

    datasets = [full_train_df, test_df]

    # Apply feature engineering to both train and test data
    for df in datasets:
        df['gm_risk_feature'] = (
            0.3 * df["curvature"] + 0.2 * (df["lighting"] == "night").astype(int) +
            0.1 * (df["weather"] != "clear").astype(int) + 0.2 * (df["speed_limit"] >= 60).astype(int) +
            0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
        )
        for col in df.select_dtypes(include='bool').columns:
            df[col] = df[col].astype(int)

    # Separate features and target
    X = full_train_df.drop('accident_risk', axis=1)
    y = full_train_df['accident_risk']
    test_features = test_df.drop('id', axis=1)

    # Encoding and Alignment
    X = pd.get_dummies(X, columns=X.select_dtypes(include='object').columns, drop_first=True)
    test_features = pd.get_dummies(test_features, columns=test_features.select_dtypes(include='object').columns, drop_first=True)
    X_aligned, test_aligned = X.align(test_features, join='left', axis=1, fill_value=0)
    
    print("Data preparation complete!")
    return X_aligned, y, test_aligned, test_ids

# ==============================================================================
# STATION 2: QUALITY CONTROL (CROSS-VALIDATION)
# ==============================================================================
def run_cross_validation(model, model_name, X, y):
    """Runs a 5-fold cross-validation for a given model and returns the average RMSE."""
    print(f"\n--- 2. Running CV for {model_name} ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)
        
    avg_rmse = np.mean(rmse_scores)
    print(f"Average CV RMSE for {model_name}: {avg_rmse}")
    return avg_rmse

# ==============================================================================
# STATION 3: FINAL ASSEMBLY (SUBMISSION)
# ==============================================================================
def create_submission(models, weights, X_train, y_train, X_test, test_ids, filename):
    """Trains final models, blends predictions, and creates a submission file."""
    print(f"\n--- 3. Creating Submission File: {filename} ---")
    
    final_preds = {}
    for name, model in models.items():
        print(f"Training final {name} model...")
        model.fit(X_train, y_train)
        final_preds[name] = model.predict(X_test)
    
    # Blend predictions using the provided weights
    blended_predictions = np.zeros_like(final_preds['xgb'])
    for name, weight in weights.items():
        blended_predictions += final_preds[name] * weight
        
    submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': blended_predictions})
    submission_df.to_csv(filename, index=False)
    print(f"Submission file '{filename}' created successfully!")
    print(submission_df.head())

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # Define file paths
    COMP_PATH = 'train.csv'
    ORIG_PATH = 'synthetic_road_accidents_100k.csv'
    TEST_PATH = 'test.csv'

    # Run the full data preparation pipeline
    X_aligned, y, test_aligned, test_ids = prepare_data(COMP_PATH, ORIG_PATH, TEST_PATH)

    # Define our tuned models
    tuned_xgb = XGBRegressor(n_estimators=617, learning_rate=0.01636, max_depth=8,
                             subsample=0.854, colsample_bytree=0.747, random_state=42)
    
    tuned_lgbm = LGBMRegressor(n_estimators=697, learning_rate=0.0518, num_leaves=71,
                               max_depth=7, min_child_samples=100, subsample=0.916,
                               colsample_bytree=0.724, verbosity=-1, random_state=42)
    
    # Add our new model, CatBoost, with default settings for now
    catboost = CatBoostRegressor(random_state=42, verbose=0) # verbose=0 keeps it quiet

    # --- Let's quickly validate each model's individual performance ---
    run_cross_validation(tuned_xgb, "Tuned XGBoost", X_aligned, y)
    run_cross_validation(tuned_lgbm, "Tuned LightGBM", X_aligned, y)
    run_cross_validation(catboost, "CatBoost", X_aligned, y)
    
    # Define the final models and weights for our 3-model ensemble
    # A 40/40/20 split is a good starting point
    final_models = {
        'xgb': tuned_xgb,
        'lgbm': tuned_lgbm,
        'cat': catboost
    }
    
    final_weights = {
        'xgb': 0.4,
        'lgbm': 0.4,
        'cat': 0.2
    }
    
    # Run the final submission pipeline
    create_submission(final_models, final_weights, X_aligned, y, test_aligned, test_ids, 'final_3_model_blend.csv')
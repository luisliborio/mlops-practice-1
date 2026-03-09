# run.py
import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# project configuration
import config

def main():
    print(f"Starting Experiment: [{config.RUN_TAG}]")
    
    # Check execution mode based on config
    pretrained_path = getattr(config, 'PRETRAINED_MODEL_PATH', None)
    is_eval_mode = pretrained_path is not None and os.path.exists(pretrained_path)
    
    if is_eval_mode:
        print(f"   [MODE: EVALUATION] Loading pre-trained model from: {pretrained_path}")
    else:
        print(f"   [MODE: TRAINING] Training model from scratch...")

    # ---------------------------------------------------------
    # 1. Data Loading & Preparation
    # ---------------------------------------------------------
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(f"File not found: {config.DATASET_PATH}")
    
    df = pd.read_parquet(config.DATASET_PATH)
    
    # Select features and target
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].copy()
    
    # Basic Cleaning (DropNA)
    valid_indices = X.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]    

    # ---------------------------------------------------------
    # 2. Split
    # ---------------------------------------------------------
    # keep X_test locked for comparison later
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED
    )
    
    print(f"   Data Split: {X_train_full.shape[0]} Train samples | {X_test.shape[0]} Test samples")

    # =========================================================
    # OPTION 2: EVALUATION MODE (Load Pre-trained Model)
    # =========================================================
    if is_eval_mode:
        # Initialize an empty model and load the saved state
        final_model = xgb.XGBRegressor()
        final_model.load_model(pretrained_path)
        
        # Predict on Test Set
        y_pred_test = final_model.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"     Pre-trained Test Set Performance: MAE=${test_mae:.2f} | RMSE=${test_rmse:.2f}")
        
        # Prepare mock variables for the report since we skipped training
        final_params = "Loaded from pre-trained model"
        cv_mae_mean = None
        cv_rmse_mean = None
        avg_best_epoch = None

    # =========================================================
    # OPTION 1: TRAINING MODE (From Scratch)
    # =========================================================
    else:
        # ---------------------------------------------------------
        # 3. CV Loop (Capturing Best Iterations)
        # ---------------------------------------------------------
        print(f"   Running {config.CV_FOLDS}-Fold Cross-Validation...")
        
        kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
        
        cv_scores = {'mae': [], 'rmse': [], 'r2': []}
        best_iterations = [] # Store optimal step for each fold
        
        fold = 1
        for train_idx, val_idx in kf.split(X_train_full):
            X_fold_train, X_fold_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_fold_train, y_fold_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
            
            model = xgb.XGBRegressor(**config.MODEL_PARAMS)
            
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)],
                verbose=False
            )
            
            # Capture the best iteration
            best_step = model.best_iteration + 1
            best_iterations.append(best_step)
            
            # Predict
            preds_val = model.predict(X_fold_val)
            rmse_val = np.sqrt(mean_squared_error(y_fold_val, preds_val))
            mae_val = mean_absolute_error(y_fold_val, preds_val)
            r2_val = r2_score(y_fold_val, preds_val)
            
            cv_scores['rmse'].append(rmse_val)
            cv_scores['mae'].append(mae_val)
            cv_scores['r2'].append(r2_val)
            
            print(f"     Fold {fold}: Best Iteration={best_step} | MAE=${mae_val:.2f} | RMSE=${rmse_val:.2f}")
            fold += 1

        # Calculate Average Optimal Epoch
        avg_best_epoch = int(np.mean(best_iterations))
        std_best_epoch = np.std(best_iterations)
        cv_mae_mean = np.mean(cv_scores['mae'])
        cv_rmse_mean = np.mean(cv_scores['rmse'])
        
        print("-" * 40)
        print(f"    CV Results: Avg MAE=${cv_mae_mean:.2f}")
        print(f"    Optimal Training Epochs found: {best_iterations}")
        print(f"    FINAL STRATEGY: Retraining for fixed {avg_best_epoch} epochs (std: {std_best_epoch:.1f})")
        print("-" * 40)

        # ---------------------------------------------------------
        # 4. Final Training (Fixed Epochs, No Early Stopping)
        # ---------------------------------------------------------
        print("   Retraining on 100% of Training Data...")
        
        # Update params to force exact number of estimators
        final_params = config.MODEL_PARAMS.copy()
        final_params['n_estimators'] = avg_best_epoch
        final_params['early_stopping_rounds'] = None
        
        final_model = xgb.XGBRegressor(**final_params)
        
        final_model.fit(
            X_train_full, y_train_full,
            verbose=False
        )
        
        # Predict on Test Set
        y_pred_test = final_model.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"     Final Test Set Performance: MAE=${test_mae:.2f} | RMSE=${test_rmse:.2f}")

    # ---------------------------------------------------------
    # 5. Export Report (Applies to BOTH modes)
    # ---------------------------------------------------------
    if not os.path.exists(config.REPORTS_DIR):
        os.makedirs(config.REPORTS_DIR)
        
    report_data = {
        'tag': config.RUN_TAG,
        'features': config.FEATURES,
        'params': final_params,
        'cv_metrics': {'mae_mean': cv_mae_mean, 'rmse_mean': cv_rmse_mean},
        'test_metrics': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2},
        'y_test_true': y_test.values,
        'y_test_pred': y_pred_test,
        'avg_best_epoch': avg_best_epoch
    }
    
    output_file = os.path.join(config.REPORTS_DIR, f"{config.RUN_TAG}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(report_data, f)
        
    print(f"     Report saved to: {output_file}")
    
    # ---------------------------------------------------------
    # 6. Save Model (ONLY in Training mode)
    # ---------------------------------------------------------
    if not is_eval_mode:
        model_path = getattr(config, 'MODEL_PATH', None)
        if model_path is not None:
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            final_model.save_model(model_path)
            print(f"     Model saved to: {model_path}")
            
    print("="*60)

if __name__ == "__main__":
    main()
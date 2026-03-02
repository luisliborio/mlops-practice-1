# run.py
import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# project configuration
import config

def main():
    print(f"Starting Classification Experiment: [{config.RUN_TAG}]")
    
    # ---------------------------------------------------------
    # 1. Data Loading & Preparation
    # ---------------------------------------------------------
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(f"File not found: {config.DATASET_PATH}")
    
    df = pd.read_parquet(config.DATASET_PATH)
    
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].copy()
    
    # Basic Cleaning (DropNA)
    valid_indices = X.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    # Target Transformation
    # Raw: 1=Credit, 2=Cash, 3=No Charge, 4=Dispute
    # Step 1: Merge 3 and 4 into 'Other' (mapped to 3 initially)
    y = y.replace({4: 3, 5:3})
    # Step 2: Shift to 0-based index for XGBoost
    # New Mapping: 0=Credit, 1=Cash, 2=Other
    y = y - 1
    
    print(f"Y UNIQUE: {np.unique(y)}")

    # ---------------------------------------------------------
    # 2. Split
    # ---------------------------------------------------------
    # keep X_test locked for comparison later
    # Stratify is critical for unbalanced classes
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED,
        stratify=y
    )
    
    print(f"   Data Split: {X_train_full.shape[0]} Train samples | {X_test.shape[0]} Test samples")

    # ---------------------------------------------------------
    # 3. CV Loop (Capturing Best Iterations)
    # ---------------------------------------------------------
    print(f"   Running {config.CV_FOLDS}-Fold Cross-Validation...")
    
    kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    
    cv_scores = {'f1': [], 'acc': []}
    best_iterations = [] 
    
    fold = 1
    for train_idx, val_idx in kf.split(X_train_full, y_train_full):
        X_fold_train, X_fold_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        model = xgb.XGBClassifier(**config.MODEL_PARAMS)
        
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)],
            verbose=False
        )
        
        best_step = model.best_iteration + 1
        best_iterations.append(best_step)
        
        preds_val = model.predict(X_fold_val)
        
        # Weighted F1 is standard for unbalanced multiclass
        f1 = f1_score(y_fold_val, preds_val, average='weighted')
        acc = accuracy_score(y_fold_val, preds_val)
        
        cv_scores['f1'].append(f1)
        cv_scores['acc'].append(acc)
        
        print(f"     Fold {fold}: Best Iteration={best_step} | F1-Weighted={f1:.4f}")
        fold += 1

    avg_best_epoch = int(np.mean(best_iterations))
    
    print("-" * 40)
    print(f"   CV Results: Avg F1-Weighted={np.mean(cv_scores['f1']):.4f}")
    print(f"   Final Strategy: Retrain for {avg_best_epoch} epochs")
    print("-" * 40)

    # ---------------------------------------------------------
    # 4. Final Training
    # ---------------------------------------------------------
    # Update params to force exact number of estimators
    final_params = config.MODEL_PARAMS.copy()
    final_params['n_estimators'] = avg_best_epoch
    final_params['early_stopping_rounds'] = None
    
    final_model = xgb.XGBClassifier(**final_params)
    
    final_model.fit(
        X_train_full, y_train_full,
        verbose=False
    )
    
    # Predict on Test Set
    y_pred_test = final_model.predict(X_test)
    
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_acc = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    
    print(f"     Final Test Performance: F1-Weighted={test_f1:.4f} | Acc={test_acc:.4f}")

    # ---------------------------------------------------------
    # 5. Export Report
    # ---------------------------------------------------------
    if not os.path.exists(config.REPORTS_DIR):
        os.makedirs(config.REPORTS_DIR)
        
    report_data = {
        'tag': config.RUN_TAG,
        'features': config.FEATURES,
        'params': final_params,
        'cv_metrics': {'f1_mean': np.mean(cv_scores['f1'])},
        'test_metrics': {'f1': test_f1, 'acc': test_acc, 'conf_matrix': conf_matrix},
        'y_test_true': y_test.values,
        'y_test_pred': y_pred_test,
        'class_map': {0: 'Credit', 1: 'Cash', 2: 'Other'}
    }
    
    output_file = os.path.join(config.REPORTS_DIR, f"{config.RUN_TAG}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(report_data, f)
        
    print(f"   Report saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()
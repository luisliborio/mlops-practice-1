# config.py
import os

# ==========================================
# 1. RUN IDENTITY
# ==========================================
# e.g., 'xgb_cls_baseline', 'xgb_cls_tuned'
RUN_TAG = "xgb_cls_5feats"

# ==========================================
# 2. DATA PATHS
# ==========================================
DATASET_PATH = "data/training_taxi_data.parquet" # inputs
REPORTS_DIR = "runs/classification/1"            # outputs

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
FEATURES = [
    'fare_amount',   # 3feats run
    'trip_distance', # 3feats run
    'PULocationID',  # 3feats run
    'pickup_hour',    # 4feats run
    'trip_type',     # 5feats run
]

TARGET = 'payment_type'

# ==========================================
# 4. MODEL HYPERPARAMETERS (XGBClassifier)
# ==========================================
# Note: eval_metric and early_stopping are handled inside the CV loop in run.py
MODEL_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'multi:softmax',
    'num_class': 3,               # 0=Credit, 1=Cash, 2=Other (merged 3+4)
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 50,
    'n_jobs': -1,
    'random_state': 42
}

# ==========================================
# 5. TRAINING STRATEGY
# ==========================================
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
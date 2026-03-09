# config.py
import os

# ==========================================
# 1. RUN IDENTITY
# ==========================================
# e.g., 'xgb_baseline', 'xgb_4_feats'
RUN_TAG = "xgb_5feats_v4_data02"

# ==========================================
# 2. DATA PATHS
# ==========================================
DATASET_PATH = "data/training_taxi_data.parquet" # inputs
REPORTS_DIR = "runs/regression/4"                # outputs
PRETRAINED_MODEL_PATH = None #f"models/regression/xgb_5feats_v3_data01.pkl" # evaluate model only
MODEL_PATH = f"models/regression/{RUN_TAG}.pkl" # None for not saving

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
# List of features to be used in the current run
FEATURES = [
    'fare_amount',       # included at 3feats
    'duration_minutes',  # included at 3feats
    'trip_distance',     # included at 3feats
    'PULocationID',    # included at 4feats
    'DOLocationID',    # included at 5feats
    # 'pickup_hour'      # not tested
]

TARGET = 'tip_amount'

# ==========================================
# 4. MODEL HYPERPARAMETERS (XGBoost)
# ==========================================
# Note: eval_metric and early_stopping are handled inside the CV loop in run.py
MODEL_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'reg:squarederror',
    'early_stopping_rounds': 50,
    'eval_metric': 'rmse',
    'n_jobs': -1,
    'random_state': 42
}

# ==========================================
# 5. TRAINING STRATEGY
# ==========================================
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5  # Number of Cross-Validation folds
EARLY_STOPPING_ROUNDS = 50
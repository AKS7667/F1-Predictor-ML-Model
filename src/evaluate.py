import pandas as pd
import numpy as np
import xgboost as xgb
from train import split_features_target, evaluate, CONFIG   

# 1. load saved features ( parquet produced by features.py )
df = pd.read_parquet("data/f1_features.parquet")

# 2. recreate the same test split train.py used
test_df = df[df["Season"].isin(CONFIG["test_seasons"])]     
X_test, y_test, groups_test = split_features_target(test_df)

# 3. load the saved models
fold_models = []
for i in range(5):
    m = xgb.XGBRegressor()
    m.load_model(f"models/fold_{i}.json")
    fold_models.append(m)

# 4. score with the same evaluate() train.py uses
results = evaluate(fold_models, X_test, y_test, groups_test)

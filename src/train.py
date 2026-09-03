"""
================================================================================
F1 RACE FINISHING POSITION PREDICTOR
================================================================================

GOAL
----
Predict the finishing position of each driver in a Formula 1 Grand Prix using
historical race data (qualifying position, recent form, constructor strength,
circuit characteristics, weather, etc.).

SUCCESS METRICS

We evaluate the model on a held-out set of recent races using:
  - MAE (Mean Absolute Error) on finishing position
      Target: < 3.0 positions on average.
      Rationale: Naive baseline (predict grid position) typically scores
      ~3.5-4.0 MAE. Beating this meaningfully is the bar for "useful".

  - Spearman rank correlation per race (represents how well we predict order)
      Target: > 0.65 averaged across test races.

  - Top-3 (podium) hit rate
      Target: > 60% of actual podium finishers in our predicted top-3.

  - Top-10 (points) hit rate
      Target: > 75%.

"""

import numpy as np
import pandas as pd
import os


from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# XGBoost is our chosen learner. 
import xgboost as xgb


# Centralizing config here
CONFIG = {

    "data_path": "data/f1_features.parquet",

    # Seasons used for training. 
    "train_seasons": list(range(2014, 2025)),  # turbo-hybrid era only;
                                              
    "test_seasons":  [2025],

    # ---- Model hyperparameters ----

    "xgb_params": {
        # Regression with squared error: matches our MAE target
        "objective": "reg:squarederror",

        # Number of boosting rounds. Avoids
        # under- or over-fitting from a hard-coded round count.
        "n_estimators": 2000,

        # Tree depth. Depths >8 overfit quickly on
        # datasets of this size.
        "max_depth": 6,

        # Learning rate. Small LR + many trees = standard recipe for stable
        # boosting.
        "learning_rate": 0.05,

        # Row subsampling per tree. Acts as regularisation — each tree sees
        # 80% of rows, which decorrelates trees and reduces variance. 
        "subsample": 0.8,

        # Column subsampling per tree. Same idea applied to features.
        "colsample_bytree": 0.8,

        # L2 regularisation on leaf weights. Discourages any single leaf
        # from dominating predictions; 
        "reg_lambda": 1.0,

        # Minimum sum of instance weights in a child node. Prevents the tree
        # from creating leaves based on, say, a single freak race. 
        "min_child_weight": 3,

        # Reproducibility. Without this, every run gives slightly different
        # numbers.
        "random_state": 42,

        # Use histogram-based algorithm: ~5-10x faster than the exact algo
        "tree_method": "hist",

        # Suppress per-iteration training spam; we'll log via callbacks.
        "verbosity": 1,
    },

    #  Cross-validation 

    "cv_folds": 5,

    # Early stopping rounds: stop if validation MAE doesn't improve for
    # this many iterations. 
    "early_stopping_rounds": 50,
}



# DATA LOADING

def load_data(path: str) -> pd.DataFrame:
    
    # Load the pre-processed race-level dataset.
    
    df = pd.read_parquet(path)
    df["race_id"] = df["Season"].astype(str) + "_" + df["Round"].astype(str)

    # Sanity checks. These have caught bugs in upstream ETL more than once.
    assert df["Position"].between(1, 24).all(), \
        "finishing_position out of expected range — check DNF handling in ETL"
    assert df["race_id"].notna().all(), "race_id has nulls — joins broke upstream"

    return df


def split_features_target(df: pd.DataFrame):
    """
    Separate the dataframe into:
      X  — feature matrix
      y  — target vector (finishing_position)
      groups — race_id, used for group-aware CV

    Categorical columns are converted to pandas 'category' dtype so XGBoost's
    native categorical support can handle them without one-hot blowup.
    """
    df.loc[:, "race_id"] = df["Season"].astype(str) + "_" + df["Round"].astype(str)

    target_col = "Position"
 
    drop_cols = ["Position", "Season", "Round", "race_id",
             "Abbreviation", "TeamName", "Location", "Circuit"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    groups = df["race_id"].copy()

    # Convert object/string columns to categorical for XGBoost native handling.
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    return X, y, groups


# =============================================================================
# TRAINING
# =============================================================================
def train_with_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, config: dict):
    """
    Train XGBoost using GroupKFold cross-validation. To avoid leaking information between races, we group by race_id. This ensures that all drivers from
  the same race are either in the training set or the validation set, but never split across both. 

    Returns a list of trained models (one per fold) plus per-fold metrics.
    Ensembling fold models at inference time gives an MAE improvement
    over picking the single best fold model (variance reduction).
    """
    cv = GroupKFold(n_splits=config["cv_folds"])
    fold_models = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # enable_categorical=True is required for the native cat handling we
        # set up above. 
        model = xgb.XGBRegressor(
            **config["xgb_params"],
            enable_categorical=True,
            early_stopping_rounds=config["early_stopping_rounds"],
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        fold_metrics.append({"fold": fold_idx, "mae": mae,
                             "best_iter": model.best_iteration})
        fold_models.append(model)

        print(f"[fold {fold_idx}] MAE={mae:.3f}  best_iter={model.best_iteration}")

        os.makedirs("models", exist_ok=True)
        for i, m in enumerate(fold_models):
            m.save_model(f"models/fold_{i}.json")

    return fold_models, fold_metrics


# EVALUATION

def evaluate(models, X_test: pd.DataFrame, y_test: pd.Series,
             race_ids: pd.Series) -> dict:
    """
    Evaluate the ensemble on held-out races.

    We compute the four success metrics defined at the top of the file. The
    rank-based metrics (Spearman, top-k hit rates) are computed *per race*
    """
    # Average predictions across fold models — simple uniform ensemble.

    preds = np.mean([m.predict(X_test) for m in models], axis=0)

    overall_mae = mean_absolute_error(y_test, preds)
    rank_mae = mean_absolute_error(y_test.rank(), pd.Series(preds).rank())

    # Per-race metrics
    df_eval = pd.DataFrame({
        "race_id": race_ids.values,
        "y_true": y_test.values,
        "y_pred": preds,
    })

    spearman_per_race = []
    podium_hits = []
    points_hits = []
    rank_mae=[]

    for _, race in df_eval.groupby("race_id"):
        # Spearman rank correlation between predicted and actual finishing
        # order. 
        rho, _ = spearmanr(race["y_true"], race["y_pred"])
        spearman_per_race.append(rho)

        # Hit rate = |predicted top-k ∩ actual top-k| / k
        actual_top3 = set(race.nsmallest(3, "y_true").index)
        pred_top3   = set(race.nsmallest(3, "y_pred").index)
        podium_hits.append(len(actual_top3 & pred_top3) / 3)

        actual_top10 = set(race.nsmallest(10, "y_true").index)
        pred_top10   = set(race.nsmallest(10, "y_pred").index)
        points_hits.append(len(actual_top10 & pred_top10) / 10)

        race["y_rank"] = race["y_pred"].rank(method="first")
        rank_mae_per_race = mean_absolute_error(race["y_true"].rank(method="first"), race["y_rank"])
        rank_mae.append(rank_mae_per_race)

    return {
        "mae": overall_mae,
        "rank_mae": np.mean(rank_mae),
        "spearman_mean": np.mean(spearman_per_race),
        "podium_hit_rate": np.mean(podium_hits),
        "points_hit_rate": np.mean(points_hits),
    }

# MAIN

def main():
    # 1. Load
    df = load_data(CONFIG["data_path"])

    # 2. Train/test split by season — see CONFIG comment on why not random.
    train_df = df[df["Season"].isin(CONFIG["train_seasons"])]
    train_df = train_df.dropna(subset=["Position"])           # incomplete races can't train
    
    test_df  = df[df["Season"].isin(CONFIG["test_seasons"])]
    test_df  = test_df.dropna(subset=["Position"])            # unscored races can't be evaluated

    X_train, y_train, groups_train = split_features_target(train_df)
    X_test,  y_test,  groups_test  = split_features_target(test_df)


    # 3. Train with grouped CV
    print(f"Training on seasons: {CONFIG['train_seasons']}")
    models, fold_metrics = train_with_cv(X_train, y_train, groups_train, CONFIG)

    cv_mae = np.mean([m["mae"] for m in fold_metrics])
    print(f"\nCV MAE (mean across folds): {cv_mae:.3f}")

    # 4. Evaluate on held-out season
    results = evaluate(models, X_test, y_test, groups_test)

    print(f"\n==== Held-out Test Results for {CONFIG['test_seasons'][0]} - {CONFIG['test_seasons'][-1]} ====")
    print(f"  MAE              : {results['mae']:.3f}   (target: < 3.0)")
    print(f"  Rank MAE         : {results['rank_mae']:.3f} (target: < 3.0)")
    print(f"  Spearman (mean)  : {results['spearman_mean']:.3f} (target: > 0.65)")
    print(f"  Podium hit rate  : {results['podium_hit_rate']:.3f} (target: > 0.60)")
    print(f"  Points hit rate  : {results['points_hit_rate']:.3f} (target: > 0.75)")

    print("Tested on seasons:", CONFIG["test_seasons"])

if __name__ == "__main__":
    main()

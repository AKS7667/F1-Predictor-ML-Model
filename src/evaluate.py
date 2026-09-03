import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from train import split_features_target, evaluate, CONFIG   
from predict import coverage_report


FEATURES = "data/f1_features.parquet"

def run(seasons):

    # 1. load saved features ( parquet produced by features.py )
    df = pd.read_parquet(FEATURES)

    # 2. recreate the same test split train.py used
    test_df = df[df["Season"].isin(CONFIG["test_seasons"])] 
    if test_df.empty:
            raise ValueError(f"No rows for seasons {seasons}") 
    # 2.1 drop unscored races so metrics stay valid
    n_before = len(test_df)
    test_df = test_df.dropna(subset=["Position"])
    dropped = n_before - len(test_df)
    if dropped:
        print(f"\nDropped {dropped} unscored rows before scoring.")  

    X_test, y_test, groups_test = split_features_target(test_df)

    # 3. load the saved models
    fold_models = []
    for i in range(5):
        m = xgb.XGBRegressor()
        m.load_model(f"models/fold_{i}.json")
        fold_models.append(m)

    # 4. score with the same evaluate() train.py uses
    results = evaluate(fold_models, X_test, y_test, groups_test)

    # 5. report what's in the matrix — makes the test set explicit
    print("=== Data coverage ===")
    coverage_report(df)

    # 6. score and print
    results = evaluate(fold_models, X_test, y_test, groups_test)

    print(f"\n==== Held-out Test Results for {CONFIG['test_seasons'][0]} - {CONFIG['test_seasons'][-1]}====")
    print(f"  MAE              : {results['mae']:.3f}   (target: < 3.0)")
    print(f"  Rank MAE         : {results['rank_mae']:.3f} (target: < 3.0)")
    print(f"  Spearman (mean)  : {results['spearman_mean']:.3f} (target: > 0.65)")
    print(f"  Podium hit rate  : {results['podium_hit_rate']:.3f} (target: > 0.60)")
    print(f"  Points hit rate  : {results['points_hit_rate']:.3f} (target: > 0.75)")

    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", type=int, nargs="+", default=CONFIG["test_seasons"],
                   help="season(s) to evaluate on; defaults to CONFIG['test_seasons']")
    args = p.parse_args()
    run(args.seasons)
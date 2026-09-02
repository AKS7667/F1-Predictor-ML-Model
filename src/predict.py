import glob
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from train import split_features_target   # adjust import if it lives elsewhere

FEATURES = "data/f1_features.parquet"
MODEL_GLOB = "models/*.json"


def load_models(pattern=MODEL_GLOB):
    models = []
    for path in sorted(glob.glob(pattern)):
        m = xgb.XGBRegressor()
        m.load_model(path)
        models.append(m)
    if not models:
        raise FileNotFoundError(f"No models matched {pattern}")
    return models


def predict_race(season, rnd, df=None, models=None):
    if df is None:
        df = pd.read_parquet(FEATURES)
    if models is None:
        models = load_models()

    race = df[(df["Season"] == season) & (df["Round"] == rnd)].copy()
    if race.empty:
        raise ValueError(f"No rows for season={season} round={rnd}")

    X = split_features_target(race)[0]
    race["pred"] = np.mean([m.predict(X) for m in models], axis=0)

    out = race.sort_values("pred").reset_index(drop=True)
    out["PredPos"] = np.arange(1, len(out) + 1)
    out["pred"] = out["pred"].round(2)

    cols = ["PredPos", "Abbreviation", "pred"]
    if out["Position"].notna().any():
        out["Actual"] = out["Position"]
        cols.append("Actual")

    return out[cols]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("season", type=int)
    p.add_argument("round", type=int)
    args = p.parse_args()
    print(predict_race(args.season, args.round).to_string(index=False))
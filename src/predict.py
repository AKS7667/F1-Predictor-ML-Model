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
def coverage_report(df=None):
    if df is None:
        df = pd.read_parquet(FEATURES)

    status = (df.groupby(["Season", "Round"])["Position"]
                .apply(lambda s: "complete" if s.notna().all()
                                 else "predictable" if s.notna().any()
                                 else "quali_only"))

    complete = status[status == "complete"].index
    predictable = status[status == "predictable"].index
    quali_only = status[status == "quali_only"].index

    if len(complete):
        last = max(complete)
        print(f"Complete races (scored): {len(complete)}, through {last[0]} R{last[1]}")
    if len(predictable):
        print(f"Partially scored: {len(predictable)} -> {list(predictable)}")
    if len(quali_only):
        print(f"Quali only, no result yet: {len(quali_only)} -> {list(quali_only)}")
    if not len(predictable) and not len(quali_only):
        print("Every race in the matrix is complete.")

    return status

def predict_race(season, rnd, rain, df=None, models=None):
    if df is None:
        df = pd.read_parquet(FEATURES)
    if models is None:
        models = load_models()

    race = df[(df["Season"] == season) & (df["Round"] == rnd)].copy()
    if race.empty:
        raise ValueError(f"No rows for season={season} round={rnd}")

    if rain is not None:
        race["RainAtStart"] = int(rain)

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
    p.add_argument("--coverage", action="store_true",
                   help="list which races are scored / predictable, then exit")
    p.add_argument("--rain", type=int, choices=[0, 1], default=None,
            help="override rain at start (0 dry, 1 wet); omit to use stored value")
    args = p.parse_args()

    print(predict_race(args.season, args.round, rain=args.rain).to_string(index=False))


    if args.coverage:
        coverage_report()
    elif args.season is not None and args.round is not None:
        print(predict_race(args.season, args.round, rain=args.rain).to_string(index=False))
    else:
        p.error("give SEASON and ROUND, or use --coverage")
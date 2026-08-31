race = df[(df["Season"] == season) & (df["Round"] == rnd)]
X = split_features_target(race)[0]              # features only
race = race.copy()
race["pred"] = np.mean([m.predict(X) for m in fold_models], axis=0)
out = race.sort_values("pred")[["Abbreviation", "pred"]]
if race["Position"].notna().all():
    out["actual"] = race.sort_values("pred")["Position"]   # comparison if available
print(out)
# 🏎️ F1 Race Prediction

A machine learning model that predicts the finishing order of a Formula 1 race.

It uses an XGBoost regressor trained on 2021–2024 seasons with engineered features (driver form, qualifying pace, constructor strength, circuit history, weather, and DNF risk). Models are trained and validated with 5-fold cross-validation grouped by race, so no race appears in both training and validation — the reported scores reflect performance on races the model hasn't seen.The model is trained with a squared-error objective — it learns to predict each driver's expected finishing position, and the drivers are then sorted by that prediction to produce the race order.

Because it's a regression sorted into a ranking, the raw predicted values tend to compress toward the middle of the field (a likely winner might score ~3.5, not 1.0). That's expected — the order is the product, not the raw number, which is why the model is judged on rank-based metrics alongside raw error.

## 🏁 Results

The model is validated with a **temporal split**: it trains on past seasons and is tested on a *later* season it has never seen. 

- **Trained on:** 2021–2024
- **Tested on:** held-out 2025 (and confirmed on live 2026 data)

| Metric | 2025 (held-out) | 2026 (live) | What it means |
|---|---|---|---|
| MAE | 3.403 | 3.787 | Avg positions off, on the raw regression output |
| Rank MAE | 3.482 | 3.636| Avg positions off, on the sorted finishing order |
| Spearman | 0.639| 0.652| Rank correlation between predicted and actual order |
| Podium hit rate | 0.625 | 0.639 | Fraction of the true top 3 the model puts on the podium |
| Points hit rate | 0.746 | 0.742 | Fraction of the true top 10 the model puts in the points |

Cross-validation MAE ≈ test MAE, which indicates the model isn't overfitting.

---
## 📁 Project Structure
 
```
f1-predictor/
├── src/
│   ├── data_pipeline.py   # Fetches raw session data from FastF1 (results, laps, weather)
│   ├── features.py        # Builds the feature matrix from the raw parquets
│   ├── train.py           # Trains the 5-fold XGBoost ensemble, saves models to models/
│   ├── evaluate.py        # Loads saved models + features, scores on the test season
│   └── predict.py         # Predicts a single race's order; --coverage lists available races
├── notebooks/             # Walkthrough / exploration notebook               
├── data/                  # Feature parquet + raw session parquets (committed)
├── models/                # Saved fold models (fold_0.json … fold_4.json)
├── environment.yml
└── README.md
```
 
---

## ⚙️ Setup
 
The project uses a conda environment. Everything is pinned in `environment.yml`.
 
```bash
# Clone
git clone https://github.com/AKS7667/F1-Predictior-ML-Model.git
cd f1-predictor
 
# Create and activate the environment
conda env create -f environment.yml
conda activate f1-predictor
```
 
The feature parquet and trained models are committed, so you can run predictions immediately — no data fetch or training required.


## 🚥 Usage
 
### Predict a race
 
```bash
# Predict the order for a given season + round
python src/predict.py 2025 1
```
 
Output is the predicted finishing order. If the race has already happened, an `Actual` column is shown next to the prediction so you can compare.
 
```bash
# See which races are available to predict / score
python src/predict.py --coverage
 
# Override rain at race start (0 = dry, 1 = wet) for a what-if prediction
python src/predict.py 2025 1 --rain 1
```
 
### Score the model
 
```bash
# Loads the saved models and scores them on the held-out test season
python src/evaluate.py                 # uses CONFIG["test_seasons"]
python src/evaluate.py --seasons 2026  # override
python src/evaluate.py --seasons 2025 2026
```

---
 
## 🔧 Advanced
 
### a. Tweak the config
 
Training seasons, test season, and model parameters live in the `CONFIG` block at the top of `train.py`. Change the values, re-run training:
 
```python
CONFIG = {
    "train_seasons": [2021, 2022, 2023, 2024],
    "test_seasons":  [2025],
    # model params ...
}
```
 
```bash
python src/train.py
```
 
> ⚠️ Keep the test season *out* of the training seasons. Adding it leaks the data you're trying to evaluate on.
 
### b. Add or remove a feature
 
Features are built in `features.py`. To **remove** one, drop it from the feature matrix. To **add** one, there's a fully commented **FP2 lap-time feature** in `features.py` that works as a copy-paste template — it shows the full pattern (pull a session, aggregate per driver, merge back, guard against leakage with `shift(1)`).
 
Current features:
 
```
GridPosition, QualiGapToPole, DriverForm, ConstructorForm,
CircuitOvertaking, DriverCircuitAvg, RainAtStart, DNFRate
```
 
Raw columns available to build new features from (the committed parquets): results, laps, and weather sessions -

```
results = pd.read_parquet("data/f1_all_results.parquet")
laps = pd.read_parquet("data/f1_all_laps.parquet")
weather = pd.read_parquet("data/f1_all_weather.parquet")

res_col = results.columns
laps_col = laps.columns
wea_cols = weather.columns
```
 
> ⚠️ Watch for leakage: rolling form features use `shift(1)` so a race can't see its own result, and weather is taken at race *start*, not aggregated across the session.
 
### c. Fetch more data (older or newer seasons)
 
`data_pipeline.py` fetches from FastF1 and is resumable (it skips sessions already saved). Add the seasons you want to the `YEARS` list and run it, then rebuild features:
 
```bash
python src/data_pipeline.py
python src/features.py
```
 
Notes:
- 🏁 **Predicting an upcoming race:** you can only predict a race *after its qualifying has happened* — grid position and quali pace are inputs. This isn't season-ahead prediction; it's "predict this weekend's race once quali is done."
- 🔁 **Re-pulling recent races:** FastF1 data for a very recent session can be incomplete right after the session. Re-run the pipeline later to pick up the finalized data (the cache means completed sessions aren't re-downloaded).
- ⏳ **Rate limits:** FastF1 allows 500 calls/hour. The pipeline detects the limit and waits it out rather than bypassing it, so a large first-time fetch may pause. Cached sessions don't count against the limit.
---

## 🏆 Limitations & Future Work
 
- **DNFs and incidents are unpredictable.** The model predicts *pace*, so it can't foresee a first-lap crash or a mechanical retirement. A race with several DNFs will score poorly no matter how good the model is — this is variance, not a defect.
- **Regression-then-sort compresses predictions.** Trained on squared error, the model predicts expected position, so values cluster in the mid-field and clustered values get broken by noisy order. The next step is `rank:pairwise` — an XGBoost learning-to-rank objective that optimizes the *order within each race* directly.
- **Tie sensitivity.** When two drivers get near-identical predictions, the sort order between them is essentially arbitrary. A ranking objective would address this too.
- **Sprint Race and qualifying** The model doesn't incorporate any sprints and their qualifying into their predictions at all, which might be leaving a valuable piece of information on the table. Adding it as a feature without affecting non-sprint races could increase accuracy. 

## 📬 Contact

Arsh Somani - asomani@crimson.ua.edu   
Project: https://github.com/AKS7667/F1-Predictior-ML-Model


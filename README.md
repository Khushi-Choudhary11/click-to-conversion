# Engage2Value: From Clicks to Conversions

Predict how much a customer spends in a session (`purchaseValue`) from multi-session digital behaviour: traffic source, device, geography and engagement. A Kaggle competition (late submission) solved with scikit-learn, XGBoost and LightGBM, scored by R².

## Problem

The dataset has session-level records from a large digital commerce platform. Each row is one unique user session.

| | |
|---|---|
| Train | 116,023 sessions, 52 columns (115,787 rows after removing 236 duplicates) |
| Test | 29,006 sessions |
| Target | `purchaseValue`, the amount spent in the session |
| Metric | `r2_score` between predictions and the true target |
| Submission | CSV with `id` and `purchaseValue` |

Feature groups: user behaviour and session metrics (`totalHits`, `pageViews`, `sessionNumber`, `sessionStart`), device and technical attributes (`deviceType`, `os`, `browser`, `gclIdPresent`), traffic and marketing source (`userChannel`, `trafficSource.*`), geography (`geoNetwork.*`, `locationCountry`, `geoCluster`) and identifiers (`userId`, `sessionId`).

Data: [Engage2Value competition on Kaggle](https://www.kaggle.com/competitions/engage-2-value-from-clicks-to-conversions). The files are not included in this repo (competition rules).

## Approach

1. **EDA.** `purchaseValue` is extremely right-skewed (mean about 2.7e7, median 0). Desktop users and the Americas have the highest average purchase value. High engagement does not always mean a purchase. `pageViews` and `totalHits` are 0.99 correlated, and `sessionId`, `date` and `sessionStart` are 0.91 correlated.
2. **Cleaning.** Dropped 236 duplicate rows, 5 columns with more than 70% missing values, and 20 columns with a single constant value (for example `device.screenResolution`, `totals.visits`). Outliers were kept: there are too many to remove safely, so tree models handle them.
3. **Feature engineering.** `days_active`, `hits_per_page`, `hits_per_session` and `page_per_session`. All three ratio features correlate with the target more than the raw counts they come from.
4. **Encoding.** Median imputation and scaling for numeric columns, one-hot for categories with fewer than 10 values, and target encoding for high-cardinality columns (`browser`, `trafficSource.keyword`, `geoNetwork.city`, `userId` and others), all in a `ColumnTransformer`.
5. **Models.** Eight baselines (Linear, Ridge, Random Forest, Gradient Boosting, KNN, XGBoost, LightGBM, AdaBoost), then `RandomizedSearchCV` for Random Forest, AdaBoost, Gradient Boosting, XGBoost and LightGBM. XGBoost used `GroupKFold` by `userId`.

## Results

| Model | CV R² after tuning | Public leaderboard R² |
|---|---|---|
| Random Forest | 0.947 | about 0.54 |
| AdaBoost | 0.974 | about 0.40 |
| Gradient Boosting | 0.976 | about 0.28 |
| **XGBoost (submitted)** | 0.868 | **about 0.68** |
| LightGBM | 0.831 | about 0.45 |

XGBoost had the lowest cross-validation score but the best leaderboard score, and its strong regularisation (`gamma`, `reg_alpha`, `reg_lambda`, subsampling) is the likely reason it held up.

## Known issues (read before trusting the CV numbers)

The large gap between cross-validation (0.83 to 0.98) and the leaderboard (0.28 to 0.68) is a warning sign, and the notebook has issues that explain at least part of it:

- **Target-encoding leakage.** The preprocessing, including `TargetEncoder`, is fitted with `fit_transform(X, y)` on the whole training set **before** the train/validation split and before cross-validation. Each row's own purchase value leaks into its encoded features, which inflates every validation score. Even plain Linear Regression reached R² 0.998 on the hold-out, which should not be possible on this target.
- **Misaligned groups.** The `GroupKFold` groups for XGBoost come from the first rows of the dataframe and do not line up with the rows of the shuffled `train_test_split`.
- **Noisy metric.** The target is zero-inflated and heavy-tailed: the top 1% of sessions make about 44% of all revenue, so R² varies a lot between splits and a single leaderboard score is a noisy estimate.
- **Target scale.** The model is trained on the raw target although EDA shows a log transform makes it much closer to normal.

A leakage-free rework (all preprocessing inside a `Pipeline`, repeated random hold-outs, raw-scale target) is a planned follow-up. In that version the honest hold-out R² was about 0.4, which matches a leaderboard score of about 0.40.

## Run it

The notebook reads from `/kaggle/input/engage-2-value-from-clicks-to-conversions/`, so the simplest way to run it is on Kaggle (late submissions are open). To run it locally, download the three CSV files and change the paths in the data-loading cells.

```bash
pip install -r requirements.txt
jupyter lab click-to-conversion.ipynb
```

## Files

| File | Purpose |
|---|---|
| `click-to-conversion.ipynb` | Full notebook with outputs |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT |

## License

[MIT](LICENSE). The competition data remains subject to Kaggle's rules.

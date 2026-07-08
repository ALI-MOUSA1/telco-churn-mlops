# Telco Customer Churn — MLOps Pipeline

A standardized, reproducible MLOps pipeline for predicting customer churn on the
**IBM Telco Customer Churn** dataset. Covers data versioning, experiment tracking
with MLflow, multi-model training/evaluation, and Docker-based deployment.

## Project Structure

```
project1/
├── data/
│   ├── v1/   # raw data (telco_churn_raw.csv, exported from Telco_customer_churn.xlsx)
│   ├── v2/   # cleaned data (missing values handled, encoded)
│   └── v3/   # feature-engineered + normalized data
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── mlflow_utils.py
├── run_pipeline.py
├── check_registry.py     # utility: confirm registered model in MLflow registry
├── convert.py             # utility: xlsx -> csv converter
├── test_request.json      # sample inference payload for the deployed model
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Setup

```bash
pip install -r requirements.txt
```

Raw dataset is at `data/v1/telco_churn_raw.csv` (7,043 customers, IBM Telco Churn dataset).

## 2. Run the full pipeline

```bash
python run_pipeline.py --register
```

This runs, in order:
1. `load_data` — confirms `data/v1/` is present
2. `preprocess` — cleans v1 → writes `data/v2/telco_churn_clean.csv`
3. `feature_engineering` — builds v2 → writes `data/v3/telco_churn_features.csv` (+ fitted scaler)
4. `train_models` — trains Logistic Regression, Random Forest, XGBoost, CatBoost on **each** of v1/v2/v3 (12 runs total), logging every run to MLflow
5. `evaluate` — pulls all MLflow runs, prints a comparison table, picks the best
6. `log_to_mlflow` — registers the best model to the MLflow Model Registry as `Production`

## 3. Results

All 12 runs (4 models × 3 dataset versions) were trained and evaluated on a held-out
test set. Full comparison:

| Model              | Dataset | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---------------------|---------|----------|-----------|--------|----------|---------|
| Logistic Regression | v1      | 0.8138   | 0.6591    | 0.6203 | **0.6391** | 0.8585 |
| CatBoost             | v1      | 0.8152   | 0.6717    | 0.5963 | 0.6317   | **0.8657** |
| XGBoost              | v1      | 0.8124   | 0.6687    | 0.5829 | 0.6229   | 0.8645 |
| Logistic Regression | v3      | 0.8070   | 0.6594    | 0.5642 | 0.6081   | 0.8493 |
| Random Forest        | v1      | 0.8131   | 0.6920    | 0.5348 | 0.6033   | 0.8641 |
| Logistic Regression | v2      | 0.7970   | 0.6294    | 0.5722 | 0.5994   | 0.8453 |
| XGBoost              | v2      | 0.7956   | 0.6352    | 0.5401 | 0.5838   | 0.8472 |
| CatBoost             | v3      | 0.7970   | 0.6429    | 0.5294 | 0.5806   | 0.8471 |
| XGBoost              | v3      | 0.7977   | 0.6508    | 0.5134 | 0.5740   | 0.8423 |
| CatBoost             | v2      | 0.7928   | 0.6331    | 0.5214 | 0.5718   | 0.8468 |
| Random Forest        | v2      | 0.7906   | 0.6396    | 0.4840 | 0.5510   | 0.8468 |
| Random Forest        | v3      | 0.7885   | 0.6319    | 0.4866 | 0.5498   | 0.8437 |

**Best model: Logistic Regression trained on v1 (raw data)** — F1-score 0.6391,
ROC-AUC 0.8585 — selected automatically by `evaluate.py` and registered in MLflow
as `ChurnModel`, version 1, stage `Production`.

**Observation:** models trained on v1 (minimally processed raw data) slightly
outperformed their v2/v3 counterparts across the board. This suggests that, for
this dataset, the added categorical encoding/feature engineering introduced more
noise/dimensionality than useful signal for these particular model families —
worth further investigation (e.g. feature selection, regularization tuning) in
future iterations.

## 4. Inspect experiments in MLflow

Tracking backend: SQLite (`mlflow.db`), local to this project.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open **http://127.0.0.1:5000**. Each run logs: tags (`model_name`, `dataset_version`),
params (hyperparameters + `seed`), metrics (`accuracy`, `precision`, `recall`,
`f1_score`, `roc_auc`, `cv_f1_mean`, `validation_f1`), and artifacts
(`confusion_matrix.png`, serialized model).

To verify the registered model from the command line instead of the UI:
```bash
python check_registry.py
```
Expected output:
```
ChurnModel 1 Production f42e0f38bfad458da63cdabb4a1a2829
```

## 5. Git workflow

```bash
git init
git add .
git commit -m "Complete pipeline: data versioning, MLflow tracking, model registry, Docker deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 6. Deploy the best model with MLflow + Docker

**Build the image directly from the MLflow Model Registry:**
```bash
mlflow models build-docker --model-uri "models:/ChurnModel/Production" --name churn-model-image
```

**Run the container:**
```bash
docker run -p 5001:8080 churn-model-image
```

**Test inference** using the included sample payload (`test_request.json`,
built from a real customer record):
```powershell
Invoke-RestMethod -Uri "http://localhost:5001/invocations" -Method Post -InFile "test_request.json" -ContentType "application/json"
```

Example result: `predictions: [0]` (model predicted "no churn" for that customer).

## 7. Reproducibility notes

- All random operations (train/val/test split, K-Fold CV, model init) use a
  fixed `SEED = 42` (see `src/train.py`).
- Each dataset version is processed and evaluated **independently**: v1 uses
  a minimal numeric-only encoding of the raw data, v2 adds cleaning + full
  encoding, v3 adds engineered features + normalization.
- Cross-validation (5-fold, stratified) is used during training for stability;
  the validation split is used for model selection; the test split is
  evaluated exactly once per model/version.
- Leakage columns (`Churn Label`, `Churn Score`, `Churn Reason`) were
  identified and excluded from all feature sets, since they directly encode
  or are only populated for the churn outcome itself.

## 8. Conclusion

Across all 12 experiments, **Logistic Regression on the raw (v1) dataset**
achieved the best F1-score (0.6391) while CatBoost on v1 achieved the best
ROC-AUC (0.8657) — both close contenders. The final deployed model
(Logistic Regression / v1) balances precision and recall reasonably well
(65.9% / 62.0%) for a churn-prediction use case, though there is room for
improvement, particularly in recall — meaning a portion of customers who do
churn are not being flagged. Future work could explore class-imbalance
handling (e.g. class weighting or SMOTE), more extensive hyperparameter
search, and further feature engineering informed by which v1 features
correlate most strongly with churn.
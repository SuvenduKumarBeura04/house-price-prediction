# 🏠 House Price Prediction — Advanced Edition

An end-to-end machine learning project that predicts residential house sale prices using
the **Ames Housing Dataset** (Kaggle "House Prices: Advanced Regression Techniques").

**Flow:** Kaggle Notebook (train) → GitHub (store/version) → Streamlit Community Cloud (deploy, free)

---

## 📁 Project Structure

```
house-price-prediction/
│
├── data/
│   ├── train.csv              # download from Kaggle (see below) — NOT committed to git
│   ├── test.csv                # download from Kaggle (see below) — NOT committed to git
│   └── README.md               # instructions to get the data
│
├── notebooks/
│   └── house_price_advanced.ipynb   # the full training notebook (paste into Kaggle)
│
├── src/
│   ├── features.py              # feature engineering functions
│   └── pipeline.py              # builds the sklearn preprocessing+model pipeline
│
├── models/
│   └── house_price_pipeline.joblib   # saved trained pipeline (generated after training)
│
├── app/
│   └── app.py                   # Streamlit app — loads the pipeline, takes user input, predicts
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to use this project (step by step)

### 1. Get the dataset
Go to the Kaggle competition page:
`https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data`

Download `train.csv` and `test.csv` and place them inside the `data/` folder.
(You need a free Kaggle account and to accept the competition rules — just a checkbox.)

### 2. Train the model on Kaggle (recommended — free compute, dataset pre-attached)
1. Open the competition page → "New Notebook"
2. Copy the contents of `notebooks/house_price_advanced.ipynb` into your Kaggle notebook
   (or upload the `.ipynb` file directly using "File → Upload Notebook")
3. Run all cells. At the end, it saves `house_price_pipeline.joblib`
4. Download that file from the notebook's "Output" panel
5. Place it inside `models/` in this project folder

> You can also run the same notebook locally in VS Code / Jupyter if you have the
> CSVs in `data/` — just make sure the file paths in the first cell match.

### 3. Test the Streamlit app locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
Open the local URL it gives you (usually `http://localhost:8501`) and try a prediction.

### 4. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: house price prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/house-price-prediction.git
git push -u origin main
```

> ⚠️ `data/*.csv` and `models/*.joblib` are excluded by `.gitignore` by default since they
> can be large / are regenerated. If your model file is small (a few MB, which it usually is
> for this dataset), you can remove it from `.gitignore` so the app has something to load
> immediately after deployment — see the note in `.gitignore`.

### 5. Deploy for free on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub → "New app"
3. Select your repo, branch `main`, and set the main file path to `app/app.py`
4. Click "Deploy" — you'll get a free public URL in a couple of minutes

---

## 🧠 What makes this the "advanced" version

- Log-transform of the skewed target (`SalePrice`) and skewed numeric features
- Proper handling of missing values based on what "missing" *means* per column
  (e.g. missing `PoolQC` = no pool, not an error)
- Engineered features: `TotalSF`, `HouseAge`, `TotalBath`, `IsRemodeled`, etc.
- Ordinal encoding for quality-based columns (`Po < Fa < TA < Gd < Ex`) instead of
  naive one-hot encoding, which would lose the ordering information
- Everything (encoding + scaling + model) wrapped in a single `scikit-learn Pipeline`,
  so the exact same transformations used in training are applied to new user input —
  no manual re-implementation of preprocessing in the app
- Multiple models compared with 5-Fold Cross-Validation: Ridge, Random Forest, XGBoost, LightGBM
- A weighted blend of the top models instead of relying on a single "best" model
- Clean separation of concerns: `src/` (reusable code) vs `notebooks/` (experimentation)
  vs `app/` (deployment)

## 🔭 Optional next-level additions (once the base version is working and deployed)
- SHAP values in the Streamlit app to explain *why* a prediction came out the way it did
- Optuna for smarter hyperparameter tuning instead of RandomizedSearchCV
- MLflow to track experiments across model runs
- `pytest` unit tests for `src/features.py` + a GitHub Actions workflow to run them on every push

## 🛠️ Tech Stack
Python · pandas · numpy · scikit-learn · XGBoost · LightGBM · matplotlib · seaborn · Streamlit

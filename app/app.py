"""
Streamlit app for House Price Prediction.

Loads the trained pipeline (preprocessing + model baked in together) and lets
the user enter a handful of the most meaningful property details. All the
engineered/ordinal features are computed automatically behind the scenes
using src/features.py — the user never has to know they exist.
"""

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make src/ importable regardless of where streamlit is launched from
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features import full_feature_pipeline  # noqa: E402


def format_indian_currency(amount):
    """Format a number with Indian-style comma grouping (e.g. 1,16,72,920)."""
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return s
    last_three = s[-3:]
    remaining = s[:-3]
    groups = []
    while len(remaining) > 2:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.insert(0, remaining)
    return ",".join(groups) + "," + last_three


MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "house_price_pipeline.joblib"
)
SKEWED_COLS_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "skewed_cols.json"
)

# The trained pipeline was fit on ALL raw Ames Housing columns, so every prediction
# needs every column present — even ones the user doesn't manually enter. These are
# reasonable "typical house" defaults (the dataset's most common category / a sensible
# midpoint value) for everything the form doesn't ask about. The user's actual inputs
# always override these where they overlap.
DEFAULT_ROW = {
    "MSSubClass": 20,
    "MSZoning": "RL",
    "LotFrontage": 70,
    "Street": "Pave",
    "Alley": "None",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "OverallCond": 5,
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0,
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "Unf",
    "BsmtFinSF1": 0,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 500,
    "Heating": "GasA",
    "HeatingQC": "TA",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "LowQualFinSF": 0,
    "KitchenAbvGr": 1,
    "KitchenQual": "TA",
    "TotRmsAbvGrd": 6,
    "Functional": "Typ",
    "FireplaceQu": "None",
    "GarageType": "Attchd",
    "GarageYrBlt": 2000,
    "GarageFinish": "Unf",
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "WoodDeckSF": 0,
    "OpenPorchSF": 0,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "PoolQC": "None",
    "Fence": "None",
    "MiscFeature": "None",
    "MiscVal": 0,
    "MoSold": 6,
    "SaleType": "WD",
    "SaleCondition": "Normal",
}

st.set_page_config(
    page_title="House Price Predictor", page_icon="🏠", layout="centered"
)

st.title("🏠 House Price Predictor")
st.write(
    "Enter a few details about the property below. The model will estimate "
    "its sale price using the Ames Housing dataset (Kaggle: House Prices — "
    "Advanced Regression Techniques)."
)


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_skewed_cols():
    # These are the exact columns that were log1p-transformed during training
    # (to fix their skewed distributions) BEFORE being fed into the pipeline.
    # That step happened outside the saved sklearn Pipeline, so we have to
    # replicate it manually here at prediction time using the same column list
    # the training notebook exported.
    if not SKEWED_COLS_PATH.exists():
        return []
    with open(SKEWED_COLS_PATH) as f:
        return json.load(f)


def apply_skew_correction(df, skewed_cols):
    df = df.copy()
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


pipeline = load_pipeline()
skewed_cols = load_skewed_cols()

if pipeline is None:
    st.error(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Train the model first (see notebooks/house_price_advanced.ipynb), "
        "then place `house_price_pipeline.joblib` inside the `models/` folder."
    )
    st.stop()

if not skewed_cols:
    st.warning(
        "No `models/skewed_cols.json` found — predictions may be inaccurate. "
        "See the notebook's final cells for how to export this file."
    )

st.subheader("Property details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1=worst, 10=best)", 1, 10, 6)
    gr_liv_area = st.number_input("Above-ground living area (sq ft)", 300, 6000, 1500)
    year_built = st.number_input("Year built", 1870, 2026, 2000)
    total_bsmt_sf = st.number_input("Total basement area (sq ft)", 0, 4000, 800)
    garage_cars = st.slider("Garage capacity (cars)", 0, 5, 2)
    full_bath = st.slider("Full bathrooms", 0, 4, 2)

with col2:
    neighborhood = st.selectbox(
        "Neighborhood",
        [
            "NAmes",
            "CollgCr",
            "OldTown",
            "Edwards",
            "Somerst",
            "Gilbert",
            "NridgHt",
            "Sawyer",
            "NWAmes",
            "SawyerW",
            "BrkSide",
            "Crawfor",
            "Mitchel",
            "NoRidge",
            "Timber",
            "IDOTRR",
            "ClearCr",
            "StoneBr",
            "SWISU",
            "MeadowV",
            "Blmngtn",
            "BrDale",
            "Veenker",
            "NPkVill",
            "Blueste",
        ],
    )
    house_style = st.selectbox(
        "House style",
        ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "1.5Unf", "2.5Unf", "2.5Fin"],
    )
    bedrooms = st.slider("Bedrooms above ground", 0, 8, 3)
    half_bath = st.slider("Half bathrooms", 0, 2, 1)
    year_sold = st.number_input("Year of sale", 2006, 2026, 2024)
    lot_area = st.number_input("Lot area (sq ft)", 1000, 50000, 9000)

st.caption(
    "Only the most impactful fields are shown here. Every other feature the "
    "model needs is filled with sensible dataset-wide defaults automatically."
)

predict_clicked = st.button("Predict Sale Price", type="primary")

if predict_clicked:
    # Fields the user actually controls in the form
    user_inputs = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "YearBuilt": year_built,
        "YearRemodAdd": year_built,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": gr_liv_area * 0.6,
        "2ndFlrSF": (
            gr_liv_area * 0.4 if house_style in ["2Story", "2.5Fin", "2.5Unf"] else 0
        ),
        "GarageCars": garage_cars,
        "GarageArea": garage_cars * 250,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "BedroomAbvGr": bedrooms,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
        "YrSold": year_sold,
        "LotArea": lot_area,
        "Fireplaces": 0,
    }

    # Start from the full set of default columns the pipeline expects, then
    # overlay the user's actual answers on top — this guarantees every raw
    # column the trained ColumnTransformer needs is present.
    full_row = {**DEFAULT_ROW, **user_inputs}
    input_row = pd.DataFrame([full_row])

    try:
        processed = full_feature_pipeline(input_row)
        processed = apply_skew_correction(processed, skewed_cols)
        prediction_log = pipeline.predict(processed)
        prediction_usd = np.expm1(prediction_log)[
            0
        ]  # reverse the log1p used during training

        USD_TO_INR = 83.5  # approximate — update as needed, exchange rates fluctuate
        prediction_inr = prediction_usd * USD_TO_INR

        st.success(f"### Estimated Sale Price: ₹{format_indian_currency(prediction_inr)}")
        st.caption(
            f"(≈ ${prediction_usd:,.0f} USD, converted at an approximate rate of "
            f"1 USD = ₹{USD_TO_INR}) — this is a model estimate based on historical "
            "Ames, Iowa housing data, treat it as a ballpark figure, not an appraisal."
        )
    except Exception as e:
        st.error(
            "Couldn't generate a prediction — this usually means the input "
            f"columns don't fully match what the trained pipeline expects.\n\nDetails: {e}"
        )

st.divider()
st.caption(
    "Built with scikit-learn + Streamlit · Dataset: Ames Housing / "
    "Kaggle House Prices: Advanced Regression Techniques"
)

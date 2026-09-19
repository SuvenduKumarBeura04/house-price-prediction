"""
Feature engineering functions for the House Price Prediction project.

Kept as plain functions (not notebook cells) so the exact same logic can be
imported both by the training notebook AND the Streamlit app — this guarantees
that whatever transformation happens during training also happens on new user
input at prediction time, with zero risk of the two drifting apart.
"""

import numpy as np
import pandas as pd

# Columns where a missing value actually means "does not exist", not "unknown"
NONE_MEANS_ABSENT = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

# Numeric columns where missing also means "does not exist" -> fill with 0
ZERO_MEANS_ABSENT = [
    "GarageYrBlt", "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# Ordinal quality-style columns: order matters, so we map them to numbers
# instead of one-hot encoding (which would throw the ordering away)
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values based on what 'missing' actually means for each column."""
    df = df.copy()

    for col in NONE_MEANS_ABSENT:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ZERO_MEANS_ABSENT:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: fill with the median per neighborhood (houses in the same
    # neighborhood tend to have similar lot frontage)
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    # Remaining categorical columns: fill with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remaining numeric columns: fill with median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new, more informative features from the raw columns."""
    df = df.copy()

    df["TotalSF"] = (
        df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0)
    )

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    df["TotalBath"] = (
        df.get("FullBath", 0)
        + 0.5 * df.get("HalfBath", 0)
        + df.get("BsmtFullBath", 0)
        + 0.5 * df.get("BsmtHalfBath", 0)
    )

    df["TotalPorchSF"] = (
        df.get("OpenPorchSF", 0)
        + df.get("EnclosedPorch", 0)
        + df.get("3SsnPorch", 0)
        + df.get("ScreenPorch", 0)
    )

    df["HasPool"] = (df.get("PoolArea", 0) > 0).astype(int)
    df["HasGarage"] = (df.get("GarageArea", 0) > 0).astype(int)
    df["HasBasement"] = (df.get("TotalBsmtSF", 0) > 0).astype(int)
    df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)

    return df


def encode_ordinal_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Map quality-style categorical columns to ordered numeric values."""
    df = df.copy()
    for col in ORDINAL_QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)
    return df


def full_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete feature-engineering flow in the correct order."""
    df = handle_missing_values(df)
    df = engineer_features(df)
    df = encode_ordinal_quality(df)
    return df

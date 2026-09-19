"""
Builds the full scikit-learn preprocessing + model pipeline.

Wrapping everything (encoding, scaling, model) in ONE Pipeline object means:
  - training code and the deployed app use IDENTICAL preprocessing
  - saving/loading the whole thing is a single joblib.dump/load call
  - no risk of "it worked in the notebook but broke in the app" bugs
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """
    numeric_cols: list of numeric column names (already includes engineered +
                  ordinal-encoded quality columns, since those are numeric by this point)
    categorical_cols: list of true nominal categorical column names (e.g. Neighborhood)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return preprocessor


def build_model_pipeline(preprocessor, model) -> Pipeline:
    """Combine preprocessing + a given regressor into one deployable pipeline."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Calculate comprehensive regression metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    err = np.abs(y_true - y_pred)
    acc20 = float((err <= 20.0).mean())
    acc10 = float((err <= 10.0).mean())
    acc5 = float((err <= 5.0).mean())
    # Use ±10 minutes as the default accuracy threshold
    acc = acc10
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "accuracy": acc, "accuracy5": acc5, "accuracy10": acc10, "accuracy20": acc20}


def train_eval_knn(X, y, n_neighbors: int = 15, test_size: float = 0.2, random_state: int = 42):
    # Train and evaluate K-Nearest Neighbors regressor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pre = ColumnTransformer([
        ("num", Pipeline(steps=[("impute", SimpleImputer()), ("scale", StandardScaler())]), X.columns.tolist()),
    ])

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return pipe, _metrics(y_test, preds)


def train_eval_linear(X, y, test_size: float = 0.2, random_state: int = 42):
    # Train and evaluate Linear Regression model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pre = ColumnTransformer([
        ("num", Pipeline(steps=[("impute", SimpleImputer()), ("scale", StandardScaler())]), X.columns.tolist()),
    ])
    model = LinearRegression(n_jobs=None)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return pipe, _metrics(y_test, preds)


def train_eval_rf(X, y, n_estimators: int = 300, max_depth: Optional[int] = None, test_size: float = 0.2, random_state: int = 42):
    # Train and evaluate Random Forest regressor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pre = ColumnTransformer([
        ("num", SimpleImputer(), X.columns.tolist()),
    ])
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return pipe, _metrics(y_test, preds)


def train_eval_xgb(X, y, test_size: float = 0.2, random_state: int = 42):
    # Train and evaluate XGBoost regressor
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not available")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        tree_method="hist",
        random_state=random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    return model, _metrics(y_test, preds)


def train_eval_catboost(X, y, test_size: float = 0.2, random_state: int = 42):
    # Train and evaluate CatBoost regressor
    if CatBoostRegressor is None:
        raise RuntimeError("catboost is not available")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.05,
        iterations=1000,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    preds = model.predict(X_test)
    return model, _metrics(y_test, preds)




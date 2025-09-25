from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.features.engineering import (
    add_basic_features,
    augment_with_weather,
    add_derived_features,
    drop_outliers_iqr,
)


def load_raw(jl_csv: str | Path, yt_csv: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load raw delivery data for both Jilin and Yantai cities
    df_jl = pd.read_csv(jl_csv)
    df_yt = pd.read_csv(yt_csv)
    return df_jl, df_yt


def build_features(
    # Construct feature set from raw data including weather and derived features
    df: pd.DataFrame,
    city_lat: float,
    city_lon: float,
    datetime_col: str = "accept_time",
) -> pd.DataFrame:
    df1 = add_basic_features(df)  # Add basic time and distance features
    df2 = augment_with_weather(df1, datetime_col=datetime_col, mode="city", city_lat=city_lat, city_lon=city_lon)  # Add weather data
    df3 = add_derived_features(df2)  # Add derived features like buckets and flags
    # Retain only rows with valid target values
    if "ata_minutes" in df3.columns:
        df3 = df3[df3["ata_minutes"].notna()]
    # Apply outlier filtering while preserving dataset integrity
    filtered = drop_outliers_iqr(df3, cols=["ata_minutes", "distance_km", "speed_kmh"], k=1.5)
    if len(filtered) == 0:
        return df3.reset_index(drop=True)
    return filtered.reset_index(drop=True)


def split_xy(df: pd.DataFrame, target: str = "ata_minutes") -> Tuple[pd.DataFrame, pd.Series]:
    # Separate features and target variable for model training
    # Select numeric features and encode categorical variables
    work = df.copy()
    if "city" in work.columns:
        work["city_code"] = pd.Categorical(work["city"]).codes
    cat_cols = [c for c in ["temp_bucket", "wind_bucket", "distance_bucket", "holiday_name", "day_of_week"] if c in work.columns]
    for c in cat_cols:
        work[f"{c}_code"] = pd.Categorical(work[c]).codes

    feature_cols = [
        c
        for c in [
            "distance_km",
            "hour",
            "is_weekend",
            "week_of_year",
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "is_business_hour",
            "is_peak_hour",
            "is_rain",
            "speed_kmh",
            "city_code",
        ]
        if c in work.columns
    ] + [f"{c}_code" for c in cat_cols]

    X = work[feature_cols].astype(float)
    y = pd.to_numeric(work[target], errors="coerce")
    # Remove rows with missing target values
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y




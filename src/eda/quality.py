from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def basic_report(df: pd.DataFrame) -> Dict[str, Any]:
    # Generate basic data quality report with summary statistics
    rep: Dict[str, Any] = {}
    rep["num_rows"] = int(len(df))
    rep["columns"] = list(df.columns)
    rep["null_counts"] = {c: int(df[c].isna().sum()) for c in df.columns}
    rep["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

    # Compute descriptive statistics for important numeric columns
    for col in ["ata_minutes", "distance_km", "lat", "lng", "accept_gps_lat", "accept_gps_lng", "delivery_gps_lat", "delivery_gps_lng"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                rep[f"describe_{col}"] = {
                    "min": float(s.min()),
                    "p25": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "p75": float(s.quantile(0.75)),
                    "max": float(s.max()),
                }
    return rep


def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    # Parse timestamp columns with multiple format support
    out = df.copy()
    # Attempt strict parsing first, then use flexible parsing
    if "accept_time" in out.columns:
        t = pd.to_datetime(out["accept_time"], format="%m-%d %H:%M:%S", errors="coerce")
        t2 = pd.to_datetime(out["accept_time"], errors="coerce")
        out["accept_time"] = t.fillna(t2)
    if "delivery_time" in out.columns:
        t = pd.to_datetime(out["delivery_time"], format="%m-%d %H:%M:%S", errors="coerce")
        t2 = pd.to_datetime(out["delivery_time"], errors="coerce")
        out["delivery_time"] = t.fillna(t2)
    return out


def _recompute_targets(out: pd.DataFrame) -> pd.DataFrame:
    # Recalculate delivery time from accept and delivery timestamps
    if {"accept_time", "delivery_time"}.issubset(out.columns):
        at = pd.to_datetime(out["accept_time"], errors="coerce")
        dt = pd.to_datetime(out["delivery_time"], errors="coerce")
        out["ata_minutes"] = (dt - at).dt.total_seconds() / 60.0
    return out


def _clip_outliers(out: pd.DataFrame) -> pd.DataFrame:
    # Remove extreme outliers from delivery time and distance data
    # Filter out invalid and extreme values
    if "ata_minutes" in out.columns:
        s = pd.to_numeric(out["ata_minutes"], errors="coerce")
        # Remove negative and zero duration entries
        out = out[s > 0].copy()
        # Cap extreme values at 99th percentile
        if s.notna().sum() > 50:
            hi = s.quantile(0.99)
            out.loc[:, "ata_minutes"] = np.clip(out["ata_minutes"], 0, hi)
    if "distance_km" in out.columns:
        s = pd.to_numeric(out["distance_km"], errors="coerce")
        if s.notna().sum() > 50:
            hi = s.quantile(0.995)
            out.loc[:, "distance_km"] = np.clip(out["distance_km"], 0, hi)
    return out


def _sanitize_coords(out: pd.DataFrame) -> pd.DataFrame:
    # Validate and clean GPS coordinate data
    # Verify coordinates are within valid geographic ranges
    for lat_col, lng_col in [("lat", "lng"), ("accept_gps_lat", "accept_gps_lng"), ("delivery_gps_lat", "delivery_gps_lng")]:
        if {lat_col, lng_col}.issubset(out.columns):
            lat = pd.to_numeric(out[lat_col], errors="coerce")
            lng = pd.to_numeric(out[lng_col], errors="coerce")
            mask = lat.between(-90, 90) & lng.between(-180, 180)
            out.loc[~mask, [lat_col, lng_col]] = np.nan
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Apply comprehensive data cleaning pipeline
    out = df.copy()
    out = _parse_times(out)
    out = _recompute_targets(out)
    # Remove duplicate records based on key identifiers
    subset_keys = [c for c in ["order_id", "accept_time", "delivery_time"] if c in out.columns]
    if subset_keys:
        out = out.drop_duplicates(subset=subset_keys)

    out = _sanitize_coords(out)
    out = _clip_outliers(out)

    return out.reset_index(drop=True)



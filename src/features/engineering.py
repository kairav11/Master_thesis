from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests


def _parse_md_hms_with_year(s: pd.Series, year: int = 2022) -> pd.Series:
    # Parse timestamp strings with month-day hour:minute:second format
    t = pd.to_datetime(s, format="%m-%d %H:%M:%S", errors="coerce")
    # Assign the specified year to parsed timestamps
    t = t.apply(lambda x: x.replace(year=year) if pd.notna(x) else x)
    # Use generic parsing if specific format fails
    if t.notna().mean() < 0.6:
        t2 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        t2 = t2.apply(lambda x: x.replace(year=year) if pd.notna(x) else x)
        t = t.fillna(t2)
    return t


def add_basic_features(df: pd.DataFrame, assumed_year: int = 2022) -> pd.DataFrame:
    # Add fundamental features including delivery time and distance calculations
    out = df.copy()
    # Calculate actual time of arrival in minutes
    if {"delivery_time", "accept_time"}.issubset(out.columns):
        out["accept_time"] = _parse_md_hms_with_year(out["accept_time"], year=assumed_year)
        out["delivery_time"] = _parse_md_hms_with_year(out["delivery_time"], year=assumed_year)
        out["ata_minutes"] = (out["delivery_time"] - out["accept_time"]).dt.total_seconds() / 60

    # Compute haversine distance between GPS coordinates
    has_cols = {"accept_gps_lat", "accept_gps_lng", "delivery_gps_lat", "delivery_gps_lng"}.issubset(out.columns)
    if has_cols:
        R = 6371.0
        lat1 = np.radians(pd.to_numeric(out["accept_gps_lat"], errors="coerce"))
        lon1 = np.radians(pd.to_numeric(out["accept_gps_lng"], errors="coerce"))
        lat2 = np.radians(pd.to_numeric(out["delivery_gps_lat"], errors="coerce"))
        lon2 = np.radians(pd.to_numeric(out["delivery_gps_lng"], errors="coerce"))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        out["distance_km"] = 2 * R * np.arcsin(np.sqrt(a))
    else:
        out["distance_km"] = np.nan

    return out


def _weather_cache_path_csv(lat: float, lon: float, start_date: str, end_date: str) -> str:
    # Generate cache file path for weather data based on location and date range
    key = json.dumps({"lat": round(lat, 4), "lon": round(lon, 4), "start": start_date, "end": end_date})
    h = hashlib.md5(key.encode()).hexdigest()
    os.makedirs(".weather_cache", exist_ok=True)
    return os.path.join(".weather_cache", f"{h}.csv")


def fetch_hourly_weather(lat: float, lon: float, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    # Retrieve hourly weather data from Open-Meteo API with caching
    cache_fp = _weather_cache_path_csv(lat, lon, start_date, end_date)
    if use_cache and os.path.exists(cache_fp):
        return pd.read_csv(cache_fp, parse_dates=["dt_hour"])  # type: ignore[arg-type]

    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=30,
    )
    r.raise_for_status()
    hourly = (r.json().get("hourly") or {})
    if "time" not in hourly:
        wdf = pd.DataFrame(columns=["dt_hour", "temperature_2m", "precipitation", "wind_speed_10m"])  # empty
    else:
        wdf = pd.DataFrame(hourly)
        wdf["time"] = pd.to_datetime(wdf["time"], errors="coerce")
        wdf["dt_hour"] = wdf["time"].dt.floor("H")
        wdf = wdf.drop_duplicates(subset=["dt_hour"]).reset_index(drop=True)
        wdf = wdf[["dt_hour", "temperature_2m", "precipitation", "wind_speed_10m"]]
    if use_cache and not wdf.empty:
        wdf.to_csv(cache_fp, index=False)
    return wdf


def add_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    # Extract temporal features from datetime column
    out = df.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    out["dt_hour"] = out[datetime_col].dt.floor("h")
    out["hour"] = out[datetime_col].dt.hour.astype("Int64")
    out["day_of_week"] = out[datetime_col].dt.dayofweek.astype("Int64")
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype("Int64")
    out["week_of_year"] = out[datetime_col].dt.isocalendar().week.astype("Int64")
    out["date"] = out[datetime_col].dt.date.astype("string")
    return out


def add_holiday_features(df: pd.DataFrame, datetime_col: str, country_code: str = "CN") -> pd.DataFrame:
    # Add holiday flags and names using the holidays library
    import holidays as pyholidays

    out = df.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    out["date"] = out[datetime_col].dt.date
    years = sorted({d.year for d in out["date"] if pd.notna(d)})
    try:
        cal = None
        for y in years:
            h = pyholidays.country_holidays(country_code, years=y)
            cal = h if cal is None else cal + h
    except Exception:
        cal = None
    if len(out):
        flags, names = zip(*([(1, str(cal.get(d))) if cal and cal.get(d) else (0, "") for d in out["date"]]))
    else:
        flags, names = [], []
    out["is_holiday"] = list(flags)
    out["holiday_name"] = pd.Series(names, index=out.index).astype("string") if len(out) else pd.Series(dtype="string")
    return out


def augment_with_weather(
    # Add weather data to the dataset using either city coordinates or individual GPS points
    df: pd.DataFrame,
    datetime_col: str = "accept_time",
    lat_col: str = "lat",
    lon_col: str = "lng",
    mode: str = "city",  # "city" or "latlng"
    city_lat: float | None = None,
    city_lon: float | None = None,
    country_code: str = "CN",
    round_deg: float = 0.1,
    max_workers: int = 8,
) -> pd.DataFrame:
    base = add_time_features(df, datetime_col)
    base = add_holiday_features(base, datetime_col, country_code=country_code)

    base["date"] = pd.to_datetime(base["date"])
    start_date = base["date"].min()
    end_date = base["date"].max()
    if pd.isna(start_date) or pd.isna(end_date):
        base["temperature_2m"] = pd.NA
        base["precipitation"] = pd.NA
        base["wind_speed_10m"] = pd.NA
        return base
    start_s = start_date.strftime("%Y-%m-%d")
    end_s = end_date.strftime("%Y-%m-%d")

    # Use city-level weather data when mode is "city"
        if city_lat is None or city_lon is None:
            raise ValueError("Provide city_lat and city_lon when mode='city'.")
        wdf = fetch_hourly_weather(city_lat, city_lon, start_s, end_s, use_cache=True)
        return base.merge(wdf, on="dt_hour", how="left")

    # Use individual GPS coordinates for weather data
    base["_lat"] = pd.to_numeric(base.get(lat_col, pd.NA), errors="coerce")
    base["_lon"] = pd.to_numeric(base.get(lon_col, pd.NA), errors="coerce")
    base["_lat_r"] = (base["_lat"] / round_deg).round() * round_deg
    base["_lon_r"] = (base["_lon"] / round_deg).round() * round_deg

    locs = base[["_lat_r", "_lon_r"]].dropna().drop_duplicates()
    if len(locs) > 200:
        locs = locs.sample(200, random_state=42)

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _, row in locs.iterrows():
            la, lo = float(row["_lat_r"]), float(row["_lon_r"])
            futures[ex.submit(fetch_hourly_weather, la, lo, start_s, end_s, True)] = (la, lo)

        frames = []
        for fut in as_completed(futures):
            la, lo = futures[fut]
            try:
                w = fut.result()
                if not w.empty:
                    w["_lat_r"] = la
                    w["_lon_r"] = lo
                    frames.append(w)
            except Exception:
                continue

    if frames:
        weather_all = pd.concat(frames, ignore_index=True)
        base = base.merge(
            weather_all,
            left_on=["dt_hour", "_lat_r", "_lon_r"],
            right_on=["dt_hour", "_lat_r", "_lon_r"],
            how="left",
        )
    else:
        base["temperature_2m"] = pd.NA
        base["precipitation"] = pd.NA
        base["wind_speed_10m"] = pd.NA

    return base.drop(columns=["_lat", "_lon", "_lat_r", "_lon_r"], errors="ignore")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create derived features including business hours, weather buckets, and speed calculations
    out = df.copy()
    # Create flags for business and peak hours
    if "hour" in out.columns:
        out["is_business_hour"] = out["hour"].between(9, 18).astype("Int64")
        out["is_peak_hour"] = out["hour"].isin([11, 12, 13, 17, 18, 19, 20]).astype("Int64")

    # Create categorical buckets for weather variables
    if "precipitation" in out.columns:
        out["is_rain"] = (pd.to_numeric(out["precipitation"], errors="coerce") > 0).astype("Int64")
    if "temperature_2m" in out.columns:
        temp = pd.to_numeric(out["temperature_2m"], errors="coerce")
        out["temp_bucket"] = pd.cut(
            temp, bins=[-np.inf, 0, 10, 20, 30, np.inf], labels=["<=0C", "0-10C", "10-20C", "20-30C", ">30C"],
        ).astype("string")
    if "wind_speed_10m" in out.columns:
        wind = pd.to_numeric(out["wind_speed_10m"], errors="coerce")
        out["wind_bucket"] = pd.cut(
            wind, bins=[-np.inf, 5, 10, 20, np.inf], labels=["calm", "breeze", "windy", "strong"],
        ).astype("string")

    # Calculate delivery speed and create distance buckets
    if {"distance_km", "ata_minutes"}.issubset(out.columns):
        dist = pd.to_numeric(out["distance_km"], errors="coerce")
        mins = pd.to_numeric(out["ata_minutes"], errors="coerce")
        out["speed_kmh"] = np.where(mins > 0, dist / (mins / 60.0), np.nan)
        out["distance_bucket"] = pd.cut(
            dist, bins=[-np.inf, 1, 3, 5, 10, np.inf], labels=["<=1km", "1-3km", "3-5km", "5-10km", ">10km"],
        ).astype("string")

    return out


def drop_outliers_iqr(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    # Remove outliers using interquartile range method
    def iqr_bounds(s: pd.Series, k_: float = 1.5) -> tuple[float, float]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return q1 - k_ * iqr, q3 + k_ * iqr

    out = df.copy()
    mask = pd.Series(True, index=out.index)
    for col in cols:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        lo, hi = iqr_bounds(s, k_=k)
        mask &= s.between(lo, hi, inclusive="both")
    return out[mask]




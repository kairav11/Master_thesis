from __future__ import annotations

"""
ATA EDA Toolkit

Reusable plotting and EDA utilities for exploring Actual Time of Arrival (ATA, minutes).

Dependencies: Python 3.10+, pandas, numpy, matplotlib, seaborn, scipy (optional)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import warnings
import os
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy import stats  # optional (LOWESS not used; correlations, residuals)
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ------------------------------
# Global styling and utilities
# ------------------------------

PALETTE: List[tuple] = []  # populated in setup_plotting


def setup_plotting(style: str = "whitegrid", context: str = "notebook") -> None:
    """Configure seaborn/matplotlib defaults for legible plots.

    Parameters
    ----------
    style: seaborn style (e.g., "whitegrid", "darkgrid")
    context: seaborn context (e.g., "paper", "notebook", "talk", "poster")
    """
    sns.set_theme(style=style, context=context)
    sns.set_palette("colorblind")
    global PALETTE
    PALETTE = list(sns.color_palette("colorblind"))
    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "savefig.dpi": 150,
    })


def ensure_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    """Raise a clear error if any required column is missing.

    Parameters
    ----------
    df: input DataFrame
    required_cols: list of required column names
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _warn_or_skip(condition: bool, message: str) -> bool:
    if condition:
        return True
    warnings.warn(message)
    return False


def _dropna_for(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if not present:
        return df.iloc[0:0]
    return df.dropna(subset=present)


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_int(s: pd.Series) -> pd.Series:
    # Handles pandas Int64 nullable to int where possible
    return pd.to_numeric(s, errors="coerce").astype(float)


def _top_n_category(df: pd.DataFrame, col: str, top_n: int = 20) -> pd.Series:
    counts = df[col].value_counts(dropna=False)
    keep = set(counts.head(top_n).index)
    mapped = df[col].where(df[col].isin(keep), other="Other")
    return mapped


def savefig(fig: plt.Figure, outdir: Optional[str], name: str) -> List[str]:
    """Save a figure to PNG and SVG; return saved paths.

    Parameters
    ----------
    fig: matplotlib Figure
    outdir: output directory (created if needed). If None, no saving.
    name: base filename without extension
    """
    saved: List[str] = []
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        png = os.path.join(outdir, f"{name}.png")
        svg = os.path.join(outdir, f"{name}.svg")
        fig.savefig(png)
        fig.savefig(svg)
        saved.extend([png, svg])
    return saved


# ------------------------------
# Target distribution
# ------------------------------

def plot_target_hist(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Histogram + KDE of ata_minutes, annotated with skew, mean, median."""
    if not _warn_or_skip("ata_minutes" in df.columns, "plot_target_hist: 'ata_minutes' missing"):
        return (plt.figure(), plt.gca())
    s = _coerce_numeric(df["ata_minutes"]).dropna()
    fig, ax = plt.subplots()
    sns.histplot(s, bins=50, kde=True, ax=ax, color="#4c78a8")
    mean_v, med_v = float(s.mean()), float(s.median())
    skew_v = float(s.skew())
    ax.axvline(mean_v, color="orange", linestyle="--", label=f"Mean={mean_v:.2f}")
    ax.axvline(med_v, color="green", linestyle=":", label=f"Median={med_v:.2f}")
    ax.set_title(f"ATA Distribution (skew={skew_v:.2f}, N={len(s)})")
    ax.set_xlabel("ATA (minutes)")
    ax.legend()
    savefig(fig, outdir, "01_target_hist")
    return fig, ax


def plot_target_box(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Boxplot of ata_minutes with whiskers and outlier count."""
    if not _warn_or_skip("ata_minutes" in df.columns, "plot_target_box: 'ata_minutes' missing"):
        return (plt.figure(), plt.gca())
    s = _coerce_numeric(df["ata_minutes"]).dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int((s < lo).sum() + (s > hi).sum())
    fig, ax = plt.subplots()
    sns.boxplot(x=s, ax=ax, color="#72b7b2")
    ax.set_title(f"ATA Boxplot (outliers≈{outliers}, N={len(s)})")
    ax.set_xlabel("ATA (minutes)")
    savefig(fig, outdir, "02_target_box")
    return fig, ax


# ------------------------------
# Univariate features
# ------------------------------

def _apply_hatch(ax: plt.Axes) -> None:
    hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    for i, p in enumerate(getattr(ax, "patches", [])):
        try:
            p.set_hatch(hatches[i % len(hatches)])
            p.set_edgecolor("black")
        except Exception:
            continue


def plot_categorical_by_count(df: pd.DataFrame, col: str, top_n: int = 20, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Barplot of category counts (top N)."""
    if not _warn_or_skip(col in df.columns, f"plot_categorical_by_count: '{col}' missing"):
        return (plt.figure(), plt.gca())
    work = df.copy()
    work[col] = _top_n_category(work, col, top_n=top_n)
    counts = work[col].value_counts()
    fig, ax = plt.subplots(figsize=(8, max(3, min(10, len(counts) * 0.3))))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color=PALETTE[0])
    _apply_hatch(ax)
    ax.set_title(f"Counts by {col} (top {top_n})")
    ax.set_xlabel("Count")
    ax.set_ylabel(col)
    savefig(fig, outdir, f"cat_count_{col}")
    return fig, ax


def plot_ata_by_category(df: pd.DataFrame, col: str, top_n: int = 20, show_violin: bool = True, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Box/violin plot of ATA per category (top N categories)."""
    if not _warn_or_skip({col, "ata_minutes"}.issubset(df.columns), f"plot_ata_by_category: need '{col}', 'ata_minutes'"):
        return (plt.figure(), plt.gca())
    work = df.copy()
    work[col] = _top_n_category(work, col, top_n=top_n)
    work = _dropna_for(work, [col, "ata_minutes"])  # only columns used
    fig, ax = plt.subplots(figsize=(10, max(3, min(12, work[col].nunique() * 0.3))))
    if show_violin:
        sns.violinplot(data=work, x="ata_minutes", y=col, inner="box", ax=ax, palette="colorblind")
    else:
        sns.boxplot(data=work, x="ata_minutes", y=col, ax=ax, palette="colorblind")
    ax.set_title(f"ATA by {col} (top {top_n})")
    ax.set_xlabel("ATA (minutes)")
    ax.set_ylabel(col)
    savefig(fig, outdir, f"ata_by_{col}")
    return fig, ax


def plot_numeric_hist(df: pd.DataFrame, col: str, bins: int = 50, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Histogram + KDE for a numeric feature with mean/std annotations."""
    if not _warn_or_skip(col in df.columns, f"plot_numeric_hist: '{col}' missing"):
        return (plt.figure(), plt.gca())
    s = _coerce_numeric(df[col]).dropna()
    fig, ax = plt.subplots()
    sns.histplot(s, bins=bins, kde=True, ax=ax, color=PALETTE[1])
    ax.axvline(float(s.mean()), color="orange", linestyle="--", label=f"Mean={float(s.mean()):.2f}")
    ax.set_title(f"Distribution: {col} (std={float(s.std()):.2f}, N={len(s)})")
    ax.legend()
    savefig(fig, outdir, f"num_hist_{col}")
    return fig, ax


# ------------------------------
# Relationships with ATA
# ------------------------------

def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="pearson")), float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def plot_scatter_with_trend(df: pd.DataFrame, x: str, y: str = "ata_minutes", sample: int = 5000, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Scatter with optional downsampling and a simple linear trendline; shows Pearson/Spearman."""
    needed = {x, y}
    if not _warn_or_skip(needed.issubset(df.columns), f"plot_scatter_with_trend: need {needed}"):
        return (plt.figure(), plt.gca())
    work = _dropna_for(df[[x, y]].copy(), [x, y])
    if len(work) == 0:
        warnings.warn("plot_scatter_with_trend: empty after dropping NA")
        return (plt.figure(), plt.gca())
    if len(work) > sample:
        work = work.sample(sample, random_state=42)
    xnum = _coerce_numeric(work[x]).to_numpy()
    ynum = _coerce_numeric(work[y]).to_numpy()
    fig, ax = plt.subplots()
    ax.scatter(xnum, ynum, alpha=0.35, s=10, color=PALETTE[0])
    # Simple linear fit
    if len(work) >= 3:
        try:
            slope, intercept = np.polyfit(xnum, ynum, 1)
            xs = np.linspace(xnum.min(), xnum.max(), 100)
            ax.plot(xs, slope * xs + intercept, color=PALETTE[2], linewidth=2, label="Linear trend")
        except Exception:
            pass
    p, s = _pearson_spearman(xnum, ynum)
    ax.set_title(f"{y} vs {x} (N={len(work)}, r={p:.2f}, ρ={s:.2f})")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    slug = f"scatter_{x}_vs_{y}"
    savefig(fig, outdir, slug)
    return fig, ax


def plot_distance_vs_ata(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    return plot_scatter_with_trend(df, x="distance_km", y="ata_minutes", outdir=outdir)


def plot_speed_vs_ata(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    return plot_scatter_with_trend(df, x="speed_kmh", y="ata_minutes", outdir=outdir)


def plot_temperature_vs_ata(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    return plot_scatter_with_trend(df, x="temperature_2m", y="ata_minutes", outdir=outdir)


def plot_precipitation_vs_ata(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    return plot_scatter_with_trend(df, x="precipitation", y="ata_minutes", outdir=outdir)


def plot_wind_vs_ata(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    return plot_scatter_with_trend(df, x="wind_speed_10m", y="ata_minutes", outdir=outdir)


# ------------------------------
# Time patterns
# ------------------------------

def _ensure_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hour" not in out.columns and "accept_time" in out.columns:
        out["hour"] = pd.to_datetime(out["accept_time"], errors="coerce").dt.hour
    if "day_of_week" not in out.columns and "accept_time" in out.columns:
        out["day_of_week"] = pd.to_datetime(out["accept_time"], errors="coerce").dt.dayofweek
    if "week_of_year" not in out.columns and ("accept_time" in out.columns or "date" in out.columns):
        t = pd.to_datetime(out.get("accept_time", out.get("date")), errors="coerce")
        out["week_of_year"] = t.dt.isocalendar().week.astype("Int64")
    if "date" not in out.columns and "accept_time" in out.columns:
        out["date"] = pd.to_datetime(out["accept_time"], errors="coerce").dt.date
    return out


def _mean_ci(series: pd.Series) -> Tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) == 0:
        return float("nan"), float("nan")
    m = float(s.mean())
    if len(s) <= 1:
        return m, float("nan")
    se = float(s.std(ddof=1) / math.sqrt(len(s)))
    ci = 1.96 * se
    return m, ci


def plot_ata_by_hour(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    work = _ensure_time_parts(df)
    if not _warn_or_skip({"hour", "ata_minutes"}.issubset(work.columns), "plot_ata_by_hour: need 'hour', 'ata_minutes'"):
        return (plt.figure(), plt.gca())
    tmp = _dropna_for(work, ["hour", "ata_minutes"]).groupby("hour", as_index=False)["ata_minutes"].apply(lambda s: pd.Series({"mean": s.mean(), "ci": _mean_ci(s)[1]}))
    fig, ax = plt.subplots()
    ax.errorbar(tmp["hour"], tmp["mean"], yerr=tmp["ci"], fmt="-o", color="#4c78a8")
    ax.set_title(f"Mean ATA by Hour (N={int(work['hour'].notna().sum())}, 95% CI)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("ATA (minutes)")
    ax.axvspan(11, 13, color="orange", alpha=0.1)
    ax.axvspan(17, 20, color="orange", alpha=0.1)
    savefig(fig, outdir, "03_ata_by_hour")
    return fig, ax


def plot_ata_by_dow(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    work = _ensure_time_parts(df)
    if not _warn_or_skip({"day_of_week", "ata_minutes"}.issubset(work.columns), "plot_ata_by_dow: need 'day_of_week', 'ata_minutes'"):
        return (plt.figure(), plt.gca())
    tmp = _dropna_for(work, ["day_of_week", "ata_minutes"]).groupby("day_of_week", as_index=False)["ata_minutes"].mean()
    fig, ax = plt.subplots()
    sns.pointplot(data=tmp, x="day_of_week", y="ata_minutes", ax=ax, color="#4c78a8")
    ax.set_title("Mean ATA by Day of Week (0=Mon...6=Sun)")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("ATA (minutes)")
    ax.axvspan(4.5, 6.5, color="orange", alpha=0.1)  # mark weekend
    savefig(fig, outdir, "04_ata_by_dow")
    return fig, ax


def plot_ata_by_week(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    work = _ensure_time_parts(df)
    if not _warn_or_skip({"week_of_year", "ata_minutes"}.issubset(work.columns), "plot_ata_by_week: need 'week_of_year', 'ata_minutes'"):
        return (plt.figure(), plt.gca())
    tmp = _dropna_for(work, ["week_of_year", "ata_minutes"]).groupby("week_of_year", as_index=False)["ata_minutes"].mean()
    fig, ax = plt.subplots()
    ax.plot(tmp["week_of_year"], tmp["ata_minutes"], marker="o", color="#4c78a8")
    ax.set_title("Weekly Mean ATA")
    ax.set_xlabel("ISO Week")
    ax.set_ylabel("ATA (minutes)")
    savefig(fig, outdir, "05_ata_by_week")
    return fig, ax


def plot_daily_ts(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    work = df.copy()
    if "date" not in work.columns:
        if "accept_time" in work.columns:
            work["date"] = pd.to_datetime(work["accept_time"], errors="coerce").dt.date
    if not _warn_or_skip({"date", "ata_minutes"}.issubset(work.columns), "plot_daily_ts: need 'date', 'ata_minutes'"):
        return (plt.figure(), plt.gca())
    tmp = _dropna_for(work, ["date", "ata_minutes"]).groupby("date", as_index=False)["ata_minutes"].mean()
    if tmp.empty:
        warnings.warn("plot_daily_ts: no data after grouping")
        return (plt.figure(), plt.gca())
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.sort_values("date")
    tmp["roll7"] = tmp["ata_minutes"].rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tmp["date"], tmp["ata_minutes"], alpha=0.6, label="Daily mean")
    ax.plot(tmp["date"], tmp["roll7"], color="orange", label="7d rolling mean")
    ax.set_title("Daily Mean ATA")
    ax.set_xlabel("Date")
    ax.set_ylabel("ATA (minutes)")
    ax.legend()
    savefig(fig, outdir, "06_daily_ts")
    return fig, ax


def plot_hour_dow_heatmap(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    work = _ensure_time_parts(df)
    if not _warn_or_skip({"hour", "day_of_week", "ata_minutes"}.issubset(work.columns), "plot_hour_dow_heatmap: need hour, day_of_week, ata_minutes"):
        return (plt.figure(), plt.gca())
    tmp = _dropna_for(work, ["hour", "day_of_week", "ata_minutes"]).groupby(["day_of_week", "hour"], as_index=False)["ata_minutes"].mean()
    if tmp.empty:
        warnings.warn("plot_hour_dow_heatmap: empty after grouping")
        return (plt.figure(), plt.gca())
    pivot = tmp.pivot(index="day_of_week", columns="hour", values="ata_minutes")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, ax=ax, cmap="coolwarm", center=0)
    ax.set_title("Mean ATA by Day-of-Week and Hour")
    savefig(fig, outdir, "07_hour_dow_heatmap")
    return fig, ax


# ------------------------------
# Spatial views
# ------------------------------

def plot_geo_scatter(df: pd.DataFrame, lon_col: str = "delivery_gps_lng", lat_col: str = "delivery_gps_lat", color: str = "ata_minutes", sample: int = 10000, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    cols = {lon_col, lat_col, color}
    if not _warn_or_skip(cols.issubset(df.columns), f"plot_geo_scatter: need {cols}"):
        return (plt.figure(), plt.gca())
    work = _dropna_for(df[[lon_col, lat_col, color]].copy(), [lon_col, lat_col, color])
    if len(work) > sample:
        work = work.sample(sample, random_state=42)
    fig, ax = plt.subplots(figsize=(6, 6))
    h = ax.scatter(work[lon_col], work[lat_col], c=work[color], s=8, alpha=0.6, cmap="viridis")
    cb = fig.colorbar(h, ax=ax)
    cb.set_label(color)
    ax.set_title(f"Geo scatter colored by {color} (N={len(work)})")
    ax.set_xlabel(lon_col)
    ax.set_ylabel(lat_col)
    savefig(fig, outdir, "08_geo_scatter")
    return fig, ax


def plot_city_region_ata(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    if not _warn_or_skip("ata_minutes" in df.columns, "plot_city_region_ata: 'ata_minutes' missing"):
        return (plt.figure(), plt.gca())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "city" in df.columns:
        top_city = _top_n_category(df, "city", top_n=20)
        tmp = df.assign(city=top_city).dropna(subset=["city", "ata_minutes"]).groupby("city", as_index=False).agg(mean_ata=("ata_minutes", "mean"), sem_ata=("ata_minutes", lambda s: s.std(ddof=1) / math.sqrt(len(s))))
        sns.barplot(data=tmp.sort_values("mean_ata", ascending=False), x="mean_ata", y="city", ax=axes[0], color="#4c78a8")
        axes[0].errorbar(tmp.sort_values("mean_ata", ascending=False)["mean_ata"], range(len(tmp)), xerr=tmp.sort_values("mean_ata", ascending=False)["sem_ata"], fmt="none", ecolor="black", alpha=0.6)
        axes[0].set_title("Mean ATA by City (top 20)")
        axes[0].set_xlabel("ATA (minutes)")
        axes[0].set_ylabel("city")
    else:
        axes[0].axis("off")
    if "region_id" in df.columns:
        top_region = _top_n_category(df, "region_id", top_n=20)
        tmp2 = df.assign(region_id=top_region).dropna(subset=["region_id", "ata_minutes"]).groupby("region_id", as_index=False).agg(mean_ata=("ata_minutes", "mean"), sem_ata=("ata_minutes", lambda s: s.std(ddof=1) / math.sqrt(len(s))))
        sns.barplot(data=tmp2.sort_values("mean_ata", ascending=False), x="mean_ata", y="region_id", ax=axes[1], color="#72b7b2")
        axes[1].errorbar(tmp2.sort_values("mean_ata", ascending=False)["mean_ata"], range(len(tmp2)), xerr=tmp2.sort_values("mean_ata", ascending=False)["sem_ata"], fmt="none", ecolor="black", alpha=0.6)
        axes[1].set_title("Mean ATA by Region (top 20)")
        axes[1].set_xlabel("ATA (minutes)")
        axes[1].set_ylabel("region_id")
    else:
        axes[1].axis("off")
    savefig(fig, outdir, "09_city_region_ata")
    return fig, axes[0]


# ------------------------------
# Weather effects
# ------------------------------

def plot_weather_boxes(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    cats = [c for c in ["is_rain", "temp_bucket", "wind_bucket"] if c in df.columns]
    if not cats or "ata_minutes" not in df.columns:
        warnings.warn("plot_weather_boxes: required columns missing")
        return (plt.figure(), plt.gca())
    n = len(cats)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    for ax, col in zip(axes, cats):
        d = _dropna_for(df[[col, "ata_minutes"]].copy(), [col, "ata_minutes"])
        if d.empty:
            continue
        try:
            sns.violinplot(data=d, x=col, y="ata_minutes", inner="box", ax=ax, color="#72b7b2")
        except Exception:
            sns.boxplot(data=d, x=col, y="ata_minutes", ax=ax, color="#72b7b2")
        ax.set_title(f"ATA by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("ATA (minutes)")
    savefig(fig, outdir, "10_weather_boxes")
    return fig, axes[0]


def plot_weather_scatter(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    cols = [c for c in ["temperature_2m", "precipitation", "wind_speed_10m"] if c in df.columns]
    if not cols or "ata_minutes" not in df.columns:
        warnings.warn("plot_weather_scatter: required columns missing")
        return (plt.figure(), plt.gca())
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    for ax, col in zip(axes, cols):
        work = _dropna_for(df[[col, "ata_minutes"]], [col, "ata_minutes"]).copy()
        if len(work) > 5000:
            work = work.sample(5000, random_state=42)
        xnum = _coerce_numeric(work[col]).to_numpy()
        ynum = _coerce_numeric(work["ata_minutes"]).to_numpy()
        ax.scatter(xnum, ynum, alpha=0.3, s=10, color="#4c78a8")
        # linear trend
        if len(work) >= 3:
            try:
                slope, intercept = np.polyfit(xnum, ynum, 1)
                xs = np.linspace(xnum.min(), xnum.max(), 100)
                ax.plot(xs, slope * xs + intercept, color="orange")
            except Exception:
                pass
        p, s = _pearson_spearman(xnum, ynum)
        ax.set_title(f"{col} vs ATA (r={p:.2f}, ρ={s:.2f})")
        ax.set_xlabel(col)
        ax.set_ylabel("ATA (minutes)")
    savefig(fig, outdir, "11_weather_scatter")
    return fig, axes[0]


# ------------------------------
# Distance & speed
# ------------------------------

def plot_ata_by_distance_bucket(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    if not _warn_or_skip({"distance_bucket", "ata_minutes"}.issubset(df.columns), "plot_ata_by_distance_bucket: need 'distance_bucket', 'ata_minutes'"):
        return (plt.figure(), plt.gca())
    work = _dropna_for(df, ["distance_bucket", "ata_minutes"]).copy()
    fig, ax = plt.subplots()
    sns.boxplot(data=work, x="distance_bucket", y="ata_minutes", ax=ax, color="#72b7b2")
    ax.set_title("ATA by Distance Bucket")
    ax.set_xlabel("distance_bucket")
    ax.set_ylabel("ATA (minutes)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    savefig(fig, outdir, "12_ata_by_distance_bucket")
    return fig, ax


def plot_speed_vs_distance(df: pd.DataFrame, sample: int = 5000, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    if not _warn_or_skip({"distance_km", "speed_kmh"}.issubset(df.columns), "plot_speed_vs_distance: need 'distance_km', 'speed_kmh'"):
        return (plt.figure(), plt.gca())
    work = _dropna_for(df[["distance_km", "speed_kmh"]], ["distance_km", "speed_kmh"]).copy()
    if len(work) > sample:
        work = work.sample(sample, random_state=42)
    fig, ax = plt.subplots()
    hb = ax.hexbin(work["distance_km"], work["speed_kmh"], gridsize=40, cmap="viridis", mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("count")
    ax.set_title("Speed vs Distance (hexbin)")
    ax.set_xlabel("distance_km")
    ax.set_ylabel("speed_kmh")
    savefig(fig, outdir, "13_speed_vs_distance")
    return fig, ax


# ------------------------------
# Courier & AOI
# ------------------------------

def plot_ata_by_courier(df: pd.DataFrame, top_n: int = 15, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    if not _warn_or_skip({"courier_id", "ata_minutes"}.issubset(df.columns), "plot_ata_by_courier: need 'courier_id','ata_minutes'"):
        return (plt.figure(), plt.gca())
    work = df.copy()
    work["courier_id"] = _top_n_category(work, "courier_id", top_n=top_n)
    work = _dropna_for(work, ["courier_id", "ata_minutes"]).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=work, x="courier_id", y="ata_minutes", ax=ax, color="#72b7b2")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(f"ATA by Courier (top {top_n})")
    ax.set_xlabel("courier_id")
    ax.set_ylabel("ATA (minutes)")
    savefig(fig, outdir, "14_ata_by_courier")
    return fig, ax


def plot_ata_by_aoi_type(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    if not _warn_or_skip({"aoi_type", "ata_minutes"}.issubset(df.columns), "plot_ata_by_aoi_type: need 'aoi_type','ata_minutes'"):
        return (plt.figure(), plt.gca())
    work = _dropna_for(df, ["aoi_type", "ata_minutes"]).copy()
    fig, ax = plt.subplots()
    sns.boxplot(data=work, x="aoi_type", y="ata_minutes", ax=ax, color="#72b7b2")
    ax.set_title("ATA by AOI Type")
    ax.set_xlabel("aoi_type")
    ax.set_ylabel("ATA (minutes)")
    savefig(fig, outdir, "15_ata_by_aoi_type")
    return fig, ax


# ------------------------------
# Correlations
# ------------------------------

def plot_corr_heatmap(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    num_df = df.select_dtypes(include=[np.number])
    if "ata_minutes" in df.columns and "ata_minutes" not in num_df.columns:
        num_df = num_df.join(_coerce_numeric(df["ata_minutes"]))
    if num_df.shape[1] < 2:
        warnings.warn("plot_corr_heatmap: not enough numeric columns")
        return (plt.figure(), plt.gca())
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (numeric features)")
    savefig(fig, outdir, "16_corr_heatmap")
    return fig, ax


def plot_pairgrid(df: pd.DataFrame, cols: Sequence[str], outdir: Optional[str] = None) -> Optional[sns.axisgrid.PairGrid]:
    for c in cols:
        if c not in df.columns:
            warnings.warn(f"plot_pairgrid: column '{c}' missing, skipping")
            return None
    data = _dropna_for(df[cols], cols)
    if data.empty:
        warnings.warn("plot_pairgrid: empty after dropping NA")
        return None
    g = sns.PairGrid(data)
    g = g.map_diag(sns.kdeplot, fill=True)
    g = g.map_offdiag(sns.scatterplot, s=10, alpha=0.3)
    savefig(g.fig, outdir, "17_pairgrid")
    return g


# ------------------------------
# Outliers via simple linear fits
# ------------------------------

def _annotate_top_residuals(ax: plt.Axes, x: np.ndarray, y: np.ndarray, k: int = 10) -> None:
    if len(x) < 3:
        return
    try:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        resid = y - yhat
        if SCIPY_AVAILABLE and len(x) > 3:
            # Studentized residuals approximation
            _, _, _, _, mse = np.polyfit(x, y, 1, full=True)
            mse = mse[0] / (len(x) - 2) if len(mse) else np.var(resid)
            se = np.sqrt(mse * (1/len(x) + (x - x.mean())**2 / ((x - x.mean())**2).sum()))
            stud = resid / (se + 1e-9)
            idx = np.argsort(np.abs(stud))[-k:]
        else:
            idx = np.argsort(np.abs(resid))[-k:]
        for i in idx:
            ax.annotate("•", (x[i], y[i]), color="red")
    except Exception:
        pass


def plot_outliers_distance_ata(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    if not _warn_or_skip({"distance_km", "ata_minutes"}.issubset(df.columns), "plot_outliers_distance_ata: need 'distance_km','ata_minutes'"):
        return (plt.figure(), plt.gca())
    work = _dropna_for(df, ["distance_km", "ata_minutes"]).copy()
    fig, ax = plt.subplots()
    ax.scatter(work["distance_km"], work["ata_minutes"], alpha=0.3, s=10, color="#4c78a8")
    _annotate_top_residuals(ax, _coerce_numeric(work["distance_km"]).to_numpy(), _coerce_numeric(work["ata_minutes"]).to_numpy())
    ax.set_title("Outliers: distance vs ATA (top residuals marked)")
    ax.set_xlabel("distance_km")
    ax.set_ylabel("ATA (minutes)")
    savefig(fig, outdir, "18_outliers_distance_ata")
    return fig, ax


def plot_outliers_speed_ata(df: pd.DataFrame, outdir: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    if not _warn_or_skip({"speed_kmh", "ata_minutes"}.issubset(df.columns), "plot_outliers_speed_ata: need 'speed_kmh','ata_minutes'"):
        return (plt.figure(), plt.gca())
    work = _dropna_for(df, ["speed_kmh", "ata_minutes"]).copy()
    fig, ax = plt.subplots()
    ax.scatter(work["speed_kmh"], work["ata_minutes"], alpha=0.3, s=10, color="#4c78a8")
    _annotate_top_residuals(ax, _coerce_numeric(work["speed_kmh"]).to_numpy(), _coerce_numeric(work["ata_minutes"]).to_numpy())
    ax.set_title("Outliers: speed vs ATA (top residuals marked)")
    ax.set_xlabel("speed_kmh")
    ax.set_ylabel("ATA (minutes)")
    savefig(fig, outdir, "19_outliers_speed_ata")
    return fig, ax


# ------------------------------
# CLI and demo helpers
# ------------------------------

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["accept_time", "delivery_time", "dt_hour", "date"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def run_eda_suite(df: pd.DataFrame, outdir: str) -> List[str]:
    """Run a standard EDA suite and save figures to outdir; return saved paths."""
    setup_plotting()
    os.makedirs(outdir, exist_ok=True)
    saved: List[str] = []

    # Core target
    saved += savefig(plot_target_hist(df, outdir)[0], outdir, "01_target_hist")
    saved += savefig(plot_target_box(df, outdir)[0], outdir, "02_target_box")

    # Time
    plot_ata_by_hour(df, outdir)
    plot_ata_by_dow(df, outdir)
    plot_ata_by_week(df, outdir)
    plot_daily_ts(df, outdir)
    plot_hour_dow_heatmap(df, outdir)

    # Relationships
    plot_distance_vs_ata(df, outdir)
    plot_speed_vs_ata(df, outdir)

    # Weather
    plot_weather_boxes(df, outdir)
    plot_weather_scatter(df, outdir)

    # City/region and buckets
    plot_city_region_ata(df, outdir)
    plot_ata_by_distance_bucket(df, outdir)

    # Correlations
    plot_corr_heatmap(df, outdir)

    return saved


# ------------------------------
# Synthetic data for smoke tests
# ------------------------------

def make_fake_df(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset with a similar schema for quick testing."""
    rng = np.random.default_rng(seed)
    hours = rng.integers(6, 23, size=n)
    dow = rng.integers(0, 7, size=n)
    dist = rng.gamma(shape=2.0, scale=2.0, size=n)  # skewed distance
    speed = np.clip(rng.normal(loc=15, scale=5, size=n), 1, 50)
    temp = rng.normal(loc=22, scale=6, size=n)
    precip = np.clip(rng.gamma(1.5, 0.5, size=n) - 0.5, 0, None)
    wind = np.clip(rng.normal(10, 4, size=n), 0, None)
    is_rain = (precip > 0.1).astype(int)
    base = 60 + 5 * (hours >= 17) + 5 * (hours <= 9) + 3 * is_rain
    ata = base + 8 * np.sqrt(dist) + rng.normal(0, 10, size=n)
    date = pd.Timestamp("2022-06-01") + pd.to_timedelta(rng.integers(0, 120, size=n), unit="D")
    accept_time = date + pd.to_timedelta(hours, unit="H")

    df = pd.DataFrame({
        "order_id": rng.integers(1_000_000, 9_999_999, size=n),
        "region_id": rng.integers(1, 100, size=n),
        "city": rng.choice(["CityA", "CityB", "CityC", "CityD"], size=n, replace=True),
        "courier_id": rng.integers(100, 200, size=n),
        "lng": rng.uniform(120, 130, size=n),
        "lat": rng.uniform(30, 45, size=n),
        "aoi_id": rng.integers(1, 500, size=n),
        "aoi_type": rng.integers(0, 15, size=n),
        "accept_time": accept_time,
        "delivery_time": accept_time + pd.to_timedelta(np.clip(ata, 5, None), unit="m"),
        "distance_km": dist,
        "ata_minutes": ata,
        "hour": hours,
        "day_of_week": dow,
        "week_of_year": pd.Timestamp.today().isocalendar().week,
        "date": date.date,
        "temperature_2m": temp,
        "precipitation": precip,
        "wind_speed_10m": wind,
        "is_weekend": (dow >= 5).astype(int),
        "is_rain": is_rain,
        "speed_kmh": speed,
        "distance_bucket": pd.cut(dist, [-np.inf, 1, 3, 5, 10, np.inf], labels=["<=1km","1-3km","3-5km","5-10km",">10km"]).astype(str),
        "temp_bucket": pd.cut(temp, [-np.inf, 0, 10, 20, 30, np.inf], labels=["<=0C","0-10C","10-20C","20-30C",">30C"]).astype(str),
        "wind_bucket": pd.cut(wind, [-np.inf, 5, 10, 20, np.inf], labels=["calm","breeze","windy","strong"]).astype(str),
        "delivery_gps_lng": rng.uniform(120, 130, size=n),
        "delivery_gps_lat": rng.uniform(30, 45, size=n),
    })
    return df


def quick_demo(outdir: str = "_eda_demo") -> None:
    df = make_fake_df(2000)
    os.makedirs(outdir, exist_ok=True)
    setup_plotting()
    # a few plots
    plot_target_hist(df, outdir)
    plot_ata_by_hour(df, outdir)
    plot_distance_vs_ata(df, outdir)
    plot_corr_heatmap(df, outdir)
    print(f"Demo plots saved under: {outdir}")


# ------------------------------
# CLI
# ------------------------------

def _cli() -> None:
    p = argparse.ArgumentParser(description="ATA EDA plotting toolkit")
    p.add_argument("csv", type=str, help="Path to CSV")
    p.add_argument("--outdir", type=str, default="ata_eda_output", help="Directory to save figures")
    p.add_argument("--query", type=str, default="", help="Optional pandas query string to subset data")
    args = p.parse_args()

    setup_plotting()
    df = pd.read_csv(args.csv)
    df = _parse_dates(df)
    if args.query:
        try:
            df = df.query(args.query)
        except Exception as e:
            warnings.warn(f"Failed to apply query: {e}")

    paths = run_eda_suite(df, outdir=args.outdir)
    print("Saved figures:")
    for pth in sorted(os.listdir(args.outdir)):
        if pth.lower().endswith((".png", ".svg")):
            print(os.path.join(args.outdir, pth))


if __name__ == "__main__":
    # Example usage (notebook):
    # from ata_eda import setup_plotting, plot_target_hist, plot_corr_heatmap, make_fake_df
    # setup_plotting()
    # df = make_fake_df(5000)
    # plot_target_hist(df)
    # plot_corr_heatmap(df)
    # plt.show()
    #
    # CLI:
    # python ata_eda.py your_data.csv --outdir eda_output --query "city == 'Jilin'"
    _cli()



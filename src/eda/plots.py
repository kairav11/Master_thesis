from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_dir(path: str) -> None:
    # Create output directory if it does not exist
    os.makedirs(path, exist_ok=True)


def _save(fig, out_fp: str) -> None:
    # Save matplotlib figure to file with proper formatting
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150)
    plt.close(fig)


def _apply_hatch_to_bars(ax: plt.Axes) -> None:
    # Apply hatch patterns to bar charts for improved accessibility
    hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    for i, p in enumerate(getattr(ax, "patches", [])):
        try:
            p.set_hatch(hatches[i % len(hatches)])
            p.set_edgecolor("black")
        except Exception:
            continue


def _apply_hatch_to_boxplot(ax: plt.Axes) -> None:
    # Apply hatch patterns to box plots for visual distinction
    hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    artists = getattr(ax, "artists", [])
    for i, a in enumerate(artists):
        try:
            a.set_hatch(hatches[i % len(hatches)])
            a.set_edgecolor("black")
        except Exception:
            continue

def plot_correlation(df: pd.DataFrame, out_dir: str, title: str) -> None:
    # Generate correlation matrix heatmap for numeric features
    num_cols = [
        c for c in [
            "ata_minutes","distance_km","hour","day_of_week","week_of_year",
            "temperature_2m","precipitation","wind_speed_10m","is_weekend",
            "is_business_hour","is_peak_hour","is_rain","speed_kmh",
        ] if c in df.columns
    ]
    if len(num_cols) < 2:
        return
    corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="crest", center=0, ax=ax)
    ax.set_title(f"Correlation Matrix - {title}")
    _save(fig, os.path.join(out_dir, "correlation_matrix.png"))


def plot_distributions(df: pd.DataFrame, out_dir: str, title: str) -> None:
    # Create distribution plots for key numeric variables
    to_plot = [c for c in ["ata_minutes","distance_km","temperature_2m","precipitation","wind_speed_10m"] if c in df.columns]
    if not to_plot:
        return
    n = len(to_plot)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for i, col in enumerate(to_plot):
        sns.histplot(pd.to_numeric(df[col], errors="coerce"), bins=50, kde=True, ax=axes[i], color=sns.color_palette("colorblind")[0])
        axes[i].set_title(f"Distribution: {col}")
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    fig.suptitle(f"Distributions - {title}")
    _save(fig, os.path.join(out_dir, "distributions.png"))


def plot_box_by_hour(df: pd.DataFrame, out_dir: str, title: str) -> None:
    # Generate box plots showing delivery time distribution by hour
    if not {"hour","ata_minutes"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df, x="hour", y="ata_minutes", ax=ax, palette="colorblind")
    _apply_hatch_to_boxplot(ax)
    ax.set_title(f"ATA by Hour - {title}")
    _save(fig, os.path.join(out_dir, "ata_by_hour.png"))


def plot_scatter_distance_ata(df: pd.DataFrame, out_dir: str, title: str) -> None:
    # Create scatter plot showing relationship between distance and delivery time
    if not {"distance_km","ata_minutes"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    hue = "is_rain" if "is_rain" in df.columns else None
    sns.scatterplot(data=df.sample(min(len(df), 10000), random_state=42), x="distance_km", y="ata_minutes", hue=hue, style=hue, alpha=0.4, ax=ax, palette="colorblind")
    ax.set_title(f"ATA vs Distance - {title}")
    _save(fig, os.path.join(out_dir, "ata_vs_distance.png"))


def plot_time_series(df: pd.DataFrame, out_dir: str, title: str) -> None:
    # Generate time series plot of average delivery times by hour
    if "accept_time" not in df.columns or "ata_minutes" not in df.columns:
        return
    t = pd.to_datetime(df["accept_time"], errors="coerce").dt.floor("h")
    series = pd.DataFrame({"dt_hour": t, "ata_minutes": pd.to_numeric(df["ata_minutes"], errors="coerce")})
    series = series.dropna().groupby("dt_hour", as_index=False).mean()
    if series.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series["dt_hour"], series["ata_minutes"], linewidth=1)
    ax.set_title(f"Hourly Mean ATA - {title}")
    ax.set_xlabel("Time")
    ax.set_ylabel("ATA (min)")
    _save(fig, os.path.join(out_dir, "hourly_mean_ata.png"))


def generate_eda_plots(df: pd.DataFrame, out_dir: str, title: str) -> None:
    # Generate comprehensive set of exploratory data analysis plots
    _ensure_dir(out_dir)
    # Filter outliers using IQR method for cleaner visualizations
    work = df.copy()
    if "ata_minutes" in work.columns:
        s = pd.to_numeric(work["ata_minutes"], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        work = work[s.between(lo, hi, inclusive="both")]
    work = work.dropna(how="any")
    plot_correlation(work, out_dir, title)
    plot_distributions(work, out_dir, title)
    plot_box_by_hour(work, out_dir, title)
    plot_scatter_distance_ata(work, out_dir, title)
    plot_time_series(work, out_dir, title)
    _plot_order_hour_distribution(work, out_dir, title)
    _plot_orders_vs_weekend(work, out_dir, title)
    # Include additional analysis plots
    

def _plot_orders_vs_weekend(df: pd.DataFrame, out_dir: str, title: str) -> None:
    # Compare order volumes between weekdays and weekends
    if not {"is_weekend"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    tmp = df["is_weekend"].value_counts().sort_index()
    sns.barplot(x=tmp.index.map({0: "Weekday", 1: "Weekend"}), y=tmp.values, ax=ax, palette="colorblind")
    ax.set_title(f"Order Counts: Weekend vs Weekday - {title}")
    ax.set_xlabel("")
    ax.set_ylabel("Orders")
    _save(fig, os.path.join(out_dir, "orders_weekend_vs_weekday.png"))


def _plot_order_hour_distribution(df: pd.DataFrame, out_dir: str, title: str) -> None:
    # Show distribution of orders throughout the day
    # Extract hour information from timestamp if not present
    work = df.copy()
    if "hour" not in work.columns and "accept_time" in work.columns:
        work["hour"] = pd.to_datetime(work["accept_time"], errors="coerce").dt.hour
    if "hour" not in work.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(work["hour"].dropna(), bins=24, discrete=True, ax=ax, color=sns.color_palette("colorblind")[1])
    ax.set_title(f"Order Time Distribution by Hour - {title}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Orders")
    _save(fig, os.path.join(out_dir, "order_hour_distribution.png"))



from __future__ import annotations

import argparse
from pathlib import Path
import json

from src.data.loaders import load_raw, build_features, split_xy
from src.eda.quality import basic_report, clean_data
from src.eda.plots import generate_eda_plots
from ata_eda import run_eda_suite
from src.models.classical import (
    train_eval_knn,
    train_eval_rf,
    train_eval_xgb,
    train_eval_catboost,
    train_eval_linear,
)
from src.models.transformers import train_eval_simple_transformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jl_csv", type=str, default=str(Path("delivery_jl.csv").resolve()))
    p.add_argument("--yt_csv", type=str, default=str(Path("delivery_yt.csv").resolve()))
    p.add_argument("--city_lat_jl", type=float, default=43.837)  # Jilin
    p.add_argument("--city_lon_jl", type=float, default=126.549)
    p.add_argument("--city_lat_yt", type=float, default=37.463)  # Yantai
    p.add_argument("--city_lon_yt", type=float, default=121.447)
    p.add_argument("--which", type=str, default="all", help="jl|yt|all")
    p.add_argument("--out_json", type=str, default="metrics.json", help="Path to save metrics JSON")
    p.add_argument("--out_csv", type=str, default="metrics.csv", help="Path to save flattened CSV")
    p.add_argument("--out_pivot_prefix", type=str, default="metrics_pivot_", help="Prefix for per-city pivot CSVs")
    args = p.parse_args()

    df_jl, df_yt = load_raw(args.jl_csv, args.yt_csv)
    # Cleaning first
    df_jl = clean_data(df_jl)
    df_yt = clean_data(df_yt)
    # Defer EDA until just before ML (on feature set)

    results = {}
    eda_reports = {}

    def run_city(df, lat, lon, tag):
        feats = build_features(df, city_lat=lat, city_lon=lon, datetime_col="accept_time")
        # EDA just before ML on features
        try:
            generate_eda_plots(feats, out_dir=f"eda_{tag}_features", title=f"{tag} (features)")
        except Exception:
            pass
        try:
            eda_reports[tag] = basic_report(feats)
        except Exception:
            eda_reports[tag] = {"error": "EDA report failed"}
        # Full ATA EDA suite from ata_eda.py
        try:
            run_eda_suite(feats, outdir=f"ata_eda_{tag}")
        except Exception:
            pass
        X, y = split_xy(feats, target="ata_minutes")
        if len(X) < 5:
            results[tag] = {"error": f"Not enough samples after preprocessing: {len(X)}"}
            return

        res_city = {}
        # Classical
        _, res_city["Linear"] = train_eval_linear(X, y)
        _, res_city["KNN"] = train_eval_knn(X, y)
        _, res_city["RandomForest"] = train_eval_rf(X, y)
        try:
            _, res_city["XGBoost"] = train_eval_xgb(X, y)
        except Exception as e:
            res_city["XGBoost"] = {"error": str(e)}
        try:
            _, res_city["CatBoost"] = train_eval_catboost(X, y)
        except Exception as e:
            res_city["CatBoost"] = {"error": str(e)}

        # Transformer
        _, res_city["Transformer"] = train_eval_simple_transformer(X, y)

        results[tag] = res_city

    if args.which in ("jl", "all"):
        run_city(df_jl, args.city_lat_jl, args.city_lon_jl, "Jilin")
    if args.which in ("yt", "all"):
        run_city(df_yt, args.city_lat_yt, args.city_lon_yt, "Yantai")

    # Round all float values to 3 decimals for stable reporting
    def _round_vals(obj):
        if isinstance(obj, dict):
            return {k: _round_vals(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_vals(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 3)
        return obj

    # include EDA summaries (from features) under a special key
    results_all = {"EDA": eda_reports, "metrics": results}
    rounded = _round_vals(results_all)

    # Print to console
    print(json.dumps(rounded, indent=2))

    # Persist JSON
    try:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rounded, f, indent=2)
    except Exception:
        pass

    # Persist CSV (flatten per city/model)
    try:
        import csv
        rows = []
        metrics_section = rounded.get("metrics", {}) if isinstance(rounded, dict) else {}
        for city, models in metrics_section.items():
            if isinstance(models, dict):
                for model_name, metrics in models.items():
                    row = {"city": city, "model": model_name}
                    if isinstance(metrics, dict):
                        row.update(metrics)
                    else:
                        row["value"] = str(metrics)
                    rows.append(row)
        if rows:
            fieldnames = sorted({k for r in rows for k in r.keys()})
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    except Exception:
        pass

    # Persist per-city pivot tables: rows=metrics, columns=models
    try:
        import csv
        for city, models in metrics_section.items():
            if not isinstance(models, dict):
                continue
            # Collect all metric keys across models
            model_names = [m for m in models.keys()]
            metric_keys = set()
            for m in model_names:
                md = models.get(m, {})
                if isinstance(md, dict):
                    metric_keys.update(md.keys())
            metric_keys = sorted(metric_keys)

            # Build table: first column 'metric', subsequent columns per model
            out_path = f"{args.out_pivot_prefix}{city}.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric"] + model_names)
                for mk in metric_keys:
                    row = [mk]
                    for m in model_names:
                        mv = models.get(m, {})
                        val = mv.get(mk, "") if isinstance(mv, dict) else ""
                        row.append(val)
                    writer.writerow(row)
    except Exception:
        pass


if __name__ == "__main__":
    main()



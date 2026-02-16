from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained baseline models.")
    p.add_argument("--run-dir", required=True, help="Path to artifacts run directory")
    p.add_argument(
        "--test-csv",
        default="./data/processed/test_external.csv",
        help="Path to external holdout CSV",
    )
    p.add_argument("--label-col", default="label", help="Label column name")
    p.add_argument(
        "--group-cols",
        default="study_id,tissue",
        help="Comma-separated columns for subgroup metrics",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing run manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_cols = manifest.get("feature_columns", [])
    if not feature_cols:
        raise SystemExit("Run manifest missing feature_columns.")

    test_df = pd.read_csv(args.test_csv)
    missing_cols = [c for c in feature_cols + [args.label_col] if c not in test_df.columns]
    if missing_cols:
        raise SystemExit(f"Missing required columns in test csv: {missing_cols}")

    x_test = test_df[feature_cols]
    y_test = test_df[args.label_col].astype(int).to_numpy()

    metrics = {}
    predictions = pd.DataFrame({"y_true": y_test})
    subgroup_rows = []
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    for model_name, model_path in manifest.get("models", {}).items():
        model = joblib.load(model_path)
        y_prob = model.predict_proba(x_test)[:, 1]
        predictions[f"{model_name}_prob"] = y_prob
        metrics[model_name] = compute_metrics(y_test, y_prob)
        subgroup_rows.extend(
            compute_subgroup_metrics(
                df=test_df,
                y_prob=y_prob,
                y_col=args.label_col,
                group_cols=group_cols,
                model_name=model_name,
            )
        )

    (run_dir / "test_predictions.csv").write_text(
        predictions.to_csv(index=False), encoding="utf-8"
    )
    (run_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(run_dir / "metrics_subgroups.csv", index=False)

    print("External holdout metrics")
    print(json.dumps(metrics, indent=2))
    print(f"Subgroup rows: {len(subgroup_df)}")


def compute_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n_samples": int(len(y_true)),
    }


def compute_subgroup_metrics(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    y_col: str,
    group_cols: list[str],
    model_name: str,
) -> list[dict]:
    rows = []
    tmp = df.copy()
    tmp["_y_prob"] = y_prob

    for group_col in group_cols:
        if group_col not in tmp.columns:
            continue
        for key, gdf in tmp.groupby(group_col):
            if gdf[y_col].nunique() < 2:
                continue
            m = compute_metrics(gdf[y_col].to_numpy(), gdf["_y_prob"].to_numpy())
            rows.append(
                {
                    "model": model_name,
                    "group_col": group_col,
                    "group_value": str(key),
                    **m,
                }
            )
    return rows


if __name__ == "__main__":
    main()


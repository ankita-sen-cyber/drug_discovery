from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train baseline RNA-seq models.")
    p.add_argument("--data-dir", default="./data/processed", help="Directory with train/val csv")
    p.add_argument("--output-dir", default="./artifacts", help="Artifacts root directory")
    p.add_argument("--label-col", default="label", help="Label column name")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--require-xgboost",
        action="store_true",
        help="Fail training if xgboost package is unavailable",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            f"Expected {train_path} and {val_path}. Run kdense-prepare-data first."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    feature_cols = detect_feature_columns(train_df, label_col=args.label_col)

    x_train = train_df[feature_cols]
    y_train = train_df[args.label_col].astype(int)
    x_val = val_df[feature_cols]
    y_val = val_df[args.label_col].astype(int)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    model_paths = {}

    elastic = build_elasticnet_model(seed=args.seed)
    elastic.fit(x_train, y_train)
    y_prob_elastic = elastic.predict_proba(x_val)[:, 1]
    metrics["elasticnet"] = compute_metrics(y_true=y_val, y_prob=y_prob_elastic)
    elastic_path = run_dir / "model_elasticnet.joblib"
    joblib.dump(elastic, elastic_path)
    model_paths["elasticnet"] = str(elastic_path.resolve())

    xgb_model = None
    xgb_unavailable_reason = None
    try:
        xgb_model = build_xgboost_model(seed=args.seed)
        xgb_model.fit(x_train, y_train)
        y_prob_xgb = xgb_model.predict_proba(x_val)[:, 1]
        metrics["xgboost"] = compute_metrics(y_true=y_val, y_prob=y_prob_xgb)
        xgb_path = run_dir / "model_xgboost.joblib"
        joblib.dump(xgb_model, xgb_path)
        model_paths["xgboost"] = str(xgb_path.resolve())
    except Exception as exc:
        xgb_unavailable_reason = str(exc)
        if args.require_xgboost:
            raise SystemExit(
                "XGBoost training failed and --require-xgboost was set. "
                f"Reason: {xgb_unavailable_reason}"
            )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "data_dir": str(data_dir.resolve()),
        "label_col": args.label_col,
        "seed": args.seed,
        "feature_columns": feature_cols,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "models": model_paths,
        "metrics_val": metrics,
        "xgboost_unavailable_reason": xgb_unavailable_reason,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    predictions = pd.DataFrame({"y_true": y_val.to_numpy(), "elasticnet_prob": y_prob_elastic})
    if xgb_model is not None:
        predictions["xgboost_prob"] = xgb_model.predict_proba(x_val)[:, 1]
    predictions.to_csv(run_dir / "val_predictions.csv", index=False)

    (run_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved baseline artifacts to {run_dir}")
    print(json.dumps(metrics, indent=2))


def detect_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    excluded = {"sample_id", "study_id", "tissue", "split", label_col}
    feature_cols = [
        c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise SystemExit("No numeric feature columns detected for training.")
    return feature_cols


def build_elasticnet_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    C=1.0,
                    solver="saga",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_xgboost_model(seed: int):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError("xgboost not installed. Install xgboost to enable this baseline.") from exc
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
    )


def compute_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


if __name__ == "__main__":
    main()


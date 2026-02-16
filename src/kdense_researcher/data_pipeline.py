from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESERVED_COLUMNS = {
    "sample_id",
    "study_id",
    "tissue",
    "label",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare grouped train/val/external splits from a harmonized RNA-seq table."
    )
    p.add_argument("--input-csv", required=True, help="Path to input harmonized table")
    p.add_argument(
        "--output-dir",
        default="./data/processed",
        help="Directory for train/val/test_external CSV files",
    )
    p.add_argument("--label-col", default="label", help="Binary label column")
    p.add_argument(
        "--group-col",
        default="study_id",
        help="Grouping column for leakage-safe split",
    )
    p.add_argument("--test-frac", type=float, default=0.2, help="External holdout group fraction")
    p.add_argument("--val-frac", type=float, default=0.2, help="Validation group fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    validate_input(df, args.label_col, args.group_col)

    split_df, split_info = split_by_group(
        df=df,
        group_col=args.group_col,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test_external"].copy()

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test_external.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    feature_cols = detect_feature_columns(train_df, label_col=args.label_col)
    manifest = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "label_col": args.label_col,
        "group_col": args.group_col,
        "seed": args.seed,
        "test_frac": args.test_frac,
        "val_frac": args.val_frac,
        "n_rows": int(len(df)),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "splits": split_info,
        "outputs": {
            "train": str(train_path.resolve()),
            "val": str(val_path.resolve()),
            "test_external": str(test_path.resolve()),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote splits to {output_dir} | "
        f"train={len(train_df)} val={len(val_df)} test_external={len(test_df)}"
    )


def validate_input(df: pd.DataFrame, label_col: str, group_col: str) -> None:
    missing = [c for c in (label_col, group_col) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    if df[label_col].isna().any():
        raise SystemExit(f"Label column '{label_col}' has missing values.")
    unique_labels = sorted(pd.unique(df[label_col]))
    if len(unique_labels) != 2:
        raise SystemExit(
            f"Label column '{label_col}' must be binary. Found labels: {unique_labels}"
        )
    if df[group_col].isna().any():
        raise SystemExit(f"Group column '{group_col}' has missing values.")


def split_by_group(
    df: pd.DataFrame,
    group_col: str,
    test_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    groups = pd.unique(df[group_col]).tolist()
    if len(groups) < 5:
        raise SystemExit(
            f"Need at least 5 unique groups in '{group_col}' for grouped splitting. "
            f"Found {len(groups)}."
        )

    rng = np.random.default_rng(seed)
    groups = list(groups)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_test = max(1, int(round(n_groups * test_frac)))
    n_remaining = n_groups - n_test
    n_val = max(1, int(round(n_remaining * val_frac)))
    n_train = n_groups - n_test - n_val
    if n_train < 1:
        raise SystemExit("Split fractions leave no training groups. Adjust val/test fractions.")

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    train_groups = set(groups[n_test + n_val :])

    split_df = df.copy()
    split_df["split"] = "train"
    split_df.loc[split_df[group_col].isin(val_groups), "split"] = "val"
    split_df.loc[split_df[group_col].isin(test_groups), "split"] = "test_external"

    info = {
        "n_groups_total": n_groups,
        "n_groups_train": len(train_groups),
        "n_groups_val": len(val_groups),
        "n_groups_test_external": len(test_groups),
        "groups_train": sorted(map(str, train_groups)),
        "groups_val": sorted(map(str, val_groups)),
        "groups_test_external": sorted(map(str, test_groups)),
    }
    return split_df, info


def detect_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    excluded = RESERVED_COLUMNS | {label_col, "split"}
    feature_cols = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise SystemExit("No numeric feature columns detected for training.")
    return feature_cols


if __name__ == "__main__":
    main()


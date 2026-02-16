from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Harmonize drug response and RNA expression tables into one training table "
            "with binary labels."
        )
    )
    p.add_argument("--response-csv", required=True, help="Drug response table (e.g., GDSC/CTRP)")
    p.add_argument("--expression-csv", required=True, help="Expression table")
    p.add_argument(
        "--output-csv",
        default="./data/raw/harmonized_rnaseq.csv",
        help="Output harmonized CSV path",
    )
    p.add_argument(
        "--summary-json",
        default="./data/raw/harmonized_summary.json",
        help="Output summary JSON path",
    )

    p.add_argument("--drug-name", required=True, help="Drug name filter value (e.g., cisplatin)")
    p.add_argument("--drug-col", default="drug_name", help="Drug name column in response table")
    p.add_argument(
        "--response-id-col",
        default="sample_id",
        help="Sample/cell line identifier column in response table",
    )
    p.add_argument(
        "--response-value-col",
        default="ln_ic50",
        help="Numeric response value column (e.g., ln_ic50 or auc)",
    )
    p.add_argument(
        "--response-direction",
        choices=["lower_better", "higher_better"],
        default="lower_better",
        help="Whether lower or higher response values indicate sensitivity",
    )

    p.add_argument(
        "--expression-orientation",
        choices=["samples_by_genes", "genes_by_samples"],
        default="samples_by_genes",
        help="Layout of expression table",
    )
    p.add_argument(
        "--expression-id-col",
        default="sample_id",
        help=(
            "Sample identifier column in expression table if samples_by_genes; "
            "ignored for genes_by_samples"
        ),
    )
    p.add_argument(
        "--gene-col",
        default="gene",
        help="Gene name column if genes_by_samples orientation",
    )

    p.add_argument(
        "--id-map-csv",
        default="",
        help="Optional mapping table for response IDs to expression IDs",
    )
    p.add_argument(
        "--id-map-response-col",
        default="response_id",
        help="Response ID column in id-map table",
    )
    p.add_argument(
        "--id-map-expression-col",
        default="expression_id",
        help="Expression ID column in id-map table",
    )

    p.add_argument(
        "--metadata-csv",
        default="",
        help="Optional metadata table containing sample_id and tissue columns",
    )
    p.add_argument("--metadata-id-col", default="sample_id", help="Metadata sample ID column")
    p.add_argument("--tissue-col", default="tissue", help="Metadata tissue column")

    p.add_argument(
        "--lower-quantile",
        type=float,
        default=0.33,
        help="Lower quantile for confident class boundary",
    )
    p.add_argument(
        "--upper-quantile",
        type=float,
        default=0.67,
        help="Upper quantile for confident class boundary",
    )
    p.add_argument(
        "--study-id",
        default="GDSC",
        help="Study ID assigned to all rows (can be refined later)",
    )
    p.add_argument(
        "--min-expression-features",
        type=int,
        default=100,
        help="Minimum number of numeric expression features required",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    validate_quantiles(args.lower_quantile, args.upper_quantile)

    response_df = pd.read_csv(args.response_csv)
    expr_df = pd.read_csv(args.expression_csv)

    response_df = standardize_response(
        response_df=response_df,
        drug_name=args.drug_name,
        drug_col=args.drug_col,
        id_col=args.response_id_col,
        value_col=args.response_value_col,
    )
    expr_df = standardize_expression(
        expr_df=expr_df,
        orientation=args.expression_orientation,
        expression_id_col=args.expression_id_col,
        gene_col=args.gene_col,
    )

    if args.id_map_csv:
        id_map_df = pd.read_csv(args.id_map_csv)
        id_map_df = id_map_df.rename(
            columns={
                args.id_map_response_col: "response_id",
                args.id_map_expression_col: "expression_id",
            }
        )
        require_columns(id_map_df, ["response_id", "expression_id"], "id-map table")
        id_map_df["response_id"] = normalize_ids(id_map_df["response_id"])
        id_map_df["expression_id"] = normalize_ids(id_map_df["expression_id"])
        response_df = response_df.merge(id_map_df, on="response_id", how="left")
    else:
        response_df["expression_id"] = response_df["response_id"]

    merged = response_df.merge(expr_df, on="expression_id", how="inner")
    if merged.empty:
        raise SystemExit("No matched rows after joining response and expression tables.")

    merged = assign_labels(
        merged=merged,
        value_col="response_value",
        lower_q=args.lower_quantile,
        upper_q=args.upper_quantile,
        direction=args.response_direction,
    )

    if args.metadata_csv:
        meta_df = pd.read_csv(args.metadata_csv)
        meta_df = meta_df.rename(
            columns={
                args.metadata_id_col: "expression_id",
                args.tissue_col: "tissue",
            }
        )
        require_columns(meta_df, ["expression_id"], "metadata table")
        meta_df["expression_id"] = normalize_ids(meta_df["expression_id"])
        merged = merged.merge(meta_df[["expression_id", "tissue"]], on="expression_id", how="left")
    else:
        merged["tissue"] = "unknown"

    feature_cols = detect_expression_feature_cols(merged)
    if len(feature_cols) < args.min_expression_features:
        raise SystemExit(
            f"Detected only {len(feature_cols)} numeric expression features; "
            f"minimum required is {args.min_expression_features}."
        )

    out_df = merged[["expression_id", "tissue", "label", *feature_cols]].copy()
    out_df = out_df.rename(columns={"expression_id": "sample_id"})
    out_df.insert(1, "study_id", args.study_id)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    summary = build_summary(
        input_response=args.response_csv,
        input_expression=args.expression_csv,
        output_csv=str(output_csv),
        merged=merged,
        out_df=out_df,
        feature_cols=feature_cols,
        lower_q=args.lower_quantile,
        upper_q=args.upper_quantile,
        response_direction=args.response_direction,
        study_id=args.study_id,
    )
    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Harmonization complete | "
        f"rows={len(out_df)} features={len(feature_cols)} "
        f"labels={out_df['label'].value_counts().to_dict()} "
        f"output={output_csv}"
    )


def standardize_response(
    response_df: pd.DataFrame,
    drug_name: str,
    drug_col: str,
    id_col: str,
    value_col: str,
) -> pd.DataFrame:
    require_columns(response_df, [drug_col, id_col, value_col], "response table")
    out = response_df.copy()
    out = out[out[drug_col].astype(str).str.lower() == drug_name.lower()].copy()
    if out.empty:
        raise SystemExit(
            f"No response rows found for drug '{drug_name}' in column '{drug_col}'."
        )
    out = out.rename(columns={id_col: "response_id", value_col: "response_value"})
    out["response_id"] = normalize_ids(out["response_id"])
    out["response_value"] = pd.to_numeric(out["response_value"], errors="coerce")
    out = out.dropna(subset=["response_id", "response_value"]).copy()
    if out.empty:
        raise SystemExit("No valid response rows after cleaning numeric values.")
    return out[["response_id", "response_value"]]


def standardize_expression(
    expr_df: pd.DataFrame,
    orientation: str,
    expression_id_col: str,
    gene_col: str,
) -> pd.DataFrame:
    if orientation == "samples_by_genes":
        require_columns(expr_df, [expression_id_col], "expression table")
        out = expr_df.copy().rename(columns={expression_id_col: "expression_id"})
        out["expression_id"] = normalize_ids(out["expression_id"])
        out = out.dropna(subset=["expression_id"]).copy()
        return out

    require_columns(expr_df, [gene_col], "expression table (genes_by_samples)")
    tmp = expr_df.copy().rename(columns={gene_col: "gene"})
    tmp["gene"] = tmp["gene"].astype(str)
    transposed = tmp.set_index("gene").T.reset_index().rename(columns={"index": "expression_id"})
    transposed["expression_id"] = normalize_ids(transposed["expression_id"])
    return transposed


def assign_labels(
    merged: pd.DataFrame,
    value_col: str,
    lower_q: float,
    upper_q: float,
    direction: str,
) -> pd.DataFrame:
    out = merged.copy()
    q_low = float(out[value_col].quantile(lower_q))
    q_high = float(out[value_col].quantile(upper_q))
    if q_low >= q_high:
        raise SystemExit(
            f"Invalid quantile boundaries: q_low={q_low} >= q_high={q_high}. "
            "Adjust lower/upper quantiles."
        )

    out["label"] = pd.NA
    if direction == "lower_better":
        out.loc[out[value_col] <= q_low, "label"] = 1
        out.loc[out[value_col] >= q_high, "label"] = 0
    else:
        out.loc[out[value_col] >= q_high, "label"] = 1
        out.loc[out[value_col] <= q_low, "label"] = 0

    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    if out["label"].nunique() != 2:
        raise SystemExit(
            f"Labeling produced non-binary class set: {sorted(out['label'].unique().tolist())}"
        )
    return out


def detect_expression_feature_cols(df: pd.DataFrame) -> list[str]:
    reserved = {"response_id", "expression_id", "response_value", "label", "tissue", "study_id"}
    features = [
        c
        for c in df.columns
        if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
    ]
    return features


def build_summary(
    input_response: str,
    input_expression: str,
    output_csv: str,
    merged: pd.DataFrame,
    out_df: pd.DataFrame,
    feature_cols: list[str],
    lower_q: float,
    upper_q: float,
    response_direction: str,
    study_id: str,
) -> dict[str, Any]:
    return {
        "input_response": str(Path(input_response).resolve()),
        "input_expression": str(Path(input_expression).resolve()),
        "output_csv": str(Path(output_csv).resolve()),
        "study_id": study_id,
        "rows_merged_before_labeling": int(len(merged)),
        "rows_after_labeling": int(len(out_df)),
        "label_distribution": out_df["label"].value_counts().to_dict(),
        "n_features": len(feature_cols),
        "lower_quantile": lower_q,
        "upper_quantile": upper_q,
        "response_direction": response_direction,
    }


def require_columns(df: pd.DataFrame, cols: list[str], table_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {table_name}: {missing}")


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().replace({"": pd.NA})


def validate_quantiles(lower_q: float, upper_q: float) -> None:
    if not (0.0 < lower_q < upper_q < 1.0):
        raise SystemExit("Quantiles must satisfy 0 < lower_quantile < upper_quantile < 1.")


if __name__ == "__main__":
    main()


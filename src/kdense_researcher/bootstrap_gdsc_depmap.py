from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Bootstrap local GDSC+DepMap dataset files and emit response/expression/"
            "harmonized tables for KDense pipeline."
        )
    )
    p.add_argument("--out-dir", default="./data/raw", help="Output data directory")

    # Local file inputs (preferred)
    p.add_argument("--gdsc-csv", default="", help="Path to GDSC response CSV/XLSX converted to CSV")
    p.add_argument("--depmap-expression-csv", default="", help="Path to DepMap expression CSV")
    p.add_argument("--depmap-model-csv", default="", help="Path to DepMap model metadata CSV")

    # Optional download URLs (use only if direct links available)
    p.add_argument("--gdsc-url", default="", help="Direct URL to GDSC response CSV")
    p.add_argument("--depmap-expression-url", default="", help="Direct URL to DepMap expression CSV")
    p.add_argument("--depmap-model-url", default="", help="Direct URL to DepMap model metadata CSV")

    p.add_argument("--drug-name", default="cisplatin", help="Drug name filter")
    p.add_argument("--lineage", default="Lung", help="Optional lineage filter from DepMap metadata")
    p.add_argument(
        "--response-value-col",
        default="ln_ic50",
        help="Output response metric column name (generated from detected source metric)",
    )
    p.add_argument("--max-genes", type=int, default=5000, help="Max expression genes to keep")
    p.add_argument("--min-matched-samples", type=int, default=20, help="Min matched samples required")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdsc_path = resolve_input_path(args.gdsc_csv, args.gdsc_url, out_dir / "gdsc_response.csv")
    depmap_expr_path = resolve_input_path(
        args.depmap_expression_csv, args.depmap_expression_url, out_dir / "depmap_expression.csv"
    )
    depmap_model_path = resolve_input_path(
        args.depmap_model_csv, args.depmap_model_url, out_dir / "depmap_model.csv"
    )

    gdsc = pd.read_csv(gdsc_path)
    depmap_expr = pd.read_csv(depmap_expr_path)
    depmap_model = pd.read_csv(depmap_model_path)

    response = build_response_table(
        gdsc=gdsc,
        depmap_model=depmap_model,
        drug_name=args.drug_name,
        response_value_col=args.response_value_col,
        lineage=args.lineage,
    )
    expression = build_expression_table(depmap_expr=depmap_expr, max_genes=args.max_genes)

    harmonized = response.merge(
        expression,
        left_on="sample_id",
        right_on="sample_id",
        how="inner",
    )
    if len(harmonized) < args.min_matched_samples:
        raise SystemExit(
            f"Only {len(harmonized)} matched samples after merge. "
            f"Need at least {args.min_matched_samples}."
        )

    response_out = out_dir / "response.csv"
    expression_out = out_dir / "expression.csv"
    harmonized_out = out_dir / "harmonized_rnaseq.csv"
    summary_out = out_dir / "bootstrap_summary.json"

    response.to_csv(response_out, index=False)
    expression.to_csv(expression_out, index=False)
    harmonized.to_csv(harmonized_out, index=False)

    summary = {
        "drug_name": args.drug_name,
        "lineage_filter": args.lineage,
        "inputs": {
            "gdsc": str(gdsc_path.resolve()),
            "depmap_expression": str(depmap_expr_path.resolve()),
            "depmap_model": str(depmap_model_path.resolve()),
        },
        "outputs": {
            "response_csv": str(response_out.resolve()),
            "expression_csv": str(expression_out.resolve()),
            "harmonized_csv": str(harmonized_out.resolve()),
        },
        "counts": {
            "response_rows": int(len(response)),
            "expression_rows": int(len(expression)),
            "harmonized_rows": int(len(harmonized)),
            "expression_features": int(len([c for c in harmonized.columns if c not in {"sample_id", "study_id", "tissue", "drug_name", args.response_value_col}])),
        },
        "label_ready_note": (
            "Run kdense-harmonize on response.csv + expression.csv to create binary labels."
        ),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Bootstrapped data written to {out_dir}")
    print(json.dumps(summary["counts"], indent=2))


def resolve_input_path(local_path: str, url: str, download_target: Path) -> Path:
    if local_path:
        p = Path(local_path)
        if not p.exists():
            raise SystemExit(f"Local input file not found: {p}")
        return p
    if url:
        download_target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {download_target}")
        urllib.request.urlretrieve(url, download_target)  # noqa: S310
        return download_target
    raise SystemExit(
        f"Need one of local path or URL for target: {download_target.name}"
    )


def build_response_table(
    gdsc: pd.DataFrame,
    depmap_model: pd.DataFrame,
    drug_name: str,
    response_value_col: str,
    lineage: str,
) -> pd.DataFrame:
    gdsc_cols = {c.lower(): c for c in gdsc.columns}
    depmap_cols = {c.lower(): c for c in depmap_model.columns}

    drug_col = find_col(gdsc_cols, ["drug_name", "drug", "drug name"])
    cosmic_col = find_col(gdsc_cols, ["cosmic_id", "cosmic id", "cosmicid"])
    cell_line_col = find_col(gdsc_cols, ["cell_line_name", "cell line name", "cell line"])
    metric_col = find_col(gdsc_cols, ["ln_ic50", "ic50", "auc", "log_ic50"])

    model_depmap_col = find_col(depmap_cols, ["depmap_id", "depmap id", "modelid", "model_id"])
    model_cosmic_col = optional_col(depmap_cols, ["cosmicid", "cosmic_id", "cosmic id"])
    model_name_col = optional_col(
        depmap_cols,
        [
            "stripped_cell_line_name",
            "strippedcelllinename",
            "cell_line_name",
            "celllinename",
            "ccle_name",
            "cclename",
        ],
    )
    model_lineage_col = optional_col(depmap_cols, ["oncotreelineage", "lineage"])

    g = gdsc.copy()
    g = g[g[drug_col].astype(str).str.lower() == drug_name.lower()].copy()
    g["cosmic_id_norm"] = normalize_numericish_id(g[cosmic_col])
    g["cell_line_name_norm"] = normalize_name(g[cell_line_col])
    g["response_value"] = pd.to_numeric(g[metric_col], errors="coerce")
    g = g.dropna(subset=["response_value"]).copy()

    m = depmap_model.copy()
    m["depmap_id"] = m[model_depmap_col].astype(str).str.strip()
    if model_cosmic_col:
        m["cosmic_id_norm"] = normalize_numericish_id(m[model_cosmic_col])
    else:
        m["cosmic_id_norm"] = pd.NA
    if model_name_col:
        m["cell_line_name_norm"] = normalize_name(m[model_name_col])
    else:
        m["cell_line_name_norm"] = pd.NA
    if model_lineage_col and lineage:
        m = m[m[model_lineage_col].astype(str).str.lower() == lineage.lower()].copy()

    # First attempt join by COSMIC ID, then fallback by normalized cell-line name.
    by_cosmic = g.merge(m[["depmap_id", "cosmic_id_norm"]], on="cosmic_id_norm", how="left")
    unresolved = by_cosmic["depmap_id"].isna()
    if unresolved.any():
        name_map = (
            m[["depmap_id", "cell_line_name_norm"]]
            .dropna(subset=["cell_line_name_norm"])
            .drop_duplicates(subset=["cell_line_name_norm"])
        )
        # Use the merged frame for boolean indexing to keep index alignment stable.
        unresolved_rows = by_cosmic.loc[unresolved, ["cell_line_name_norm"]].copy()
        by_name = unresolved_rows.merge(name_map, on="cell_line_name_norm", how="left")
        by_cosmic.loc[unresolved, "depmap_id"] = by_name["depmap_id"].to_numpy()

    out = by_cosmic.dropna(subset=["depmap_id"]).copy()
    if out.empty:
        raise SystemExit("No response rows mapped to DepMap IDs.")

    out = (
        out.groupby("depmap_id", as_index=False)["response_value"]
        .median()
        .rename(columns={"depmap_id": "sample_id"})
    )
    out["study_id"] = "GDSC_DEPMap"
    out["tissue"] = lineage.lower() if lineage else "unknown"
    out["drug_name"] = drug_name
    out = out.rename(columns={"response_value": response_value_col})
    out = out[["sample_id", "study_id", "tissue", "drug_name", response_value_col]]
    return out


def build_expression_table(depmap_expr: pd.DataFrame, max_genes: int) -> pd.DataFrame:
    id_col = None
    for cand in ["DepMap_ID", "depmap_id", "ModelID", "modelid", "Unnamed: 0"]:
        if cand in depmap_expr.columns:
            id_col = cand
            break
    if id_col is None:
        raise SystemExit(
            "DepMap expression table must contain one of: "
            "DepMap_ID, depmap_id, ModelID, modelid, Unnamed: 0"
        )
    out = depmap_expr.copy().rename(columns={id_col: "sample_id"})
    out["sample_id"] = out["sample_id"].astype(str).str.strip()

    numeric_cols = [c for c in out.columns if c != "sample_id" and pd.api.types.is_numeric_dtype(out[c])]
    if not numeric_cols:
        raise SystemExit("No numeric expression features found in DepMap expression table.")

    # Drop versioned gene IDs from column names to clean symbols like "EGFR (ENSG...)"
    rename_map = {}
    for c in numeric_cols:
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*\(", c)
        rename_map[c] = m.group(1) if m else c
    out = out.rename(columns=rename_map)

    # Keep top genes by variance for manageable baseline size.
    numeric_cols = [c for c in out.columns if c != "sample_id" and pd.api.types.is_numeric_dtype(out[c])]
    variances = out[numeric_cols].var(axis=0, skipna=True)
    top_cols = variances.sort_values(ascending=False).head(max_genes).index.tolist()
    out = out[["sample_id", *top_cols]].copy()
    out = out.drop_duplicates(subset=["sample_id"])
    return out


def find_col(cols_map: dict[str, str], candidates: list[str]) -> str:
    for cand in candidates:
        if cand in cols_map:
            return cols_map[cand]
    raise SystemExit(f"Could not detect required column from candidates: {candidates}")


def optional_col(cols_map: dict[str, str], candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in cols_map:
            return cols_map[cand]
    return None


def normalize_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", "", regex=True)
        .str.strip()
        .replace({"": pd.NA})
    )


def normalize_numericish_id(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .astype("Int64")
        .astype(str)
        .replace({"<NA>": pd.NA})
    ).reindex(series.index)


if __name__ == "__main__":
    main()

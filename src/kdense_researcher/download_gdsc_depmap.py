from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from .bootstrap_gdsc_depmap import main as bootstrap_main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download GDSC + DepMap files into data/raw and optionally bootstrap."
    )
    p.add_argument("--out-dir", default="./data/raw", help="Output directory")
    p.add_argument(
        "--gdsc-index-url",
        default="https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/",
        help="GDSC release index URL",
    )
    p.add_argument(
        "--depmap-article-id",
        type=int,
        default=27993248,
        help="DepMap Figshare article ID (e.g., 27993248 for 24Q4)",
    )
    p.add_argument(
        "--gdsc-csv-name",
        default="",
        help=(
            "Optional explicit GDSC csv filename from index "
            "(e.g., GDSC2_fitted_dose_response_24Jul22.csv)"
        ),
    )
    p.add_argument(
        "--run-bootstrap",
        action="store_true",
        help="Run kdense-bootstrap-gdsc immediately after download",
    )
    p.add_argument("--drug-name", default="cisplatin", help="Drug for bootstrap stage")
    p.add_argument("--lineage", default="Lung", help="Lineage for bootstrap stage")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdsc_name = args.gdsc_csv_name or discover_gdsc_csv_name(args.gdsc_index_url)
    gdsc_url = urllib.parse.urljoin(args.gdsc_index_url, gdsc_name)
    gdsc_path = out_dir / "gdsc_response.csv"
    download_file(gdsc_url, gdsc_path)

    files = discover_depmap_files(args.depmap_article_id)
    expr_url = files["OmicsExpressionProteinCodingGenesTPMLogp1.csv"]["download_url"]
    model_url = files["Model.csv"]["download_url"]

    expr_path = out_dir / "depmap_expression.csv"
    model_path = out_dir / "depmap_model.csv"
    download_file(expr_url, expr_path)
    download_file(model_url, model_path)

    summary = {
        "gdsc_index_url": args.gdsc_index_url,
        "gdsc_selected_file": gdsc_name,
        "gdsc_url": gdsc_url,
        "depmap_article_id": args.depmap_article_id,
        "depmap_expression_name": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "depmap_model_name": "Model.csv",
        "outputs": {
            "gdsc_response_csv": str(gdsc_path.resolve()),
            "depmap_expression_csv": str(expr_path.resolve()),
            "depmap_model_csv": str(model_path.resolve()),
        },
    }
    (out_dir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Download complete.")
    print(json.dumps(summary, indent=2))

    if args.run_bootstrap:
        # Reuse existing CLI by emulating argv.
        import sys

        prev_argv = sys.argv[:]
        try:
            sys.argv = [
                "kdense-bootstrap-gdsc",
                "--gdsc-csv",
                str(gdsc_path),
                "--depmap-expression-csv",
                str(expr_path),
                "--depmap-model-csv",
                str(model_path),
                "--drug-name",
                args.drug_name,
                "--lineage",
                args.lineage,
                "--out-dir",
                str(out_dir),
            ]
            bootstrap_main()
        finally:
            sys.argv = prev_argv


def discover_gdsc_csv_name(index_url: str) -> str:
    html = fetch_text(index_url)
    candidates = re.findall(r'href="([^"]+\.csv)"', html, flags=re.IGNORECASE)
    if not candidates:
        raise SystemExit(f"No CSV files detected at GDSC index URL: {index_url}")

    # Prefer fitted GDSC2 file, then fitted GDSC1.
    ranked = sorted(
        candidates,
        key=lambda n: (
            0 if "GDSC2_fitted_dose_response" in n else
            1 if "GDSC1_fitted_dose_response" in n else
            2,
            n,
        ),
    )
    chosen = ranked[0]
    print(f"Selected GDSC CSV: {chosen}")
    return chosen


def discover_depmap_files(article_id: int) -> dict:
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    raw = fetch_text(api_url)
    payload = json.loads(raw)
    files = payload.get("files", [])
    by_name = {f.get("name"): f for f in files}
    required = ["OmicsExpressionProteinCodingGenesTPMLogp1.csv", "Model.csv"]
    missing = [n for n in required if n not in by_name]
    if missing:
        available = sorted([k for k in by_name.keys() if isinstance(k, str)])
        raise SystemExit(
            f"Missing required DepMap files in article {article_id}: {missing}. "
            f"Available files: {available[:50]}"
        )
    return by_name


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read().decode("utf-8")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310


if __name__ == "__main__":
    main()


# KDense Drug Discovery (Reproducible Runbook)

This repository currently supports:
- Data download/bootstrap for GDSC + DepMap
- Harmonization to a binary drug-response training table
- Train/val/test split
- Baseline model training and evaluation
- LLM+RAG API for frontend hypothesis output

Use the steps below exactly to reproduce.

## 1) Environment

```bash
cd "/Users/ankitasen/Desktop/Kyushu University/drug_discovery/drug_discovery"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2) Download raw datasets (GDSC + DepMap)

```bash
kdense-download-gdsc-depmap \
  --out-dir ./data/raw \
  --drug-name cisplatin \
  --lineage Lung
```

Expected files after download:
- `./data/raw/gdsc_response.csv`
- `./data/raw/depmap_expression.csv`
- `./data/raw/depmap_model.csv`
- `./data/raw/download_summary.json`

## 3) Bootstrap aligned response/expression tables

```bash
kdense-bootstrap-gdsc \
  --gdsc-csv ./data/raw/gdsc_response.csv \
  --depmap-expression-csv ./data/raw/depmap_expression.csv \
  --depmap-model-csv ./data/raw/depmap_model.csv \
  --drug-name cisplatin \
  --lineage Lung \
  --out-dir ./data/raw
```

Expected files:
- `./data/raw/response.csv`
- `./data/raw/expression.csv`
- `./data/raw/harmonized_rnaseq.csv`
- `./data/raw/bootstrap_summary.json`

## 4) Add binary labels (quantile binarization)

```bash
kdense-harmonize \
  --response-csv ./data/raw/response.csv \
  --expression-csv ./data/raw/expression.csv \
  --drug-name cisplatin \
  --drug-col drug_name \
  --response-id-col sample_id \
  --response-value-col ln_ic50 \
  --response-direction lower_better \
  --expression-orientation samples_by_genes \
  --expression-id-col sample_id \
  --study-id GDSC_DEPMap \
  --output-csv ./data/raw/harmonized_rnaseq.csv \
  --summary-json ./data/raw/harmonized_summary.json
```

## 5) Split into train/val/test_external

Current data has one `study_id` group, so use `sample_id` grouping for now:

```bash
kdense-prepare-data \
  --input-csv ./data/raw/harmonized_rnaseq.csv \
  --output-dir ./data/processed \
  --label-col label \
  --group-col sample_id \
  --seed 42
```

Expected files:
- `./data/processed/train.csv`
- `./data/processed/val.csv`
- `./data/processed/test_external.csv`
- `./data/processed/manifest.json`

## 6) Train baseline models

```bash
kdense-train-baseline \
  --data-dir ./data/processed \
  --output-dir ./artifacts \
  --label-col label \
  --seed 42
```

Expected run dir:
- `./artifacts/run_<timestamp>/`

Inside run dir:
- `manifest.json`
- `metrics_val.json`
- `val_predictions.csv`
- `model_elasticnet.joblib`
- optional: `model_xgboost.joblib` (if xgboost installed)

## 7) Evaluate external holdout

```bash
kdense-evaluate \
  --run-dir ./artifacts/<run_id> \
  --test-csv ./data/processed/test_external.csv \
  --label-col label \
  --group-cols study_id,tissue
```

Expected outputs in run dir:
- `metrics_test.json`
- `metrics_subgroups.csv`
- `test_predictions.csv`

## 8) Run backend API (LLM + RAG, no mock results)

Start Ollama first in another terminal:

```bash
ollama serve
```

Then backend:

```bash
source .venv/bin/activate
export KDENSE_LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3.1:8b
export KDENSE_DOCS_DIR=./data/literature
kdense-api --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

If LLM output is invalid or Ollama is unreachable, `/api/analyze` returns HTTP `502`.

## 9) Run frontend (`molecu-map`)

In sibling repo:

```bash
cd "/Users/ankitasen/Desktop/Kyushu University/drug_discovery/molecu-map"
npm install
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Frontend behavior:
- Calls `POST /api/analyze`
- Does not generate mock pathway/target results
- Shows explicit backend errors

## Frontend query examples

Use these in the `molecu-map` input box:

1. `NSCLC RNA-seq shows EGFR overexpression, KRAS G12D mutation, PI3K/AKT/mTOR activation, PTEN loss, and elevated IL6-STAT3 signaling after cisplatin treatment.`
2. `Cisplatin-resistant lung adenocarcinoma samples show TP53 mutation, NF-kB activation, high BCL2, and reduced apoptosis markers.`
3. `Lung tumor cohort with HER2 amplification, CDK4/6 hyperactivity, RB1 loss, and MAPK pathway activation linked to poor platinum response.`
4. `BRCA1-deficient tumors with homologous recombination defects and DNA damage response dependence suggest PARP combination strategies.`
5. `NSCLC organoids show EMT signatures, inflammatory JAK-STAT3 signaling, and persistent survival pathway activation under cisplatin pressure.`

## Optional: PubMed literature ingestion for RAG

```bash
kdense-ingest-pubmed \
  --query "cisplatin resistance NSCLC RNA-seq" \
  --max-results 25 \
  --out-dir ./data/literature
```

If TLS cert issues occur:

```bash
kdense-ingest-pubmed --query "..." --ca-bundle /path/to/ca.pem
```

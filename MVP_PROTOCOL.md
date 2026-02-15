# MVP Protocol: Anticancer Drug Response (Cisplatin)

## Objective

Build a closed-loop AI researcher that:

1. Predicts cisplatin response from pretreatment RNA-seq.
2. Infers mechanisms (genes/pathways/upstream regulators).
3. Identifies failure modes and remediates.
4. Outputs a shortlist of wet-lab testable interventions.

## Scope

- Phenotype: cisplatin `sensitive` vs `resistant`
- Data: 3-5 public datasets
- Models: Elastic Net + XGBoost (+ optional MLP)
- One external holdout dataset for final validation

## Data inclusion rules

- Pretreatment samples only
- RNA-seq expression matrices with usable metadata
- Keep source/study/tissue metadata for split and failure analysis
- Exclude samples with missing drug response labels

## Splitting strategy

- Grouped split by source/study/tissue
- External holdout locked before model selection
- No leakage of duplicate or closely related profiles across splits

## Metrics

- Primary: AUPRC
- Secondary: AUROC, balanced accuracy, Brier score
- Required: subgroup metrics by tissue/source

## Mechanism extraction

- Feature importance: ENet coefficients + SHAP
- Enrichment: Hallmark/Reactome/KEGG
- Upstream regulators: TF activity inference
- Consensus rule: promote only signals supported by >=2 methods

## Failure analysis

- Error slices by tissue/source/platform
- Domain shift checks
- Label noise around threshold bands
- Data quality flags and outlier cluster checks

## Closed-loop remediation

Apply top 2-3 remediation actions:

- Label threshold refinement
- Feature stability filtering or pathway-level features
- Source/tissue reweighting
- Retrain and compare to locked baseline

## Outputs

- `baseline_report.md`
- `failure_matrix.csv`
- `mechanism_consensus.csv`
- `wetlab_shortlist.csv` (10-20 candidates)

## Wet-lab shortlist fields

- `target`
- `type` (`gene`, `pathway`, `regulator`, `drug_combo`)
- `predicted_effect` (`sensitize`, `reverse_resistance`)
- `evidence_score` (0-1)
- `model_support`
- `feasibility_note`
- `safety_flag`


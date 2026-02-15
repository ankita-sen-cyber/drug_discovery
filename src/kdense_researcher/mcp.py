from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MCPContextItem:
    name: str
    content: str
    source: str


@dataclass(frozen=True)
class MCPToolSpec:
    tool_id: str
    category: str
    purpose: str
    server_hint: str
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCPToolCallPlan:
    tool_id: str
    reason: str
    arguments: dict[str, str]


class MCPProvider(Protocol):
    def list_tools(self) -> list[MCPToolSpec]:
        ...

    def fetch_context(self, question: str, top_k: int = 3) -> list[MCPContextItem]:
        ...


class NullMCPProvider:
    """Default provider with a static tool catalog and no live MCP calls."""

    def __init__(self, profile: str = "chembl"):
        self.profile = profile

    def list_tools(self) -> list[MCPToolSpec]:
        if self.profile == "chembl":
            return build_chembl_tool_catalog()
        return build_drug_discovery_tool_catalog()

    def fetch_context(self, question: str, top_k: int = 3) -> list[MCPContextItem]:
        return []


def build_drug_discovery_tool_catalog() -> list[MCPToolSpec]:
    # ChEMBL entries mirror tool names documented in:
    # https://github.com/Augmented-Nature/ChEMBL-MCP-Server
    # Additional non-ChEMBL tools are retained for full workflow coverage.
    return [
        MCPToolSpec(
            tool_id="chembl.search_compounds",
            category="compound",
            purpose="Find compounds by name, synonym, or scaffold term.",
            server_hint="chembl",
            required_args=("query",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.get_compound_info",
            category="compound",
            purpose="Retrieve canonical ChEMBL compound details and properties.",
            server_hint="chembl",
            required_args=("chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_similar_compounds",
            category="compound",
            purpose="Find structurally similar compounds for analog analysis.",
            server_hint="chembl",
            required_args=("smiles",),
            optional_args=("similarity_threshold", "limit"),
        ),
        MCPToolSpec(
            tool_id="chembl.search_substructure",
            category="compound",
            purpose="Find compounds containing a substructure.",
            server_hint="chembl",
            required_args=("smiles",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.analyze_molecular_properties",
            category="compound",
            purpose="Assess molecular descriptors and drug-likeness.",
            server_hint="chembl",
            required_args=("chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.get_compound_bioactivities",
            category="bioactivity",
            purpose="Pull activity records for a compound across assays/targets.",
            server_hint="chembl",
            required_args=("chembl_id",),
            optional_args=("activity_type", "limit"),
        ),
        MCPToolSpec(
            tool_id="chembl.search_activities",
            category="bioactivity",
            purpose="Retrieve activity data with flexible filters.",
            server_hint="chembl",
            required_args=("filters",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.analyze_sar",
            category="bioactivity",
            purpose="Summarize structure-activity relationships.",
            server_hint="chembl",
            required_args=("chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.predict_bioactivity",
            category="bioactivity",
            purpose="Estimate likely bioactivity from compound properties.",
            server_hint="chembl",
            required_args=("smiles",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_targets",
            category="target",
            purpose="Find targets and target classes for mechanism mapping.",
            server_hint="chembl",
            required_args=("query",),
            optional_args=("organism", "limit"),
        ),
        MCPToolSpec(
            tool_id="chembl.get_target_info",
            category="target",
            purpose="Retrieve target metadata including identifiers/classification.",
            server_hint="chembl",
            required_args=("target_chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_assays",
            category="assay",
            purpose="Find assays relevant to a target or compound context.",
            server_hint="chembl",
            required_args=("query",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.get_assay_info",
            category="assay",
            purpose="Retrieve assay protocol/details for evidence quality checks.",
            server_hint="chembl",
            required_args=("assay_chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.get_mechanism_of_action",
            category="mechanism",
            purpose="Get known mechanism-of-action annotations.",
            server_hint="chembl",
            required_args=("chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.get_drug_indications",
            category="clinical",
            purpose="Retrieve disease indications linked to compound entities.",
            server_hint="chembl",
            required_args=("chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_by_target_chembl_id",
            category="target",
            purpose="Resolve target-centric records by ChEMBL target id.",
            server_hint="chembl",
            required_args=("target_chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_by_target_name",
            category="target",
            purpose="Find target records by common target name.",
            server_hint="chembl",
            required_args=("target_name",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_by_organism",
            category="target",
            purpose="Filter targets by organism.",
            server_hint="chembl",
            required_args=("organism",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_by_target_type",
            category="target",
            purpose="Filter targets by target type.",
            server_hint="chembl",
            required_args=("target_type",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_by_protein_class",
            category="target",
            purpose="Filter targets by protein class.",
            server_hint="chembl",
            required_args=("protein_class",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_by_uniprot",
            category="target",
            purpose="Map UniProt ids to ChEMBL target records.",
            server_hint="chembl",
            required_args=("uniprot_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.advanced_target_search",
            category="target",
            purpose="Multi-criteria target search across classes and metadata.",
            server_hint="chembl",
            required_args=("filters",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.analyze_target_landscape",
            category="target",
            purpose="Summarize a target landscape for a disease area.",
            server_hint="chembl",
            required_args=("query",),
        ),
        MCPToolSpec(
            tool_id="chembl.analyze_target_druggability",
            category="target",
            purpose="Assess target druggability and tractability clues.",
            server_hint="chembl",
            required_args=("target_chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.compare_targets",
            category="target",
            purpose="Compare targets to prioritize intervention candidates.",
            server_hint="chembl",
            required_args=("target_chembl_ids",),
        ),
        MCPToolSpec(
            tool_id="chembl.find_novel_targets",
            category="target",
            purpose="Find potentially underexplored targets for hypotheses.",
            server_hint="chembl",
            required_args=("disease_or_pathway",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_target_pathways",
            category="pathway",
            purpose="Fetch pathway links for a target.",
            server_hint="chembl",
            required_args=("target_chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_target_diseases",
            category="disease",
            purpose="Fetch disease associations for a target.",
            server_hint="chembl",
            required_args=("target_chembl_id",),
        ),
        MCPToolSpec(
            tool_id="opentargets.search_target",
            category="target",
            purpose="Resolve genes/targets and disease links for prioritization.",
            server_hint="opentargets",
            required_args=("query",),
            optional_args=("limit",),
        ),
        MCPToolSpec(
            tool_id="opentargets.get_target_disease_associations",
            category="target",
            purpose="Rank evidence for target-disease associations.",
            server_hint="opentargets",
            required_args=("ensembl_id",),
            optional_args=("disease_id", "limit"),
        ),
        MCPToolSpec(
            tool_id="reactome.pathway_enrichment",
            category="pathway",
            purpose="Enrich pathways from candidate gene list.",
            server_hint="reactome",
            required_args=("gene_symbols",),
            optional_args=("species",),
        ),
        MCPToolSpec(
            tool_id="stringdb.get_interactions",
            category="network",
            purpose="Build PPI context to identify network hubs and modules.",
            server_hint="stringdb",
            required_args=("gene_symbols",),
            optional_args=("species", "score_threshold"),
        ),
        MCPToolSpec(
            tool_id="pubmed.search",
            category="literature",
            purpose="Find recent papers supporting or contradicting hypotheses.",
            server_hint="pubmed",
            required_args=("query",),
            optional_args=("date_from", "max_results"),
        ),
        MCPToolSpec(
            tool_id="pubmed.get_abstract",
            category="literature",
            purpose="Retrieve abstract text for evidence extraction.",
            server_hint="pubmed",
            required_args=("pmid",),
        ),
        MCPToolSpec(
            tool_id="geo.search_series",
            category="dataset",
            purpose="Find public RNA-seq cohorts for model training/validation.",
            server_hint="geo",
            required_args=("query",),
            optional_args=("organism", "limit"),
        ),
        MCPToolSpec(
            tool_id="geo.get_series_metadata",
            category="dataset",
            purpose="Fetch cohort metadata to assess label quality/confounders.",
            server_hint="geo",
            required_args=("gse_id",),
        ),
    ]


def build_chembl_tool_catalog() -> list[MCPToolSpec]:
    return [t for t in build_drug_discovery_tool_catalog() if t.server_hint == "chembl"]


def build_anticancer_tool_plan(
    drug_name: str,
    cancer_type: str,
    phenotype: str = "drug resistance",
    profile: str = "chembl",
) -> list[MCPToolCallPlan]:
    chembl_plan = [
        MCPToolCallPlan(
            tool_id="chembl.search_compounds",
            reason="Resolve canonical compound entities and synonyms.",
            arguments={"query": drug_name, "limit": "10"},
        ),
        MCPToolCallPlan(
            tool_id="chembl.get_compound_info",
            reason="Pull canonical identifiers/properties for downstream joins.",
            arguments={"chembl_id": "<from_search_compounds_top_hit>"},
        ),
        MCPToolCallPlan(
            tool_id="chembl.search_targets",
            reason="Identify known targets and mechanism anchors.",
            arguments={"query": drug_name, "limit": "20"},
        ),
        MCPToolCallPlan(
            tool_id="chembl.get_mechanism_of_action",
            reason="Collect curated mechanism annotations.",
            arguments={"chembl_id": "<from_search_compounds_top_hit>"},
        ),
        MCPToolCallPlan(
            tool_id="chembl.search_activities",
            reason="Collect potency/activity evidence linked to targets.",
            arguments={
                "filters": f"compound={drug_name};activity_type=IC50;organism=Homo sapiens",
                "limit": "100",
            },
        ),
        MCPToolCallPlan(
            tool_id="chembl.search_target_diseases",
            reason="Cross-check target-disease associations in ChEMBL records.",
            arguments={"target_chembl_id": "<from_search_targets_top_target>"},
        ),
        MCPToolCallPlan(
            tool_id="chembl.search_target_pathways",
            reason="Map targets to pathways for intervention hypotheses.",
            arguments={"target_chembl_id": "<from_search_targets_top_target>"},
        ),
    ]

    if profile == "chembl":
        return chembl_plan

    return [
        *chembl_plan,
        MCPToolCallPlan(
            tool_id="geo.search_series",
            reason="Find relevant RNA-seq datasets for training and holdout.",
            arguments={
                "query": f"{cancer_type} RNA-seq {drug_name} {phenotype}",
                "organism": "Homo sapiens",
                "limit": "25",
            },
        ),
        MCPToolCallPlan(
            tool_id="pubmed.search",
            reason="Gather recent literature for mechanism support.",
            arguments={
                "query": f"{drug_name} resistance {cancer_type} mechanisms",
                "date_from": "2018-01-01",
                "max_results": "25",
            },
        ),
        MCPToolCallPlan(
            tool_id="reactome.pathway_enrichment",
            reason="Map candidate genes to pathways for intervention design.",
            arguments={"gene_symbols": "<from_model_top_genes>", "species": "Homo sapiens"},
        ),
        MCPToolCallPlan(
            tool_id="opentargets.get_target_disease_associations",
            reason="Prioritize targets by disease relevance evidence.",
            arguments={"ensembl_id": "<candidate_target_ensembl>", "limit": "20"},
        ),
    ]

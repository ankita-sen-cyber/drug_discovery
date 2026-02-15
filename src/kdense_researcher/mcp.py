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

    def list_tools(self) -> list[MCPToolSpec]:
        return build_drug_discovery_tool_catalog()

    def fetch_context(self, question: str, top_k: int = 3) -> list[MCPContextItem]:
        return []


def build_drug_discovery_tool_catalog() -> list[MCPToolSpec]:
    # These are tool contracts expected from connected MCP servers.
    # They are intentionally server-agnostic with a server_hint.
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
            tool_id="chembl.get_compound",
            category="compound",
            purpose="Retrieve canonical ChEMBL compound details and properties.",
            server_hint="chembl",
            required_args=("chembl_id",),
        ),
        MCPToolSpec(
            tool_id="chembl.search_activities",
            category="bioactivity",
            purpose="Fetch activity values for a target, assay type, or compound.",
            server_hint="chembl",
            required_args=("target_or_compound",),
            optional_args=("assay_type", "activity_type", "limit"),
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


def build_anticancer_tool_plan(
    drug_name: str,
    cancer_type: str,
    phenotype: str = "drug resistance",
) -> list[MCPToolCallPlan]:
    return [
        MCPToolCallPlan(
            tool_id="chembl.search_compounds",
            reason="Resolve canonical compound entities and synonyms.",
            arguments={"query": drug_name, "limit": "10"},
        ),
        MCPToolCallPlan(
            tool_id="chembl.search_targets",
            reason="Identify known targets and mechanism anchors.",
            arguments={"query": drug_name, "limit": "20"},
        ),
        MCPToolCallPlan(
            tool_id="chembl.search_activities",
            reason="Collect potency/activity evidence linked to targets.",
            arguments={
                "target_or_compound": drug_name,
                "assay_type": "B",
                "activity_type": "IC50",
                "limit": "100",
            },
        ),
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

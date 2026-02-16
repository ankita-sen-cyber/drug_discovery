from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .llm import build_default_llm, get_llm_provider_name
from .rag import LocalRAG


def analyze_text_with_llm_and_rag(
    input_data: str,
    docs_dir: Path,
    drug_name: str = "cisplatin",
    cancer_type: str = "NSCLC",
) -> dict[str, Any]:
    rag = LocalRAG(AppConfig(docs_dir=docs_dir).rag)
    if docs_dir.exists():
        rag.index_dir(docs_dir)
    rag_chunks = rag.query(input_data, top_k=6)

    llm = build_default_llm()
    llm_provider = get_llm_provider_name(llm)
    prompt = _build_structured_prompt(
        input_data=input_data,
        rag_chunks=rag_chunks,
        drug_name=drug_name,
        cancer_type=cancer_type,
    )
    try:
        raw = llm.complete(prompt)
        parsed = _parse_llm_json(raw)
    except Exception as exc:
        raise RuntimeError(f"LLM invocation failed: {exc}") from exc
    if parsed is None:
        raise RuntimeError("LLM output could not be parsed as JSON.")

    normalized = _normalize_payload(parsed)
    if not normalized["pathways"] and not normalized["targets"]:
        raise RuntimeError("LLM returned empty pathways/targets.")
    return {
        "pathways": normalized["pathways"],
        "targets": normalized["targets"],
        "hypothesis": normalized["hypothesis"],
        "metadata": {
            "mode": "llm_rag",
            "llm_provider": llm_provider,
            "rag_enabled": bool(rag_chunks),
            "rag_sources": [c.source for c in rag_chunks],
        },
    }


def _build_structured_prompt(
    input_data: str,
    rag_chunks: list[Any],
    drug_name: str,
    cancer_type: str,
) -> str:
    rag_block = "\n\n".join(
        [
            f"[RAG:{i+1}] source={c.source}\n{c.text[:1200]}"
            for i, c in enumerate(rag_chunks)
        ]
    )
    return (
        "You are KDense oncology research assistant.\n"
        "Task: extract pathways, gene targets, and a concise mechanistic hypothesis from input.\n"
        "Context: anticancer drug-response focus.\n"
        f"Drug context: {drug_name}\n"
        f"Cancer context: {cancer_type}\n\n"
        "Return ONLY valid JSON (no markdown, no commentary) with this exact schema:\n"
        "{\n"
        '  "pathways": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "description": "string",\n'
        '      "genes": ["GENE1", "GENE2"],\n'
        '      "confidence": 0.0,\n'
        '      "category": "Oncology|Signaling|Immunology|Inflammation|Cell Cycle|DNA Repair|Other"\n'
        "    }\n"
        "  ],\n"
        '  "targets": [\n'
        "    {\n"
        '      "name": "string",\n'
        '      "type": "string",\n'
        '      "description": "string",\n'
        '      "druggability": "High|Medium|Low",\n'
        '      "knownDrugs": ["DrugA"],\n'
        '      "pathways": ["Pathway Name"]\n'
        "    }\n"
        "  ],\n"
        '  "hypothesis": {\n'
        '    "summary": "1-2 sentences",\n'
        '    "mechanism": "Mechanistic explanation",\n'
        '    "intervention": "Testable intervention idea",\n'
        '    "validation": "Wet-lab validation suggestion",\n'
        '    "confidence": 0.0\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Provide 3-8 pathways and 3-8 targets when evidence supports them.\n"
        "- Confidence values must be between 0 and 1.\n"
        "- Use uppercase gene symbols where possible.\n"
        "- Keep target/pathway names human-readable.\n\n"
        f"User input:\n{input_data}\n\n"
        f"RAG context:\n{rag_block}\n"
    )


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None

    # First pass: direct JSON parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Second pass: extract outer-most JSON object.
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = text[first : last + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    pathways_raw = payload.get("pathways", [])
    targets_raw = payload.get("targets", [])
    hypothesis_raw = payload.get("hypothesis", {})

    pathways: list[dict[str, Any]] = []
    for p in pathways_raw if isinstance(pathways_raw, list) else []:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "")).strip()
        if not name:
            continue
        genes = p.get("genes", [])
        genes_list = [str(g).strip().upper() for g in genes if str(g).strip()] if isinstance(genes, list) else []
        confidence = _clamp01(p.get("confidence", 0.6))
        pathways.append(
            {
                "name": name,
                "description": str(p.get("description", "")).strip() or "No description provided.",
                "genes": genes_list[:12],
                "confidence": confidence,
                "category": str(p.get("category", "Other")).strip() or "Other",
            }
        )

    targets: list[dict[str, Any]] = []
    for t in targets_raw if isinstance(targets_raw, list) else []:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name", "")).strip()
        if not name:
            continue
        druggability = str(t.get("druggability", "Medium")).strip().title()
        if druggability not in {"High", "Medium", "Low"}:
            druggability = "Medium"
        known_drugs = t.get("knownDrugs", [])
        linked_pathways = t.get("pathways", [])
        targets.append(
            {
                "name": name,
                "type": str(t.get("type", "Target")).strip() or "Target",
                "description": str(t.get("description", "")).strip() or "No description provided.",
                "druggability": druggability,
                "knownDrugs": [str(d).strip() for d in known_drugs if str(d).strip()] if isinstance(known_drugs, list) else [],
                "pathways": [str(p).strip() for p in linked_pathways if str(p).strip()] if isinstance(linked_pathways, list) else [],
            }
        )

    if not isinstance(hypothesis_raw, dict):
        hypothesis_raw = {}
    hypothesis = {
        "summary": str(hypothesis_raw.get("summary", "")).strip() or "Insufficient evidence for a concise hypothesis.",
        "mechanism": str(hypothesis_raw.get("mechanism", "")).strip() or "No mechanism provided.",
        "intervention": str(hypothesis_raw.get("intervention", "")).strip() or "No intervention suggested.",
        "validation": str(hypothesis_raw.get("validation", "")).strip() or "No validation plan provided.",
        "confidence": _clamp01(hypothesis_raw.get("confidence", 0.5)),
    }

    return {"pathways": pathways[:8], "targets": targets[:8], "hypothesis": hypothesis}


def _clamp01(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, x))

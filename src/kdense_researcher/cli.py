from __future__ import annotations

import argparse
from pathlib import Path

from .agent import ResearchAgent
from .config import AppConfig
from .llm import build_default_llm
from .mcp import NullMCPProvider
from .rag import LocalRAG


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KDense AI Researcher CLI")
    p.add_argument("--query", help="Research question")
    p.add_argument(
        "--docs-dir",
        default="./data/literature",
        help="Directory containing .txt/.md context documents",
    )
    p.add_argument(
        "--drug",
        default="cisplatin",
        help="Drug name used for the anticancer tool plan",
    )
    p.add_argument(
        "--cancer-type",
        default="NSCLC",
        help="Cancer type used for the anticancer tool plan",
    )
    p.add_argument(
        "--list-tools",
        action="store_true",
        help="List available MCP tool definitions for this workflow",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig(docs_dir=Path(args.docs_dir))
    rag = LocalRAG(config.rag)
    if config.docs_dir.exists():
        rag.index_dir(config.docs_dir)
    llm = build_default_llm()
    mcp = NullMCPProvider()
    if args.list_tools:
        print("\n=== KDense MCP Tool Catalog ===\n")
        for t in mcp.list_tools():
            print(
                f"- {t.tool_id} [{t.category}] via {t.server_hint}\n"
                f"  purpose: {t.purpose}\n"
                f"  required: {', '.join(t.required_args)}\n"
                f"  optional: {', '.join(t.optional_args) if t.optional_args else '-'}"
            )
        return

    if not args.query:
        raise SystemExit("--query is required unless --list-tools is set.")

    agent = ResearchAgent(rag=rag, llm=llm, mcp=mcp)
    result = agent.run(args.query, drug_name=args.drug, cancer_type=args.cancer_type)

    print("\n=== KDense AI Researcher ===\n")
    print(result.answer)
    print("\n--- Planned MCP Tool Calls ---")
    for p in result.planned_tools:
        print(f"{p.tool_id} | reason={p.reason} | args={p.arguments}")
    print("\n--- RAG Sources ---")
    for s in result.rag_sources:
        print(s)
    print("\n--- MCP Sources ---")
    for s in result.mcp_sources:
        print(s)


if __name__ == "__main__":
    main()

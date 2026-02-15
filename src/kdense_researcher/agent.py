from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM
from .mcp import MCPProvider, MCPToolCallPlan, build_anticancer_tool_plan
from .rag import LocalRAG


@dataclass(frozen=True)
class ResearchAnswer:
    answer: str
    rag_sources: list[str]
    mcp_sources: list[str]
    planned_tools: list[MCPToolCallPlan]


class ResearchAgent:
    def __init__(self, rag: LocalRAG, llm: LLM, mcp: MCPProvider):
        self.rag = rag
        self.llm = llm
        self.mcp = mcp

    def run(
        self,
        query: str,
        drug_name: str = "cisplatin",
        cancer_type: str = "NSCLC",
    ) -> ResearchAnswer:
        rag_chunks = self.rag.query(query)
        mcp_items = self.mcp.fetch_context(query, top_k=3)
        tool_plan = build_anticancer_tool_plan(
            drug_name=drug_name,
            cancer_type=cancer_type,
            phenotype="drug resistance",
        )
        prompt = self._build_prompt(query, rag_chunks, mcp_items, tool_plan)
        response = self.llm.complete(prompt)
        return ResearchAnswer(
            answer=response,
            rag_sources=[c.source for c in rag_chunks],
            mcp_sources=[m.source for m in mcp_items],
            planned_tools=tool_plan,
        )

    def _build_prompt(self, query, rag_chunks, mcp_items, tool_plan) -> str:
        rag_block = "\n\n".join(
            [f"[RAG:{i+1}] source={c.source}\n{c.text}" for i, c in enumerate(rag_chunks)]
        )
        mcp_block = "\n\n".join(
            [
                f"[MCP:{i+1}] name={m.name} source={m.source}\n{m.content}"
                for i, m in enumerate(mcp_items)
            ]
        )
        plan_block = "\n".join(
            [
                f"- {t.tool_id} | reason={t.reason} | args={t.arguments}"
                for t in tool_plan
            ]
        )
        return (
            "You are KDense AI Researcher.\n"
            "Task: infer anticancer drug-response mechanisms and propose testable targets.\n"
            "Rules:\n"
            "- Separate evidence from inference.\n"
            "- Provide failure risks/confounders.\n"
            "- Output: 1) findings 2) mechanisms 3) target shortlist 4) validation plan.\n\n"
            f"User query:\n{query}\n\n"
            f"Planned MCP tool calls:\n{plan_block}\n\n"
            f"RAG context:\n{rag_block}\n\n"
            f"MCP context:\n{mcp_block}\n"
        )

# KDense AI Researcher (MCP + RAG Starter)

Minimal starter to build an AI researcher for anticancer drug-response discovery.

## What this includes

- Local RAG over text/markdown documents (no external vector DB required)
- MCP-ready context provider interface for tools/resources
- Research agent orchestration loop
- CLI entrypoint
- Anticancer MVP protocol template

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run:

```bash
kdense-research \
  --query "What mechanisms may drive cisplatin resistance in NSCLC?" \
  --docs-dir ./data/literature
```

List drug-discovery MCP tool definitions:

```bash
kdense-research --list-tools
```

Run with explicit drug/cancer context for tool planning:

```bash
kdense-research \
  --query "Propose hypotheses for resistance and intervention targets" \
  --drug cisplatin \
  --cancer-type NSCLC
```

Optional environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4.1-mini`)

If `OPENAI_API_KEY` is missing, the CLI falls back to a deterministic stub model.

## Project layout

- `src/kdense_researcher/`: agent code
- `MVP_PROTOCOL.md`: anticancer drug-response protocol
- `data/literature/`: place `.txt` or `.md` files here for RAG context

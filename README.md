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

List only ChEMBL-MCP-Server tools:

```bash
kdense-research --list-tools --tool-profile chembl
```

Run with explicit drug/cancer context for tool planning:

```bash
kdense-research \
  --query "Propose hypotheses for resistance and intervention targets" \
  --drug cisplatin \
  --cancer-type NSCLC
```

Ingest PubMed abstracts into local RAG docs:

```bash
kdense-ingest-pubmed \
  --query "cisplatin resistance NSCLC RNA-seq" \
  --max-results 25 \
  --out-dir ./data/literature
```

Optional for higher NCBI API limits:

```bash
export NCBI_API_KEY=your_ncbi_api_key
```

Optional environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4.1-mini`)

If `OPENAI_API_KEY` is missing, the CLI falls back to a deterministic stub model.

## MCP config (ChEMBL via Podman)

A ready config file is included at:

- `mcp_servers.chembl.json`

Contents:

```json
{
  "mcpServers": {
    "chembl": {
      "command": "podman",
      "args": ["run", "-i", "--rm", "chembl-mcp-server"]
    }
  }
}
```

Add this server block to your Codex MCP client settings and restart the session.

## Project layout

- `src/kdense_researcher/`: agent code
- `MVP_PROTOCOL.md`: anticancer drug-response protocol
- `data/literature/`: place `.txt` or `.md` files here for RAG context

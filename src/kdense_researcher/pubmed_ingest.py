from __future__ import annotations

import argparse
import os
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: str


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch PubMed abstracts and save markdown files for RAG."
    )
    p.add_argument("--query", required=True, help="PubMed query string")
    p.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of PubMed records to fetch",
    )
    p.add_argument(
        "--out-dir",
        default="./data/literature",
        help="Output directory for markdown files",
    )
    p.add_argument(
        "--email",
        default="",
        help="Contact email for NCBI requests (recommended)",
    )
    p.add_argument(
        "--retmax-hard-limit",
        type=int,
        default=200,
        help="Safety cap to prevent accidental huge downloads",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.max_results < 1:
        raise SystemExit("--max-results must be >= 1")
    if args.max_results > args.retmax_hard_limit:
        raise SystemExit(
            f"--max-results exceeds hard limit ({args.retmax_hard_limit}). "
            "Increase --retmax-hard-limit explicitly if needed."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pmids = search_pmids(
            query=args.query,
            max_results=args.max_results,
            email=args.email,
            api_key=os.getenv("NCBI_API_KEY", ""),
        )
    except urllib.error.URLError as exc:
        raise SystemExit(
            "PubMed search request failed. Check internet access/DNS and retry. "
            f"Underlying error: {exc}"
        ) from exc
    if not pmids:
        print("No PubMed IDs found for query.")
        return

    try:
        articles = fetch_articles(
            pmids=pmids,
            email=args.email,
            api_key=os.getenv("NCBI_API_KEY", ""),
        )
    except urllib.error.URLError as exc:
        raise SystemExit(
            "PubMed fetch request failed. Check internet access/DNS and retry. "
            f"Underlying error: {exc}"
        ) from exc
    if not articles:
        print("No articles returned by PubMed fetch.")
        return

    written = 0
    for article in articles:
        if not article.abstract.strip():
            continue
        filename = f"{article.pmid}_{slug(article.title)}.md"
        path = out_dir / filename
        path.write_text(render_markdown(article), encoding="utf-8")
        written += 1

    print(f"Fetched {len(articles)} records; wrote {written} markdown files to {out_dir}")


def search_pmids(query: str, max_results: int, email: str, api_key: str) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max_results),
        "sort": "relevance",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = f"{ESEARCH_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = resp.read().decode("utf-8")
    import json

    data = json.loads(payload)
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_articles(pmids: list[str], email: str, api_key: str) -> list[PubMedArticle]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = f"{EFETCH_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        xml_text = resp.read().decode("utf-8")
    return parse_pubmed_xml(xml_text)


def parse_pubmed_xml(xml_text: str) -> list[PubMedArticle]:
    root = ET.fromstring(xml_text)
    out: list[PubMedArticle] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = text_or_empty(article.find(".//PMID"))
        title = text_or_empty(article.find(".//ArticleTitle"))
        journal = text_or_empty(article.find(".//Journal/Title"))
        year = text_or_empty(article.find(".//PubDate/Year"))
        if not year:
            year = text_or_empty(article.find(".//PubDate/MedlineDate"))[:4]

        abstract_parts: list[str] = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = (node.attrib.get("Label", "") or "").strip()
            content = text_or_empty(node).strip()
            if not content:
                continue
            if label:
                abstract_parts.append(f"{label}: {content}")
            else:
                abstract_parts.append(content)

        out.append(
            PubMedArticle(
                pmid=pmid or "unknown",
                title=title or "Untitled",
                abstract="\n\n".join(abstract_parts),
                journal=journal or "Unknown Journal",
                year=year or "Unknown",
            )
        )
    return out


def text_or_empty(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:80] if cleaned else "article"


def render_markdown(article: PubMedArticle) -> str:
    abstract = article.abstract.strip()
    if not abstract:
        abstract = "No abstract available."
    wrapped = "\n".join(textwrap.wrap(abstract, width=100))
    return (
        f"# {article.title}\n\n"
        f"- PMID: {article.pmid}\n"
        f"- Journal: {article.journal}\n"
        f"- Year: {article.year}\n\n"
        "## Abstract\n\n"
        f"{wrapped}\n"
    )


if __name__ == "__main__":
    main()

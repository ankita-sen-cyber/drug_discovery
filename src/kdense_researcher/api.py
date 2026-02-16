from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .llm_analysis import analyze_text_with_llm_and_rag


class KDenseAPIHandler(BaseHTTPRequestHandler):
    server_version = "KDenseAPI/0.1"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": "Not Found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/analyze":
            self._send_json({"error": "Not Found"}, status=404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw or "{}")
            input_data = str(payload.get("data", "")).strip()
            if not input_data:
                self._send_json({"error": "Field 'data' is required."}, status=400)
                return
            docs_dir = str(payload.get("docs_dir", "")).strip() or os.getenv(
                "KDENSE_DOCS_DIR", "./data/literature"
            )
            drug_name = str(payload.get("drug_name", "")).strip() or "cisplatin"
            cancer_type = str(payload.get("cancer_type", "")).strip() or "NSCLC"

            result = analyze_text_with_llm_and_rag(
                input_data=input_data,
                docs_dir=Path(docs_dir),
                drug_name=drug_name,
                cancer_type=cancer_type,
            )
            self._send_json(result, status=200)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=400)
        except RuntimeError as exc:
            # Explicit non-mock failure path: caller should retry after fixing runtime deps
            # (e.g., Ollama server, model availability, MCP connectivity).
            self._send_json({"error": str(exc)}, status=502)
        except Exception as exc:  # defensive fallback
            self._send_json({"error": f"Internal server error: {exc}"}, status=500)

    def log_message(self, fmt: str, *args) -> None:
        # Keep console output concise.
        print(f"[kdense-api] {self.address_string()} - {fmt % args}")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KDense lightweight API server")
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    return p


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), KDenseAPIHandler)
    print(f"KDense API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

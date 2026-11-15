"""A small HTTP proxy in front of a `llama-server` running some other embedding model, so rummage and the plain
dense check can run on it unchanged: it rewrites rummage's EmbeddingGemma prompts (`task: search result | query:
...` and `title: ... | text: ...`) into the prompts that model's card asks for, and passes everything else through.

    python bench/model_proxy.py --profile harrier --listen 18190 --upstream http://127.0.0.1:18091

Profiles (prompts from each model card):
  embeddinggemma  passes the text as is (EmbeddingGemma 1 and 2 share the prompt format)
  instruct        `Instruct: <task>\\nQuery: <query>`, documents as plain text (Qwen3-Embedding, harrier-oss-v1)
  plain           queries and documents as plain text (BGE-M3)
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QUERY = "task: search result | query: "
DOCUMENT = re.compile(r"title: (.*?) \| text: (.*)", re.DOTALL)
TASK = "Given a web search query, retrieve relevant passages that answer the query"


def _document(text: str) -> tuple[str, str] | None:
    match = DOCUMENT.fullmatch(text)
    if not match:
        return None
    title = match.group(1)
    return ("" if title == "none" else title), match.group(2)


def rewrite(profile: str, text: str) -> str:
    if profile == "embeddinggemma":
        return text
    if text.startswith(QUERY):
        query = text[len(QUERY):]
        return f"Instruct: {TASK}\nQuery: {query}" if profile == "instruct" else query
    document = _document(text)
    if document:
        title, body = document
        return f"{title}\n{body}" if title else body
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=["embeddinggemma", "instruct", "plain"], required=True)
    parser.add_argument("--listen", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    upstream = args.upstream.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        def _forward(self, body: bytes | None) -> None:
            request = urllib.request.Request(upstream + self.path, data=body, method=self.command,
                                             headers={"Content-Type": "application/json"} if body else {})
            try:
                with urllib.request.urlopen(request, timeout=900) as r:
                    status, payload = r.status, r.read()
            except urllib.error.HTTPError as error:
                status, payload = error.code, error.read()
            except (urllib.error.URLError, OSError) as error:
                status, payload = 502, json.dumps({"error": str(error)}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._forward(None)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            for key in ("input", "content"):
                value = body.get(key)
                if isinstance(value, str):
                    body[key] = rewrite(args.profile, value)
                elif isinstance(value, list):
                    body[key] = [rewrite(args.profile, v) if isinstance(v, str) else v for v in value]
            self._forward(json.dumps(body).encode())

        def log_message(self, *_) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", args.listen), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

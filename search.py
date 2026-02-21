"""
Seek — semantic code search CLI.

Interactive mode:
    python search.py

One-shot mode:
    python search.py "function that reads a file"
    python search.py "auth middleware" --top-k 10
    python search.py "db connection pool" --ext py,go
    python search.py "error handling" --path src/
    python search.py "jwt decode" --json
"""
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("RUST_LOG", "error")

import argparse
import json
import logging
import re
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cocoindex
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

from config import TOP_K_DEFAULT, EMBED_MODEL, database_url
from index import seek_index_flow

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

_model = SentenceTransformer(EMBED_MODEL)

_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def bold(t: str)   -> str: return _c("1",  t)
def dim(t: str)    -> str: return _c("2",  t)
def cyan(t: str)   -> str: return _c("36", t)
def green(t: str)  -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)

# Table name comes from CocoIndex (not user input) — validate before interpolating.
_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$", re.IGNORECASE)


def _table_name() -> str:
    name = cocoindex.utils.get_target_storage_default_name(
        seek_index_flow, "code_embeddings"
    )
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(f"Unexpected table name from CocoIndex: {name!r}")
    return name


def search(
    pool: ConnectionPool,
    query: str,
    *,
    top_k: int = TOP_K_DEFAULT,
    ext_filter: list[str] | None = None,
    path_filter: str | None = None,
) -> list[dict]:
    if not query.strip():
        return []

    query_vector = _model.encode(query).tolist()
    table = _table_name()

    filter_clauses: list[str] = []
    filter_params: list = []

    if ext_filter:
        pattern = r"\.(" + "|".join(re.escape(e.lstrip(".")) for e in ext_filter) + r")$"
        filter_clauses.append("filename ~ %s")
        filter_params.append(pattern)

    if path_filter:
        filter_clauses.append("filename ILIKE %s")
        filter_params.append(f"%{path_filter}%")

    where = ("WHERE " + " AND ".join(filter_clauses)) if filter_clauses else ""

    query_sql = f"""
        SELECT filename, location, code, embedding <=> %s::vector AS distance
        FROM {table}
        {where}
        ORDER BY distance
        LIMIT %s
    """
    params = filter_params + [query_vector, top_k]

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_sql, params)
                rows = cur.fetchall()
    except Exception as exc:
        log.error("Database query failed: %s", exc)
        raise

    return [
        {
            "filename": r[0],
            "location": r[1],
            "code": r[2],
            # pgvector <=> is cosine distance; convert to similarity score
            "score": round(1.0 - r[3], 4),
        }
        for r in rows
    ]


def _score_color(score: float) -> str:
    if score >= 0.70:
        return green(f"{score:.3f}")
    if score >= 0.40:
        return yellow(f"{score:.3f}")
    return dim(f"{score:.3f}")


def _print_results(results: list[dict], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(results, indent=2))
        return

    print(f"\n{bold(f'--- {len(results)} result(s) ---')}")
    for i, r in enumerate(results, 1):
        snippet = r["code"].rstrip()
        truncated = len(snippet) > 300
        print(
            f"\n{bold(str(i))}. {cyan(r['filename'])}"
            f"  {dim(r['location'])}"
            f"  score: {_score_color(r['score'])}"
        )
        print(snippet[:300] + (dim(" …") if truncated else ""))
        print(dim("-" * 60))
    print()


def _run_query(pool: ConnectionPool, args: argparse.Namespace, query: str) -> None:
    ext_filter = [e.lstrip(".") for e in args.ext.split(",")] if args.ext else None

    try:
        results = search(
            pool,
            query,
            top_k=args.top_k,
            ext_filter=ext_filter,
            path_filter=args.path or None,
        )
    except Exception as exc:
        print(f"Search error: {exc}", file=sys.stderr)
        return

    if not results:
        print("No results found.\n")
    else:
        _print_results(results, as_json=args.json)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seek",
        description="Seek — semantic code search across your local repos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python search.py "function that reads a file"
  python search.py "auth middleware" --top-k 10
  python search.py "db connection" --ext py,go
  python search.py "error handling" --path src/
  python search.py "jwt decode" --json | jq '.[].filename'
""",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (omit to enter interactive REPL mode)",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        metavar="N",
        help=f"Number of results (default: {TOP_K_DEFAULT})",
    )
    parser.add_argument(
        "--ext",
        metavar="py,go,ts",
        help="Filter by file extension(s), comma-separated",
    )
    parser.add_argument(
        "--path",
        metavar="SUBSTR",
        help="Filter results to filenames containing this substring",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON",
    )
    args = parser.parse_args()

    try:
        db_url = database_url()
    except EnvironmentError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        pool = ConnectionPool(db_url, open=True)
    except Exception as exc:
        print(f"Could not connect to database: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.query:
            _run_query(pool, args, args.query)
        else:
            print(bold("Seek") + " — Semantic Code Search")
            print(dim("Type a query and press Enter. Empty input quits.\n"))
            while True:
                try:
                    query = input(cyan("Search: ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not query:
                    break
                _run_query(pool, args, query)
    finally:
        pool.close()


if __name__ == "__main__":
    main()

# vector_cleanup.py

import os
import sys
import json
from urllib.parse import quote_plus
import re
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
import psycopg
from collections import Counter  # add at the top with imports


load_dotenv()

# ---- PG / LangChain setup ----
PG_USER = os.getenv("PG_USER", "doadmin")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT", "25060")
PG_DB = os.getenv("PG_DATABASE", "emmett_ai")

if PG_PASSWORD is None or PG_HOST is None or PG_DB is None:
    raise RuntimeError("PG connection env vars not set (PG_PASSWORD / PG_HOST / PG_DATABASE).")

ENC_PWD = quote_plus(PG_PASSWORD)

# For psycopg (SQL)
PG_DSN = f"postgresql://{PG_USER}:{ENC_PWD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=require"

# For LangChain PGVector
PG_SA_URL = f"postgresql+psycopg://{PG_USER}:{ENC_PWD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=require"

EMBEDDING_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
COLLECTION = os.getenv("COLLECTION", "docs")
MANIFEST_NAME = "manifest.json"

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def make_pg_vectorstore():
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=PG_SA_URL,
        use_jsonb=True,
        create_extension=False,
    )


def load_manifest(path: str = "."):
    mpath = os.path.join(path, MANIFEST_NAME)
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(path: str, manifest: dict):
    mpath = os.path.join(path, MANIFEST_NAME)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# ---------- DB helpers ----------
def list_all_sources_raw() -> list[str]:
    """
    Return ALL metadata['source'] values (no DISTINCT),
    one per embedding row. This lets us detect exact duplicates.
    """
    conn = psycopg.connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT (cmetadata->>'source') AS source
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND cmetadata ? 'source'
                ORDER BY 1;
                """,
                (COLLECTION,),
            )
            rows = cur.fetchall()
            return [r[0] for r in rows if r[0] is not None]
    finally:
        conn.close()

def list_distinct_sources() -> list[str]:
    """
    Return all distinct metadata['source'] values from this collection.
    Useful just to show the user what exists.
    """
    conn = psycopg.connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT (cmetadata->>'source') AS source
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND cmetadata ? 'source'
                ORDER BY 1;
                """,
                (COLLECTION,),
            )
            rows = cur.fetchall()
            return [r[0] for r in rows if r[0] is not None]
    finally:
        conn.close()


def find_sources_matching(user_input: str) -> list[str]:
    """
    Find all distinct sources whose value contains the user_input (case-insensitive).
    This allows matching just on filename like 'BHE_Internship_Report (1).pdf'.
    """
    pattern = f"%{user_input}%"
    conn = psycopg.connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT (cmetadata->>'source') AS source
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND (cmetadata->>'source') ILIKE %s
                ORDER BY 1;
                """,
                (COLLECTION, pattern),
            )
            rows = cur.fetchall()
            return [r[0] for r in rows if r[0] is not None]
    finally:
        conn.close()


def count_vectors_for_source(full_source: str) -> int:
    """
    Count embeddings for a specific *full* source value.
    """
    conn = psycopg.connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND (cmetadata->>'source') = %s;
                """,
                (COLLECTION, full_source),
            )
            (count,) = cur.fetchone()
            return int(count or 0)
    finally:
        conn.close()


# ---------- Delete helper ----------

def delete_file_vectors(full_source: str, update_manifest: bool = True) -> int:
    """
    Delete all vectors where metadata['source'] == full_source
    using raw SQL, and optionally remove it from manifest.json.
    Returns number of deleted vectors (best effort).
    """
    before = count_vectors_for_source(full_source)
    print(f"Before delete: {before} vectors for {full_source!r}")

    # Perform the actual delete in Postgres
    conn = psycopg.connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                AND (cmetadata->>'source') = %s;
                """,
                (COLLECTION, full_source),
            )
        conn.commit()
    finally:
        conn.close()

    after = count_vectors_for_source(full_source)
    print(f"After delete:  {after} vectors for {full_source!r}")

    if update_manifest:
        manifest = load_manifest(".")
        if full_source in manifest:
            manifest.pop(full_source, None)
            save_manifest(".", manifest)

    # Number of deleted rows
    return before - after

def normalize_source_name(src: str) -> str:
    """
    Normalize a source string so we can spot likely duplicates.

    - Take only the basename (drop directories)
    - Remove common ' (1)', ' (2)' style suffixes before extension
    - Lowercase for case-insensitive comparison
    """
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)

    # Strip " (1)", " (2)", etc. at the end of the name
    name = re.sub(r"\s+\(\d+\)$", "", name)

    return (name + ext).lower()


def find_duplicate_sources_by_normalized_name() -> dict[str, list[str]]:
    """
    Group distinct sources by normalized name, and keep only groups
    that have more than one distinct variant.
    Example:
      'report.pdf' and 'report (1).pdf' will be grouped together.
    """
    sources = list_distinct_sources()
    groups: dict[str, list[str]] = {}

    for src in sources:
        key = normalize_source_name(src)
        groups.setdefault(key, []).append(src)

    return {k: v for k, v in groups.items() if len(v) > 1}


def find_exact_duplicate_sources() -> dict[str, int]:
    """
    Look at ALL sources (no DISTINCT) and count occurrences.
    Returns only sources that show up more than once.
    This detects cases where the same file was re-ingested
    with exactly the same 'source' string.
    """
    all_sources = list_all_sources_raw()
    counts = Counter(all_sources)
    return {src: cnt for src, cnt in counts.items() if cnt > 1}



def ask_and_delete_duplicate():
    """
    Show duplicate-looking files, let the user pick one to delete.

    - First show groups of distinct sources that normalize to the same name
      (e.g. 'file.pdf' vs 'file (1).pdf')
    - Then show exact duplicate 'source' strings that appear multiple times
      in the DB.
    """
    dup_groups = find_duplicate_sources_by_normalized_name()
    exact_dups = find_exact_duplicate_sources()

    if not dup_groups and not exact_dups:
        print("No duplicate-looking files found.")
        return

    index_to_source: list[str] = []
    counter = 1

    if dup_groups:
        print("\nDuplicate-looking groups (normalized name):")
        for key in sorted(dup_groups.keys()):
            variants = dup_groups[key]
            print(f"\nGroup: {key}")
            for src in variants:
                print(f"  {counter}. {src}")
                index_to_source.append(src)
                counter += 1

    if exact_dups:
        print("\nExact duplicate 'source' values (same string used multiple times):")
        for src in sorted(exact_dups.keys()):
            count = exact_dups[src]
            print(f"  {counter}. {src}  (occurs {count} times)")
            index_to_source.append(src)
            counter += 1

    choice = input(
        "\nEnter the NUMBER of the file you want to delete "
        "(or press Enter to cancel):\n> "
    ).strip()

    if not choice:
        print("Cancelled.")
        return

    try:
        idx = int(choice)
        if idx < 1 or idx > len(index_to_source):
            raise ValueError
    except ValueError:
        print("Invalid number. Aborting.")
        return

    full_source = index_to_source[idx - 1]

    vec_count = count_vectors_for_source(full_source)
    print(f"\nYou selected: {full_source}")
    print(f"Found {vec_count} vectors for this file.")

    confirm = input("Do you really want to delete these vectors? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Deletion cancelled.")
        return

    deleted = delete_file_vectors(full_source, update_manifest=True)

    print("\nResult:")
    print(f"Deleted vectors for: {full_source}")
    print("(Exact count may not be returned by PGVector.)")


# ---------- Interactive flow ----------

def ask_and_delete_one():
    """
    Show a numbered list of all distinct metadata['source'] values.
    User selects a number -> we verify -> delete that source only.
    """
    sources = list_distinct_sources()

    if not sources:
        print("No files found in the vector database.")
        return

    print("\nFiles currently indexed:")
    for idx, src in enumerate(sources, start=1):
        print(f"{idx}. {src}")

    choice = input(
        "\nEnter the NUMBER of the file you want to delete "
        "(or press Enter to cancel):\n> "
    ).strip()

    if not choice:
        print("Cancelled.")
        return

    try:
        idx = int(choice)
        if idx < 1 or idx > len(sources):
            raise ValueError
    except ValueError:
        print("Invalid number. Aborting.")
        return

    full_source = sources[idx - 1]

    # Count vectors belonging to that exact source
    vec_count = count_vectors_for_source(full_source)
    print(f"\nYou selected: {full_source}")
    print(f"Found {vec_count} vectors for this file.")

    confirm = input("Do you really want to delete these vectors? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Deletion cancelled.")
        return

    deleted = delete_file_vectors(full_source, update_manifest=True)

    print("\nResult:")
    print(f"Deleted vectors for: {full_source}")
    print("(Exact count may not be returned by PGVector.)")

if __name__ == "__main__":
    # Special mode: show duplicate-looking files and optionally delete one
    if len(sys.argv) >= 2 and sys.argv[1] == "--dups":
        ask_and_delete_duplicate()
        sys.exit(0)

    # If you pass an argument (other than --dups), we treat it as a "search string" (usually filename)
    # and still follow the safe logic (match -> confirm -> delete).
    if len(sys.argv) >= 2:
        # existing search_str logic...
        ...
    else:
        ask_and_delete_one()

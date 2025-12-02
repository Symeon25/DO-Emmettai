import os
import tiktoken
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

import psycopg
from urllib.parse import quote_plus

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from langchain_community.callbacks import get_openai_callback

# ------------- Postgres / PGVector config -------------

PG_USER = os.getenv("PG_USER", "doadmin")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT", "25060")
PG_DB = os.getenv("PG_DATABASE", "emmett_ai")

ENC_PWD = quote_plus(PG_PASSWORD)
PG_DSN = f"postgresql://{PG_USER}:{ENC_PWD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=require"
PG_SA_URL = f"postgresql+psycopg://{PG_USER}:{ENC_PWD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=require"

os.environ["PG_CONN_STR"] = PG_SA_URL

# Optional check
with psycopg.connect(PG_DSN) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        print(cur.fetchone())

# ------------- Model / embedding config --------------

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
EMBED_PRICE_PER_1M = float(os.getenv("EMBED_PRICE_PER_1M", "0.13"))
COLLECTION = os.getenv("COLLECTION", "docs")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def make_pg_vectorstore() -> PGVector:
    """Create a PGVector-backed vectorstore (no filesystem, no data folder)."""
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=PG_SA_URL,
        use_jsonb=True,
        create_extension=False,  # set True only once to create the extension
    )

# Single global vectorstore instance
vectorstore: PGVector = make_pg_vectorstore()

# If you still want BM25, you can keep it as an in-memory corpus of Documents
BM25_CORPUS: List[Document] = []


# ------------- Token counting / cost helpers -------------

MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached": 0.075},
}

def compute_cost(prompt_tokens, completion_tokens, model=MODEL, cached_fraction=0.0):
    p = MODEL_PRICES.get(model, MODEL_PRICES["gpt-4o-mini"])
    cached_tokens = int(prompt_tokens * cached_fraction)
    normal_tokens = prompt_tokens - cached_tokens
    cost_input = (normal_tokens * p["input"] + cached_tokens * p["cached"]) / 1_000_000
    cost_output = (completion_tokens * p["output"]) / 1_000_000
    return cost_input + cost_output

def _count_tokens_texts(texts, model_name: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(t or "")) for t in texts)


# ------------- Splitting (still generic, no folders) -------------

def split_docs(docs: List[Document]) -> List[Document]:
    """Split arbitrary Documents into smaller chunks using token-based splitter."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=60,
        encoding_name="cl100k_base",
    )
    return splitter.split_documents(docs)


# ------------- Direct vector-space ingestion ----------------

from typing import Dict, List
from openai import BadRequestError, APIError  # add this import at the top

# ... keep everything above as-is ...

def add_chunks_to_vectorstore(
    chunks: List[Document],
    collection_name: str | None = None,
) -> Dict:
    """
    Take a list of already-prepared chunks (Documents) and add them
    directly to PGVector. No filesystem, no 'data' directory.

    Returns a small summary with embedding token usage & cost.

    This version is more defensive:
    - ensures page_content is str
    - skips empty / extremely large chunks
    - surfaces clearer errors for embedding 400s
    """
    # Optional: allow overriding collection name at call-time
    global vectorstore
    if collection_name and collection_name != COLLECTION:
        # Rebuild vectorstore for a different collection if needed
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=PG_SA_URL,
            use_jsonb=True,
            create_extension=False,
        )

    texts = []
    metadatas = []
    ids = []

    for i, c in enumerate(chunks):
        md = dict(c.metadata or {})

        # ---- make sure content is a safe string ----
        text = c.page_content if c.page_content is not None else ""
        if not isinstance(text, str):
            text = str(text)

        # skip completely empty chunks
        if not text.strip():
            continue

        # OPTIONAL: very defensive guard against extreme size
        # (your chunk_size is 300 tokens, so this normally won't trigger)
        if len(text) > 20000:  # about 20k characters as a sanity check
            print(
                "Skipping very large chunk from",
                md.get("filename") or md.get("source") or "unknown",
            )
            continue

        # ---- ensure filename and ids are stable ----
        filename = md.get("filename")
        if not filename:
            src = md.get("source")
            if src:
                filename = os.path.basename(src)
            else:
                filename = "unknown_file"

        md["filename"] = filename

        # Make a readable, file-based doc_id
        doc_id = md.get("doc_id", f"{filename}::chunk::{i}")
        md["chunk_id"] = i
        md["doc_id"] = doc_id

        texts.append(text)
        metadatas.append(md)
        ids.append(doc_id)

    # If everything was filtered out, just return empty usage
    if not texts:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

    embed_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    }

    try:
        # This will call OpenAIEmbeddings internally
        with get_openai_callback() as cb:
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        emb_tokens = getattr(cb, "total_embedding_tokens", 0) or getattr(cb, "total_tokens", 0)
        if emb_tokens == 0:
            emb_tokens = _count_tokens_texts(texts, EMBEDDING_MODEL)

        embed_usage["total_tokens"] = emb_tokens
        embed_usage["prompt_tokens"] = emb_tokens
        embed_usage["total_cost"] = (emb_tokens / 1_000_000.0) * EMBED_PRICE_PER_1M

        # Optional BM25
        BM25_CORPUS.extend(
            [
                Document(page_content=texts[i], metadata=metadatas[i])
                for i in range(len(texts))
            ]
        )

        return embed_usage

    except BadRequestError as e:
        # This is the typical "HTTP 400" situation at the embedding layer
        # We raise a clearer message; your Streamlit layer will catch it per-file.
        raise RuntimeError(
            f"Embedding request was rejected (400 Bad Request). "
            f"This often means the content or embedding model is invalid. "
            f"Details: {e}"
        ) from e

    except APIError as e:
        # Other OpenAI API errors (5xx, network, etc.)
        raise RuntimeError(f"OpenAI API error while embedding chunks: {e}") from e

    except Exception as e:
        # Anything unexpected
        raise RuntimeError(f"Unexpected error while adding chunks to vector store: {e}") from e

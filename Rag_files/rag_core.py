# rag_core.py
"""
Core RAG plumbing: retrieval, reranking, question rewriting and usage tracking.

This module does NOT generate final answers – it only returns:
- rewritten question
- context (joined chunks)
- filenames
- structured sources list
- used_db flag
- confidence

The LangGraph agent (Agent.py) is responsible for turning this into a
user-facing Markdown answer using its own system prompt.
"""

import os
import json
import re
from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from Rag_files.vector_store import (
    vectorstore,       # PGVector instance
    MODEL,             # chat model name ("gpt-4o-mini", etc.)
    compute_cost,      # cost calculator
)

# ----------------- Basic config -----------------

TOP_K = int(os.getenv("TOP_K", "15"))
BASE_K = max(TOP_K, 20)           # at least 20 candidates
FETCH_K = max(TOP_K * 2, 40)      # at least 40 for MMR diversity

import re

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to",
    "in", "on", "with", "at", "by", "from", "is", "are",
    "was", "were", "be", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "why", "when",
}

def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def _doc_relevance_scores(question: str, docs) -> list[float]:
    """
    Compute a simple relevance score in [0, 1] for each doc based on
    overlap between question keywords and doc content.
    """
    q_tokens = _extract_keywords(question)
    if not q_tokens:
        return [0.0] * len(docs)

    scores: list[float] = []
    for d in docs:
        text = (d.page_content or "").lower()
        if not text:
            scores.append(0.0)
            continue
        hits = sum(1 for t in q_tokens if t in text)
        scores.append(hits / len(q_tokens))
    return scores
# ----------------- Retrievers: Dense + LLM Reranker -----------------
# Tracks which filenames were most recently used per session (for soft continuity)
ACTIVE_FILES: Dict[str, list[str]] = {}

# If you want MMR back, you can switch this to search_type="mmr" as in the comments
# in the original vector_chat.py.
dense = vectorstore.as_retriever(search_kwargs={"k": 20})

# LLM used for rewriting & reranking (no final answers here)
rerank_llm = ChatOpenAI(
    temperature=0.0,
    top_p=0.9,
    model=MODEL,
    streaming=False,
    stream_usage=True,
)

def rerank_with_llm(
    question: str,
    docs,
    top_k: int = 5,
    preferred_files: list[str] | None = None,
):
    """
    Rerank retrieved chunks using an LLM and return up to `top_k` UNIQUE docs.

    `preferred_files` is a SOFT hint: if provided, the reranker should
    prefer chunks coming from these filenames, unless others are clearly
    more relevant to the question.
    """
    if not docs:
        return []

    # Build a context block that also shows the source filename
    formatted_chunks = []
    for i, d in enumerate(docs):
        md = d.metadata or {}
        src = md.get("source", "unknown")
        filename = md.get("filename") or (os.path.basename(src) if src else "unknown")
        formatted_chunks.append(
            f"[Doc {i} | Source: {filename}]\n{d.page_content}"
        )

    context_block = "\n\n---\n\n".join(formatted_chunks)

    preferred_str = ""
    if preferred_files:
        preferred_str = (
            "Preferred source filenames (soft preference, not a hard filter):\n"
            + ", ".join(preferred_files)
            + "\n\n"
        )

    prompt = f"""
You are a retrieval reranker.

User question:
{question}

{preferred_str}Here are candidate document chunks:

{context_block}

Instructions:
- Prefer chunks whose Source filename is in the preferred list above, IF they are still relevant.
- However, do NOT ignore clearly more relevant chunks from other sources.
- Return the indices (comma-separated) of the TOP {top_k} most relevant [Doc i] chunks.
- Only return the indices, e.g.: 0,2,3,5,6
"""

    raw = rerank_llm.invoke(prompt).content

    # --- robustly parse indices from model output ---
    candidates = re.findall(r"\d+", raw)

    indices = []
    for c in candidates:
        idx = int(c)
        if 0 <= idx < len(docs):
            indices.append(idx)

    # Deduplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    # Enforce top_k cap
    unique_indices = unique_indices[:top_k]

    # Fallback if model output is unusable
    if not unique_indices:
        return docs[: min(top_k, len(docs))]

    return [docs[i] for i in unique_indices]



def dense_with_rerank(
    query: str,
    top_k: int = TOP_K,
    preferred_files: list[str] | None = None,
):
    """
    Retrieve candidates and rerank them.
    If `preferred_files` is provided, the reranker will softly prefer
    chunks coming from those filenames (document continuity).
    """
    retrieved = dense.invoke(query)   # 20 docs from retriever
    top_docs = rerank_with_llm(
        query,
        retrieved,
        top_k=top_k,
        preferred_files=preferred_files,
    )
    return top_docs


# This is what the RAG chain uses:
#retriever_runnable = RunnableLambda(lambda q: dense_with_rerank(q))

# ----------------- Prompts for rewriting -----------------

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You rewrite the user's latest message into a single, standalone question "
                "for use with a retrieval system over energy-related documents.\n"
                "\n"
                "Context:\n"
                "- The chat history is about energy projects (e.g. wind, solar, heat pumps, "
                "  industrial decarbonisation, policy reports, financial models).\n"
                "- The user may refer to previous projects or documents with pronouns like "
                "  'it', 'they', 'this project', 'that study', 'this document'.\n"
                "\n"
                "Requirements:\n"
                "1. Use the SAME LANGUAGE as the user's latest message.\n"
                "2. Use the chat history ONLY to replace ambiguous references like "
                "   'it', 'they', 'this project', 'that document' with explicit terms "
                "   (e.g. 'the 10 MW onshore wind farm in Spain from the feasibility study').\n"
                "3. Preserve ALL explicit technical and financial details from the latest message "
                "   and history (e.g. capacities, units, years, technologies, project names), "
                "   but do NOT invent new ones.\n"
                "4. Do NOT introduce new facts, numbers, metrics (e.g. IRR, NPV, LCOE) or assumptions "
                "   that were not mentioned.\n"
                "5. Do NOT change units or numerical values (kW, kWh, MW, MWh, tCO2, € values, etc.).\n"
                "6. If the latest user message is already a standalone question, return it unchanged.\n"
                "7. Do NOT answer the question.\n"
                "8. Return ONLY the rewritten question, without any explanation, preamble, or quotes."
            ),
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)



contextualize_q_chain = rewrite_prompt | rerank_llm | StrOutputParser()

def maybe_rewrite(question: str, chat_history):
    if not chat_history or len(chat_history) < 2:
        return question
    return contextualize_q_chain.invoke({"question": question, "chat_history": chat_history[:]})

# ----------------- Formatting helpers -----------------

def extract_filenames(docs, max_files: int = 3) -> list[str]:
    """
    Extract up to `max_files` unique filenames from retrieved docs,
    preserving order (first = most relevant).
    """
    names: list[str] = []
    for d in docs:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        filename = md.get("filename") or (os.path.basename(src) if src else "unknown")
        names.append(filename)

    # dedupe while preserving order
    unique = list(dict.fromkeys(names))
    return unique[:max_files]


def format_docs(docs):
    """
    Format retrieved docs into a text block with [Source: <filename>] headers.
    """
    formatted = []
    for d in docs:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        filename = md.get("filename") or (os.path.basename(src) if src else "unknown")
        formatted.append(f"[Source: {filename}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)

# ----------------- Chat history (shared with agent / UI) -----------------

_histories: Dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _histories:
        _histories[session_id] = InMemoryChatMessageHistory()
    return _histories[session_id]

def reset_history(session_id: str) -> None:
    """
    Clear the conversation history for a given session_id.
    """
    sid = session_id or "default-session"
    if sid in _histories:
        try:
            _histories[sid].clear()
        except Exception:
            _histories[sid] = InMemoryChatMessageHistory()

def sync_history_from_messages(session_id: str, messages: List[Dict[str, Any]]) -> None:
    """
    Rebuild InMemoryChatMessageHistory from a list of
    {'role': 'user'|'assistant', 'content': str} messages.
    """
    hist = get_history(session_id)
    try:
        hist.clear()
    except Exception:
        _histories[session_id] = InMemoryChatMessageHistory()
        hist = _histories[session_id]

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if not content:
            continue

        if role == "user":
            hist.add_message(HumanMessage(content=content))
        elif role == "assistant":
            hist.add_message(AIMessage(content=content))

# ----------------- Usage tracking (RAG-side only) -----------------

USAGE_TOTALS = defaultdict(float)  # keys: prompt_tokens, completion_tokens, total_tokens, total_cost
SESSION_TOTALS = defaultdict(lambda: defaultdict(float))  # per-session buckets

DEFAULT_SESSION_ID = os.getenv("SESSION_ID", "default-session")

def run_rag(
    question: str,
    session_id: str | None = None,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Core RAG operation:
    - optionally rewrites to a standalone question using chat history
    - retrieves & reranks documents
    - formats context and builds filename + sources metadata
    - tracks token usage for the rewriting / rerank LLM

    Returns a dict with keys:
      - question: rewritten standalone question (str)
      - context: joined document chunks (str)
      - filenames: list[str]
      - sources: list[dict] with {file, full_path, chunk_id, snippet, relevance}
      - used_db: bool
      - confidence: float in [0, 1]
      - avg_relevance: float in [0, 1]
    """
    sid = session_id or DEFAULT_SESSION_ID

    with get_openai_callback() as cb:
        hist = _histories.get(sid)
        chat_history = hist.messages if hist is not None else []

        # 1) Rewrite to standalone question (if needed)
        standalone = maybe_rewrite(question, chat_history)

        # 2) Retrieve & rerank with soft document continuity
        #    Use previously active filenames for this session as a soft preference.
        preferred_files = ACTIVE_FILES.get(sid)

        docs = dense_with_rerank(
            standalone,
            top_k=top_k,
            preferred_files=preferred_files,
        )
        context_text = format_docs(docs)
        # filenames already capped to 3 by default
        filenames = extract_filenames(docs)  # or extract_filenames(docs, max_files=3)

        # Update active files for this session for future turns
        ACTIVE_FILES[sid] = filenames


    # ---- token & cost accounting (for RAG pipeline only) ----
    cost = compute_cost(cb.prompt_tokens, cb.completion_tokens, model=MODEL)

    USAGE_TOTALS["prompt_tokens"]     += cb.prompt_tokens
    USAGE_TOTALS["completion_tokens"] += cb.completion_tokens
    USAGE_TOTALS["total_tokens"]      += cb.total_tokens
    USAGE_TOTALS["total_cost"]        += cost

    if sid not in SESSION_TOTALS:
        SESSION_TOTALS[sid] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

    SESSION_TOTALS[sid]["prompt_tokens"]     += cb.prompt_tokens
    SESSION_TOTALS[sid]["completion_tokens"] += cb.completion_tokens
    SESSION_TOTALS[sid]["total_tokens"]      += cb.total_tokens
    SESSION_TOTALS[sid]["total_cost"]        += cost

    print(
        f"[RAG core / SESSION {sid}] "
        f"Prompt={cb.prompt_tokens}  Completion={cb.completion_tokens}  "
        f"Total={cb.total_tokens}  Cost=${cost:.6f}"
    )

    # ---- relevance & confidence ----
    scores = _doc_relevance_scores(standalone, docs)
    max_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Confidence: how well the best doc matches the question (0–1)
    confidence = float(max_score)

    # Decide whether DB was used at all (your chosen heuristic)
    used_db = len(docs) > 0

    # Build structured "sources" metadata
    sources = []
    max_source_docs = min(len(docs), 10)  # or 5 if you want it very tight

    for i, d in enumerate(docs[:max_source_docs]):
        md = d.metadata or {}
        src = md.get("source", "unknown")
        filename = md.get("filename") or (os.path.basename(src) if src else "unknown")
        sources.append(
            {
                "file": filename,
                "full_path": src,
                "chunk_id": md.get("chunk_id"),
                "snippet": d.page_content[:280],
                "relevance": scores[i] if i < len(scores) else None,
            }
        )

    return {
        "question": standalone,
        "context": context_text,
        "filenames": filenames,
        "sources": sources,
        "used_db": used_db,
        "confidence": confidence,
        "avg_relevance": avg_score,
    }

# ----------------- Tool: rag_lookup (pure retrieval) -----------------

@tool("rag_lookup")
def rag_lookup(question: str, session_id: str | None = None) -> str:
    """
    Retrieve context from the local document database (RAG).

    Input:
      - question: natural language question as a string.

    Output (JSON string) with keys:
      - question: rewritten standalone question (str)
      - context: joined document chunks (str)
      - filenames: list[str]
      - sources: list[dict] with {file, full_path, chunk_id, snippet}
      - used_db: bool
      - confidence: float in [0, 1]
    """
    out = run_rag(question, session_id=session_id)
    print(f"[RAG TOOL] called for session={session_id}, used_db={out['used_db']}, filenames={out['filenames']}")
    return json.dumps(out, ensure_ascii=False)

__all__ = [
    "run_rag",
    "rag_lookup",
    "USAGE_TOTALS",
    "SESSION_TOTALS",
    "DEFAULT_SESSION_ID",
    "reset_history",
    "sync_history_from_messages",
]

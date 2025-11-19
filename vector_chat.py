import os
import json
from collections import defaultdict
from operator import itemgetter

from dotenv import load_dotenv
load_dotenv()

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_community.callbacks import get_openai_callback
from langchain_core.tools import tool

# 👉 our own vector / embeddings backend
from vector_store import (
    vectorstore,       # PGVector instance
    BM25_CORPUS,       # in-memory docs for BM25
    MODEL,             # chat model name ("gpt-4o-mini", etc.)
    compute_cost,      # cost calculator
)

# ----------------- Basic config -----------------

TOP_K = int(os.getenv("TOP_K", "5"))

# ----------------- RRF fusion -----------------

def rrf_fuse(result_lists, weights=None, k=TOP_K, c=60, id_key="doc_id"):
    """
    Reciprocal Rank Fusion to combine multiple ranked lists (bm25 + dense).
    """
    weights = weights or [1.0] * len(result_lists)
    scores, by_id = {}, {}
    for i, docs in enumerate(result_lists):
        w = weights[i]
        for rank, d in enumerate(docs):
            did = (d.metadata or {}).get(id_key) or d.page_content
            by_id[did] = d
            scores[did] = scores.get(did, 0.0) + w * (1.0 / (c + rank + 1))
    return [
        by_id[did]
        for did, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ]

# ----------------- Retrievers: dense + BM25 + hybrid -----------------

# Dense retriever (PGVector)
dense = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.7},
)

# BM25 retriever – only build if we actually have documents
if BM25_CORPUS:
    bm25 = BM25Retriever.from_documents(BM25_CORPUS, k=TOP_K)
else:
    print("BM25_CORPUS is empty; BM25 disabled until ingestion runs. Using dense-only retriever.")

    class DenseOnlyBM25:
        def __init__(self, dense_ret):
            self.dense = dense_ret

        def invoke(self, query: str):
            return self.dense.invoke(query)

        def get_relevant_documents(self, query: str):
            return self.dense.get_relevant_documents(query)

    bm25 = DenseOnlyBM25(dense)


class HybridRRFRetriever:
    """
    Hybrid retriever that fuses BM25 and dense (PGVector) results with RRF.
    """
    def __init__(self, bm25_ret, dense_ret, weights=(0.4, 0.6), k=TOP_K, c=60, id_key="doc_id"):
        self.bm25 = bm25_ret
        self.dense = dense_ret
        self.weights = list(weights)
        self.k = k
        self.c = c
        self.id_key = id_key

    def _retrieve(self, query: str):
        bm25_docs  = self.bm25.invoke(query)
        dense_docs = self.dense.invoke(query)
        return rrf_fuse(
            [bm25_docs, dense_docs],
            weights=self.weights,
            k=self.k,
            c=self.c,
            id_key=self.id_key,
        )

    def invoke(self, query: str, **kwargs):
        return self._retrieve(query)

    def get_relevant_documents(self, query: str):
        return self._retrieve(query)


# Instantiate the hybrid retriever
hybrid = HybridRRFRetriever(bm25, dense, weights=(0.4, 0.6), k=TOP_K, c=60, id_key="doc_id")

# ----------------- LLM + prompts -----------------

llm = ChatOpenAI(
    temperature=0.0,
    top_p=0.9,
    model=MODEL,
    streaming=True,
    stream_usage=True,
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user message, rewrite the message into a standalone question. "
            "Only return the rewritten question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

def format_docs(docs):
    """
    Format retrieved docs into a text block with [Source: <filename>] headers.
    """
    formatted = []
    for d in docs:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        filename = md.get("filename") or os.path.basename(src) if src else "unknown"
        formatted.append(f"[Source: {filename}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write in a very detailed, elegant tone, formatting all answers in Markdown. "
            "You are a helpful assistant. If the user expresses gratitude (e.g., 'thank you' or 'thanks'), "
            "reply with a polite, friendly response (e.g., 'You're very welcome!'). "
            "Otherwise, use ONLY the provided context to answer accurately and concisely. "
            "Retrieve data only from the database and at the end of your answer, always append a tag "
            "in the format: [File: <filenames>]. "
            "If the question is not answerable from the database context, try to answer but mention "
            "it is not from the database, then append [File: None]."
        ),
        MessagesPlaceholder("chat_history"),
        ("system", "Context:\n{context}"),
        ("human", "{question}"),
    ]
)

contextualize_q_chain = rewrite_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write in a very detailed, elegant tone, formatting all answers in Markdown. "
            "You are a helpful assistant. If the user expresses gratitude (e.g., 'thank you' or 'thanks'), "
            "reply with a polite, friendly response (e.g., 'You're very welcome!'). "
            "Otherwise, use ONLY the provided context to answer accurately and concisely. "
            "Retrieve data only from the database and at the end of your answer, always append a tag "
            "in the format: [File: <filenames>]. "
            "If the question is not answerable from the database context, try to answer but mention "
            "it is not from the database, then append [File: None]."
        ),
        ("system", "Context:\n{context}"),
        ("human", "{question}"),
    ]
)

def maybe_rewrite(question, chat_history):
    if not chat_history or len(chat_history) < 2:
        return question
    return contextualize_q_chain.invoke({"question": question, "chat_history": chat_history[:]})

# ----------------- RAG chain -----------------

retriever_runnable = RunnableLambda(lambda q: hybrid.invoke(q))
format_runnable = RunnableLambda(format_docs)

rag_chain = (
    RunnablePassthrough()
    .assign(standalone_question=lambda x: maybe_rewrite(x["question"], x.get("chat_history", [])))
    .assign(context=(itemgetter("standalone_question") | retriever_runnable | format_runnable))
    .assign(question=itemgetter("standalone_question"))
    | answer_prompt
    | llm
    | StrOutputParser()
)

# ----------------- Chat history (memory) -----------------

_histories: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _histories:
        _histories[session_id] = InMemoryChatMessageHistory()
    return _histories[session_id]

chat_chain = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

SESSION_ID = os.getenv("SESSION_ID", "default-session")

def is_trivial(query: str) -> bool:
    q = (query or "").strip().lower()
    return len(q.split()) <= 2 and q in {
        "hi", "hey", "hello", "yo", "sup", "good morning", "good evening"
    }

# ----------------- Usage tracking -----------------

USAGE_TOTALS = defaultdict(float)  # keys: prompt_tokens, completion_tokens, total_tokens, total_cost
SESSION_TOTALS = defaultdict(lambda: defaultdict(float))  # per-session buckets


def chat(message: str, history, session_id: str | None = None) -> str:
    """
    Main chat entrypoint used by the frontend.

    - `session_id` controls which conversation memory is used.
    - If None, falls back to the global SESSION_ID.
    - `history` is only for UI; memory is kept server-side in _histories.
    """
    if is_trivial(message):
        return "Hi I am EMMETT.ai! How can I assist you today?"

    sid = session_id or SESSION_ID

    with get_openai_callback() as cb:
        result = chat_chain.invoke(
            {"question": message},
            config={
                "callbacks": [cb],
                "configurable": {"session_id": sid},
            },
        )

    cost = compute_cost(cb.prompt_tokens, cb.completion_tokens, model=MODEL)

    # Global totals
    USAGE_TOTALS["prompt_tokens"]     += cb.prompt_tokens
    USAGE_TOTALS["completion_tokens"] += cb.completion_tokens
    USAGE_TOTALS["total_tokens"]      += cb.total_tokens
    USAGE_TOTALS["total_cost"]        += cost

    # Per-session totals
    SESSION_TOTALS[sid]["prompt_tokens"]     += cb.prompt_tokens
    SESSION_TOTALS[sid]["completion_tokens"] += cb.completion_tokens
    SESSION_TOTALS[sid]["total_tokens"]      += cb.total_tokens
    SESSION_TOTALS[sid]["total_cost"]        += cost

    print(
        f"[SESSION {sid}] Prompt={cb.prompt_tokens}  Completion={cb.completion_tokens}  "
        f"Total={cb.total_tokens}  Cost=${cost:.6f}  ||  "
        f"[TOTAL USAGE] Tokens={USAGE_TOTALS['total_tokens']:.0f}  "
        f"Cost=${USAGE_TOTALS['total_cost']:.4f}"
    )

    return result


def reset_history(session_id: str = SESSION_ID):
    """
    Clear the conversation history for a given session_id.
    Used when the UI starts a 'New conversation'.
    """
    sid = session_id or SESSION_ID
    if sid in _histories:
        try:
            _histories[sid].clear()
        except Exception:
            _histories[sid] = InMemoryChatMessageHistory()


# ----------------- RAG helper + tool (optional) -----------------

def rag_answer(question: str, session_id: str = SESSION_ID, top_k: int = TOP_K):
    """
    Returns a structured result for agents or debugging.
    """
    hist = _histories.get(session_id)
    chat_history = hist.messages if hist is not None else []
    standalone = maybe_rewrite(question, chat_history)

    docs = hybrid.invoke(standalone)
    context_text = format_docs(docs)
    final = (answer_prompt | llm | StrOutputParser()).invoke(
        {"context": context_text, "question": standalone}
    )

    sources = []
    for d in docs[:top_k]:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        filename = md.get("filename") or os.path.basename(src) if src else "unknown"
        sources.append(
            {
                "file": filename,
                "full_path": src,
                "chunk_id": md.get("chunk_id"),
                "snippet": d.page_content[:280],
            }
        )

    confidence = min(1.0, len(docs) / 5.0)

    return {
        "answer": final,
        "sources": sources,
        "used_db": True,
        "confidence": confidence,
    }


@tool("rag_lookup")
def rag_lookup(question: str) -> str:
    """
    Retrieve an answer from the local document database.
    Returns a JSON string with: answer, sources, used_db, confidence.
    """
    out = rag_answer(question)
    return json.dumps(out, ensure_ascii=False)


__all__ = ["chat", "USAGE_TOTALS", "SESSION_ID", "reset_history", "rag_lookup", "rag_answer"]


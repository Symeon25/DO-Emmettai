import os
import json
import re
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
from langchain_community.callbacks import get_openai_callback
from langchain_core.tools import tool

from langchain_core.messages import HumanMessage, AIMessage


from vector_store import (
    vectorstore,       # PGVector instance
    MODEL,             # chat model name ("gpt-4o-mini", etc.)
    compute_cost,      # cost calculator
)

# ----------------- Basic config -----------------TOP_K

TOP_K = int(os.getenv("TOP_K", "15"))
BASE_K = max(TOP_K, 20)           # at least 20 candidates
FETCH_K = max(TOP_K * 2, 40)      # at least 40 for MMR diversity

# ----------------- Retrievers: Dense + LLM Reranker -----------------
"""
dense = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": BASE_K, "fetch_k": FETCH_K, "lambda_mult": 0.7},
)
"""
dense = vectorstore.as_retriever( search_kwargs={"k": 20} )


answer_llm = ChatOpenAI(
    temperature=0.0,
    top_p=0.9,
    model=MODEL,
    streaming=True,
    stream_usage=True,
)

rerank_llm = ChatOpenAI(
    temperature=0.0,
    top_p=0.9,
    model=MODEL,
    streaming=False,
    stream_usage=True,
)

def rerank_with_llm(question: str, docs, top_k: int = 5):
    """
    Rerank retrieved chunks using an LLM and return up to `top_k` UNIQUE docs.
    """
    if not docs:
        return []

    context_block = "\n\n---\n\n".join(
        f"[Doc {i}]\n{d.page_content}" for i, d in enumerate(docs)
    )

    prompt = f"""
You are a retrieval reranker.

User question:
{question}

Here are candidate document chunks:

{context_block}

Return the indices (comma-separated) of the TOP {top_k} most relevant [Doc i] chunks.
Only return the indices, e.g.: 0,2,3,5,6
"""

    raw = rerank_llm.invoke(prompt).content

    # --- robustly parse indices from model output ---
    # Extract all integers from the response (e.g. "Top: 0, 1, 5" or "[0, 2, 5]")
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


def dense_with_rerank(query: str, top_k: int = TOP_K):
    retrieved = dense.invoke(query)   # 20 docs from MMR
    top_docs = rerank_with_llm(query, retrieved, top_k=top_k)
    return top_docs


# This is what the RAG chain uses:
retriever_runnable = RunnableLambda(lambda q: dense_with_rerank(q))



# ----------------- LLM + prompts -----------------

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

def extract_filenames(docs):
    names = []
    for d in docs:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        filename = md.get("filename") or (os.path.basename(src) if src else "unknown")
        names.append(filename)
    return list(dict.fromkeys(names))  # dedupe in order


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

"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the provided context to answer accurately and concisely."),
        MessagesPlaceholder("chat_history"),
        ("system", "Context:\n{context}"),
        ("human", "{question}"),
    ]
)
"""

contextualize_q_chain = rewrite_prompt | rerank_llm | StrOutputParser()


"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write in a very detailed, elegant tone, formatting all answers in Markdown. "
            "You are a helpful assistant. If the user expresses gratitude (e.g., 'thank you' or 'thanks'), "
            "reply with a polite, friendly response. "
            "Use the retrieved Context for factual information. "
            "You may also consider the retrieved file names (in the `filenames` field) "
            "when deciding if information is available. "
            "Do NOT include filenames in your answer; the system will handle that."
        ),
        MessagesPlaceholder("chat_history"),
        ("system", "Retrieved files: {filenames}"),  # <-- invisible to user, visible to LLM
        ("system", "Context:\n{context}"),
        ("human", "{question}"),
    ]
)
"""


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant for an energy transition consultancy. "
            "You help analyze and explain energy-related documents such as feasibility studies, "
            "technical reports, contracts, policy documents, financial models, and project proposals.\n\n"
            "Writing style:\n"
            "- Be clear, structured, and professional.\n"
            "- Use Markdown with headings, bullet points, and tables where helpful.\n"
            "- Be concise but thorough: start with a short summary, then add detail.\n\n"
            "Use of context:\n"
            "- Treat the provided Context as your primary source of truth.\n"
            "- Only state detailed facts, figures, or quotes if they appear in the Context.\n"
            "- You may use general domain knowledge for basic explanations, but do NOT invent "
            "specific numbers, project details, or claims not supported by the Context.\n"
            "- If the Context is missing, incomplete, or does not answer the question, say so clearly "
            "and explain what additional information would be needed.\n\n"
            "Retrieved files and filenames:\n"
            "- You may use the retrieved file names (provided as metadata) to infer what type of "
            "information might be available.\n"
            "- Apart from the final Sources line described below, never mention or list filenames "
            "or file paths anywhere else in your answer.\n\n"
            "Sources:\n"
            "- At the end of your answer, IF AND ONLY IF your answer actually uses information "
            "from the document Context, append a final line in the format:\n"
            "  [Sources: filename1, filename2]\n"
            "- If your answer does NOT rely on the document Context (for example, simple greetings, "
            "chit-chat, or very general explanations), DO NOT add a Sources line.\n"
            "- Never mention filenames anywhere else in the answer.\n\n"
            "Energy & technical guidance:\n"
            "- Be careful with units (kW vs kWh, MW vs MWh, tCO2 vs kgCO2, etc.). "
            "Always state units explicitly when using numbers.\n"
            "- If you derive numbers (e.g., annual energy from hourly data, CO2 savings from kWh), "
            "show your calculation steps and assumptions.\n"
            "- If there are uncertainties or assumptions, explicitly list them.\n\n"
            "Behavior:\n"
            "- If the user expresses gratitude (e.g. 'thank you' or 'thanks'), reply with a brief, "
            "friendly response.\n"
            "- If a question is ambiguous, briefly state the assumptions you are making.\n"
            "- If something might be interpreted as legal, financial, or regulatory advice, add a short "
            "disclaimer that this is an AI-generated analysis and should be checked by a qualified expert."
        ),
        MessagesPlaceholder("chat_history"),

        # Hidden RAG metadata
        (
            "system",
            "You will receive a list of retrieved filenames as metadata. "
            "They help you reason about what might be in the documents."
        ),
        ("system", "FILENAME_METADATA: {filenames}"),

        ("system", "Context from retrieved documents:\n{context}"),
        ("human", "{question}"),
    ]
)



def maybe_rewrite(question, chat_history):
    if not chat_history or len(chat_history) < 2:
        return question
    return contextualize_q_chain.invoke({"question": question, "chat_history": chat_history[:]})

# ----------------- RAG chain -----------------

#retriever_runnable = RunnableLambda(lambda q: hybrid.invoke(q))
format_runnable = RunnableLambda(format_docs)


rag_chain = (
    RunnablePassthrough()
    .assign(
        standalone_question=lambda x: maybe_rewrite(
            x["question"], x.get("chat_history", [])
        )
    )
    # Retrieve docs
    .assign(docs=itemgetter("standalone_question") | retriever_runnable)
    # Extract filenames
    .assign(filenames=lambda x: extract_filenames(x["docs"]))
    # Format docs into clean context text
    .assign(context=itemgetter("docs") | format_runnable)
    # Pass rewritten question
    .assign(question=itemgetter("standalone_question"))
    # LLM answer
    .assign(
        answer=answer_prompt
        | answer_llm
        | StrOutputParser()
    )
    | itemgetter("answer")
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
        "hi", "hey", "hello", "yo", "sup", "good morning", "good evening", "Hallo" 
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

def sync_history_from_messages(session_id: str, messages: list[dict]) -> None:
    """
    Rebuild LangChain's InMemoryChatMessageHistory for a session_id
    from a list of {'role': 'user'|'assistant', 'content': str} messages.
    """
    hist = get_history(session_id)
    try:
        hist.clear()
    except Exception:
        # If clear() doesn't exist for some reason, just overwrite
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
        # ignore other roles (system, tool) for now, or handle them if you store them

# ----------------- RAG helper + tool -----------------
# Agent tool

def rag_answer(question: str, session_id: str = SESSION_ID, top_k: int = TOP_K):
    """
    Returns a structured result for agents or debugging.
    """
    hist = _histories.get(session_id)
    chat_history = hist.messages if hist is not None else []
    standalone = maybe_rewrite(question, chat_history)

    docs = dense_with_rerank(standalone, top_k=top_k)
    context_text = format_docs(docs)

    final = (answer_prompt | answer_llm | StrOutputParser()).invoke(
        {
            "context": context_text,
            "question": standalone,
            "chat_history": chat_history,  # <-- IMPORTANT
        }
    )

    sources = []
    for d in docs[:top_k]:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        filename = md.get("filename") or (os.path.basename(src) if src else "unknown")
        sources.append(
            {
                "file": filename,
                "full_path": src,
                "chunk_id": md.get("chunk_id"),
                "snippet": d.page_content[:280],
            }
        )

    confidence = min(1.0, len(docs) / float(top_k or 1))

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


__all__ = ["chat", "USAGE_TOTALS", "SESSION_ID", "reset_history", "rag_lookup", "rag_answer", "sync_history_from_messages"]



# Agent.py  (LangGraph agent using rag_core.rag_lookup)

from typing import Dict
from typing_extensions import Annotated, TypedDict
import operator
import os

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    AnyMessage,
    AIMessage,
)

# ✅ import the new RAG tool from rag_core
from Rag_files.rag_core import rag_lookup, SESSION_TOTALS, USAGE_TOTALS
from Rag_files.vector_store import compute_cost, MODEL as COST_MODEL
from langchain_community.callbacks import get_openai_callback


load_dotenv()

# --------- Model ---------
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=model_name, temperature=0)

# --------- Tools---------
tools = [rag_lookup]
tools_by_name: Dict[str, callable] = {t.name: t for t in tools}
model_with_tools = llm.bind_tools(tools)

# --------- State ---------
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    session_id: str
# --------- LLM node ---------
def llm_call(state: MessagesState):
    """
    Main chat LLM node.

    - For document/project-specific questions, it should call `rag_lookup` before answering.
    - For generic FAQ / concept questions, it may answer directly.
    - For greetings / thanks / tool-management, it should answer directly.
    - It also owns the full answer style and we track token/cost usage per session.
    """
    sid = state.get("session_id", "default-session")
    with get_openai_callback() as cb:
        ai_msg = model_with_tools.invoke(
            [
                SystemMessage(
                    content=(
                        # === Role & domain ===
                        "You are an AI assistant for an energy transition consultancy. "
                        "You analyze and explain energy-related documents such as feasibility studies, "
                        "technical reports, contracts, policy documents, financial models, and project proposals.\n\n"

                        # === Writing style ===
                        "Writing style:\n"
                        "- Be clear, structured, and professional.\n"
                        "- Always respond in Markdown with headings, bullet points, and tables where helpful.\n"
                        "- Be concise but thorough: start with a short summary, then add detail.\n\n"

                        # === Use of tools & context (RAG output schema) ===
                        "RAG tool & context:\n"
                        "- You have a tool `rag_lookup` that returns JSON with the fields "
                        "`question`, `context`, `filenames`, `sources`, `used_db`, and `confidence`.\n"
                        "- Treat `context` as your primary source of truth for detailed facts, figures, and quotes.\n"
                        "- Only state detailed numbers or project-specific claims when they are supported by `context`.\n"
                        "- You may use general domain knowledge for high-level background, but do NOT invent "
                        "specific numbers, project details, or document-based claims that are not supported by `context`.\n"
                        "- If `used_db` is false or the context is clearly insufficient, say this explicitly and "
                        "explain what additional information would be needed.\n\n"

                        # === Filenames / sources handling ===
                        "Filenames and sources:\n"
                        "- `filenames` and `sources` help you reason about which documents were used; they are metadata only.\n"
                        "- At the end of your answer, IF AND ONLY IF your answer relies on information from the document context, "
                        "append a final line in the format:\n"
                        "  [Sources: filename1, filename2]\n"
                        "- Use ONLY the `filenames` field from the JSON for this line.\n"
                        "- Never mention filenames or file paths elsewhere in the answer.\n"
                        "- If you answer purely from general knowledge (no document context), do NOT add a Sources line.\n\n"

                        # === Document continuity ===
                        "Document continuity:\n"
                        "- If earlier in the conversation you answered based on a particular document (as indicated by `filenames`), "
                        "and the user asks a follow-up question without mentioning a new document or project, treat that document as "
                        "the default context.\n"
                        "- However, if the new RAG context clearly indicates that a different document is more relevant to the latest "
                        "question, use that instead. Prefer continuity, but do not ignore obviously more relevant documents.\n\n"

                        # === Energy & technical calculations ===
                        "Energy, units, and calculations:\n"
                        "- Be careful with units (kW vs kWh, MW vs MWh, tCO2 vs kgCO2, etc.) and always state units explicitly.\n"
                        "- When you derive numbers (e.g., annual energy, CO2 savings, financial metrics), show the calculation "
                        "steps and assumptions in simple Markdown-friendly formatting.\n"
                        "- Use clear plain-text math such as:\n"
                        "  - `x = (a - b) / c`\n"
                        "  - `P = V * I`\n"
                        "  - `E = m * c^2`\n"
                        "- Show multi-step calculations line by line (e.g. `Step 1`, `Step 2`, `Step 3`).\n"
                        "- Round numbers only when appropriate and mention key assumptions.\n\n"

                        # === General behavior ===
                        "General behavior:\n"
                        "- If the user expresses gratitude (e.g. 'thank you', 'thanks'), reply with a brief, friendly response.\n"
                        "- If a question is ambiguous, briefly state the assumptions you are making before answering.\n"
                        "- If something might be interpreted as legal, financial, or regulatory advice, add a short disclaimer "
                        "that this is AI-generated analysis and should be checked by a qualified expert.\n\n"

                        # === Tool usage policy (UPDATED) ===
                        "Tool usage policy:\n"
                        "1. Default behavior – always use `rag_lookup` for real questions:\n"
                        "   - For ANY user query that is not just a greeting, pure chit-chat, a small thank-you, or a question "
                        "     about tools/how you work, you MUST first call the `rag_lookup` tool before answering.\n"
                        "   - This includes both document-specific and generic domain questions such as:\n"
                        "     • 'What is NPV?'\n"
                        "     • 'Explain capacity factor.'\n"
                        "     • 'What are the assumptions in this feasibility study?'\n"
                        "     • 'Summarize the main findings of the report.'\n"
                        "   - When in doubt, CALL `rag_lookup`.\n\n"
                        "   After calling `rag_lookup`:\n"
                        "   - Read the JSON and treat `context` as your primary source of truth for detailed facts and numbers.\n"
                        "   - If `used_db` is false, `context` is empty, or `confidence` is clearly low, explicitly say that the "
                        "     documents did not provide sufficient information and then you may fall back to general domain knowledge.\n\n"

                        "2. When you may answer WITHOUT `rag_lookup`:\n"
                        "   You may answer directly (no RAG call) ONLY if the message is one of these:\n"
                        "   - Greetings ('hi', 'hello', 'good morning', etc.).\n"
                        "   - Pure chit-chat not asking for information or analysis.\n"
                        "   - Short thank-you messages ('thanks', 'thank you', 'great, that helps').\n"
                        "   - Questions about your tools or behavior (e.g. 'what tools do you have?', 'how do you work?').\n\n"

                        "3. Multiple tools (future-proofing):\n"
                        "   - In the future, additional tools may be provided.\n"
                        "   - For any information or analysis request, call `rag_lookup` first, then other tools if needed.\n"
                        "   - For pure calculator-style tasks, you may use an appropriate calculator tool AFTER `rag_lookup` has "
                        "     been called (unless the user message is clearly only a math calculation).\n\n"

                        "4. Order of operations:\n"
                        "   - Step 1: Check if the message is greeting / thanks / chit-chat / tool-management → if yes, answer directly.\n"
                        "   - Step 2: Otherwise, you MUST call `rag_lookup`.\n"
                        "   - Step 3: Read the JSON from `rag_lookup` and use `context` for facts and figures.\n"
                        "   - Step 4: If another tool is needed, call it AFTER `rag_lookup`.\n"
                        "   - Step 5: Produce the final Markdown answer, using document context where available.\n\n"


                         # === Answer structure & database indicator ===
                        "Answer structure:\n"
                        "- At the VERY TOP of every non-chit-chat answer, add a one-line indicator of whether the answer is based on "
                        "document database context or on general knowledge.\n"
                        "- Use the `used_db` and `confidence` fields from the `rag_lookup` JSON to decide which variant to use:\n"
                        "  • If `used_db` is true AND `confidence` is medium or high AND `context` is non-empty, start with:\n"
                        "    `Context source: ✅ Based on your document database.`\n"
                        "  • If `used_db` is false OR `context` is empty OR `confidence` is clearly low, start with:\n"
                        "    `Context source: ⚠️ No strong match in your document database; answering mainly from general knowledge.`\n"
                        "- After this first line, add a blank line and THEN continue with your normal Markdown answer "
                        "(e.g. a title like `# Characteristics of High-Risk Customers for Bank Churning`, summary, details, etc.).\n"
                        "- Do NOT include filenames in this first line. Filenames are only used in the final `[Sources: ...]` line if applicable.\n\n"
                        
                        "- Never show the raw JSON from any tool in your answer.\n"
                    )

                )

            ] + state["messages"]
        )
        # ---- usage accounting for the Agent LLM ----
        cost = compute_cost(cb.prompt_tokens, cb.completion_tokens, model=COST_MODEL)

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

        # Log to terminal with session id (which encodes the user)
        print(
            f"[AGENT LLM / SESSION {sid}] "
            f"Prompt={cb.prompt_tokens}  Completion={cb.completion_tokens}  "
            f"Total={cb.total_tokens}  Cost=${cost:.6f}"
        )

    return {
        "messages": [ai_msg],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "session_id": sid,
    }


# --------- Tool node ---------
def tool_node(state: MessagesState):
    """
    Executes tool calls (here: rag_lookup) and returns ToolMessages.
    Also passes session_id into the tool so RAG usage is tracked per session.
    """
    last = state["messages"][-1]
    results = []
    sid = state.get("session_id", "default-session")

    for tc in getattr(last, "tool_calls", []) or []:
        tool_fn = tools_by_name[tc["name"]]
        args = dict(tc["args"])  # copy model-proposed args
        # ensure session_id is present for rag_lookup
        if "session_id" not in args:
            args["session_id"] = sid

        observation = tool_fn.invoke(args)
        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tc["id"],
            )
        )
    return {"messages": results, "session_id": sid}


# --------- Router ---------
def should_continue(state: MessagesState):
    """
    If the LLM requested a tool_call, go to tool_node, otherwise end.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None) and state.get("llm_calls", 0) == 1:
        return "tool_node"
    return END

# --------- Build graph ---------
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

rag_agent = agent_builder.compile()

# Optional: small helper to run it from plain text
def run_agent(user_input: str):
    msgs = [HumanMessage(content=user_input)]
    result = rag_agent.invoke({"messages": msgs, "llm_calls": 0, "session_id": "debug-session"})
    return result["messages"]


def agent_chat(message: str, history: list[dict], session_id: str | None = None) -> str:
    """
    Adapter used by Streamlit.

    - `message`: the latest user input (string).
    - `history`: Streamlit-style history: [{"role": "user"|"assistant", "content": str}, ...]
    - `session_id`: we can pass it as context to the agent (optional).
    """
    msgs: list[AnyMessage] = []

    # Optional: give the agent a hint about session id (not required, but nice)
    if session_id:
        msgs.append(
            SystemMessage(
                content=(
                    f"Conversation session id: {session_id}. "
                    f"Use this only as context, not in the reply."
                )
            )
        )

    # Convert Streamlit-style messages into LangChain messages
    for m in history:
        role = m.get("role")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))

    # Add the current user message
    msgs.append(HumanMessage(content=message))
    
    # Run the LangGraph agent with explicit session_id
    sid = session_id or "default-session"
    result = rag_agent.invoke({"messages": msgs, "llm_calls": 0, "session_id": sid})

    # Get the last AI message from the result
    last_ai = None
    for m in reversed(result["messages"]):
        if getattr(m, "type", None) == "ai":
            last_ai = m
            break

    return last_ai.content if last_ai is not None else ""

if __name__ == "__main__":
    msgs = run_agent("Summarize the main conclusions of the feasibility study.")
    for m in msgs:
        try:
            m.pretty_print()
        except Exception:
            print(type(m), m)

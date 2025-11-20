import os
import time
import traceback
import streamlit as st

# ---- Import your existing backend without altering it ----
from vector_chat import (
    chat,
    USAGE_TOTALS,
    SESSION_ID,
    reset_history,
)

from vector_store import (
        add_chunks_to_vectorstore,
        embeddings,
        split_docs,
        COLLECTION
    )

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,                    
    UnstructuredPowerPointLoader, 
    UnstructuredExcelLoader, 
    DataFrameLoader,     
)
import pandas as pd

import base64

def load_logo_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"

logo_data = load_logo_base64("logo/LogoAI2.png")



# ------------------------- Page config -------------------------
st.set_page_config(
    page_title="EMMETT.ai",
    page_icon="💬",
    layout="wide",
)

# ------------------------- Login -------------------------
APP_USERNAME = os.getenv("APP_USERNAME")   # default optional
APP_PASSWORD = os.getenv("APP_PASSWORD")             # no default on purpose

if "authed" not in st.session_state:
    st.session_state.authed = False
if "user" not in st.session_state:
    st.session_state.user = None

def require_login():
    if st.session_state.authed:
        return

    #st.image("logo/LogoAI2.png", width=300)
    st.markdown(
    f"<img src='{logo_data}' style='width:300px;'>",
    unsafe_allow_html=True
    )

    st.markdown("### 🔒 Please log in to continue")
    with st.form("login-form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if APP_PASSWORD is None:
            # If you forgot to set APP_PASSWORD on the server, fail safely
            st.error("Server login is not configured correctly (APP_PASSWORD missing).")
        elif username == APP_USERNAME and password == APP_PASSWORD:
            st.session_state.authed = True
            st.session_state.user = username
            st.success("Login successful ✅")
            st.rerun()
        else:
            st.error("Invalid username or password ❌")


require_login()
if not st.session_state.authed:
    st.stop()  # prevent rest of UI from rendering until logged in
if "session_id" not in st.session_state:
    st.session_state.conv_counter = 1
    st.session_state.session_id = f"{st.session_state.user}-conv-1"

if "past_conversations" not in st.session_state:
    # Each item: {"id": str, "title": str, "messages": [...]}
    st.session_state.past_conversations = []

if "messages" not in st.session_state:
    # Chat history for the CURRENT conversation
    st.session_state.messages = []

# ------------------------- Sidebar ----------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_upload_files" not in st.session_state:
    st.session_state.last_upload_files = []
if "last_skipped_files" not in st.session_state:
    st.session_state.last_skipped_files = []
if "last_ingest_changed" not in st.session_state:
    st.session_state.last_ingest_changed = []
if "last_ingest_failed" not in st.session_state:
    st.session_state.last_ingest_failed = []



with st.sidebar:
    #st.image("logo/LogoAI2.png", width=200)
    st.markdown(
    f"<img src='{logo_data}' style='width:200px;'>",
    unsafe_allow_html=True
    )

    #st.header("⚙️ Settings")

    #st.subheader("Session")
    #st.text_input("Session ID (from backend)", value=str(SESSION_ID), disabled=True)
    st.subheader("📁 Add documents")

    uploaded_files = st.file_uploader(
        "Drop files to ingest",
        type=["pdf", "docx", "txt", "xlsx", "xls", "pptx", "ppt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    # --- Handle newly uploaded files (this run) ---
    if uploaded_files:
        os.makedirs("data", exist_ok=True)  # BASE_DIR

        saved_files = []
        skipped_files = []

        for f in uploaded_files:
            # Simple check: empty file
            if f.size == 0:
                skipped_files.append(f.name)
                continue

            save_path = os.path.join("data", f.name)
            with open(save_path, "wb") as out:
                out.write(f.read())
            saved_files.append(f.name)

        # Store in session so messages survive rerun
        st.session_state.last_upload_files = saved_files
        st.session_state.last_skipped_files = skipped_files

        # Show ingest button only when we have something to ingest
        if saved_files and st.button("🔄 Upload documents"):
            changed_files = []
            failed_files = []

            try:
                with st.spinner("Indexing documents into the vector store…"):
                    all_docs = []

                    # 1) Load each saved file into Documents
                    for fname in saved_files:
                        path = os.path.join("data", fname)
                        ext = os.path.splitext(path)[1].lower()

                        try:
                            if ext == ".pdf":
                                loader = PyPDFLoader(path)

                            elif ext == ".docx":
                                loader = Docx2txtLoader(path)

                            elif ext == ".txt":
                                loader = TextLoader(path, autodetect_encoding=True)

                            # 📊 Excel (xls/xlsx) via UnstructuredExcelLoader
                            elif ext in [".xlsx", ".xls"]:
                                loader= UnstructuredExcelLoader(path, mode="single")

                            # 📽 PowerPoint (ppt/pptx) via UnstructuredPowerPointLoader
                            elif ext in [".pptx", ".ppt"]:
                                loader = UnstructuredPowerPointLoader(path, mode="single")

                            else:
                                # Unknown/unsupported type
                                failed_files.append(fname)
                                continue

                            docs = loader.load()
                            for d in docs:
                                d.metadata["filename"] = fname      # uploaded file name
                                d.metadata["source"] = path         # optional but useful

                            all_docs.extend(docs)
                            changed_files.append(fname)

                        except Exception as e:
                            failed_files.append(fname)
                            print(f"[WARN] Failed to load {path}: {e}")
                            st.warning(f"Failed to load {fname}: {e}")
                            st.exception(e)


                    # 2) Split into chunks
                    if all_docs:
                        chunks = split_docs(all_docs)

                        # 3) Add chunks directly to PGVector
                        usage = add_chunks_to_vectorstore(chunks)

                        print(
                            f"Indexed files: {', '.join(changed_files)} | "
                            f"Embedded {usage['total_tokens']} tokens (cost ≈ ${usage['total_cost']:.6f})"
                        )


                # Save ingest results in session state so we can show messages after rerun
                st.session_state.last_ingest_changed = changed_files
                st.session_state.last_ingest_failed = failed_files

                # ✅ Update the "uploaded" summary to only show successfully indexed files
                if changed_files:
                    st.session_state.last_upload_files = changed_files
                else:
                    st.session_state.last_upload_files = []

                # Reset uploader so file list disappears, then rerun
                st.session_state.uploader_key += 1
                st.rerun()

            except Exception as e:
                st.error("Error during ingestion")
                st.exception(e)


    # --- Persistent messages (shown even after uploader reset) ---
    if st.session_state.last_upload_files:
        st.success(
            f"Uploaded {len(st.session_state.last_upload_files)} file(s): "
            + ", ".join(st.session_state.last_upload_files)
        )


    if st.session_state.last_skipped_files:
        st.warning(
            "Skipped empty file(s): "
            + ", ".join(st.session_state.last_skipped_files)
        )

    if st.session_state.last_ingest_changed:
        st.info(
            "Upload complete!"
            f"{len(st.session_state.last_ingest_changed)} file(s): "
            + ", ".join(st.session_state.last_ingest_changed)
        )

    if st.session_state.last_ingest_failed:
        st.warning(
            "The following file(s) failed and were removed:\n\n"
            + ", ".join(st.session_state.last_ingest_failed)
        )


    st.divider()

    st.subheader("💬 Conversations")

    #st.markdown(f"**Current conversation ID:** `{st.session_state.session_id}`")

    # Show past conversations with ability to "see" and "load" them
    if st.session_state.past_conversations:
        st.markdown("**Past conversations:**")
        for idx, conv in enumerate(st.session_state.past_conversations):
            # Give the expander a unique key as well (optional but safer)
            with st.expander(f"{conv['title']}", expanded=False):
                # Show messages from that past conversation
                for m in conv["messages"]:
                    role_label = "👤 User" if m["role"] == "user" else "🤖 Assistant"
                    st.markdown(f"**{role_label}:** {m['content']}")

                # 👉 Button to load this conversation as the active one
                if st.button(
                    "Load this conversation",
                    key=f"load_{idx}_{conv['id']}",   # ✅ now guaranteed unique
                ):
                    # 1. Switch current session_id to this past one
                    st.session_state.session_id = conv["id"]

                    # 2. Restore UI messages
                    st.session_state.messages = conv["messages"].copy()

                    # 3. Rerun so the main chat area shows this conversation
                    st.rerun()
    else:
        st.caption("No past conversations yet.")


    # Button to start a new conversation
    if st.button("➕ New conversation"):
        # 1. Save current conversation to history (if non-empty)
        if st.session_state.get("messages"):
            first_msg = next(
                (m for m in st.session_state.messages if m["role"] == "user"),
                None,
            )
            title = (first_msg["content"][:40] + "…") if first_msg else "Untitled"

            st.session_state.past_conversations.append(
                {
                    "id": st.session_state.session_id,
                    "title": title,
                    "messages": st.session_state.messages.copy(),
                }
            )

        # 2. Increment conversation counter and build a new session_id
        st.session_state.conv_counter += 1
        new_session_id = f"{st.session_state.user}-conv-{st.session_state.conv_counter}"
        st.session_state.session_id = new_session_id

        # 3. Clear UI messages
        st.session_state.messages = []

        # 4. Reset backend history for the *new* session id (fresh memory)
        reset_history(st.session_state.session_id)

        # 5. Rerun to clear chat display
        st.rerun()

    st.divider()


    st.subheader("💳 Usage (chat)")
    try:
        st.metric("Prompt tokens", int(USAGE_TOTALS.get("prompt_tokens", 0)))
        st.metric("Completion tokens", int(USAGE_TOTALS.get("completion_tokens", 0)))
        st.metric("Total tokens", int(USAGE_TOTALS.get("total_tokens", 0)))
        st.metric("Total cost", f"${float(USAGE_TOTALS.get('total_cost', 0.0)):.4f}")
        st.caption("Totals reflect this Python process; they reset on restart.")
    except Exception:
        st.info("Usage totals not available.")

# ------------------------- Main area ---------------------------

#st.image("LogoAI2.png", width=300)
st.markdown(
    f"<img src='{logo_data}' style='width:300px;'>",
    unsafe_allow_html=True
    )
# messages already initialized above, so no need to re-check here, but safe:
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Type your question…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            start = time.time()
            response_text = chat(
                user_input,
                st.session_state.messages,
                session_id=st.session_state.session_id,  # 👈 key change
            )
            elapsed = time.time() - start

            placeholder.markdown(response_text, unsafe_allow_html=False)
            st.caption(f"Responded in {elapsed:.2f}s")
        except Exception:
            placeholder.error("Something went wrong while calling the backend.")
            st.exception(traceback.format_exc())
            response_text = ""

    if response_text:
        st.session_state.messages.append({"role": "assistant", "content": response_text})

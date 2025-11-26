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
    WebBaseLoader,    
)
import pandas as pd
import tempfile
import base64

def load_logo_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"

logo_data = load_logo_base64("logo/LogoAI2.png")

#------------LLM for Title--------------------------------------
from openai import OpenAI
client = OpenAI()

def generate_conversation_title(messages):
    """
    Use the LLM to generate a clean 3–4 word title summarizing the conversation.
    Falls back to a truncated first user message if the LLM fails.
    """
    # collect user messages
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]

    # if no user messages yet
    if not user_msgs:
        return "New conversation"

    preview_text = "\n".join(user_msgs[:3])

    prompt = f"""
    Summarize the following conversation topic into a short, professional title.
    - Maximum 4 words
    - No punctuation
    - No quotes
    - Capitalize Each Word

    Conversation:
    {preview_text}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15,
        )
        # ✅ correct access pattern for new client:
        title = resp.choices[0].message.content.strip()
        if title:
            return title
    except Exception:
        # if anything goes wrong, fall back below
        pass

    # fallback: use first user message truncated
    first = user_msgs[0]
    return (first[:40] + "…") if len(first) > 40 else first



# ------------------------- Page config -------------------------
st.set_page_config(
    page_title="EMMETT.ai",
    page_icon="💬",
    layout="wide",
)
# ------------------------- Users-------------------------
import json
import bcrypt
from history_store import load_user_conversations, save_user_conversations


USERS = {}

def _load_users_from_json(path: str = "users.json"):

    global USERS

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        USERS = {}
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = []
    except Exception:
        # Corrupt or invalid JSON -> act as if no users
        USERS = {}
        return

    USERS = {u["username"]: u["password_hash"].encode("utf-8") for u in data}

def verify_user(username: str, password: str) -> bool:
    """Return True if username/password is valid."""
    if not USERS:
        _load_users_from_json()
    pw_hash = USERS.get(username)
    if not pw_hash:
        return False
    return bcrypt.checkpw(password.encode("utf-8"), pw_hash)

def create_user(username: str, password: str, path: str = "users.json"):
    """
    Creates a NEW user in users.json.
    If username already exists -> raise ValueError.
    Stores BOTH:
      - password_plain (for internal use)
      - password_hash (for auth)
    """
    users_list = []

    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                users_list = json.load(f)
            if not isinstance(users_list, list):
                users_list = []
        except Exception:
            # if file is corrupt, start fresh list
            users_list = []

    # ❗ Check duplicate username
    for u in users_list:
        if u.get("username") == username:
            raise ValueError("Username already exists")

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    users_list.append(
        {
            "username": username,
            "password_plain": password,
            "password_hash": pw_hash,
        }
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(users_list, f, indent=2)

    _load_users_from_json(path)


# ------------------------- Login -------------------------

if "authed" not in st.session_state:
    st.session_state.authed = False
if "user" not in st.session_state:
    st.session_state.user = None

def require_login():
    if st.session_state.authed:
        return

    st.markdown(
        f"<img src='{logo_data}' style='width:300px;margin-bottom:25px;'>",
        unsafe_allow_html=True
    )
    #st.markdown("### 🔒 Please log in")

    with st.form("login-form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        mode = st.radio(
            "Action",
            ["Login", "Create new account"],
            horizontal=True,
        )
        submit = st.form_submit_button("Continue")

    if submit:
        if not username or not password:
            st.error("Please enter both a username and a password.")
            return

        # Make sure USERS is loaded
        if not USERS:
            _load_users_from_json()

        if mode == "Login":
            # 👉 One generic error for:
            #    - username does not exist
            #    - password is wrong
            if verify_user(username, password):
                st.session_state.authed = True
                st.session_state.user = username
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Incorrect username or password. Please try again.")

        else:  # "Create new account"
            try:
                create_user(username, password)
            except ValueError:
                # Username already in use
                st.error("This username is already taken. Please choose another one.")
                return

            st.success("Account created and logged in 🎉")
            st.session_state.authed = True
            st.session_state.user = username
            st.rerun()


require_login()
if not st.session_state.authed:
    st.stop()


# ---------- Initialize per-user state ----------
user = st.session_state.user

# Load past conversations from disk on first load for this user
if "past_conversations" not in st.session_state:
    st.session_state.past_conversations = load_user_conversations(user)

# Determine conversation counter based on existing convs
if "conv_counter" not in st.session_state:
    if st.session_state.past_conversations:
        # infer max N from ids like "<user>-conv-N"
        import re
        max_n = 0
        for conv in st.session_state.past_conversations:
            m = re.search(r"-conv-(\d+)$", conv["id"])
            if m:
                n = int(m.group(1))
                max_n = max(max_n, n)
        st.session_state.conv_counter = max_n + 1
    else:
        st.session_state.conv_counter = 1

if "session_id" not in st.session_state:
    st.session_state.session_id = f"{user}-conv-{st.session_state.conv_counter}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------- Main area ---------------------------

#st.image("LogoAI2.png", width=300)
st.markdown(
    f"<img src='{logo_data}' style='width:300px;'>",
    unsafe_allow_html=True
    )
# messages already initialized above, so no need to re-check here, but safe:

st.divider()
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
                session_id=st.session_state.session_id, 
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

        # Optionally auto-save this ongoing conversation as well
        # Find existing entry or create one
        current_id = st.session_state.session_id
        updated = False

        for conv in st.session_state.past_conversations:
            if conv["id"] == current_id:
                conv["messages"] = st.session_state.messages.copy()
                updated = True
                break

        if not updated:
            # Not in list yet, add with a generated title
            title = generate_conversation_title(st.session_state.messages)
            st.session_state.past_conversations.append(
                {
                    "id": current_id,
                    "title": title,
                    "messages": st.session_state.messages.copy(),
                }
            )

        save_user_conversations(user, st.session_state.past_conversations)


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
if "url_input_key" not in st.session_state:
    st.session_state.url_input_key = 0
if "last_file_embedding_info" not in st.session_state:
    st.session_state.last_file_embedding_info = ""
if "last_url_embedding_info" not in st.session_state:
        st.session_state.last_url_embedding_info = ""

with st.sidebar:
    st.markdown(
        f"<img src='{logo_data}' style='width:200px;'>",
        unsafe_allow_html=True
    )
    st.divider()
    st.subheader("📁 Add documents")

    # Simple, stable uploader: NO dynamic key
    uploaded_files = st.file_uploader(
        "Drop your files:",
        type=["pdf", "docx", "txt", "pptx", "ppt"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )

    if uploaded_files and st.button("🔄 Upload documents"):
        all_docs = []
        changed_files = []
        failed_files = []
        skipped_files = []

        try:
            with st.spinner("Indexing documents into the vector store…"):
                for f in uploaded_files:
                    # Skip empty files
                    if f.size == 0:
                        skipped_files.append(f.name)
                        continue

                    ext = os.path.splitext(f.name)[1].lower()

                    # Save to a temporary file (no fixed data folder)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name

                    try:
                        # Choose loader based on extension
                        if ext == ".pdf":
                            loader = PyPDFLoader(tmp_path)
                        elif ext == ".docx":
                            loader = Docx2txtLoader(tmp_path)
                        elif ext == ".txt":
                            loader = TextLoader(tmp_path, autodetect_encoding=True)
                        elif ext in [".pptx", ".ppt"]:
                            loader = UnstructuredPowerPointLoader(tmp_path, mode="single")
                        else:
                            failed_files.append(f.name)
                            continue

                        docs = loader.load()
                        for d in docs:
                            d.metadata["filename"] = f.name
                            d.metadata["source"] = f.name

                        all_docs.extend(docs)
                        changed_files.append(f.name)

                    except Exception as e:
                        failed_files.append(f.name)
                        st.warning(f"Failed to load {f.name}: {e}")
                        st.exception(e)

                    finally:
                        # Clean up temp file
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                # Split into chunks + store in PGVector
                if all_docs:
                    try:
                        chunks = split_docs(all_docs)
                        usage = add_chunks_to_vectorstore(chunks)

                        embedding_msg = (
                            f"Indexed {len(chunks)} chunks from {len(changed_files)} file(s). "
                            f"(≈ {usage['total_tokens']} tokens, cost ≈ ${usage['total_cost']:.6f})"
                        )
                        # Store message so it survives st.rerun()
                        st.session_state.last_file_embedding_info = embedding_msg

                        print(
                            f"Indexed files: {', '.join(changed_files)} | "
                            f"Embedded {usage['total_tokens']} tokens (cost ≈ ${usage['total_cost']:.6f})"
                        )
                    except Exception as e:
                        st.error("Error while adding documents to the vector store.")
                        st.exception(e)
                        # Clear previous info if this run failed
                        st.session_state.last_file_embedding_info = ""

            # Store messages in session (optional)
            st.session_state.last_ingest_changed = changed_files
            st.session_state.last_ingest_failed = failed_files
            st.session_state.last_skipped_files = skipped_files
            st.session_state.last_upload_files = changed_files if changed_files else []
            st.session_state.uploader_key += 1
            st.rerun()

        except Exception as e:
            st.error("Error during ingestion")
            st.exception(e)

    # --- Persistent messages ---

    if st.session_state.last_skipped_files:
        st.warning(
            "Skipped empty file(s): "
            + ", ".join(st.session_state.last_skipped_files)
        )

    if st.session_state.last_ingest_changed:
        st.info(
            "Upload complete! "
            f"{len(st.session_state.last_ingest_changed)} file(s): "
            + ", ".join(st.session_state.last_ingest_changed)
        )

    if st.session_state.last_ingest_failed:
        st.warning(
            "The following file(s) failed and were removed:\n\n"
            + ", ".join(st.session_state.last_ingest_failed)
        )

    if st.session_state.last_file_embedding_info:
        st.info(st.session_state.last_file_embedding_info)

    st.divider()

# ================== WEB PAGE INGESTION ==================
    st.subheader("🌐 Add web pages")

    if "last_ingest_urls_ok" not in st.session_state:
        st.session_state.last_ingest_urls_ok = []
    if "last_ingest_urls_failed" not in st.session_state:
        st.session_state.last_ingest_urls_failed = []

    url_text = st.text_area(
        "Paste your website URL:",
        placeholder="https://example.com/article",
        height=100,
        key=f"url_input_{st.session_state.url_input_key}",
    )

    if st.button("🔗 Upload URL"):
        # Reset status
        st.session_state.last_ingest_urls_ok = []
        st.session_state.last_ingest_urls_failed = []

        # Parse URLs (one per line)
        raw_urls = [u.strip() for u in url_text.splitlines() if u.strip()]
        urls = []
        for u in raw_urls:
            if not (u.startswith("http://") or u.startswith("https://")):
                st.warning(f"Skipping invalid URL (must start with http/https): {u}")
                continue
            urls.append(u)

        if not urls:
            st.error("No valid URLs provided.")
        else:
            all_docs = []
            ok_urls = []
            failed_urls = []

            with st.spinner("Uploading web page…"):
                for url in urls:
                    try:
                        loader = WebBaseLoader(url)
                        docs = loader.load()

                        if not docs:
                            failed_urls.append(url)
                            continue

                        # Tag documents so you know they came from the web
                        for d in docs:
                            d.metadata["source"] = url
                            d.metadata["filename"] = url
                            d.metadata["doc_type"] = "webpage"

                        all_docs.extend(docs)
                        ok_urls.append(url)
                    except Exception as e:
                        failed_urls.append(url)
                        st.warning(f"Failed to load {url}: {e}")

                if all_docs:
                    try:
                        chunks = split_docs(all_docs)
                        usage = add_chunks_to_vectorstore(chunks)

                        embedding_msg = (
                            f"Indexed {len(chunks)} chunks from {len(ok_urls)} URL(s). "
                            f"(≈ {usage['total_tokens']} tokens, cost ≈ ${usage['total_cost']:.6f})"
                        )
                        # Store for later display after rerun
                        st.session_state.last_url_embedding_info = embedding_msg

                    except Exception as e:
                        st.error("Error while adding web pages to the vector store.")
                        st.exception(e)
                        # Clear any previous info if this run failed
                        st.session_state.last_url_embedding_info = ""

            # Save status in session for display
            st.session_state.last_ingest_urls_ok = ok_urls
            st.session_state.last_ingest_urls_failed = failed_urls
        st.session_state.url_input_key += 1
        st.rerun()

    # Show URL ingest status
    if st.session_state.last_ingest_urls_ok:
        st.success(
            "Successfully uploaded web page:\n" + "\n".join(st.session_state.last_ingest_urls_ok)
        )
    if st.session_state.last_ingest_urls_failed:
        st.warning(
            "Failed to upload web page:\n" + "\n".join(st.session_state.last_ingest_urls_failed)
        )
    
    if st.session_state.last_url_embedding_info:
        st.info(st.session_state.last_url_embedding_info)

    if st.session_state.last_ingest_urls_failed:
        st.warning(
            "Failed web pages:\n\n" + "\n".join(st.session_state.last_ingest_urls_failed)
        )

    # ================== END WEB PAGE INGESTION ==================


    st.divider()
    st.subheader("💬 Conversations")

    current_id = st.session_state.session_id

    if st.session_state.past_conversations:
        st.markdown("Past conversations:")

        for idx, conv in enumerate(st.session_state.past_conversations):

            # Title row with load + delete buttons
            c1, c2, c3 = st.columns([0.7, 0.15, 0.15])  

            with c1:
                exp = st.expander(
                    f"{'🟢 ' if conv['id'] == current_id else ''}{conv['title']}",
                    expanded=False,
                )

            with c2:
                load_clicked = st.button(
                    "📄",
                    key=f"load_icon_{idx}_{conv['id']}",
                    help="Load this conversation",
                )

            with c3:
                delete_clicked = st.button(
                    "🗑️",
                    key=f"delete_{idx}_{conv['id']}",
                    help="Delete this conversation",
                )

            # ---- LOAD LOGIC (icon) ----
            if load_clicked:
                st.session_state.session_id = conv["id"]
                st.session_state.messages = conv["messages"].copy()
                st.rerun()

            # ---- DELETE LOGIC ----
            if delete_clicked:
                st.session_state.past_conversations = [
                    c for c in st.session_state.past_conversations
                    if c["id"] != conv["id"]
                ]
                save_user_conversations(user, st.session_state.past_conversations)

                if conv["id"] == current_id:
                    st.session_state.messages = []
                    st.session_state.session_id = (
                        f"{user}-conv-{st.session_state.conv_counter}"
                    )
                    reset_history(st.session_state.session_id)

                st.rerun()

            # ---- CONTENT INSIDE EXPANDER ----
            with exp:
                for m in conv["messages"]:
                    role_label = "👤 User" if m["role"] == "user" else "🤖 Assistant"
                    st.markdown(f"**{role_label}:** {m['content']}")

    else:
        st.caption("No past conversations yet.")

    # ---------- NEW CONVERSATION ----------
    if st.button("➕ New conversation"):
        if st.session_state.get("messages"):
            current_id = st.session_state.session_id

            # Generate a short topic title
            title = generate_conversation_title(st.session_state.messages)

            # Update or append conversation
            updated = False
            for conv in st.session_state.past_conversations:
                if conv["id"] == current_id:
                    conv["title"] = title
                    conv["messages"] = st.session_state.messages.copy()
                    updated = True
                    break

            if not updated:
                st.session_state.past_conversations.append(
                    {
                        "id": current_id,
                        "title": title,
                        "messages": st.session_state.messages.copy(),
                    }
                )

            save_user_conversations(user, st.session_state.past_conversations)

        # Start a fresh conversation
        st.session_state.conv_counter += 1
        new_session_id = f"{user}-conv-{st.session_state.conv_counter}"
        st.session_state.session_id = new_session_id
        st.session_state.messages = []
        reset_history(st.session_state.session_id)
        st.rerun()

    st.divider()


    st.subheader("💳 Usage (chat)")
    try:
        st.metric("Prompt tokens", int(USAGE_TOTALS.get("prompt_tokens", 0)))
        st.metric("Completion tokens", int(USAGE_TOTALS.get("completion_tokens", 0)))
        st.metric("Total tokens", int(USAGE_TOTALS.get("total_tokens", 0)))
        st.metric("Total cost", f"${float(USAGE_TOTALS.get('total_cost', 0.0)):.4f}")
    except Exception:
        st.info("Usage totals not available.")



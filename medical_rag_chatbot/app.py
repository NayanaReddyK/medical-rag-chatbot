"""
app.py
------
Streamlit Medical RAG Chatbot UI.

Run:
    streamlit run app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG — Medical Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.severity-red    { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600; }
.severity-orange { background:#ffedd5; color:#9a3412; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600; }
.severity-yellow { background:#fef9c3; color:#854d0e; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600; }
.severity-green  { background:#dcfce7; color:#166534; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600; }
.source-badge    { background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:4px; font-size:11px; }
.drug-badge      { background:#ede9fe; color:#5b21b6; padding:2px 8px; border-radius:4px; font-size:11px; }
.disclaimer      { font-size:11px; color:#6b7280; border-top:1px solid #e5e7eb; margin-top:12px; padding-top:8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":     [],   # list of {role, content, sources?, intent?}
        "pipeline":     None,
        "initialized":  False,
        "model":        os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "top_k":        5,
        "threshold":    0.35,
        "use_hybrid":   True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loader (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(model: str, use_hybrid: bool, top_k: int, threshold: float):
    from llm.rag_pipeline import RAGPipeline
    p = RAGPipeline(
        model=model,
        use_hybrid=use_hybrid,
        top_k=top_k,
        score_threshold=threshold,
    )
    p.initialize()
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💊 MedRAG")
    st.caption("Medical Q&A powered by MedQuAD + DrugBank")
    st.divider()

    # Model selector
    model_choice = st.selectbox(
    "LLM Model",
    ["gpt-4o-mini", "gpt-4o", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "mock", "groq"],
    index=["gpt-4o-mini", "gpt-4o", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "mock","groq"].index(
        st.session_state.model
    ),
    help="mock = demo mode without an API key",
)
    st.session_state.model = model_choice

    st.divider()
    st.subheader("Retrieval settings")

    top_k = st.slider("Results to retrieve (k)", 1, 10, st.session_state.top_k)
    st.session_state.top_k = top_k

    threshold = st.slider("Score threshold", 0.0, 1.0, st.session_state.threshold, 0.05)
    st.session_state.threshold = threshold

    use_hybrid = st.toggle("Hybrid retrieval (dense + TF-IDF)", st.session_state.use_hybrid)
    st.session_state.use_hybrid = use_hybrid

    st.divider()

    # Quick drug lookup
    st.subheader("Drug lookup")
    drug_query = st.text_input("Search drug name", placeholder="e.g. warfarin")
    if drug_query:
        if st.button("Search interactions", use_container_width=True):
            st.session_state.messages.append({
                "role":    "user",
                "content": f"What are the drug interactions for {drug_query}?",
            })
            st.rerun()

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ For informational purposes only.\n"
        "Always consult a healthcare professional."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────
st.title("Medical RAG Chatbot")
st.caption("Ask about symptoms, conditions, drug interactions, or treatment guidelines.")

# Load pipeline (shows spinner)
with st.spinner("Loading knowledge base and models..."):
    pipeline = load_pipeline(
        model=st.session_state.model,
        use_hybrid=st.session_state.use_hybrid,
        top_k=st.session_state.top_k,
        threshold=st.session_state.threshold,
    )


# ── Render chat history ───────────────────────────────────────────────────────
def _severity_badge(severity: str) -> str:
    s = severity.lower()
    if any(x in s for x in ["contraindicated", "major", "severe"]):
        return f'<span class="severity-red">{severity}</span>'
    elif any(x in s for x in ["serious", "avoid"]):
        return f'<span class="severity-orange">{severity}</span>'
    elif any(x in s for x in ["moderate", "caution"]):
        return f'<span class="severity-yellow">{severity}</span>'
    else:
        return f'<span class="severity-green">{severity}</span>'


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)} retrieved)", expanded=False):
        for i, chunk in enumerate(sources, 1):
            source   = chunk.get("source", "").upper()
            category = chunk.get("category", "")
            score    = chunk.get("score", 0.0)
            ctype    = chunk.get("chunk_type", "")
            meta     = chunk.get("metadata", {})

            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(
                    f'<span class="source-badge">{source}</span> '
                    f'<span class="drug-badge">{category}</span>',
                    unsafe_allow_html=True,
                )
            with col2:
                if ctype == "interaction" and "severity" in meta:
                    from llm.prompt_engine import extract_severity_label
                    label = extract_severity_label(meta["severity"])
                    st.markdown(_severity_badge(label), unsafe_allow_html=True)
                else:
                    st.caption(ctype.replace("_", " ").title())
            with col3:
                st.caption(f"Score: {score:.2f}")

            st.text(chunk.get("text", "")[:300] + ("..." if len(chunk.get("text","")) > 300 else ""))
            if i < len(sources):
                st.divider()


def render_intent_badges(intent):
    """Show small intent indicator badges."""
    badges = []
    if intent and intent.is_drug_interaction:
        badges.append("💊 Drug interaction query")
    if intent and intent.is_symptom_query:
        badges.append("🩺 Symptom query")
    if intent and intent.drugs_mentioned:
        drugs = ", ".join(d.title() for d in intent.drugs_mentioned)
        badges.append(f"🔍 Drugs detected: {drugs}")
    if badges:
        st.caption(" · ".join(badges))


# Render all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_intent_badges(msg.get("intent"))
            render_sources(msg.get("sources", []))


# ── Chat input ────────────────────────────────────────────────────────────────
# Sample questions
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    sample_qs = [
        "Can I take ibuprofen with warfarin?",
        "What are the symptoms of heart failure?",
        "How does metformin work for diabetes?",
        "What drugs interact with atorvastatin?",
        "What is COPD and how is it managed?",
    ]
    cols = st.columns(len(sample_qs))
    for col, q in zip(cols, sample_qs):
        if col.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Build chat history for context (exclude sources/intent fields)
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # all but current
    ]

    # Stream assistant response
    with st.chat_message("assistant"):
        # Get sources before streaming
        token_gen, sources, intent = pipeline.stream_with_sources(
            user_input, chat_history=chat_history
        )

        # Render intent badges immediately
        render_intent_badges(intent)

        # Stream the response
        response_placeholder = st.empty()
        full_response = ""
        for token in token_gen:
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

        # Show sources after response
        render_sources(sources)

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "sources": sources,
        "intent":  intent,
    })

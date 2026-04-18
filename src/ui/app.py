"""Streamlit demo UI for Clinical RAG Assistant."""

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="\U0001fa7a",
    layout="wide",
)


# ── Sidebar: health + metrics ─────────────────────────────────────────────
with st.sidebar:
    st.title("Clinical RAG Assistant")
    st.caption("Evidence-based clinical Q&A with source attribution")

    # Health check
    st.subheader("System Status")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        status = health.get("status", "unknown")
        if status == "ok":
            st.success(f"**Status:** {status}")
        else:
            st.warning(f"**Status:** {status}")
        st.markdown(f"**Model:** `{health.get('model', 'N/A')}`")
        st.markdown(f"**DB Connected:** {health.get('db_connected', False)}")
        st.markdown(f"**Chunks Loaded:** {health.get('chunks_loaded', 0):,}")
    except Exception:
        st.error("API unreachable at " + API_BASE)

    st.divider()

    # Metrics
    st.subheader("Performance Metrics")
    try:
        metrics = requests.get(f"{API_BASE}/metrics", timeout=3).json()
        col1, col2 = st.columns(2)
        col1.metric("Total Queries", metrics.get("total_queries", 0))
        col2.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.0f}ms")

        col3, col4 = st.columns(2)
        hall_rate = metrics.get("hallucination_rate", 0) * 100
        col3.metric("Hallucination Rate", f"{hall_rate:.1f}%")
        col4.metric("Safety Overrides", metrics.get("safety_override_count", 0))

        col5, col6 = st.columns(2)
        col5.metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.1%}")
        col6.metric("p95 Latency", f"{metrics.get('p95_latency_ms', 0):.0f}ms")
    except Exception:
        st.caption("Metrics unavailable")

    st.divider()

    # Data sources
    st.subheader("Data Sources")
    try:
        sources = requests.get(f"{API_BASE}/sources", timeout=3).json()
        for src in sources:
            st.markdown(
                f"**{src['source_type'].upper()}**: "
                f"{src.get('count', 0):,} docs / "
                f"{src.get('chunk_count', 0):,} chunks"
            )
    except Exception:
        st.caption("Sources unavailable")

    st.divider()
    st.caption(
        "4-layer safety: confidence thresholding, source grounding, "
        "hallucination detection, direct advice filtering"
    )


# ── Main: query interface ─────────────────────────────────────────────────
st.header("Ask a Clinical Question")

with st.form("query_form"):
    question = st.text_area(
        "Question",
        placeholder="e.g., What are the indications for massive transfusion protocol activation in trauma patients?",
        height=80,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        top_k = st.slider("Sources to retrieve", min_value=1, max_value=20, value=5)
    with col_b:
        confidence_threshold = st.slider(
            "Confidence threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05
        )
    with col_c:
        domain_filter = st.text_input(
            "Domain filter (optional)", placeholder="e.g., transfusion, trauma"
        )

    submitted = st.form_submit_button("Submit Query", type="primary", use_container_width=True)

if submitted and question.strip():
    payload = {
        "question": question.strip(),
        "top_k": top_k,
        "confidence_threshold": confidence_threshold,
    }
    if domain_filter.strip():
        payload["domain_filter"] = domain_filter.strip()

    with st.spinner("Retrieving evidence and generating answer..."):
        try:
            resp = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Is the backend running at " + API_BASE + "?")
            st.stop()
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.stop()

    # ── Answer ─────────────────────────────────────────────────────────
    st.subheader("Answer")

    # Safety banner
    if not data.get("is_safe", True):
        st.warning("Safety guardrails modified this response: " + (data.get("override_reason") or "unknown reason"))
    elif data.get("override_reason"):
        st.info("Note: " + data["override_reason"])

    st.markdown(data.get("answer", "No answer returned."))

    # ── Confidence indicators ──────────────────────────────────────────
    st.subheader("Confidence & Grounding")
    col_conf, col_ground, col_latency, col_model = st.columns(4)

    confidence = data.get("confidence", 0)
    grounding = data.get("grounding_score", 0)
    latency = data.get("latency_ms", 0)

    with col_conf:
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(min(confidence, 1.0))

    with col_ground:
        st.metric("Source Grounding", f"{grounding:.1%}")
        st.progress(min(grounding, 1.0))

    with col_latency:
        st.metric("Latency", f"{latency:,}ms")

    with col_model:
        st.metric("Model", data.get("model", "N/A"))

    # ── Source cards ───────────────────────────────────────────────────
    sources = data.get("sources", [])
    cited = set(data.get("cited_indices", []))

    if sources:
        st.subheader(f"Sources ({len(sources)})")

        for src in sources:
            idx = src.get("index", 0)
            is_cited = idx in cited
            badge = " (cited)" if is_cited else ""

            source_type = src.get("source_type", "unknown").upper()
            type_colors = {
                "PUBMED": "blue",
                "FDA": "red",
                "CLINICAL_TRIALS": "green",
            }
            color = type_colors.get(source_type, "gray")

            with st.expander(
                f"[{idx}] {src.get('title', 'Untitled')} -- {source_type}{badge}",
                expanded=is_cited,
            ):
                mcol1, mcol2, mcol3 = st.columns([2, 1, 1])
                with mcol1:
                    url = src.get("url", "")
                    citation = src.get("citation", "")
                    if url:
                        st.markdown(f"[{citation}]({url})")
                    else:
                        st.markdown(f"`{citation}`")
                with mcol2:
                    year = src.get("year")
                    if year:
                        st.markdown(f"**Year:** {year}")
                with mcol3:
                    rel = src.get("relevance_score")
                    if rel is not None:
                        st.markdown(f"**Relevance:** {rel:.3f}")

                st.markdown(f"> {src.get('text_preview', 'No preview available.')}")
    else:
        st.info("No sources retrieved for this query.")

elif submitted:
    st.warning("Please enter a question.")


# ── Example queries ────────────────────────────────────────────────────
st.divider()
st.subheader("Example Queries")
examples = [
    "What are the indications for massive transfusion protocol activation in trauma?",
    "What is the recommended ratio of packed RBCs to FFP in massive transfusion?",
    "How does tranexamic acid (TXA) affect mortality in trauma patients?",
    "What are the FDA requirements for AI/ML-based clinical decision support software?",
    "What vital sign parameters predict the need for life-saving interventions in trauma?",
]
for ex in examples:
    st.code(ex, language=None)

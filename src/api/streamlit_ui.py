"""Streamlit demo UI for clinical RAG."""

import streamlit as st


def main():
    """Streamlit app for clinical RAG demo."""
    st.set_page_config(
        page_title="Clinical RAG Assistant",
        page_icon="medical/clinical_RAG.png",
        layout="wide",
    )

    st.title("Clinical RAG Assistant")
    st.markdown("Evidence-based answers to clinical questions")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
        )
        domain_filter = st.selectbox(
            "Medical Domain",
            options=["All", "Trauma", "Critical Care", "Transfusion"],
        )

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input(
            "Ask a clinical question",
            placeholder="e.g., What is the mortality rate in hemorrhagic shock?",
        )

    with col2:
        submit_button = st.button("Search", use_container_width=True)

    # Results display
    if submit_button and query:
        with st.spinner("Searching medical literature..."):
            # Implementation placeholder: call API
            pass

    # Footer
    st.markdown("---")
    st.markdown("Clinical RAG Assistant | Always verify with current guidelines and expert review")


if __name__ == "__main__":
    main()

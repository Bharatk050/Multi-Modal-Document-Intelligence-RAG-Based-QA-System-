import streamlit as st
import os
from vector_BM25_store import VectorStore
from llm_qa import LLMQA, SimpleQA
import config

st.set_page_config(
    page_title="RAG multi-model"
)

# ---- Session State Init ----
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ---- Load Vector Store & QA System ----
if not st.session_state.loaded:
    faiss_file = os.path.join(config.VECTOR_STORE_PATH, "index.faiss")
    alt_faiss_file = f"{config.VECTOR_STORE_PATH}.faiss"

    if os.path.exists(faiss_file) or os.path.exists(alt_faiss_file):
        with st.spinner("Loading pre-processed data..."):
            try:
                vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
                vector_store.load(config.VECTOR_STORE_PATH)
                st.session_state.vector_store = vector_store

                try:
                    qa_system = LLMQA(model_name=config.LLM_MODEL)
                    st.session_state.qa_system = qa_system
                    st.success("✓ LLM model loaded successfully")
                except Exception as llm_error:
                    st.warning(f"⚠️ Using simple QA (LLM model failed: {str(llm_error)[:100]}...)")
                    st.info("💡 Tip: SimpleQA will provide direct excerpts from the document instead of generated answers.")
                    st.session_state.qa_system = SimpleQA()

                st.session_state.loaded = True

            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.session_state.loaded = False

# ---- Main Title ----
st.title(" Multi-Modal RAG ")
st.markdown("Ask questions about the Qatar IMF Report")

# ---- Sidebar ----
with st.sidebar:
    st.header("File Status")

    retrieval_mode = None  # default

    if st.session_state.loaded and st.session_state.vector_store:
        st.success(" Ready to use ")

        # Chunk statistics
        total = len(st.session_state.vector_store.chunks)
        text_count = sum(1 for c in st.session_state.vector_store.chunks if c.get('type') == 'text')
        table_count = sum(1 for c in st.session_state.vector_store.chunks if c.get('type') == 'table')
        image_count = sum(1 for c in st.session_state.vector_store.chunks if c.get('type') == 'image')

        st.markdown(f"**Total Chunks:** {total}")
        st.markdown(f"- Text: `{text_count}`")
        st.markdown(f"- Tables: `{table_count}`")
        st.markdown(f"- Images: `{image_count}`")

        st.markdown("---")

        # Retrieval mode toggle (Hybrid vs Vector only)
        retrieval_mode = st.radio(
            "Retrieval Mode",
            ["Hybrid (BM25 + Vector)", "Vector only"],
            key="retrieval_mode",
        )   

        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    else:
        st.error(" Data Not Found!")
        st.markdown("---")
        st.subheader(" Setup Required")
        st.markdown("""
        Please run the following commands:
        
        **Step 1: Process Document**
        ```bash
        python process_document.py
        ```
        
        **Step 2: Create Embeddings**
        ```bash
        python create_embeddings.py
        ```
        
        **Step 3: Restart App**
        ```bash
        streamlit run app.py
        ```
        """)

# ---- Main Chat Interface ----
if st.session_state.loaded and st.session_state.vector_store and st.session_state.qa_system:
    st.markdown("---")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                with st.expander(f"📚 View Citations ({len(message['citations'])} sources)"):
                    for cite in message["citations"]:
                        # Type emoji
                        type_emoji = "📝" if cite['type'] == 'text' else "📊" if cite['type'] == 'table' else "🖼️"
                        
                        # Relevance color
                        score = cite['relevance_score']
                        if score >= 0.7:
                            score_color = "🟢"
                        elif score >= 0.4:
                            score_color = "🟡"
                        else:
                            score_color = "🔴"
                        
                        st.markdown(f"""
---
**{cite['rank']}. {type_emoji} {cite['source']}**

| Property | Value |
|----------|-------|
| 📄 Page | {cite.get('page', 'N/A')} |
| 📁 Type | {cite['type'].capitalize()} |
| {score_color} Relevance | {score:.1%} |
""")
                        # Show preview if available
                        if 'preview' in cite:
                            st.caption(f"Preview: _{cite['preview']}_")

    # User input
    query = st.chat_input("Ask a question about the document...")

    if query:
        # Append user msg to history
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                vs = st.session_state.vector_store
                mode = st.session_state.get("retrieval_mode", "Hybrid (BM25 + Vector)")

                # ---- Hybrid vs Vector-only retrieval ----
                try:
                    if mode.startswith("Hybrid") and hasattr(vs, "search_hybrid"):
                        search_results = vs.search_hybrid(
                            query,
                            k=5,          # final top-k
                            top_k_vec=20, # candidates from FAISS
                            top_k_bm25=20,
                            alpha=0.6,    # BM25 weight
                        )
                    else:
                        # Fallback: vector-only (search_vector or original search)
                        if hasattr(vs, "search_vector"):
                            search_results = vs.search_vector(query, k=5)
                        else:
                            search_results = vs.search(query, k=5)

                except Exception as e:
                    st.error(f"Error during retrieval: {e}")
                    search_results = []

                # Generate answer with citations
                result = st.session_state.qa_system.generate_answer_with_citations(
                    query, search_results
                )

                st.markdown(result['answer'])

                # Format citations nicely
                if result['citations']:
                    with st.expander(f"📚 View Citations ({len(result['citations'])} sources)"):
                        for cite in result['citations']:
                            # Type emoji
                            type_emoji = "📝" if cite['type'] == 'text' else "📊" if cite['type'] == 'table' else "🖼️"
                            
                            # Relevance color
                            score = cite['relevance_score']
                            if score >= 0.7:
                                score_color = "🟢"
                            elif score >= 0.4:
                                score_color = "🟡"
                            else:
                                score_color = "🔴"
                            
                            st.markdown(f"""
---
**{cite['rank']}. {type_emoji} {cite['source']}**

| Property | Value |
|----------|-------|
| 📄 Page | {cite.get('page', 'N/A')} |
| 📁 Type | {cite['type'].capitalize()} |
| {score_color} Relevance | {score:.1%} |
""")
                            # Show preview if available
                            if 'preview' in cite:
                                st.caption(f"Preview: _{cite['preview']}_")

                # Save assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "citations": result['citations']
                })

else:
    st.info(" Follow steps")
    st.markdown("""
    Step 1: `python process_document.py`  
    Step 2: `python create_embeddings.py`  
    Step 3: `streamlit run app.py`
    """)

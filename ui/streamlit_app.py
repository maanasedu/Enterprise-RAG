import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import uuid


st.set_page_config(
    page_title='Enterprise RAG Platform',
    page_icon='🔍',
    layout='wide'
)


API_URL = 'http://localhost:8000'


def check_api_health():
    try:
        r = requests.get(f'{API_URL}/health', timeout=3)
        return r.ok, r.json() if r.ok else {}
    except Exception:
        return False, {}


def upload_document(file_bytes, filename):
    try:
        response = requests.post(
            f'{API_URL}/upload',
            files={'file': (filename, file_bytes, 'application/octet-stream')},
            timeout=120
        )
        return response.ok, response.json()
    except Exception as e:
        return False, {'detail': str(e)}


# ── Session State Init ────────────────────────────────────────────────────────
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'turn_count' not in st.session_state:
    st.session_state.turn_count = 0


# Keep a stable copy of session_id to prevent resets
SESSION_ID = st.session_state.session_id


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title('⚙️ Configuration')

    st.markdown('**Session**')
    st.caption(f'ID: `{SESSION_ID}`')
    st.caption(f'Turns: {st.session_state.turn_count}')

    if st.button('🗑️ Clear Conversation', use_container_width=True):
        try:
            requests.post(
                f'{API_URL}/clear_session',
                json={'session_id': SESSION_ID},
                timeout=5
            )
        except Exception:
            pass

        st.session_state.messages = []
        st.session_state.turn_count = 0
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()

    st.markdown('**Settings**')

    top_k = st.slider('Sources per query', 1, 10, 5)

    st.divider()

    st.markdown('**API Status**')

    is_healthy, health_data = check_api_health()

    if is_healthy:
        st.success('✅ Online')

        if health_data.get('components'):
            c = health_data['components']
            st.caption(f"📊 {c.get('faiss_vectors', 0):,} vectors indexed")
            st.caption(f"📄 {c.get('bm25_documents', 0):,} BM25 docs")
            st.caption(f"💬 {c.get('active_sessions', 0)} active sessions")
    else:
        st.error('❌ Offline')
        st.caption('Run: uvicorn api.rag_api:app --reload --port 8000')

    st.divider()

    st.markdown('**📤 Upload Document**')

    uploaded_file = st.file_uploader(
        'Upload PDF or DOCX',
        type=['pdf', 'docx', 'doc'],
        help='Max 20MB. Indexed immediately.'
    )

    if uploaded_file is not None:
        if st.button(
            '📥 Index Document',
            use_container_width=True,
            type='primary'
        ):
            if not is_healthy:
                st.error('API is offline')
            else:
                with st.spinner(f'Indexing {uploaded_file.name}...'):
                    file_bytes = uploaded_file.read()

                    success, result = upload_document(
                        file_bytes,
                        uploaded_file.name
                    )

                if success:
                    st.success(
                        f'✅ Indexed!\n'
                        f'• Chunks added: {result.get("chunks_added", 0)}\n'
                        f'• Total vectors: {result.get("total_vectors", 0):,}'
                    )
                    st.rerun()
                else:
                    st.error(
                        f'Failed: {result.get("detail", "Unknown error")}'
                    )


# ── Main Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['💬 Chat', '📊 Evaluation', 'ℹ️ About'])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.title('🔍 Enterprise RAG Platform')

    if not is_healthy:
        st.error(
            '⚠️ API offline. '
            'Run: `uvicorn api.rag_api:app --reload --port 8000`'
        )

    elif health_data.get('components', {}).get('faiss_vectors', 0) == 0:
        st.warning('⚠️ No documents indexed. Upload a document in the sidebar.')

    else:
        vectors = health_data.get('components', {}).get('faiss_vectors', 0)
        st.caption(
            f'📚 Searching across {vectors:,} indexed chunks · '
            f'Session: `{SESSION_ID}`'
        )

    # Suggested questions shown only when chat is empty
    if not st.session_state.messages and is_healthy:
        st.markdown('**Try asking:**')

        suggestions = [
            'What is retrieval augmented generation?',
            'How does dense retrieval work?',
            'What are the limitations of RAG systems?',
            'Explain BM25 and how it differs from dense embeddings',
        ]

        cols = st.columns(2)

        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(
                    suggestion,
                    key=f'sug_{i}',
                    use_container_width=True
                ):
                    st.session_state['pending_question'] = suggestion
                    st.rerun()

    # Display existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

            if msg.get('sources'):
                with st.expander(
                    f"📄 {len(msg['sources'])} sources · "
                    f"⏱️ {msg.get('latency_ms', 0):.0f}ms · "
                    f"Turn {msg.get('turn_number', '?')}"
                ):
                    for i, src in enumerate(msg['sources']):
                        score = (
                            src.get('relevance_score') or
                            src.get('score') or 0
                        )

                        meta = src.get('metadata', {})
                        filename = meta.get('filename', 'Unknown source')

                        st.markdown(
                            f'**Source {i + 1}** — `{filename}` '
                            f'(score: {score:.3f})'
                        )

                        st.text(src.get('text', '')[:300] + '...')

                        if i < len(msg['sources']) - 1:
                            st.divider()

    # Handle suggestion button clicks
    pending = st.session_state.pop('pending_question', None)

    # Chat input
    question = st.chat_input('Ask a question about your documents...')

    if pending:
        question = pending

    if question:
        if not is_healthy:
            st.error('API is offline.')

        else:
            # Show user message immediately
            st.session_state.messages.append({
                'role': 'user',
                'content': question
            })

            with st.chat_message('user'):
                st.markdown(question)

            with st.chat_message('assistant'):
                with st.spinner('Searching and generating...'):
                    try:
                        response = requests.post(
                            f'{API_URL}/query',
                            json={
                                'question': question,
                                'top_k': top_k,
                                'session_id': SESSION_ID
                            },
                            timeout=60
                        )

                        if response.ok:
                            data = response.json()

                            answer = data.get(
                                'answer',
                                'No answer returned.'
                            )

                            sources = data.get('sources', [])

                            latency = data.get('latency_ms', 0)
                            turn_number = data.get('turn_number', 0)

                            st.markdown(answer)

                            if sources:
                                with st.expander(
                                    f'📄 {len(sources)} sources · '
                                    f'Turn {turn_number}'
                                ):
                                    for i, src in enumerate(sources):
                                        score = (
                                            src.get('relevance_score') or
                                            src.get('score') or 0
                                        )

                                        meta = src.get('metadata', {})
                                        filename = meta.get(
                                            'filename',
                                            'Unknown'
                                        )

                                        st.markdown(
                                            f'**Source {i + 1}** — '
                                            f'`{filename}` '
                                            f'(score: {score:.3f})'
                                        )

                                        st.text(
                                            src.get('text', '')[:300] + '...'
                                        )

                                        if i < len(sources) - 1:
                                            st.divider()

                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': answer,
                                'sources': sources,
                                'latency_ms': latency,
                                'turn_number': turn_number
                            })

                            st.session_state.turn_count += 1

                        else:
                            try:
                                error_detail = response.json().get(
                                    'detail',
                                    response.text
                                )
                            except Exception:
                                error_detail = response.text

                            st.error(
                                f'Error {response.status_code}: '
                                f'{error_detail}'
                            )

                            st.session_state.messages.append({
                                'role': 'assistant',
                                'content': f'Error: {error_detail}'
                            })

                    except requests.exceptions.Timeout:
                        st.error(
                            '⏱️ Request timed out. Please try again.'
                        )

                    except Exception as e:
                        st.error(f'Unexpected error: {str(e)}')


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: EVALUATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title('📊 Evaluation Dashboard')

    st.info(
        'Run `python scripts/run_evaluation.py` to generate results, '
        'then upload the CSV here.'
    )

    col1, col2 = st.columns(2)

    with col1:
        eval_file = st.file_uploader(
            'Upload eval_results.csv',
            type='csv',
            key='eval_upload'
        )

    with col2:
        chunk_file = st.file_uploader(
            'Upload chunking_comparison.csv',
            type='csv',
            key='chunk_upload'
        )

    if eval_file:
        df = pd.read_csv(eval_file)

        metrics = [
            'faithfulness',
            'answer_relevancy',
            'context_precision',
            'context_recall'
        ]

        st.subheader('Overall Scores')

        cols = st.columns(4)

        for col, metric in zip(cols, metrics):
            if metric in df.columns:
                val = df[metric].mean()
                delta = val - 0.5

                col.metric(
                    metric.replace('_', ' ').title(),
                    f'{val:.3f}',
                    delta=f'{delta:+.3f} vs 0.5 baseline',
                    delta_color='normal'
                )

        st.divider()

        st.subheader('Score Distributions')

        col_a, col_b = st.columns(2)

        with col_a:
            if 'faithfulness' in df.columns:
                fig = px.histogram(
                    df,
                    x='faithfulness',
                    title='Faithfulness Distribution',
                    color_discrete_sequence=['#1E40AF'],
                    nbins=20
                )

                fig.update_layout(showlegend=False)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with col_b:
            if 'answer_relevancy' in df.columns:
                fig = px.histogram(
                    df,
                    x='answer_relevancy',
                    title='Answer Relevancy Distribution',
                    color_discrete_sequence=['#166534'],
                    nbins=20
                )

                fig.update_layout(showlegend=False)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        available = [
            m for m in metrics
            if m in df.columns
        ]

        if len(available) >= 3:
            means = [
                df[m].mean()
                for m in available
            ]

            labels = [
                m.replace('_', ' ').title()
                for m in available
            ]

            fig = px.line_polar(
                r=means + [means[0]],
                theta=labels + [labels[0]],
                line_close=True,
                title='RAGAS Metrics Radar'
            )

            fig.update_traces(
                fill='toself',
                fillcolor='rgba(30, 64, 175, 0.2)'
            )

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 1])
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with st.expander('View Raw Data'):
            st.dataframe(
                df,
                use_container_width=True
            )

    if chunk_file:
        st.subheader('Chunking Strategy Comparison')

        df_chunk = pd.read_csv(
            chunk_file,
            index_col=0
        )

        st.dataframe(
            df_chunk.style.highlight_max(
                axis=0,
                color='#DCFCE7'
            ),
            use_container_width=True
        )

        if 'context_precision' in df_chunk.columns:
            fig = px.bar(
                df_chunk.reset_index(),
                x='index',
                y='context_precision',
                title='Context Precision by Chunking Strategy',
                color='context_precision',
                color_continuous_scale='Blues',
                labels={
                    'index': 'Strategy',
                    'context_precision': 'Context Precision'
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title('ℹ️ About This Project')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Architecture')

        st.markdown("""
        **Retrieval Pipeline**
        - BM25 sparse retrieval
        - BGE-large dense embeddings
        - Reciprocal Rank Fusion (RRF)
        - Cohere cross-encoder reranking

        **Generation**
        - openai/gpt-oss-120b via Groq
        - Conversation memory (last 3 turns)
        - Context truncation for token safety
        - Retry logic with fallback

        **Evaluation**
        - RAGAS: faithfulness, relevancy, precision, recall
        """)

    with col2:
        st.subheader('Benchmark Results')

        results_data = {
            'Metric': [
                'Answer Relevancy',
                'Context Recall',
                'Context Precision',
                'Faithfulness'
            ],
            'Score': [
                0.900,
                0.721,
                0.877,
                0.842
            ],
            'Samples': [
                45,
                45,
                45,
                45
            ]
        }

        st.dataframe(
            pd.DataFrame(results_data),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            'Evaluated on 45 samples using openai/gpt-oss-120b evaluator '
            'with BGE-large embeddings (free stack).'
        )

    st.subheader('Tech Stack')

    tech_cols = st.columns(4)

    stacks = [
        ('🧠 LLM', 'gpt-oss-120b\n(Groq API)'),
        ('🔢 Embeddings', 'BAAI/bge-large\n(Local)'),
        ('🗄️ Vector DB', 'FAISS\n(Local disk)'),
        ('⚡ API', 'FastAPI\n(Async)'),
    ]

    for col, (title, content) in zip(tech_cols, stacks):
        col.metric(title, content)
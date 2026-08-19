import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ArXivNavigator", page_icon="🔬", layout="wide")

DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/{os.getenv('POSTGRES_DB')}"
engine = create_engine(DATABASE_URL)


@st.cache_data(ttl=30)
def load_data():
    query = "SELECT * FROM query_logs ORDER BY id DESC"
    return pd.read_sql(query, engine)


df_all = load_data()

st.title("🔬 ArXivNavigator — Self-Healing RAG for AI Research")

if df_all.empty:
    st.warning("No queries logged yet. Run some queries against the API first.")
    st.stop()

# ---------- Sidebar: filters + refresh ----------
with st.sidebar:
    st.header("Filters")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    categories = sorted(df_all['category'].dropna().unique().tolist())
    category_choice = st.selectbox("Category", ["All"] + categories)
    selected_categories = categories if category_choice == "All" else [category_choice]

    query_types = sorted(df_all['query_type'].dropna().unique().tolist())
    query_type_choice = st.selectbox("Query Type", ["All"] + query_types)
    selected_types = query_types if query_type_choice == "All" else [query_type_choice]

# Apply filters
df = df_all[
    df_all['category'].isin(selected_categories) &
    df_all['query_type'].isin(selected_types)
].copy()

last_synced = pd.to_datetime(df_all['timestamp'].max())
st.caption(f"{len(df)} of {len(df_all)} queries shown | Last synced: {last_synced.strftime('%b %d, %Y at %I:%M %p')}")

if df.empty:
    st.warning("No queries match the current filters. Adjust the filters in the sidebar.")
    st.stop()

# ---------- Tabs ----------
tab_live, tab_overview, tab_deep_dive, tab_log = st.tabs(
    ["💬 Live Query", "📊 Overview", "🔍 Deep Dive", "📋 Query Log"]
)

# ---------- TAB: Live Query ----------
with tab_live:
    st.subheader("Ask a Research Question")
    with st.form("live_query"):
        q_col, cat_col = st.columns([3, 1])
        query_input = q_col.text_input("Query", placeholder="Ask a research question...", label_visibility="collapsed")
        category_input = cat_col.selectbox("Category", ["cs_ai", "cs_lg", "cs_cl", "cs_cv", "cs_ir"], label_visibility="collapsed")
        submitted = st.form_submit_button("Search")

    if submitted and query_input:
        import requests
        with st.spinner("Retrieving, generating, and evaluating... this can take a minute."):
            try:
                resp = requests.post(
                    os.getenv("API_URL", "http://127.0.0.1:8000") + "/query",
                    json={"query": query_input, "category": category_input},
                    timeout=600
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(result['answer'])
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Faithfulness", f"{result['faithfulness']:.2f}" if result['faithfulness'] is not None else "N/A")
                    sc2.metric("Answer Relevancy", f"{result['answer_relevancy']:.2f}" if result['answer_relevancy'] is not None else "N/A")
                    sc3.metric("Context Relevance", f"{result['context_relevance']:.2f}" if result['context_relevance'] is not None else "N/A")
                    sc4.metric("Composite", f"{result['composite_score']:.2f}")
                    if result['healing_triggered']:
                        st.info(f"Healing was triggered. Winning strategy: {result.get('winning_strategy') or 'original answer kept'}")
                    st.cache_data.clear()
                    st.caption("New result saved — click 'Refresh Data' in the sidebar to see it reflected in the charts.")
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# ---------- TAB: Overview ----------
with tab_overview:
    st.subheader("RAG Health")

    avg_composite = df['composite_score'].mean()
    avg_faithfulness = df['faithfulness'].mean()
    avg_relevancy = df['answer_relevancy'].mean()
    avg_context = df['context_relevance'].mean()

    gauge_color = "red" if avg_composite < 0.6 else "yellow" if avg_composite < 0.75 else "green"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_composite,
        number={'valueformat': '.2f'},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 0.6], 'color': '#fde2e1'},
                {'range': [0.6, 0.75], 'color': '#fff3cd'},
                {'range': [0.75, 1.0], 'color': '#d4edda'},
            ],
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': 0.7}
        },
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Faithfulness", f"{avg_faithfulness:.2f}")
    c2.metric("Answer Relevancy", f"{avg_relevancy:.2f}")
    c3.metric("Context Relevance", f"{avg_context:.2f}")
    st.caption("Weighted 40% Faithfulness, 40% Answer Relevancy, 20% Context Relevance — averaged across filtered queries")

    st.divider()

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("**Healing Trigger Rate**")
        healed_count = df['healing_triggered'].sum()
        total = len(df)
        rate = healed_count / total * 100 if total else 0
        fig_pie = px.pie(
            names=['Healed', 'No Healing Needed'],
            values=[healed_count, total - healed_count],
            color_discrete_sequence=['#f0ad4e', '#5cb85c'], hole=0.5
        )
        fig_pie.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(f"{rate:.0f}% of queries triggered healing ({healed_count}/{total})")

    with row1_col2:
        st.markdown("**Average Composite Score by Category**")
        cat_avg = df.groupby('category')['composite_score'].mean().reset_index()
        fig_bar = px.bar(cat_avg, x='category', y='composite_score', color='composite_score',
                          color_continuous_scale='RdYlGn', range_color=[0, 1])
        fig_bar.add_hline(y=0.7, line_dash="dash", annotation_text="Threshold")
        fig_bar.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------- TAB: Deep Dive ----------
with tab_deep_dive:
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("**Strategy Effectiveness by Query Type**")
        healed_df = df[df['healing_triggered'] & df['winning_strategy'].notna()]
        if not healed_df.empty:
            pivot = healed_df.pivot_table(index='winning_strategy', columns='query_type', values='composite_score', aggfunc='mean')
            fig_heat = px.imshow(pivot, color_continuous_scale='Blues', aspect='auto', labels=dict(color="Avg Composite"))
            fig_heat.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No winning strategies logged yet for the current filters.")

    with row2_col2:
        st.markdown("**Query Complexity vs Final Quality**")
        plot_df = df.copy()
        plot_df['complexity'] = plot_df['query'].apply(lambda q: len(q.split()) * (sum(len(w) for w in q.split()) / max(len(q.split()), 1)))
        plot_df['healing_label'] = plot_df['healing_triggered'].map({True: 'Healing triggered', False: 'No healing needed'})
        fig_scatter = px.scatter(
            plot_df, x='complexity', y='composite_score', color='healing_label', size='healing_attempts',
            color_discrete_map={'Healing triggered': 'red', 'No healing needed': 'green'}, hover_data=['query']
        )
        fig_scatter.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.subheader("Healing Journey")

    healed_queries = df[df['healing_triggered']].sort_values('id', ascending=False)
    if not healed_queries.empty:
        query_options = {row['id']: f"{row['query'][:60]}..." for _, row in healed_queries.iterrows()}
        selected_id = st.selectbox("Select a healed query to inspect", options=list(query_options.keys()),
                                    format_func=lambda x: query_options[x])
        attempts_df = pd.read_sql(f"SELECT * FROM healing_attempts WHERE query_log_id = {selected_id} ORDER BY attempt_number", engine)
        if not attempts_df.empty:
            attempts_df['label'] = attempts_df.apply(
                lambda r: 'Initial' if r['attempt_number'] == 0 else (r['strategy_name'] or f"Attempt {r['attempt_number']}"), axis=1
            )
            fig_journey = go.Figure()
            fig_journey.add_trace(go.Scatter(
                x=attempts_df['attempt_number'], y=attempts_df['composite_score'],
                mode='lines+markers+text', text=attempts_df['label'], textposition='top center',
                marker=dict(
                    size=14,
                    color=['gold' if w else ('green' if s >= 0.7 else 'red') for w, s in zip(attempts_df['is_winner'], attempts_df['composite_score'])],
                    symbol=['star' if w else 'circle' for w in attempts_df['is_winner']],
                ),
                line=dict(color='gray', dash='dot')
            ))
            fig_journey.add_hline(y=0.7, line_dash="dash", annotation_text="Healing threshold")
            fig_journey.update_layout(xaxis_title="Attempt", yaxis_title="Composite Score", yaxis_range=[0, 1], height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig_journey, use_container_width=True)
            st.caption("Gold star = attempt returned to user. Other points tried and correctly discarded.")
    else:
        st.info("No healed queries match the current filters.")

    st.divider()
    st.subheader("Healing Impact: Before vs After")

    impact_df = pd.read_sql(
        """
        SELECT ql.id, ql.query, ql.query_type,
               MIN(CASE WHEN ha.attempt_number = 0 THEN ha.composite_score END) AS before_score,
               MAX(CASE WHEN ha.is_winner THEN ha.composite_score END) AS after_score
        FROM query_logs ql
        JOIN healing_attempts ha ON ha.query_log_id = ql.id
        WHERE ql.healing_triggered = true
        GROUP BY ql.id, ql.query, ql.query_type
        ORDER BY ql.id DESC
        LIMIT 15
        """, engine
    )
    impact_df = impact_df[impact_df['id'].isin(df['id'])]

    if not impact_df.empty:
        impact_df['query_short'] = impact_df['query'].str.slice(0, 35)
        impact_df = impact_df.sort_values('before_score')
        fig_impact = go.Figure()
        fig_impact.add_trace(go.Bar(y=impact_df['query_short'], x=impact_df['before_score'], name='Before Healing', orientation='h', marker_color='#d9534f'))
        fig_impact.add_trace(go.Bar(y=impact_df['query_short'], x=impact_df['after_score'], name='After Healing', orientation='h', marker_color='#5cb85c'))
        fig_impact.add_vline(x=0.7, line_dash="dash", annotation_text="Threshold")
        fig_impact.update_layout(barmode='group', height=max(300, len(impact_df) * 35), xaxis_title="Composite Score", margin=dict(t=20, b=20))
        st.plotly_chart(fig_impact, use_container_width=True)
        avg_improvement = ((impact_df['after_score'] - impact_df['before_score']) / impact_df['before_score'].replace(0, 0.01) * 100).mean()
        st.caption(f"+{avg_improvement:.0f}% average improvement across {len(impact_df)} healed queries")
    else:
        st.info("No healing impact data for the current filters.")

# ---------- TAB: Query Log ----------
with tab_log:
    st.subheader("Recent Query Log")
    display_df = df[['query', 'query_type', 'category', 'healing_triggered', 'winning_strategy',
                      'faithfulness', 'answer_relevancy', 'context_relevance', 'composite_score', 'timestamp']].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "query": st.column_config.TextColumn("Query", width="large"),
            "query_type": st.column_config.TextColumn("Type", width="small"),
            "category": st.column_config.TextColumn("Category", width="small"),
            "healing_triggered": st.column_config.CheckboxColumn("Healed", width="small"),
            "winning_strategy": st.column_config.TextColumn("Strategy", width="medium"),
            "faithfulness": st.column_config.NumberColumn("Faithfulness", format="%.2f", width="small"),
            "answer_relevancy": st.column_config.NumberColumn("Relevancy", format="%.2f", width="small"),
            "context_relevance": st.column_config.NumberColumn("Context", format="%.2f", width="small"),
            "composite_score": st.column_config.NumberColumn("Composite", format="%.2f", width="small"),
            "timestamp": st.column_config.DatetimeColumn("Time", format="MMM D, h:mm a", width="medium"),
        }
    )

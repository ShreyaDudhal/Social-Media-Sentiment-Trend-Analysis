"""
Social Media Sentiment and Trend Analysis Using Big Data Technologies
Streamlit Analytics Dashboard Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from src.data_preprocessing import map_and_preprocess_columns
from src.spark_processor import SparkAnalyticsProcessor
from src.sentiment_analysis import analyze_sentiment, get_sentiment_over_time
from src.hashtag_analysis import analyze_hashtags_and_trends
from src.engagement_analysis import compute_engagement_breakdowns
from src.network_analysis import build_social_network_graph, compute_network_metrics, generate_interactive_network_fig
from src.community_detection import detect_communities, generate_community_network_fig
from src.mongodb_handler import MongoDBHandler
from utils.helpers import convert_df_to_csv, format_number

# Page Config
st.set_page_config(
    page_title="Social Media Sentiment & Trend Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
    }

    .subtitle-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-top: 8px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* KPI Cards Styling */
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .kpi-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }

    .kpi-subtext {
        color: #10b981;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Status Badge Styling */
    .status-box {
        padding: 12px 16px;
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.9rem;
        margin-bottom: 16px;
    }
    .status-connected {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .status-disconnected {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    /* Tab Header styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.92rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if 'spark_processor' not in st.session_state:
    st.session_state.spark_processor = SparkAnalyticsProcessor()

if 'mongo_handler' not in st.session_state:
    st.session_state.mongo_handler = MongoDBHandler()

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>Social Media Sentiment & Trend Analysis</h1>
    <p>Big Data Analytics using PySpark, MongoDB, and Social Network Analysis</p>
    <div class="subtitle-badge">BIG DATA ANALYTICS MINI PROJECT</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/analytics.png", width=64)
    st.title("Control Panel")
    st.markdown("---")
    
    st.subheader("📁 Dataset Selection")
    data_option = st.radio(
        "Choose Data Source:",
        ("Use Sample Dataset", "Upload CSV Dataset"),
        index=0
    )
    
    raw_df = None
    if data_option == "Use Sample Dataset":
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_social_media.csv")
        if os.path.exists(sample_path):
            raw_df = pd.read_csv(sample_path)
            st.success("Loaded demonstration dataset (350 records)")
        else:
            st.error("Sample dataset file not found at data/sample_social_media.csv")
    else:
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded dataset loaded ({len(raw_df)} rows)")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                
    st.markdown("---")
    st.subheader("🍃 MongoDB Status")
    mongo_info = st.session_state.mongo_handler.get_status()
    if mongo_info['status'] == 'CONNECTED':
        st.markdown(f'<div class="status-box status-connected">🟢 <b>CONNECTED</b><br><small>{mongo_info["message"]}</small></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-box status-disconnected">🟡 <b>NOT CONNECTED</b><br><small>{mongo_info["message"]}</small></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("⚡ Processing Engine")
    if st.session_state.spark_processor.is_active:
        st.info("🔥 **PySpark 3.x Engine**: Active (`local[*]`)")
    else:
        st.info("💻 **Pandas Engine**: Active (Local fallback)")

# If no dataset available, stop execution gracefully
if raw_df is None or raw_df.empty:
    st.warning("Please select a data source from the sidebar to begin analysis.")
    st.stop()

# Preprocess & Normalize Dataset
df_clean, preprocessing_report = map_and_preprocess_columns(raw_df)

# Run Sentiment Analysis Engine
df_analyzed, sentiment_summary = analyze_sentiment(df_clean)

# Compute Overview KPIs via PySpark Engine
kpi_results = st.session_state.spark_processor.compute_overview_kpis(df_analyzed)

# Store in MongoDB if connected
if st.session_state.mongo_handler.is_connected:
    st.session_state.mongo_handler.store_processed_posts(df_analyzed)
    st.session_state.mongo_handler.store_analysis_summary({
        'kpis': kpi_results,
        'sentiment': sentiment_summary
    })

# Warning banner if columns were missing
if preprocessing_report['missing_columns']:
    missing_str = ", ".join(preprocessing_report['missing_columns'])
    st.warning(f"⚠️ Notice: The following optional columns were missing and automatically filled with default values: **{missing_str}**.")

# Define 8 Dashboard Tabs
tabs = st.tabs([
    "📊 Overview",
    "😄 Sentiment Analysis",
    "🔥 Trend Analysis",
    "👤 User & Engagement",
    "🕸️ Social Network",
    "🧩 Community Detection",
    "🔍 Data Explorer",
    "ℹ️ About Project"
])

# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with tabs[0]:
    st.subheader("Exploratory Data Analysis & System Summary")
    st.caption(f"Processed with {kpi_results['engine_used']}")
    
    # Row 1: KPI Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Posts</div>
            <div class="kpi-value">{format_number(kpi_results['total_posts'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Users</div>
            <div class="kpi-value">{format_number(kpi_results['total_users'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Likes</div>
            <div class="kpi-value">{format_number(kpi_results['total_likes'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Comments</div>
            <div class="kpi-value">{format_number(kpi_results['total_comments'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Shares</div>
            <div class="kpi-value">{format_number(kpi_results['total_shares'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Engagement</div>
            <div class="kpi-value">{kpi_results['avg_engagement']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sentiment Pills
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.info(f"🟢 **Positive Posts**: {sentiment_summary['positive_count']} ({sentiment_summary['positive_pct']}%)")
    with sc2:
        st.info(f"⚪ **Neutral Posts**: {sentiment_summary['neutral_count']} ({sentiment_summary['neutral_pct']}%)")
    with sc3:
        st.info(f"🔴 **Negative Posts**: {sentiment_summary['negative_count']} ({sentiment_summary['negative_pct']}%)")

    st.markdown("---")
    
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📋 Dataset Preview Table")
        st.dataframe(df_analyzed[['user_id', 'post', 'timestamp', 'likes', 'comments', 'shares', 'sentiment', 'engagement']].head(10), use_container_width=True)
        
    with col_right:
        st.subheader("📈 Summary Statistics")
        stats_df = pd.DataFrame({
            "Metric": ["Average Likes", "Average Comments", "Average Shares", "Maximum Likes", "Maximum Shares", "Average Engagement"],
            "Value": [
                kpi_results['avg_likes'],
                kpi_results['avg_comments'],
                kpi_results['avg_shares'],
                kpi_results['max_likes'],
                kpi_results['max_shares'],
                kpi_results['avg_engagement']
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ==========================================
# TAB 2: SENTIMENT ANALYSIS
# ==========================================
with tabs[1]:
    st.subheader("😄 Sentiment Analysis (VADER NLP Model)")
    st.caption("Classifies post text into Positive, Neutral, or Negative using VADER Compound Thresholds (>= 0.05 Positive, <= -0.05 Negative).")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Analyzed Posts", sentiment_summary['total_posts'])
    m2.metric("Positive Posts", f"{sentiment_summary['positive_count']} ({sentiment_summary['positive_pct']}%)")
    m3.metric("Neutral Posts", f"{sentiment_summary['neutral_count']} ({sentiment_summary['neutral_pct']}%)")
    m4.metric("Negative Posts", f"{sentiment_summary['negative_count']} ({sentiment_summary['negative_pct']}%)")
    
    col_pie, col_bar = st.columns(2)
    with col_pie:
        # Donut Chart
        fig_donut = px.pie(
            values=[sentiment_summary['positive_count'], sentiment_summary['neutral_count'], sentiment_summary['negative_count']],
            names=['Positive', 'Neutral', 'Negative'],
            color=['Positive', 'Neutral', 'Negative'],
            color_discrete_map={'Positive': '#10b981', 'Neutral': '#94a3b8', 'Negative': '#ef4444'},
            hole=0.4,
            title="Sentiment Distribution Breakdown"
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_bar:
        # Bar Chart
        fig_sent_bar = px.bar(
            x=['Positive', 'Neutral', 'Negative'],
            y=[sentiment_summary['positive_count'], sentiment_summary['neutral_count'], sentiment_summary['negative_count']],
            color=['Positive', 'Neutral', 'Negative'],
            color_discrete_map={'Positive': '#10b981', 'Neutral': '#94a3b8', 'Negative': '#ef4444'},
            labels={'x': 'Sentiment Category', 'y': 'Number of Posts'},
            title="Sentiment Post Count Comparison"
        )
        st.plotly_chart(fig_sent_bar, use_container_width=True)
        
    # Sentiment Over Time
    st.subheader("📅 Sentiment Distribution Over Time")
    sentiment_time_df = get_sentiment_over_time(df_analyzed)
    if not sentiment_time_df.empty:
        fig_time = px.line(
            sentiment_time_df,
            x='date',
            y=['Positive', 'Neutral', 'Negative'],
            color_discrete_map={'Positive': '#10b981', 'Neutral': '#64748b', 'Negative': '#ef4444'},
            title="Daily Sentiment Trend Tracker",
            labels={'value': 'Number of Posts', 'date': 'Date', 'variable': 'Sentiment'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
    st.subheader("📑 Detailed Post Sentiment Table")
    st.dataframe(
        df_analyzed[['user_id', 'post', 'sentiment', 'compound_score', 'pos_score', 'neu_score', 'neg_score']].head(20),
        use_container_width=True
    )


# ==========================================
# TAB 3: TREND ANALYSIS
# ==========================================
with tabs[2]:
    st.subheader("🔥 Hashtag Frequency & Trending Topic Analysis")
    
    st.info("💡 **Trending Topic Algorithm**: Trending Score = Likes + Comments + Shares per hashtag topic. Trending topics are identified using keyword frequency and total audience engagement.")
    
    trend_df, trend_summary = analyze_hashtags_and_trends(df_analyzed, top_n=10)
    
    if not trend_df.empty:
        col_t1, col_t2 = st.columns([3, 2])
        with col_t1:
            fig_trend = px.bar(
                trend_df,
                x='Trending Score',
                y='Hashtag',
                orientation='h',
                color='Trending Score',
                color_continuous_scale='Viridis',
                title="Top 10 Trending Topics by Score"
            )
            fig_trend.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col_t2:
            st.markdown("### Top 10 Trending Topics")
            st.dataframe(trend_df[['Rank', 'Hashtag', 'Posts', 'Trending Score', 'Likes', 'Shares']], use_container_width=True, hide_index=True)
    else:
        st.warning("No hashtags detected in the current dataset.")


# ==========================================
# TAB 4: USER & ENGAGEMENT ANALYSIS
# ==========================================
with tabs[3]:
    st.subheader("👤 User Activity & Engagement Analytics")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.markdown("### Top 10 Most Active Users")
        active_users_df = st.session_state.spark_processor.get_most_active_users(df_analyzed, top_n=10)
        st.dataframe(active_users_df, use_container_width=True, hide_index=True)
        
        fig_users = px.bar(
            active_users_df,
            x='user_id',
            y='post_count',
            color='post_count',
            title="Posts per Most Active User",
            labels={'user_id': 'User ID', 'post_count': 'Post Count'}
        )
        st.plotly_chart(fig_users, use_container_width=True)
        
    with col_u2:
        st.markdown("### Top 10 Most Shared / Retweeted Posts")
        shared_posts_df = st.session_state.spark_processor.get_most_shared_posts(df_analyzed, top_n=10)
        st.dataframe(shared_posts_df[['Rank', 'user_id', 'post', 'shares', 'likes', 'engagement']], use_container_width=True, hide_index=True)
        
        fig_shared = px.bar(
            shared_posts_df,
            x='user_id',
            y='shares',
            color='shares',
            title="Top Posts by Shares / Retweets",
            labels={'user_id': 'User ID', 'shares': 'Shares / Retweets'}
        )
        st.plotly_chart(fig_shared, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Engagement Breakdowns")
    eng_kpis, eng_by_sentiment, eng_over_time = compute_engagement_breakdowns(df_analyzed)
    
    if not eng_by_sentiment.empty:
        col_ebs, col_eot = st.columns(2)
        with col_ebs:
            fig_eng_sent = px.bar(
                eng_by_sentiment,
                x='sentiment',
                y='avg_engagement',
                color='sentiment',
                color_discrete_map={'Positive': '#10b981', 'Neutral': '#94a3b8', 'Negative': '#ef4444'},
                title="Average Engagement by Sentiment Category"
            )
            st.plotly_chart(fig_eng_sent, use_container_width=True)
            
        with col_eot:
            if not eng_over_time.empty:
                fig_eng_time = px.area(
                    eng_over_time,
                    x='date',
                    y='daily_engagement',
                    title="Total Daily Engagement Over Time"
                )
                st.plotly_chart(fig_eng_time, use_container_width=True)


# ==========================================
# TAB 5: SOCIAL NETWORK ANALYSIS
# ==========================================
with tabs[4]:
    st.subheader("🕸️ Social Network Analysis (SNA)")
    st.info("💡 **Network Graph Concept**: User mentions (`@username`) construct a directed graph where Nodes = Users and Edges = Mention Interactions. **Degree centrality** indicates how connected a user is within the interaction network.")
    
    # Build Network Graph
    G = build_social_network_graph(df_analyzed)
    net_metrics, centrality_df = compute_network_metrics(G)
    
    # Metrics Row
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Total Nodes (Users)", net_metrics['num_nodes'])
    n2.metric("Total Edges (Interactions)", net_metrics['num_edges'])
    n3.metric("Average Degree", net_metrics['avg_degree'])
    n4.metric("Network Density", net_metrics['density'])
    
    st.markdown("---")
    
    # Network Graph Visualization
    st.subheader("🌐 Interactive User Interaction Graph")
    max_nodes_slider = st.slider("Select maximum users to display in graph:", min_value=10, max_value=100, value=40, step=5)
    
    net_fig = generate_interactive_network_fig(G, top_n_nodes=max_nodes_slider)
    st.plotly_chart(net_fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🏆 Top 10 Users by Degree Centrality")
    if not centrality_df.empty:
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.dataframe(centrality_df.head(10), use_container_width=True, hide_index=True)
        with c_right:
            fig_cent = px.bar(
                centrality_df.head(10),
                x='User',
                y='Degree Centrality',
                color='Degree Centrality',
                title="Degree Centrality Ranks"
            )
            st.plotly_chart(fig_cent, use_container_width=True)
    else:
        st.warning("No interaction mentions found to calculate degree centrality.")


# ==========================================
# TAB 6: COMMUNITY DETECTION
# ==========================================
with tabs[5]:
    st.subheader("🧩 Community Detection Analysis")
    st.info("💡 **Community Detection Concept**: Community detection identifies groups of users with relatively dense connections using NetworkX **Greedy Modularity Communities** algorithm.")
    
    G = build_social_network_graph(df_analyzed)
    comm_summary, comm_df, raw_communities = detect_communities(G)
    
    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Communities Detected", comm_summary['num_communities'])
    cm2.metric("Largest Community Size", comm_summary['largest_community_size'])
    cm3.metric("Smallest Community Size", comm_summary['smallest_community_size'])
    
    st.markdown("---")
    st.subheader("🎨 Community Partition Network Visualization")
    comm_fig = generate_community_network_fig(G, raw_communities, top_n_nodes=40)
    st.plotly_chart(comm_fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("👥 User Community Assignment Table")
    if not comm_df.empty:
        selected_comm = st.selectbox("Filter by Community:", options=["All"] + sorted(comm_df['Community ID'].unique().tolist()))
        if selected_comm != "All":
            filtered_comm_df = comm_df[comm_df['Community ID'] == selected_comm]
        else:
            filtered_comm_df = comm_df
        st.dataframe(filtered_comm_df, use_container_width=True, hide_index=True)


# ==========================================
# TAB 7: DATA EXPLORER
# ==========================================
with tabs[6]:
    st.subheader("🔍 Interactive Data Explorer & Filter")
    
    e_col1, e_col2, e_col3, e_col4 = st.columns(4)
    with e_col1:
        sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Neutral", "Negative"])
    with e_col2:
        user_search = st.text_input("Search User ID:", "")
    with e_col3:
        hashtag_search = st.text_input("Search Hashtag:", "")
    with e_col4:
        sort_by = st.selectbox("Sort By:", ["engagement", "likes", "shares", "comments", "compound_score"])
        
    df_exp = df_analyzed.copy()
    
    if sentiment_filter != "All":
        df_exp = df_exp[df_exp['sentiment'] == sentiment_filter]
        
    if user_search.strip():
        df_exp = df_exp[df_exp['user_id'].str.contains(user_search.strip(), case=False, na=False)]
        
    if hashtag_search.strip():
        df_exp = df_exp[df_exp['hashtags'].str.contains(hashtag_search.strip(), case=False, na=False)]
        
    df_exp = df_exp.sort_values(by=sort_by, ascending=False)
    
    st.write(f"Showing **{len(df_exp)}** matching records:")
    st.dataframe(df_exp[['user_id', 'post', 'timestamp', 'likes', 'comments', 'shares', 'sentiment', 'compound_score', 'hashtags', 'engagement']], use_container_width=True)
    
    st.markdown("---")
    st.subheader("📥 Export Processed Dataset")
    csv_bytes = convert_df_to_csv(df_exp)
    st.download_button(
        label="📄 Download Processed Dataset (CSV)",
        data=csv_bytes,
        file_name="processed_social_media.csv",
        mime="text/csv"
    )


# ==========================================
# TAB 8: ABOUT PROJECT & SYLLABUS MAPPING
# ==========================================
with tabs[7]:
    st.subheader("ℹ️ About This Project")
    
    st.markdown("""
    ### Project Title:
    **"Social Media Sentiment and Trend Analysis Using Big Data Technologies"**

    #### 🎯 Problem Statement:
    Social media platforms generate vast volumes of unstructured text and user interaction data in real time. Manually extracting actionable insights, public sentiment trends, and influencer network dynamics from this data is unfeasible. This project addresses this challenge by employing **Big Data Analytics frameworks (PySpark)**, **NoSQL Storage (MongoDB)**, **Natural Language Processing (VADER Sentiment)**, and **Social Network Analysis (NetworkX)** to analyze post engagement, hashtag popularity, sentiment distribution, and user community structures.

    ---

    #### 📌 Objectives:
    1. **Textual Data Analysis**: Process social media post text and normalize dataset variations.
    2. **Sentiment Classification**: Categorize posts into Positive, Neutral, and Negative using VADER compound scoring.
    3. **Hashtag & Trend Identification**: Rank popular hashtags and compute dynamic Trending Scores (`Likes + Comments + Shares`).
    4. **User Activity & Engagement Analytics**: Identify top active users and quantify audience interaction metrics.
    5. **Social Interaction Network**: Construct a directed graph mapping user-to-user mention networks.
    6. **Degree Centrality Calculation**: Measure user connectedness and centrality within the interaction graph.
    7. **Community Detection**: Partition the user network into distinct clusters using Greedy Modularity algorithms.
    8. **Distributed Big Data Processing**: Execute schema inspection and aggregate metrics using Apache Spark DataFrames.
    9. **NoSQL Database Storage**: Store processed records and analytical summaries in MongoDB collections (`posts` and `analysis_results`).
    10. **Interactive Data Visualization**: Present real-time analytics dashboards via Streamlit and Plotly.

    ---

    #### 🎓 Mapping to Big Data Analytics Lab Syllabus:

    | Syllabus Topic | Project Implementation & Proof of Concept |
    | :--- | :--- |
    | **MongoDB / NoSQL Databases** | Integrated PyMongo document handler storing processed social media records in `social_media_db.posts` and summary metrics in `social_media_db.analysis_results` with graceful offline fallback. |
    | **PySpark / Spark Engine** | Initialized `SparkSession` in `local[*]` mode. Performed DataFrame operations: `select`, `filter`, `groupBy`, `count`, `sum`, `avg`, `max`, `orderBy`, and `withColumn`. |
    | **Descriptive Analytics & Statistics** | Calculated statistical summaries, average likes/shares/comments, maximum engagement metrics, and temporal distributions. |
    | **Social Network Analysis (SNA)** | Created directed interaction graph using `NetworkX`, computed node/edge counts, average degree, density, and **Degree Centrality**. |
    | **Community Detection Algorithm** | Implemented **Greedy Modularity Community Partitioning** to group densely connected user subgraphs. |
    | **Data Visualization** | Rendered interactive Plotly donut charts, time-series lines, 2D network graphs, and Streamlit KPI metrics. |
    | **Mini Project Application** | Fully functional local Big Data analytics mini-project application ready for college demonstration. |

    ---

    #### 🛠️ Technology Stack:
    - **Language**: Python 3.10+
    - **Big Data Processing Engine**: PySpark (Apache Spark 3.x)
    - **NoSQL Database**: MongoDB / PyMongo
    - **NLP & Sentiment**: NLTK / VADER Sentiment Intensity Analyzer
    - **Graph Mining**: NetworkX
    - **Dashboard UI**: Streamlit
    - **Data Visualization**: Plotly Express & Graph Objects
    - **Data Manipulation**: Pandas & NumPy
    """)

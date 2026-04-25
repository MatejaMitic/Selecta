import time
import os
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Creative Portfolio Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
* {
    --radius: 12px;
    --radius-compact: 8px;
    --bg-light: #FFFFFF;
    --bg-slightly-lighter: #F5F5F5;
    --text-primary: #1A1A1A;
    --text-secondary: rgba(26, 26, 26, 0.65);
    --text-tertiary: rgba(26, 26, 26, 0.45);
    --border-light: #D0D0CE;
    --accent: #5B4FD6;
    --accent-dark: #4A3FC4;
    --success: #0D6F44;
    --warning: #C17600;
    --error: #B70000;
    --transition: 0.12s ease;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg-light);
}

[data-testid="stSidebar"] {
    background: var(--bg-slightly-lighter);
}

html, body, [data-testid="stMarkdownContainer"] {
    background: var(--bg-light);
    color: var(--text-primary);
}

main {
    background: var(--bg-light);
}

main .block-container {
    padding: 32px 24px;
    max-width: 1400px;
    background: var(--bg-light);
}

h1, h2, h3, h4, h5, h6 {
    font-weight: 500;
    letter-spacing: -0.02em;
    line-height: 1.3;
    color: var(--text-primary);
}

h1 {
    font-size: 28px;
    margin: 0 0 16px 0;
}

h2 {
    font-size: 22px;
    margin: 0 0 12px 0;
}

h3 {
    font-size: 18px;
    margin: 0 0 12px 0;
}

p {
    margin: 0;
    line-height: 1.6;
    color: var(--text-secondary);
    font-size: 14px;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 2px solid var(--border-light);
    gap: 32px;
    padding: 0;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 400;
    color: var(--text-secondary);
    padding: 12px 0;
    border-bottom: 3px solid transparent;
    transition: color var(--transition), border-color var(--transition);
}

.stTabs [aria-selected="true"] {
    border-bottom-color: var(--accent);
    color: var(--text-primary);
    background: transparent;
    font-weight: 500;
}

.recommendation-prune {
    background: #FEF3F3;
    border-left: 3px solid var(--error);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-secondary);
}

.recommendation-pursue {
    background: #F0FAF7;
    border-left: 3px solid var(--success);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-secondary);
}

.recommendation-review {
    background: #FEF7EC;
    border-left: 3px solid var(--warning);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-secondary);
}

.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stNumberInput > div > div > input {
    border-radius: var(--radius-compact);
    border: 1px solid var(--border-light);
    background: var(--bg-light);
    color: var(--text-primary);
    font-size: 14px;
    padding: 10px 12px;
    transition: border-color var(--transition), box-shadow var(--transition);
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {
    border-color: var(--accent);
    outline: none;
    box-shadow: 0 0 0 3px rgba(91, 79, 214, 0.12);
}

.stButton > button {
    background-color: var(--accent);
    color: #FFFFFF;
    border: none;
    border-radius: var(--radius);
    font-weight: 500;
    font-size: 14px;
    padding: 10px 20px;
    height: 40px;
    transition: background-color var(--transition), opacity var(--transition);
}

.stButton > button:hover {
    background-color: var(--accent-dark);
}

.stButton > button:active {
    opacity: 0.92;
}

[data-testid="metric-container"] {
    background: var(--bg-slightly-lighter);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    padding: 16px;
}

[data-testid="stMetricValue"] {
    font-weight: 500;
    font-size: 24px;
    color: var(--text-primary);
}

[data-testid="stMetricDelta"] {
    font-size: 12px;
    color: var(--text-tertiary);
}

[data-testid="stMetricLabel"] {
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: 400;
}

.stDivider {
    border: none;
    height: 1px;
    background-color: var(--border-light);
    margin: 24px 0;
}

[data-testid="stAlert"] {
    background: var(--bg-slightly-lighter);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    color: var(--text-primary);
}

.stSlider > div > div > div {
    color: var(--accent);
}

.stMultiSelect [data-baseweb="tag"] {
    background-color: var(--accent);
}

.stMultiSelect > div > div > div {
    border: 1px solid var(--border-light);
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

DATA_PATH = Path("Smadex_Creative_Intelligence_Dataset_FULL")

@st.cache_data
def load_smadex_data():
    """Load Smadex dataset and generate mock pruning analysis."""
    try:
        creatives = pd.read_csv(DATA_PATH / "creatives.csv")
        campaigns = pd.read_csv(DATA_PATH / "campaigns.csv")
        creative_summary = pd.read_csv(DATA_PATH / "creative_summary.csv")
        
        # Merge data
        df = creatives.merge(campaigns[["campaign_id", "daily_budget_usd"]], on="campaign_id", how="left")
        df = df.merge(creative_summary[["creative_id", "total_clicks", "total_impressions", "total_conversions", "total_revenue_usd", "overall_ctr", "creative_status"]], 
                     on="creative_id", how="left")
        
        # Calculate conversion rate for revenue proxy
        df["conv_rate"] = df["total_conversions"] / (df["total_clicks"] + 1)
        df["revenue_proxy"] = df["total_revenue_usd"].fillna(0)
        df["ctr"] = df["overall_ctr"].fillna(0)
        
        return df, creatives, campaigns, creative_summary
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

@st.cache_data
def generate_pruning_tree(df):
    """Generate mock pruning analysis with tree structure."""
    np.random.seed(42)
    
    # Group creatives by campaign for tree structure
    campaigns = df["campaign_id"].unique()
    
    tree_nodes = []
    
    for campaign_id in campaigns[:15]:  # Limit to 15 campaigns for UI
        campaign_creatives = df[df["campaign_id"] == campaign_id].copy()
        
        for idx, row in campaign_creatives.iterrows():
            # Generate mock metrics vector similarity
            similarity = np.random.uniform(0.3, 0.99)
            
            # Calculate whether to prune based on revenue and similarity
            revenue = row["revenue_proxy"]
            revenue_norm = (revenue - df["revenue_proxy"].min()) / (df["revenue_proxy"].max() - df["revenue_proxy"].min() + 1)
            
            # Higher revenue = more tolerance for similarity
            tolerance = 0.5 + (revenue_norm * 0.35)
            
            if similarity > tolerance:
                recommendation = "PRUNE"
                reason = f"Market saturation: {similarity:.0%} similar to existing creative. Recommend pausing to reduce budget waste."
            elif similarity > tolerance * 0.85:
                recommendation = "REVIEW"
                reason = f"High market overlap: {similarity:.0%} similar creatives. Monitor performance before scaling spend."
            else:
                recommendation = "PURSUE"
                reason = f"Low market saturation: {similarity:.0%} similarity. Good candidate for scaling investment."
            
            tree_nodes.append({
                "creative_id": row["creative_id"],
                "campaign_id": campaign_id,
                "app_name": row["app_name"],
                "theme": row["theme"],
                "format": row["format"],
                "language": row["language"],
                "similarity": similarity,
                "tolerance": tolerance,
                "revenue_proxy": revenue,
                "ctr": row["ctr"],
                "conv_rate": row["conv_rate"],
                "recommendation": recommendation,
                "reason": reason,
                "creative_path": row.get("creative_path", ""),
            })
    
    return pd.DataFrame(tree_nodes)

def render_splash():
    """Render loading splash screen."""
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            "<div style='padding: 64px; text-align: center;'>"
            "<h1 style='color: #1A1A1A; font-size: 32px; margin-bottom: 16px;'>Creative Portfolio Analysis</h1>"
            "<p style='color: rgba(26, 26, 26, 0.6); font-size: 15px;'>Loading analysis...</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        progress = st.progress(0)
        for percent in range(0, 101, 5):
            time.sleep(0.08)
            progress.progress(percent)
    time.sleep(0.25)
    placeholder.empty()

def render_header():
    st.markdown(
        "<div style='padding: 48px 0; margin-bottom: 40px; border-bottom: 1px solid #E4E4E2;'>"
        "<h1 style='margin-bottom: 12px; color: #1A1A1A;'>Creative Portfolio Analysis</h1>"
        "<p style='color: rgba(26, 26, 26, 0.6); font-size: 15px; line-height: 1.6; max-width: 600px;'>Identify which creatives to continue investing in and which to retire based on market similarity and financial performance.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

def render_tree_view(tree_df, filter_recommendation=None, min_revenue=0, max_similarity=1.0):
    """Render the pruning tree view."""
    
    # Apply filters
    filtered = tree_df.copy()
    if filter_recommendation and len(filter_recommendation) > 0:
        filtered = filtered[filtered["recommendation"].isin(filter_recommendation)]
    filtered = filtered[filtered["revenue_proxy"] >= min_revenue]
    filtered = filtered[filtered["similarity"] <= max_similarity]
    
    if filtered.empty:
        st.warning("No creatives match the selected filters.")
        return
    
    # Display as organized cards
    sorted_df = filtered.sort_values("similarity", ascending=False)
    
    for idx, row in sorted_df.iterrows():
        if row['recommendation'] == 'PRUNE':
            css_class = "recommendation-prune"
            label = "Pause"
        elif row['recommendation'] == 'REVIEW':
            css_class = "recommendation-review"
            label = "Monitor"
        else:
            css_class = "recommendation-pursue"
            label = "Scale"
        
        col1, col2, col3 = st.columns([0.8, 2, 1.2], gap="medium")
        
        with col1:
            st.markdown(f"<p style='font-weight: 500; font-size: 14px; color: #1A1A1A;'>{label}</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<p style='font-weight: 500; font-size: 14px; color: #1A1A1A; margin-bottom: 4px;'>{row['app_name']}</p><p style='font-size: 13px; color: rgba(26, 26, 26, 0.6);'>{row['theme']} · {row['format']}</p>", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"<p style='font-size: 13px; color: rgba(26, 26, 26, 0.6);'><strong style='color: #1A1A1A;'>${row['revenue_proxy']:.0f}</strong> revenue</p><p style='font-size: 13px; color: rgba(26, 26, 26, 0.6);'>{row['similarity']:.0%} similarity</p>", unsafe_allow_html=True)
        
        st.markdown(
            f"<div class='{css_class}'>"
            f"{row['reason']}"
            f"</div>",
            unsafe_allow_html=True
        )
        st.divider()

def render_analytics(tree_df):
    """Render analytics summary."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prune_count = len(tree_df[tree_df["recommendation"] == "PRUNE"])
        st.metric("Pause", prune_count, f"{prune_count/len(tree_df)*100:.0f}%")
    
    with col2:
        pursue_count = len(tree_df[tree_df["recommendation"] == "PURSUE"])
        st.metric("Scale", pursue_count, f"{pursue_count/len(tree_df)*100:.0f}%")
    
    with col3:
        review_count = len(tree_df[tree_df["recommendation"] == "REVIEW"])
        st.metric("Monitor", review_count, f"{review_count/len(tree_df)*100:.0f}%")
    
    with col4:
        total_revenue = tree_df["revenue_proxy"].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")

def main():
    render_splash()
    render_header()
    
    # Load data
    df, creatives, campaigns, creative_summary = load_smadex_data()
    
    if df is None:
        st.error("Failed to load Smadex dataset from workspace.")
        return
    
    # Generate tree analysis
    tree_df = generate_pruning_tree(df)
    
    # Sidebar controls
    st.sidebar.markdown("### Filters", help=None)
    
    filter_rec = st.sidebar.multiselect(
        "Status",
        ["PRUNE", "PURSUE", "REVIEW"],
        default=["PRUNE", "PURSUE", "REVIEW"]
    )
    
    min_revenue = st.sidebar.slider(
        "Minimum Revenue ($)",
        0.0,
        tree_df["revenue_proxy"].max(),
        0.0,
        step=100.0
    )
    
    max_sim = st.sidebar.slider(
        "Maximum Similarity",
        0.3,
        1.0,
        1.0,
        step=0.05
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size: 13px; line-height: 1.6; color: rgba(26, 26, 26, 0.6);'>"
        "<p style='margin: 0 0 8px 0; font-weight: 500; color: #1A1A1A;'>How recommendations work</p>"
        "<p style='margin: 0 0 12px 0;'><strong>Pause:</strong> High market overlap with existing campaigns. Reduces wasted spend.</p>"
        "<p style='margin: 0 0 12px 0;'><strong>Monitor:</strong> Overlap detected. Track performance before scaling.</p>"
        "<p style='margin: 0;'><strong>Scale:</strong> Low market saturation. Recommended for investment.</p>"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Recommendations", "Analytics", "Details"])
    
    with tab1:
        st.markdown("### Portfolio Recommendations")
        render_tree_view(
            tree_df,
            filter_recommendation=filter_rec if len(filter_rec) > 0 else None,
            min_revenue=min_revenue,
            max_similarity=max_sim
        )
    
    with tab2:
        st.markdown("### Summary")
        render_analytics(tree_df)
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### Market Saturation")
            st.bar_chart(tree_df["similarity"].value_counts().sort_index())
        with col2:
            st.markdown("#### Revenue by Status")
            rev_by_rec = tree_df.groupby("recommendation")["revenue_proxy"].sum()
            st.bar_chart(rev_by_rec)
    
    with tab3:
        st.markdown("### All Creatives")
        display_df = tree_df[["app_name", "theme", "format", "similarity", "revenue_proxy", "ctr", "recommendation"]].copy()
        display_df["similarity"] = display_df["similarity"].map(lambda x: f"{x:.0%}")
        display_df["revenue_proxy"] = display_df["revenue_proxy"].map(lambda x: f"${x:.0f}")
        display_df["ctr"] = display_df["ctr"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df, use_container_width=True, height=600)


if __name__ == "__main__":
    main()

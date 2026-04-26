"""
Selecta-style UI wired to VisualAddSimilarity backend (main mock + boost + similarity).

Run locally from this repo root:
  .\\scripts\\run_local_ui.ps1
  # or: py -m pip install -e ".[ui]" && py -m visual_add_similarity.streamlit_launcher

Local dev only (optional): set `SMADEX_DATASET` to a zip or Smadex folder; otherwise the app uses the bundled dataset when present.
"""
from __future__ import annotations

import html
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import hdbscan

from visual_add_similarity.backend import (
    BackendWeights,
    SmadexDatasetRepository,
    run_backend_for_application,
)


def _resolve_dataset_path() -> str:
    """
    Workspace data location (not shown in UI). Production: replace with your API/DB layer.
    Resolution: SMADEX_DATASET env, then bundled selecta-reference folder for local demos.
    """
    env = os.environ.get("SMADEX_DATASET", "").strip()
    if env and Path(env).exists():
        return env
    ref = Path(__file__).resolve().parent / "Smadex_Creative_Intelligence_Dataset_FULL"
    if ref.is_dir():
        return str(ref)
    return ""


STATUS_LABELS = {"PRUNE": "Hold", "REVIEW": "Watch", "PURSUE": "Grow"}
STATUS_KEYS = {"Hold": "PRUNE", "Watch": "REVIEW", "Grow": "PURSUE"}


def _short_reason(recommendation: str) -> str:
    if recommendation == "PRUNE":
        return "Looks like creatives that are already earning more."
    if recommendation == "REVIEW":
        return "Close to something that is working — check before you scale."
    return "Visually distinct from what is winning — good candidate to lean on."


STYLE = """
<style>
  :root {
    --bg: #fbfbfd;
    --surface: #ffffff;
    --sidebar: #f5f5f7;
    --text: #1d1d1f;
    --text2: #6e6e73;
    --text3: #86868b;
    --line: rgba(0, 0, 0, 0.08);
    --accent: #0071e3;
    --accent-hover: #0077ed;
    --radius: 12px;
    --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.06);
    --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: var(--font);
    -webkit-font-smoothing: antialiased;
  }

  [data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: 1px solid var(--line) !important;
  }

  [data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text2);
    margin: 20px 0 8px 0;
    text-transform: none;
  }

  main .block-container {
    padding: 28px 32px 48px;
    max-width: 1080px;
  }

  h1, h2, h3 {
    font-family: var(--font);
    font-weight: 600;
    letter-spacing: -0.022em;
    color: var(--text);
  }

  h1 { font-size: 28px; line-height: 1.15; margin: 0 0 6px 0; font-weight: 600; }
  h2 { font-size: 22px; line-height: 1.2; margin: 0 0 8px 0; font-weight: 600; }
  h3 { font-size: 17px; line-height: 1.25; margin: 0 0 12px 0; font-weight: 600; }

  p, span, label { color: var(--text2); }

  .hero-sub {
    font-size: 15px;
    line-height: 1.45;
    color: var(--text3);
    margin: 0 0 28px 0;
    max-width: 42em;
  }

  [data-testid="stTabs"] { margin-top: 8px; }
  [data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--line);
    background: transparent;
    padding: 0;
  }
  [data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 500;
    color: var(--text3);
    padding: 10px 14px;
    border-radius: 8px 8px 0 0;
    border: none !important;
  }
  [data-testid="stTabs"] [aria-selected="true"] {
    color: var(--text) !important;
    font-weight: 600;
    background: var(--surface) !important;
    box-shadow: 0 -1px 0 var(--surface);
    border-bottom: 2px solid var(--accent) !important;
  }

  .stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 980px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.35rem !important;
    transition: background 0.15s ease !important;
  }
  .stButton > button:hover { background: var(--accent-hover) !important; }

  [data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
  [data-testid="stNumberInput"] input {
    border-radius: 10px !important;
    border: 1px solid var(--line) !important;
    background: var(--surface) !important;
    font-size: 14px !important;
  }

  [data-testid="stMetricValue"] {
    font-weight: 600 !important;
    font-size: 26px !important;
    letter-spacing: -0.02em;
    color: var(--text) !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text3) !important;
    text-transform: none !important;
  }

  [data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    padding: 16px 18px !important;
    box-shadow: none !important;
  }

  .cu-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 16px 18px;
    margin-bottom: 10px;
    box-shadow: var(--shadow);
  }
  .cu-card-hold { border-left: 3px solid #aeaeb2; }
  .cu-card-watch { border-left: 3px solid #d4a574; }
  .cu-card-grow { border-left: 3px solid #6bab90; }

  .cu-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 6px;
  }
  .cu-title {
    font-size: 15px;
    font-weight: 500;
    color: var(--text);
    letter-spacing: -0.015em;
    line-height: 1.35;
  }
  .cu-meta {
    font-size: 13px;
    font-weight: 500;
    color: var(--text2);
    text-align: right;
    white-space: nowrap;
  }
  .cu-reason {
    font-size: 13px;
    line-height: 1.5;
    color: var(--text3);
    margin: 10px 0 0 0;
  }

  hr { border: none; border-top: 1px solid var(--line); margin: 20px 0; }

  [data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    background: var(--surface);
    margin-bottom: 8px;
  }

  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header[data-testid="stHeader"] { background: var(--bg); border-bottom: 1px solid var(--line); }
</style>
"""


def load_merged_df(dataset_path: str) -> Optional[pd.DataFrame]:
    p = Path(dataset_path)
    try:
        if p.is_file() and p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p, "r") as zf:
                creatives = pd.read_csv(zf.open("creatives.csv"))
                campaigns = pd.read_csv(zf.open("campaigns.csv"))
                creative_summary = pd.read_csv(zf.open("creative_summary.csv"))
        elif p.is_dir():
            creatives = pd.read_csv(p / "creatives.csv")
            campaigns = pd.read_csv(p / "campaigns.csv")
            creative_summary = pd.read_csv(p / "creative_summary.csv")
        else:
            return None
    except Exception:
        return None

    df = creatives.merge(
        campaigns[["campaign_id", "daily_budget_usd"]], on="campaign_id", how="left"
    )
    df = df.merge(
        creative_summary[
            [
                "creative_id",
                "total_clicks",
                "total_impressions",
                "total_conversions",
                "total_revenue_usd",
                "overall_ctr",
                "creative_status",
            ]
        ],
        on="creative_id",
        how="left",
    )
    df["conv_rate"] = df["total_conversions"] / (df["total_clicks"] + 1)
    df["revenue_proxy"] = df["total_revenue_usd"].fillna(0)
    df["ctr"] = df["overall_ctr"].fillna(0)
    return df


@st.cache_data(show_spinner="Scoring portfolio…")
def build_tree_from_backend(
    dataset_path: str,
    max_campaigns: int,
    main_w: float,
    boost_w: float,
    sim_w: float,
) -> pd.DataFrame:
    repo = SmadexDatasetRepository(dataset_path)
    weights = BackendWeights(
        main_weight=main_w,
        boost_weight=boost_w,
        similarity_penalty_weight=sim_w,
    )
    df = load_merged_df(dataset_path)
    if df is None or df.empty:
        return pd.DataFrame()

    rev_min = float(df["revenue_proxy"].min())
    rev_max = float(df["revenue_proxy"].max())
    rev_rng = max(rev_max - rev_min, 1e-8)

    campaign_ids = sorted(df["campaign_id"].unique().tolist())[:max_campaigns]
    nodes: List[dict] = []

    for campaign_id in campaign_ids:
        try:
            result = run_backend_for_application(
                str(int(campaign_id)),
                repository=repo,
                weights=weights,
            )
        except (ValueError, FileNotFoundError):
            continue

        sub = df[df["campaign_id"] == campaign_id]
        for _, row in sub.iterrows():
            ad_id = str(int(row["creative_id"]))
            if ad_id not in result.final_scores:
                continue
            revenue = float(row["revenue_proxy"])
            revenue_norm = (revenue - rev_min) / rev_rng
            sim_penalty = float(result.similarity_penalties.get(ad_id, 0.0))
            similarity = sim_penalty
            tolerance = 0.5 + (revenue_norm * 0.35)
            score = float(result.final_scores[ad_id])

            if similarity > tolerance:
                recommendation = "PRUNE"
            elif similarity > tolerance * 0.85:
                recommendation = "REVIEW"
            else:
                recommendation = "PURSUE"
            reason = _short_reason(recommendation)

            nodes.append(
                {
                    "creative_id": row["creative_id"],
                    "campaign_id": campaign_id,
                    "app_name": row["app_name"],
                    "theme": row.get("theme", ""),
                    "format": row.get("format", ""),
                    "language": row.get("language", ""),
                    "similarity": similarity,
                    "tolerance": tolerance,
                    "revenue_proxy": revenue,
                    "ctr": row["ctr"],
                    "conv_rate": row["conv_rate"],
                    "recommendation": recommendation,
                    "reason": reason,
                    "creative_path": str(row.get("asset_file", "")),
                    "portfolio_score": score,
                }
            )

    return pd.DataFrame(nodes)


def _list_app_names(df: pd.DataFrame) -> List[str]:
    return sorted(df["app_name"].astype(str).dropna().unique().tolist(), key=str.lower)


@st.cache_data(show_spinner="Analyzing…")
def build_tree_for_single_app(
    dataset_path: str,
    app_name_query: str,
    main_w: float,
    boost_w: float,
    sim_w: float,
) -> pd.DataFrame:
    """Run backend for one app name (same contract as portfolio builder)."""
    app_key = (app_name_query or "").strip()
    if not app_key:
        return pd.DataFrame()

    repo = SmadexDatasetRepository(dataset_path)
    weights = BackendWeights(
        main_weight=main_w,
        boost_weight=boost_w,
        similarity_penalty_weight=sim_w,
    )
    df = load_merged_df(dataset_path)
    if df is None or df.empty:
        return pd.DataFrame()

    mask = df["app_name"].astype(str).str.strip().str.lower() == app_key.lower()
    if not mask.any():
        return pd.DataFrame()

    try:
        result = run_backend_for_application(app_key, repository=repo, weights=weights)
    except (ValueError, FileNotFoundError):
        return pd.DataFrame()

    rev_min = float(df["revenue_proxy"].min())
    rev_max = float(df["revenue_proxy"].max())
    rev_rng = max(rev_max - rev_min, 1e-8)

    subset = df.loc[mask]
    nodes: List[dict] = []
    for _, row in subset.iterrows():
        ad_id = str(int(row["creative_id"]))
        if ad_id not in result.final_scores:
            continue
        revenue = float(row["revenue_proxy"])
        revenue_norm = (revenue - rev_min) / rev_rng
        sim_penalty = float(result.similarity_penalties.get(ad_id, 0.0))
        similarity = sim_penalty
        tolerance = 0.5 + (revenue_norm * 0.35)
        score = float(result.final_scores[ad_id])

        if similarity > tolerance:
            recommendation = "PRUNE"
        elif similarity > tolerance * 0.85:
            recommendation = "REVIEW"
        else:
            recommendation = "PURSUE"
        reason = _short_reason(recommendation)

        nodes.append(
            {
                "creative_id": row["creative_id"],
                "campaign_id": row["campaign_id"],
                "app_name": row["app_name"],
                "theme": row.get("theme", ""),
                "format": row.get("format", ""),
                "language": row.get("language", ""),
                "similarity": similarity,
                "tolerance": tolerance,
                "revenue_proxy": revenue,
                "ctr": row["ctr"],
                "conv_rate": row["conv_rate"],
                "recommendation": recommendation,
                "reason": reason,
                "creative_path": str(row.get("asset_file", "")),
                "portfolio_score": score,
            }
        )

    return pd.DataFrame(nodes)


def _init_history() -> None:
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    if "last_search_snapshot" not in st.session_state:
        st.session_state.last_search_snapshot = None
    if "last_cluster_snapshot" not in st.session_state:
        st.session_state.last_cluster_snapshot = None


def _append_history(entry: Dict[str, Any]) -> None:
    _init_history()
    hist: List[Dict[str, Any]] = st.session_state.analysis_history
    hist.insert(0, entry)
    st.session_state.analysis_history = hist[:50]


@st.cache_data(show_spinner=False)
def _build_cluster_for_filters(
    dataset_path: str,
    app_name: str,
    country: Optional[str],
    os_name: Optional[str],
    language: Optional[str],
) -> pd.DataFrame:
    root = Path(dataset_path)
    if root.is_file() and root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root, "r") as zf:
            creatives_df = pd.read_csv(zf.open("creatives.csv"))
            daily_df = pd.read_csv(zf.open("creative_daily_country_os_stats.csv"))
    else:
        creatives_df = pd.read_csv(root / "creatives.csv")
        daily_df = pd.read_csv(root / "creative_daily_country_os_stats.csv")

    app_creatives = creatives_df[creatives_df["app_name"] == app_name].copy()
    if app_creatives.empty:
        return pd.DataFrame()

    # Language filter (auto picks mode).
    target_language = language
    if "language" in app_creatives.columns:
        if not target_language:
            target_language = app_creatives["language"].mode(dropna=True).iloc[0]
        app_creatives = app_creatives[app_creatives["language"] == target_language]
    creative_ids = app_creatives["creative_id"]
    if creative_ids.empty:
        return pd.DataFrame()

    filtered_stats = daily_df[daily_df["creative_id"].isin(creative_ids)].copy()
    if filtered_stats.empty:
        return pd.DataFrame()

    # Country/OS filters (auto picks most frequent pair for comparability).
    target_country = country
    target_os = os_name
    if not target_country or not target_os:
        pair = filtered_stats[["country", "os"]].value_counts().index[0]
        target_country = target_country or pair[0]
        target_os = target_os or pair[1]

    filtered_stats = filtered_stats[
        (filtered_stats["country"] == target_country) & (filtered_stats["os"] == target_os)
    ].copy()
    if filtered_stats.empty:
        return pd.DataFrame()

    creative_ids = pd.Index(creative_ids)[
        pd.Index(creative_ids).isin(filtered_stats["creative_id"].unique())
    ]
    if creative_ids.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for creative_id in creative_ids:
        c = filtered_stats[filtered_stats["creative_id"] == creative_id]
        rows.append(
            {
                "creative_id": int(creative_id),
                "country": target_country,
                "os": target_os,
                "language": target_language,
                "total_impressions": c["impressions"].sum(),
                "mean_impressions": c["impressions"].mean(),
                "std_impressions": c["impressions"].std(),
                "total_clicks": c["clicks"].sum(),
                "mean_clicks": c["clicks"].mean(),
                "ctr": (c["clicks"].sum() / c["impressions"].sum()) * 100 if c["impressions"].sum() > 0 else 0.0,
                "conversion_rate": (c["conversions"].sum() / c["clicks"].sum()) * 100 if c["clicks"].sum() > 0 else 0.0,
                "total_revenue": c["revenue_usd"].sum(),
                "mean_revenue": c["revenue_usd"].mean(),
                "revenue_per_impression": c["revenue_usd"].sum() / c["impressions"].sum() if c["impressions"].sum() > 0 else 0.0,
                "engagement_rate": ((c["clicks"].sum() + c["conversions"].sum()) / c["impressions"].sum()) * 100 if c["impressions"].sum() > 0 else 0.0,
                "video_completion_rate": (c["video_completions"].sum() / c["impressions"].sum()) * 100 if c["impressions"].sum() > 0 else 0.0,
                "cpc": c["spend_usd"].sum() / c["clicks"].sum() if c["clicks"].sum() > 0 else 0.0,
                "cpa": c["spend_usd"].sum() / c["conversions"].sum() if c["conversions"].sum() > 0 else 0.0,
                "cpm": (c["spend_usd"].sum() / c["impressions"].sum()) * 1000 if c["impressions"].sum() > 0 else 0.0,
                "rpc": c["revenue_usd"].sum() / c["clicks"].sum() if c["clicks"].sum() > 0 else 0.0,
                "revenue_per_conversion": c["revenue_usd"].sum() / c["conversions"].sum() if c["conversions"].sum() > 0 else 0.0,
                "viewability_rate": (c["viewable_impressions"].sum() / c["impressions"].sum()) * 100 if c["impressions"].sum() > 0 else 0.0,
                "click_to_conversion_rate": (c["conversions"].sum() / c["clicks"].sum()) * 100 if c["clicks"].sum() > 0 else 0.0,
            }
        )

    features_df = pd.DataFrame(rows)
    if features_df.empty:
        return pd.DataFrame()

    num = features_df.select_dtypes(include=["float64", "int64"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaled = StandardScaler().fit_transform(num)
    n_components = min(10, scaled.shape[0], scaled.shape[1])
    if n_components < 3:
        return pd.DataFrame()

    pca_data = PCA(n_components=n_components).fit_transform(scaled)
    pca_df = pd.DataFrame(pca_data[:, :3], columns=["PC1", "PC2", "PC3"])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="leaf",
        allow_single_cluster=True,
    )
    labels = clusterer.fit_predict(pca_df[["PC1", "PC2", "PC3"]].to_numpy())
    if (labels == -1).any():
        next_label = (labels.max() + 1) if labels.max() >= 0 else 0
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = next_label
                next_label += 1

    pca_df["cluster"] = labels
    pca_df["creative_id"] = features_df["creative_id"].values
    pca_df["country"] = target_country
    pca_df["os"] = target_os
    pca_df["language"] = target_language
    return pca_df


@st.cache_data(show_spinner=False)
def _get_app_filter_options(dataset_path: str, app_name: str) -> Dict[str, List[str]]:
    root = Path(dataset_path)
    if root.is_file() and root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root, "r") as zf:
            creatives_df = pd.read_csv(zf.open("creatives.csv"), usecols=["creative_id", "app_name", "language"])
            daily_df = pd.read_csv(
                zf.open("creative_daily_country_os_stats.csv"),
                usecols=["creative_id", "country", "os"],
            )
    else:
        creatives_df = pd.read_csv(root / "creatives.csv", usecols=["creative_id", "app_name", "language"])
        daily_df = pd.read_csv(
            root / "creative_daily_country_os_stats.csv",
            usecols=["creative_id", "country", "os"],
        )

    app_creatives = creatives_df[creatives_df["app_name"] == app_name]
    if app_creatives.empty:
        return {"languages": [], "countries": [], "oses": []}
    ids = app_creatives["creative_id"].unique()
    app_daily = daily_df[daily_df["creative_id"].isin(ids)]

    languages = sorted(app_creatives["language"].dropna().astype(str).unique().tolist())
    countries = sorted(app_daily["country"].dropna().astype(str).unique().tolist())
    oses = sorted(app_daily["os"].dropna().astype(str).unique().tolist())
    return {"languages": languages, "countries": countries, "oses": oses}


def render_splash() -> None:
    placeholder = st.empty()
    with placeholder.container():
        bar = st.progress(0)
        for pct in (0.35, 0.7, 1.0):
            time.sleep(0.06)
            bar.progress(pct)
        placeholder.empty()


def render_header() -> None:
    st.markdown(
        '<h1>Portfolio</h1><p class="hero-sub">Hold, watch, or grow — from revenue and how '
        "similar each creative looks to the rest of the set.</p>",
        unsafe_allow_html=True,
    )


def render_tree_view(
    tree_df: pd.DataFrame,
    filter_recommendation: Optional[List[str]],
    min_revenue: float,
    max_similarity: float,
) -> None:
    filtered = tree_df.copy()
    if filter_recommendation:
        filtered = filtered[filtered["recommendation"].isin(filter_recommendation)]
    filtered = filtered[filtered["revenue_proxy"] >= min_revenue]
    filtered = filtered[filtered["similarity"] <= max_similarity]

    if filtered.empty:
        st.caption("Nothing matches these filters. Try widening them in the sidebar.")
        return

    sorted_df = filtered.sort_values("similarity", ascending=False)
    for _, row in sorted_df.iterrows():
        rec = str(row["recommendation"])
        label = STATUS_LABELS.get(rec, rec)
        card = "cu-card-hold" if rec == "PRUNE" else "cu-card-watch" if rec == "REVIEW" else "cu-card-grow"
        title = " · ".join(
            html.escape(str(x))
            for x in (row["app_name"], row.get("theme", ""), row.get("format", ""))
            if str(x).strip()
        )
        rev = float(row["revenue_proxy"])
        if rev >= 1_000_000:
            rev_s = f"{rev / 1_000_000:.1f}M"
        elif rev >= 1000:
            rev_s = f"{rev / 1000:.1f}k"
        else:
            rev_s = f"{rev:.0f}"
        ov = int(round(float(row["similarity"]) * 100))
        sc = f"{float(row['portfolio_score']):.1f}"
        meta = html.escape(f"{rev_s} revenue · {ov}% overlap · score {sc}")
        body = html.escape(str(row["reason"]))
        st.markdown(
            f'<div class="cu-card {card}">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">'
            f'<div><span class="cu-badge">{html.escape(label)}</span>'
            f'<div class="cu-title">{title}</div></div>'
            f'<div class="cu-meta">{meta}</div></div>'
            f'<p class="cu-reason">{body}</p></div>',
            unsafe_allow_html=True,
        )


def render_analytics(tree_df: pd.DataFrame) -> None:
    if tree_df.empty:
        return
    col1, col2, col3, col4 = st.columns(4)
    n = len(tree_df)
    with col1:
        prune_count = int((tree_df["recommendation"] == "PRUNE").sum())
        st.metric("Hold", prune_count, f"{prune_count / n * 100:.0f}%")
    with col2:
        pursue_count = int((tree_df["recommendation"] == "PURSUE").sum())
        st.metric("Grow", pursue_count, f"{pursue_count / n * 100:.0f}%")
    with col3:
        review_count = int((tree_df["recommendation"] == "REVIEW").sum())
        st.metric("Watch", review_count, f"{review_count / n * 100:.0f}%")
    with col4:
        total_revenue = float(tree_df["revenue_proxy"].sum())
        st.metric("Revenue", f"${total_revenue:,.0f}")


def render_cluster_view(cluster_snap: Optional[Dict[str, Any]]) -> None:
    if not cluster_snap:
        st.caption("Run an app search first to generate clusters.")
        return

    cdf = cluster_snap.get("df")
    if cdf is None or cdf.empty:
        st.caption(
            f"No cluster points for {cluster_snap['app_name']} with "
            f"country={cluster_snap['country']}, language={cluster_snap['language']}, os={cluster_snap['os']}."
        )
        return

    st.caption(
        f"{cluster_snap['app_name']} · country={cluster_snap['country']} · "
        f"language={cluster_snap['language']} · os={cluster_snap['os']}"
    )
    fig = px.scatter_3d(
        cdf,
        x="PC1",
        y="PC2",
        z="PC3",
        color=cdf["cluster"].astype(str),
        hover_data=["creative_id", "cluster", "country", "os", "language"],
        title="Creative clusters (PCA 3D + HDBSCAN)",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.9))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=560)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Portfolio",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(STYLE, unsafe_allow_html=True)

    render_splash()
    render_header()

    dataset_path = _resolve_dataset_path()
    if not dataset_path:
        st.error("Creative data is not available. Your administrator needs to connect the workspace.")
        return

    st.sidebar.markdown("### Model mix")
    main_w = st.sidebar.slider("Core score", 0.0, 30.0, 10.0, 0.5)
    boost_w = st.sidebar.slider("Revenue lift", 0.0, 5.0, 1.2, 0.05)
    sim_w = st.sidebar.slider("Similarity", 0.0, 5.0, 1.4, 0.05)
    max_campaigns = st.sidebar.slider("Campaigns", 1, 180, 24, help="How many campaigns to score in the overview.")

    preview = load_merged_df(dataset_path)
    if preview is None:
        st.error("Creative data could not be loaded. Contact support.")
        return

    tree_df = build_tree_from_backend(
        dataset_path,
        max_campaigns=int(max_campaigns),
        main_w=float(main_w),
        boost_w=float(boost_w),
        sim_w=float(sim_w),
    )

    if tree_df.empty:
        st.error("No scores returned. Check that image assets exist for this dataset.")
        return

    st.sidebar.markdown("### Filters")
    filter_labels = st.sidebar.multiselect(
        "Show",
        list(STATUS_KEYS.keys()),
        default=list(STATUS_KEYS.keys()),
    )
    filter_rec = [STATUS_KEYS[k] for k in filter_labels] if filter_labels else list(STATUS_KEYS.values())
    min_revenue = st.sidebar.slider(
        "Min revenue",
        0.0,
        float(tree_df["revenue_proxy"].max()),
        0.0,
        step=100.0,
    )
    max_sim = st.sidebar.slider(
        "Max overlap",
        0.0,
        1.0,
        1.0,
        step=0.05,
    )

    _init_history()

    tab_search, tab_history, tab1, tab2, tab3 = st.tabs(
        ["Search", "History", "Overview", "Summary", "Table"]
    )

    with tab_search:
        st.markdown("### Find an app")
        app_choices = _list_app_names(preview)
        search_query = st.text_input(
            "Name",
            value=st.session_state.get("search_name", ""),
            placeholder="Type an app name",
            help="Matches app name in your data, ignoring case.",
            key="search_name",
        )
        picked = st.selectbox(
            "Or choose",
            options=["—"] + app_choices,
            index=0,
            help="Overrides the text field.",
            key="search_pick",
        )
        effective_preview = (search_query or "").strip()
        if picked and picked != "—":
            effective_preview = picked.strip()

        options = {"languages": [], "countries": [], "oses": []}
        if effective_preview:
            options = _get_app_filter_options(dataset_path, effective_preview)

        st.markdown("#### Audience filters")
        c1, c2, c3 = st.columns(3)
        with c1:
            language_choice = st.selectbox(
                "Language",
                options=["Auto"] + options["languages"] if options["languages"] else ["Auto"],
                index=0,
                key="search_language",
            )
        with c2:
            country_choice = st.selectbox(
                "Country",
                options=["Auto"] + options["countries"] if options["countries"] else ["Auto"],
                index=0,
                key="search_country",
            )
        with c3:
            os_choice = st.selectbox(
                "OS",
                options=["Auto"] + options["oses"] if options["oses"] else ["Auto"],
                index=0,
                key="search_os",
            )

        submitted = st.button("Analyze", key="search_analyze")

        effective_query = (search_query or "").strip()
        if picked and picked != "—":
            effective_query = picked.strip()

        if submitted:
            if not effective_query:
                st.warning("Enter an app name or pick one from the list.")
            else:
                tree_search = build_tree_for_single_app(
                    dataset_path,
                    effective_query,
                    float(main_w),
                    float(boost_w),
                    float(sim_w),
                )
                if tree_search.empty:
                    st.session_state.last_search_snapshot = None
                    st.session_state.last_cluster_snapshot = None
                    st.error("No match. Try the picker or check spelling.")
                    _append_history(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "app_name": effective_query,
                            "ok": False,
                            "tree_df": None,
                            "error": "No creatives found for that name.",
                        }
                    )
                else:
                    resolved_name = str(tree_search["app_name"].iloc[0])
                    lang = None if language_choice == "Auto" else language_choice
                    ctry = None if country_choice == "Auto" else country_choice
                    os_name = None if os_choice == "Auto" else os_choice
                    cluster_df = _build_cluster_for_filters(
                        dataset_path=dataset_path,
                        app_name=resolved_name,
                        country=ctry,
                        os_name=os_name,
                        language=lang,
                    )
                    snap = tree_search.copy()
                    st.session_state.last_search_snapshot = {
                        "app_name": resolved_name,
                        "df": snap,
                    }
                    st.session_state.last_cluster_snapshot = {
                        "app_name": resolved_name,
                        "country": "Auto" if ctry is None else ctry,
                        "language": "Auto" if lang is None else lang,
                        "os": "Auto" if os_name is None else os_name,
                        "df": cluster_df.copy() if not cluster_df.empty else pd.DataFrame(),
                    }
                    st.success(f"{resolved_name} · {len(snap)} creatives")
                    _append_history(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "app_name": resolved_name,
                            "ok": True,
                            "tree_df": snap,
                            "error": None,
                        }
                    )

        snap = st.session_state.get("last_search_snapshot")
        if snap and snap.get("df") is not None and not snap["df"].empty:
            st.divider()
            st.markdown(f"#### {snap['app_name']}")
            st.markdown("##### 3D clusters")
            render_cluster_view(st.session_state.get("last_cluster_snapshot"))
            st.markdown("##### Recommendations")
            render_tree_view(
                snap["df"],
                filter_recommendation=filter_rec if filter_rec else None,
                min_revenue=min_revenue,
                max_similarity=max_sim,
            )

    with tab_history:
        st.markdown("### History")
        hist: List[Dict[str, Any]] = st.session_state.get("analysis_history", [])
        if not hist:
            st.caption("Runs you start from Search show up here.")
        else:
            for i, entry in enumerate(hist):
                title = f"{entry['app_name']} · {entry['timestamp']}"
                if not entry.get("ok", True):
                    title += " · failed"
                with st.expander(title, expanded=(i == 0)):
                    if entry.get("error"):
                        st.error(entry["error"])
                    elif entry.get("tree_df") is not None and not entry["tree_df"].empty:
                        df_h = entry["tree_df"]
                        st.caption(f"{len(df_h)} creatives")
                        render_tree_view(
                            df_h,
                            filter_recommendation=None,
                            min_revenue=0.0,
                            max_similarity=1.0,
                        )
                    else:
                        st.write("No stored results for this entry.")

    with tab1:
        st.markdown("### Overview")
        render_tree_view(
            tree_df,
            filter_recommendation=filter_rec if filter_rec else None,
            min_revenue=min_revenue,
            max_similarity=max_sim,
        )
    with tab2:
        st.markdown("### Summary")
        render_analytics(tree_df)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.caption("Overlap index")
            st.bar_chart(tree_df[["similarity"]])
        with col2:
            st.caption("Revenue by call")
            rev_by = tree_df.groupby("recommendation")["revenue_proxy"].sum().rename(
                index=lambda x: STATUS_LABELS.get(str(x), str(x))
            )
            st.bar_chart(rev_by)

        st.divider()
        st.markdown("### 3D clusters (from Search filters)")
        render_cluster_view(st.session_state.get("last_cluster_snapshot"))
    with tab3:
        st.markdown("### All creatives")
        display_df = tree_df[
            [
                "app_name",
                "theme",
                "format",
                "similarity",
                "revenue_proxy",
                "ctr",
                "portfolio_score",
                "recommendation",
            ]
        ].copy()
        display_df["similarity"] = display_df["similarity"].map(lambda x: f"{x:.0%}")
        display_df["revenue_proxy"] = display_df["revenue_proxy"].map(lambda x: f"${x:.0f}")
        display_df["ctr"] = display_df["ctr"].map(lambda x: f"{x:.2%}")
        display_df["portfolio_score"] = display_df["portfolio_score"].map(lambda x: f"{x:.2f}")
        display_df["recommendation"] = display_df["recommendation"].map(
            lambda x: STATUS_LABELS.get(str(x), str(x))
        )
        st.dataframe(display_df, use_container_width=True, height=600)


if __name__ == "__main__":
    main()

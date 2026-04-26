import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


DATASET_DIR = os.path.join(os.path.dirname(__file__), "Smadex_Creative_Intelligence_Dataset_FULL")


def extract_to_csv(app_name="VaultRun Pay", country=None, os_name=None, language=None):
    # Load creatives.csv to find creative_ids for the specified app_name
    creatives_df = pd.read_csv(os.path.join(DATASET_DIR, "creatives.csv"))

    # Filter creatives for the specified app_name
    app_creatives = creatives_df[creatives_df["app_name"] == app_name].copy()

    # Choose a single language so creatives are comparable
    if "language" in app_creatives.columns and len(app_creatives) > 0:
        target_language = language if language is not None else app_creatives["language"].mode(dropna=True).iloc[0]
        app_creatives = app_creatives[app_creatives["language"] == target_language]
    else:
        target_language = language

    creative_ids = app_creatives["creative_id"]

    # Load creative_daily_country_os_stats.csv and filter rows for the relevant creative_ids
    daily_stats_df = pd.read_csv(os.path.join(DATASET_DIR, "creative_daily_country_os_stats.csv"))
    filtered_stats = daily_stats_df[daily_stats_df['creative_id'].isin(creative_ids)]

    # Choose a single (country, os) so creatives are comparable
    if len(filtered_stats) > 0 and {"country", "os"}.issubset(filtered_stats.columns):
        if country is not None and os_name is not None:
            target_country, target_os = country, os_name
        else:
            target_country, target_os = (
                filtered_stats[["country", "os"]]
                .value_counts()
                .index[0]
            )
        filtered_stats = filtered_stats[(filtered_stats["country"] == target_country) & (filtered_stats["os"] == target_os)]

        # Keep only creatives that actually have data in this country/os
        creative_ids = pd.Index(creative_ids)[pd.Index(creative_ids).isin(filtered_stats["creative_id"].unique())]
    else:
        target_country, target_os = country, os_name

    # Engineer features for each creative
    creative_features = []

    for creative_id in creative_ids:
        creative_data = filtered_stats[filtered_stats['creative_id'] == creative_id]

        # Compute features for the creative
        features = {
            "creative_id": creative_id,
            "country": target_country,
            "os": target_os,
            "language": target_language,
            "total_impressions": creative_data["impressions"].sum(),
            "mean_impressions": creative_data["impressions"].mean(),
            "std_impressions": creative_data["impressions"].std(),
            "total_clicks": creative_data["clicks"].sum(),
            "mean_clicks": creative_data["clicks"].mean(),
            "ctr": (creative_data["clicks"].sum() / creative_data["impressions"].sum()) * 100 if creative_data["impressions"].sum() > 0 else 0,
            "conversion_rate": (creative_data["conversions"].sum() / creative_data["clicks"].sum()) * 100 if creative_data["clicks"].sum() > 0 else 0,
            "total_revenue": creative_data["revenue_usd"].sum(),
            "mean_revenue": creative_data["revenue_usd"].mean(),
            "revenue_per_impression": creative_data["revenue_usd"].sum() / creative_data["impressions"].sum() if creative_data["impressions"].sum() > 0 else 0,
        }

        # Additional marketing features
        features.update({
            "engagement_rate": ((creative_data["clicks"].sum() + creative_data["conversions"].sum()) / creative_data["impressions"].sum()) * 100 if creative_data["impressions"].sum() > 0 else 0,
            "video_completion_rate": (creative_data["video_completions"].sum() / creative_data["impressions"].sum()) * 100 if creative_data["impressions"].sum() > 0 else 0,
            "cpc": creative_data["spend_usd"].sum() / creative_data["clicks"].sum() if creative_data["clicks"].sum() > 0 else 0,
            "cpa": creative_data["spend_usd"].sum() / creative_data["conversions"].sum() if creative_data["conversions"].sum() > 0 else 0,
            "cpm": (creative_data["spend_usd"].sum() / creative_data["impressions"].sum()) * 1000 if creative_data["impressions"].sum() > 0 else 0,
            "rpc": creative_data["revenue_usd"].sum() / creative_data["clicks"].sum() if creative_data["clicks"].sum() > 0 else 0,
            "revenue_per_conversion": creative_data["revenue_usd"].sum() / creative_data["conversions"].sum() if creative_data["conversions"].sum() > 0 else 0,
            "viewability_rate": (creative_data["viewable_impressions"].sum() / creative_data["impressions"].sum()) * 100 if creative_data["impressions"].sum() > 0 else 0,
            "click_to_conversion_rate": (creative_data["conversions"].sum() / creative_data["clicks"].sum()) * 100 if creative_data["clicks"].sum() > 0 else 0,
        })

        creative_features.append(features)

    # Convert to DataFrame for comparison
    creative_features_df = pd.DataFrame(creative_features)

    return creative_features_df
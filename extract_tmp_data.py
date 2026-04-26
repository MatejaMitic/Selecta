import pandas as pd
import numpy as np
import os


DATASET_DIR = os.path.join(os.path.dirname(__file__), "Smadex_Creative_Intelligence_Dataset_FULL")


def _safe_div(numer, denom):
    denom = np.asarray(denom, dtype=float)
    numer = np.asarray(numer, dtype=float)
    out = np.zeros_like(numer, dtype=float)
    mask = denom != 0
    out[mask] = numer[mask] / denom[mask]
    return out


def _linear_slope(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2 or np.all(~np.isfinite(y)):
        return 0.0
    x = np.arange(n, dtype=float)
    y2 = np.where(np.isfinite(y), y, np.nanmean(y))
    vx = x - x.mean()
    vy = y2 - y2.mean()
    denom = np.dot(vx, vx)
    if denom == 0:
        return 0.0
    return float(np.dot(vx, vy) / denom)


def _peak_relative_day(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return 0.0
    idx = int(np.nanargmax(y)) if np.any(np.isfinite(y)) else 0
    return 0.0 if n == 1 else float(idx / (n - 1))


def _auc_normalized(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return 0.0
    y2 = np.where(np.isfinite(y), y, 0.0)
    auc = float(np.trapz(y2, dx=1.0))
    scale = float(np.nanmax(y2)) if np.any(np.isfinite(y2)) else 0.0
    if scale <= 0:
        return 0.0
    # Normalize to [0, ~1] by dividing by (max * (n-1)) trapezoid width
    denom = scale * max(1, (n - 1))
    return float(auc / denom)


def _cv(y):
    y = np.asarray(y, dtype=float)
    mu = float(np.nanmean(y)) if y.size else 0.0
    sd = float(np.nanstd(y)) if y.size else 0.0
    if mu == 0 or not np.isfinite(mu):
        return 0.0
    return float(sd / mu)


def _autocorr_lag(y, lag=7):
    y = np.asarray(y, dtype=float)
    if y.size <= lag:
        return 0.0
    a = y[:-lag]
    b = y[lag:]
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _first_last_ratio(y, k=7, which="first"):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0
    y2 = np.where(np.isfinite(y), y, 0.0)
    total = float(np.sum(y2))
    if total == 0:
        return 0.0
    if which == "first":
        part = float(np.sum(y2[: min(k, y2.size)]))
    else:
        part = float(np.sum(y2[max(0, y2.size - k) :]))
    return float(part / total)


def _changepoint_relative_day(y):
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return 0.0
    y2 = np.where(np.isfinite(y), y, 0.0)
    d = np.abs(np.diff(y2))
    idx = int(np.argmax(d)) + 1  # changepoint after the jump
    return 0.0 if n == 1 else float(idx / (n - 1))


def _max_single_drop(y):
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return 0.0
    y2 = np.where(np.isfinite(y), y, 0.0)
    diffs = np.diff(y2)
    drops = -diffs[diffs < 0]
    return float(np.max(drops)) if drops.size else 0.0


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

    # Engineer time-series features for each creative (CTR/IPM over days)
    creative_features = []

    if len(filtered_stats) == 0:
        return pd.DataFrame()

    df = filtered_stats.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["creative_id", "date"])

    for creative_id in creative_ids:
        g = df[df["creative_id"] == creative_id]
        if len(g) == 0:
            continue

        # Fill missing days in range with zeros so shape features are comparable
        start = g["date"].min()
        end = g["date"].max()
        all_days = pd.date_range(start=start, end=end, freq="D")
        g2 = (
            g.set_index("date")[["impressions", "clicks", "conversions"]]
            .reindex(all_days)
            .fillna(0.0)
        )

        impressions = g2["impressions"].to_numpy(dtype=float)
        clicks = g2["clicks"].to_numpy(dtype=float)
        conv = g2["conversions"].to_numpy(dtype=float)

        ctr = _safe_div(clicks, impressions) * 100.0
        ipm = _safe_div(conv, impressions) * 1000.0

        n = ctr.size
        half = max(1, n // 2)

        ctr_slope = _linear_slope(ctr)
        ctr_slope_first = _linear_slope(ctr[:half])
        ctr_slope_second = _linear_slope(ctr[half:]) if n > half else 0.0
        ctr_slope_ratio = float(ctr_slope_second / ctr_slope_first) if ctr_slope_first != 0 else 0.0

        ctr_peak_rel = _peak_relative_day(ctr)
        ctr_peak = float(np.nanmax(ctr)) if n else 0.0
        ctr_end = float(ctr[-1]) if n else 0.0
        ctr_peak_to_end_ratio = float(ctr_peak / ctr_end) if ctr_end != 0 else 0.0

        ctr_auc_norm = _auc_normalized(ctr)
        ctr_cv = _cv(ctr)
        ctr_autocorr_lag7 = _autocorr_lag(ctr, lag=7)
        ctr_first7_ratio = _first_last_ratio(ctr, k=7, which="first")
        ctr_last7_ratio = _first_last_ratio(ctr, k=7, which="last")
        ctr_cp_rel = _changepoint_relative_day(ctr)
        ctr_max_drop = _max_single_drop(ctr)

        ipm_slope = _linear_slope(ipm)
        ipm_peak_rel = _peak_relative_day(ipm)
        ipm_first7_ratio = _first_last_ratio(ipm, k=7, which="first")
        ipm_last7_ratio = _first_last_ratio(ipm, k=7, which="last")

        total_days_active = int(n)
        log_total_days_active = float(np.log1p(total_days_active))

        creative_features.append(
            {
                "creative_id": creative_id,
                "country": target_country,
                "os": target_os,
                "language": target_language,
                # CTR shape (12 features)
                "ctr_slope": ctr_slope,
                "ctr_slope_first_half": ctr_slope_first,
                "ctr_slope_second_half": ctr_slope_second,
                "ctr_slope_ratio": ctr_slope_ratio,
                "ctr_peak_relative_day": ctr_peak_rel,
                "ctr_peak_to_end_ratio": ctr_peak_to_end_ratio,
                "ctr_auc_normalized": ctr_auc_norm,
                "ctr_cv": ctr_cv,
                "ctr_autocorr_lag7": ctr_autocorr_lag7,
                "ctr_first7d_ratio": ctr_first7_ratio,
                "ctr_last7d_ratio": ctr_last7_ratio,
                "ctr_changepoint_relative_day": ctr_cp_rel,
                "ctr_max_single_drop": ctr_max_drop,
                # IPM confirmation (4 features)
                "ipm_slope": ipm_slope,
                "ipm_peak_relative_day": ipm_peak_rel,
                "ipm_first7d_ratio": ipm_first7_ratio,
                "ipm_last7d_ratio": ipm_last7_ratio,
                # Lifespan context (2 features)
                "total_days_active": total_days_active,
                "log_total_days_active": log_total_days_active,
            }
        )

    return pd.DataFrame(creative_features)
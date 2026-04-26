from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence
import zipfile

import numpy as np
import pandas as pd

from .layer import AdRecord, run_similarity_layer


@dataclass(frozen=True)
class BackendWeights:
    # Main signal is intentionally dominant.
    main_weight: float = 10.0
    boost_weight: float = 1.2
    similarity_penalty_weight: float = 1.4
    clamp_result: bool = True
    min_score: float = -2.0
    max_score: float = 12.0


@dataclass(frozen=True)
class BackendResult:
    application: str
    main_scores: Dict[str, float]
    boost_scores: Dict[str, float]
    similarity_penalties: Dict[str, float]
    final_scores: Dict[str, float]


class SmadexDatasetRepository:
    """
    Resolves ads by app name or campaign id from either a .zip or an unzipped
    folder (same layout as the Selecta repo: creatives.csv + assets/...).
    """

    def __init__(self, dataset_path: str, *, extract_dir: str = "data_cache/assets") -> None:
        self._path = Path(dataset_path)
        self.extract_dir = Path(extract_dir)
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        if self._path.is_file() and self._path.suffix.lower() == ".zip":
            self._mode = "zip"
            self.zip_path = str(self._path)
        elif self._path.is_dir():
            self._mode = "dir"
            self.zip_path = ""
        else:
            raise ValueError(
                f"dataset_path must be a .zip file or a directory: {dataset_path}"
            )
        self._creative_table = self._load_creative_table()

    def _load_creative_table(self) -> pd.DataFrame:
        if self._mode == "zip":
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                creatives = pd.read_csv(
                    zf.open("creatives.csv"),
                    usecols=["creative_id", "campaign_id", "app_name", "asset_file"],
                )
                summary = pd.read_csv(
                    zf.open("creative_summary.csv"),
                    usecols=["creative_id", "total_revenue_usd"],
                )
        else:
            root = self._path
            creatives = pd.read_csv(
                root / "creatives.csv",
                usecols=["creative_id", "campaign_id", "app_name", "asset_file"],
            )
            summary = pd.read_csv(
                root / "creative_summary.csv",
                usecols=["creative_id", "total_revenue_usd"],
            )
        merged = creatives.merge(summary, on="creative_id", how="left")
        merged["total_revenue_usd"] = merged["total_revenue_usd"].fillna(0.0)
        return merged

    def _extract_asset(self, asset_member: str) -> str:
        if self._mode == "dir":
            local = self._path / Path(asset_member)
            if not local.exists():
                raise FileNotFoundError(f"Asset not found: {local}")
            return str(local)

        out_path = self.extract_dir / Path(asset_member).name
        if out_path.exists():
            return str(out_path)

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            with zf.open(asset_member, "r") as src:
                out_path.write_bytes(src.read())
        return str(out_path)

    def get_ads_for_application(self, app_identifier: str) -> List[AdRecord]:
        app_key = app_identifier.strip()
        if not app_key:
            raise ValueError("Application identifier cannot be empty.")

        mask_by_name = self._creative_table["app_name"].astype(str).str.lower() == app_key.lower()
        if app_key.isdigit():
            mask_by_campaign = self._creative_table["campaign_id"].astype(str) == app_key
        else:
            mask_by_campaign = pd.Series(False, index=self._creative_table.index)

        subset = self._creative_table[mask_by_name | mask_by_campaign].copy()
        if subset.empty:
            raise ValueError(
                f"No ads found for application identifier '{app_identifier}'. "
                "Try app_name (string) or campaign_id (number)."
            )

        records: List[AdRecord] = []
        for _, row in subset.iterrows():
            records.append(
                AdRecord(
                    ad_id=str(int(row["creative_id"])),
                    image_path=self._extract_asset(str(row["asset_file"])),
                    revenue=float(row["total_revenue_usd"]),
                )
            )
        return records


def _mock_main_layer(application: str, ad_ids: Sequence[str]) -> Dict[str, float]:
    """
    Main layer mock: deterministic pseudo-random score in [0, 1].
    """
    scores: Dict[str, float] = {}
    for ad_id in ad_ids:
        key = f"{application}:{ad_id}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        raw = int(digest[:8], 16) / 0xFFFFFFFF
        scores[ad_id] = float(raw)
    return scores


def _normalized_revenue_boost(records: Sequence[AdRecord]) -> Dict[str, float]:
    revenues = np.asarray([record.revenue for record in records], dtype=np.float32)
    if revenues.size == 0:
        return {}
    min_rev = float(np.min(revenues))
    max_rev = float(np.max(revenues))
    rng = max(max_rev - min_rev, 1e-8)
    norm = (revenues - min_rev) / rng
    return {record.ad_id: float(norm[i]) for i, record in enumerate(records)}


def run_backend_for_application(
    application: str,
    *,
    repository: SmadexDatasetRepository,
    weights: BackendWeights | None = None,
) -> BackendResult:
    cfg = weights or BackendWeights()
    records = repository.get_ads_for_application(application)
    ad_ids = [record.ad_id for record in records]

    main_scores = _mock_main_layer(application, ad_ids)
    boost_scores = _normalized_revenue_boost(records)
    similarity_penalties = run_similarity_layer(records)

    final_scores: Dict[str, float] = {}
    for ad_id in ad_ids:
        value = (
            cfg.main_weight * main_scores[ad_id]
            + cfg.boost_weight * boost_scores[ad_id]
            - cfg.similarity_penalty_weight * similarity_penalties[ad_id]
        )
        if cfg.clamp_result:
            value = float(np.clip(value, cfg.min_score, cfg.max_score))
        final_scores[ad_id] = float(value)

    return BackendResult(
        application=application,
        main_scores=main_scores,
        boost_scores=boost_scores,
        similarity_penalties=similarity_penalties,
        final_scores=final_scores,
    )


# Backwards-compatible name
SmadexZipAdRepository = SmadexDatasetRepository

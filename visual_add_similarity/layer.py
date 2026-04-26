from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class AdRecord:
    ad_id: str
    image_path: str
    revenue: float


@dataclass(frozen=True)
class SimilarityLayerConfig:
    image_size: Tuple[int, int] = (96, 96)
    color_hist_bins: int = 16
    tiny_grid_size: Tuple[int, int] = (24, 24)
    edge_weight: float = 0.7
    similarity_floor: float = 0.70
    similarity_power: float = 2.0
    revenue_weight: float = 1.0
    epsilon: float = 1e-8


@dataclass(frozen=True)
class SimilarityLayerResult:
    ad_ids: List[str]
    penalties: np.ndarray
    uniqueness: np.ndarray
    similarity_matrix: np.ndarray
    penalty_by_ad: Dict[str, float]
    uniqueness_by_ad: Dict[str, float]


def _read_image_from_path(path: str, size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize(size)
    return np.asarray(rgb, dtype=np.uint8)


def _extract_features(img_rgb: np.ndarray, cfg: SimilarityLayerConfig) -> np.ndarray:
    float_img = img_rgb.astype(np.float32) / 255.0

    # Color distribution.
    color_hist_parts: List[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(
            float_img[:, :, channel],
            bins=cfg.color_hist_bins,
            range=(0.0, 1.0),
            density=True,
        )
        color_hist_parts.append(hist.astype(np.float32))
    color_hist = np.concatenate(color_hist_parts)

    # Spatial appearance.
    tiny = np.asarray(
        Image.fromarray(img_rgb).resize(cfg.tiny_grid_size).convert("RGB"),
        dtype=np.float32,
    ).reshape(-1)
    tiny = tiny / 255.0

    # Edge profile via OpenCV if available; otherwise grayscale gradients.
    if cv2 is not None:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=80, threshold2=160).astype(np.float32) / 255.0
        edge_vec = np.asarray(
            Image.fromarray((edges * 255.0).astype(np.uint8)).resize(cfg.tiny_grid_size),
            dtype=np.float32,
        ).reshape(-1)
        edge_vec = edge_vec / 255.0
    else:
        gray = np.asarray(Image.fromarray(img_rgb).convert("L"), dtype=np.float32) / 255.0
        grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        grad_mag = np.clip(grad_x + grad_y, 0.0, 1.0)
        edge_vec = np.asarray(
            Image.fromarray((grad_mag * 255.0).astype(np.uint8)).resize(cfg.tiny_grid_size),
            dtype=np.float32,
        ).reshape(-1)
        edge_vec = edge_vec / 255.0

    return np.concatenate([color_hist, tiny, cfg.edge_weight * edge_vec]).astype(np.float32)


def _l2_normalize_rows(x: np.ndarray, eps: float) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _compute_penalties(
    similarity: np.ndarray, revenues: np.ndarray, cfg: SimilarityLayerConfig
) -> np.ndarray:
    n = similarity.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    rev = revenues.astype(np.float32)
    rev_min = float(np.min(rev))
    rev_max = float(np.max(rev))
    rev_range = max(rev_max - rev_min, cfg.epsilon)
    rev_norm = (rev - rev_min) / rev_range

    penalties = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        sim_row = similarity[i].copy()
        sim_row[i] = 0.0

        better_mask = rev > rev[i]
        if not np.any(better_mask):
            penalties[i] = 0.0
            continue

        sim_signal = np.clip(
            (sim_row - cfg.similarity_floor) / max(1.0 - cfg.similarity_floor, cfg.epsilon),
            0.0,
            1.0,
        )
        sim_signal = np.power(sim_signal, cfg.similarity_power)

        rev_gap = np.clip(rev_norm - rev_norm[i], 0.0, 1.0)
        weights = sim_signal * np.power(rev_gap, cfg.revenue_weight) * better_mask

        penalties[i] = float(np.sum(weights))

    # Normalize into [0, 1] to make combination with other layers stable.
    max_pen = float(np.max(penalties))
    if max_pen > cfg.epsilon:
        penalties = penalties / max_pen
    penalties = np.clip(penalties, 0.0, 1.0)
    return penalties


def compute_similarity_layer(
    records: Sequence[AdRecord],
    *,
    config: Optional[SimilarityLayerConfig] = None,
) -> SimilarityLayerResult:
    cfg = config or SimilarityLayerConfig()
    ad_ids = [record.ad_id for record in records]
    revenues = np.asarray([record.revenue for record in records], dtype=np.float32)

    feature_rows: List[np.ndarray] = []
    for record in records:
        img = _read_image_from_path(record.image_path, size=cfg.image_size)
        feature_rows.append(_extract_features(img, cfg))

    feature_matrix = np.stack(feature_rows, axis=0) if feature_rows else np.zeros((0, 1))
    feature_matrix = _l2_normalize_rows(feature_matrix, eps=cfg.epsilon)
    similarity_matrix = np.clip(feature_matrix @ feature_matrix.T, -1.0, 1.0).astype(np.float32)

    penalties = _compute_penalties(similarity_matrix, revenues, cfg)
    uniqueness = np.clip(1.0 - penalties, 0.0, 1.0).astype(np.float32)

    penalty_by_ad = {ad_id: float(penalties[i]) for i, ad_id in enumerate(ad_ids)}
    uniqueness_by_ad = {ad_id: float(uniqueness[i]) for i, ad_id in enumerate(ad_ids)}

    return SimilarityLayerResult(
        ad_ids=ad_ids,
        penalties=penalties,
        uniqueness=uniqueness,
        similarity_matrix=similarity_matrix,
        penalty_by_ad=penalty_by_ad,
        uniqueness_by_ad=uniqueness_by_ad,
    )


def run_similarity_layer(
    records: Sequence[AdRecord],
    *,
    config: Optional[SimilarityLayerConfig] = None,
) -> Dict[str, float]:
    """
    Backend-friendly one-call API.
    Input: ad_id + image_path + revenue
    Output: dict[ad_id] -> penalty score
    """
    return compute_similarity_layer(records, config=config).penalty_by_ad

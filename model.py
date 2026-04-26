from extract_tmp_data import extract_to_csv
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Below this threshold, HDBSCAN cannot estimate density reliably → use hierarchical
SMALL_SAMPLE_THRESHOLD = 20


def _auto_cut_dendrogram(Z, n_samples):
    """
    Automatically choose number of clusters by finding the largest
    gap between successive merge distances in the linkage matrix.
    Falls back to sqrt(n) heuristic if no clear gap found.
    """
    merge_distances = Z[:, 2]
    gaps = np.diff(merge_distances)

    # Look for the biggest jump in the last (n-1) merges — that's where clusters fuse
    if len(gaps) >= 2:
        # Ignore trivially small distances at the start
        significant = gaps[gaps > np.median(gaps) * 0.1]
        if len(significant) > 0:
            cut_idx = np.argmax(gaps) + 1          # +1 because gaps is diff of distances
            n_clusters = n_samples - cut_idx        # clusters remaining after that merge
            n_clusters = max(2, min(n_clusters, n_samples - 1))
            return n_clusters

    # Fallback: sqrt heuristic
    return max(2, int(np.sqrt(n_samples)))


def _cluster_small(pca_array, n_samples):
    """
    Agglomerative (Ward) clustering for small datasets.
    Automatically selects k via largest dendrogram gap.
    Returns (labels, n_clusters, linkage_matrix).
    """
    Z = linkage(pca_array, method="ward")
    n_clusters = _auto_cut_dendrogram(Z, n_samples)
    labels = fcluster(Z, n_clusters, criterion="maxclust") - 1  # 0-indexed
    print(f"Agglomerative (Ward): auto-selected k={n_clusters} via dendrogram gap analysis")
    return labels, n_clusters, Z


def _cluster_large(pca_array, n_samples):
    """
    HDBSCAN for larger datasets. Noise points stay as -1.
    Returns (labels, n_clusters, None).
    """
    min_cluster_size = max(2, int(np.sqrt(n_samples) * 0.5))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(pca_array)
    n_clusters = len(np.unique(labels[labels >= 0]))
    n_noise = int((labels == -1).sum())
    print(f"HDBSCAN: {n_clusters} cluster(s), {n_noise} noise point(s) "
          f"(min_cluster_size={min_cluster_size})")
    return labels, n_clusters, None


def clusterization(app_name="VaultRun Pay", country=None, os_name=None, language=None, show_plots=None):
    """
    Runs: feature extraction → PCA (auto-components) → clustering → plots.

    Strategy:
      n < 20  →  Agglomerative / Ward  (no density assumptions, works with any n)
      n ≥ 20  →  HDBSCAN              (density-based, no k needed)

    Returns the PCA dataframe with cluster labels.
    """
    if show_plots is None:
        show_plots = int(os.environ.get("SHOW_PLOTS", "1"))

    # ── 1. Features ──────────────────────────────────────────────────────────
    data = extract_to_csv(app_name, country=country, os_name=os_name, language=language)
    numeric = data.select_dtypes(include=["float64", "int64"])
    n_samples = len(numeric)
    print(f"Dataset: {n_samples} samples, {numeric.shape[1]} numeric features")

    # ── 2. Standardize ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric)

    # ── 3. PCA – retain ≥ 95% variance, min 3 components for 3D plot ─────────
    VARIANCE_THRESHOLD = 0.95
    max_components = min(n_samples - 1, scaled_data.shape[1])  # n-1 for small n
    max_components = max(max_components, 1)

    pca_full = PCA(n_components=max_components)
    pca_full.fit(scaled_data)
    cumulative_variance = pca_full.explained_variance_ratio_.cumsum()

    n_components = int(np.searchsorted(cumulative_variance, VARIANCE_THRESHOLD) + 1)
    n_components = max(n_components, min(3, max_components))   # want 3 for plot, but cap at max
    n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    print(f"PCA: {n_components} component(s) → {cumulative_variance[-1]:.1%} variance retained")

    col_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_data, columns=col_names)

    # ── 4. Clustering – strategy chosen by sample size ────────────────────────
    cluster_input = pca_df[col_names].to_numpy()

    if n_samples < SMALL_SAMPLE_THRESHOLD:
        print(f"n={n_samples} < {SMALL_SAMPLE_THRESHOLD} → using Agglomerative / Ward")
        cluster_labels, n_clusters, linkage_matrix = _cluster_small(cluster_input, n_samples)
    else:
        print(f"n={n_samples} ≥ {SMALL_SAMPLE_THRESHOLD} → using HDBSCAN")
        cluster_labels, n_clusters, linkage_matrix = _cluster_large(cluster_input, n_samples)

    pca_df["cluster"] = cluster_labels

    # ── 5. 3-D scatter ────────────────────────────────────────────────────────
    pc1 = pca_df["PC1"]
    pc2 = pca_df["PC2"] if "PC2" in pca_df else pd.Series(np.zeros(n_samples))
    pc3 = pca_df["PC3"] if "PC3" in pca_df else pd.Series(np.zeros(n_samples))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    mask_noise = cluster_labels == -1
    mask_clustered = ~mask_noise

    if mask_clustered.any():
        sc = ax.scatter(
            pc1[mask_clustered], pc2[mask_clustered], pc3[mask_clustered],
            c=cluster_labels[mask_clustered], cmap="tab20",
            alpha=0.9, edgecolors="k", linewidths=0.3, label="Clustered",
        )
        plt.colorbar(sc, ax=ax, pad=0.1, label="Cluster ID")

    if mask_noise.any():
        ax.scatter(
            pc1[mask_noise], pc2[mask_noise], pc3[mask_noise],
            c="lightgray", marker="x", alpha=0.6,
            label=f"Noise ({mask_noise.sum()})",
        )

    method = "Ward" if n_samples < SMALL_SAMPLE_THRESHOLD else "HDBSCAN"
    ax.set_title(f"{method} — {n_clusters} cluster(s) | n={n_samples} | {n_components}D PCA")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if show_plots:
        plt.show(block=True)
    else:
        plt.close()

    # ── 6. Dendrogram (small n only) ──────────────────────────────────────────
    if linkage_matrix is not None:
        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix, labels=[str(i) for i in range(n_samples)],
                   color_threshold=linkage_matrix[-(n_clusters - 1), 2])
        plt.title(f"Dendrogram — Ward linkage (k={n_clusters} auto-selected)")
        plt.xlabel("Sample index")
        plt.ylabel("Merge distance")
        plt.tight_layout()
        if show_plots:
            plt.show(block=True)
        else:
            plt.close()

    # ── 7. Cumulative explained variance ─────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="--")
    plt.axhline(VARIANCE_THRESHOLD, color="red", linestyle=":", label=f"{VARIANCE_THRESHOLD:.0%} threshold")
    plt.axvline(n_components, color="orange", linestyle=":", label=f"Chosen: {n_components} PCs")
    plt.title("Cumulative explained variance")
    plt.xlabel("Principal components"); plt.ylabel("Cumulative variance")
    plt.legend(); plt.grid(True); plt.tight_layout()
    if show_plots:
        plt.show(block=True)
    else:
        plt.close()

    print("Cumulative explained variance:", cumulative_variance)
    return pca_df
    


if __name__ == "__main__":
    # clusterization(app_name="StormByte Saga", country='US', os_name='iOS', language='en')
    clusterization(app_name="GlitchWave Legends", country='UK', os_name='Android', language='en')
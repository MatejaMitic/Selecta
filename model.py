from extract_tmp_data import extract_to_csv
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def clusterization(app_name="VaultRun Pay", country=None, os_name=None, language=None, show_plots=None):
    """
    Runs: feature extraction -> PCA (3D) -> HDBSCAN clustering -> plots/CSVs.

    Returns the PCA dataframe with cluster labels.
    """
    # By default, save plots to files (avoid hanging on plt.show() in batch runs).
    # Set SHOW_PLOTS=1 in env if you want interactive windows.
    if show_plots is None:
        show_plots = os.environ.get("SHOW_PLOTS", "0") == "1"

    # Build features dataset in-memory
    data = extract_to_csv(app_name, country=country, os_name=os_name, language=language)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=["float64", "int64"]))

    # Apply PCA
    n_components = min(10, scaled_data.shape[0], scaled_data.shape[1])
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    # Create a DataFrame with the PCA results
    if pca_data.shape[1] < 3:
        raise ValueError("Need at least 3 PCA components to run 3D clustering/plotting.")

    pca_df = pd.DataFrame(pca_data[:, :3], columns=["PC1", "PC2", "PC3"])

    # Nonparametric clustering (no need to pre-set number of clusters)
    #
    # HDBSCAN returns labels in {-1, 0, 1, ...} where -1 means "noise".
    # Since you mentioned possible 1-instance "classes", we convert noise points
    # into singleton clusters (each noise point gets its own unique label).
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method = 'leaf',
        allow_single_cluster = True
    )
    cluster_labels = clusterer.fit_predict(pca_df[["PC1", "PC2", "PC3"]].to_numpy())

    noise_mask = cluster_labels == -1
    if noise_mask.any():
        next_label = (cluster_labels.max() + 1) if (cluster_labels.max() >= 0) else 0
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == -1:
                cluster_labels[i] = next_label
                next_label += 1

    pca_df["cluster"] = cluster_labels
    n_clusters = len(pd.unique(cluster_labels))
    print(f"HDBSCAN clusters found: {n_clusters} (after singleton-noise relabel)")

    # Plot 3D PCA with cluster color coding
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        pca_df["PC3"],
        c=pca_df["cluster"],
        cmap="tab20",
        alpha=0.9,
        edgecolors="k",
        linewidths=0.2,
    )
    ax.set_title(f"HDBSCAN Clusters on 3D PCA (k={n_clusters})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    if show_plots:
        plt.show(block=True)
    else:
        plt.close()

    # Cumulative explained variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    if show_plots:
        plt.show(block=True)
    else:
        plt.close()

    # Better way to estimate information loss:
    # Use the cumulative explained variance to decide the number of components that retain a desired level of variance (e.g., 95%).
    print("Cumulative explained variance:", cumulative_variance)

    return pca_df


if __name__ == "__main__":
    clusterization(app_name="VaultRun Pay", country=None, os_name=None, language=None)

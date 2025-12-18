import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from utils.preprocess import prepare_data

def run_clustering_comparison():
    print("Loading and preparing data...")
    full_df = prepare_data()
    full_df=full_df.dropna()

    # 1. Split Features and Labels
    # We use labels ONLY for validation (to see if the cluster matches the action)
    X = full_df.drop(columns=['action']).astype(float)
    y_true = full_df['action']
    
    # 2. Scaling (Critical for all distance-based algorithms)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. PCA (Dimensionality Reduction)
    # We reduce to 2 dimensions for simple visualization and robust clustering
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Data projected to 2 Principal Components for clustering.")

    # --- MODEL 1: K-MEANS ---
    print("\n" + "="*40)
    print("Running K-MEANS (Hard Clustering)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_pca)
    evaluate_clusters("K-Means", y_true, km_labels, X_pca)

    # --- MODEL 2: DBSCAN ---
    print("\n" + "="*40)
    print("Running DBSCAN (Density Based)...")
    # Note: DBSCAN does NOT take n_clusters. It finds them automatically.
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    dbscan = DBSCAN(eps=0.5, min_samples=5) 
    db_labels = dbscan.fit_predict(X_pca)
    evaluate_clusters("DBSCAN", y_true, db_labels, X_pca)

    # --- MODEL 3: GAUSSIAN MIXTURE (GMM) ---
    print("\n" + "="*40)
    print("Running GMM (Probabilistic/Soft Clustering)...")
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm_labels = gmm.fit_predict(X_pca)
    evaluate_clusters("GMM", y_true, gmm_labels, X_pca)

    # --- PLOTTING COMPARISON ---
    plot_comparison(X_pca, km_labels, db_labels, gmm_labels)

def evaluate_clusters(model_name, y_true, labels, X_data):
    """
    Prints Silhouette score and a Cross-Tabulation of Cluster vs Real Action.
    """
    # Check if the model found any clusters (DBSCAN might mark everything as noise: -1)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"[{model_name}] Warning: Found only 1 cluster/noise. Cannot compute Silhouette.")
        return

    # 1. Silhouette Score (Higher is better, max 1.0)
    sil = silhouette_score(X_data, labels)
    print(f"[{model_name}] Silhouette Score: {sil:.4f}")
    
    # 2. Confusion Matrix (Map Clusters to Real Actions)
    # This helps you see: "Did Cluster 0 capture all the Serves?"
    print(f"[{model_name}] Cluster vs. True Label mapping:")
    df_res = pd.DataFrame({'Real Action': y_true, 'Cluster ID': labels})
    print(pd.crosstab(df_res['Real Action'], df_res['Cluster ID']))

def plot_comparison(X, km_labels, db_labels, gmm_labels):
    """
    Plots the same 2D data colored by 3 different algorithms.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot K-Means
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=km_labels, palette='viridis', ax=axes[0], legend='full')
    axes[0].set_title('K-Means (Target: 3 Clusters)')
    
    # Plot DBSCAN
    # Note: Label -1 is "Noise" (Outliers)
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=db_labels, palette='deep', ax=axes[1], legend='full')
    axes[1].set_title('DBSCAN (Auto-detected)')
    
    # Plot GMM
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=gmm_labels, palette='viridis', ax=axes[2], legend='full')
    axes[2].set_title('Gaussian Mixture (GMM)')
    
    plt.tight_layout()
    plt.savefig('./models/clustering_comparison.png')
    print("\nSaved comparison plot to './models/clustering_comparison.png'")

if __name__ == "__main__":
    run_clustering_comparison()
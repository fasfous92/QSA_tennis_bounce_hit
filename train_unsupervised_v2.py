import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Replaced Agglomerative with Birch
from sklearn.cluster import Birch, SpectralClustering
from sklearn.metrics import silhouette_score
from utils.preprocess import prepare_data

def run_large_scale_clustering():
    full_df = prepare_data()
    full_df.dropna(inplace=True)
    
    X = full_df.drop(columns=['action'])
    y_true = full_df['action']
    
    # 1. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Data shape: {X_pca.shape}. Switching to memory-efficient algorithms.")

    # --- MODEL 1: BIRCH (Memory-Efficient Hierarchical) ---
    print("\n" + "="*40)
    print("Running BIRCH (Efficient Hierarchical)...")
    
    # threshold: The radius of the subcluster. Lower = more fine-grained splitting.
    # branching_factor: Max number of subclusters in each node.
    # n_clusters=3: The final aggregation step will produce 3 clusters.
    brc = Birch(n_clusters=3, threshold=0.5, branching_factor=50)
    brc_labels = brc.fit_predict(X_pca)
    
    evaluate_clusters("BIRCH", y_true, brc_labels, X_pca)

    # --- MODEL 2: Spectral (ONLY ON SUBSET) ---
    # Spectral clustering ALSO fails on large data (O(N^2) memory).
    # We must train it on a subset to avoid crashing, then we can extrapolate or just analyze the subset.
    print("\n" + "="*40)
    print("Running Spectral Clustering (On Subset)...")
    
    # Take a random sample of 10,000 points max to prevent memory crash
    if len(X_pca) > 10000:
        indices = np.random.choice(len(X_pca), 10000, replace=False)
        X_subset = X_pca[indices]
        y_subset = y_true.iloc[indices]
    else:
        X_subset = X_pca
        y_subset = y_true

    spec = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42, n_jobs=-1)
    spec_labels = spec.fit_predict(X_subset)
    
    evaluate_clusters("Spectral (Subset)", y_subset, spec_labels, X_subset)

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=brc_labels, palette='viridis', ax=axes[0], s=10)
    axes[0].set_title('BIRCH (Hierarchical for Large Data)')
    
    sns.scatterplot(x=X_subset[:,0], y=X_subset[:,1], hue=spec_labels, palette='coolwarm', ax=axes[1], s=10)
    axes[1].set_title('Spectral (Subset Only)')
    
    plt.tight_layout()
    plt.savefig('./models/large_scale_clustering.png')
    print("\nSaved plot to './models/large_scale_clustering.png'")

def evaluate_clusters(model_name, y_true, labels, X_data):
    # Silhouette score is slow on large data; computing on a sample is safer
    if len(X_data) > 20000:
        # Calculate score on a random sample of 10k points for speed
        idx = np.random.choice(len(X_data), 10000, replace=False)
        sil = silhouette_score(X_data[idx], labels[idx])
        print(f"[{model_name}] Silhouette Score (Sampled): {sil:.4f}")
    else:
        sil = silhouette_score(X_data, labels)
        print(f"[{model_name}] Silhouette Score: {sil:.4f}")

    df_res = pd.DataFrame({'Real Action': y_true, 'Cluster ID': labels})
    print(pd.crosstab(df_res['Real Action'], df_res['Cluster ID']))

if __name__ == "__main__":
    run_large_scale_clustering()
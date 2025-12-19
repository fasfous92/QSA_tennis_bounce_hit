import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
import json
from preprocess import prepare_data # Assuming this exists

# Ensure models directory exists
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def evaluate_performance(df):
    print("--- Final Pipeline Evaluation ---")
    labels = ['air', 'hit', 'bounce']
    y_true = df['action']
    y_pred = df['final_prediction']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Final Confusion Matrix')
    plt.show()

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

def perform_fast_umap(df, target_col=None, plot=False):
    """
    Fits UMAP and returns the embedding, the fitted reducer, and the scaler.
    """
    print("Preparing data for Fast UMAP...")

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None

    # 1. Standardize (CRITICAL to save this scaler for inference)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Fast UMAP
    print("Running UMAP (Optimized for speed)...")
    reducer = umap.UMAP(
        n_neighbors=10,
        n_components=2,
        metric='euclidean',
        init='pca',
        n_epochs=300,
        low_memory=True,
        random_state=42,
        n_jobs=-1
    )
    
    embedding = reducer.fit_transform(X_scaled)
    
    if plot:
        umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        if target_col is not None:
            umap_df[target_col] = y.values

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='UMAP1', y='UMAP2', 
            hue=target_col if target_col else None,
            data=umap_df, palette='plasma', s=10, alpha=0.7
        )
        plt.title('Fast UMAP Projection')
        plt.show()
    
    return embedding, reducer, scaler

def cluster_with_gmm(umap_data, n_components=3, plot=False):
    """
    Fits GMM and returns labels and the fitted model.
    """
    print(f"Running GMM with {n_components} components...")
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(umap_data)
    
    if plot:
        plot_df = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2'])
        plot_df['cluster_id'] = labels
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=plot_df, x='UMAP1', y='UMAP2', 
            hue='cluster_id', palette='tab10', s=10, alpha=0.6
        )
        plt.title(f"Gaussian Mixture Models (k={n_components})")
        plt.show()
    
    return labels, gmm

def first_clustering(df):
    """
    Stage 1: Separate 'Air' from 'Potential Events'.
    Saves: Scaler, UMAP, GMM, and Top Clusters list.
    """
    print("\n--- STAGE 1: Event Detection ---")
    df['action_bin'] = df['action'] == 'air' 
    best_to_keep = ['action_bin', 'delta_vy', 'jerk_mag', 'cosine_sim']
    
    # Filter for UMAP training
    df_test = df[best_to_keep].copy()
    df_test.dropna(inplace=True)
    
    # Run Pipeline
    embedding, reducer, scaler = perform_fast_umap(df_test, 'action_bin')
    labels, gmm = cluster_with_gmm(embedding, 7)
    
    # Identify Event Clusters
    ct = pd.crosstab(df_test['action_bin'], labels)
    print("Stage 1 Crosstab:\n", ct)
    
    # Find clusters with most 'False' (meaning NOT air = Events)
    event_counts = ct.loc[False] 
    top_clusters = event_counts.nlargest(1).index.tolist() # Adjust logic if needed
    print(f"Selected Event Clusters: {top_clusters}")

    # Save Artifacts
    print("Saving Stage 1 models...")
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'stage1_scaler.joblib'))
    joblib.dump(reducer, os.path.join(MODEL_DIR, 'stage1_umap.joblib'))
    joblib.dump(gmm, os.path.join(MODEL_DIR, 'stage1_gmm.joblib'))
    with open(os.path.join(MODEL_DIR, 'stage1_top_clusters.json'), 'w') as f:
        json.dump(top_clusters, f)

    # Apply to dataframe
    df['first_cluster'] = labels
    df['is_event'] = df['first_cluster'].isin(top_clusters)
    
    return df

def second_clustering(df):
    """
    Stage 2: Classify Events into 'Hit', 'Bounce', or 'Air'.
    Saves: Scaler, UMAP, GMM, and Cluster Mapping.
    """
    print("\n--- STAGE 2: Event Classification ---")
    df_second = df.copy()
    
    if 'is_event' in df_second.columns:
        df_second = df_second[df_second['is_event'] == True]
    
    best_to_keep = ['action', 'ay', 'delta_vy', 'cosine_sim', 'speed_delta']
    df_train = df_second[best_to_keep].dropna()

    print(f"Running 2nd Stage Clustering on {len(df_train)} points...")

    # Run Pipeline
    embedding, reducer, scaler = perform_fast_umap(df_train, 'action')
    labels, gmm = cluster_with_gmm(embedding, 7)
    
    df_train['cluster_id'] = labels

    # Create Mapping based on Ground Truth (Training Only)
    ct = pd.crosstab(df_train['cluster_id'], df_train['action'])
    ct_pct = ct.div(ct.sum(axis=1), axis=0)
    print("\nStage 2 Cluster Composition:\n", ct_pct)

    cluster_mapping = {}
    for cluster_id, row in ct_pct.iterrows():
        pct_hit = row.get('hit', 0)
        pct_bounce = row.get('bounce', 0)
        
        # Heuristics
        if pct_hit > 0.08 and pct_hit > pct_bounce:
            cluster_mapping[cluster_id] = 'hit'
        elif pct_bounce > 0.08:
            cluster_mapping[cluster_id] = 'bounce'
        else:
            cluster_mapping[cluster_id] = 'air'
            
    # Fix keys for JSON (must be strings)
    # But for pandas map we need ints. We save as string keys in JSON, convert back on load.
    save_mapping = {str(k): v for k, v in cluster_mapping.items()}

    # Save Artifacts
    print("Saving Stage 2 models...")
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'stage2_scaler.joblib'))
    joblib.dump(reducer, os.path.join(MODEL_DIR, 'stage2_umap.joblib'))
    joblib.dump(gmm, os.path.join(MODEL_DIR, 'stage2_gmm.joblib'))
    with open(os.path.join(MODEL_DIR, 'stage2_mapping.json'), 'w') as f:
        json.dump(save_mapping, f)

    # Apply to dataframe for evaluation
    # We must map back to the original df indices
    df_train['predicted_action'] = df_train['cluster_id'].map(cluster_mapping)
    
    return df_train, cluster_mapping

if __name__ == "__main__":
    
    df = prepare_data()
    
    # 1. Run Stage 1 (and save models)
    df = first_clustering(df)
    
    # 2. Run Stage 2 (and save models)
    df_second_results, cluster_mapping = second_clustering(df)
    print(f"Stage 2 Mapping: {cluster_mapping}")

    # 3. Merge Results
    df['final_prediction'] = 'air'
    
    if not df_second_results.empty:
        # Align via index
        df.loc[df_second_results.index, 'final_prediction'] = df_second_results['predicted_action']

    # 4. Evaluate
    print("\n--- Final Prediction Counts ---")
    print(df['final_prediction'].value_counts())

    total_events = df[df['final_prediction'].isin(['hit', 'bounce'])].shape[0]
    print(f"Total Detected Events: {total_events}")
    
    evaluate_performance(df)

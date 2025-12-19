import pandas as pd
import numpy as np
import joblib
import json
import os
import umap
import os
import dotenv

dotenv.load_dotenv()
TEST_DATA=os.getenv('TEST_DATA')



# --- CONFIG ---
MODEL_DIR = 'models'

class UnsupervisedInference:
    def __init__(self, model_dir=MODEL_DIR):
        print(f"Loading models from {model_dir}...")
        
        # Load Stage 1 Artifacts
        self.s1_scaler = joblib.load(os.path.join(model_dir, 'stage1_scaler.joblib'))
        self.s1_umap = joblib.load(os.path.join(model_dir, 'stage1_umap.joblib'))
        self.s1_gmm = joblib.load(os.path.join(model_dir, 'stage1_gmm.joblib'))
        with open(os.path.join(model_dir, 'stage1_top_clusters.json'), 'r') as f:
            self.s1_top_clusters = json.load(f)
            
        # Load Stage 2 Artifacts
        self.s2_scaler = joblib.load(os.path.join(model_dir, 'stage2_scaler.joblib'))
        self.s2_umap = joblib.load(os.path.join(model_dir, 'stage2_umap.joblib'))
        self.s2_gmm = joblib.load(os.path.join(model_dir, 'stage2_gmm.joblib'))
        with open(os.path.join(model_dir, 'stage2_mapping.json'), 'r') as f:
            # JSON keys are strings, convert to int for mapping
            self.s2_mapping = {int(k): v for k, v in json.load(f).items()}

        print("Models loaded successfully.")

    def predict(self, df):
        """
        Runs the full 2-stage pipeline on a new DataFrame.
        Returns the DataFrame with a 'final_prediction' column.
        """
        df = df.copy()
        
        # --- STAGE 1: Event Detection ---
        # 1. Feature Prep (MUST match training exactly)
        s1_features = ['delta_vy', 'jerk_mag', 'cosine_sim']
        
     
        # Extract features
        X_s1 = df[s1_features].copy()
        # Handle NaNs (important for new data)
        X_s1 = X_s1.fillna(0) 
        
        # 2. Transform
        # Note: UMAP transform is slower than fit_transform.
        print("Running Stage 1 Inference...")
        X_s1_scaled = self.s1_scaler.transform(X_s1)
        X_s1_umap = self.s1_umap.transform(X_s1_scaled)
        
        # 3. Predict Cluster
        s1_labels = self.s1_gmm.predict(X_s1_umap)
        
        # 4. Filter Candidates
        df['stage1_cluster'] = s1_labels
        df['is_candidate'] = df['stage1_cluster'].isin(self.s1_top_clusters)
        
        candidates = df[df['is_candidate'] == True].copy()
        
        # --- STAGE 2: Classification ---
        if candidates.empty:
            print("No events detected in this video.")
            df['pred_action'] = 'air'
            return df
            
        print(f"Running Stage 2 on {len(candidates)} candidate frames...")
        
        # 1. Feature Prep
        s2_features = ['ay', 'delta_vy', 'cosine_sim', 'speed_delta']
        
      
        X_s2 = candidates[s2_features].copy()
        X_s2 = X_s2.fillna(0)
        
        # 2. Transform
        X_s2_scaled = self.s2_scaler.transform(X_s2)
        X_s2_umap = self.s2_umap.transform(X_s2_scaled)
        
        # 3. Predict Cluster
        s2_labels = self.s2_gmm.predict(X_s2_umap)
        candidates['stage2_cluster'] = s2_labels
        
        # 4. Map to Labels
        candidates['predicted_label'] = candidates['stage2_cluster'].map(self.s2_mapping)
        
        # --- MERGE RESULTS ---
        df['pred_action'] = 'air'
        
        # Align indexes to update only the candidates
        # We use .fillna('air') in case mapping had gaps (though it shouldn't)
        final_labels = candidates['predicted_label'].fillna('air')
        df.loc[candidates.index, 'pred_action'] = final_labels
        
        return df

# --- USAGE EXAMPLE ---
# if __name__ == "__main__":
#     # 1. Initialize Pipeline
#     pipeline = UnsupervisedInference()
    
#     # 2. Load New Data
#     # Ensure this df has 'delta_vy', 'jerk_mag', etc. calculated!
#     # You might need to run your 'calculate_physics_features(df)' helper first.
#     new_df,original_df = prepare_data_test(TEST_DATA) # Load your NEW video dataframe
    
#     # 3. Run Inference
#     results_df = pipeline.predict(new_df)
    
#     # 4. View Results
#     print("\nInference Complete. Predictions:")
#     print(results_df['pred_action'].value_counts())
    
   
import pandas as pd
import numpy as np
from preprocess import preprocessing_per_file
import os 
import dotenv
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler
import warnings



#load dotenv
dotenv.load_dotenv()
DATA_FOLDER=os.environ.get("DATA_FOLDER")

def create_flattened_sequences(df, window_size=30, step_size=10):
    """
    Transforms a DataFrame of frame-by-frame data into flattened sequence vectors.
    
    Args:
        df: DataFrame containing features (x, y, velocity, etc.) and 'action'.
        window_size: Number of frames to include in one sequence (e.g., 30 frames ~ 1 sec).
        step_size: How many frames to slide the window (overlap). 
                   Lower step_size = more data samples but highly correlated.
    
    Returns:
        X_flat: Numpy array of shape (N_samples, n_features * window_size)
        y_flat: Numpy array of labels corresponding to each sequence.
    """
    
    # 1. Separate Features and Targets
    # We drop 'action' because we don't want to flatten the label, just the physics.
    # We also drop metadata like 'video_id' if you have it.
    feature_cols = [c for c in df.columns if c != 'action']
    data = df[feature_cols].values
    labels = df['action'].values
    
    sequences = []
    seq_labels = []
    
    # 2. Create Sliding Windows
    # Note: If you have a 'video_id' column, you should loop through groups 
    # to avoid creating a window that crosses two different videos.
    # Here we assume continuous data or that boundaries don't matter much.
    
    num_samples = len(df)
    
    for start_idx in range(0, num_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Extract the window (Shape: window_size x n_features)
        window = data[start_idx:end_idx]
        
        # FLATTEN: Convert (30, 4) -> (120,)
        # This puts [x0, y0, ..., x29, y29] in one single row
        flat_vector = window.flatten()
        
        sequences.append(flat_vector)
        
        # Assign Label: We take the label of the LAST frame in the window
        # (Or you could take the mode/majority vote of the window)
        seq_labels.append(labels[end_idx - 1])
        
    return np.array(sequences), np.array(seq_labels)




def prepare_data(folder_path=DATA_FOLDER):


    frames_list = []

    print(f"Acessing data from {folder_path}")

    # 2. Loop through every file in the folder
    for filename in tqdm(os.listdir(folder_path)):

        filepath = os.path.join(folder_path, filename)
        df_temp = pd.read_json(filepath)
        df_temp=df_temp.T
        print(df_temp.shape)
        
        df_temp=preprocessing_per_file(df_temp,num=5)
        #after one video prepare data
        feature_cols = df_temp.columns.drop('action')
        scaler = StandardScaler()
        df_temp[feature_cols] = scaler.fit_transform(df_temp[feature_cols])
        X_seq, y_seq = create_flattened_sequences(df_temp, window_size=30, step_size=5)
        print(X_seq.shape)    
        
        break
        frames_list.append(df_temp)
        


    # # Concatenate all files into one distinct DataFrame
    # full_df = pd.concat(frames_list)
    # print(full_df.isna().sum())

    # print(f"Successfully created 'full_df' with shape: {full_df.shape}")
    return None
        

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # 1. Load your feature-engineered data
    full_df = prepare_data()
    
    # # Optional: Ensure features are scaled BEFORE flattening
    # # (Scaling after flattening is also okay, but scaling before is often cleaner)
    # feature_cols = full_df.columns.drop('action')
    # scaler = StandardScaler()
    # full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])

    # # 2. Run the Flattening
    # print(f"Original Shape: {full_df.shape} (Frames x Features)")
    
    # X_seq, y_seq = create_flattened_sequences(full_df, window_size=30, step_size=5)
    
    # print(f"Flattened Shape: {X_seq.shape} (Samples x [Features*30])")
    
    # # 3. Now run your Clustering on X_seq instead of X
    # from sklearn.cluster import Birch
    
    # print("Running Clustering on Sequences...")
    # model = Birch(n_clusters=10, threshold=0.5) # Threshold might need adjustment for higher dims
    # clusters = model.fit_predict(X_seq)
    
    # # Evaluate
    # df_res = pd.DataFrame({'Real Label': y_seq, 'Cluster ID': clusters})
    # print(pd.crosstab(df_res['Real Label'], df_res['Cluster ID']))
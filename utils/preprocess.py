import numpy as np
import glob
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
import os 
import dotenv
import warnings
warnings.filterwarnings('ignore')


#load dotenv
dotenv.load_dotenv()
DATA_FOLDER=os.environ.get("DATA_FOLDER")


def cosine_similarity_rows(df, window_size=4, threshold=1e-5):
    '''
    This function would compute the consine similarity between the velocity vectore before and after ou
    point. This could help us detect a sudden or a big angle change depicting hence a potential
    hit or bounce.      
    '''
    vx_before = df['x_smooth'] - df['x_smooth'].shift(window_size).bfill()
    vy_before = df['y_smooth'] - df['y_smooth'].shift(window_size).bfill()
   
    vx_after = df['x_smooth'].shift(-window_size).ffill() - df['x_smooth']
    vy_after = df['y_smooth'].shift(-window_size).ffill() - df['y_smooth']
    
    A = np.array([vx_before, vy_before], dtype=np.float32).T
    B = np.array([vx_after, vy_after], dtype=np.float32).T 
    
    #Dot product
    dot_products = np.sum(A * B, axis=1)
    
    #compute of norm
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    
   # to avoid diviing by zero
    valid_mask = (norm_A > threshold) & (norm_B > threshold)
    
    similarities = np.ones(len(df), dtype=np.float32)
    
    
    similarities[valid_mask] = dot_products[valid_mask] / (norm_A[valid_mask] * norm_B[valid_mask])
    
    # Use the same function to compute the velocity magnitude change
    delta_speed = norm_B - norm_A 
    
    #return np.clip(similarities, -1.0, 1.0), delta_speed
    return similarities, delta_speed


def get_window_slope(data, window_size):
    """
    Calculates the slope of a moving window efficiently.
    opposite slopes specialy in y might indicate a hit
    """
    windows = sliding_window_view(data, window_size)
    
    #compute the delta t to compute the slope
    x = np.arange(window_size)
    
    N = window_size
    sum_x = np.sum(x)
    sum_x_sq = np.sum(x**2)
    delta = N * sum_x_sq - sum_x**2
    
    sum_y = np.sum(windows, axis=1)
    sum_xy = np.sum(windows * x, axis=1)
    
    slopes = (N * sum_xy - sum_x * sum_y) / delta
    
    return slopes

def windowed_acceleration(df, window_size=5):
    
    vx_data = df['x_smooth'].values.astype(float)
    vy_data = df['y_smooth'].values.astype(float)
    
    # get the slope which is the velcity (windowed)
    vx_trend = get_window_slope(vx_data, window_size)
    vy_trend = get_window_slope(vy_data, window_size)
    
    # differentiate to get the acceleration
    ax = np.diff(vx_trend)
    ay = np.diff(vy_trend)
    
    # Pad the result to match original length
    # The sliding window reduces length by (window_size - 1)
    # The diff reduces length by 1
    # Total lost = window_size
    
    pad_start = window_size // 2
    pad_end = max(0, len(df) - len(ax) - pad_start)
    
    ax = np.pad(ax, (pad_start, pad_end), constant_values=0)
    ay = np.pad(ay, (pad_start, pad_end), constant_values=0)
    
    return ax, ay


def fit_piecewise_linear(df, breakpoints):
    """
    The main idea behind this is to reconstruct our smooth and parabolic 
    time series and make it fit in picewise linear function. This might help 
    us making hits and  bounces more relevent and hence increase their selctability.
    breakpoints are points where we observe certain peaks hence we use them as boundaries 
    for our piecewise linear reconstruction.
    """
   
    boundaries = sorted(list(set([0, len(df)] + list(breakpoints))))
    
    # Prepare an array to hold the linear reconstruction 
    linear_fit = np.full(len(df), np.nan)
    
    # List to store the equation of each line: (slope, intercept)
    line_equations = []

    # 2. Iterate through each segment
    for i in range(len(boundaries) - 1):
        start, end = int(boundaries[i]), int(boundaries[i+1])
        
        X_segment = np.arange(start, end).reshape(-1, 1) 
        y_segment = df['y'].iloc[start:end].values
        
        if len(y_segment) < 2:
            continue
            
        # fit Linear Model
        model = LinearRegression()
        model.fit(X_segment, y_segment)
        
        y_pred = model.predict(X_segment)
        
        linear_fit[start:end] = y_pred
        
        line_equations.append({
            'segment': (start, end),
            'slope': model.coef_[0],      
            'intercept': model.intercept_
        })
        
    return linear_fit, line_equations


def merge_close_points(points, threshold=10):
    """
    Groups detected peak points that are within 'threshold' frames of each other 
    and returns the average frame index for each group. The main observation was that 
    certain slopes were considered as peaks hence we tried to merge them.
    
    """
    if len(points) == 0:
        return np.array([], dtype=int)
    
    sorted_pts = np.sort(points)
    
    merged_points = []
    current_cluster = [sorted_pts[0]]
    
    for i in range(1, len(sorted_pts)):
        # if this point is close to the last one in the cluster, add it
        if sorted_pts[i] - current_cluster[-1] < threshold:
            current_cluster.append(sorted_pts[i])
        else:
            # Cluster finished: calculate mean and save
            mean_frame = np.mean(current_cluster)
            merged_points.append(mean_frame)
            # Start new cluster
            current_cluster = [sorted_pts[i]]
            
    # last cluster
    if current_cluster:
        merged_points.append(np.mean(current_cluster))
        
    return np.array(merged_points, dtype=int)


def calculate_peak_distances(total_frames, peak_indices):
    """
    For every point we calculate the distance to the closest peak.
    If the distance 0 this might be a peak, but also close points are harder to determine 
    but this also helps us detect air points (very far from peaks).
    """
    if len(peak_indices) == 0:
        return np.zeros(total_frames)
    
    all_frames = np.arange(total_frames)
    
    # We subtract every point from every peak, finding the absolute min
    distances = np.abs(all_frames[:, None] - peak_indices[None, :])
    
    # Get the minimum value along the rows (closest peak for each frame)
    min_distances = np.min(distances, axis=1)
    
    return min_distances


def Kinematics(df, x_col='x', y_col='y', fps=30, smooth_window=3):
    """
    Detects most of the kinematics variables:
    -velocity on x and y
    -acceleration on x and y
    -jerk on x and y
    -turning angle

    """
    # 1. PREPROCESSING: Smooth the data to reduce tracking noise
    # We use a rolling mean, but Savitzky-Golay is better for production
    #df = df.copy()
    # df[f'{x_col}_smooth'] = df[x_col].rolling(window=smooth_window, center=True).mean().fillna(df[x_col])
    # df[f'{y_col}_smooth'] = df[y_col].rolling(window=smooth_window, center=True).mean().fillna(df[y_col])

    # Time delta (dt)
    dt = 1 / fps

    # 2. KINEMATICS: Calculate Velocity and Acceleration
    # Velocity (First Derivative)
    df['vx'] = np.gradient(df[f'{x_col}_smooth'], dt)
    df['vy'] = np.gradient(df[f'{y_col}_smooth'], dt)
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)

    # Acceleration (Second Derivative)
    df['ax'] = np.gradient(df['vx'], dt)
    df['ay'] = np.gradient(df['vy'], dt)
    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2)

    # Jerk (Third Derivative - The "Shock" detector)
    # High jerk indicates a sudden physical event (collision)
    df['jerk_mag'] = np.gradient(df['acc_mag'], dt)

    # 3. CURVATURE: Calculate Turning Angle (0 to 180 degrees)
    # A generic "straight" flight has angle ~0. A hit has angle > 0.
    # We use dot product of normalized velocity vectors
    vx_norm = df['vx'] / (df['speed'] + 1e-6) # Avoid div/0
    vy_norm = df['vy'] / (df['speed'] + 1e-6)
    
    # Calculate dot product between vector t and t-1
    # We shift the normalized vectors by 1 frame to compare t vs t-1
    dot_product = (vx_norm * vx_norm.shift(1)) + (vy_norm * vy_norm.shift(1))
    # Clip to valid range for arccos [-1, 1]
    dot_product = dot_product.clip(-1.0, 1.0)
    df['turn_angle_deg'] = np.degrees(np.arccos(dot_product)).fillna(0)
  
    return df




def preprocess(df):
    #create a smooth version for better fixing of censors sensibility
    df['x_smooth'] = savgol_filter(df['x'], window_length=7, polyorder=2)
    df['y_smooth'] = savgol_filter(df['y'], window_length=7, polyorder=2)
    
    df=Kinematics(df, fps=30)
    #cosine similarity between velcoity before/after and also difference of speed 
    df['cosine_sim'],df['speed_delta'] = cosine_similarity_rows(df, 4)

    # #get the acceleration 
    # ax, ay = windowed_acceleration(df, window_size=10)

    # # Total magnitude of windowed acceleration
    # acc_magnitude = np.sqrt(ax**2 + ay**2)

    # df['acc_x'] = ax
    # df['acc_y'] = ay
    # df['acc_mag'] = acc_magnitude

    
    #detect pick points using scipy module
    y_data = df['y_smooth'].values
    peaks, _ = find_peaks(y_data, prominence=20, distance=5)
    valleys, _ = find_peaks(-y_data, prominence=20, distance=5)
    raw_points = np.concatenate([peaks, valleys])

    # Merge Close Peak points to avoid miss interpretations
    final_events = merge_close_points(raw_points, threshold=15)

    # --- CRITICAL FIX: MAP TO DATAFRAME INDEX ---
    # We use these 0-based positions to grab the REAL frame numbers from df.index
    actual_event_frames = df.index[final_events]
    old_event_frames=df.index[raw_points]
    # 3. Calculate Distances
    # We pass the 0-based 'final_events' because 'calculate_peak_distances' 
    # generates a 0-based range internally.
    dist_array = calculate_peak_distances(len(df), final_events)
    df['dist_to_event'] = dist_array  # Assigning to DF automatically aligns it to DF index

    to_drop=['x', 'y','x_smooth', 'y_smooth']
    df=df.drop(columns=to_drop)
    
    return   df


def preprocessing_per_file(df,num=3):
    df=df[df['visible']==True].copy()
    preprocess_df=preprocess(df)
  
    #df.drop(columns=['visible'])
    
    eps = 1e-15
    for i in range(1, num):
        df['x_lag_{}'.format(i)] = df['x'].shift(i)
        df['x_lag_inv_{}'.format(i)] = df['x'].shift(-i)
        df['y_lag_{}'.format(i)] = df['y'].shift(i)
        df['y_lag_inv_{}'.format(i)] = df['y'].shift(-i) 
        df['x_diff_{}'.format(i)] = abs(df['x_lag_{}'.format(i)] - df['x'])
        df['y_diff_{}'.format(i)] = df['y_lag_{}'.format(i)] - df['y']
        df['x_diff_inv_{}'.format(i)] = abs(df['x_lag_inv_{}'.format(i)] - df['x'])
        df['y_diff_inv_{}'.format(i)] = df['y_lag_inv_{}'.format(i)] - df['y']
        df['x_div_{}'.format(i)] = abs(df['x_diff_{}'.format(i)]/(df['x_diff_inv_{}'.format(i)] + eps))
        df['y_div_{}'.format(i)] = df['y_diff_{}'.format(i)]/(df['y_diff_inv_{}'.format(i)] + eps)
    
    
    # for i in range(1, num):
    #     df = df[df['x_lag_{}'.format(i)].notna()]
    #     df = df[df['x_lag_inv_{}'.format(i)].notna()]
        
        
    colnames_x = ['x_diff_{}'.format(i) for i in range(1, num)] + \
                    ['x_diff_inv_{}'.format(i) for i in range(1, num)] + \
                    ['x_div_{}'.format(i) for i in range(1, num)]
    colnames_y = ['y_diff_{}'.format(i) for i in range(1, num)] + \
                    ['y_diff_inv_{}'.format(i) for i in range(1, num)] + \
                    ['y_div_{}'.format(i) for i in range(1, num)]
    colnames = colnames_x + colnames_y

    features = df[colnames]

    
    
        
    return preprocess_df.join(features)



def prepare_data(folder_path=DATA_FOLDER):


    frames_list = []

    print(f"Acessing data from {folder_path}")

    # 2. Loop through every file in the folder
    for filename in tqdm(os.listdir(folder_path)):

        filepath = os.path.join(folder_path, filename)
        df_temp = pd.read_json(filepath)
        df_temp=df_temp.T
        
        df_temp=preprocessing_per_file(df_temp,num=5)
        
        frames_list.append(df_temp)
        


    # Concatenate all files into one distinct DataFrame
    full_df = pd.concat(frames_list)
    print(full_df.isna().sum())

    print(f"Successfully created 'full_df' with shape: {full_df.shape}")
    return full_df

def prepare_data_test(folder_path=DATA_FOLDER):  
    # here we assume that all our test data would have a visible "True"
    # meaning x and y values are there not nan
    ASSUMPTION=False #can be changes if decides else our model would just get avg. for the nan values


    frames_list = []
    frames_original=[]

    print(f"Acessing data from {DATA_FOLDER}")

    # 2. Loop through every file in the folder
    for filename in tqdm(os.listdir(DATA_FOLDER)):

        filepath = os.path.join(DATA_FOLDER, filename)
        df_temp = pd.read_json(filepath)
        df_temp=df_temp.T
        if not ASSUMPTION:
            df_temp['x'] = df_temp['x'].ffill().bfill()
            df_temp['y'] = df_temp['y'].ffill().bfill() #better then avg
            df_temp['visible'] = True
        
        
        frames_original.append(df_temp.copy())
        
        df_temp=preprocessing_per_file(df_temp,num=5)
        
        frames_list.append(df_temp)


    # Concatenate all files into one distinct DataFrame
    print(len(frames_list))
    print(len(frames_original))
    full_df = pd.concat(frames_list)
    original_df=pd.concat(frames_original)

    print(f"Successfully created 'full_df' with shape: {full_df.shape}")
    print(f"original df  shape: {original_df.shape}")

    return full_df,original_df
        
if __name__ == "__main__":
    
    full_df,original_df=prepare_data_test()

    #full_df.to_csv('full_data_preprocessed.csv')
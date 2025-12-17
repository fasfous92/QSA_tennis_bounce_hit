import json
import numpy as np
import matplotlib.pyplot as plt

def plot_ball_timeseries(point_ID,test_y,test_x):
    
    with open(f'data/ball_data_{point_ID}.json', 'r') as file:
        ball_data = json.load(file)
    ball_data = {int(frame): data for frame, data in ball_data.items()}
    
    bounces_data = {}
    hits_data = {}

    # ---- Iterate through all frames ----
    for frame_str, info in ball_data.items():
        action = info.get("action", None)
        frame = int(frame_str)

        if action == "bounce":
            bounces_data[frame] = info

        elif action == "hit":
            hits_data[frame] = info


    dico_ball = ball_data
    hits = hits_data
    bounces = bounces_data
    # 1) Sort frames and build arrays
    frames = sorted(dico_ball.keys())

    frames_visible = []
    x_vals = []
    y_vals = []

    for f in frames:
        d = dico_ball[f]
        # keep only visible points with valid x,y
        if d.get("visible") and d.get("x") is not None and d.get("y") is not None:
            frames_visible.append(f)
            x_vals.append(d["x"])
            y_vals.append(d["y"])

    frames_visible = np.array(frames_visible)
    x_vals = np.array(x_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)

    # 2) Create figure with 2 subplots: x vs frame, y vs frame
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_x, ax_y = axes

    # Plot x
    ax_x.plot(frames_visible, x_vals, marker='o', linestyle='-', label='x (pixels)')
    ax_x.plot(frames_visible, test_x, marker='x', linestyle='-', label='test_x')

    ax_x.set_ylabel("x (pixels)")
    ax_x.legend(loc="upper left")

    # Plot y
    ax_y.plot(frames_visible, y_vals, marker='o', linestyle='-', label='y (pixels)')
    ax_y.plot(frames_visible, test_y, marker='x', linestyle='-', label='test')
    ax_y.set_ylabel("y (pixels)")
    ax_y.set_xlabel("Frame")
    ax_y.legend(loc="upper left")

    # 3) Add vertical lines for hits (green) and bounces (red)
    hits = sorted(hits)
    bounces = sorted(bounces)

    for h in hits:
        ax_x.axvline(h, color='g', linestyle='--', alpha=0.7, label='Hit')
        ax_y.axvline(h, color='g', linestyle='--', alpha=0.7, label='Hit')

    for b in bounces:
        ax_x.axvline(b, color='r', linestyle='--', alpha=0.7,  label='Bounce')
        ax_y.axvline(b, color='r', linestyle='--', alpha=0.7,label='Bounce')

    # Optional: x-limits around data & events
    all_frames_for_limits = frames_visible.tolist() + hits + bounces
    if all_frames_for_limits:
        x_min = min(all_frames_for_limits) - 5
        x_max = max(all_frames_for_limits) + 5
        ax_x.set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.legend()
    plt.show()

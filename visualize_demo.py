#!/usr/bin/env python
"""
Quick demo script to visualize a single sample of MediaPipe data
This shows RAW data in (T, 75) format WITHOUT temporal preprocessing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# MediaPipe bone connections (0-indexed)
BONES = [
    (1, 0), (2, 0), (3, 1), (4, 2),  # Face
    (5, 0), (6, 0),  # Shoulders to nose
    (7, 5), (9, 7), (8, 6), (10, 8),  # Arms
    (11, 9), (13, 9), (12, 10), (14, 10),  # Hands
    (15, 5), (16, 6), (16, 15),  # Torso
    (17, 15), (19, 17), (18, 16), (20, 18),  # Legs
    (21, 19), (23, 19), (22, 20), (24, 20),  # Feet
]

ACTIONS = ['Arm_Swing', 'Body_pose', 'chest_expansion', 'Drumming', 'Frog_Pose',
           'Marcas_Forward', 'Marcas_Shaking', 'Sing_Clap', 'Squat_Pose', 
           'Tree_Pose', 'Twist_Pose']


def load_csv_raw(csv_file):
    """
    Load CSV file with min-max normalization (same as Enhanced-MMASD)
    Returns data in (T, 75) format
    """
    df = pd.read_csv(csv_file)
    
    # Drop label columns if present
    if 'Action_Label' in df.columns:
        df = df.drop(['Action_Label'], axis=1, errors='ignore')
    if 'ASD_Label' in df.columns:
        df = df.drop(['ASD_Label'], axis=1, errors='ignore')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
    
    # Min-max normalization: scale data to [0, 1] range
    df_min = df.min().min()
    df_max = df.max().max()
    normalized_df = (df - df_min) / (df_max - df_min)
    
    return normalized_df.values  # (T, 75)


def plot_skeleton(ax, data, frame_idx, title=""):
    """Plot skeleton for a single frame
    Args:
        data: numpy array of shape (T, 75)
        frame_idx: which frame to visualize
    """
    # Get single frame and reshape to (25, 3)
    frame = data[frame_idx]
    keypoints = frame.reshape(25, 3)
    x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
    
    # Plot joints
    ax.scatter(x, y, z, c='red', marker='o', s=50, alpha=0.8)
    
    # Plot bones
    for j1, j2 in BONES:
        ax.plot([x[j1], x[j2]], [y[j1], y[j2]], [z[j1], z[j2]], 
                'b-', linewidth=2, alpha=0.6)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    
    # Equal aspect
    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min()) / 2.0
    mid = [(x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2]
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)


def main():
    print("\n" + "="*60)
    print("Quick MediaPipe Data Visualization Demo")
    print("Raw data in (T, 75) format - NO temporal preprocessing")
    print("="*60)
    
    # Check if a file path is provided as command-line argument
    if len(sys.argv) > 1:
        sample_csv = sys.argv[1]
        if not os.path.exists(sample_csv):
            print(f"Error: File not found: {sample_csv}")
            return
        # Extract action folder from path
        path_parts = sample_csv.split(os.sep)
        action_folder = path_parts[-2] if len(path_parts) > 1 else "Unknown"
        sample_name = os.path.basename(sample_csv)
    else:
        # Find a sample CSV file
        data_path = './3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit'
        
        # Get all CSV files
        csv_files = []
        action_folders = sorted([d for d in os.listdir(data_path) 
                                if os.path.isdir(os.path.join(data_path, d))])
        
        for action_folder in action_folders:
            if action_folder not in ['Arm_Swing', 'Body_pose', 'chest_expansion', 'Drumming', 
                                     'Frog_Pose', 'Marcas_Forward', 'Marcas_Shaking', 
                                     'Sing_Clap', 'Squat_Pose', 'Tree_Pose', 'Twist_Pose']:
                continue
            
            action_path = os.path.join(data_path, action_folder)
            files = sorted(glob.glob(os.path.join(action_path, '*.csv')))
            for f in files:
                csv_files.append((f, action_folder))
        
        if not csv_files:
            print("Error: No CSV files found!")
            return
        
        # Use first file
        sample_csv, action_folder = csv_files[0]
        sample_name = os.path.basename(sample_csv)
    
    print(f"\nLoading sample: {sample_name}")
    print(f"Action: {action_folder}")
    
    # Load raw data
    data = load_csv_raw(sample_csv)
    
    print(f"\nSample details:")
    print(f"  Name: {sample_name}")
    print(f"  Action: {action_folder}")
    print(f"  Data shape: {data.shape} (T, 75)")
    print(f"    T={data.shape[0]} (temporal frames - ORIGINAL, no resizing)")
    print(f"    75 = 25 joints × 3 coordinates (x,y,z)")
    print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
    
    # Visualize 6 frames
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'{action_folder} - {sample_name}\nRaw Data Shape: {data.shape}', 
                 fontsize=14, fontweight='bold')
    
    frame_indices = np.linspace(0, data.shape[0]-1, 6, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        plot_skeleton(ax, data, frame_idx, f'Frame {frame_idx}/{data.shape[0]-1}')
    
    plt.tight_layout()
    
    # Save
    os.makedirs('./visualizations', exist_ok=True)
    save_path = './visualizations/demo_output.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60 + "\n")
    
    # Show plot
    plt.show()


if __name__ == '__main__':
    main()


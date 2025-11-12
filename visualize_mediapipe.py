#!/usr/bin/env python3
"""
Visualization script for MediaPipe dataloader
Shows skeleton stick figures with bones and keypoints
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feeders.feeder_mediapipe import Feeder
from feeders.bone_pairs import mediapipe_pairs


# MediaPipe joint names (0-indexed)
JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_heel', 'right_heel', 'left_foot', 'right_foot'
]

# Convert 1-indexed pairs to 0-indexed
BONE_CONNECTIONS = [(v1 - 1, v2 - 1) for v1, v2 in mediapipe_pairs]


def visualize_skeleton_3d(data, frame_idx=0, title="MediaPipe 3D Skeleton", show_labels=True):
    """
    Visualize a single frame of skeleton in 3D
    
    Args:
        data: numpy array of shape (C, T, V, M) where C=3, V=25
        frame_idx: which frame to visualize
        title: plot title
        show_labels: whether to show joint labels
    """
    C, T, V, M = data.shape
    
    # Get the frame (C, V, M)
    frame_data = data[:, frame_idx, :, 0]  # Shape: (3, 25)
    
    # Extract x, y, z coordinates
    xs = frame_data[0, :]  # x coordinates
    ys = frame_data[1, :]  # y coordinates  
    zs = frame_data[2, :]  # z coordinates
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot keypoints with different colors for different body parts
    # Face points (0-4)
    ax.scatter(xs[0:5], ys[0:5], zs[0:5], c='yellow', marker='o', s=100, 
               label='Face', alpha=0.9, edgecolors='black', linewidths=1)
    # Upper body (5-14)
    ax.scatter(xs[5:15], ys[5:15], zs[5:15], c='red', marker='o', s=100, 
               label='Upper Body', alpha=0.9, edgecolors='black', linewidths=1)
    # Lower body (15-24)
    ax.scatter(xs[15:25], ys[15:25], zs[15:25], c='blue', marker='o', s=100, 
               label='Lower Body', alpha=0.9, edgecolors='black', linewidths=1)
    
    # Add keypoint labels
    if show_labels:
        for i in range(V):
            ax.text(xs[i], ys[i], zs[i], f' {i}', fontsize=8, fontweight='bold')
    
    # Draw bone connections
    for v1, v2 in BONE_CONNECTIONS:
        if v1 < V and v2 < V:
            ax.plot3D([xs[v1], xs[v2]], 
                     [ys[v1], ys[v2]], 
                     [zs[v1], zs[v2]], 
                     'darkgreen', linewidth=2.5, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\nFrame {frame_idx}/{T-1}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=15, azim=45)
    
    plt.tight_layout()
    return fig


def visualize_skeleton_2d_multiview(data, frame_idx=0, title="MediaPipe 2D Skeleton (Multi-view)"):
    """
    Visualize a single frame in 2D with three different projections (XY, XZ, YZ)
    
    Args:
        data: numpy array of shape (C, T, V, M) where C=3, V=25
        frame_idx: which frame to visualize
        title: plot title
    """
    C, T, V, M = data.shape
    
    # Get the frame (C, V, M)
    frame_data = data[:, frame_idx, :, 0]  # Shape: (3, 25)
    
    # Extract x, y, z coordinates
    xs = frame_data[0, :]
    ys = frame_data[1, :]
    zs = frame_data[2, :]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    projections = [
        (xs, ys, 'X', 'Y', 'Front View (XY)'),
        (xs, zs, 'X', 'Z', 'Top View (XZ)'),
        (zs, ys, 'Z', 'Y', 'Side View (ZY)')
    ]
    
    for ax, (coord1, coord2, label1, label2, view_title) in zip(axes, projections):
        # Plot keypoints with different colors
        ax.scatter(coord1[0:5], coord2[0:5], c='yellow', marker='o', s=100, 
                  label='Face', alpha=0.9, edgecolors='black', linewidths=1, zorder=3)
        ax.scatter(coord1[5:15], coord2[5:15], c='red', marker='o', s=100, 
                  label='Upper Body', alpha=0.9, edgecolors='black', linewidths=1, zorder=3)
        ax.scatter(coord1[15:25], coord2[15:25], c='blue', marker='o', s=100, 
                  label='Lower Body', alpha=0.9, edgecolors='black', linewidths=1, zorder=3)
        
        # Draw bone connections
        for v1, v2 in BONE_CONNECTIONS:
            if v1 < V and v2 < V:
                ax.plot([coord1[v1], coord1[v2]], 
                       [coord2[v1], coord2[v2]], 
                       'darkgreen', linewidth=2, alpha=0.6, zorder=2)
        
        # Add keypoint labels
        for i in range(V):
            ax.text(coord1[i], coord2[i], f' {i}', fontsize=7, fontweight='bold')
        
        ax.set_xlabel(label1, fontsize=11, fontweight='bold')
        ax.set_ylabel(label2, fontsize=11, fontweight='bold')
        ax.set_title(view_title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        ax.set_aspect('equal', adjustable='box')
    
    fig.suptitle(f'{title} - Frame {frame_idx}/{T-1}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def visualize_multiple_frames_3d(data, frame_indices=None, title="MediaPipe Skeleton Sequence"):
    """
    Visualize multiple frames in a 2x2 grid in 3D
    
    Args:
        data: numpy array of shape (C, T, V, M)
        frame_indices: list of frame indices to visualize (default: evenly spaced 4 frames)
        title: plot title
    """
    C, T, V, M = data.shape
    
    if frame_indices is None:
        # Select 4 evenly spaced frames
        frame_indices = [0, T//3, 2*T//3, T-1]
    
    fig = plt.figure(figsize=(16, 14))
    
    for subplot_idx, frame_idx in enumerate(frame_indices[:4]):
        if frame_idx >= T:
            continue
        
        ax = fig.add_subplot(2, 2, subplot_idx + 1, projection='3d')
        
        # Get the frame
        frame_data = data[:, frame_idx, :, 0]
        xs = frame_data[0, :]
        ys = frame_data[1, :]
        zs = frame_data[2, :]
        
        # Plot keypoints
        ax.scatter(xs[0:5], ys[0:5], zs[0:5], c='yellow', marker='o', s=80, alpha=0.9)
        ax.scatter(xs[5:15], ys[5:15], zs[5:15], c='red', marker='o', s=80, alpha=0.9)
        ax.scatter(xs[15:25], ys[15:25], zs[15:25], c='blue', marker='o', s=80, alpha=0.9)
        
        # Draw bones
        for v1, v2 in BONE_CONNECTIONS:
            if v1 < V and v2 < V:
                ax.plot3D([xs[v1], xs[v2]], 
                         [ys[v1], ys[v2]], 
                         [zs[v1], zs[v2]], 
                         'darkgreen', linewidth=2, alpha=0.6)
        
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title(f'Frame {frame_idx}', fontsize=11, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.view_init(elev=15, azim=45)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


def print_joint_info():
    """Print information about MediaPipe joint structure"""
    print(f"\n{'='*70}")
    print("MediaPipe Pose Joint Structure (25 keypoints)")
    print(f"{'='*70}")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:2d}: {name:20s}", end='')
        if (i + 1) % 2 == 0:
            print()
    if len(JOINT_NAMES) % 2 == 1:
        print()
    print(f"{'='*70}\n")
    
    print(f"{'='*70}")
    print(f"Bone Connections ({len(BONE_CONNECTIONS)} bones)")
    print(f"{'='*70}")
    for i, (v1, v2) in enumerate(BONE_CONNECTIONS):
        print(f"  {i+1:2d}. {JOINT_NAMES[v1]:20s} <--> {JOINT_NAMES[v2]:20s}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize MediaPipe skeleton data')
    parser.add_argument('--data-path', type=str, 
                       default='data/mediapipe/mediapipe_data.npz',
                       help='Path to MediaPipe data file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                       help='Dataset split to visualize')
    parser.add_argument('--sample-idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--frame-idx', type=int, default=0,
                       help='Frame index to visualize')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for saving visualizations')
    parser.add_argument('--show-labels', action='store_true', default=True,
                       help='Show joint labels in 3D view')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save visualizations to file')
    parser.add_argument('--bone', action='store_true',
                       help='Visualize bone modality instead of joint')
    parser.add_argument('--normalization', action='store_true',
                       help='Apply normalization to data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print joint structure information
    print_joint_info()
    
    # Load data using feeder
    print(f"{'='*70}")
    print(f"Loading MediaPipe data from: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"{'='*70}\n")
    
    try:
        feeder = Feeder(
            data_path=args.data_path,
            split=args.split,
            window_size=64,  # Use standard window size for visualization
            p_interval=[0.95],  # Use test mode p_interval for stable visualization
            debug=False,
            bone=args.bone,
            normalization=args.normalization
        )
        
        print(f"Dataset loaded successfully!")
        print(f"  Total samples: {len(feeder)}")
        print(f"  Data shape: {feeder.data.shape} (N, C, T, V, M)")
        
        # Get a sample
        if args.sample_idx >= len(feeder):
            print(f"\n⚠ Warning: Sample index {args.sample_idx} out of range. Using sample 0.")
            args.sample_idx = 0
        
        data, label, idx = feeder[args.sample_idx]
        
        print(f"\nSample {args.sample_idx}:")
        print(f"  Shape: {data.shape}")
        print(f"  Label: {label}")
        print(f"  Data range: [{data.min():.4f}, {data.max():.4f}]")
        
        C, T, V, M = data.shape
        
        if args.frame_idx >= T:
            print(f"\n⚠ Warning: Frame index {args.frame_idx} out of range. Using frame 0.")
            args.frame_idx = 0
        
        # Visualization 1: 3D skeleton (single frame)
        print(f"\n{'='*70}")
        print("Creating 3D skeleton visualization (single frame)...")
        print(f"{'='*70}")
        
        modality_str = "Bone" if args.bone else "Joint"
        fig1 = visualize_skeleton_3d(
            data, 
            frame_idx=args.frame_idx,
            title=f"MediaPipe 3D Skeleton ({modality_str} Modality) - Sample {args.sample_idx}",
            show_labels=args.show_labels
        )
        
        if not args.no_save:
            output_file1 = os.path.join(args.output_dir, 
                                       f'mediapipe_{args.split}_sample{args.sample_idx}_frame{args.frame_idx}_3d.png')
            plt.figure(fig1.number)
            plt.savefig(output_file1, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_file1}")
        
        # Visualization 2: 2D multi-view (single frame)
        print(f"\n{'='*70}")
        print("Creating 2D multi-view visualization...")
        print(f"{'='*70}")
        
        fig2 = visualize_skeleton_2d_multiview(
            data,
            frame_idx=args.frame_idx,
            title=f"MediaPipe 2D Skeleton ({modality_str} Modality) - Sample {args.sample_idx}"
        )
        
        if not args.no_save:
            output_file2 = os.path.join(args.output_dir, 
                                       f'mediapipe_{args.split}_sample{args.sample_idx}_frame{args.frame_idx}_2d.png')
            plt.figure(fig2.number)
            plt.savefig(output_file2, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_file2}")
        
        # Visualization 3: Multiple frames in 3D
        print(f"\n{'='*70}")
        print("Creating multi-frame 3D visualization...")
        print(f"{'='*70}")
        
        fig3 = visualize_multiple_frames_3d(
            data,
            frame_indices=[0, T//3, 2*T//3, T-1],
            title=f"MediaPipe Skeleton Sequence ({modality_str} Modality) - Sample {args.sample_idx}"
        )
        
        if not args.no_save:
            output_file3 = os.path.join(args.output_dir, 
                                       f'mediapipe_{args.split}_sample{args.sample_idx}_sequence.png')
            plt.figure(fig3.number)
            plt.savefig(output_file3, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_file3}")
        
        print(f"\n{'='*70}")
        print("Visualization complete!")
        print(f"{'='*70}\n")
        
        # Show plots
        plt.show()
        
    except FileNotFoundError:
        print(f"\n❌ Error: Data file not found: {args.data_path}")
        print("\nPlease ensure the MediaPipe data has been preprocessed.")
        print("Run: python data/mediapipe/preprocess_mediapipe_data.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

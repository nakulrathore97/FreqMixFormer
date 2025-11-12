#!/usr/bin/env python
"""
Visualization script for MediaPipe skeleton data as it's passed to the model.

This script:
1. Loads data using the actual feeder (feeder_mmasd.py)
2. Shows the data structure after all preprocessing
3. Visualizes skeleton in 3D with bone connections
4. Shows temporal information (multiple frames)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import torch
from matplotlib.animation import FuncAnimation
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feeders.feeder_mmasd import Feeder
from graph.mediapipe import Graph


# MediaPipe bone pairs (0-indexed, matching graph/mediapipe.py)
MEDIAPIPE_BONES = [
    # Face
    (1, 0), (2, 0),  # eyes to nose
    (3, 1), (4, 2),  # ears to eyes
    # Upper body
    (5, 0), (6, 0),  # shoulders to nose
    (7, 5), (9, 7),  # left arm: shoulder -> elbow -> wrist
    (8, 6), (10, 8),  # right arm: shoulder -> elbow -> wrist
    (11, 9), (13, 9),  # left wrist to pinky and index
    (12, 10), (14, 10),  # right wrist to pinky and index
    # Torso
    (15, 5), (16, 6),  # hips from shoulders
    (16, 15),  # right hip from left hip
    # Legs
    (17, 15), (19, 17),  # left leg: hip -> knee -> ankle
    (18, 16), (20, 18),  # right leg: hip -> knee -> ankle
    # Feet
    (21, 19), (23, 19),  # left ankle to heel and foot
    (22, 20), (24, 20),  # right ankle to heel and foot
]

# Joint names for MediaPipe (25 joints)
JOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_pinky',     # 11
    'right_pinky',    # 12
    'left_index',     # 13
    'right_index',    # 14
    'left_hip',       # 15
    'right_hip',      # 16
    'left_knee',      # 17
    'right_knee',     # 18
    'left_ankle',     # 19
    'right_ankle',    # 20
    'left_heel',      # 21
    'right_heel',     # 22
    'left_foot',      # 23
    'right_foot',     # 24
]

# Action labels
ACTION_LABELS = [
    'Arm_Swing',
    'Body_pose',
    'chest_expansion',
    'Drumming',
    'Frog_Pose',
    'Marcas_Forward',
    'Marcas_Shaking',
    'Sing_Clap',
    'Squat_Pose',
    'Tree_Pose',
    'Twist_Pose'
]


def visualize_skeleton_3d(data, frame_idx, ax=None, title=None):
    """
    Visualize a single frame of skeleton data in 3D
    
    Args:
        data: numpy array of shape (C, T, V, M) where C=3, V=25 joints
        frame_idx: which frame to visualize
        ax: matplotlib 3D axis (if None, creates new figure)
        title: plot title
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates for this frame (C, V, M) -> (3, 25, 1)
    frame_data = data[:, frame_idx, :, 0]  # Shape: (3, 25)
    
    # Get x, y, z coordinates
    x = frame_data[0, :]  # x coordinates of all joints
    y = frame_data[1, :]  # y coordinates
    z = frame_data[2, :]  # z coordinates
    
    # Plot joints as scatter points
    ax.scatter(x, y, z, c='red', marker='o', s=50, alpha=0.8, label='Joints')
    
    # Plot bones (connections between joints)
    for bone_idx, (joint1, joint2) in enumerate(MEDIAPIPE_BONES):
        if joint1 < len(x) and joint2 < len(x):
            ax.plot([x[joint1], x[joint2]], 
                   [y[joint1], y[joint2]], 
                   [z[joint1], z[joint2]], 
                   'b-', linewidth=2, alpha=0.6)
    
    # Add joint labels for key points
    key_joints = [0, 5, 6, 9, 10, 15, 16, 19, 20]  # nose, shoulders, wrists, hips, ankles
    for i in key_joints:
        if i < len(x):
            ax.text(x[i], y[i], z[i], f'{i}', fontsize=8, color='darkgreen')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    return ax


def visualize_multiple_frames(data, label, sample_name, num_frames=6, save_path=None):
    """
    Visualize multiple frames of skeleton data
    
    Args:
        data: numpy array of shape (C, T, V, M)
        label: action label
        sample_name: name of the sample
        num_frames: number of frames to display
        save_path: path to save the figure
    """
    C, T, V, M = data.shape
    
    # Select evenly spaced frames
    frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
    
    # Create subplots
    cols = 3
    rows = (num_frames + cols - 1) // cols
    fig = plt.figure(figsize=(15, 5 * rows))
    
    action_name = ACTION_LABELS[label] if label < len(ACTION_LABELS) else f"Action_{label}"
    fig.suptitle(f'Sample: {sample_name}\nAction: {action_name}\nData Shape: {data.shape}', 
                 fontsize=14, fontweight='bold')
    
    for idx, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        visualize_skeleton_3d(data, frame_idx, ax, 
                            title=f'Frame {frame_idx}/{T-1}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def visualize_data_structure(data, label, sample_name):
    """
    Visualize the structure and statistics of the data
    
    Args:
        data: numpy array of shape (C, T, V, M)
        label: action label
        sample_name: name of the sample
    """
    C, T, V, M = data.shape
    action_name = ACTION_LABELS[label] if label < len(ACTION_LABELS) else f"Action_{label}"
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Data Structure Analysis\nSample: {sample_name} | Action: {action_name}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Joint trajectories over time (X coordinate)
    ax = axes[0, 0]
    for v in range(min(V, 10)):  # Plot first 10 joints
        ax.plot(data[0, :, v, 0], label=f'J{v}', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('X Coordinate')
    ax.set_title('X Coordinate Trajectories (first 10 joints)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Data distribution per coordinate
    ax = axes[0, 1]
    coord_names = ['X', 'Y', 'Z']
    for c in range(C):
        coord_data = data[c, :, :, 0].flatten()
        ax.hist(coord_data, bins=50, alpha=0.5, label=coord_names[c])
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Coordinates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Mean movement per joint (variance over time)
    ax = axes[0, 2]
    movement = []
    for v in range(V):
        joint_variance = np.var(data[:, :, v, 0], axis=1).mean()
        movement.append(joint_variance)
    ax.bar(range(V), movement, color='steelblue', alpha=0.7)
    ax.set_xlabel('Joint Index')
    ax.set_ylabel('Variance (Movement)')
    ax.set_title('Movement per Joint')
    ax.grid(True, alpha=0.3)
    
    # 4. Heatmap of joint activity over time (magnitude)
    ax = axes[1, 0]
    magnitude = np.sqrt(np.sum(data[:, :, :, 0]**2, axis=0))  # Shape: (T, V)
    im = ax.imshow(magnitude.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Joint Index')
    ax.set_title('Joint Activity Heatmap (Magnitude)')
    plt.colorbar(im, ax=ax)
    
    # 5. Data statistics table
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Data Shape: (C={C}, T={T}, V={V}, M={M})
    
    C: Coordinates (X, Y, Z)
    T: Temporal frames
    V: Vertices (joints)
    M: Maximum number of people
    
    Statistics:
    - Min value: {data.min():.4f}
    - Max value: {data.max():.4f}
    - Mean: {data.mean():.4f}
    - Std: {data.std():.4f}
    
    Per Coordinate:
    - X: [{data[0].min():.3f}, {data[0].max():.3f}]
    - Y: [{data[1].min():.3f}, {data[1].max():.3f}]
    - Z: [{data[2].min():.3f}, {data[2].max():.3f}]
    
    Action: {action_name}
    Label: {label}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    # 6. Graph structure visualization
    ax = axes[1, 2]
    ax.axis('off')
    graph_text = f"""
    MediaPipe Graph Structure:
    
    Total Joints: {V}
    Total Bones: {len(MEDIAPIPE_BONES)}
    
    Key Joints:
    0: nose
    5-6: left/right shoulder
    9-10: left/right wrist
    15-16: left/right hip
    19-20: left/right ankle
    
    Body Parts:
    - Face: 5 joints (0-4)
    - Arms: 10 joints (5-14)
    - Torso/Hips: 2 joints (15-16)
    - Legs/Feet: 8 joints (17-24)
    """
    ax.text(0.1, 0.5, graph_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    return fig


def visualize_batch_data(data_loader, num_samples=4, save_dir='./visualizations'):
    """
    Visualize multiple samples from a data loader batch
    
    Args:
        data_loader: PyTorch DataLoader
        num_samples: number of samples to visualize
        save_dir: directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nVisualizing {num_samples} samples from data loader...")
    print(f"Batch size: {data_loader.batch_size}")
    print(f"Number of workers: {data_loader.num_workers}")
    print(f"Dataset size: {len(data_loader.dataset)}")
    
    # Get a batch
    data_iter = iter(data_loader)
    data_batch, label_batch, index_batch = next(data_iter)
    
    print(f"\nBatch data shape: {data_batch.shape}")
    print(f"Batch labels shape: {label_batch.shape}")
    
    # Convert to numpy
    data_batch = data_batch.numpy()
    label_batch = label_batch.numpy()
    
    # Visualize individual samples
    for i in range(min(num_samples, len(data_batch))):
        data = data_batch[i]
        label = label_batch[i]
        sample_name = data_loader.dataset.sample_name[index_batch[i]]
        
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"  Name: {sample_name}")
        print(f"  Label: {label} ({ACTION_LABELS[label]})")
        print(f"  Shape: {data.shape}")
        print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
        
        # Visualize multiple frames
        fig1 = visualize_multiple_frames(
            data, label, sample_name, num_frames=6,
            save_path=os.path.join(save_dir, f'sample_{i+1}_frames.png')
        )
        plt.close(fig1)
        
        # Visualize data structure
        fig2 = visualize_data_structure(data, label, sample_name)
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}_structure.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close(fig2)
    
    print(f"\n{'='*60}")
    print(f"Visualizations saved to {save_dir}/")


def create_animation(data, label, sample_name, save_path=None, fps=10):
    """
    Create an animation of the skeleton moving through time
    
    Args:
        data: numpy array of shape (C, T, V, M)
        label: action label
        sample_name: name of the sample
        save_path: path to save animation (gif)
        fps: frames per second
    """
    C, T, V, M = data.shape
    action_name = ACTION_LABELS[label] if label < len(ACTION_LABELS) else f"Action_{label}"
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_idx):
        ax.clear()
        visualize_skeleton_3d(data, frame_idx, ax, 
                            title=f'{action_name} - Frame {frame_idx}/{T-1}\nSample: {sample_name}')
        return ax,
    
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    
    if save_path:
        print(f"Saving animation to {save_path} (this may take a while)...")
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved!")
    
    return anim, fig


def main():
    parser = argparse.ArgumentParser(description='Visualize MediaPipe skeleton data')
    parser.add_argument('--data-path', type=str, 
                       default='./3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit',
                       help='Path to MMASD dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--window-size', type=int, default=64,
                       help='Window size for temporal data')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of samples to visualize')
    parser.add_argument('--save-dir', type=str, default='./visualizations/mediapipe_data',
                       help='Directory to save visualizations')
    parser.add_argument('--create-animation', action='store_true',
                       help='Create animated gif of first sample')
    parser.add_argument('--bone', action='store_true',
                       help='Use bone modality')
    parser.add_argument('--vel', action='store_true',
                       help='Use velocity modality')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for data loader')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MediaPipe Skeleton Data Visualization")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"Window size: {args.window_size}")
    print(f"Bone modality: {args.bone}")
    print(f"Velocity modality: {args.vel}")
    
    # Initialize feeder (same as used in main.py)
    print("\nInitializing data feeder...")
    feeder = Feeder(
        data_path=args.data_path,
        split=args.split,
        window_size=args.window_size,
        p_interval=[0.95],
        bone=args.bone,
        vel=args.vel,
        debug=False,
        train_val_split=0.8
    )
    
    print(f"Dataset loaded: {len(feeder)} samples")
    print(f"Number of classes: {len(set(feeder.label))}")
    
    # Initialize graph
    graph = Graph(labeling_mode='spatial')
    print(f"\nGraph structure:")
    print(f"  Number of nodes: {graph.num_node}")
    print(f"  Number of edges: {len(graph.neighbor)}")
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    
    # Visualize batch data
    visualize_batch_data(data_loader, num_samples=args.num_samples, save_dir=args.save_dir)
    
    # Create animation if requested
    if args.create_animation:
        print("\nCreating animation of first sample...")
        data, label, _ = feeder[0]
        sample_name = feeder.sample_name[0]
        
        anim, fig = create_animation(
            data, label, sample_name,
            save_path=os.path.join(args.save_dir, 'sample_animation.gif'),
            fps=10
        )
        plt.close(fig)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()


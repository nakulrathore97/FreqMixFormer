#!/usr/bin/env python
"""
Visualization script for MediaPipe skeleton data loaded from the dataset.
This script helps visualize:
1. Skeleton pose in 3D space
2. Temporal evolution of poses
3. Data statistics
4. Sample animations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns

# MediaPipe bone connections (1-indexed, convert to 0-indexed for plotting)
MEDIAPIPE_PAIRS = [
    (2, 1), (3, 1),  # eyes to nose
    (4, 2), (5, 3),  # ears to eyes
    (6, 1), (7, 1),  # shoulders to nose (center connection)
    (8, 6), (9, 7),  # elbows to shoulders
    (10, 8), (11, 9),  # wrists to elbows
    (12, 10), (13, 11),  # pinky to wrists
    (14, 10), (15, 11),  # index to wrists
    (16, 6), (17, 7),  # hips to shoulders
    (18, 16), (19, 17),  # knees to hips
    (20, 18), (21, 19),  # ankles to knees
    (22, 20), (23, 21),  # heels to ankles
    (24, 20), (25, 21),  # feet to ankles
]

# Convert to 0-indexed
MEDIAPIPE_PAIRS_0_INDEXED = [(i-1, j-1) for i, j in MEDIAPIPE_PAIRS]

# Keypoint names
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_heel', 'right_heel', 'left_foot', 'right_foot'
]


def load_mediapipe_data(data_path, split='train', num_samples=None):
    """Load MediaPipe data from NPZ file."""
    print(f"Loading data from: {data_path}")
    npz_data = np.load(data_path)
    
    if split == 'train':
        data = npz_data['x_train']
        labels = np.where(npz_data['y_train'] > 0)[1]
    else:
        data = npz_data['x_test']
        labels = np.where(npz_data['y_test'] > 0)[1]
    
    # Data shape: (N, T, 75) where 75 = 1*25*3
    # Reshape to (N, C, T, V, M) = (N, 3, T, 25, 1)
    N, T, _ = data.shape
    data = data.reshape((N, T, 1, 25, 3)).transpose(0, 4, 1, 3, 2)
    
    print(f"Data shape: {data.shape}")
    print(f"Number of samples: {N}")
    print(f"Number of frames: {T}")
    print(f"Number of unique labels: {len(np.unique(labels))}")
    
    if num_samples is not None:
        data = data[:num_samples]
        labels = labels[:num_samples]
    
    return data, labels


def plot_skeleton_3d(ax, skeleton, title="3D Skeleton Pose", show_labels=False):
    """
    Plot a single skeleton frame in 3D.
    
    Args:
        ax: matplotlib 3D axis
        skeleton: array of shape (3, V) where V=25 joints
        title: plot title
        show_labels: whether to show joint labels
    """
    # skeleton shape: (3, 25) - (coords, joints)
    x, y, z = skeleton[0], skeleton[1], skeleton[2]
    
    # Plot joints
    ax.scatter(x, y, z, c='red', s=50, marker='o', alpha=0.8)
    
    # Plot bones (connections)
    for i, j in MEDIAPIPE_PAIRS_0_INDEXED:
        if np.all(skeleton[:, i] != 0) and np.all(skeleton[:, j] != 0):
            ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 
                   'b-', linewidth=2, alpha=0.6)
    
    # Add labels if requested
    if show_labels:
        for i, name in enumerate(KEYPOINT_NAMES):
            if np.all(skeleton[:, i] != 0):
                ax.text(x[i], y[i], z[i], name, fontsize=6)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), 
                         z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_sample(data, labels, sample_idx=0, label_names=None):
    """Visualize multiple frames from a single sample."""
    sample = data[sample_idx]  # Shape: (3, T, 25, 1)
    label = labels[sample_idx]
    
    # Remove the person dimension
    sample = sample[:, :, :, 0]  # Shape: (3, T, 25)
    
    # Find valid frames (non-zero)
    valid_frames = []
    for t in range(sample.shape[1]):
        if np.any(sample[:, t, :] != 0):
            valid_frames.append(t)
    
    print(f"\nSample {sample_idx}:")
    print(f"  Label: {label}" + (f" ({label_names[label]})" if label_names else ""))
    print(f"  Total frames: {sample.shape[1]}")
    print(f"  Valid frames: {len(valid_frames)}")
    
    # Plot 4 frames evenly distributed
    if len(valid_frames) >= 4:
        frame_indices = [valid_frames[i * len(valid_frames) // 4] 
                        for i in range(4)]
    else:
        frame_indices = valid_frames[:4]
    
    fig = plt.figure(figsize=(16, 4))
    label_str = f" ({label_names[label]})" if label_names else f" {label}"
    fig.suptitle(f"Sample {sample_idx} - Label:{label_str} - Frames from sequence", 
                 fontsize=14, fontweight='bold')
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        skeleton = sample[:, frame_idx, :]  # Shape: (3, 25)
        plot_skeleton_3d(ax, skeleton, title=f"Frame {frame_idx}")
    
    plt.tight_layout()
    return fig


def create_animation(data, sample_idx=0, output_path=None):
    """Create an animation of a skeleton sequence."""
    sample = data[sample_idx][:, :, :, 0]  # Shape: (3, T, 25)
    
    # Find valid frames
    valid_frames = []
    for t in range(sample.shape[1]):
        if np.any(sample[:, t, :] != 0):
            valid_frames.append(t)
    
    if len(valid_frames) == 0:
        print("No valid frames found!")
        return None
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame_num):
        ax.clear()
        frame_idx = valid_frames[frame_num % len(valid_frames)]
        skeleton = sample[:, frame_idx, :]
        plot_skeleton_3d(ax, skeleton, 
                        title=f"Sample {sample_idx} - Frame {frame_idx}/{sample.shape[1]}")
        return ax,
    
    anim = animation.FuncAnimation(fig, update, frames=len(valid_frames),
                                  interval=50, blit=False)
    
    if output_path:
        print(f"Saving animation to {output_path}")
        anim.save(output_path, writer='pillow', fps=20)
    
    return anim


def plot_data_statistics(data, labels, label_names=None):
    """Plot various statistics about the dataset."""
    N, C, T, V, M = data.shape
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Label distribution
    ax1 = plt.subplot(2, 3, 1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if label_names:
        label_texts = [label_names[l] for l in unique_labels]
    else:
        label_texts = [f"Class {l}" for l in unique_labels]
    ax1.bar(range(len(unique_labels)), counts)
    ax1.set_xticks(range(len(unique_labels)))
    ax1.set_xticklabels(label_texts, rotation=45, ha='right')
    ax1.set_ylabel('Count')
    ax1.set_title('Label Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Valid frames per sample
    ax2 = plt.subplot(2, 3, 2)
    valid_frames_per_sample = []
    for i in range(N):
        sample = data[i, :, :, :, 0]
        valid_count = np.sum([np.any(sample[:, t, :] != 0) for t in range(T)])
        valid_frames_per_sample.append(valid_count)
    ax2.hist(valid_frames_per_sample, bins=30, edgecolor='black')
    ax2.set_xlabel('Number of Valid Frames')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title(f'Valid Frames Distribution\n(Mean: {np.mean(valid_frames_per_sample):.1f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Joint activity heatmap (which joints are used most)
    ax3 = plt.subplot(2, 3, 3)
    joint_activity = np.zeros(V)
    for v in range(V):
        joint_activity[v] = np.sum(data[:, :, :, v, :] != 0)
    ax3.barh(range(V), joint_activity)
    ax3.set_yticks(range(V))
    ax3.set_yticklabels(KEYPOINT_NAMES, fontsize=8)
    ax3.set_xlabel('Activity Count')
    ax3.set_title('Joint Activity')
    ax3.grid(True, alpha=0.3)
    
    # 4. Coordinate range distribution
    ax4 = plt.subplot(2, 3, 4)
    non_zero_data = data[data != 0]
    data_x = data[:, 0, :, :, :][data[:, 0, :, :, :] != 0]
    data_y = data[:, 1, :, :, :][data[:, 1, :, :, :] != 0]
    data_z = data[:, 2, :, :, :][data[:, 2, :, :, :] != 0]
    ax4.hist([data_x, data_y, data_z], bins=50, label=['X', 'Y', 'Z'], alpha=0.7)
    ax4.set_xlabel('Coordinate Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Coordinate Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Average skeleton visualization
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    # Compute mean skeleton across all samples and frames
    mean_skeleton = np.zeros((3, V))
    count = np.zeros(V)
    for i in range(N):
        for t in range(T):
            skeleton = data[i, :, t, :, 0]
            mask = np.any(skeleton != 0, axis=0)
            mean_skeleton[:, mask] += skeleton[:, mask]
            count[mask] += 1
    count[count == 0] = 1  # Avoid division by zero
    mean_skeleton /= count
    plot_skeleton_3d(ax5, mean_skeleton, title="Average Skeleton Pose")
    
    # 6. Sample variance across time
    ax6 = plt.subplot(2, 3, 6)
    temporal_variance = []
    for i in range(min(N, 50)):  # Sample 50 sequences
        sample = data[i, :, :, :, 0]
        # Compute variance across time for each joint
        valid_frames = [t for t in range(T) if np.any(sample[:, t, :] != 0)]
        if len(valid_frames) > 1:
            sample_variance = np.var(sample[:, valid_frames, :], axis=1).mean()
            temporal_variance.append(sample_variance)
    ax6.hist(temporal_variance, bins=20, edgecolor='black')
    ax6.set_xlabel('Temporal Variance')
    ax6.set_ylabel('Count')
    ax6.set_title('Motion Variance per Sample')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_samples_by_label(data, labels, label=0, num_samples=4):
    """Compare multiple samples of the same label."""
    # Find samples with this label
    sample_indices = np.where(labels == label)[0]
    
    if len(sample_indices) == 0:
        print(f"No samples found for label {label}")
        return None
    
    sample_indices = sample_indices[:num_samples]
    
    fig = plt.figure(figsize=(16, 4 * len(sample_indices)))
    
    for i, sample_idx in enumerate(sample_indices):
        sample = data[sample_idx, :, :, :, 0]  # Shape: (3, T, 25)
        
        # Get 4 frames from this sample
        valid_frames = [t for t in range(sample.shape[1]) 
                       if np.any(sample[:, t, :] != 0)]
        if len(valid_frames) >= 4:
            frame_indices = [valid_frames[j * len(valid_frames) // 4] 
                           for j in range(4)]
        else:
            frame_indices = valid_frames[:4]
        
        for j, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(len(sample_indices), 4, 
                               i * 4 + j + 1, projection='3d')
            skeleton = sample[:, frame_idx, :]
            plot_skeleton_3d(ax, skeleton, 
                           title=f"Sample {sample_idx}, Frame {frame_idx}")
    
    fig.suptitle(f"Comparison of Samples with Label {label}", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MediaPipe skeleton data')
    parser.add_argument('--data_path', type=str, 
                       default='data/mediapipe/mediapipe_data.npz',
                       help='Path to the NPZ data file')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Which split to visualize')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of sample to visualize')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to load (None for all)')
    parser.add_argument('--mode', type=str, default='statistics',
                       choices=['statistics', 'sample', 'animation', 'compare'],
                       help='Visualization mode')
    parser.add_argument('--save', action='store_true',
                       help='Save figures instead of showing')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for saved figures')
    
    args = parser.parse_args()
    
    # Load data
    data, labels = load_mediapipe_data(args.data_path, args.split, args.num_samples)
    
    # Load label names if available
    import os
    label_names = None
    label_map_path = os.path.join(os.path.dirname(args.data_path), 'label_map.txt')
    if os.path.exists(label_map_path):
        label_names = {}
        with open(label_map_path, 'r') as f:
            for line in f:
                idx, name = line.strip().split(': ', 1)
                label_names[int(idx)] = name
        print(f"Loaded {len(label_names)} label names")
    
    # Create output directory if saving
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform visualization based on mode
    if args.mode == 'statistics':
        print("Generating statistics...")
        fig = plot_data_statistics(data, labels, label_names)
        if args.save:
            output_path = os.path.join(args.output_dir, 
                                      f'{args.split}_statistics.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_path}")
        else:
            plt.show()
    
    elif args.mode == 'sample':
        print(f"Visualizing sample {args.sample_idx}...")
        fig = visualize_sample(data, labels, args.sample_idx, label_names)
        if args.save:
            output_path = os.path.join(args.output_dir, 
                                      f'{args.split}_sample_{args.sample_idx}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_path}")
        else:
            plt.show()
    
    elif args.mode == 'animation':
        print(f"Creating animation for sample {args.sample_idx}...")
        output_path = None
        if args.save:
            output_path = os.path.join(args.output_dir, 
                                      f'{args.split}_sample_{args.sample_idx}.gif')
        anim = create_animation(data, args.sample_idx, output_path)
        if not args.save and anim:
            plt.show()
    
    elif args.mode == 'compare':
        print(f"Comparing samples with label {labels[args.sample_idx]}...")
        fig = compare_samples_by_label(data, labels, 
                                      labels[args.sample_idx], num_samples=4)
        if fig:
            if args.save:
                output_path = os.path.join(args.output_dir, 
                                          f'{args.split}_compare_label_{labels[args.sample_idx]}.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved to {output_path}")
            else:
                plt.show()


if __name__ == "__main__":
    main()


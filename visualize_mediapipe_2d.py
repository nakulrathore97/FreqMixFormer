#!/usr/bin/env python
"""
2D Visualization script for MediaPipe skeleton data.
Creates clean 2D stick figure visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os

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


def plot_skeleton_2d(ax, skeleton, title="2D Skeleton", show_labels=False, invert_y=True):
    """
    Plot a single skeleton frame in 2D.
    
    Args:
        ax: matplotlib axis
        skeleton: array of shape (3, V) where V=25 joints
        title: plot title
        show_labels: whether to show joint labels
        invert_y: whether to invert Y axis (so skeleton appears upright)
    """
    # skeleton shape: (3, 25) - (coords, joints)
    x, y = skeleton[0], skeleton[1]
    
    # Invert y-axis so skeleton is upright
    if invert_y:
        y = -y
    
    # Plot bones (connections) first
    for i, j in MEDIAPIPE_PAIRS_0_INDEXED:
        if np.all(skeleton[:, i] != 0) and np.all(skeleton[:, j] != 0):
            ax.plot([x[i], x[j]], [y[i], y[j]], 
                   'b-', linewidth=3, alpha=0.7, solid_capstyle='round')
    
    # Plot joints on top
    valid_joints = np.where(np.any(skeleton != 0, axis=0))[0]
    if len(valid_joints) > 0:
        ax.scatter(x[valid_joints], y[valid_joints], 
                  c='red', s=100, marker='o', alpha=0.9, 
                  edgecolors='darkred', linewidths=2, zorder=5)
    
    # Add labels if requested
    if show_labels:
        for i in valid_joints:
            ax.text(x[i], y[i], KEYPOINT_NAMES[i], 
                   fontsize=8, ha='right', va='bottom')
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits with padding
    if len(valid_joints) > 0:
        x_range = x[valid_joints].max() - x[valid_joints].min()
        y_range = y[valid_joints].max() - y[valid_joints].min()
        padding = max(x_range, y_range) * 0.1
        
        ax.set_xlim(x[valid_joints].min() - padding, x[valid_joints].max() + padding)
        ax.set_ylim(y[valid_joints].min() - padding, y[valid_joints].max() + padding)


def visualize_sample_2d(data, labels, sample_idx=0, label_names=None, num_frames=8):
    """Visualize multiple frames from a single sample in 2D."""
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
    
    # Plot frames evenly distributed
    if len(valid_frames) >= num_frames:
        frame_indices = [valid_frames[i * len(valid_frames) // num_frames] 
                        for i in range(num_frames)]
    else:
        frame_indices = valid_frames[:num_frames]
    
    # Create figure with grid layout
    rows = 2
    cols = num_frames // rows
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 5*rows))
    
    label_str = f" ({label_names[label]})" if label_names else f" {label}"
    fig.suptitle(f"Sample {sample_idx} - Label:{label_str} - 2D Skeleton Sequence", 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten() if num_frames > 1 else [axes]
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes_flat[i]
        skeleton = sample[:, frame_idx, :]  # Shape: (3, 25)
        plot_skeleton_2d(ax, skeleton, title=f"Frame {frame_idx}")
    
    plt.tight_layout()
    return fig


def create_single_frame_2d(data, labels, sample_idx=0, frame_idx=None, 
                           label_names=None, figsize=(8, 10)):
    """Create a single clean 2D skeleton image."""
    sample = data[sample_idx][:, :, :, 0]  # Shape: (3, T, 25)
    label = labels[sample_idx]
    
    # Find valid frames
    valid_frames = []
    for t in range(sample.shape[1]):
        if np.any(sample[:, t, :] != 0):
            valid_frames.append(t)
    
    # If no frame specified, use middle frame
    if frame_idx is None:
        frame_idx = valid_frames[len(valid_frames) // 2] if valid_frames else 0
    
    fig, ax = plt.subplots(figsize=figsize)
    skeleton = sample[:, frame_idx, :]  # Shape: (3, 25)
    
    label_str = f"{label_names[label]}" if label_names else f"Label {label}"
    plot_skeleton_2d(ax, skeleton, 
                    title=f"Sample {sample_idx}: {label_str}\nFrame {frame_idx}", 
                    show_labels=False)
    
    plt.tight_layout()
    return fig


def create_animation_2d(data, labels, sample_idx=0, label_names=None, output_path=None):
    """Create a 2D animation of a skeleton sequence."""
    sample = data[sample_idx][:, :, :, 0]  # Shape: (3, T, 25)
    label = labels[sample_idx]
    
    # Find valid frames
    valid_frames = []
    for t in range(sample.shape[1]):
        if np.any(sample[:, t, :] != 0):
            valid_frames.append(t)
    
    if len(valid_frames) == 0:
        print("No valid frames found!")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    label_str = f"{label_names[label]}" if label_names else f"Label {label}"
    
    def update(frame_num):
        ax.clear()
        frame_idx = valid_frames[frame_num % len(valid_frames)]
        skeleton = sample[:, frame_idx, :]
        plot_skeleton_2d(ax, skeleton, 
                        title=f"Sample {sample_idx}: {label_str}\nFrame {frame_idx}/{sample.shape[1]}")
        return ax,
    
    anim = animation.FuncAnimation(fig, update, frames=len(valid_frames),
                                  interval=50, blit=False)
    
    if output_path:
        print(f"Saving animation to {output_path}")
        anim.save(output_path, writer='pillow', fps=20)
        print(f"Animation saved!")
    
    return anim


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MediaPipe skeleton data in 2D')
    parser.add_argument('--data_path', type=str, 
                       default='data/mediapipe/mediapipe_data.npz',
                       help='Path to the NPZ data file')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Which split to visualize')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of sample to visualize')
    parser.add_argument('--frame_idx', type=int, default=None,
                       help='Specific frame to visualize (None for middle frame)')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'sequence', 'animation'],
                       help='Visualization mode')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of frames to show in sequence mode')
    parser.add_argument('--save', action='store_true',
                       help='Save figures instead of showing')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for saved figures')
    
    args = parser.parse_args()
    
    # Load data
    data, labels = load_mediapipe_data(args.data_path, args.split)
    
    # Load label names if available
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
    if args.mode == 'single':
        print(f"Creating single frame visualization for sample {args.sample_idx}...")
        fig = create_single_frame_2d(data, labels, args.sample_idx, 
                                     args.frame_idx, label_names)
        if args.save:
            frame_str = f"_frame_{args.frame_idx}" if args.frame_idx else ""
            output_path = os.path.join(args.output_dir, 
                                      f'{args.split}_sample_{args.sample_idx}_2d{frame_str}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved to {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    elif args.mode == 'sequence':
        print(f"Creating sequence visualization for sample {args.sample_idx}...")
        fig = visualize_sample_2d(data, labels, args.sample_idx, 
                                 label_names, args.num_frames)
        if args.save:
            output_path = os.path.join(args.output_dir, 
                                      f'{args.split}_sample_{args.sample_idx}_2d_sequence.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved to {output_path}")
            plt.close(fig)
        else:
            plt.show()
    
    elif args.mode == 'animation':
        print(f"Creating 2D animation for sample {args.sample_idx}...")
        output_path = None
        if args.save:
            output_path = os.path.join(args.output_dir, 
                                      f'{args.split}_sample_{args.sample_idx}_2d.gif')
        anim = create_animation_2d(data, labels, args.sample_idx, 
                                  label_names, output_path)
        if not args.save and anim:
            plt.show()


if __name__ == "__main__":
    main()


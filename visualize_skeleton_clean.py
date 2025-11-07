#!/usr/bin/env python
"""
Clean 2D skeleton visualization for MediaPipe pose data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import argparse

# MediaPipe bone connections (0-indexed)
MEDIAPIPE_PAIRS = [
    (1, 0), (2, 0),  # eyes to nose
    (3, 1), (4, 2),  # ears to eyes  
    (5, 0), (6, 0),  # shoulders to nose
    (7, 5), (8, 6),  # elbows to shoulders
    (9, 7), (10, 8),  # wrists to elbows
    (11, 9), (12, 10),  # pinky to wrists
    (13, 9), (14, 10),  # index to wrists
    (15, 5), (16, 6),  # hips to shoulders
    (17, 15), (18, 16),  # knees to hips
    (19, 17), (20, 18),  # ankles to knees
    (21, 19), (22, 20),  # heels to ankles
    (23, 19), (24, 20),  # feet to ankles
]

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_heel', 'right_heel', 'left_foot', 'right_foot'
]


def load_data(data_path, split='train'):
    """Load MediaPipe data from NPZ file."""
    npz_data = np.load(data_path)
    
    if split == 'train':
        data = npz_data['x_train']
        labels = np.where(npz_data['y_train'] > 0)[1]
    else:
        data = npz_data['x_test']
        labels = np.where(npz_data['y_test'] > 0)[1]
    
    # Load label names
    label_names = {}
    label_map_path = os.path.join(os.path.dirname(data_path), 'label_map.txt')
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            for line in f:
                idx, name = line.strip().split(': ', 1)
                label_names[int(idx)] = name
    
    return data, labels, label_names


def get_skeleton(data, sample_idx, frame_idx=None):
    """Extract skeleton from data for given sample and frame."""
    N, T, _ = data.shape
    
    # Reshape: (T, 75) -> (T, M*V*C) -> (T, M, V, C)
    sample_flat = data[sample_idx]  # (T, 75)
    sample_reshaped = sample_flat.reshape((T, 1, 25, 3))  # (T, M=1, V=25, C=3)
    sample = sample_reshaped.transpose(3, 0, 1, 2)  # (C=3, T, M=1, V=25)
    
    # Find valid frames
    valid_frames = []
    for t in range(T):
        frame = sample[:, t, 0, :]  # (3, 25)
        if np.any(frame != 0):
            valid_frames.append(t)
    
    # Pick frame
    if frame_idx is None:
        frame_idx = valid_frames[len(valid_frames) // 2] if valid_frames else 0
    elif frame_idx not in valid_frames:
        frame_idx = valid_frames[0] if valid_frames else 0
    
    skeleton = sample[:, frame_idx, 0, :]  # (3, 25)
    return skeleton, frame_idx, valid_frames


def plot_skeleton_2d(ax, skeleton, title="", show_labels=False):
    """Plot skeleton in 2D (front view)."""
    x, y, z = skeleton[0], skeleton[1], skeleton[2]
    
    # Invert Y so person is upright
    y = -y
    
    # Plot bones first (connections)
    for i, j in MEDIAPIPE_PAIRS:
        if x[i] != 0 and x[j] != 0:
            ax.plot([x[i], x[j]], [y[i], y[j]], 
                   'dodgerblue', linewidth=4, alpha=0.8, solid_capstyle='round', zorder=1)
    
    # Plot joints on top
    valid_joints = np.where(np.any(skeleton != 0, axis=0))[0]
    if len(valid_joints) > 0:
        ax.scatter(x[valid_joints], y[valid_joints], 
                  c='orangered', s=120, marker='o', alpha=1.0,
                  edgecolors='darkred', linewidths=2.5, zorder=5)
    
    # Add labels if requested
    if show_labels:
        for i in valid_joints:
            ax.text(x[i] + 0.05, y[i] + 0.05, KEYPOINT_NAMES[i], 
                   fontsize=7, ha='left', va='bottom', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Styling
    ax.set_xlabel('X (Left ← → Right)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Down ← → Up)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f9f9f9')
    
    # Set axis limits with padding
    if len(valid_joints) > 0:
        x_range = x[valid_joints].max() - x[valid_joints].min()
        y_range = y[valid_joints].max() - y[valid_joints].min()
        padding = max(x_range, y_range) * 0.15
        
        x_center = (x[valid_joints].max() + x[valid_joints].min()) / 2
        y_center = (y[valid_joints].max() + y[valid_joints].min()) / 2
        axis_range = max(x_range, y_range) / 2 + padding
        
        ax.set_xlim(x_center - axis_range, x_center + axis_range)
        ax.set_ylim(y_center - axis_range, y_center + axis_range)


def visualize_single_frame(data, labels, label_names, sample_idx=0, frame_idx=None, 
                          show_labels=False, save_path=None):
    """Create a single clean skeleton image."""
    skeleton, frame_idx, valid_frames = get_skeleton(data, sample_idx, frame_idx)
    label = labels[sample_idx]
    label_name = label_names.get(label, f"Label {label}")
    
    fig, ax = plt.subplots(figsize=(10, 12), facecolor='white')
    
    title = (f"Sample {sample_idx}: {label_name}\n"
            f"Frame {frame_idx}/{data.shape[1]} "
            f"({len(valid_frames)} valid frames)")
    
    plot_skeleton_2d(ax, skeleton, title=title, show_labels=show_labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()
    
    return fig


def visualize_sequence(data, labels, label_names, sample_idx=0, num_frames=8, save_path=None):
    """Visualize a sequence of frames."""
    skeleton_full, _, valid_frames = get_skeleton(data, sample_idx)
    label = labels[sample_idx]
    label_name = label_names.get(label, f"Label {label}")
    
    # Select evenly spaced frames
    if len(valid_frames) >= num_frames:
        frame_indices = [valid_frames[i * len(valid_frames) // num_frames] 
                        for i in range(num_frames)]
    else:
        frame_indices = valid_frames[:num_frames]
    
    # Create grid
    rows = 2
    cols = (num_frames + 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 5*rows), facecolor='white')
    
    fig.suptitle(f"Sample {sample_idx}: {label_name} - Sequence", 
                fontsize=18, fontweight='bold', y=0.995)
    
    axes_flat = axes.flatten() if num_frames > 1 else [axes]
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes_flat[i]
        skeleton, _, _ = get_skeleton(data, sample_idx, frame_idx)
        plot_skeleton_2d(ax, skeleton, title=f"Frame {frame_idx}", show_labels=False)
    
    # Hide extra subplots
    for i in range(len(frame_indices), len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()
    
    return fig


def create_animation_gif(data, labels, label_names, sample_idx=0, save_path=None):
    """Create an animated GIF of the skeleton sequence."""
    N, T, _ = data.shape
    sample_flat = data[sample_idx].reshape((T, 1, 25, 3))
    sample = sample_flat.transpose(3, 0, 1, 2)  # (C, T, M, V)
    
    label = labels[sample_idx]
    label_name = label_names.get(label, f"Label {label}")
    
    # Find valid frames
    valid_frames = []
    for t in range(T):
        frame = sample[:, t, 0, :]
        if np.any(frame != 0):
            valid_frames.append(t)
    
    if not valid_frames:
        print("No valid frames found!")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 12), facecolor='white')
    
    def update(frame_num):
        ax.clear()
        frame_idx = valid_frames[frame_num % len(valid_frames)]
        skeleton = sample[:, frame_idx, 0, :]
        
        title = (f"Sample {sample_idx}: {label_name}\n"
                f"Frame {frame_idx}/{T} ({frame_num+1}/{len(valid_frames)})")
        
        plot_skeleton_2d(ax, skeleton, title=title, show_labels=False)
        return ax,
    
    anim = animation.FuncAnimation(fig, update, frames=len(valid_frames),
                                  interval=50, blit=False)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved!")
        plt.close()
    else:
        plt.show()
    
    return anim


def main():
    parser = argparse.ArgumentParser(description='Visualize MediaPipe skeleton in 2D')
    parser.add_argument('--data_path', type=str, 
                       default='data/mediapipe/mediapipe_data.npz',
                       help='Path to NPZ data file')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset split')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--frame_idx', type=int, default=None,
                       help='Frame index (None for middle frame)')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'sequence', 'animation'],
                       help='Visualization mode')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of frames for sequence mode')
    parser.add_argument('--show_labels', action='store_true',
                       help='Show joint labels')
    parser.add_argument('--save', action='store_true',
                       help='Save output')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.split} data from {args.data_path}...")
    data, labels, label_names = load_data(args.data_path, args.split)
    print(f"Loaded {len(data)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualization
    if args.mode == 'single':
        save_path = None
        if args.save:
            frame_str = f"_frame{args.frame_idx}" if args.frame_idx is not None else ""
            save_path = os.path.join(args.output_dir, 
                                    f'skeleton_2d_sample{args.sample_idx}{frame_str}.png')
        
        visualize_single_frame(data, labels, label_names, 
                             args.sample_idx, args.frame_idx, 
                             args.show_labels, save_path)
    
    elif args.mode == 'sequence':
        save_path = None
        if args.save:
            save_path = os.path.join(args.output_dir, 
                                    f'skeleton_2d_sample{args.sample_idx}_sequence.png')
        
        visualize_sequence(data, labels, label_names, 
                         args.sample_idx, args.num_frames, save_path)
    
    elif args.mode == 'animation':
        save_path = None
        if args.save:
            save_path = os.path.join(args.output_dir, 
                                    f'skeleton_2d_sample{args.sample_idx}.gif')
        
        create_animation_gif(data, labels, label_names, 
                           args.sample_idx, save_path)


if __name__ == "__main__":
    main()


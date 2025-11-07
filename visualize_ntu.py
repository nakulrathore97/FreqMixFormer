#!/usr/bin/env python
"""
Visualization script for NTU RGB+D skeleton data.
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
import os

# NTU RGB+D skeleton connections (1-indexed, convert to 0-indexed for plotting)
NTU_PAIRS = [
    (2, 1), (2, 21), (21, 3), (3, 4),  # head
    (21, 5), (5, 6), (6, 7), (7, 8), (8, 23), (23, 22),  # left arm
    (21, 9), (9, 10), (10, 11), (11, 12), (12, 25), (25, 24),  # right arm
    (1, 13), (13, 14), (14, 15), (15, 16),  # left leg
    (1, 17), (17, 18), (18, 19), (19, 20)  # right leg
]

# Convert to 0-indexed
NTU_PAIRS_0_INDEXED = [(i-1, j-1) for i, j in NTU_PAIRS]

# Joint names for NTU skeleton
JOINT_NAMES = [
    'base_spine', 'mid_spine', 'neck', 'head',  # 0-3
    'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand',  # 4-7
    'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand',  # 8-11
    'left_hip', 'left_knee', 'left_ankle', 'left_foot',  # 12-15
    'right_hip', 'right_knee', 'right_ankle', 'right_foot',  # 16-19
    'spine', 'left_hand_tip', 'left_thumb',  # 20-22
    'right_hand_tip', 'right_thumb'  # 23-24
]

# NTU RGB+D action labels (60 classes)
NTU_ACTION_LABELS = [
    'drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop',
    'pickup', 'throw', 'sitting down', 'standing up (from sitting position)', 'clapping',
    'reading', 'writing', 'tear up paper', 'wear jacket', 'take off jacket',
    'wear a shoe', 'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap',
    'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 'reach into pocket',
    'hopping (one foot jumping)', 'jump up', 'make a phone call/answer phone', 'playing with phone/tablet', 'typing on a keyboard',
    'pointing to something with finger', 'taking a selfie', 'check time (from watch)', 'rub two hands together', 'nod head/bow',
    'shake head', 'wipe face', 'salute', 'put the palms together', 'cross hands in front (say stop)',
    'sneeze/cough', 'staggering', 'falling', 'touch head (headache)', 'touch chest (stomachache/heart pain)',
    'touch back (backache)', 'touch neck (neckache)', 'nausea or vomiting condition', 'use a fan (with hand or paper)/feeling warm', 'punching/slapping other person',
    'kicking other person', 'pushing other person', 'pat on back of other person', 'point finger at the other person', 'hugging other person',
    'giving something to other person', 'touch other person\'s pocket', 'handshaking', 'walking towards each other', 'walking apart from each other'
]

# NTU RGB+D 120 additional action labels (classes 61-120)
NTU120_ADDITIONAL_LABELS = [
    'put on headphone', 'take off headphone', 'shoot at the basket', 'bounce ball', 'tennis bat swing',
    'juggling table tennis balls', 'hush (quite)', 'flick hair', 'thumb up', 'thumb down',
    'make ok sign', 'make victory sign', 'staple book', 'counting money', 'cutting nails',
    'cutting paper (using scissors)', 'snapping fingers', 'open bottle', 'sniff (smell)', 'squat down',
    'toss a coin', 'fold paper', 'ball up paper', 'play magic cube', 'apply cream on face',
    'apply cream on hand back', 'put on bag', 'take off bag', 'put something into a bag', 'take something out of a bag',
    'open a box', 'move heavy objects', 'shake fist', 'throw up cap/hat', 'hands up (both hands)',
    'cross arms', 'arm circles', 'arm swings', 'running on the spot', 'butt kicks (kick backward)',
    'cross toe touch', 'side kick', 'yawn', 'stretch oneself', 'blow nose',
    'hit other person with something', 'wield knife towards other person', 'knock over other person (hit with body)', 'grab other person\'s stuff', 'shoot at other person with a gun',
    'step on foot', 'high-five', 'cheers and drink', 'carry something with other person', 'take a photo of other person',
    'follow other person', 'whisper in other person\'s ear', 'exchange things with other person', 'support somebody with hand', 'finger-guessing game (playing rock-paper-scissors)'
]

NTU120_ACTION_LABELS = NTU_ACTION_LABELS + NTU120_ADDITIONAL_LABELS


def load_ntu_data(data_path, split='train', num_samples=None):
    """Load NTU data from NPZ file."""
    print(f"Loading data from: {data_path}")
    npz_data = np.load(data_path)
    
    if split == 'train':
        data = npz_data['x_train']
        labels = np.where(npz_data['y_train'] > 0)[1]
    else:
        data = npz_data['x_test']
        labels = np.where(npz_data['y_test'] > 0)[1]
    
    # Data shape: (N, T, 150) where 150 = 2*25*3 (2 people, 25 joints, 3 coords)
    # Reshape to (N, C, T, V, M) = (N, 3, T, 25, 2)
    N, T, _ = data.shape
    data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
    
    print(f"Data shape: {data.shape}")
    print(f"Number of samples: {N}")
    print(f"Number of frames: {T}")
    print(f"Number of unique labels: {len(np.unique(labels))}")
    print(f"Label range: {labels.min()} - {labels.max()}")
    
    if num_samples is not None:
        data = data[:num_samples]
        labels = labels[:num_samples]
    
    return data, labels


def plot_skeleton_3d(ax, skeleton, title="3D Skeleton Pose", show_labels=False, color='red'):
    """
    Plot a single skeleton frame in 3D.
    
    Args:
        ax: matplotlib 3D axis
        skeleton: array of shape (3, V) where V=25 joints
        title: plot title
        show_labels: whether to show joint labels
        color: color for the joints
    """
    # skeleton shape: (3, 25) - (coords, joints)
    x, y, z = skeleton[0], skeleton[1], skeleton[2]
    
    # Check if skeleton has any valid data
    if not np.any(skeleton != 0):
        ax.text(0.5, 0.5, 0.5, 'No data', transform=ax.transAxes,
               ha='center', va='center', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        return
    
    # Plot joints
    valid_joints = np.any(skeleton != 0, axis=0)
    if np.any(valid_joints):
        ax.scatter(x[valid_joints], y[valid_joints], z[valid_joints], 
                  c=color, s=50, marker='o', alpha=0.8)
    
    # Plot bones (connections)
    for i, j in NTU_PAIRS_0_INDEXED:
        if valid_joints[i] and valid_joints[j]:
            ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 
                   'b-', linewidth=2, alpha=0.6)
    
    # Add labels if requested
    if show_labels:
        for i, name in enumerate(JOINT_NAMES):
            if valid_joints[i]:
                ax.text(x[i], y[i], z[i], name, fontsize=6)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    valid_x = x[valid_joints]
    valid_y = y[valid_joints]
    valid_z = z[valid_joints]
    
    if len(valid_x) > 0:
        max_range = np.array([valid_x.max()-valid_x.min(), 
                            valid_y.max()-valid_y.min(), 
                            valid_z.max()-valid_z.min()]).max() / 2.0
        mid_x = (valid_x.max()+valid_x.min()) * 0.5
        mid_y = (valid_y.max()+valid_y.min()) * 0.5
        mid_z = (valid_z.max()+valid_z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_sample(data, labels, sample_idx=0, label_names=None, show_both_people=True):
    """Visualize multiple frames from a single sample."""
    sample = data[sample_idx]  # Shape: (3, T, 25, 2)
    label = labels[sample_idx]
    
    # Find valid frames (non-zero)
    valid_frames = []
    for t in range(sample.shape[1]):
        if np.any(sample[:, t, :, :] != 0):
            valid_frames.append(t)
    
    print(f"\nSample {sample_idx}:")
    print(f"  Label: {label}" + (f" ({label_names[label]})" if label_names else ""))
    print(f"  Total frames: {sample.shape[1]}")
    print(f"  Valid frames: {len(valid_frames)}")
    
    # Check if there are two people
    person1_valid = np.any(sample[:, :, :, 0] != 0)
    person2_valid = np.any(sample[:, :, :, 1] != 0)
    print(f"  Person 1: {'Present' if person1_valid else 'Not present'}")
    print(f"  Person 2: {'Present' if person2_valid else 'Not present'}")
    
    # Plot 4 frames evenly distributed
    if len(valid_frames) >= 4:
        frame_indices = [valid_frames[i * len(valid_frames) // 4] 
                        for i in range(4)]
    else:
        frame_indices = valid_frames[:4] if valid_frames else [0, 0, 0, 0]
    
    # Determine number of rows (1 or 2 depending on whether we show both people)
    num_people = 2 if (show_both_people and person1_valid and person2_valid) else 1
    fig = plt.figure(figsize=(16, 4 * num_people))
    label_str = f" ({label_names[label]})" if label_names else f" {label}"
    fig.suptitle(f"Sample {sample_idx} - Label:{label_str} - Frames from sequence", 
                 fontsize=14, fontweight='bold')
    
    for i, frame_idx in enumerate(frame_indices):
        # Plot person 1
        ax = fig.add_subplot(num_people, 4, i+1, projection='3d')
        skeleton1 = sample[:, frame_idx, :, 0]  # Shape: (3, 25)
        plot_skeleton_3d(ax, skeleton1, title=f"Person 1 - Frame {frame_idx}", color='red')
        
        # Plot person 2 if present
        if num_people == 2:
            ax = fig.add_subplot(num_people, 4, 4 + i+1, projection='3d')
            skeleton2 = sample[:, frame_idx, :, 1]  # Shape: (3, 25)
            plot_skeleton_3d(ax, skeleton2, title=f"Person 2 - Frame {frame_idx}", color='blue')
    
    plt.tight_layout()
    return fig


def create_animation(data, sample_idx=0, output_path=None, show_both_people=True):
    """Create an animation of a skeleton sequence."""
    sample = data[sample_idx]  # Shape: (3, T, 25, 2)
    
    # Find valid frames
    valid_frames = []
    for t in range(sample.shape[1]):
        if np.any(sample[:, t, :, :] != 0):
            valid_frames.append(t)
    
    if len(valid_frames) == 0:
        print("No valid frames found!")
        return None
    
    # Check if there are two people
    person1_valid = np.any(sample[:, :, :, 0] != 0)
    person2_valid = np.any(sample[:, :, :, 1] != 0)
    num_people = 2 if (show_both_people and person1_valid and person2_valid) else 1
    
    fig = plt.figure(figsize=(10 * num_people, 8))
    
    def update(frame_num):
        fig.clear()
        frame_idx = valid_frames[frame_num % len(valid_frames)]
        
        # Plot person 1
        ax1 = fig.add_subplot(1, num_people, 1, projection='3d')
        skeleton1 = sample[:, frame_idx, :, 0]
        plot_skeleton_3d(ax1, skeleton1, 
                        title=f"Person 1 - Frame {frame_idx}/{sample.shape[1]}", 
                        color='red')
        
        # Plot person 2 if present
        if num_people == 2:
            ax2 = fig.add_subplot(1, num_people, 2, projection='3d')
            skeleton2 = sample[:, frame_idx, :, 1]
            plot_skeleton_3d(ax2, skeleton2, 
                           title=f"Person 2 - Frame {frame_idx}/{sample.shape[1]}", 
                           color='blue')
        
        return fig,
    
    anim = animation.FuncAnimation(fig, update, frames=len(valid_frames),
                                  interval=50, blit=False)
    
    if output_path:
        print(f"Saving animation to {output_path}")
        anim.save(output_path, writer='pillow', fps=20)
    
    return anim


def plot_data_statistics(data, labels, label_names=None):
    """Plot various statistics about the dataset."""
    N, C, T, V, M = data.shape
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Label distribution
    ax1 = plt.subplot(3, 3, 1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    ax1.bar(range(len(unique_labels)), counts)
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Label Distribution ({len(unique_labels)} classes)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Valid frames per sample
    ax2 = plt.subplot(3, 3, 2)
    valid_frames_per_sample = []
    for i in range(N):
        valid_count = np.sum([np.any(data[i, :, t, :, :] != 0) for t in range(T)])
        valid_frames_per_sample.append(valid_count)
    ax2.hist(valid_frames_per_sample, bins=50, edgecolor='black')
    ax2.set_xlabel('Number of Valid Frames')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title(f'Valid Frames Distribution\n(Mean: {np.mean(valid_frames_per_sample):.1f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Joint activity heatmap (which joints are used most)
    ax3 = plt.subplot(3, 3, 3)
    joint_activity = np.zeros(V)
    for v in range(V):
        joint_activity[v] = np.sum(data[:, :, :, v, :] != 0)
    ax3.barh(range(V), joint_activity)
    ax3.set_yticks(range(V))
    ax3.set_yticklabels(JOINT_NAMES, fontsize=7)
    ax3.set_xlabel('Activity Count')
    ax3.set_title('Joint Activity')
    ax3.grid(True, alpha=0.3)
    
    # 4. Coordinate range distribution
    ax4 = plt.subplot(3, 3, 4)
    data_x = data[:, 0, :, :, :][data[:, 0, :, :, :] != 0]
    data_y = data[:, 1, :, :, :][data[:, 1, :, :, :] != 0]
    data_z = data[:, 2, :, :, :][data[:, 2, :, :, :] != 0]
    ax4.hist([data_x, data_y, data_z], bins=50, label=['X', 'Y', 'Z'], alpha=0.7)
    ax4.set_xlabel('Coordinate Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Coordinate Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Average skeleton visualization (Person 1)
    ax5 = fig.add_subplot(3, 3, 5, projection='3d')
    mean_skeleton = np.zeros((3, V))
    count = np.zeros(V)
    for i in range(N):
        for t in range(T):
            skeleton = data[i, :, t, :, 0]
            mask = np.any(skeleton != 0, axis=0)
            mean_skeleton[:, mask] += skeleton[:, mask]
            count[mask] += 1
    count[count == 0] = 1
    mean_skeleton /= count
    plot_skeleton_3d(ax5, mean_skeleton, title="Average Skeleton (Person 1)", color='red')
    
    # 6. Number of people per sample
    ax6 = plt.subplot(3, 3, 6)
    people_count = []
    for i in range(N):
        person1_present = np.any(data[i, :, :, :, 0] != 0)
        person2_present = np.any(data[i, :, :, :, 1] != 0)
        people_count.append(int(person1_present) + int(person2_present))
    unique_counts, counts = np.unique(people_count, return_counts=True)
    ax6.bar(unique_counts, counts)
    ax6.set_xlabel('Number of People')
    ax6.set_ylabel('Number of Samples')
    ax6.set_title('People per Sample Distribution')
    ax6.set_xticks(unique_counts)
    ax6.grid(True, alpha=0.3)
    
    # 7. Top 15 most frequent action classes
    ax7 = plt.subplot(3, 3, 7)
    unique_labels, counts = np.unique(labels, return_counts=True)
    top_15_idx = np.argsort(counts)[-15:]
    top_15_labels = unique_labels[top_15_idx]
    top_15_counts = counts[top_15_idx]
    if label_names:
        top_15_names = [label_names[l][:20] + '...' if len(label_names[l]) > 20 
                       else label_names[l] for l in top_15_labels]
    else:
        top_15_names = [f"Class {l}" for l in top_15_labels]
    ax7.barh(range(len(top_15_labels)), top_15_counts)
    ax7.set_yticks(range(len(top_15_labels)))
    ax7.set_yticklabels(top_15_names, fontsize=8)
    ax7.set_xlabel('Count')
    ax7.set_title('Top 15 Most Frequent Actions')
    ax7.grid(True, alpha=0.3)
    
    # 8. Temporal variance
    ax8 = plt.subplot(3, 3, 8)
    temporal_variance = []
    for i in range(min(N, 100)):
        sample = data[i, :, :, :, 0]
        valid_frames = [t for t in range(T) if np.any(sample[:, t, :] != 0)]
        if len(valid_frames) > 1:
            sample_variance = np.var(sample[:, valid_frames, :], axis=1).mean()
            temporal_variance.append(sample_variance)
    ax8.hist(temporal_variance, bins=30, edgecolor='black')
    ax8.set_xlabel('Temporal Variance')
    ax8.set_ylabel('Count')
    ax8.set_title('Motion Variance per Sample')
    ax8.grid(True, alpha=0.3)
    
    # 9. Average skeleton visualization (Person 2)
    ax9 = fig.add_subplot(3, 3, 9, projection='3d')
    mean_skeleton2 = np.zeros((3, V))
    count2 = np.zeros(V)
    for i in range(N):
        for t in range(T):
            skeleton = data[i, :, t, :, 1]
            mask = np.any(skeleton != 0, axis=0)
            mean_skeleton2[:, mask] += skeleton[:, mask]
            count2[mask] += 1
    count2[count2 == 0] = 1
    mean_skeleton2 /= count2
    plot_skeleton_3d(ax9, mean_skeleton2, title="Average Skeleton (Person 2)", color='blue')
    
    plt.tight_layout()
    return fig


def compare_samples_by_label(data, labels, label=0, num_samples=4, label_names=None):
    """Compare multiple samples of the same label."""
    # Find samples with this label
    sample_indices = np.where(labels == label)[0]
    
    if len(sample_indices) == 0:
        print(f"No samples found for label {label}")
        return None
    
    sample_indices = sample_indices[:num_samples]
    
    fig = plt.figure(figsize=(16, 4 * len(sample_indices)))
    
    for i, sample_idx in enumerate(sample_indices):
        sample = data[sample_idx, :, :, :, 0]  # Shape: (3, T, 25) - only person 1
        
        # Get 4 frames from this sample
        valid_frames = [t for t in range(sample.shape[1]) 
                       if np.any(sample[:, t, :] != 0)]
        if len(valid_frames) >= 4:
            frame_indices = [valid_frames[j * len(valid_frames) // 4] 
                           for j in range(4)]
        else:
            frame_indices = valid_frames[:4] if valid_frames else [0, 0, 0, 0]
        
        for j, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(len(sample_indices), 4, 
                               i * 4 + j + 1, projection='3d')
            skeleton = sample[:, frame_idx, :]
            plot_skeleton_3d(ax, skeleton, 
                           title=f"Sample {sample_idx}, Frame {frame_idx}",
                           color='red')
    
    label_str = f" ({label_names[label]})" if label_names else ""
    fig.suptitle(f"Comparison of Samples with Label {label}{label_str}", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize NTU RGB+D skeleton data')
    parser.add_argument('--data_path', type=str, 
                       default='data/ntu/NTU60_CS.npz',
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
    parser.add_argument('--single_person', action='store_true',
                       help='Only show person 1 (for two-person actions)')
    
    args = parser.parse_args()
    
    # Load data
    data, labels = load_ntu_data(args.data_path, args.split, args.num_samples)
    
    # Determine label names based on dataset
    label_names = None
    if '120' in args.data_path.lower():
        label_names = {i: NTU120_ACTION_LABELS[i] for i in range(len(NTU120_ACTION_LABELS))}
        print(f"Using NTU RGB+D 120 action labels (120 classes)")
    else:
        label_names = {i: NTU_ACTION_LABELS[i] for i in range(len(NTU_ACTION_LABELS))}
        print(f"Using NTU RGB+D action labels (60 classes)")
    
    # Create output directory if saving
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    show_both = not args.single_person
    
    # Perform visualization based on mode
    if args.mode == 'statistics':
        print("Generating statistics...")
        fig = plot_data_statistics(data, labels, label_names)
        if args.save:
            dataset_name = 'ntu120' if '120' in args.data_path.lower() else 'ntu60'
            output_path = os.path.join(args.output_dir, 
                                      f'{dataset_name}_{args.split}_statistics.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_path}")
        else:
            plt.show()
    
    elif args.mode == 'sample':
        print(f"Visualizing sample {args.sample_idx}...")
        fig = visualize_sample(data, labels, args.sample_idx, label_names, show_both)
        if args.save:
            dataset_name = 'ntu120' if '120' in args.data_path.lower() else 'ntu60'
            output_path = os.path.join(args.output_dir, 
                                      f'{dataset_name}_{args.split}_sample_{args.sample_idx}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_path}")
        else:
            plt.show()
    
    elif args.mode == 'animation':
        print(f"Creating animation for sample {args.sample_idx}...")
        output_path = None
        if args.save:
            dataset_name = 'ntu120' if '120' in args.data_path.lower() else 'ntu60'
            output_path = os.path.join(args.output_dir, 
                                      f'{dataset_name}_{args.split}_sample_{args.sample_idx}.gif')
        anim = create_animation(data, args.sample_idx, output_path, show_both)
        if not args.save and anim:
            plt.show()
    
    elif args.mode == 'compare':
        print(f"Comparing samples with label {labels[args.sample_idx]}...")
        fig = compare_samples_by_label(data, labels, 
                                      labels[args.sample_idx], 
                                      num_samples=4,
                                      label_names=label_names)
        if fig:
            if args.save:
                dataset_name = 'ntu120' if '120' in args.data_path.lower() else 'ntu60'
                output_path = os.path.join(args.output_dir, 
                                          f'{dataset_name}_{args.split}_compare_label_{labels[args.sample_idx]}.png')
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved to {output_path}")
            else:
                plt.show()


if __name__ == "__main__":
    main()


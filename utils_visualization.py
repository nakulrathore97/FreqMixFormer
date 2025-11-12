"""
Utility functions for visualizing data during training/testing.
Can be imported and used in main.py or other scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


# MediaPipe bone connections (0-indexed)
MEDIAPIPE_BONES = [
    (1, 0), (2, 0), (3, 1), (4, 2),  # Face
    (5, 0), (6, 0),  # Shoulders to nose
    (7, 5), (9, 7), (8, 6), (10, 8),  # Arms
    (11, 9), (13, 9), (12, 10), (14, 10),  # Hands
    (15, 5), (16, 6), (16, 15),  # Torso
    (17, 15), (19, 17), (18, 16), (20, 18),  # Legs
    (21, 19), (23, 19), (22, 20), (24, 20),  # Feet
]

# MMASD action labels
ACTION_LABELS = [
    'Arm_Swing', 'Body_pose', 'chest_expansion', 'Drumming', 'Frog_Pose',
    'Marcas_Forward', 'Marcas_Shaking', 'Sing_Clap', 'Squat_Pose', 
    'Tree_Pose', 'Twist_Pose'
]


def plot_skeleton_frame(ax, data, frame_idx, title="", show_labels=False):
    """
    Plot a single frame of skeleton data in 3D
    
    Args:
        ax: matplotlib 3D axis
        data: numpy array of shape (C, T, V, M) where C=3, V=25
        frame_idx: which frame to plot
        title: plot title
        show_labels: whether to show joint index labels
    """
    # Extract coordinates for this frame
    coords = data[:, frame_idx, :, 0]  # Shape: (3, 25)
    x, y, z = coords[0], coords[1], coords[2]
    
    # Plot joints
    ax.scatter(x, y, z, c='red', marker='o', s=50, alpha=0.8, label='Joints')
    
    # Plot bones
    for j1, j2 in MEDIAPIPE_BONES:
        if j1 < len(x) and j2 < len(x):
            ax.plot([x[j1], x[j2]], [y[j1], y[j2]], [z[j1], z[j2]], 
                    'b-', linewidth=2, alpha=0.6)
    
    # Show labels for key joints
    if show_labels:
        key_joints = [0, 5, 6, 9, 10, 15, 16, 19, 20]
        for i in key_joints:
            if i < len(x):
                ax.text(x[i], y[i], z[i], f'{i}', fontsize=8, color='darkgreen')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min()) / 2.0
    mid_x, mid_y, mid_z = (x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=20, azim=45)


def visualize_sample(data, label=None, sample_name=None, num_frames=6, 
                     save_path=None, show_plot=False):
    """
    Visualize multiple frames of a single sample
    
    Args:
        data: numpy array of shape (C, T, V, M)
        label: action label (int)
        sample_name: name of the sample
        num_frames: number of frames to display
        save_path: path to save figure (if None, doesn't save)
        show_plot: whether to display the plot
        
    Returns:
        matplotlib figure object
    """
    C, T, V, M = data.shape
    
    # Select evenly spaced frames
    frame_indices = np.linspace(0, T-1, num_frames, dtype=int)
    
    # Create subplots
    cols = 3
    rows = (num_frames + cols - 1) // cols
    fig = plt.figure(figsize=(15, 5 * rows))
    
    # Title
    title_parts = []
    if sample_name:
        title_parts.append(f'Sample: {sample_name}')
    if label is not None:
        action_name = ACTION_LABELS[label] if label < len(ACTION_LABELS) else f"Action_{label}"
        title_parts.append(f'Action: {action_name}')
    title_parts.append(f'Shape: {data.shape}')
    
    if title_parts:
        fig.suptitle('\n'.join(title_parts), fontsize=14, fontweight='bold')
    
    # Plot each frame
    for idx, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        plot_skeleton_frame(ax, data, frame_idx, 
                          title=f'Frame {frame_idx}/{T-1}',
                          show_labels=(idx == 0))  # Show labels only on first frame
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def visualize_batch(data_batch, label_batch, sample_names=None, 
                    num_samples=4, save_dir=None, show_plot=False):
    """
    Visualize multiple samples from a batch
    
    Args:
        data_batch: torch.Tensor or numpy array of shape (B, C, T, V, M)
        label_batch: torch.Tensor or numpy array of shape (B,)
        sample_names: list of sample names (optional)
        num_samples: number of samples to visualize
        save_dir: directory to save figures (if None, doesn't save)
        show_plot: whether to display plots
        
    Returns:
        list of matplotlib figure objects
    """
    # Convert to numpy if needed
    if hasattr(data_batch, 'cpu'):
        data_batch = data_batch.cpu().numpy()
    if hasattr(label_batch, 'cpu'):
        label_batch = label_batch.cpu().numpy()
    
    figures = []
    num_samples = min(num_samples, len(data_batch))
    
    for i in range(num_samples):
        data = data_batch[i]
        label = label_batch[i]
        sample_name = sample_names[i] if sample_names else f"sample_{i}"
        
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f'{sample_name}_vis.png')
        
        fig = visualize_sample(data, label, sample_name, 
                             save_path=save_path, show_plot=show_plot)
        figures.append(fig)
        
        if not show_plot:
            plt.close(fig)
    
    return figures


def print_data_info(data, label=None, sample_name=None):
    """
    Print information about a data sample
    
    Args:
        data: numpy array or torch.Tensor of shape (C, T, V, M)
        label: action label (int)
        sample_name: name of the sample
    """
    # Convert to numpy if needed
    if hasattr(data, 'cpu'):
        data = data.cpu().numpy()
    
    print("\n" + "="*60)
    print("Data Information")
    print("="*60)
    
    if sample_name:
        print(f"Sample: {sample_name}")
    
    if label is not None:
        action_name = ACTION_LABELS[label] if label < len(ACTION_LABELS) else f"Action_{label}"
        print(f"Action: {action_name} (label={label})")
    
    print(f"\nShape: {data.shape}")
    print(f"  C={data.shape[0]} (coordinates: X, Y, Z)")
    print(f"  T={data.shape[1]} (temporal frames)")
    print(f"  V={data.shape[2]} (vertices/joints)")
    print(f"  M={data.shape[3]} (max people)")
    
    print(f"\nValue Statistics:")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std: {data.std():.6f}")
    
    print(f"\nPer Coordinate:")
    coord_names = ['X', 'Y', 'Z']
    for c in range(min(3, data.shape[0])):
        print(f"  {coord_names[c]}: [{data[c].min():.4f}, {data[c].max():.4f}], "
              f"mean={data[c].mean():.4f}, std={data[c].std():.4f}")
    
    # Check for zero frames
    frame_magnitudes = np.sqrt(np.sum(data**2, axis=(0, 2, 3)))
    zero_frames = np.sum(frame_magnitudes < 1e-6)
    valid_frames = data.shape[1] - zero_frames
    print(f"\nFrame Info:")
    print(f"  Total frames: {data.shape[1]}")
    print(f"  Valid frames: {valid_frames}")
    print(f"  Zero frames: {zero_frames}")
    
    print("="*60 + "\n")


def quick_visualize(data_loader, num_samples=1, save_dir='./visualizations/quick_check'):
    """
    Quick visualization of samples from a data loader
    Useful for debugging or checking data during training
    
    Args:
        data_loader: PyTorch DataLoader
        num_samples: number of samples to visualize
        save_dir: directory to save visualizations
        
    Example usage in main.py:
        from utils_visualization import quick_visualize
        quick_visualize(self.data_loader['train'], num_samples=2)
    """
    print(f"\nQuick visualization of {num_samples} samples...")
    
    # Get a batch
    data_iter = iter(data_loader)
    try:
        data_batch, label_batch, index_batch = next(data_iter)
    except StopIteration:
        print("Data loader is empty!")
        return
    
    # Get sample names if available
    sample_names = None
    if hasattr(data_loader.dataset, 'sample_name'):
        sample_names = [data_loader.dataset.sample_name[idx] for idx in index_batch]
    
    # Visualize
    visualize_batch(data_batch, label_batch, sample_names,
                   num_samples=num_samples, save_dir=save_dir, show_plot=False)
    
    # Print info for first sample
    if len(data_batch) > 0:
        data = data_batch[0]
        label = label_batch[0]
        sample_name = sample_names[0] if sample_names else None
        print_data_info(data, label, sample_name)
    
    print(f"Visualizations saved to {save_dir}/")


if __name__ == '__main__':
    """
    Example usage of visualization utilities
    """
    print("Visualization Utilities Demo")
    print("Import these functions in your training script:")
    print()
    print("from utils_visualization import visualize_sample, visualize_batch, quick_visualize")
    print()
    print("Examples:")
    print("1. Visualize a single sample:")
    print("   visualize_sample(data, label, 'sample.csv', save_path='output.png')")
    print()
    print("2. Visualize a batch:")
    print("   visualize_batch(data_batch, label_batch, num_samples=4, save_dir='./vis')")
    print()
    print("3. Quick check during training:")
    print("   quick_visualize(data_loader, num_samples=2)")
    print()
    print("For full demos, use:")
    print("  - visualize_demo.py (quick single sample)")
    print("  - visualize_mediapipe_data.py (comprehensive analysis)")


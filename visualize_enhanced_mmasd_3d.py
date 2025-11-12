import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the dataset class from Enhanced-MMASD
class CustomSkeletonDataset:
    def __init__(self, csv_file):
        """
        Load a single 3D skeleton CSV file
        """
        self.csv_file = csv_file
        
    def load_and_normalize(self):
        """
        Load and normalize data (similar to Enhanced-MMASD preprocessing)
        """
        df = pd.read_csv(self.csv_file)
        
        # Drop labels if present
        if 'Action_Label' in df.columns:
            df = df.drop(['Action_Label'], axis=1, errors='ignore')
        if 'ASD_Label' in df.columns:
            df = df.drop(['ASD_Label'], axis=1, errors='ignore')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
            
        # Min-max normalization: scale data to [0, 1] range
        # This matches the normalization in feeder_mediapipe.py when normalization=True
        df_min = df.min().min()
        df_max = df.max().max()
        normalized_data = (df - df_min) / (df_max - df_min)
        
        return normalized_data.values, df.values  # Return both normalized and original

def visualize_3d_skeleton_frame(skeleton_data, frame_idx=0, title="3D Skeleton - Single Frame"):
    """
    Visualize a single frame of 3D skeleton keypoints
    
    Args:
        skeleton_data: numpy array of shape (num_frames, 75)
        frame_idx: which frame to visualize
        title: plot title
    """
    # Joint names based on CSV header
    joint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
        'left_heel', 'right_heel', 'left_foot', 'right_foot'
    ]
    
    # Get single frame
    frame = skeleton_data[frame_idx]
    
    # Reshape to (25, 3) - 25 keypoints with x, y, z coordinates
    num_keypoints = len(frame) // 3
    keypoints_3d = frame.reshape(num_keypoints, 3)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot keypoints with different colors for different body parts
    xs, ys, zs = keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2]
    
    # Face points (0-4)
    ax.scatter(xs[0:5], ys[0:5], zs[0:5], c='yellow', marker='o', s=100, label='Face', alpha=0.8)
    # Upper body (5-14)
    ax.scatter(xs[5:15], ys[5:15], zs[5:15], c='red', marker='o', s=100, label='Upper Body', alpha=0.8)
    # Lower body (15-24)
    ax.scatter(xs[15:25], ys[15:25], zs[15:25], c='blue', marker='o', s=100, label='Lower Body', alpha=0.8)
    
    # Add keypoint labels
    for i, (x, y, z) in enumerate(keypoints_3d):
        ax.text(x, y, z, f'{i}', fontsize=7, fontweight='bold')
    
    # Define skeleton connections based on actual MediaPipe Pose structure
    connections = [
        # Face
        (0, 1), (0, 2),  # nose to eyes
        (1, 3), (2, 4),  # eyes to ears
        # Upper body
        (5, 6),  # shoulders
        (0, 5), (0, 6),  # nose to shoulders
        (5, 7), (7, 9),  # left arm: shoulder -> elbow -> wrist
        (6, 8), (8, 10),  # right arm: shoulder -> elbow -> wrist
        (9, 11), (9, 13),  # left wrist to pinky and index
        (10, 12), (10, 14),  # right wrist to pinky and index
        # Torso
        (5, 15), (6, 16),  # shoulders to hips
        (15, 16),  # hips
        # Legs
        (15, 17), (17, 19),  # left leg: hip -> knee -> ankle
        (16, 18), (18, 20),  # right leg: hip -> knee -> ankle
        # Feet
        (19, 21), (19, 23),  # left ankle to heel and foot
        (20, 22), (20, 24),  # right ankle to heel and foot
    ]
    
    # Draw connections
    for connection in connections:
        if connection[0] < num_keypoints and connection[1] < num_keypoints:
            points = keypoints_3d[[connection[0], connection[1]]]
            ax.plot3D(*points.T, 'b-', linewidth=2, alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'{title}\nFrame {frame_idx}', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig


def visualize_multiple_frames(skeleton_data, frame_indices=[0, 30, 60, 90]):
    """
    Visualize multiple frames in a grid
    """
    fig = plt.figure(figsize=(16, 12))
    
    for idx, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(skeleton_data):
            continue
            
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        
        # Get single frame
        frame = skeleton_data[frame_idx]
        num_keypoints = len(frame) // 3
        keypoints_3d = frame.reshape(num_keypoints, 3)
        
        # Plot keypoints with different colors
        xs, ys, zs = keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2]
        
        # Face points (0-4)
        ax.scatter(xs[0:5], ys[0:5], zs[0:5], c='yellow', marker='o', s=50, alpha=0.8)
        # Upper body (5-14)
        ax.scatter(xs[5:15], ys[5:15], zs[5:15], c='red', marker='o', s=50, alpha=0.8)
        # Lower body (15-24)
        ax.scatter(xs[15:25], ys[15:25], zs[15:25], c='blue', marker='o', s=50, alpha=0.8)
        
        # Draw skeleton connections based on actual joint structure
        connections = [
            # Face
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Upper body
            (5, 6), (0, 5), (0, 6),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (9, 11), (9, 13), (10, 12), (10, 14),
            # Torso
            (5, 15), (6, 16), (15, 16),
            # Legs
            (15, 17), (17, 19), (16, 18), (18, 20),
            # Feet
            (19, 21), (19, 23), (20, 22), (20, 24),
        ]
        
        for connection in connections:
            if connection[0] < num_keypoints and connection[1] < num_keypoints:
                points = keypoints_3d[[connection[0], connection[1]]]
                ax.plot3D(*points.T, 'b-', linewidth=1.5, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_idx}', fontweight='bold')
    
    plt.tight_layout()
    return fig


# Main usage
if __name__ == "__main__":
    # Check if a file path is provided as command-line argument
    if len(sys.argv) > 1:
        sample_csv = sys.argv[1]
        if not os.path.exists(sample_csv):
            print(f"Error: File not found: {sample_csv}")
            sys.exit(1)
    else:
        # Default path to a sample CSV file from Enhanced-MMASD
        sample_csv = "3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit/Arm_Swing/processed_Arm_Swing_P1_R1_0.csv"
        
        # Check if file exists
        if not os.path.exists(sample_csv):
            print(f"File not found: {sample_csv}")
            print("\nLooking for available CSV files...")
            
            # Try to find a sample file
            base_dir = "3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit"
            if os.path.exists(base_dir):
                for subdir in os.listdir(base_dir):
                    subdir_path = os.path.join(base_dir, subdir)
                    if os.path.isdir(subdir_path):
                        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
                        if csv_files:
                            sample_csv = os.path.join(subdir_path, csv_files[0])
                            print(f"Found sample file: {sample_csv}")
                            break
    
    if os.path.exists(sample_csv):
        print(f"\n{'='*60}")
        print(f"Loading: {sample_csv}")
        print(f"{'='*60}\n")
        
        # Joint names based on CSV header
        joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'left_heel', 'right_heel', 'left_foot', 'right_foot'
        ]
        
        # Load data using the dataloader approach (same as Enhanced-MMASD)
        dataset = CustomSkeletonDataset(sample_csv)
        normalized_data, original_data = dataset.load_and_normalize()
        
        print(f"Loaded skeleton data shape: {original_data.shape}")
        print(f"Number of frames: {original_data.shape[0]}")
        print(f"Number of features per frame: {original_data.shape[1]}")
        print(f"Number of keypoints: {original_data.shape[1] // 3}")
        print(f"\nData range - Min: {original_data.min():.4f}, Max: {original_data.max():.4f}")
        print(f"Normalized range - Min: {normalized_data.min():.4f}, Max: {normalized_data.max():.4f}")
        
        print(f"\n{'='*60}")
        print("Joint Structure (MediaPipe Pose):")
        print(f"{'='*60}")
        for i, name in enumerate(joint_names):
            print(f"  {i:2d}: {name}")
        print(f"{'='*60}")
        
        # Visualize a single frame (frame 0)
        print(f"\n{'='*60}")
        print("Creating single frame visualization...")
        print(f"{'='*60}")
        fig1 = visualize_3d_skeleton_frame(original_data, frame_idx=0, 
                                           title=f"3D Skeleton Visualization (Enhanced-MMASD Dataloader)\n{os.path.basename(sample_csv)}")
        plt.savefig('visualizations/enhanced_mmasd_single_frame.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: visualizations/enhanced_mmasd_single_frame.png")
        
        # Visualize multiple frames
        print(f"\n{'='*60}")
        print("Creating multiple frames visualization...")
        print(f"{'='*60}")
        fig2 = visualize_multiple_frames(original_data, frame_indices=[0, 30, 60, 90])
        plt.savefig('visualizations/enhanced_mmasd_multiple_frames.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: visualizations/enhanced_mmasd_multiple_frames.png")
        
        print(f"\n{'='*60}")
        print("Visualization complete!")
        print(f"{'='*60}\n")
        
        # Display the plots
        plt.show()
    else:
        print("❌ No CSV file found. Please check the data directory.")
        print("Expected directory: 3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit/")


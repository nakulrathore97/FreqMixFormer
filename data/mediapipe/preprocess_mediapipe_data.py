#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

# MediaPipe pose landmark indices (25 points)
# 0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows, 9-10: wrists
# 11-12: pinky, 13-14: index, 15-16: hips, 17-18: knees, 19-20: ankles
# 21-22: heels, 23-24: feet

def load_csv_file(csv_path):
    """
    Load a single CSV file and extract skeleton data
    Returns: numpy array of shape (T, 25, 3) where T is number of frames
    """
    df = pd.read_csv(csv_path)
    
    # Remove the label columns
    df_coords = df.drop(['Action_Label', 'ASD_Label'], axis=1, errors='ignore')
    
    # Reshape to (T, 25, 3) - T frames, 25 joints, 3 coordinates (x,y,z)
    num_frames = len(df_coords)
    num_coords = df_coords.shape[1]
    
    # Ensure we have 75 coordinates (25 joints * 3)
    assert num_coords == 75, f"Expected 75 coordinates, got {num_coords}"
    
    data = df_coords.values.reshape(num_frames, 25, 3)
    
    return data

def process_dataset(data_root, max_frames=300, train_ratio=0.8):
    """
    Process all CSV files in the dataset
    
    Args:
        data_root: Root directory containing action folders
        max_frames: Maximum number of frames to consider (for padding)
        train_ratio: Ratio of training samples
    
    Returns:
        Dictionary with train/test splits
    """
    
    # Get all action folders (exclude hidden/system folders)
    action_folders = sorted([d for d in os.listdir(data_root) 
                           if os.path.isdir(os.path.join(data_root, d)) 
                           and not d.startswith('.')])
    
    print(f"Found {len(action_folders)} action classes:")
    for i, action in enumerate(action_folders):
        print(f"  {i}: {action}")
    
    # Create label mapping
    label_map = {action: idx for idx, action in enumerate(action_folders)}
    
    all_samples = []
    all_labels = []
    all_names = []
    
    # Process each action folder
    for action_name in tqdm(action_folders, desc="Processing actions"):
        action_path = os.path.join(data_root, action_name)
        csv_files = glob.glob(os.path.join(action_path, "*.csv"))
        
        label = label_map[action_name]
        
        for csv_file in tqdm(csv_files, desc=f"  Loading {action_name}", leave=False):
            try:
                # Load skeleton data
                skeleton_data = load_csv_file(csv_file)
                
                # Get filename for sample name
                filename = os.path.basename(csv_file)
                
                all_samples.append(skeleton_data)
                all_labels.append(label)
                all_names.append(f"{action_name}_{filename}")
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
    
    print(f"\nTotal samples loaded: {len(all_samples)}")
    
    # Find max frames in dataset
    max_frames_in_data = max([s.shape[0] for s in all_samples])
    print(f"Max frames in dataset: {max_frames_in_data}")
    
    # Use the larger of the two for padding
    target_frames = max(max_frames, max_frames_in_data)
    print(f"Using target frames: {target_frames}")
    
    # Pad all samples to same length and create fixed array
    # Shape: (N, T, V, C, M) = (N_samples, T_frames, 25_joints, 3_coords, 1_person)
    N = len(all_samples)
    padded_data = np.zeros((N, target_frames, 25, 3, 1), dtype=np.float32)
    
    for i, sample in enumerate(tqdm(all_samples, desc="Padding samples")):
        T = sample.shape[0]
        # Only take up to target_frames
        T_actual = min(T, target_frames)
        padded_data[i, :T_actual, :, :, 0] = sample[:T_actual]
    
    # Reshape to NTU format: (N, T, V*C*M) then (N, C, T, V, M)
    # NTU format expects: (N, T, 2, 25, 3) but MediaPipe has only 1 person
    N, T, V, C, M = padded_data.shape
    
    # Reshape to (N, T, M, V, C) then (N, C, T, V, M)
    data_reshaped = padded_data.transpose(0, 1, 4, 2, 3)  # (N, T, M, V, C)
    
    # For compatibility with NTU feeder, reshape to (N, T, M*V*C)
    data_flat = data_reshaped.reshape(N, T, M * V * C)
    
    # Create labels array
    labels = np.array(all_labels)
    
    # Split into train/test
    indices = np.arange(N)
    train_idx, test_idx = train_test_split(
        indices, 
        train_size=train_ratio, 
        random_state=42,
        stratify=labels
    )
    
    print(f"\nTrain samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    # Create one-hot encoded labels
    num_classes = len(label_map)
    y_train = np.zeros((len(train_idx), num_classes))
    y_test = np.zeros((len(test_idx), num_classes))
    
    for i, idx in enumerate(train_idx):
        y_train[i, labels[idx]] = 1
    
    for i, idx in enumerate(test_idx):
        y_test[i, labels[idx]] = 1
    
    return {
        'x_train': data_flat[train_idx],
        'y_train': y_train,
        'x_test': data_flat[test_idx],
        'y_test': y_test,
        'label_map': label_map,
        'num_classes': num_classes
    }

def main():
    # Path to your data
    data_root = "/root/3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit"
    output_dir = "/root/FreqMixFormer/data/mediapipe"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing MediaPipe dataset...")
    dataset = process_dataset(data_root, max_frames=300, train_ratio=0.8)
    
    # Save as npz file
    output_path = os.path.join(output_dir, "mediapipe_data.npz")
    np.savez(output_path, **dataset)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"Number of classes: {dataset['num_classes']}")
    print(f"Training samples: {dataset['x_train'].shape}")
    print(f"Test samples: {dataset['x_test'].shape}")
    
    # Save label mapping
    label_map_path = os.path.join(output_dir, "label_map.txt")
    with open(label_map_path, 'w') as f:
        for action, label in sorted(dataset['label_map'].items(), key=lambda x: x[1]):
            f.write(f"{label}: {action}\n")
    
    print(f"Label mapping saved to: {label_map_path}")

if __name__ == "__main__":
    main()


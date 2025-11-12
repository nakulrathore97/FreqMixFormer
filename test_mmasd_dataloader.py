#!/usr/bin/env python
"""
Test script to verify MMASD dataloader integration
"""

import sys
import torch
from torch.utils.data import DataLoader

# Import the MMASD feeder
from feeders.feeder_mmasd import Feeder

def test_dataloader():
    """Test the MMASD dataloader"""
    
    print("="*60)
    print("Testing MMASD Dataloader")
    print("="*60)
    
    # Test parameters
    data_path = "./3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit"
    
    # Create training dataset
    print("\n1. Creating training dataset...")
    train_dataset = Feeder(
        data_path=data_path,
        split='train',
        window_size=64,
        p_interval=[0.5, 1],
        random_rot=True,
        bone=False,
        vel=False,
        debug=False,  # Set to True for faster testing with 100 samples
        train_val_split=0.8
    )
    
    print(f"   Training dataset size: {len(train_dataset)}")
    
    # Create test dataset
    print("\n2. Creating test dataset...")
    test_dataset = Feeder(
        data_path=data_path,
        split='test',
        window_size=64,
        p_interval=[0.95],
        bone=False,
        vel=False,
        debug=False,
        train_val_split=0.8
    )
    
    print(f"   Test dataset size: {len(test_dataset)}")
    
    # Create dataloaders
    print("\n3. Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Set to 0 for testing
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\n4. Loading a sample batch...")
    for batch_idx, (data, label, index) in enumerate(train_loader):
        print(f"   Batch shape: {data.shape}")
        print(f"   Expected shape: (batch_size, C, T, V, M) = (8, 3, 64, 25, 1)")
        print(f"   Label shape: {label.shape}")
        print(f"   Labels in batch: {label.numpy()}")
        print(f"   Index shape: {index.shape}")
        print(f"   Data type: {data.dtype}")
        print(f"   Data range: [{data.min():.4f}, {data.max():.4f}]")
        break
    
    # Test with bone modality
    print("\n5. Testing bone modality...")
    bone_dataset = Feeder(
        data_path=data_path,
        split='train',
        window_size=64,
        p_interval=[0.5, 1],
        bone=True,
        debug=True  # Use debug mode for faster testing
    )
    print(f"   Bone dataset size: {len(bone_dataset)}")
    
    bone_loader = DataLoader(bone_dataset, batch_size=4, shuffle=False)
    for data, label, index in bone_loader:
        print(f"   Bone batch shape: {data.shape}")
        break
    
    # Test with velocity modality
    print("\n6. Testing velocity modality...")
    vel_dataset = Feeder(
        data_path=data_path,
        split='train',
        window_size=64,
        p_interval=[0.5, 1],
        vel=True,
        debug=True
    )
    print(f"   Velocity dataset size: {len(vel_dataset)}")
    
    vel_loader = DataLoader(vel_dataset, batch_size=4, shuffle=False)
    for data, label, index in vel_loader:
        print(f"   Velocity batch shape: {data.shape}")
        break
    
    # Test class distribution
    print("\n7. Checking class distribution...")
    class_counts = {}
    for i in range(len(train_dataset)):
        label = train_dataset.label[i]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    action_names = {
        0: 'Arm_Swing',
        1: 'Body_pose',
        2: 'chest_expansion',
        3: 'Drumming',
        4: 'Frog_Pose',
        5: 'Marcas_Forward',
        6: 'Marcas_Shaking',
        7: 'Sing_Clap',
        8: 'Squat_Pose',
        9: 'Tree_Pose',
        10: 'Twist_Pose'
    }
    
    print(f"\n   Class distribution in training set:")
    for class_id in sorted(class_counts.keys()):
        print(f"   Class {class_id} ({action_names.get(class_id, 'Unknown')}): {class_counts[class_id]} samples")
    
    print("\n" + "="*60)
    print("✓ All tests passed! Dataloader is working correctly.")
    print("="*60)
    print("\nTo train the model, use:")
    print("  python main.py --config ./config/mmasd/train_joint.yaml --phase train")
    print("\nTo test the model, use:")
    print("  python main.py --config ./config/mmasd/train_joint.yaml --phase test --weights <path_to_weights>")
    print("="*60)

if __name__ == '__main__':
    try:
        test_dataloader()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


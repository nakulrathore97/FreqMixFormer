#!/usr/bin/env python
"""
Example: How to integrate visualization into main.py training loop

This demonstrates how to visualize data samples during training or testing
by adding a few lines to main.py
"""

import sys
import os
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feeders.feeder_mmasd import Feeder
from utils_visualization import quick_visualize, visualize_sample, print_data_info


def example_1_quick_check():
    """
    Example 1: Quick check of data loader
    This is useful for debugging data loading issues
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Check of Data Loader")
    print("="*70)
    
    # Load config (same as main.py does)
    config_path = './config/mmasd/train_joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create feeder (same as main.py)
    train_feeder_args = config['train_feeder_args']
    train_feeder_args['debug'] = True  # Use debug mode for quick demo
    
    feeder = Feeder(**train_feeder_args)
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    
    # Quick visualization - just add this line in main.py!
    quick_visualize(data_loader, num_samples=2, 
                   save_dir='./visualizations/training_check')
    
    print("\n✓ Visualizations saved to ./visualizations/training_check/")


def example_2_visualize_specific_sample():
    """
    Example 2: Visualize a specific sample
    This is useful when you want to examine particular samples
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Visualize Specific Sample")
    print("="*70)
    
    # Load config
    config_path = './config/mmasd/train_joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create feeder
    test_feeder_args = config['test_feeder_args']
    test_feeder_args['debug'] = True
    feeder = Feeder(**test_feeder_args)
    
    # Get a specific sample (e.g., index 5)
    data, label, idx = feeder[5]
    sample_name = feeder.sample_name[5]
    
    # Print detailed information
    print_data_info(data, label, sample_name)
    
    # Visualize it
    visualize_sample(data, label, sample_name, num_frames=6,
                    save_path='./visualizations/specific_sample.png')
    
    print("\n✓ Visualization saved to ./visualizations/specific_sample.png")


def example_3_visualize_during_training():
    """
    Example 3: How to add visualization to training loop in main.py
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Visualization During Training (code snippet)")
    print("="*70)
    
    code_snippet = '''
# In main.py, import the utility at the top:
from utils_visualization import quick_visualize, visualize_batch

class Processor():
    def __init__(self, arg):
        # ... existing initialization code ...
        
        # OPTIONAL: Visualize some training samples after loading data
        if arg.phase == 'train' and not arg.train_feeder_args['debug']:
            print("\\nVisualizing training data samples...")
            quick_visualize(self.data_loader['train'], 
                          num_samples=3,
                          save_dir=os.path.join(arg.work_dir, 'data_viz'))
    
    def train(self, epoch, save_model=False):
        # ... existing training code ...
        
        # OPTIONAL: Visualize a batch at the start of first epoch
        if epoch == 0:
            loader = self.data_loader['train']
            data_iter = iter(loader)
            data_batch, label_batch, index_batch = next(data_iter)
            
            # Get sample names
            sample_names = [loader.dataset.sample_name[idx] 
                          for idx in index_batch[:3]]
            
            # Visualize
            visualize_batch(data_batch[:3], label_batch[:3], sample_names,
                          save_dir=os.path.join(self.arg.work_dir, 'epoch0_viz'))
            
            print(f"First batch visualization saved!")
        
        # ... rest of training loop ...
    
    def eval(self, epoch, save_score=False, loader_name=['test'], ...):
        # ... existing eval code ...
        
        # OPTIONAL: Visualize misclassified samples
        if wrong_file is not None:
            # After evaluation, visualize samples that were misclassified
            # This helps understand what the model is getting wrong
            pass
'''
    
    print(code_snippet)
    print("\n✓ Copy and adapt this code to main.py")


def example_4_compare_modalities():
    """
    Example 4: Compare joint vs bone vs velocity modalities
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Compare Different Modalities")
    print("="*70)
    
    # Load config
    config_path = './config/mmasd/train_joint.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    train_args = config['train_feeder_args']
    train_args['debug'] = True
    
    # Create feeders for each modality
    print("\nLoading joint modality...")
    feeder_joint = Feeder(**{**train_args, 'bone': False, 'vel': False})
    
    print("Loading bone modality...")
    feeder_bone = Feeder(**{**train_args, 'bone': True, 'vel': False})
    
    print("Loading velocity modality...")
    feeder_vel = Feeder(**{**train_args, 'bone': False, 'vel': True})
    
    # Get same sample from each
    idx = 0
    data_joint, label, _ = feeder_joint[idx]
    data_bone, _, _ = feeder_bone[idx]
    data_vel, _, _ = feeder_vel[idx]
    sample_name = feeder_joint.sample_name[idx]
    
    # Visualize all three
    print(f"\nVisualizing sample: {sample_name}")
    
    visualize_sample(data_joint, label, f"{sample_name} (Joint)", 
                    save_path='./visualizations/modality_joint.png')
    
    visualize_sample(data_bone, label, f"{sample_name} (Bone)", 
                    save_path='./visualizations/modality_bone.png')
    
    visualize_sample(data_vel, label, f"{sample_name} (Velocity)", 
                    save_path='./visualizations/modality_velocity.png')
    
    print("\n✓ Modality comparisons saved to ./visualizations/modality_*.png")
    print("\nNote the differences:")
    print("  - Joint: Shows absolute positions")
    print("  - Bone: Shows relative positions (joint-to-joint vectors)")
    print("  - Velocity: Shows temporal changes (frame-to-frame differences)")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("MEDIAPIPE DATA VISUALIZATION EXAMPLES")
    print("="*70)
    print("\nThese examples show how to integrate visualization into main.py")
    print("or use visualization utilities for debugging and analysis.")
    
    try:
        # Example 1: Quick check
        example_1_quick_check()
        
        # Example 2: Specific sample
        example_2_visualize_specific_sample()
        
        # Example 3: Training integration (just shows code)
        example_3_visualize_during_training()
        
        # Example 4: Compare modalities
        example_4_compare_modalities()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nCheck the ./visualizations/ directory for outputs.")
        print("\nFor more options, see:")
        print("  - VISUALIZATION_README.md")
        print("  - visualize_mediapipe_data.py --help")
        print("  - utils_visualization.py")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


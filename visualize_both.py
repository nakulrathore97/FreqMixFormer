#!/usr/bin/env python
"""visualize_both.py - Visualize the same sample with both methods"""

import sys
import os
import glob
import re
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*60)
print("Synchronized Visualization Script")
print("Both scripts now use (T, 75) format WITHOUT temporal preprocessing")
print("="*60)

# Find the first CSV file (sorted order)
data_path = './3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit'

csv_files = []
action_folders = sorted([d for d in os.listdir(data_path) 
                        if os.path.isdir(os.path.join(data_path, d))])

for action_folder in action_folders:
    if action_folder not in ['Arm_Swing', 'Body_pose', 'chest_expansion', 'Drumming', 
                             'Frog_Pose', 'Marcas_Forward', 'Marcas_Shaking', 
                             'Sing_Clap', 'Squat_Pose', 'Tree_Pose', 'Twist_Pose']:
        continue
    
    action_path = os.path.join(data_path, action_folder)
    files = sorted(glob.glob(os.path.join(action_path, '*.csv')))
    csv_files.extend(files)

if not csv_files:
    print("Error: No CSV files found!")
    sys.exit(1)

# Use first file
sample_path = csv_files[0]
sample_name = os.path.basename(sample_path)

print(f"\n{'='*60}")
print(f"Both scripts will visualize: {sample_path}")
print(f"Sample name: {sample_name}")
print(f"{'='*60}\n")

# Update visualize_demo.py to use this specific file
with open('visualize_demo.py', 'r') as f:
    demo_content = f.read()

# Find and replace the sample selection to use our specific file
demo_modified = re.sub(
    r'# Use first file\s+sample_csv, action_folder = csv_files\[0\]',
    f'# Use specific file\n    sample_csv = "{sample_path}"\n    action_folder = os.path.basename(os.path.dirname(sample_csv))',
    demo_content
)

with open('visualize_demo_temp.py', 'w') as f:
    f.write(demo_modified)

# Update visualize_enhanced_mmasd_3d.py to use the same file
with open('visualize_enhanced_mmasd_3d.py', 'r') as f:
    enhanced_content = f.read()

enhanced_modified = re.sub(
    r'sample_csv = ".*?"',
    f'sample_csv = "{sample_path}"',
    enhanced_content
)

with open('visualize_enhanced_mmasd_3d_temp.py', 'w') as f:
    f.write(enhanced_modified)

print("\n" + "="*60)
print("1. Running visualize_demo.py (raw T, 75 format)...")
print("="*60 + "\n")
result1 = subprocess.run([sys.executable, 'visualize_demo_temp.py'])

print("\n" + "="*60)
print("2. Running visualize_enhanced_mmasd_3d.py (raw T, 75 format)...")
print("="*60 + "\n")
result2 = subprocess.run([sys.executable, 'visualize_enhanced_mmasd_3d_temp.py'])

# Cleanup
os.remove('visualize_demo_temp.py')
os.remove('visualize_enhanced_mmasd_3d_temp.py')

print("\n" + "="*60)
print("Comparison Complete!")
print("="*60)
print("\nOutput files:")
print("  - visualizations/demo_output.png (raw T×75 data)")
print("  - visualizations/enhanced_mmasd_single_frame.png (raw T×75 data)")
print("  - visualizations/enhanced_mmasd_multiple_frames.png (raw T×75 data)")
print("\nBoth visualizations now show the SAME preprocessing:")
print("  ✓ Same (T, 75) format")
print("  ✓ Same min-max normalization")
print("  ✓ NO temporal resizing")
print("  ✓ SAME sample file")
print("="*60 + "\n")


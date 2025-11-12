# MediaPipe Data Visualization

This document explains how to visualize the MediaPipe skeleton data that is passed to the model in `main.py`.

## Overview

The visualization tools help you understand:
- How the MediaPipe skeleton data is structured (C, T, V, M format)
- What the data looks like after preprocessing by `feeder_mmasd.py`
- How joints are connected in the MediaPipe graph
- How the skeleton moves through time (temporal information)

## Data Format

The data fed to the model has the shape: **(C, T, V, M)**

- **C = 3**: Coordinates (X, Y, Z)
- **T = 64**: Temporal frames (configurable via `window_size`)
- **V = 25**: Vertices/Joints (MediaPipe pose landmarks)
- **M = 1**: Maximum number of people

### MediaPipe Joints (25 landmarks)

```
0:  nose              9:  left_wrist       18: right_knee
1:  left_eye         10:  right_wrist      19: left_ankle
2:  right_eye        11:  left_pinky       20: right_ankle
3:  left_ear         12:  right_pinky      21: left_heel
4:  right_ear        13:  left_index       22: right_heel
5:  left_shoulder    14:  right_index      23: left_foot
6:  right_shoulder   15:  left_hip         24: right_foot
7:  left_elbow       16:  right_hip
8:  right_elbow      17:  left_knee
```

### Actions (11 classes)

```
0: Arm_Swing          5: Marcas_Forward      9:  Tree_Pose
1: Body_pose          6: Marcas_Shaking      10: Twist_Pose
2: chest_expansion    7: Sing_Clap
3: Drumming           8: Squat_Pose
4: Frog_Pose
```

## Visualization Scripts

### 1. Quick Demo (`visualize_demo.py`)

A simple script to quickly visualize a single sample.

**Usage:**
```bash
python visualize_demo.py
```

**Features:**
- Loads a single sample using debug mode (fast)
- Shows 6 evenly-spaced frames
- Displays skeleton with bone connections in 3D
- Saves output to `./visualizations/demo_output.png`

**Output:**
- Simple visualization showing how the skeleton looks across different frames

---

### 2. Comprehensive Visualization (`visualize_mediapipe_data.py`)

A full-featured script with multiple visualization modes.

**Basic Usage:**
```bash
python visualize_mediapipe_data.py --num-samples 4
```

**Advanced Usage:**
```bash
# Visualize test split with bone modality
python visualize_mediapipe_data.py --split test --bone --num-samples 2

# Create animated GIF
python visualize_mediapipe_data.py --num-samples 1 --create-animation

# Visualize with velocity modality
python visualize_mediapipe_data.py --vel --num-samples 3

# Custom data path and save directory
python visualize_mediapipe_data.py \
    --data-path ./path/to/data \
    --save-dir ./my_visualizations \
    --num-samples 5
```

**Arguments:**
- `--data-path`: Path to MMASD dataset (default: `./3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit`)
- `--split`: Dataset split - `train` or `test` (default: `train`)
- `--window-size`: Temporal window size (default: `64`)
- `--num-samples`: Number of samples to visualize (default: `4`)
- `--save-dir`: Output directory (default: `./visualizations/mediapipe_data`)
- `--create-animation`: Create animated GIF of first sample
- `--bone`: Use bone modality (velocity between joints)
- `--vel`: Use velocity modality (temporal differences)
- `--batch-size`: Data loader batch size (default: `8`)

**Output Files:**

For each sample, the script generates two files:

1. **`sample_N_frames.png`**: Shows 6 frames of the skeleton moving
   - 3D skeleton visualization with bone connections
   - Evenly spaced frames across the temporal dimension
   - Joint indices labeled for key points

2. **`sample_N_structure.png`**: Data structure analysis with 6 subplots:
   - **Joint trajectories**: X-coordinates over time for first 10 joints
   - **Coordinate distribution**: Histogram of X, Y, Z values
   - **Movement per joint**: Variance showing which joints move most
   - **Activity heatmap**: Joint activity magnitude over time
   - **Data statistics**: Min/max/mean/std and shape information
   - **Graph structure**: MediaPipe skeleton graph details

3. **`sample_animation.gif`** (optional): Animated skeleton movement

---

## Understanding the Visualizations

### Frame Visualization
- **Red dots**: Individual joints (25 landmarks)
- **Blue lines**: Bone connections between joints
- **Green labels**: Joint indices for key points

### Data Structure Analysis

1. **Trajectories**: Shows how joints move over time
   - Smooth curves indicate stable tracking
   - Sudden jumps may indicate occlusions or errors

2. **Distribution**: Shows data normalization
   - Values should be in [0, 1] range after min-max normalization
   - All coordinates should have similar ranges

3. **Movement**: Indicates which body parts are most active
   - Higher variance = more movement
   - Useful for understanding action characteristics

4. **Heatmap**: Shows temporal patterns
   - Bright areas = high activity
   - Can reveal periodic or sustained movements

### Bone vs Joint Modality

**Joint modality (default):**
- Shows absolute positions of joints
- Good for pose-based actions (e.g., Tree_Pose, Frog_Pose)

**Bone modality (`--bone`):**
- Shows relative positions (differences between connected joints)
- Good for actions involving joint relationships
- More robust to global position shifts

**Velocity modality (`--vel`):**
- Shows temporal differences (frame-to-frame changes)
- Good for dynamic actions (e.g., Drumming, Arm_Swing)
- Emphasizes motion over pose

---

## Data Pipeline

The visualization shows data **after** all preprocessing steps:

```
Raw CSV → Load & Normalize → Reshape → Crop/Resize → Augmentation → Model
  (T,75)    (T,25,3)         (3,T,25,1)   (3,64,25,1)   (C,T,V,M)
```

**Preprocessing steps (from `feeder_mmasd.py`):**

1. **Load CSV**: Read MediaPipe landmarks (T frames × 75 values)
2. **Min-Max Normalize**: Scale to [0, 1] per sample
3. **Reshape**: Convert to (T, 25, 3) format
4. **Transpose**: Convert to (C, T, V, M) = (3, T, 25, 1)
5. **Crop/Resize**: Adjust to window_size frames (default 64)
6. **Augmentation** (training only):
   - Random rotation (`random_rot`)
   - Random shift (`random_shift`)
   - Random move (`random_move`)
7. **Modality Transform** (optional):
   - Bone: Compute joint-to-joint vectors
   - Velocity: Compute frame-to-frame differences

---

## Verifying Data Quality

Use visualizations to check:

✅ **Skeleton structure**: All 25 joints should be visible and properly connected

✅ **Movement**: Skeleton should move smoothly across frames

✅ **Value range**: All coordinates should be in [0, 1] after normalization

✅ **Temporal consistency**: No sudden jumps or discontinuities

✅ **Action clarity**: Motion should match the action label

❌ **Warning signs**:
- All joints at same position (tracking failure)
- Extreme values outside [0, 1]
- Disconnected or inverted skeleton
- No movement for dynamic actions

---

## Integration with Training

The visualizations show **exactly** what the model sees during training/testing.

**From `main.py`:**
```python
# Data loader (line 242-248)
data_loader['train'] = torch.utils.data.DataLoader(
    dataset=Feeder(**arg.train_feeder_args),  # Uses feeder_mmasd.py
    batch_size=arg.batch_size,
    shuffle=True,
    num_workers=arg.num_worker,
)

# Training loop (line 373-381)
for batch_idx, (data, label, index) in enumerate(process):
    data = data.float().cuda(self.output_device)  # Shape: (B, C, T, V, M)
    output = self.model(data)  # Forward pass
```

**From `config/mmasd/train_joint.yaml`:**
- `window_size: 64` → T dimension
- `num_point: 25` → V dimension  
- `num_person: 1` → M dimension
- Graph: `graph.mediapipe.Graph` → Defines bone connections

---

## Examples

### Example 1: Visualize training data
```bash
python visualize_mediapipe_data.py --split train --num-samples 3
```

### Example 2: Visualize test data with bone modality
```bash
python visualize_mediapipe_data.py --split test --bone --num-samples 2
```

### Example 3: Quick check of a single sample
```bash
python visualize_demo.py
```

### Example 4: Create animation for presentation
```bash
python visualize_mediapipe_data.py --num-samples 1 --create-animation
```

### Example 5: Analyze specific action
```bash
# First, find samples of a specific action by checking the output
python visualize_mediapipe_data.py --num-samples 10 --split train
# Look for samples with desired action label
```

---

## Troubleshooting

**Issue**: Script runs slowly
- **Solution**: Reduce `--num-samples` or use `visualize_demo.py`

**Issue**: Out of memory
- **Solution**: Reduce `--batch-size` or `--window-size`

**Issue**: Skeleton looks incorrect
- **Solution**: Check that CSV files match MediaPipe format (75 columns)

**Issue**: Animation creation fails
- **Solution**: Install `pillow`: `pip install pillow`

**Issue**: No bone connections visible
- **Solution**: Check that `MEDIAPIPE_BONES` matches joint indices in data

---

## File Locations

- **Visualization scripts**: 
  - `visualize_mediapipe_data.py` (comprehensive)
  - `visualize_demo.py` (quick demo)
  
- **Output directory**: 
  - `./visualizations/mediapipe_data/` (default)
  - `./visualizations/demo_output.png` (demo)

- **Data feeder**: 
  - `feeders/feeder_mmasd.py`

- **Graph definition**: 
  - `graph/mediapipe.py`

- **Training config**: 
  - `config/mmasd/train_joint.yaml`

---

## Additional Resources

- **Data transformation**: See `DATA_TRANSFORMATION_COMPARISON.md`
- **MMASD integration**: See `MMASD_INTEGRATION_SUMMARY.md`
- **Dataloader details**: See `MMASD_DATALOADER_README.md`
- **Main training**: See `main.py`

---

## Citation

If you use these visualization tools, please cite:

```
MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose
MMASD Dataset: https://github.com/username/Enhanced-MMASD
FreqMixFormer: [Your paper citation]
```


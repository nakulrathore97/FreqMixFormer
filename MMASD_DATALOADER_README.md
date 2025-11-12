# MMASD Dataloader Integration

This document explains how to use the MMASD (Enhanced-MMASD) dataloader with the FreqMixFormer model.

## Overview

The MMASD dataloader (`feeders/feeder_mmasd.py`) is designed to load 3D skeleton data from the Enhanced-MMASD dataset, which contains MediaPipe Pose landmarks (25 joints) extracted from videos of 11 different actions.

## Dataset Structure

The dataloader expects the following directory structure:

```
3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit/
├── Arm_Swing/
│   ├── processed_Arm_Swing_P1_R1_0.csv
│   ├── processed_Arm_Swing_P1_R2_0.csv
│   └── ...
├── Body_pose/
│   └── ...
├── chest_expansion/
│   └── ...
├── Drumming/
│   └── ...
├── Frog_Pose/
│   └── ...
├── Marcas_Forward/
│   └── ...
├── Marcas_Shaking/
│   └── ...
├── Sing_Clap/
│   └── ...
├── Squat_Pose/
│   └── ...
├── Tree_Pose/
│   └── ...
└── Twist_Pose/
    └── ...
```

Each CSV file contains frames (rows) with 75 columns representing 25 joints × 3 coordinates (x, y, z).

## Action Labels

The dataloader maps the following actions to numeric labels:

| Action Name        | Label |
|-------------------|-------|
| Arm_Swing         | 0     |
| Body_pose         | 1     |
| chest_expansion   | 2     |
| Drumming          | 3     |
| Frog_Pose         | 4     |
| Marcas_Forward    | 5     |
| Marcas_Shaking    | 6     |
| Sing_Clap         | 7     |
| Squat_Pose        | 8     |
| Tree_Pose         | 9     |
| Twist_Pose        | 10    |

## Data Format

The dataloader converts CSV data from shape `(T, 75)` to `(C, T, V, M)` format:
- **C**: 3 (x, y, z coordinates)
- **T**: Number of frames (temporal dimension)
- **V**: 25 (number of joints/vertices - MediaPipe Pose landmarks)
- **M**: 1 (number of persons)

## MediaPipe Pose Joint Structure

The 25 joints from MediaPipe Pose are:

```
 0: nose             11: right_pinky      21: left_heel
 1: left_eye         12: left_index       22: right_heel
 2: right_eye        13: right_index      23: left_foot
 3: left_ear         14: left_hip         24: right_foot
 4: right_ear        15: right_hip
 5: left_shoulder    16: left_knee
 6: right_shoulder   17: right_knee
 7: left_elbow       18: left_ankle
 8: right_elbow      19: right_ankle
 9: left_wrist       20: left_heel
10: right_wrist
```

## Configuration Files

Three configuration files are provided:

### 1. Joint Modality
**File**: `config/mmasd/train_joint.yaml`

Trains on raw joint positions.

### 2. Bone Modality
**File**: `config/mmasd/train_bone.yaml`

Trains on bone vectors (differences between connected joints).

## Testing the Dataloader

Before training, verify the dataloader works correctly:

```bash
python test_mmasd_dataloader.py
```

This will:
1. Load training and test datasets
2. Create dataloaders
3. Load sample batches
4. Test bone and velocity modalities
5. Show class distribution

## Training the Model

### Train with Joint Modality

```bash
python main.py --config ./config/mmasd/train_joint.yaml --phase train
```

### Train with Bone Modality

```bash
python main.py --config ./config/mmasd/train_bone.yaml --phase train
```

## Testing the Model

```bash
python main.py --config ./config/mmasd/train_joint.yaml --phase test --weights <path_to_checkpoint>
```

Example:
```bash
python main.py --config ./config/mmasd/train_joint.yaml --phase test --weights ./work_dir/mmasd/skefreqmixformer_joint/runs-65-12345.pt
```

## Key Parameters

### Feeder Parameters

- `data_path`: Path to the MMASD dataset directory
- `split`: 'train' or 'test'
- `window_size`: Number of frames to use (default: 64)
- `p_interval`: Probability interval for frame sampling
- `random_rot`: Apply random rotation augmentation
- `bone`: Use bone modality (joint differences)
- `vel`: Use velocity modality (temporal differences)
- `train_val_split`: Ratio for train/test split (default: 0.8)
- `debug`: Use only 100 samples for quick testing

### Model Parameters

- `num_class`: 11 (number of action classes in MMASD)
- `num_point`: 25 (number of MediaPipe joints)
- `num_person`: 1 (single person per video)
- `num_frames`: 64 (temporal window size)

### Training Parameters

- `batch_size`: 64 (adjust based on GPU memory)
- `base_lr`: 0.1 (initial learning rate)
- `num_epoch`: 65 (total training epochs)
- `warm_up_epoch`: 5 (learning rate warm-up)
- `weight_decay`: 0.0004

## Train/Test Split

The dataloader automatically splits the data with an 80/20 train/test ratio by default. The split is:
- **Deterministic**: Uses `np.random.seed(42)` for reproducibility
- **Random**: Shuffles all samples before splitting
- **Customizable**: Adjust `train_val_split` parameter (e.g., 0.8 = 80% train, 20% test)

## Data Augmentation

The following augmentation techniques are supported:

- **Random Rotation**: Randomly rotates skeleton around XYZ axes
- **Random Shift**: Randomly pads zeros at sequence beginning/end
- **Random Move**: Randomly translates the skeleton
- **Random Choose**: Randomly selects a portion of the sequence

## Bone Modality Details

When `bone=True`, the dataloader computes bone vectors as differences between connected joints:

```python
bone_vector = joint_position[child] - joint_position[parent]
```

MediaPipe bone pairs are defined in `feeders/bone_pairs.py` as `mediapipe_pairs`.

## Integration with visualize_enhanced_mmasd_3d.py

The dataloader uses the same CSV loading approach as `visualize_enhanced_mmasd_3d.py`:

```python
# From visualize_enhanced_mmasd_3d.py
df = pd.read_csv(csv_file)
df = df.drop(['Action_Label', 'ASD_Label'], axis=1, errors='ignore')
data_array = df.values  # (T, 75)
```

This ensures consistency between visualization and training.

## Troubleshooting

### Issue: "File not found" error

**Solution**: Ensure the dataset path is correct:
```bash
ls "3_75 ELEMENTS LABLES_MEDIAPIPE_Final_to_Submit/"
```

### Issue: Out of memory during training

**Solution**: Reduce batch size in config file:
```yaml
batch_size: 32  # or 16
test_batch_size: 32  # or 16
```

### Issue: Training is too slow

**Solution**: Increase `num_worker` in dataloader:
```yaml
num_worker: 4  # or 8, depending on your CPU
```

### Issue: Poor accuracy

**Solution**: Try these approaches:
1. Train for more epochs
2. Use data augmentation (random_rot, random_shift)
3. Ensemble joint and bone modalities
4. Adjust learning rate or weight decay

## Ensemble Multiple Modalities

To improve performance, train multiple models and ensemble their predictions:

1. Train joint model:
   ```bash
   python main.py --config ./config/mmasd/train_joint.yaml --phase train
   ```

2. Train bone model:
   ```bash
   python main.py --config ./config/mmasd/train_bone.yaml --phase train
   ```

3. Ensemble predictions (use existing `ensemble.py` script)

## References

- **Enhanced-MMASD**: Original multimodal autism spectrum disorder dataset
- **MediaPipe Pose**: Google's pose estimation solution (33 landmarks, we use 25)
- **FreqMixFormer**: Frequency-domain mixing transformer for skeleton-based action recognition

## Contact

For issues or questions about the MMASD dataloader integration, please refer to the main README or create an issue.


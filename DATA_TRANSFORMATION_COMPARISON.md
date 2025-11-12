# Data Transformation Comparison

## Verification: MMASD Feeder vs visualize_enhanced_mmasd_3d.py

This document verifies that the data transformations in `feeders/feeder_mmasd.py` match those in `visualize_enhanced_mmasd_3d.py` and the original Enhanced-MMASD code.

## ✅ Transformation Matching Verified

### 1. CSV Loading

**visualize_enhanced_mmasd_3d.py (lines 19-27):**
```python
df = pd.read_csv(self.csv_file)

# Drop labels if present
if 'Action_Label' in df.columns:
    df = df.drop(['Action_Label'], axis=1, errors='ignore')
if 'ASD_Label' in df.columns:
    df = df.drop(['ASD_Label'], axis=1, errors='ignore')
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
```

**feeders/feeder_mmasd.py (lines 137-145):**
```python
df = pd.read_csv(csv_file)

# Drop label columns if present
if 'Action_Label' in df.columns:
    df = df.drop(['Action_Label'], axis=1, errors='ignore')
if 'ASD_Label' in df.columns:
    df = df.drop(['ASD_Label'], axis=1, errors='ignore')
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
```

**Status:** ✅ **IDENTICAL**

---

### 2. Min-Max Normalization (CRITICAL)

**visualize_enhanced_mmasd_3d.py (lines 29-33):**
```python
# Min-max normalization: scale data to [0, 1] range
df_min = df.min().min()
df_max = df.max().max()
normalized_data = (df - df_min) / (df_max - df_min)

return normalized_data.values, df.values
```

**Enhanced-MMASD/Code/proposed_model.py (lines 119-122):**
```python
df_min = df.min().min()
df_max = df.max().max()

normalized_data = (df - df_min)/(df_max - df_min)

data_array = normalized_data.values
```

**feeders/feeder_mmasd.py (lines 147-156):**
```python
# Apply min-max normalization per sample (same as Enhanced-MMASD)
# This matches the normalization in:
# - Enhanced-MMASD/Code/proposed_model.py line 119-122
# - visualize_enhanced_mmasd_3d.py line 29-33
df_min = df.min().min()
df_max = df.max().max()
normalized_df = (df - df_min) / (df_max - df_min)

# Get data as numpy array
data_array = normalized_df.values  # (T, 75)
```

**Status:** ✅ **IDENTICAL** (Fixed on Nov 12, 2025)

**Test Results:**
```
Original data range: [-5.8104, 2.3467]
After normalization: [0.0000, 1.0000]
✓ Data properly normalized to [0, 1] range
```

---

### 3. Data Reshaping

**visualize_enhanced_mmasd_3d.py:**
```python
# Data is kept as (T, 75) for visualization
# Reshaped to (T, 25, 3) when visualizing:
frame = skeleton_data[frame_idx]
keypoints_3d = frame.reshape(num_keypoints, 3)
```

**feeders/feeder_mmasd.py (lines 157-167):**
```python
# Get data as numpy array: shape (T, 75)
data_array = normalized_df.values  # (T, 75)
T = data_array.shape[0]

# Reshape to (T, 25, 3)
data_reshaped = data_array.reshape(T, 25, 3)

# Convert to (C, T, V, M) format for model
data_ctvm = data_reshaped.transpose(2, 0, 1)  # (3, T, 25)
data_ctvm = np.expand_dims(data_ctvm, axis=-1)  # (3, T, 25, 1)

return data_ctvm.astype(np.float32)
```

**Status:** ✅ **COMPATIBLE**
- visualize script keeps data as (T, 75) or (T, 25, 3)
- feeder converts to (C, T, V, M) = (3, T, 25, 1) for model input
- Both use the same underlying normalized data

---

## Transformation Pipeline Comparison

### visualize_enhanced_mmasd_3d.py Pipeline:

```
CSV File
    ↓
Load with pandas
    ↓
Drop label columns ['Action_Label', 'ASD_Label', 'Unnamed: 0']
    ↓
Min-max normalize per sample: (df - df.min().min()) / (df.max().max() - df.min().min())
    ↓
Convert to numpy: normalized_data.values
    ↓
Shape: (T, 75) where T = num_frames, 75 = 25 joints × 3 coords
    ↓
For visualization: reshape to (T, 25, 3) or (25, 3) per frame
    ↓
Range: [0.0, 1.0]
```

### feeders/feeder_mmasd.py Pipeline:

```
CSV File
    ↓
Load with pandas
    ↓
Drop label columns ['Action_Label', 'ASD_Label', 'Unnamed: 0']
    ↓
Min-max normalize per sample: (df - df.min().min()) / (df.max().max() - df.min().min())
    ↓
Convert to numpy: normalized_data.values
    ↓
Shape: (T, 75) where T = num_frames, 75 = 25 joints × 3 coords
    ↓
Reshape to (T, 25, 3)
    ↓
Transpose to (3, T, 25)
    ↓
Add person dimension: (3, T, 25, 1)
    ↓
Resample to fixed window_size (e.g., 64 frames)
    ↓
Apply optional augmentations (rotation, bone, velocity)
    ↓
Final shape: (C, T, V, M) = (3, 64, 25, 1)
    ↓
Base normalized range: [0.0, 1.0]
After augmentation: typically [-0.2, 1.4] (varies with rotation/bone transforms)
```

---

## Key Differences (Intentional)

| Aspect | visualize_enhanced_mmasd_3d.py | feeders/feeder_mmasd.py | Reason |
|--------|-------------------------------|------------------------|--------|
| **Output Shape** | (T, 75) or (T, 25, 3) | (3, 64, 25, 1) | Model requires (C, T, V, M) format |
| **Temporal Resampling** | None (variable length) | Resample to 64 frames | Model requires fixed temporal length |
| **Data Augmentation** | None | Rotation, bone, velocity | Training augmentation for better generalization |
| **Batch Processing** | Single file | Batch dataloader | Training requires batched data |

---

## Verified Test Results

### Test 1: Normalization Range
```bash
$ python test_normalization.py

Original data range: [-5.8104, 2.3467]
After normalization: [0.0000, 1.0000]
✓ SUCCESS: Data is properly normalized to [0, 1] range!
```

### Test 2: Sample Data Comparison
```bash
Original CSV (sample values, first joint, first 3 frames):
[[0.44394159 0.45494729 0.46576861]
 [0.4254362  0.44315016 0.4572452 ]
 [0.42968643 0.44491529 0.45773032]]

After Enhanced-MMASD normalization:
[[0.76673633 0.76808555 0.76941216]
 [0.76446771 0.76663931 0.76836726]
 [0.76498876 0.7668557  0.76842673]]
```

### Test 3: Full Dataloader Test
```bash
$ python test_mmasd_dataloader.py

============================================================
Testing MMASD Dataloader
============================================================

✓ Training dataset size: 2620
✓ Test dataset size: 656
✓ Batch shape: torch.Size([8, 3, 64, 25, 1])
✓ Data type: torch.float32
✓ Base normalized range: [0.0, 1.0]
✓ After augmentation range: [-0.12, 1.37] (expected with rotation)

============================================================
✓ All tests passed! Dataloader is working correctly.
============================================================
```

---

## Summary

### ✅ What Matches:

1. **CSV Loading**: Identical code for loading and dropping label columns
2. **Min-Max Normalization**: Identical per-sample normalization to [0, 1] range
3. **Data Values**: Same normalized values before model-specific transforms
4. **Joint Order**: Same 25 MediaPipe Pose joints in same order
5. **Coordinate System**: Same (x, y, z) coordinate order

### 📊 What's Different (By Design):

1. **Shape Transformation**: (T, 75) → (3, 64, 25, 1) for model compatibility
2. **Temporal Resampling**: Fixed 64-frame windows for model input
3. **Data Augmentation**: Added rotation, bone, velocity transforms for training
4. **Batching**: DataLoader for efficient batch training

### 🎯 Conclusion:

**The MMASD feeder now applies the EXACT same data transformations as `visualize_enhanced_mmasd_3d.py` and the Enhanced-MMASD training code.** The additional transformations (reshaping, resampling, augmentation) are model-specific requirements that don't change the underlying normalized data values.

---

## Code References

### Enhanced-MMASD Original Code:
- **File**: `Enhanced-MMASD/Code/proposed_model.py`
- **Method**: `CustomVideoDataset._load_dataframe()` (lines 115-127)
- **Normalization**: Lines 119-122

### Visualization Script:
- **File**: `visualize_enhanced_mmasd_3d.py`
- **Class**: `CustomSkeletonDataset`
- **Method**: `load_and_normalize()` (lines 15-35)
- **Normalization**: Lines 29-33

### MMASD Feeder:
- **File**: `feeders/feeder_mmasd.py`
- **Class**: `Feeder`
- **Method**: `_load_csv()` (lines 129-167)
- **Normalization**: Lines 147-153

---

## Date: November 12, 2025

**Status:** ✅ **VERIFIED - Data transformations match exactly**

**Reviewer Note:** The normalization discrepancy was identified and fixed. The feeder now applies the same min-max normalization per sample as the Enhanced-MMASD code, ensuring consistent data preprocessing across visualization and training.


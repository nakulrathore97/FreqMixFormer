import numpy as np
import pandas as pd
import os
import glob
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, 
                 random_shift=False, random_move=False, random_rot=False, window_size=-1, 
                 normalization=False, debug=False, use_mmap=False, bone=False, vel=False,
                 train_val_split=0.8):
        """
        MMASD Feeder for Enhanced-MMASD 3D skeleton data
        
        :param data_path: Root directory containing action folders with CSV files
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the beginning or end of sequence
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param train_val_split: ratio for train/test split (default 0.8)
        """
        
        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.train_val_split = train_val_split
        
        # Action to label mapping based on Enhanced-MMASD structure
        self.action_to_label = {
            'Arm_Swing': 0,
            'Body_pose': 1,
            'chest_expansion': 2,
            'Drumming': 3,
            'Frog_Pose': 4,
            'Marcas_Forward': 5,
            'Marcas_Shaking': 6,
            'Sing_Clap': 7,
            'Squat_Pose': 8,
            'Tree_Pose': 9,
            'Twist_Pose': 10
        }
        
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        """Load CSV files from MMASD dataset"""
        self.sample_name = []
        self.label = []
        csv_files = []
        
        # Scan all action folders
        action_folders = [d for d in os.listdir(self.data_path) 
                         if os.path.isdir(os.path.join(self.data_path, d))]
        
        for action_folder in action_folders:
            if action_folder not in self.action_to_label:
                continue
                
            action_path = os.path.join(self.data_path, action_folder)
            label = self.action_to_label[action_folder]
            
            # Get all CSV files in this action folder
            csv_list = sorted(glob.glob(os.path.join(action_path, '*.csv')))
            
            for csv_file in csv_list:
                csv_files.append(csv_file)
                self.sample_name.append(os.path.basename(csv_file))
                self.label.append(label)
        
        # Sort to ensure reproducibility
        combined = list(zip(csv_files, self.sample_name, self.label))
        combined.sort(key=lambda x: x[1])  # Sort by filename
        csv_files, self.sample_name, self.label = zip(*combined)
        csv_files = list(csv_files)
        self.sample_name = list(self.sample_name)
        self.label = list(self.label)
        
        # Split into train and test
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(csv_files))
        split_idx = int(len(csv_files) * self.train_val_split)
        
        if self.split == 'train':
            indices = indices[:split_idx]
        elif self.split == 'test':
            indices = indices[split_idx:]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        csv_files = [csv_files[i] for i in indices]
        self.sample_name = [self.sample_name[i] for i in indices]
        self.label = [self.label[i] for i in indices]
        
        if self.debug:
            csv_files = csv_files[:100]
            self.sample_name = self.sample_name[:100]
            self.label = self.label[:100]
        
        # Load all CSV files into memory
        self.data = []
        print(f"Loading {len(csv_files)} CSV files for {self.split} split...")
        
        for i, csv_file in enumerate(csv_files):
            if (i + 1) % 500 == 0:
                print(f"  Loaded {i + 1}/{len(csv_files)} files...")
            skeleton_data = self._load_csv(csv_file)
            self.data.append(skeleton_data)
        
        self.label = np.array(self.label)
        print(f"Loaded {len(self.data)} samples from MMASD dataset ({self.split} split)")

    def _load_csv(self, csv_file):
        """
        Load and process a single CSV file
        Returns data in shape (C, T, V, M) format
        C=3 (x,y,z), T=num_frames, V=25 joints, M=1 person
        
        IMPORTANT: Applies the same min-max normalization as Enhanced-MMASD code
        """
        df = pd.read_csv(csv_file)
        
        # Drop label columns if present
        if 'Action_Label' in df.columns:
            df = df.drop(['Action_Label'], axis=1, errors='ignore')
        if 'ASD_Label' in df.columns:
            df = df.drop(['ASD_Label'], axis=1, errors='ignore')
        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0'], axis=1, errors='ignore')
        
        # Apply min-max normalization per sample (same as Enhanced-MMASD)
        # This matches the normalization in:
        # - Enhanced-MMASD/Code/proposed_model.py line 119-122
        # - visualize_enhanced_mmasd_3d.py line 29-33
        df_min = df.min().min()
        df_max = df.max().max()
        normalized_df = (df - df_min) / (df_max - df_min)
        
        # Get data as numpy array: shape (T, 75) where 75 = 25 joints * 3 coords
        data_array = normalized_df.values  # (T, 75)
        T = data_array.shape[0]
        
        # Reshape to (T, 25, 3)
        data_reshaped = data_array.reshape(T, 25, 3)
        
        # Convert to (C, T, V, M) format
        # C=3 (x,y,z), T=num_frames, V=25 joints, M=1 person
        data_ctvm = data_reshaped.transpose(2, 0, 1)  # (3, T, 25)
        data_ctvm = np.expand_dims(data_ctvm, axis=-1)  # (3, T, 25, 1)
        
        return data_ctvm.astype(np.float32)

    def get_mean_map(self):
        """Calculate mean and std for normalization"""
        data = np.array(self.data)
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        
        # Get valid frame number (non-zero frames)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        # Crop and resize to window_size
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        
        if self.bone:
            # MediaPipe Pose bone pairs (25 joints, 1-indexed converted to 0-indexed)
            bone_pairs = [
                (1, 0), (2, 0),  # eyes to nose
                (3, 1), (4, 2),  # ears to eyes
                (6, 5),  # shoulders
                (5, 0), (6, 0),  # shoulders to nose
                (7, 5), (9, 7),  # left arm
                (8, 6), (10, 8),  # right arm
                (11, 9), (13, 9),  # left wrist to fingers
                (12, 10), (14, 10),  # right wrist to fingers
                (15, 5), (16, 6),  # shoulders to hips
                (16, 15),  # hips
                (17, 15), (19, 17),  # left leg
                (18, 16), (20, 18),  # right leg
                (21, 19), (23, 19),  # left foot
                (22, 20), (24, 20),  # right foot
            ]
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in bone_pairs:
                bone_data_numpy[:, :, v1] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            data_numpy = bone_data_numpy
        
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

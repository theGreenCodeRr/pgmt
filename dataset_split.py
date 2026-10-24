import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class NTURGBD_SplitDataset(Dataset):
    """
    PGMT Tier 3 Split Dataset:
    Loads pre-extracted visual chunks (geo_fused, sem_global) from RAM and parses 
    raw skeletons on the fly. Applies strict NTU-120 X-Sub, X-Set, and Zero-Shot splits.
    """
    def __init__(self, chunk_data, skeleton_dir=None, 
                 protocol="xsub", split="train", zsl_classes=0, 
                 num_visual_frames=3, window_size=30, max_subjects=2):
        super().__init__()
        
        # FIX 1: Point exactly to skeleton_cache where the files are actually staged
        if skeleton_dir is None:
            slurm_tmp = os.environ.get("SLURM_TMPDIR", "/tmp")
            self.skeleton_dir = os.path.join(slurm_tmp, "skeleton_cache")
        else:
            self.skeleton_dir = os.path.normpath(skeleton_dir)

        # FIX 2: Strict safety check. Crash immediately if the folder is missing.
        if not os.path.exists(self.skeleton_dir):
            raise FileNotFoundError(
                f"[!] FATAL: Skeleton directory not found at '{self.skeleton_dir}'. "
                f"Please ensure your SLURM script copies the files to this exact path."
            )

        self.chunk_data = chunk_data
        
        self.protocol = protocol.lower()
        self.split = split.lower()
        self.zsl_classes = zsl_classes
        
        self.num_visual_frames = num_visual_frames
        self.window_size = window_size
        self.max_subjects = max_subjects
        self.num_joints = 25
        
        # NTU Kinect v2 Bone Connectivity
        self.bone_pairs = [
            (1, 0), (20, 1), (2, 20), (3, 2), (20, 4), (4, 5), (5, 6), (6, 7), 
            (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), 
            (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (22, 21), 
            (7, 22), (24, 23), (11, 24)
        ]
        
        # NTU-120 Official Cross-Subject (X-Sub) Train IDs (53 Subjects)
        self.ntu120_train_subjects = [
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        
        # Zero-Shot Unseen Classes (Zero-Indexed for PyTorch)
        self.unseen_classes = []
        if self.zsl_classes == 10:
            self.unseen_classes = list(range(110, 120))
        elif self.zsl_classes == 24:
            self.unseen_classes = list(range(96, 120))
            
        self.samples = self._filter_chunk()
        logger.info(f"[*] Chunk Filtered ({self.protocol.upper()} | {self.split.upper()} | ZSL: {self.zsl_classes}). Valid Samples: {len(self.samples)}")

    def _parse_filename(self, filename):
        match = re.search(r'S(\d{3})C(\d{3})P(\d{3})R\d{3}A(\d{3})', filename)
        if match:
            return {
                'setup': int(match.group(1)),
                'camera': int(match.group(2)),
                'subject': int(match.group(3)),
                'action': int(match.group(4)) - 1 # Zero-indexed for PyTorch CrossEntropy
            }
        return None

    def _filter_chunk(self):
        valid_samples = []
        for sample_id, features in self.chunk_data.items():
            meta = self._parse_filename(sample_id)
            if not meta:
                continue
                
            # 1. Protocol Filtering
            is_train_split = False
            if self.protocol == 'xsub':
                is_train_split = meta['subject'] in self.ntu120_train_subjects
            elif self.protocol == 'xset':
                is_train_split = (meta['setup'] % 2 == 0) # Even setups are train
                
            # Filter out samples not belonging to the requested split
            if self.split == 'train' and not is_train_split:
                continue
            if self.split == 'test' and is_train_split:
                continue
                
            # 2. Zero-Shot Filtering (Masking unseen classes)
            if self.zsl_classes > 0:
                is_unseen = meta['action'] in self.unseen_classes
                if self.split == 'train' and is_unseen:
                    continue # Train split must never see ZSL classes
                if self.split == 'test' and not is_unseen:
                    continue # Test split for ZSL should only evaluate unseen classes
                    
            valid_samples.append({
                'sample_id': sample_id,
                'action': meta['action'],
                'geo_fused': features['geo_fused'],
                'sem_global': features['sem_global']
            })
            
        return valid_samples

    def _parse_ntu_skeleton(self, skel_path):
        with open(skel_path, 'r') as f:
            lines = f.readlines()
            
        if not lines: raise ValueError(f"Empty skeleton file: {skel_path}")

        num_frames = int(lines[0].strip())
        kinematics = np.zeros((num_frames, self.max_subjects, self.num_joints, 3), dtype=np.float32)

        line_idx = 1
        for f_idx in range(num_frames):
            if line_idx >= len(lines): break
            num_bodies = int(lines[line_idx].strip())
            line_idx += 1
            
            for b_idx in range(num_bodies):
                line_idx += 1 # Skip body info
                num_joints_in_body = int(lines[line_idx].strip())
                line_idx += 1
                
                valid_body = b_idx < self.max_subjects
                for j_idx in range(num_joints_in_body):
                    joint_info = list(map(float, lines[line_idx].strip().split()))
                    if valid_body and j_idx < self.num_joints:
                        kinematics[f_idx, b_idx, j_idx, :] = joint_info[0:3]
                    line_idx += 1
                    
        return torch.from_numpy(kinematics)

    def _compute_kinematic_physics(self, joints_window):
        bones = torch.zeros_like(joints_window)
        for target, source in self.bone_pairs:
            bones[:, :, target, :] = joints_window[:, :, target, :] - joints_window[:, :, source, :]
            
        padded_joints = torch.cat([joints_window[0:1], joints_window], dim=0)
        motion = padded_joints[1:] - padded_joints[:-1]
        
        return torch.stack([joints_window, bones, motion], dim=0)

    def _get_temporal_anchors(self, total_frames):
        if total_frames < self.num_visual_frames:
            return np.arange(total_frames).tolist() + [total_frames - 1] * (self.num_visual_frames - total_frames)
        step = total_frames / self.num_visual_frames
        return [int(step / 2 + i * step) for i in range(self.num_visual_frames)]

    def _extract_skeleton_window(self, kinematics, anchor_idx, total_frames):
        half_win = self.window_size // 2
        start_idx = max(0, anchor_idx - half_win)
        end_idx = min(total_frames, anchor_idx + half_win + (self.window_size % 2))
        
        skel_win = kinematics[start_idx:end_idx]
        
        if skel_win.shape[0] < self.window_size:
            pad_length = self.window_size - skel_win.shape[0]
            if start_idx == 0:
                padding = skel_win[0:1].repeat(pad_length, 1, 1, 1)
                skel_win = torch.cat([padding, skel_win], dim=0)
            else:
                padding = skel_win[-1:].repeat(pad_length, 1, 1, 1)
                skel_win = torch.cat([skel_win, padding], dim=0)
                
        return skel_win

    def __getitem__(self, idx):
        # Bound retry loop for corrupted skeletons
        for attempt in range(5):
            current_idx = (idx + attempt) % len(self.samples)
            sample_info = self.samples[current_idx]
            skel_path = os.path.join(self.skeleton_dir, f"{sample_info['sample_id']}.skeleton")
            
            try:
                if not os.path.exists(skel_path):
                    raise FileNotFoundError(f"Missing skeleton file: {skel_path}")
                    
                kinematics = self._parse_ntu_skeleton(skel_path)
                if torch.isnan(kinematics).any() or torch.isinf(kinematics).any():
                    raise ValueError("Corrupted Kinect Sensor: NaN/Inf detected in joints")
                
                total_frames = kinematics.shape[0]
                if total_frames <= 0: raise ValueError("Skeleton contains 0 frames")
                
                anchor_indices = self._get_temporal_anchors(total_frames)
                physics_streams = []
                
                for anchor in anchor_indices:
                    skel_win = self._extract_skeleton_window(kinematics, anchor, total_frames)
                    physics_streams.append(self._compute_kinematic_physics(skel_win))
                    
                return {
                    "sample_id": sample_info['sample_id'],
                    "action_label": sample_info['action'],
                    "geo_fused": sample_info['geo_fused'],
                    "sem_global": sample_info['sem_global'],
                    "kinematic_streams": torch.stack(physics_streams)
                }
            except Exception as e:
                logger.debug(f"Failed to load {skel_path}: {str(e)}")
                continue
                
        # FIX 3: Do not return dummy zeros. Crash loudly if data is actually missing.
        raise RuntimeError(f"[!] FATAL: Failed to load valid skeleton data after 5 retries. Check if {self.skeleton_dir} is populated.")

    def __len__(self):
        return len(self.samples)
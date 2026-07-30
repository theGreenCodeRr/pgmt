import os
import glob
import torch
import numpy as np
import cv2
import re
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class NTURGBD_EvalDataset(Dataset):
    """
    PGMT Phase 1 Evaluation Dataset.
    Includes official NTU RGB+D splitting protocols (X-Sub, X-View) 
    and extracts ground-truth Action Labels for t-SNE and Zero-Shot testing.
    """
    def __init__(self, data_root="/pnr/lab/nurul/datasets/ntu_rgbd", 
                 protocol="xsub", split="test", 
                 num_visual_frames=3, window_size=30, max_subjects=2):
        super().__init__()
        self.data_root = os.path.normpath(data_root)
        self.rgb_dir = os.path.join(self.data_root, "rgb")
        self.skeleton_dir = os.path.join(self.data_root, "skeleton")
        
        self.num_visual_frames = num_visual_frames
        self.window_size = window_size
        self.max_subjects = max_subjects
        self.num_joints = 25
        self.protocol = protocol.lower() # 'xsub', 'xview', or 'all'
        self.split = split.lower()       # 'train', 'test', or 'all'
        
        # NTU-60 Official Cross-Subject (X-Sub) Training IDs
        self.ntu60_train_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        # NTU-60 Official Cross-View (X-View) Training Camera IDs
        self.ntu60_train_cameras = [2, 3] 

        self.bone_pairs = [
            (1, 0), (20, 1), (2, 20), (3, 2), (20, 4), (4, 5), (5, 6), (6, 7), 
            (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), 
            (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (22, 21), 
            (7, 22), (24, 23), (11, 24)
        ]
        
        self.samples = self._build_dataset_index()
        print(f"[*] Initialized PGMT Eval Dataset ({self.protocol.upper()} - {self.split.upper()}). Found {len(self.samples)} valid samples.")

    def _parse_filename(self, filename):
        """Extracts Setup (S), Camera (C), Performer (P), and Action (A) IDs."""
        match = re.search(r'S(\d{3})C(\d{3})P(\d{3})R\d{3}A(\d{3})', filename)
        if match:
            return {
                'setup': int(match.group(1)),
                'camera': int(match.group(2)),
                'subject': int(match.group(3)),
                'action': int(match.group(4))
            }
        return None

    def _is_valid_split(self, meta_info):
        """Determines if the file belongs in the requested train/test split."""
        if self.protocol == 'all' or self.split == 'all':
            return True
            
        if self.protocol == 'xsub':
            is_train = meta_info['subject'] in self.ntu60_train_subjects
            return is_train if self.split == 'train' else not is_train
            
        elif self.protocol == 'xview':
            is_train = meta_info['camera'] in self.ntu60_train_cameras
            return is_train if self.split == 'train' else not is_train
            
        elif self.protocol == 'xset': # NTU-120 Protocol
            is_train = (meta_info['setup'] % 2 == 0) # Even setups are train
            return is_train if self.split == 'train' else not is_train
            
        return False

    def _build_dataset_index(self):
        valid_samples = []
        skeleton_files = glob.glob(os.path.join(self.skeleton_dir, "*.skeleton"))
        
        for skel_path in skeleton_files:
            base_name = os.path.basename(skel_path).replace('.skeleton', '')
            meta_info = self._parse_filename(base_name)
            
            if not meta_info or not self._is_valid_split(meta_info):
                continue # Skip files not in our target split
                
            rgb_path = os.path.join(self.rgb_dir, f"{base_name}_rgb.avi")
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(self.rgb_dir, f"{base_name}.avi")
                
            if os.path.exists(rgb_path):
                valid_samples.append({
                    'sample_id': base_name,
                    'skeleton_path': skel_path,
                    'rgb_path': rgb_path,
                    'action_label': meta_info['action'] - 1 # 0-indexed for PyTorch
                })
                
        return valid_samples

    def _parse_ntu_skeleton(self, skel_path):
        with open(skel_path, 'r') as f:
            lines = f.readlines()
            
        if not lines: raise ValueError(f"Empty skeleton file: {skel_path}")

        num_frames = int(lines[0].strip())
        kinematics = np.zeros((num_frames, self.max_subjects, self.num_joints, 3), dtype=np.float32)
        anchors_2d = np.zeros((num_frames, self.max_subjects, 2), dtype=np.float32)

        line_idx = 1
        for f_idx in range(num_frames):
            if line_idx >= len(lines): break
            num_bodies = int(lines[line_idx].strip())
            line_idx += 1
            
            for b_idx in range(num_bodies):
                _ = lines[line_idx].strip().split() 
                line_idx += 1
                num_joints_in_body = int(lines[line_idx].strip())
                line_idx += 1
                
                valid_body = b_idx < self.max_subjects
                for j_idx in range(num_joints_in_body):
                    joint_info = list(map(float, lines[line_idx].strip().split()))
                    if valid_body and j_idx < self.num_joints:
                        kinematics[f_idx, b_idx, j_idx, :] = joint_info[0:3]
                        if j_idx == 0:
                            anchors_2d[f_idx, b_idx, 0] = joint_info[5] / 1920.0
                            anchors_2d[f_idx, b_idx, 1] = joint_info[6] / 1080.0
                    line_idx += 1
                    
        return torch.tensor(kinematics), torch.tensor(anchors_2d)

    def _compute_kinematic_physics(self, joints_window):
        bones = torch.zeros_like(joints_window)
        for target, source in self.bone_pairs:
            bones[:, :, target, :] = joints_window[:, :, target, :] - joints_window[:, :, source, :]
            
        motion = torch.zeros_like(joints_window)
        padded_joints = torch.cat([joints_window[0:1], joints_window], dim=0)
        motion = padded_joints[1:] - padded_joints[:-1]
        
        return torch.stack([joints_window, bones, motion], dim=0) 

    def _get_temporal_anchors(self, total_frames):
        if total_frames < self.num_visual_frames:
            return np.arange(total_frames).tolist() + [total_frames - 1] * (self.num_visual_frames - total_frames)
        step = total_frames / self.num_visual_frames
        return [int(step / 2 + i * step) for i in range(self.num_visual_frames)]

    def _extract_skeleton_window(self, kinematics, anchors_2d, anchor_idx, total_frames):
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
                
        anchor_2d = anchors_2d[anchor_idx] 
        return skel_win, anchor_2d

    def _load_rgb_frames(self, rgb_path, anchor_indices):
        cap = cv2.VideoCapture(rgb_path)
        frames_siglip = []
        frames_dinov2 = []
        
        if not cap.isOpened():
            return torch.zeros((len(anchor_indices), 3, 384, 384)), torch.zeros((len(anchor_indices), 3, 336, 336))
            
        for anchor_idx in anchor_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                f_384 = cv2.resize(frame_rgb, (384, 384))
                t_384 = torch.from_numpy(f_384).permute(2, 0, 1).float() / 255.0
                frames_siglip.append(t_384)
                
                f_336 = cv2.resize(frame_rgb, (336, 336))
                t_336 = torch.from_numpy(f_336).permute(2, 0, 1).float() / 255.0
                frames_dinov2.append(t_336)
            else:
                frames_siglip.append(torch.zeros((3, 384, 384)))
                frames_dinov2.append(torch.zeros((3, 336, 336)))
                
        cap.release()
        return torch.stack(frames_siglip), torch.stack(frames_dinov2)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        try:
            kinematics, anchors_2d = self._parse_ntu_skeleton(sample_info['skeleton_path'])
            if torch.isnan(kinematics).any() or torch.isinf(kinematics).any():
                raise ValueError("Corrupted Kinect Sensor Data")
            total_frames = kinematics.shape[0]
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))

        anchor_indices = self._get_temporal_anchors(total_frames)
        rgb_siglip, rgb_dinov2 = self._load_rgb_frames(sample_info['rgb_path'], anchor_indices)
        
        physics_streams = []
        spatial_anchors = []
        
        for anchor in anchor_indices:
            skel_win, anchor_2d = self._extract_skeleton_window(kinematics, anchors_2d, anchor, total_frames)
            multi_stream_physics = self._compute_kinematic_physics(skel_win)
            physics_streams.append(multi_stream_physics)
            spatial_anchors.append(anchor_2d)
            
        return {
            "sample_id": sample_info['sample_id'],
            "action_label": sample_info['action_label'], # NEW: Required for t-SNE and Zero-Shot!
            "visual_frames_siglip": rgb_siglip,
            "visual_frames_dinov2": rgb_dinov2,
            "kinematic_streams": torch.stack(physics_streams),
            "spatial_anchors": torch.stack(spatial_anchors)
        }

    def __len__(self):
        return len(self.samples)
import os
import glob
import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# Prevent OpenCV from spawning internal sub-threads per DataLoader worker
cv2.setNumThreads(0)

class NTURGBD_AlignedDataset(Dataset):
    """
    PGMT Phase 1 Dataset:
    Aligns sparse RGB visual frames with high-frequency kinematic skeleton windows.
    Acts as an on-the-fly physics engine computing Joints, Bones, and Motion (Velocity),
    outputting dual-resolution visual frames for the SigLIP (384) + DINOv2 (336) backbones.
    """
    def __init__(self, data_root="/pnr/lab/nurul/datasets/ntu_rgbd", 
                 num_visual_frames=3, window_size=30, max_subjects=2):
        super().__init__()
        self.data_root = os.path.normpath(data_root)
        self.rgb_dir = os.path.join(self.data_root, "rgb")
        self.skeleton_dir = os.path.join(self.data_root, "skeleton")
        
        self.num_visual_frames = num_visual_frames
        self.window_size = window_size
        self.max_subjects = max_subjects
        self.num_joints = 25  # Standard NTU Kinect v2 topology
        
        # NTU Kinect v2 Bone Connectivity (Target -> Source)
        self.bone_pairs = [
            (1, 0), (20, 1), (2, 20), (3, 2), (20, 4), (4, 5), (5, 6), (6, 7), 
            (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), 
            (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (22, 21), 
            (7, 22), (24, 23), (11, 24)
        ]
        
        self.samples = self._build_dataset_index()
        print(f"[*] Initialized PGMT Dataset. Found {len(self.samples)} valid aligned samples.")

    def _build_dataset_index(self):
        valid_samples = []
        print(f"[*] Searching for skeletons in: {self.skeleton_dir}")
        skeleton_files = glob.glob(os.path.join(self.skeleton_dir, "*.skeleton"))
        print(f"[*] Found {len(skeleton_files)} raw .skeleton files.")
        
        for skel_path in skeleton_files:
            base_name = os.path.basename(skel_path).replace('.skeleton', '')
            rgb_path = os.path.join(self.rgb_dir, f"{base_name}_rgb.avi")
            
            if not os.path.exists(rgb_path):
                rgb_path = os.path.join(self.rgb_dir, f"{base_name}.avi")
                
            if os.path.exists(rgb_path):
                valid_samples.append({
                    'sample_id': base_name,
                    'skeleton_path': skel_path,
                    'rgb_path': rgb_path
                })
                
        if len(skeleton_files) > 0 and len(valid_samples) == 0:
            print(f"[!] WARNING: Found skeletons, but NO matching RGB .avi files found in: {self.rgb_dir}")
            
        return valid_samples

    def _parse_ntu_skeleton(self, skel_path):
        """
        Parses raw skeleton file into 3D joint coordinates and 2D image anchors.
        """
        with open(skel_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            raise ValueError(f"Empty skeleton file: {skel_path}")

        num_frames = int(lines[0].strip())
        kinematics = np.zeros((num_frames, self.max_subjects, self.num_joints, 3), dtype=np.float32)
        anchors_2d = np.zeros((num_frames, self.max_subjects, 2), dtype=np.float32)

        line_idx = 1
        for f_idx in range(num_frames):
            if line_idx >= len(lines): 
                break
            num_bodies = int(lines[line_idx].strip())
            line_idx += 1
            
            for b_idx in range(num_bodies):
                line_idx += 1  # Skip body tracking status line
                num_joints_in_body = int(lines[line_idx].strip())
                line_idx += 1
                
                valid_body = b_idx < self.max_subjects
                for j_idx in range(num_joints_in_body):
                    joint_info = list(map(float, lines[line_idx].strip().split()))
                    
                    if valid_body and j_idx < self.num_joints:
                        kinematics[f_idx, b_idx, j_idx, :] = joint_info[0:3]
                        # Spine Base (Joint 0) used for 2D visual anchor
                        if j_idx == 0:
                            anchors_2d[f_idx, b_idx, 0] = joint_info[5] / 1920.0
                            anchors_2d[f_idx, b_idx, 1] = joint_info[6] / 1080.0
                    line_idx += 1
                    
        return torch.from_numpy(kinematics), torch.from_numpy(anchors_2d)

    def _compute_kinematic_physics(self, joints_window):
        """
        Takes joints window (W, S, V, 3) and computes Bones and Motion.
        Returns stacked tensor of shape (3, W, S, V, 3).
        """
        # 1. Joints (Absolute 3D positions)
        # 2. Bones (Spatial displacement vectors between adjacent joints)
        bones = torch.zeros_like(joints_window)
        for target, source in self.bone_pairs:
            bones[:, :, target, :] = joints_window[:, :, target, :] - joints_window[:, :, source, :]
            
        # 3. Motion (Temporal frame-to-frame velocity)
        padded_joints = torch.cat([joints_window[0:1], joints_window], dim=0)
        motion = padded_joints[1:] - padded_joints[:-1]
        
        return torch.stack([joints_window, bones, motion], dim=0)  # (3, W, S, V, 3)

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
        """
        Loads and resizes RGB anchor frames for SigLIP (384) and DINOv2 (336).
        Guarantees resource release via try/finally.
        """
        cap = cv2.VideoCapture(rgb_path)
        frames_siglip = []
        frames_dinov2 = []
        
        try:
            if not cap.isOpened():
                return torch.zeros((len(anchor_indices), 3, 384, 384)), torch.zeros((len(anchor_indices), 3, 336, 336))
                
            for anchor_idx in anchor_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # SigLIP resolution (384x384)
                    f_384 = cv2.resize(frame_rgb, (384, 384), interpolation=cv2.INTER_LINEAR)
                    t_384 = torch.from_numpy(f_384).permute(2, 0, 1).float() / 255.0
                    frames_siglip.append(t_384)
                    
                    # DINOv2 resolution (336x336)
                    f_336 = cv2.resize(frame_rgb, (336, 336), interpolation=cv2.INTER_LINEAR)
                    t_336 = torch.from_numpy(f_336).permute(2, 0, 1).float() / 255.0
                    frames_dinov2.append(t_336)
                else:
                    frames_siglip.append(torch.zeros((3, 384, 384), dtype=torch.float32))
                    frames_dinov2.append(torch.zeros((3, 336, 336), dtype=torch.float32))
        finally:
            cap.release()
            
        return torch.stack(frames_siglip), torch.stack(frames_dinov2)

    def __getitem__(self, idx):
        # Bounded retry loop to avoid recursive stack overflow on corrupted files
        for attempt in range(5):
            current_idx = (idx + attempt) % len(self.samples)
            sample_info = self.samples[current_idx]
            
            try:
                kinematics, anchors_2d = self._parse_ntu_skeleton(sample_info['skeleton_path'])
                
                # Check for corrupted Kinect sensor outputs
                if torch.isnan(kinematics).any() or torch.isinf(kinematics).any():
                    raise ValueError("Corrupted Kinect Sensor: NaN/Inf detected in joints")
                if torch.isnan(anchors_2d).any() or torch.isinf(anchors_2d).any():
                    raise ValueError("Corrupted Kinect Sensor: NaN/Inf detected in anchors")
                    
                total_frames = kinematics.shape[0]
                if total_frames <= 0:
                    raise ValueError("Skeleton contains 0 frames")
                    
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
                    "visual_frames_siglip": rgb_siglip,   # (T, 3, 384, 384)
                    "visual_frames_dinov2": rgb_dinov2,   # (T, 3, 336, 336)
                    "kinematic_streams": torch.stack(physics_streams),  # (T, 3, W, S, 25, 3)
                    "spatial_anchors": torch.stack(spatial_anchors)     # (T, S, 2)
                }
            except Exception:
                continue
                
        # Fallback dummy zero-batch if 5 consecutive files fail
        return {
            "sample_id": "dummy",
            "visual_frames_siglip": torch.zeros((self.num_visual_frames, 3, 384, 384), dtype=torch.float32),
            "visual_frames_dinov2": torch.zeros((self.num_visual_frames, 3, 336, 336), dtype=torch.float32),
            "kinematic_streams": torch.zeros((self.num_visual_frames, 3, self.window_size, self.max_subjects, self.num_joints, 3), dtype=torch.float32),
            "spatial_anchors": torch.zeros((self.num_visual_frames, self.max_subjects, 2), dtype=torch.float32)
        }

    def __len__(self):
        return len(self.samples)
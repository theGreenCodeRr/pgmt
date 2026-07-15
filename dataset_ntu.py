import os
import glob
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class NTURGBD_AlignedDataset(Dataset):
    """
    PGMT Phase 1 Dataset: Aligns sparse RGB visual frames with high-frequency 
    kinematic skeleton windows and extracts native 2D spatial anchors.
    """
    def __init__(self, data_root=r"E:\datasets\ntu_rgbd", num_visual_frames=3, window_size=30, max_subjects=2):
        super().__init__()
        # Use os.path.normpath to strictly prevent Windows \n or \d escape character bugs
        self.data_root = os.path.normpath(data_root)
        self.rgb_dir = os.path.join(self.data_root, "rgb")
        self.skeleton_dir = os.path.join(self.data_root, "skeleton")
        
        self.num_visual_frames = num_visual_frames
        self.window_size = window_size
        self.max_subjects = max_subjects
        self.num_joints = 25 # Standard NTU Kinect v2 format
        
        # Task 1.1: Build aligned index
        self.samples = self._build_dataset_index()
        print(f"[*] Initialized PGMT Dataset. Found {len(self.samples)} valid aligned samples.")

    def _build_dataset_index(self):
        """
        Scans directories, aligns valid RGB videos with their corresponding 
        skeleton files based on NTU naming conventions.
        """
        valid_samples = []
        
        # Add debugging to isolate the Windows Path issue
        print(f"[*] Searching for skeletons in: {self.skeleton_dir}")
        skeleton_files = glob.glob(os.path.join(self.skeleton_dir, "*.skeleton"))
        print(f"[*] Found {len(skeleton_files)} raw .skeleton files.")
        
        for skel_path in skeleton_files:
            base_name = os.path.basename(skel_path).replace('.skeleton', '')
            
            # NTU RGB videos typically append _rgb.avi
            rgb_path = os.path.join(self.rgb_dir, f"{base_name}_rgb.avi")
            
            # Fallback for exact name match if _rgb is not appended
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
        Parses the raw NTU .skeleton text file.
        Returns:
            kinematics: shape (num_frames, max_subjects, 25, 3) -> [x, y, z]
            anchors_2d: shape (num_frames, max_subjects, 2) -> [colorX, colorY] of Spine Base
        """
        with open(skel_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            raise ValueError(f"Empty skeleton file: {skel_path}")

        num_frames = int(lines[0].strip())
        
        # Initialize padded tensors with zeros (handles max_subjects = 2)
        kinematics = np.zeros((num_frames, self.max_subjects, self.num_joints, 3), dtype=np.float32)
        anchors_2d = np.zeros((num_frames, self.max_subjects, 2), dtype=np.float32)

        line_idx = 1
        for f_idx in range(num_frames):
            if line_idx >= len(lines):
                break
                
            num_bodies = int(lines[line_idx].strip())
            line_idx += 1
            
            for b_idx in range(num_bodies):
                body_info = lines[line_idx].strip().split()
                line_idx += 1
                
                num_joints_in_body = int(lines[line_idx].strip())
                line_idx += 1
                
                # Only process up to max_subjects
                valid_body = b_idx < self.max_subjects
                
                for j_idx in range(num_joints_in_body):
                    joint_info = list(map(float, lines[line_idx].strip().split()))
                    
                    if valid_body and j_idx < self.num_joints:
                        # [x, y, z] are indices 0, 1, 2
                        kinematics[f_idx, b_idx, j_idx, :] = joint_info[0:3]
                        
                        # Task 1.3: Native 2D Spatial Anchoring via Spine Base (Joint 0)
                        if j_idx == 0:
                            # [colorX, colorY] are indices 5, 6 in NTU format
                            # FIXED: Normalize by Kinect v2 HD resolution (1920x1080)
                            # Prevents neural network feature saturation from massive integer values
                            anchors_2d[f_idx, b_idx, 0] = joint_info[5] / 1920.0
                            anchors_2d[f_idx, b_idx, 1] = joint_info[6] / 1080.0
                            
                    line_idx += 1
                    
        return torch.tensor(kinematics), torch.tensor(anchors_2d)

    def _get_temporal_anchors(self, total_frames):
        """
        Task 1.2: Calculates the frame indices to sample the T visual frames.
        Samples at evenly spaced intervals (e.g., 10%, 50%, 90%).
        """
        if total_frames < self.num_visual_frames:
            return np.arange(total_frames).tolist() + [total_frames - 1] * (self.num_visual_frames - total_frames)
            
        step = total_frames / self.num_visual_frames
        anchors = [int(step / 2 + i * step) for i in range(self.num_visual_frames)]
        return anchors

    def _extract_skeleton_window(self, kinematics, anchors_2d, anchor_idx, total_frames):
        """
        Task 1.2: Extracts the [t - dt, t + dt] kinematic window.
        Uses clamp/padding if the window exceeds sequence boundaries.
        """
        half_win = self.window_size // 2
        start_idx = max(0, anchor_idx - half_win)
        end_idx = min(total_frames, anchor_idx + half_win + (self.window_size % 2))
        
        skel_win = kinematics[start_idx:end_idx]
        
        # Pad temporal dimension if at the very start/end of the video
        if skel_win.shape[0] < self.window_size:
            pad_length = self.window_size - skel_win.shape[0]
            if start_idx == 0:
                padding = skel_win[0:1].repeat(pad_length, 1, 1, 1)
                skel_win = torch.cat([padding, skel_win], dim=0)
            else:
                padding = skel_win[-1:].repeat(pad_length, 1, 1, 1)
                skel_win = torch.cat([skel_win, padding], dim=0)
                
        # The 2D anchor is taken exactly at the visual frame timestamp
        anchor_2d = anchors_2d[anchor_idx] 
        
        return skel_win, anchor_2d

    def _load_rgb_frames(self, rgb_path, anchor_indices):
        """
        Task 1.1: Extracts sparse visual frames using optimized OpenCV (cv2) frame seeking.
        Avoids decoding the entire video, keeping GPU/CPU overhead minimal.
        """
        cap = cv2.VideoCapture(rgb_path)
        frames = []
        
        if not cap.isOpened():
            print(f"[*] Warning: Could not open video {rgb_path}. Padding with zeros.")
            return torch.zeros((len(anchor_indices), 3, 224, 224))
            
        for anchor_idx in anchor_indices:
            # Jump directly to the required frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR (OpenCV default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to standard ViT resolution
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                # Convert to PyTorch tensor format (C, H, W) and normalize [0, 1]
                frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                frames.append(frame_tensor)
            else:
                # Fallback for unexpected EOF
                frames.append(torch.zeros((3, 224, 224)))
                
        cap.release()
        return torch.stack(frames)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load and parse full skeleton
        try:
            kinematics, anchors_2d = self._parse_ntu_skeleton(sample_info['skeleton_path'])
            total_frames = kinematics.shape[0]
        except Exception as e:
            # Fallback for corrupted NTU files during training
            print(f"Error parsing {sample_info['skeleton_path']}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

        # 1. Determine Temporal Anchors
        anchor_indices = self._get_temporal_anchors(total_frames)
        
        # 2. Load Visual Frames
        rgb_tensor = self._load_rgb_frames(sample_info['rgb_path'], anchor_indices)
        
        # 3. Extract Synchronized Windows
        kinematic_windows = []
        spatial_anchors = []
        
        for anchor in anchor_indices:
            skel_win, anchor_2d = self._extract_skeleton_window(kinematics, anchors_2d, anchor, total_frames)
            kinematic_windows.append(skel_win)
            spatial_anchors.append(anchor_2d)
            
        return {
            "sample_id": sample_info['sample_id'],
            "visual_frames": rgb_tensor,
            "kinematic_windows": torch.stack(kinematic_windows),  # (T, Window_Size, Max_Subjects, 25, 3)
            "spatial_anchors": torch.stack(spatial_anchors)       # (T, Max_Subjects, 2)
        }

    def __len__(self):
        return len(self.samples)
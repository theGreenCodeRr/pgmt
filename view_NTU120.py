import os
import glob
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

# Configuration
BASE_DIR = r"/pnr/lab/nurul/datasets/ntu_rgbd" 
RGB_DIR = os.path.join(BASE_DIR, "rgb")
SKELETON_DIR = os.path.join(BASE_DIR, "skeleton")

# Kinect v2 Bone Connections (25 joints)
BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
]

# NTU RGB+D 120 Action Mapping (First 60 from NTU-60, 61-120 from NTU-120)
ACTION_MAP = {
    1: "drink water", 2: "eat meal/snack", 3: "brushing teeth", 4: "brushing hair", 5: "drop",
    6: "pickup", 7: "throw", 8: "sitting down", 9: "standing up", 10: "clapping",
    11: "reading", 12: "writing", 13: "tear up paper", 14: "wear jacket", 15: "take off jacket",
    16: "wear a shoe", 17: "take off a shoe", 18: "wear on glasses", 19: "take off glasses", 20: "put on a hat/cap",
    21: "take off a hat/cap", 22: "cheer up", 23: "hand waving", 24: "kicking something", 25: "reach into pocket",
    26: "hopping", 27: "jumping up", 28: "make a phone call", 29: "playing with phone/tablet", 30: "typing on a keyboard",
    31: "pointing to something", 32: "taking a selfie", 33: "check time (from watch)", 34: "rub two hands together", 35: "nod head/bow",
    36: "shake head", 37: "wipe face", 38: "salute", 39: "put the palms together", 40: "cross hands in front",
    41: "sneeze/cough", 42: "staggering", 43: "falling", 44: "touch head", 45: "touch chest",
    46: "touch back", 47: "touch neck", 48: "nausea or vomiting", 49: "use a fan/feeling warm", 50: "punching/slapping",
    51: "kicking other person", 52: "pushing other person", 53: "pat on back of other", 54: "point finger at other", 55: "hugging other person",
    56: "giving something to other", 57: "touch other's pocket", 58: "handshaking", 59: "walking towards each other", 60: "walking apart",
    61: "put on headphone", 62: "take off headphone", 63: "shoot at basket", 64: "bounce ball", 65: "tennis bat swing",
    66: "juggling table tennis balls", 67: "hush", 68: "flick hair", 69: "thumb up", 70: "thumb down",
    71: "make ok sign", 72: "make v sign", 73: "catch cap", 74: "parade tie", 75: "play magic cube",
    76: "read book", 77: "pass article", 78: "take article", 79: "hit with stick", 80: "pull paper from dispenser",
    81: "catch object", 82: "shoot with gun", 83: "facepalm", 84: "oh ok", 85: "vomiting",
    86: "sleep", 87: "cross arms", 88: "cross legs", 89: "open bottle", 90: "pour liquid",
    91: "cut object", 92: "chop object", 93: "stir liquid", 94: "blow nose", 95: "type on keyboard",
    96: "point to", 97: "take photo", 98: "check time", 99: "rub hands", 100: "nod head/bow",
    101: "shake head", 102: "wipe face", 103: "salute", 104: "put palms together", 105: "cross hands",
    106: "sneeze/cough", 107: "staggering", 108: "falling", 109: "touch head", 110: "touch chest",
    111: "touch back", 112: "touch neck", 113: "vomiting", 114: "use fan", 115: "punch/slap",
    116: "kick", 117: "push", 118: "pat back", 119: "point at", 120: "hug"
}

class NTUViewerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("NTU RGB+D Dataset Viewer")
        self.geometry("1600x600")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.file_pairs = self._get_file_pairs()
        self.current_index = 0
        
        self.cap = None
        self.skeleton_data = []
        self.current_frame_idx = 0
        self.total_frames = 1
        
        self.is_playing = True
        self.playback_delay = 33 # ms per frame (~30 FPS)
        self.show_skeleton = ctk.BooleanVar(value=True)
        
        self._setup_gui()
        self._bind_keys()
        
        if not self.file_pairs:
            self.info_label.configure(text=f"Error: No matching files found in {BASE_DIR}")
        else:
            self.load_current_pair()
            self.play_video()

    def _get_file_pairs(self):
        pairs = []
        skeleton_files = glob.glob(os.path.join(SKELETON_DIR, "*.skeleton"))
        for skel_path in skeleton_files:
            basename = os.path.basename(skel_path).replace('.skeleton', '')
            rgb_path = os.path.join(RGB_DIR, f"{basename}_rgb.avi")
            if os.path.exists(rgb_path):
                pairs.append({'name': basename, 'skel': skel_path, 'rgb': rgb_path})
        return sorted(pairs, key=lambda x: x['name'])

    def _setup_gui(self):
        # Header Info
        self.info_label = ctk.CTkLabel(self, text="", font=("Arial", 20, "bold"))
        self.info_label.pack(pady=(10, 0))
        
        self.action_label = ctk.CTkLabel(self, text="", font=("Arial", 16), text_color="gray")
        self.action_label.pack(pady=(0, 10))
        
        # Video Display (Dynamic sizing label)
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(expand=True, fill="both", padx=20, pady=5)
        
        # Frame Scrubber
        self.slider = ctk.CTkSlider(self, from_=0, to=100, command=self.on_slider_move)
        self.slider.set(0)
        self.slider.pack(fill="x", padx=40, pady=10)
        
        # Controls Frame
        controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        controls_frame.pack(pady=10)
        
        # Navigation
        ctk.CTkButton(controls_frame, text="<< Prev", command=self.prev_file, width=80).pack(side="left", padx=5)
        self.btn_play = ctk.CTkButton(controls_frame, text="Pause", command=self.toggle_play, width=80)
        self.btn_play.pack(side="left", padx=5)
        ctk.CTkButton(controls_frame, text="Next >>", command=self.next_file, width=80).pack(side="left", padx=5)
        
        # Speed Controls
        ctk.CTkLabel(controls_frame, text="Speed:").pack(side="left", padx=(20, 5))
        self.speed_var = ctk.StringVar(value="1.0x")
        speed_opts = ctk.CTkSegmentedButton(controls_frame, values=["0.5x", "1.0x", "2.0x"], 
                                            variable=self.speed_var, command=self.change_speed)
        speed_opts.pack(side="left", padx=5)
        
        # Toggles
        ctk.CTkSwitch(controls_frame, text="Show Skeleton", variable=self.show_skeleton, 
                      command=self.force_update_frame).pack(side="left", padx=(20, 5))

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self.step_frame(-1))
        self.bind("<Right>", lambda e: self.step_frame(1))
        self.bind("<space>", lambda e: self.toggle_play())

    def get_action_name(self, filename):
        try:
            # Extract AXXX from string like 'S032C003P106R002A120'
            action_code = int(filename.split('A')[1][:3])
            return ACTION_MAP.get(action_code, "Unknown Action")
        except:
            return "Unknown Action"

    def parse_skeleton_file(self, filepath):
        frames_data = []
        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if not lines: return []
            
            num_frames = int(lines[0])
            idx = 1
            for _ in range(num_frames):
                if idx >= len(lines): break
                num_bodies = int(lines[idx])
                idx += 1
                bodies = []
                for _ in range(num_bodies):
                    idx += 1 
                    num_joints = int(lines[idx])
                    idx += 1
                    joints = []
                    for _ in range(num_joints):
                        joint_info = lines[idx].split()
                        cx, cy = float(joint_info[5]), float(joint_info[6])
                        joints.append((cx, cy))
                        idx += 1
                    bodies.append(joints)
                frames_data.append(bodies)
        except Exception as e:
            print(f"Error parsing skeleton: {e}")
        return frames_data

    def load_current_pair(self):
        if self.cap is not None:
            self.cap.release()
            
        pair = self.file_pairs[self.current_index]
        action_name = self.get_action_name(pair['name'])
        
        self.info_label.configure(text=f"[{self.current_index + 1}/{len(self.file_pairs)}] {pair['name']}")
        self.action_label.configure(text=f"Action: {action_name.title()}")
        
        self.cap = cv2.VideoCapture(pair['rgb'])
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.configure(to=max(1, self.total_frames - 1))
        
        self.skeleton_data = self.parse_skeleton_file(pair['skel'])
        
        self.current_frame_idx = 0
        self.slider.set(0)
        self.is_playing = True
        self.btn_play.configure(text="Pause")

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.configure(text="Pause" if self.is_playing else "Play")

    def change_speed(self, value):
        if value == "0.5x": self.playback_delay = 66
        elif value == "1.0x": self.playback_delay = 33
        elif value == "2.0x": self.playback_delay = 16

    def step_frame(self, direction):
        """Allows frame-by-frame scrubbing with arrow keys when paused."""
        if self.is_playing:
            self.toggle_play()
            
        new_idx = self.current_frame_idx + direction
        if 0 <= new_idx < self.total_frames:
            self.current_frame_idx = new_idx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            self.slider.set(self.current_frame_idx)
            self.force_update_frame()

    def on_slider_move(self, value):
        self.current_frame_idx = int(value)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.force_update_frame()

    def prev_file(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_pair()

    def next_file(self):
        if self.current_index < len(self.file_pairs) - 1:
            self.current_index += 1
            self.load_current_pair()

    def force_update_frame(self):
        """Draws the current frame immediately (used for scrubbing/toggling)."""
        if self.cap is not None and self.cap.isOpened():
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if ret:
                self._render_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos) # Reset position after read

    def _render_frame(self, frame):
        frame_rgb = frame.copy()
        frame_merged = frame.copy()
        frame_skeleton = np.zeros_like(frame)

        # Draw skeleton overlay if toggled on
        if self.show_skeleton.get() and self.current_frame_idx < len(self.skeleton_data):
            bodies = self.skeleton_data[self.current_frame_idx]
            for body in bodies:
                for cx, cy in body:
                    if not (cx == 0 and cy == 0):
                        cv2.circle(frame_merged, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                        cv2.circle(frame_skeleton, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                for p1, p2 in BONES:
                    if p1 < len(body) and p2 < len(body):
                        x1, y1 = body[p1]
                        x2, y2 = body[p2]
                        if not ((x1==0 and y1==0) or (x2==0 and y2==0)):
                            cv2.line(frame_merged, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.line(frame_skeleton, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        cv2.putText(frame_rgb, "RGB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_skeleton, "Skeleton", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_merged, "Merged", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        h_orig, w_orig, _ = frame_merged.shape
        frame_merged_large = cv2.resize(frame_merged, (w_orig * 2, h_orig * 2))
        right_col = cv2.vconcat([frame_skeleton, frame_rgb])
        combined_frame = cv2.hconcat([frame_merged_large, right_col])

        # Dynamic Aspect Ratio Resizing
        target_w = self.video_label.winfo_width()
        target_h = self.video_label.winfo_height()
        
        if target_w > 10 and target_h > 10: # Only resize if window is initialized
            h, w, _ = combined_frame.shape
            scale = min(target_w/w, target_h/h)
            new_w, new_h = int(w * scale), int(h * scale)
            combined_frame = cv2.resize(combined_frame, (new_w, new_h))

        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(combined_frame)
        imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

    def play_video(self):
        if self.is_playing and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self._render_frame(frame)
                self.slider.set(self.current_frame_idx)
                self.current_frame_idx += 1
            else:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_idx = 0
                
        self.after(self.playback_delay, self.play_video)

if __name__ == "__main__":
    app = NTUViewerApp()
    app.mainloop()
"""
SignBridge — Phase 9: The Final Polish
Flawless Hybrid Routing + Complete UI + Skeleton Tracking

Features:
- "Momentum" Routing: Waits 12 frames before predicting phrases to capture fluid motion
- Blue Skeleton tracking restored for perfect visual alignment
- Stable spelling buffer that ignores micro-tremors
"""

import cv2
import numpy as np
import pickle
import json
import os
import threading
import time
from collections import deque
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras

# ==========================================
# Configuration
# ==========================================
ALPHABET_MODEL_PATH = "models/alphabet_model.h5"
DYNAMIC_MODEL_PATH = "models/signbridge_dynamic_final.h5"
LABEL_MAP_PATH = "models/label_map.json"
ENCODER_FILE = "models/label_encoder.pkl"
MEDIAPIPE_MODEL = "models/hand_landmarker.task"

# Smart Hybrid Router Configuration
WRIST_VELOCITY_THRESHOLD = 0.025  # Perfect balance between still and moving
FRAMES_FOR_VELOCITY = 5
MIN_BUFFER_SIZE = 15
MOTION_PATIENCE = 12  # Frames to wait after movement stops before predicting

# Spelling Buffer Configuration
CONSISTENT_FRAMES_THRESHOLD = 15
COOLDOWN_FRAMES = 30
MAX_WORD_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.70

# Disable GPU to prevent conflicts locally
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class SignBridgeFinalApp:
    def __init__(self):
        print("🚀 Starting SignBridge Final Hybrid Application...")
        
        if not self.initialize_components():
            print("❌ Failed to initialize components")
            return
            
        # Hybrid State
        self.wrist_history = deque(maxlen=FRAMES_FOR_VELOCITY + 1)
        self.frame_buffer = deque(maxlen=30)
        self.is_recording = False
        self.still_frames = 0
        self.current_mode = "SPELLING"
        
        # Spelling State
        self.current_word = ""
        self.completed_sentence = ""
        self.prediction_history = deque(maxlen=CONSISTENT_FRAMES_THRESHOLD)
        self.last_registered_letter = ""
        self.cooldown_counter = 0
        self.current_confidence = 0.0
        self.last_prediction = None
        
        self.run()
        
    def initialize_components(self):
        try:
            print("🤖 Initializing ML components...")
            
            # Load Alphabet Model
            self.alphabet_model = keras.models.load_model(ALPHABET_MODEL_PATH, compile=False)
            with open(ENCODER_FILE, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.alphabet_classes = self.label_encoder.classes_
            print(f"✅ Alphabet model loaded: {len(self.alphabet_classes)} classes")
            
            # Load Dynamic Model
            self.dynamic_model = keras.models.load_model(DYNAMIC_MODEL_PATH, compile=False)
            with open(LABEL_MAP_PATH, 'r') as f:
                label_map = json.load(f)
            self.dynamic_labels = {v: k for k, v in label_map.items()}
            print(f"✅ Dynamic H5 model loaded: {len(self.dynamic_labels)} classes")
            
            # Initialize MediaPipe
            print("🤚 Initializing MediaPipe...")
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode
            
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5
            )
            self.landmarker = HandLandmarker.create_from_options(options)
            
            # Initialize Webcam
            print("📷 Initializing webcam...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            return False

    def extract_hand_landmarks(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                landmarks = []
                for landmark in hand_landmarks:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                return np.array(landmarks, dtype=np.float32), hand_landmarks
            return None, None
        except:
            return None, None

    def calculate_wrist_velocity(self, current_wrist):
        if len(self.wrist_history) < FRAMES_FOR_VELOCITY:
            return 0.0
        old_wrist = self.wrist_history[0]
        distance = np.sqrt(
            (current_wrist[0] - old_wrist[0])**2 +
            (current_wrist[1] - old_wrist[1])**2 +
            (current_wrist[2] - old_wrist[2])**2
        )
        return distance / FRAMES_FOR_VELOCITY

    def preprocess_dynamic(self, frame_sequence):
        processed_sequence = []
        for frame_array in frame_sequence:
            if len(frame_array) < 63:
                frame_array = np.pad(frame_array, (0, 63 - len(frame_array)), 'constant')
            elif len(frame_array) > 63:
                frame_array = frame_array[:63]
            
            wrist = frame_array[:3].copy()
            relative_landmarks = []
            relative_landmarks.extend([0.0, 0.0, 0.0])
            
            for i in range(3, len(frame_array), 3):
                if i + 2 < len(frame_array):
                    relative_landmarks.extend([
                        frame_array[i] - wrist[0],
                        frame_array[i+1] - wrist[1],
                        frame_array[i+2] - wrist[2]
                    ])
                    
            if len(relative_landmarks) < 63:
                relative_landmarks.extend([0.0] * (63 - len(relative_landmarks)))
            
            processed_sequence.append(relative_landmarks)
        return np.array(processed_sequence, dtype=np.float32)

    def route_and_predict(self, landmarks):
        if landmarks is None:
            self.current_mode = "SEARCHING"
            self.prediction_history.clear()
            return

        wrist_pos = landmarks[:3]
        self.wrist_history.append(wrist_pos)
        velocity = self.calculate_wrist_velocity(wrist_pos)
        
        # Always buffer landmarks so we have a full gesture when we need it
        self.frame_buffer.append(landmarks)

        # --- SMART MOMENTUM ROUTING LOGIC ---
        if velocity > WRIST_VELOCITY_THRESHOLD:
            # Hand is moving dynamically
            self.current_mode = "RECORDING PHRASE"
            self.is_recording = True
            self.still_frames = 0
            self.prediction_history.clear() # Interrupt spelling
        else:
            # Hand velocity is low
            if self.is_recording:
                self.still_frames += 1
                # Wait for the hand to be completely still for a moment
                if self.still_frames > MOTION_PATIENCE:
                    if len(self.frame_buffer) >= MIN_BUFFER_SIZE:
                        self.current_mode = "PROCESSING PHRASE"
                        self.trigger_dynamic_prediction()
                    
                    # Reset gesture tracking
                    self.is_recording = False
                    self.frame_buffer.clear()
                    self.still_frames = 0
            else:
                # Hand is resting -> Trigger Alphabet Model
                self.current_mode = "SPELLING"
                self.trigger_static_prediction(landmarks)

    def trigger_dynamic_prediction(self):
        try:
            frame_sequence = np.array(list(self.frame_buffer))
            processed_sequence = self.preprocess_dynamic(frame_sequence)
            
            if len(processed_sequence) < 30:
                padding = np.tile(processed_sequence[-1], (30 - len(processed_sequence), 1))
                processed_sequence = np.vstack([processed_sequence, padding])
            elif len(processed_sequence) > 30:
                indices = np.linspace(0, len(processed_sequence) - 1, 30, dtype=int)
                processed_sequence = processed_sequence[indices]
                
            input_data = np.expand_dims(processed_sequence, axis=0)
            prediction = self.dynamic_model.predict(input_data, verbose=0)
            
            idx = np.argmax(prediction[0])
            conf = prediction[0][idx]
            
            if conf > 0.75:
                phrase = self.dynamic_labels[idx].upper()
                self.last_prediction = phrase
                self.current_confidence = conf
                
                # Auto-append phrase to sentence
                if self.current_word:
                    self.completed_sentence += self.current_word + " "
                    self.current_word = ""
                self.completed_sentence += f"[{phrase}] "
                self.speak_word_async(phrase)
                
        except Exception as e:
            print(f"❌ Dynamic Error: {e}")

    def trigger_static_prediction(self, landmarks):
        try:
            landmarks_input = landmarks.reshape(1, -1)
            prediction = self.alphabet_model.predict(landmarks_input, verbose=0)
            
            idx = np.argmax(prediction[0])
            conf = prediction[0][idx]
            letter = self.alphabet_classes[idx]
            
            self.last_prediction = letter
            self.current_confidence = conf
            
            if conf >= CONFIDENCE_THRESHOLD:
                self.prediction_history.append(letter)
                if len(self.prediction_history) >= CONSISTENT_FRAMES_THRESHOLD:
                    if all(p == letter for p in self.prediction_history):
                        if self.cooldown_counter <= 0 and letter != self.last_registered_letter:
                            self.register_letter(letter)
                            self.cooldown_counter = COOLDOWN_FRAMES
                            self.prediction_history.clear()
            else:
                self.prediction_history.clear()
                
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                
        except Exception as e:
            pass

    def register_letter(self, letter):
        if letter == "space":
            if self.current_word:
                self.completed_sentence += self.current_word + " "
                self.speak_word_async(self.current_word)
                self.current_word = ""
                self.last_registered_letter = letter
        elif letter == "del":
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self.last_registered_letter = letter
        else:
            if len(self.current_word) < MAX_WORD_LENGTH:
                self.current_word += letter
                self.last_registered_letter = letter

    def speak_word_async(self, word):
        if word.strip():
            threading.Thread(target=lambda: os.system(f"say '{word}' &"), daemon=True).start()

    def draw_stable_ui(self, frame):
        height, width = frame.shape[:2]
        panel_width = 420
        panel_x = width - panel_width - 15
        panel_y = 15
        panel_height = height - 30
        
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (25, 25, 35), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (100, 100, 255), 2)
        
        cv2.putText(frame, "🎯 SignBridge Final", (panel_x + 20, panel_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        mode_color = (0, 200, 100) if self.current_mode == "SPELLING" else (0, 100, 255)
        cv2.putText(frame, f"MODE: {self.current_mode}", (panel_x + 20, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        cv2.line(frame, (panel_x + 15, panel_y + 85), (panel_x + panel_width - 15, panel_y + 85), (100, 100, 255), 2)
        
        y_offset = panel_y + 110
        cv2.putText(frame, "CURRENT WORD", (panel_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        word_box_y = y_offset + 10
        cv2.rectangle(frame, (panel_x + 15, word_box_y), (panel_x + panel_width - 15, word_box_y + 40), (50, 50, 80), 1)
        cv2.putText(frame, self.current_word, (panel_x + 25, word_box_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        y_offset = word_box_y + 60
        cv2.putText(frame, "SENTENCE", (panel_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        sentence_box_y = y_offset + 10
        cv2.rectangle(frame, (panel_x + 15, sentence_box_y), (panel_x + panel_width - 15, sentence_box_y + 80), (50, 50, 80), 1)
        
        full_text = self.completed_sentence
        words = full_text.split()
        lines, current_line = [], ""
        for word in words:
            if len(current_line + word) < 35: current_line += word + " "
            else: lines.append(current_line); current_line = word + " "
        lines.append(current_line)
        
        for i, line in enumerate(lines[-3:]): 
            cv2.putText(frame, line.strip(), (panel_x + 25, sentence_box_y + 20 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        # --- NEW: CONTROLS SECTION ---
        controls_y = sentence_box_y + 110
        cv2.putText(frame, "CONTROLS", (panel_x + 20, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        cv2.line(frame, (panel_x + 15, controls_y + 10), (panel_x + panel_width - 15, controls_y + 10), (50, 50, 80), 1)
        
        controls = [
            "C - Clear all text",
            "P - Speak out sentence",
            "T - Save transcript to text file",
            "Q - Quit Application"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (panel_x + 20, controls_y + 35 + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
        return frame

    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Restored the blue skeleton connections for perfect visual alignment."""
        if not hand_landmarks:
            return frame

        h, w = frame.shape[:2]
        
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
        # Draw the blue skeleton lines first
        for start_idx, end_idx in HAND_CONNECTIONS:
            start_pt = hand_landmarks[start_idx]
            end_pt = hand_landmarks[end_idx]
            cv2.line(frame, 
                     (int(start_pt.x * w), int(start_pt.y * h)), 
                     (int(end_pt.x * w), int(end_pt.y * h)), 
                     (255, 100, 100), 3)

        # Draw the green tracking dots over the lines
        for landmark in hand_landmarks:
            cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 6, (100, 255, 100), -1)
            cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 6, (0, 0, 0), 2)
            
        return frame

    def run(self):
        print("🎉 Final System Ready!")
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            landmarks, hand_landmarks = self.extract_hand_landmarks(frame)
            
            self.route_and_predict(landmarks)
            
            frame = self.draw_hand_landmarks(frame, hand_landmarks)
            frame = self.draw_stable_ui(frame)
            
            if self.last_prediction:
                cv2.putText(frame, f"{self.last_prediction} ({self.current_confidence*100:.0f}%)", 
                           (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 255, 50), 3)
            
            cv2.imshow("SignBridge Final", frame)
            
            # --- UPDATED KEYBOARD CONTROLS ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27: 
                break  # Quit
                
            elif key == ord('c'):
                # Clear Text with visual feedback
                self.current_word = ""
                self.completed_sentence = ""
                self.last_prediction = "🧹 CLEARED"
                self.current_confidence = 1.0
                
            elif key == ord('p'):
                # Speak the entire sentence out loud
                full_text = self.completed_sentence.strip()
                if full_text:
                    self.speak_word_async(full_text)
                    self.last_prediction = "🔊 SPEAKING"
                    self.current_confidence = 1.0
                    
            elif key == ord('t'):
                # NEW FEATURE: Export to Text File
                full_text = self.completed_sentence.strip()
                if full_text:
                    with open("SignBridge_Transcript.txt", "a") as f:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{timestamp}] {full_text}\n")
                    self.last_prediction = "💾 SAVED TO FILE"
                    self.current_confidence = 1.0
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SignBridgeFinalApp()
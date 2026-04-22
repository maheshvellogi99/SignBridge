"""
SignBridge — Phase 4: Professional Desktop GUI (OpenCV)
Stable Desktop Application using OpenCV GUI instead of CustomTkinter
"""

import cv2
import numpy as np
import pickle
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
KERAS_MODEL_PATH = "models/alphabet_model.h5"
ENCODER_FILE = "models/label_encoder.pkl"
MODEL_PATH = "models/hand_landmarker.task"

# Spelling Buffer Configuration
CONSISTENT_FRAMES_THRESHOLD = 15
COOLDOWN_FRAMES = 30
MAX_WORD_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.75

# Disable GPU to prevent conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class SignBridgeOpenCVApp:
    def __init__(self):
        """Initialize the OpenCV-based desktop application."""
        print("🚀 Starting SignBridge Desktop Application (OpenCV)...")
        
        # Initialize components
        self.initialize_components()
        
        # Start main loop
        self.run()
        
    def initialize_components(self):
        """Initialize all ML components."""
        try:
            print("🤖 Initializing ML components...")
            
            # Load Keras model
            print("📂 Loading Keras model...")
            self.model = keras.models.load_model(KERAS_MODEL_PATH, compile=False)
            
            # Load label encoder
            with open(ENCODER_FILE, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.class_names = self.label_encoder.classes_
            print(f"✅ Model loaded: {len(self.class_names)} classes")
            
            # Initialize MediaPipe
            print("🤚 Initializing MediaPipe...")
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode
            
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5
            )
            
            self.landmarker = HandLandmarker.create_from_options(options)
            print("✅ MediaPipe initialized")
            
            # Initialize webcam
            print("📷 Initializing webcam...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Error: Could not open webcam")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("✅ Webcam initialized")
            
            # Initialize spelling system
            self.setup_spelling_system()
            
            # Performance tracking
            self.fps_counter = 0
            self.fps_start_time = time.time()
            self.current_fps = 0
            
            print("✅ All components initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            exit(1)
    
    def setup_spelling_system(self):
        """Initialize spelling buffer and TTS."""
        self.current_word = ""
        self.completed_sentence = ""
        self.prediction_history = deque(maxlen=CONSISTENT_FRAMES_THRESHOLD)
        self.last_registered_letter = ""
        self.cooldown_counter = 0
        self.current_confidence = 0.0
        
        print("✅ Spelling system initialized")
    
    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks from frame."""
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
            
        except Exception as e:
            return None, None
    
    def predict_letter(self, landmarks):
        """Predict ASL letter using Keras model."""
        try:
            # Prepare input
            landmarks_input = landmarks.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(landmarks_input, verbose=0)
            
            # Get prediction
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]
            predicted_letter = self.class_names[predicted_class_idx]
            
            return predicted_letter, confidence
            
        except Exception as e:
            return None, 0.0
    
    def process_prediction(self, prediction, confidence):
        """Process prediction and update spelling buffer."""
        if prediction is None or confidence < CONFIDENCE_THRESHOLD:
            self.prediction_history.clear()
            return
        
        self.prediction_history.append(prediction)
        
        if len(self.prediction_history) >= CONSISTENT_FRAMES_THRESHOLD:
            if all(p == prediction for p in self.prediction_history):
                if self.cooldown_counter <= 0 and prediction != self.last_registered_letter:
                    self.register_letter(prediction)
                    self.cooldown_counter = COOLDOWN_FRAMES
                    self.prediction_history.clear()
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
    
    def register_letter(self, letter):
        """Register a letter and handle special cases."""
        if letter == "space":
            if self.current_word:
                word_to_speak = self.current_word
                self.completed_sentence += self.current_word + " "
                self.speak_word_async(word_to_speak)
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
        """Speak a word using macOS TTS."""
        if not word.strip():
            return
        
        def speak_worker():
            try:
                os.system(f"say '{word}' &")
            except Exception as e:
                pass
        
        threading.Thread(target=speak_worker, daemon=True).start()
    
    def speak_sentence(self):
        """Speak the complete sentence."""
        full_text = self.completed_sentence + self.current_word
        self.speak_word_async(full_text)
    
    def clear_all(self):
        """Clear all text."""
        self.current_word = ""
        self.completed_sentence = ""
        self.prediction_history.clear()
        self.last_registered_letter = ""
        self.cooldown_counter = 0
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on frame."""
        if hand_landmarks is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # Hand connections
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            start_x = int(start_point.x * width)
            start_y = int(start_point.y * height)
            end_x = int(end_point.x * width)
            end_y = int(end_point.y * height)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
        
        return frame
    
    def draw_ui_overlay(self, frame):
        """Draw professional UI overlay on frame."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for UI panel
        overlay = frame.copy()
        
        # Define panel dimensions
        panel_width = 400
        panel_x = width - panel_width - 20
        panel_y = 20
        panel_height = height - 40
        
        # Ensure panel coordinates are within frame bounds
        if panel_x < 0:
            panel_x = 10
            panel_width = width - 20
        if panel_y + panel_height > height:
            panel_height = height - panel_y - 10
        
        # Draw semi-transparent panel background
        panel_roi = frame[panel_y:panel_y + panel_height, panel_x:panel_x + panel_width]
        if panel_roi.size > 0:  # Check if ROI is valid
            overlay_panel = cv2.addWeighted(panel_roi, 0.3, np.full_like(panel_roi, (20, 20, 30), dtype=np.uint8), 0.7, 0)
            frame[panel_y:panel_y + panel_height, panel_x:panel_x + panel_width] = overlay_panel
        
        # Draw title
        cv2.putText(frame, "🎯 SignBridge Dashboard", (panel_x + 20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw current word
        cv2.putText(frame, "Current Word:", (panel_x + 20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, self.current_word, (panel_x + 20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 144, 255), 2)
        
        # Draw completed sentence
        cv2.putText(frame, "Sentence:", (panel_x + 20, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Split sentence into lines if too long
        sentence_text = self.completed_sentence + self.current_word
        if len(sentence_text) > 40:
            words = sentence_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 40:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            lines.append(current_line)
        else:
            lines = [sentence_text]
        
        for i, line in enumerate(lines[:3]):  # Max 3 lines
            cv2.putText(frame, line.strip(), (panel_x + 20, 260 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw confidence meter
        cv2.putText(frame, f"Confidence: {int(self.current_confidence * 100)}%", 
                   (panel_x + 20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Confidence bar
        bar_width = 300
        bar_height = 20
        bar_x = panel_x + 20
        bar_y = 400
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Confidence fill
        fill_width = int(bar_width * self.current_confidence)
        if self.current_confidence > 0.75:
            color = (0, 255, 0)  # Green
        elif self.current_confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
            
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                     color, -1)
        
        # Draw controls
        cv2.putText(frame, "Controls:", (panel_x + 20, 480),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, "S - Speak Sentence", (panel_x + 20, 510),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "C - Clear Text", (panel_x + 20, 540),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Q - Quit", (panel_x + 20, 570),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps}", (panel_x + 20, height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Main application loop."""
        print("🎉 SignBridge Desktop Application is ready!")
        print("Controls: S - Speak | C - Clear | Q - Quit")
        
        while True:
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks and predict
                landmarks, hand_landmarks = self.extract_hand_landmarks(frame)
                prediction = None
                confidence = 0.0
                
                if landmarks is not None:
                    prediction, confidence = self.predict_letter(landmarks)
                    self.process_prediction(prediction, confidence)
                    self.current_confidence = confidence
                
                # Draw landmarks
                frame = self.draw_hand_landmarks(frame, hand_landmarks)
                
                # Add prediction overlay
                if prediction and confidence >= CONFIDENCE_THRESHOLD:
                    cv2.putText(frame, f"{prediction} ({confidence*100:.1f}%)", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw UI overlay
                frame = self.draw_ui_overlay(frame)
                
                # Display frame
                cv2.imshow("SignBridge - Professional ASL Recognition", frame)
                
                # Update FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.speak_sentence()
                elif key == ord('c'):
                    self.clear_all()
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        
        # Cleanup
        print("👋 Shutting down...")
        self.cap.release()
        if self.landmarker:
            self.landmarker.close()
        cv2.destroyAllWindows()
        print("✅ Application closed")

def main():
    """Main entry point."""
    try:
        app = SignBridgeOpenCVApp()
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()

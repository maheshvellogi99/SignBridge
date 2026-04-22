"""
SignBridge — Phase 4: Professional Desktop GUI (Stable)
Final stable version with no flicker and optimal performance

Key fixes:
- Eliminated UI flickering by updating every frame
- Optimized UI drawing for better performance
- Maintained frame skipping for prediction only
- Streamlined rendering pipeline
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

# Performance Configuration
PREDICTION_SKIP_FRAMES = 3  # Predict every 3rd frame for better FPS

# Disable GPU to prevent conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class SignBridgeStableApp:
    def __init__(self):
        """Initialize the stable desktop application."""
        print("🚀 Starting SignBridge Stable Desktop Application...")
        
        # Frame counters for optimization
        self.frame_count = 0
        self.prediction_frame_count = 0
        
        # Initialize components
        if not self.initialize_components():
            print("❌ Failed to initialize components")
            return
        
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
                return False
            
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
            
            # Optimization: Pre-allocate arrays
            self.last_landmarks = None
            self.last_prediction = None
            self.last_confidence = 0.0
            
            print("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            return False
    
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
    
    def speak_paragraph(self):
        """Speak the complete paragraph (current word + completed sentence)."""
        full_text = self.completed_sentence + self.current_word
        if full_text.strip():
            print(f"🗣️ Speaking paragraph: '{full_text}'")
            # Use slower speech rate for better clarity
            threading.Thread(target=lambda: os.system(f"say -r 150 '{full_text}' &"), daemon=True).start()
    
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
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 100, 100), 3)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 6, (100, 255, 100), -1)
            cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)
        
        return frame
    
    def draw_stable_ui(self, frame):
        """Draw stable UI overlay on frame (no flicker)."""
        height, width = frame.shape[:2]
        
        # UI Panel dimensions
        panel_width = 420
        panel_x = width - panel_width - 15
        panel_y = 15
        panel_height = height - 30
        
        # Draw panel background
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (25, 25, 35), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                     (100, 100, 255), 2)
        
        # Header
        cv2.putText(frame, "🎯 SignBridge Pro", (panel_x + 20, panel_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(frame, (panel_x + 15, panel_y + 50), (panel_x + panel_width - 15, panel_y + 50),
                (100, 100, 255), 2)
        
        # Current Word Section
        y_offset = panel_y + 90
        cv2.putText(frame, "CURRENT WORD", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        word_box_y = y_offset + 10
        cv2.rectangle(frame, (panel_x + 15, word_box_y), (panel_x + panel_width - 15, word_box_y + 40),
                     (50, 50, 80), 1)
        cv2.putText(frame, self.current_word, (panel_x + 25, word_box_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        # Sentence Section
        y_offset = word_box_y + 60
        cv2.putText(frame, "SENTENCE", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        sentence_box_y = y_offset + 10
        sentence_box_height = 80
        cv2.rectangle(frame, (panel_x + 15, sentence_box_y), 
                     (panel_x + panel_width - 15, sentence_box_y + sentence_box_height),
                     (50, 50, 80), 1)
        
        # Display sentence with word wrap
        full_text = self.completed_sentence + self.current_word
        if len(full_text) > 35:
            words = full_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 35:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            lines.append(current_line)
        else:
            lines = [full_text]
        
        for i, line in enumerate(lines[:3]):
            cv2.putText(frame, line.strip(), (panel_x + 25, sentence_box_y + 20 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Confidence Meter
        y_offset = sentence_box_y + sentence_box_height + 20
        cv2.putText(frame, f"CONFIDENCE: {int(self.current_confidence * 100)}%", 
                   (panel_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        bar_width = panel_width - 50
        bar_height = 15
        bar_x = panel_x + 20
        bar_y = y_offset + 10
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (40, 40, 60), -1)
        
        # Confidence fill
        fill_width = int(bar_width * self.current_confidence)
        if fill_width > 0:
            if self.current_confidence > 0.75:
                color = (50, 255, 50)  # Green
            elif self.current_confidence > 0.5:
                color = (255, 255, 50)  # Yellow
            else:
                color = (255, 50, 50)  # Red
                
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                         color, -1)
        
        # Status indicator
        y_offset = bar_y + bar_height + 20
        if self.current_confidence >= CONFIDENCE_THRESHOLD:
            status_text = "✓ DETECTING"
            status_color = (50, 255, 50)
        else:
            status_text = "○ READY"
            status_color = (255, 255, 255)
        
        cv2.putText(frame, status_text, (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Controls section
        y_offset = y_offset + 40
        cv2.putText(frame, "CONTROLS", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        
        controls = [
            "S - Speak Sentence",
            "P - Speak Paragraph", 
            "C - Clear Text",
            "Q - Quit Application"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (panel_x + 25, y_offset + 25 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Performance metrics
        y_offset = y_offset + 110
        cv2.putText(frame, f"FPS: {self.current_fps}", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Main application loop (stable)."""
        print("🎉 SignBridge Stable Desktop Application is ready!")
        print("=" * 60)
        print("🎮 KEYBOARD CONTROLS:")
        print("  S - Speak the complete sentence")
        print("  P - Speak the complete paragraph (slower)")
        print("  C - Clear all text")
        print("  Q - Quit application")
        print("=" * 60)
        print("📷 Camera is active. Show clear hand signs for recognition.")
        print("✨ Stable interface with no flicker!")
        
        while True:
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Camera frame lost, attempting to reconnect...")
                    time.sleep(0.1)
                    continue
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Increment frame counter
                self.frame_count += 1
                
                # Extract landmarks (every frame for smooth video)
                landmarks, hand_landmarks = self.extract_hand_landmarks(frame)
                
                # Prediction optimization: Only predict every Nth frame
                prediction = None
                confidence = 0.0
                
                if landmarks is not None:
                    self.prediction_frame_count += 1
                    if self.prediction_frame_count % PREDICTION_SKIP_FRAMES == 0:
                        prediction, confidence = self.predict_letter(landmarks)
                        self.process_prediction(prediction, confidence)
                        self.current_confidence = confidence
                        self.last_prediction = prediction
                        self.last_confidence = confidence
                    else:
                        # Use last known values for display
                        prediction = self.last_prediction
                        confidence = self.last_confidence
                
                # Draw landmarks (every frame for smooth visual)
                frame = self.draw_hand_landmarks(frame, hand_landmarks)
                
                # Add prediction overlay on main video area
                if prediction and confidence >= CONFIDENCE_THRESHOLD:
                    cv2.putText(frame, f"{prediction} ({confidence*100:.1f}%)", 
                               (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 255, 50), 3)
                    cv2.putText(frame, f"{prediction} ({confidence*100:.1f}%)", 
                               (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
                
                # Draw stable UI overlay (every frame to prevent flicker)
                frame = self.draw_stable_ui(frame)
                
                # Display frame
                cv2.imshow("SignBridge Stable - ASL Recognition System", frame)
                
                # Update FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s'):
                    self.speak_sentence()
                    print(f"🗣️ Speaking: {self.completed_sentence + self.current_word}")
                elif key == ord('p'):
                    self.speak_paragraph()
                elif key == ord('c'):
                    self.clear_all()
                    print("🧹 Text cleared")
                
            except Exception as e:
                print(f"⚠️ Error in main loop: {e}")
                continue
        
        # Cleanup
        print("\n👋 Shutting down SignBridge Stable...")
        self.cap.release()
        if self.landmarker:
            self.landmarker.close()
        cv2.destroyAllWindows()
        print("✅ Application closed successfully")

def main():
    """Main entry point."""
    try:
        app = SignBridgeStableApp()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()

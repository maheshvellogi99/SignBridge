"""
SignBridge — Phase 3: Dynamic Spelling & Text-to-Speech
Advanced Real-Time ASL Recognition with Spelling Buffer and TTS

This script enhances the real-time ASL recognition with:
- Dynamic spelling buffer with frame consistency checking
- Word completion and sentence building
- Non-blocking macOS Text-to-Speech
- Enhanced UI showing current word and completed sentence
- Smart cooldown system to prevent letter spam

Features:
- Frame counter: 15-20 consistent frames needed to register a letter
- Cooldown system: Prevents duplicate letter registration
- Space handling: Completes words and triggers TTS
- Delete handling: Removes last letter from current word
- Threading: Non-blocking TTS using macOS native 'say' command

Controls:
- Press 'q' to quit gracefully
- Press 'c' to clear current word and sentence
- Press 's' to save current frame (optional)
"""

import cv2
import numpy as np
import pickle
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import time
import os
import threading
from collections import deque

# ==========================================
# Configuration and Paths
# ==========================================
MODEL_FILE = "models/alphabet_model.h5"
ENCODER_FILE = "models/label_encoder.pkl"
MODEL_PATH = "models/hand_landmarker.task"

# Spelling Buffer Configuration
CONSISTENT_FRAMES_THRESHOLD = 15  # Frames needed to register a letter
COOLDOWN_FRAMES = 30  # Frames to wait before registering same letter again
MAX_WORD_LENGTH = 20  # Maximum word length

# UI Configuration
CONFIDENCE_THRESHOLD = 0.75
STATUS_BAR_HEIGHT = 160  # Increased for more text
STATUS_BAR_COLOR = (40, 44, 52)  # Dark blue-gray
TEXT_COLOR = (255, 255, 255)  # White
PREDICTION_COLOR = (76, 175, 80)  # Green
CONFIDENCE_COLOR = (255, 193, 7)  # Amber
WORD_COLOR = (33, 150, 243)  # Blue
SENTENCE_COLOR = (156, 39, 176)  # Purple
LANDMARK_COLOR = (0, 255, 0)  # Bright green
CONNECTION_COLOR = (0, 0, 255)  # Blue

# MediaPipe Hands Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

class SignBridgeTTSSpeller:
    def __init__(self):
        """Initialize the advanced ASL recognition system with TTS."""
        print("="*60)
        print("  SignBridge — Dynamic Spelling & TTS")
        print("="*60)
        
        # Load model and encoder
        self.load_model()
        
        # Initialize MediaPipe
        self.setup_mediapipe()
        
        # Initialize webcam
        self.setup_camera()
        
        # Initialize UI elements
        self.setup_ui()
        
        # Initialize spelling system
        self.setup_spelling_system()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def load_model(self):
        """Load the trained model and label encoder."""
        print("📂 Loading model artifacts...")
        
        try:
            # Load Keras model
            self.model = keras.models.load_model(MODEL_FILE)
            print(f"✅ Model loaded: {MODEL_FILE}")
            
            # Load label encoder
            with open(ENCODER_FILE, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder loaded: {ENCODER_FILE}")
            
            # Get class names
            self.class_names = self.label_encoder.classes_
            print(f"📝 Available classes: {len(self.class_names)} letters")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(1)
    
    def setup_mediapipe(self):
        """Initialize MediaPipe HandLandmarker."""
        print("🤲 Initializing MediaPipe...")
        
        try:
            # Create HandLandmarker options
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5
            )
            
            # Create landmarker
            self.landmarker = HandLandmarker.create_from_options(options)
            print("✅ MediaPipe HandLandmarker initialized")
            
        except Exception as e:
            print(f"❌ Error initializing MediaPipe: {e}")
            exit(1)
    
    def setup_camera(self):
        """Initialize webcam capture."""
        print("📷 Initializing webcam...")
        
        try:
            self.cap = cv2.VideoCapture(0)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                print("❌ Error: Could not open webcam")
                exit(1)
            
            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual camera settings
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Webcam initialized: {width}x{height} @ {fps} FPS")
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            exit(1)
    
    def setup_ui(self):
        """Setup UI elements and fonts."""
        print("🎨 Setting up UI elements...")
        
        # Initialize OpenCV fonts
        self.font_large = cv2.FONT_HERSHEY_SIMPLEX
        self.font_medium = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        
        # Font scales
        self.font_scale_large = 2.0
        self.font_scale_medium = 1.2
        self.font_scale_small = 0.8
        
        # Text thickness
        self.thickness_large = 3
        self.thickness_medium = 2
        self.thickness_small = 1
        
        print("✅ UI elements ready")
    
    def setup_spelling_system(self):
        """Initialize the spelling buffer and TTS system."""
        print("🔤 Initializing spelling system...")
        
        # Spelling buffer
        self.current_word = ""
        self.completed_sentence = ""
        
        # Frame consistency tracking
        self.prediction_history = deque(maxlen=CONSISTENT_FRAMES_THRESHOLD)
        self.last_registered_letter = ""
        self.cooldown_counter = 0
        
        # TTS system
        self.tts_queue = []
        self.tts_busy = False
        
        print("✅ Spelling system ready")
    
    def speak_word_async(self, word):
        """Speak a word using macOS TTS without blocking the main thread."""
        if not word.strip():
            return
        
        def speak_worker():
            try:
                # Use macOS native 'say' command
                os.system(f"say '{word}' &")
            except Exception as e:
                print(f"TTS Error: {e}")
        
        # Run TTS in a separate thread to avoid blocking
        tts_thread = threading.Thread(target=speak_worker, daemon=True)
        tts_thread.start()
    
    def process_prediction(self, prediction, confidence):
        """Process prediction and update spelling buffer."""
        if prediction is None or confidence < CONFIDENCE_THRESHOLD:
            # Clear prediction history when no confident prediction
            self.prediction_history.clear()
            return
        
        # Add to prediction history
        self.prediction_history.append(prediction)
        
        # Check if we have enough consistent predictions
        if len(self.prediction_history) >= CONSISTENT_FRAMES_THRESHOLD:
            # Check if all recent predictions are the same
            if all(p == prediction for p in self.prediction_history):
                # Check cooldown
                if self.cooldown_counter <= 0 and prediction != self.last_registered_letter:
                    self.register_letter(prediction)
                    self.cooldown_counter = COOLDOWN_FRAMES
                    self.prediction_history.clear()
        
        # Update cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
    
    def register_letter(self, letter):
        """Register a letter and handle special cases."""
        print(f"🔤 Registering letter: {letter}")
        
        if letter == "space":
            # Complete current word and speak it
            if self.current_word:
                word_to_speak = self.current_word
                self.completed_sentence += self.current_word + " "
                print(f"🗣️ Speaking: '{word_to_speak}'")
                self.speak_word_async(word_to_speak)
                self.current_word = ""
                self.last_registered_letter = letter
                
        elif letter == "del":
            # Delete last letter from current word
            if self.current_word:
                self.current_word = self.current_word[:-1]
                print(f"🗑️ Deleted letter. Current word: '{self.current_word}'")
                self.last_registered_letter = letter
                
        else:
            # Add letter to current word
            if len(self.current_word) < MAX_WORD_LENGTH:
                self.current_word += letter
                print(f"➕ Added letter '{letter}'. Current word: '{self.current_word}'")
                self.last_registered_letter = letter
    
    def clear_all(self):
        """Clear current word and completed sentence."""
        self.current_word = ""
        self.completed_sentence = ""
        self.prediction_history.clear()
        self.last_registered_letter = ""
        self.cooldown_counter = 0
        print("🧹 Cleared all text")
    
    def extract_hand_landmarks(self, frame):
        """
        Extract 63 hand landmarks from frame using MediaPipe.
        Returns landmarks array or None if no hand detected.
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process image
            result = self.landmarker.detect(mp_image)
            
            # Check if hand detected
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                
                # Extract 63 features (21 landmarks x 3 coordinates)
                landmarks = []
                for landmark in hand_landmarks:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                return np.array(landmarks, dtype=np.float32), hand_landmarks
            
            return None, None
            
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None, None
    
    def predict_letter(self, landmarks):
        """
        Predict ASL letter from hand landmarks.
        Returns predicted letter and confidence score.
        """
        try:
            # Reshape for model input
            landmarks_input = landmarks.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(landmarks_input, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]
            predicted_letter = self.class_names[predicted_class_idx]
            
            return predicted_letter, confidence
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, 0.0
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks and connections on frame using HandLandmarker results."""
        if hand_landmarks is None:
            return frame
        
        height, width = frame.shape[:2]
        
        # Define hand connections (same as MediaPipe Hands)
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),     # Index finger
            (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
            (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (0, 17)                              # Palm
        ]
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            # Convert normalized coordinates to pixel coordinates
            start_x = int(start_point.x * width)
            start_y = int(start_point.y * height)
            end_x = int(end_point.x * width)
            end_y = int(end_point.y * height)
            
            # Draw connection line
            cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                    CONNECTION_COLOR, 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            # Draw landmark point
            cv2.circle(frame, (x, y), 5, LANDMARK_COLOR, -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)  # Black border
        
        return frame
    
    def draw_status_bar(self, frame, prediction, confidence):
        """Draw professional status bar with spelling information."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - STATUS_BAR_HEIGHT), 
                     (width, height), STATUS_BAR_COLOR, -1)
        
        # Blend overlay with original frame
        alpha = 0.9
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        y_offset = height - STATUS_BAR_HEIGHT + 25
        
        # Draw title
        title_text = "SignBridge - Dynamic Spelling & TTS"
        title_size = cv2.getTextSize(title_text, self.font_medium, 
                                   self.font_scale_medium, self.thickness_medium)[0]
        title_x = (width - title_size[0]) // 2
        
        cv2.putText(frame, title_text, (title_x, y_offset),
                   self.font_medium, self.font_scale_medium, TEXT_COLOR, 
                   self.thickness_medium)
        
        y_offset += 35
        
        # Draw completed sentence
        sentence_text = f"Sentence: {self.completed_sentence}"
        cv2.putText(frame, sentence_text, (50, y_offset),
                   self.font_medium, self.font_scale_medium, SENTENCE_COLOR, 
                   self.thickness_medium)
        
        y_offset += 30
        
        # Draw current word
        word_text = f"Current Word: {self.current_word}"
        cv2.putText(frame, word_text, (50, y_offset),
                   self.font_medium, self.font_scale_medium, WORD_COLOR, 
                   self.thickness_medium)
        
        y_offset += 30
        
        # Draw prediction info
        if prediction and confidence >= CONFIDENCE_THRESHOLD:
            # Predicted letter
            pred_text = f"Letter: {prediction}"
            pred_size = cv2.getTextSize(pred_text, self.font_large, 
                                       self.font_scale_large, self.thickness_large)[0]
            
            cv2.putText(frame, pred_text, (50, y_offset),
                       self.font_large, self.font_scale_large, PREDICTION_COLOR, 
                       self.thickness_large)
            
            # Confidence score
            conf_text = f"Confidence: {confidence*100:.1f}%"
            conf_x = 50 + pred_size[0] + 30
            
            cv2.putText(frame, conf_text, (conf_x, y_offset),
                       self.font_medium, self.font_scale_medium, CONFIDENCE_COLOR, 
                       self.thickness_medium)
            
            # Show consistency progress
            consistency_text = f"Consistency: {len(self.prediction_history)}/{CONSISTENT_FRAMES_THRESHOLD}"
            cv2.putText(frame, consistency_text, (conf_x + 200, y_offset),
                       self.font_small, self.font_scale_small, TEXT_COLOR, 
                       self.thickness_small)
        else:
            # Show "No confident prediction" message
            no_pred_text = "Show a clear hand sign (confidence > 75%)"
            cv2.putText(frame, no_pred_text, (50, y_offset),
                       self.font_medium, self.font_scale_medium, TEXT_COLOR, 
                       self.thickness_medium)
        
        # Draw FPS counter
        fps_text = f"FPS: {self.current_fps}"
        cv2.putText(frame, fps_text, (width - 150, height - STATUS_BAR_HEIGHT + 30),
                   self.font_small, self.font_scale_small, TEXT_COLOR, 
                   self.thickness_small)
        
        # Draw controls info
        controls_text = "Controls: 'q'=quit, 'c'=clear"
        cv2.putText(frame, controls_text, (width - 250, height - 30),
                   self.font_small, self.font_scale_small, TEXT_COLOR, 
                   self.thickness_small)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self):
        """Main processing loop for real-time inference with TTS."""
        print("\n🚀 Starting dynamic spelling with TTS...")
        print("   Show hand signs to spell words")
        print("   Use 'space' sign to complete words and hear them spoken")
        print("   Use 'del' sign to delete last letter")
        print("   Press 'q' to quit, 'c' to clear all text")
        print("="*60)
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract hand landmarks
                landmarks, hand_landmarks = self.extract_hand_landmarks(frame)
                
                # Initialize prediction variables
                prediction = None
                confidence = 0.0
                
                # Make prediction if hand detected
                if landmarks is not None:
                    prediction, confidence = self.predict_letter(landmarks)
                    self.process_prediction(prediction, confidence)
                
                # Draw hand landmarks
                frame = self.draw_hand_landmarks(frame, hand_landmarks)
                
                # Draw status bar with prediction and spelling info
                frame = self.draw_status_bar(frame, prediction, confidence)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('SignBridge - Dynamic Spelling & TTS', frame)
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('c'):
                    self.clear_all()
                elif key == ord('s'):
                    # Save current frame (optional feature)
                    timestamp = int(time.time())
                    filename = f"signbridge_tts_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Frame saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.landmarker:
            self.landmarker.close()
        
        print("✅ Cleanup complete")

def main():
    """Main entry point."""
    try:
        # Create and run the TTS speller system
        app = SignBridgeTTSSpeller()
        app.run()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        exit(1)

if __name__ == "__main__":
    main()

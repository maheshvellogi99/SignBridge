"""
SignBridge — Phase 6: Hybrid Inference System (v2)
Real-time desktop application with static alphabet and dynamic phrase recognition

Features:
- Motion-based routing between static and dynamic recognition
- Wrist velocity detection for phrase recording
- Exact Kaggle preprocessing replication
- H5 dynamic model integration (avoids TFLite Flex delegate issues)
- Professional UI with mode indicators
"""

import cv2
import numpy as np
import json
from collections import deque
import time
import threading
import os

# MediaPipe for hand detection
import mediapipe as mp

# TensorFlow for both models
import tensorflow as tf

# ==========================================
# Configuration
# ==========================================
ALPHABET_MODEL_PATH = "models/alphabet_model.h5"
DYNAMIC_MODEL_PATH = "models/signbridge_dynamic_final.h5"
LABEL_MAP_PATH = "models/label_map.json"
ENCODER_FILE = "models/label_encoder.pkl"
MEDIAPIPE_MODEL = "models/hand_landmarker.task"

# Motion detection thresholds
WRIST_VELOCITY_THRESHOLD = 0.02  # Minimum velocity to consider "moving"
FRAMES_FOR_VELOCITY = 5  # Number of frames back to calculate velocity
MIN_BUFFER_SIZE = 15  # Minimum frames before attempting prediction

# UI Configuration
WINDOW_NAME = "SignBridge Hybrid - ASL Recognition System"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

class HybridInferenceApp:
    def __init__(self):
        """Initialize the hybrid inference application."""
        print("🚀 Starting SignBridge Hybrid Inference System (v2)...")
        
        # Initialize components
        if not self.initialize_components():
            print("❌ Failed to initialize components")
            return
        
        # Initialize motion detection
        self.wrist_history = deque(maxlen=FRAMES_FOR_VELOCITY + 1)
        self.frame_buffer = deque(maxlen=30)
        self.is_recording = False
        self.last_prediction_time = 0
        self.prediction_cooldown = 2.0  # Seconds between predictions
        
        # UI state
        self.current_mode = "WAITING"
        self.current_phrase = ""
        self.confidence = 0.0
        
        # Start main loop
        self.run()
        
    def initialize_components(self):
        """Initialize all ML components."""
        try:
            print("🤖 Initializing ML components...")
            
            # Load alphabet model (static recognition)
            if os.path.exists(ALPHABET_MODEL_PATH):
                self.alphabet_model = tf.keras.models.load_model(ALPHABET_MODEL_PATH, compile=False)
                
                # Load label encoder
                import pickle
                with open(ENCODER_FILE, 'rb') as f:
                    self.alphabet_encoder = pickle.load(f)
                self.alphabet_classes = self.alphabet_encoder.classes_
                print(f"✅ Alphabet model loaded: {len(self.alphabet_classes)} classes")
            else:
                print("⚠️ Alphabet model not found, using dynamic only")
                self.alphabet_model = None
            
            # Load dynamic model (H5 version to avoid TFLite issues)
            if os.path.exists(DYNAMIC_MODEL_PATH):
                self.dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH, compile=False)
                print(f"✅ Dynamic H5 model loaded successfully")
                
                # Load label map
                with open(LABEL_MAP_PATH, 'r') as f:
                    label_map = json.load(f)
                
                # Swap key-values for index lookup
                self.dynamic_labels = {v: k for k, v in label_map.items()}
                print(f"✅ Dynamic labels loaded: {len(self.dynamic_labels)} classes")
            else:
                print("❌ Dynamic model not found!")
                return False
            
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
            print("✅ MediaPipe initialized")
            
            # Initialize webcam
            print("📷 Initializing webcam...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Error: Could not open webcam")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("✅ Webcam initialized")
            
            print("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            return False
    
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
    
    def calculate_wrist_velocity(self, current_wrist):
        """Calculate wrist velocity based on position history."""
        if len(self.wrist_history) < FRAMES_FOR_VELOCITY:
            return 0.0
        
        # Get position 5 frames ago
        old_wrist = self.wrist_history[0]
        
        # Calculate Euclidean distance
        distance = np.sqrt(
            (current_wrist[0] - old_wrist[0])**2 +
            (current_wrist[1] - old_wrist[1])**2 +
            (current_wrist[2] - old_wrist[2])**2
        )
        
        # Velocity = distance / time (assuming ~30fps)
        velocity = distance / FRAMES_FOR_VELOCITY
        
        return velocity
    
    def preprocess_for_dynamic_model(self, frame_sequence):
        """
        Preprocess frame sequence exactly like Kaggle preprocessing.
        Convert to wrist-relative coordinates.
        """
        processed_sequence = []
        
        for frame_array in frame_sequence:
            # Ensure we have exactly 63 features
            if len(frame_array) < 63:
                frame_array = np.pad(frame_array, (0, 63 - len(frame_array)), 'constant')
            elif len(frame_array) > 63:
                frame_array = frame_array[:63]
            
            # CRITICAL: Replicate exact Kaggle preprocessing
            # Convert to wrist-relative coordinates
            wrist_x = frame_array[0]
            wrist_y = frame_array[1]
            wrist_z = frame_array[2]
            
            # Create relative coordinates array
            relative_landmarks = []
            
            # Wrist becomes origin (0,0,0)
            relative_landmarks.extend([0.0, 0.0, 0.0])
            
            # Subtract wrist coordinates from all other landmarks
            for i in range(3, len(frame_array), 3):
                if i + 2 < len(frame_array):
                    rel_x = frame_array[i] - wrist_x
                    rel_y = frame_array[i + 1] - wrist_y
                    rel_z = frame_array[i + 2] - wrist_z
                    relative_landmarks.extend([rel_x, rel_y, rel_z])
            
            # Ensure we still have 63 features
            if len(relative_landmarks) < 63:
                relative_landmarks.extend([0.0] * (63 - len(relative_landmarks)))
            elif len(relative_landmarks) > 63:
                relative_landmarks = relative_landmarks[:63]
            
            processed_sequence.append(relative_landmarks)
        
        return np.array(processed_sequence, dtype=np.float32)
    
    def predict_dynamic_phrase(self, frame_sequence):
        """Predict dynamic phrase using H5 model."""
        try:
            # Preprocess exactly like Kaggle
            processed_sequence = self.preprocess_for_dynamic_model(frame_sequence)
            
            # Ensure we have exactly 30 frames
            if len(processed_sequence) < 30:
                # Pad with zeros
                padding = np.zeros((30 - len(processed_sequence), 63))
                processed_sequence = np.vstack([processed_sequence, padding])
            elif len(processed_sequence) > 30:
                # Truncate by sampling evenly
                indices = np.linspace(0, len(processed_sequence) - 1, 30, dtype=int)
                processed_sequence = processed_sequence[indices]
            
            # Reshape for model input: (1, 30, 63)
            input_data = np.expand_dims(processed_sequence, axis=0)
            
            # Make prediction
            prediction = self.dynamic_model.predict(input_data, verbose=0)
            
            # Get prediction and confidence
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]
            
            # Get label
            if predicted_class_idx in self.dynamic_labels:
                predicted_phrase = self.dynamic_labels[predicted_class_idx]
            else:
                predicted_phrase = "unknown"
            
            return predicted_phrase, confidence
            
        except Exception as e:
            print(f"❌ Dynamic prediction error: {e}")
            return "error", 0.0
    
    def predict_static_letter(self, landmarks):
        """Predict static letter using alphabet model."""
        try:
            if self.alphabet_model is None:
                return None, 0.0
            
            # Prepare input
            landmarks_input = landmarks.reshape(1, -1)
            
            # Make prediction
            prediction = self.alphabet_model.predict(landmarks_input, verbose=0)
            
            # Get prediction
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]
            predicted_letter = self.alphabet_classes[predicted_class_idx]
            
            return predicted_letter, confidence
            
        except Exception as e:
            return None, 0.0
    
    def update_motion_state(self, landmarks):
        """Update motion detection state and route predictions."""
        if landmarks is None:
            self.current_mode = "WAITING"
            return
        
        # Extract wrist coordinates (landmark 0: x, y, z)
        wrist_pos = landmarks[:3]
        
        # Add to wrist history
        self.wrist_history.append(wrist_pos)
        
        # Calculate velocity
        velocity = self.calculate_wrist_velocity(wrist_pos)
        
        # Add current frame to buffer
        self.frame_buffer.append(landmarks)
        
        # Motion routing logic
        if velocity > WRIST_VELOCITY_THRESHOLD:
            # Hand is moving - record for dynamic prediction
            self.current_mode = "RECORDING"
            self.is_recording = True
        else:
            # Hand stopped moving
            if self.is_recording and len(self.frame_buffer) >= MIN_BUFFER_SIZE:
                # We just finished a motion -> Trigger dynamic prediction
                self.current_mode = "PREDICTING"
                self.trigger_dynamic_prediction()
                self.is_recording = False
            else:
                # Hand is just resting/holding a static pose -> Trigger Alphabet prediction
                self.is_recording = False
                if self.current_mode != "PREDICTING":
                    self.current_mode = "WAITING"
                    
                    # --- THE FIX: ROUTE TO STATIC MODEL ---
                    # Only predict static if we have a valid alphabet model loaded
                    if self.alphabet_model is not None:
                        # Extract just the x,y coordinates for the static model (ignoring z)
                        # Assuming your alphabet model was trained on 42 features (21 landmarks * x,y)
                        static_landmarks = []
                        for i in range(0, len(landmarks), 3):
                            static_landmarks.extend([landmarks[i], landmarks[i+1]])
                            
                        letter, conf = self.predict_static_letter(np.array(static_landmarks))
                        
                        # Only update UI if confidence is high to avoid flickering
                        if letter and conf > 0.70:
                            self.current_phrase = letter
                            self.confidence = conf
                            # Optional: print to console (might be spammy, can comment out)
                            # print(f"🔤 Static Prediction: {letter} ({conf:.2f})")
    
    def trigger_dynamic_prediction(self):
        """Trigger dynamic phrase prediction."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_prediction_time < self.prediction_cooldown:
            return
        
        try:
            # Convert buffer to numpy array
            frame_sequence = np.array(list(self.frame_buffer))
            
            # Predict
            phrase, confidence = self.predict_dynamic_phrase(frame_sequence)
            
            if confidence > 0.5:  # Confidence threshold
                self.current_phrase = phrase.upper()
                self.confidence = confidence
                self.last_prediction_time = current_time
                print(f"🎯 Dynamic Prediction: {self.current_phrase} ({confidence:.2f})")
            
            # Clear buffer after prediction
            self.frame_buffer.clear()
            
        except Exception as e:
            print(f"❌ Prediction trigger error: {e}")
    
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
    
    def draw_hybrid_ui(self, frame):
        """Draw hybrid UI with mode indicators and predictions."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for mode indicator
        overlay = frame.copy()
        
        # Mode indicator background
        mode_colors = {
            "WAITING": (50, 50, 50),
            "RECORDING": (0, 100, 200),
            "PREDICTING": (0, 200, 100)
        }
        
        mode_color = mode_colors.get(self.current_mode, (50, 50, 50))
        
        # Draw mode bar at top
        cv2.rectangle(overlay, (0, 0), (width, 80), mode_color, -1)
        
        # Blend overlay
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw mode text
        mode_text = f"MODE: {self.current_mode}"
        cv2.putText(frame, mode_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw buffer status
        buffer_status = f"Buffer: {len(self.frame_buffer)}/30 frames"
        cv2.putText(frame, buffer_status, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw current phrase if available
        if self.current_phrase:
            # Phrase background
            phrase_bg_height = 120
            cv2.rectangle(frame, (0, height - phrase_bg_height), (width, height), (25, 25, 35), -1)
            
            # Phrase text
            phrase_text = self.current_phrase
            confidence_text = f"Confidence: {self.confidence:.1%}"
            
            # Calculate text position (centered)
            text_size = cv2.getTextSize(phrase_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50
            
            cv2.putText(frame, phrase_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100, 200, 255), 3)
            cv2.putText(frame, phrase_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 1)
            
            # Confidence text
            conf_text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            conf_x = (width - conf_text_size[0]) // 2
            cv2.putText(frame, confidence_text, (conf_x, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Draw instructions
        instructions = [
            "Move hand continuously → Record phrase",
            "Stop moving → Trigger prediction",
            "ESC or Q to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width - 350, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main application loop."""
        print("🎉 SignBridge Hybrid Inference System is ready!")
        print("=" * 60)
        print("🎮 Hybrid Recognition Controls:")
        print("  Move hand continuously → Record dynamic phrase")
        print("  Stop moving → Trigger prediction")
        print("  ESC or Q → Quit application")
        print("=" * 60)
        print("📷 Camera is active. Show dynamic ASL phrases!")
        print("✨ Hybrid system ready for dynamic phrase recognition!")
        
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
                
                # Extract landmarks
                landmarks, hand_landmarks = self.extract_hand_landmarks(frame)
                
                # Update motion state and routing
                self.update_motion_state(landmarks)
                
                # Draw landmarks
                frame = self.draw_hand_landmarks(frame, hand_landmarks)
                
                # Draw hybrid UI
                frame = self.draw_hybrid_ui(frame)
                
                # Display frame
                cv2.imshow(WINDOW_NAME, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                
            except Exception as e:
                print(f"⚠️ Error in main loop: {e}")
                continue
        
        # Cleanup
        print("\n👋 Shutting down SignBridge Hybrid System...")
        self.cap.release()
        if self.landmarker:
            self.landmarker.close()
        cv2.destroyAllWindows()
        print("✅ Application closed successfully")

def main():
    """Main entry point."""
    try:
        app = HybridInferenceApp()
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()

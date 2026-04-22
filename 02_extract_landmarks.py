"""
SignBridge — Phase 1, Step 2
Data Extraction: ASL Alphabet Landmarks

This script iterates exactly through the downloaded ASL Alphabet dataset,
passes each image through the MediaPipe HandLandmarker (IMAGE mode),
extracts 63 landmarks (21 points x 3 coordinates), and saves them to numpy arrays.

Output:
    data/alphabet/landmarks.npy (Features: shape (N, 63))
    data/alphabet/labels.npy (Labels: shape (N,))
"""

import os
import cv2
import glob
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path

# ==========================================
# Paths
# ==========================================
# Dataset is extracted in data/asl_alphabet_train/asl_alphabet_train
DATASET_DIR = "data/asl_alphabet_train/asl_alphabet_train"
OUTPUT_DIR  = "data/alphabet"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MediaPipe HandLandmarker Setup
# ==========================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Create HandLandmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        print("Please download and extract the dataset first:")
        print("1. Run: kaggle datasets download -d grassknoted/asl-alphabet")
        print("2. Run: unzip asl-alphabet.zip")
        print("3. Ensure the 'asl_alphabet_train/asl_alphabet_train' folder exists")
        return

    print("="*60)
    print("  SignBridge — Data Extraction (ASL Alphabet)")
    print("="*60)
    print(f"Loading images from: {DATASET_DIR}")

    # Discover all image paths and their corresponding labels
    image_paths = []
    labels = []
    
    # Iterate through each class folder (A, B, C... del, nothing, space)
    class_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    class_folders.sort()
    
    for class_name in class_folders:
        folder_path = os.path.join(DATASET_DIR, class_name)
        
        # Grab all .jpg files in this class folder
        class_images = glob.glob(os.path.join(folder_path, "*.jpg"))
        for img_path in class_images:
            image_paths.append(img_path)
            labels.append(class_name)

    total_images = len(image_paths)
    if total_images == 0:
        print("[ERROR] No images found. Check the dataset path.")
        return

    print(f"Found {total_images} images across {len(class_folders)} classes.")
    print("Starting MediaPipe extraction...")

    X_data = [] # Will store our 63-feature vectors
    y_data = [] # Will store our labels
    
    skipped_count = 0

    # Create HandLandmarker
    with HandLandmarker.create_from_options(options) as landmarker:
        # We use zip to iterate through both paths and labels simultaneously
        # Using tqdm to show an ETA progress bar
        for img_path, label in tqdm(zip(image_paths, labels), total=total_images, desc="Extracting"):
            
            # Read image
            frame = cv2.imread(img_path)
            if frame is None:
                skipped_count += 1
                continue

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process image through MediaPipe
            result = landmarker.detect(mp_image)

            # Check if any hands were detected
            if result.hand_landmarks:
                # Get the first hand detected
                hand_lms = result.hand_landmarks[0]
                
                # Extract the x, y, z for all 21 landmarks into a flat list
                # e.g., [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
                feature_vector = []
                for lm in hand_lms:
                    feature_vector.extend([lm.x, lm.y, lm.z])
                
                # Append to our dataset
                X_data.append(feature_vector)
                y_data.append(label)
            else:
                # MediaPipe didn't find a hand in this image
                skipped_count += 1

    print("\n" + "="*60)
    print("  Extraction Complete!")
    print("="*60)
    
    # Convert lists to NumPy arrays
    X_array = np.array(X_data, dtype=np.float32)
    y_array = np.array(y_data)
    
    print(f"Extracted shape:  {X_array.shape}")
    print(f"Labels shape:     {y_array.shape}")
    print(f"Skipped frames (No Hand): {skipped_count} / {total_images}")

    # Save to disk
    X_path = os.path.join(OUTPUT_DIR, "landmarks.npy")
    y_path = os.path.join(OUTPUT_DIR, "labels.npy")
    
    np.save(X_path, X_array)
    np.save(y_path, y_array)
    
    print(f"\nSaved landmarks to -> {X_path}")
    print(f"Saved labels to    -> {y_path}")

if __name__ == "__main__":
    main()

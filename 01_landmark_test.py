"""
SignBridge — Phase 1, Step 1.2
Real-Time Hand Landmark Visualization (MediaPipe Tasks API)

Opens the Mac webcam, runs MediaPipe HandLandmarker to detect both hands,
then draws all 21 3-D landmarks per hand on the live feed.

Controls:
    q  —  Quit the application cleanly
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os

# ──────────────────────────────────────────────
# MediaPipe Tasks API Setup
# ──────────────────────────────────────────────
BaseOptions       = mp.tasks.BaseOptions
HandLandmarker    = mp.tasks.vision.HandLandmarker
HandLandmarkerOpt = mp.tasks.vision.HandLandmarkerOptions
RunningMode       = mp.tasks.vision.RunningMode

# Drawing utilities (available in the new API)
mp_drawing = mp.tasks.vision.drawing_utils
mp_styles  = mp.tasks.vision.drawing_styles

# Hand connection map — 21 landmarks connected as per MediaPipe hand model
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17),            # Palm
]

# Colors (BGR format for OpenCV)
COLORS = {
    "left_landmark":   (0, 255, 128),   # Bright green
    "left_connection":  (0, 200, 100),   # Darker green
    "right_landmark":  (255, 180, 0),    # Amber/orange
    "right_connection": (200, 140, 0),   # Darker amber
    "label":           (255, 255, 255),  # White
    "hud_bg":          (30, 30, 30),     # Dark grey
    "hud_text":        (0, 255, 200),    # Teal
    "fps_good":        (0, 255, 128),    # Green
    "fps_bad":         (0, 0, 255),      # Red
    "status_on":       (0, 255, 0),      # Green
    "status_off":      (0, 0, 255),      # Red
}

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


def draw_hand_landmarks(frame, hand_landmarks, handedness_label):
    """
    Draw 21 landmarks and connections for a single hand on the frame.

    Args:
        frame: BGR image (numpy array)
        hand_landmarks: List of NormalizedLandmark (21 points)
        handedness_label: "Left" or "Right"
    """
    h, w, _ = frame.shape

    # Choose colors based on handedness
    if handedness_label == "Left":
        dot_color = COLORS["left_landmark"]
        line_color = COLORS["left_connection"]
    else:
        dot_color = COLORS["right_landmark"]
        line_color = COLORS["right_connection"]

    # Convert normalized landmarks to pixel coordinates
    points = []
    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Draw connections first (so dots appear on top)
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = points[start_idx]
        pt2 = points[end_idx]
        cv2.line(frame, pt1, pt2, line_color, 2, cv2.LINE_AA)

    # Draw landmark dots
    for i, (px, py) in enumerate(points):
        # Fingertips get bigger dots (indices 4, 8, 12, 16, 20)
        radius = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, (px, py), radius, dot_color, cv2.FILLED, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)  # white outline

    # Draw handedness label near wrist (landmark 0)
    wrist_x, wrist_y = points[0]
    label = "L" if handedness_label == "Left" else "R"
    offset_x = -30 if handedness_label == "Left" else 15
    cv2.putText(frame, label, (wrist_x + offset_x, wrist_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["label"], 2, cv2.LINE_AA)

    # Draw confidence-style z-depth indicator on fingertips
    for tip_idx in [4, 8, 12, 16, 20]:
        z_val = hand_landmarks[tip_idx].z
        # z is relative depth — negative = closer to camera
        depth_color = (0, 255, 255) if z_val < 0 else (255, 100, 100)
        cv2.circle(frame, points[tip_idx], 7, depth_color, 1, cv2.LINE_AA)


def draw_info_overlay(frame, fps, hand_count, landmark_count):
    """Draw a translucent HUD with FPS, hand detection status, and landmark info."""
    h, w, _ = frame.shape

    # --- Top-left info bar ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (310, 125), COLORS["hud_bg"], cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # FPS
    fps_color = COLORS["fps_good"] if fps >= 15 else COLORS["fps_bad"]
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2, cv2.LINE_AA)

    # Hand status
    status_color = COLORS["status_on"] if hand_count > 0 else COLORS["status_off"]
    cv2.putText(frame, f"Hands Detected: {hand_count}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    # Landmark count
    cv2.putText(frame, f"Total Landmarks: {landmark_count}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Quit hint
    cv2.putText(frame, "Press 'q' to quit", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    # --- Bottom banner ---
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 45), (w, h), COLORS["hud_bg"], cv2.FILLED)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "SignBridge  |  Phase 1: Hand Landmark Visualization",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                COLORS["hud_text"], 1, cv2.LINE_AA)


def main():
    """Main loop: capture → detect → draw → display."""
    print("\n" + "=" * 58)
    print("   SignBridge — Real-Time Hand Landmark Visualization")
    print("=" * 58)

    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model not found at: {MODEL_PATH}")
        print("  Run: curl -L -o models/hand_landmarker.task \\")
        print('    "https://storage.googleapis.com/mediapipe-models/'
              'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"')
        sys.exit(1)

    print(f"  Model loaded: {MODEL_PATH}")
    print("  Opening webcam …  Press 'q' in the window to quit.")
    print("=" * 58 + "\n")

    # ── Configure the HandLandmarker ──
    options = HandLandmarkerOpt(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,                       # Detect up to 2 hands
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open the default webcam (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera permissions:")
        print("  → System Settings > Privacy & Security > Camera")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Read actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h}")

    # FPS tracking
    prev_time = time.time()
    fps = 0.0
    frame_idx = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Empty frame received — skipping.")
                continue

            frame_idx += 1

            # Flip horizontally for a mirror-view experience
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image from the numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Timestamp in milliseconds (required for VIDEO mode)
            timestamp_ms = int(time.time() * 1000)

            # ── Run MediaPipe inference ──
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # ── Draw results ──
            hand_count = len(result.hand_landmarks)
            landmark_count = hand_count * 21  # 21 landmarks per hand

            for i in range(hand_count):
                hand_lms = result.hand_landmarks[i]

                # Get handedness — MediaPipe returns "Left"/"Right" from
                # the camera's perspective; since we flipped, we swap labels
                raw_label = result.handedness[i][0].category_name
                display_label = "Right" if raw_label == "Left" else "Left"

                draw_hand_landmarks(frame, hand_lms, display_label)

            # Calculate FPS (smoothed)
            curr_time = time.time()
            instant_fps = 1.0 / (curr_time - prev_time + 1e-9)
            fps = 0.8 * fps + 0.2 * instant_fps  # Exponential moving average
            prev_time = curr_time

            # Draw HUD overlay
            draw_info_overlay(frame, fps, hand_count, landmark_count)

            # Display the frame
            cv2.imshow("SignBridge — Hand Landmark Test", frame)

            # 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] 'q' pressed — shutting down cleanly.")
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released. Goodbye!\n")


if __name__ == "__main__":
    main()

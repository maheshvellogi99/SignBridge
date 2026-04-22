/**
 * Hybrid Router Constants
 *
 * These thresholds control the "Traffic Cop" state machine
 * that switches between Spelling (static) and Recording (dynamic) modes.
 *
 * Ported from Python desktop SignBridge pipeline.
 */

// --- Velocity Detection ---

/** Wrist velocity threshold (in normalized 0–1 coordinate space).
 *  Values above this indicate the hand is moving → dynamic phrase mode. */
export const WRIST_VELOCITY_THRESHOLD = 0.025;

/** Number of wrist history frames used to calculate velocity.
 *  Euclidean distance is measured between oldest and newest in this window. */
export const FRAMES_FOR_VELOCITY = 5;

// --- Recording Buffer ---

/** Minimum number of buffered frames required for a valid dynamic phrase.
 *  If the user stops moving but has fewer frames than this, the buffer is discarded. */
export const MIN_BUFFER_SIZE = 15;

/** Maximum number of frames to store in the frame buffer.
 *  Acts as a rolling window — oldest frames are dropped when this is exceeded. */
export const MAX_BUFFER_SIZE = 30;

// --- Patience ---

/** Number of consecutive low-velocity frames before we consider the motion "stopped".
 *  Prevents premature phrase termination during brief pauses in signing. */
export const MOTION_PATIENCE = 5;

// --- Wrist History ---

/** Number of wrist positions to retain in the rolling history window.
 *  Must be >= FRAMES_FOR_VELOCITY + 1 to calculate velocity across the full window. */
export const WRIST_HISTORY_SIZE = 6;

// --- Mode Types ---

export type RouterMode = 'WAITING' | 'SPELLING' | 'RECORDING PHRASE' | 'PROCESSING PHRASE';

/**
 * A single frame's flattened landmark data.
 * 21 keypoints × 3 coordinates (x, y, z) = 63 numbers.
 * All values are normalized to 0.0–1.0 range.
 */
export type FlatFrame = number[];

/** Normalized 3D wrist position */
export interface NormalizedPoint3D {
  x: number;
  y: number;
  z: number;
}

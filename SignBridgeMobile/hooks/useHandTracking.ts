import { useState, useRef, useCallback, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
import type { Keypoint } from '@tensorflow-models/hand-pose-detection';

import {
  WRIST_VELOCITY_THRESHOLD,
  FRAMES_FOR_VELOCITY,
  MIN_BUFFER_SIZE,
  MAX_BUFFER_SIZE,
  MOTION_PATIENCE,
  WRIST_HISTORY_SIZE,
  type RouterMode,
  type FlatFrame,
  type NormalizedPoint3D,
} from '../utils/routerConstants';

import {
  loadModels,
  ALPHABET_LABELS,
  DYNAMIC_LABELS,
} from '../utils/ModelLoader';

// Re-export for consumers
export type { Keypoint };

/**
 * Tensor dimensions used for camera resize.
 * MediaPipe Hands is optimized for 256×256 input.
 */
export const TENSOR_WIDTH = 192;
export const TENSOR_HEIGHT = 192;

// ─── Spelling Constants ──────────────────────────────────

/** Number of consecutive identical predictions required before accepting a letter */
const CONSISTENT_FRAMES_THRESHOLD = 8;
/** Cooldown frames after registering a letter (prevents rapid repeats) */
const COOLDOWN_FRAMES = 30;
/** Maximum word length */
const MAX_WORD_LENGTH = 20;
/** Minimum confidence to consider a static prediction */
const STATIC_CONFIDENCE_THRESHOLD = 0.70;
/** Minimum confidence for dynamic phrase prediction */
const DYNAMIC_CONFIDENCE_THRESHOLD = 0.75;

// ─── Hook Return Type ────────────────────────────────────

interface UseHandTrackingReturn {
  /** Whether the TF backend has been initialized */
  isTfReady: boolean;
  /** Whether ALL models (hand detector + alphabet + dynamic) are loaded */
  isModelLoaded: boolean;
  /** Current 21 hand keypoints, or null if no hand detected */
  landmarks: Keypoint[] | null;
  /** Current processing FPS */
  fps: number;
  /** Current state machine mode */
  currentMode: RouterMode;
  /** Current wrist velocity (normalized) */
  velocity: number;
  /** Number of frames in the recording buffer */
  bufferSize: number;
  /** Current word being spelled, letter by letter */
  currentWord: string;
  /** Completed sentence (words + phrase results) */
  sentence: string;
  /** Last prediction label and confidence for UI display */
  lastPrediction: { label: string; confidence: number } | null;
  /** Process a single camera frame tensor. Handles disposal internally. */
  processFrame: (imageTensor: tf.Tensor3D) => Promise<void>;
  /** Dispose of the detector and clean up resources */
  cleanup: () => void;
  /** Clear all text (word + sentence) */
  clearText: () => void;
}

/**
 * Custom hook that encapsulates:
 *  1. TensorFlow.js initialization
 *  2. Hand pose detection (MediaPipe Hands lite)
 *  3. Alphabet + Dynamic model loading
 *  4. Hybrid Router state machine
 *  5. Static spelling prediction with flicker buffer
 *  6. Dynamic phrase prediction with wrist-relative normalization
 *
 * MEMORY GUARDRAIL: Every tensor is disposed via tf.dispose().
 */
export function useHandTracking(): UseHandTrackingReturn {
  // --- Core TF state ---
  const [isTfReady, setIsTfReady] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [landmarks, setLandmarks] = useState<Keypoint[] | null>(null);
  const [fps, setFps] = useState(0);

  // --- Router state ---
  const [currentMode, setCurrentMode] = useState<RouterMode>('WAITING');
  const [velocity, setVelocity] = useState(0);
  const [bufferSize, setBufferSize] = useState(0);

  // --- Prediction output state ---
  const [currentWord, setCurrentWord] = useState('');
  const [sentence, setSentence] = useState('');
  const [lastPrediction, setLastPrediction] = useState<{
    label: string;
    confidence: number;
  } | null>(null);

  // --- Refs for models and detector ---
  const detectorRef = useRef<handPoseDetection.HandDetector | null>(null);
  const alphabetModelRef = useRef<tf.LayersModel | null>(null);
  const dynamicModelRef = useRef<tf.LayersModel | null>(null);

  // --- Processing guards ---
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(0);
  const isProcessingRef = useRef(false);
  /** Global frame counter for frame-skipping logic */
  const globalFrameCountRef = useRef(0);

  // --- Router state machine refs ---
  const wristHistoryRef = useRef<NormalizedPoint3D[]>([]);
  const frameBufferRef = useRef<FlatFrame[]>([]);
  const isRecordingRef = useRef(false);
  const stillFramesRef = useRef(0);

  // --- Spelling state refs (mutated per-frame) ---
  /** Rolling buffer of recent letter predictions for consistency check */
  const predictionHistoryRef = useRef<string[]>([]);
  /** Last letter that was successfully registered into the word */
  const lastRegisteredLetterRef = useRef('');
  /** Cooldown counter after registering a letter */
  const cooldownCounterRef = useRef(0);
  /** Ref to current word for use in callbacks without stale closures */
  const currentWordRef = useRef('');
  /** Ref to current sentence for use in callbacks */
  const sentenceRef = useRef('');

  // ─── Initialization ────────────────────────────────────

  useEffect(() => {
    let cancelled = false;

    const init = async () => {
      try {
        // Step 1: TF backend
        console.log('[HandTracking] Initializing TF backend...');
        await tf.ready();
        if (cancelled) return;
        console.log(`[HandTracking] TF backend ready: ${tf.getBackend()}`);
        setIsTfReady(true);

        // Step 2: Hand pose detector
        console.log('[HandTracking] Loading hand pose detection model...');
        const model = handPoseDetection.SupportedModels.MediaPipeHands;
        const detector = await handPoseDetection.createDetector(model, {
          runtime: 'tfjs' as const,
          modelType: 'lite' as const,
          maxHands: 1,
        });
        if (cancelled) return;
        detectorRef.current = detector;
        console.log('[HandTracking] ✅ Hand detector loaded');

        // Step 3: Load classification models
        console.log('[HandTracking] Loading classification models...');
        const { alphabetModel, dynamicModel } = await loadModels();
        if (cancelled) return;
        alphabetModelRef.current = alphabetModel;
        dynamicModelRef.current = dynamicModel;
        console.log('[HandTracking] ✅ All models loaded');

        setIsModelLoaded(true);
      } catch (error) {
        console.error('[HandTracking] Initialization failed:', error);
      }
    };

    init();
    return () => { cancelled = true; };
  }, []);

  // ─── Utility: Flatten keypoints to 63-number array ─────

  const flattenKeypoints = useCallback((keypoints: Keypoint[]): FlatFrame => {
    const flat: number[] = [];
    for (const kp of keypoints) {
      flat.push(
        kp.x / TENSOR_WIDTH,
        kp.y / TENSOR_HEIGHT,
        (kp.z ?? 0) / TENSOR_WIDTH,
      );
    }
    return flat;
  }, []);

  // ─── Utility: Calculate wrist velocity ─────────────────

  const calculateWristVelocity = useCallback((history: NormalizedPoint3D[]): number => {
    if (history.length < FRAMES_FOR_VELOCITY) return 0;
    const oldest = history[history.length - FRAMES_FOR_VELOCITY];
    const newest = history[history.length - 1];
    const dx = newest.x - oldest.x;
    const dy = newest.y - oldest.y;
    const dz = newest.z - oldest.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz) / FRAMES_FOR_VELOCITY;
  }, []);

  // ─── Static Prediction (Spelling) ─────────────────────

  /**
   * Runs the Alphabet model on the current frame's landmarks.
   * Uses a consistency buffer: the same letter must be predicted
   * for CONSISTENT_FRAMES_THRESHOLD frames in a row before it is
   * registered into the word.
   */
  const triggerStaticPrediction = useCallback((flatFrame: FlatFrame) => {
    if (!alphabetModelRef.current) return;

    // Decrement cooldown
    if (cooldownCounterRef.current > 0) {
      cooldownCounterRef.current--;
    }

    // Run prediction inside tf.tidy to auto-dispose intermediate tensors
    const result = tf.tidy(() => {
      const inputTensor = tf.tensor2d([flatFrame], [1, 63]);
      const output = alphabetModelRef.current!.predict(inputTensor) as tf.Tensor;
      const data = output.dataSync(); // synchronous read — small tensor
      console.log('[Spelling] 🔮 model.predict() fired — raw scores:', Array.from(data).map(v => v.toFixed(3)).join(', '));
      return data;
    });

    const predArray = Array.from(result);
    const idx = predArray.indexOf(Math.max(...predArray));
    const conf = predArray[idx];
    const letter = ALPHABET_LABELS[idx];

    setLastPrediction({ label: letter, confidence: conf });

    if (conf >= STATIC_CONFIDENCE_THRESHOLD) {
      predictionHistoryRef.current.push(letter);

      // Trim to max size
      if (predictionHistoryRef.current.length > CONSISTENT_FRAMES_THRESHOLD) {
        predictionHistoryRef.current.shift();
      }

      // Check consistency: all recent predictions must be the same letter
      if (predictionHistoryRef.current.length >= CONSISTENT_FRAMES_THRESHOLD) {
        const allSame = predictionHistoryRef.current.every(p => p === letter);
        if (
          allSame &&
          cooldownCounterRef.current <= 0 &&
          letter !== lastRegisteredLetterRef.current
        ) {
          // Register the letter
          registerLetter(letter);
          cooldownCounterRef.current = COOLDOWN_FRAMES;
          predictionHistoryRef.current = [];
        }
      }
    } else {
      // Low confidence → clear buffer
      predictionHistoryRef.current = [];
    }
  }, []);

  // ─── Register a letter into the word ───────────────────

  const registerLetter = useCallback((letter: string) => {
    if (letter === 'space') {
      if (currentWordRef.current) {
        sentenceRef.current += currentWordRef.current + ' ';
        setSentence(sentenceRef.current);
        currentWordRef.current = '';
        setCurrentWord('');
        console.log(`[Spelling] ⏎ SPACE — word committed`);
      }
    } else if (letter === 'del') {
      if (currentWordRef.current) {
        currentWordRef.current = currentWordRef.current.slice(0, -1);
        setCurrentWord(currentWordRef.current);
        console.log(`[Spelling] ⌫ DELETE`);
      }
    } else {
      if (currentWordRef.current.length < MAX_WORD_LENGTH) {
        currentWordRef.current += letter;
        setCurrentWord(currentWordRef.current);
        console.log(`[Spelling] ✏️ Added "${letter}" → "${currentWordRef.current}"`);
      }
    }
    lastRegisteredLetterRef.current = letter;
  }, []);

  // ─── Dynamic Prediction (Phrases) ─────────────────────

  /**
   * Runs the Dynamic model on a buffered sequence of frames.
   *
   * CRITICAL PREPROCESSING (matching Python pipeline):
   * 1. Apply wrist-relative spatial normalization
   * 2. Pad/resample to exactly 30 frames
   * 3. Shape to [1, 30, 63]
   */
  const triggerDynamicPrediction = useCallback((buffer: FlatFrame[]) => {
    if (!dynamicModelRef.current || buffer.length < MIN_BUFFER_SIZE) return;

    const result = tf.tidy(() => {
      // 1. Wrist-relative normalization
      const normalized = buffer.map(frame => {
        const wristX = frame[0];
        const wristY = frame[1];
        const wristZ = frame[2];

        const relative: number[] = [0.0, 0.0, 0.0]; // wrist is origin
        for (let i = 3; i < frame.length; i += 3) {
          relative.push(
            frame[i] - wristX,
            frame[i + 1] - wristY,
            frame[i + 2] - wristZ,
          );
        }

        // Ensure exactly 63 values
        while (relative.length < 63) relative.push(0.0);
        return relative.slice(0, 63);
      });

      // 2. Pad or resample to exactly 30 frames
      let frames30: number[][];

      if (normalized.length < 30) {
        // Pad by repeating the last frame
        frames30 = [...normalized];
        const lastFrame = normalized[normalized.length - 1];
        while (frames30.length < 30) {
          frames30.push([...lastFrame]);
        }
      } else if (normalized.length > 30) {
        // Resample: pick 30 evenly-spaced indices
        frames30 = [];
        for (let i = 0; i < 30; i++) {
          const idx = Math.round((i * (normalized.length - 1)) / 29);
          frames30.push(normalized[idx]);
        }
      } else {
        frames30 = normalized;
      }

      // 3. Shape to [1, 30, 63] and predict
      const inputTensor = tf.tensor3d([frames30], [1, 30, 63]);
      const output = dynamicModelRef.current!.predict(inputTensor) as tf.Tensor;
      const data = output.dataSync();
      console.log('[Dynamic] 🔮 model.predict() fired — raw scores:', Array.from(data).map(v => v.toFixed(3)).join(', '));
      return data;
    });

    const predArray = Array.from(result);
    const idx = predArray.indexOf(Math.max(...predArray));
    const conf = predArray[idx];
    const phrase = DYNAMIC_LABELS[idx];

    setLastPrediction({ label: phrase.toUpperCase(), confidence: conf });

    if (conf > DYNAMIC_CONFIDENCE_THRESHOLD) {
      console.log(
        `[Dynamic] 🎯 Phrase: "${phrase.toUpperCase()}" (${(conf * 100).toFixed(1)}%)`
      );

      // Commit current word if any, then append phrase
      if (currentWordRef.current) {
        sentenceRef.current += currentWordRef.current + ' ';
        currentWordRef.current = '';
        setCurrentWord('');
      }
      sentenceRef.current += `[${phrase.toUpperCase()}] `;
      setSentence(sentenceRef.current);
    } else {
      console.log(
        `[Dynamic] ⚠️ Low confidence: "${phrase}" (${(conf * 100).toFixed(1)}%) — skipping`
      );
    }
  }, []);

  // ─── Router: Route a detected hand frame ───────────────

  const routeFrame = useCallback((keypoints: Keypoint[]) => {
    // 1. Normalize wrist position
    const wrist = keypoints[0];
    const normalizedWrist: NormalizedPoint3D = {
      x: wrist.x / TENSOR_WIDTH,
      y: wrist.y / TENSOR_HEIGHT,
      z: (wrist.z ?? 0) / TENSOR_WIDTH,
    };

    // 2. Update wrist history
    const history = wristHistoryRef.current;
    history.push(normalizedWrist);
    if (history.length > WRIST_HISTORY_SIZE) history.shift();

    // 3. Calculate velocity
    const vel = calculateWristVelocity(history);
    setVelocity(vel);

    // 4. Flatten keypoints
    const flatFrame = flattenKeypoints(keypoints);

    // 5. State machine
    const isMoving = vel > WRIST_VELOCITY_THRESHOLD;

    if (isRecordingRef.current || isMoving) {
      frameBufferRef.current.push(flatFrame);
      if (frameBufferRef.current.length > MAX_BUFFER_SIZE) {
        frameBufferRef.current.shift();
      }
      setBufferSize(frameBufferRef.current.length);
    }

    if (isMoving) {
      // DYNAMIC: Hand is moving
      isRecordingRef.current = true;
      stillFramesRef.current = 0;
      setCurrentMode('RECORDING PHRASE');
      // Clear spelling buffer when entering dynamic mode
      predictionHistoryRef.current = [];
    } else {
      if (isRecordingRef.current) {
        // Was recording — check if motion stopped
        stillFramesRef.current++;

        if (stillFramesRef.current > MOTION_PATIENCE) {
          if (frameBufferRef.current.length >= MIN_BUFFER_SIZE) {
            setCurrentMode('PROCESSING PHRASE');
            console.log(
              `[Router] 🎬 DYNAMIC PHRASE COMPLETED: ${frameBufferRef.current.length} frames`
            );
            // Run dynamic prediction
            triggerDynamicPrediction([...frameBufferRef.current]);
          }
          // Reset recording state
          isRecordingRef.current = false;
          stillFramesRef.current = 0;
          frameBufferRef.current = [];
          setBufferSize(0);
        }
      } else {
        // Not recording, hand is still → SPELLING
        setCurrentMode('SPELLING');
        triggerStaticPrediction(flatFrame);
      }
    }
  }, [calculateWristVelocity, flattenKeypoints, triggerStaticPrediction, triggerDynamicPrediction]);

  // ─── Process a single camera frame ─────────────────────

  const processFrame = useCallback(async (imageTensor: tf.Tensor3D) => {
    if (isProcessingRef.current || !detectorRef.current) {
      tf.dispose(imageTensor);
      return;
    }

    // ── Frame Skipping: only run detector on every 2nd frame ──
    globalFrameCountRef.current++;
    if (globalFrameCountRef.current % 2 !== 0) {
      tf.dispose(imageTensor);
      return;
    }

    isProcessingRef.current = true;

    try {
      await tf.nextFrame();

      const hands = await detectorRef.current.estimateHands(imageTensor, {
        flipHorizontal: false,
      });

      // FPS counter
      frameCountRef.current++;
      const now = Date.now();
      if (frameCountRef.current % 10 === 0) {
        if (lastFpsTimeRef.current > 0) {
          const deltaMs = now - lastFpsTimeRef.current;
          setFps(Math.round(10000 / deltaMs));
        }
        lastFpsTimeRef.current = now;
      }

      // Route through state machine
      if (hands.length > 0 && hands[0].keypoints) {
        setLandmarks(hands[0].keypoints);
        routeFrame(hands[0].keypoints);
      } else {
        setLandmarks(null);
        setCurrentMode('WAITING');
        wristHistoryRef.current = [];
        predictionHistoryRef.current = [];
      }
    } catch (error) {
      console.error('[HandTracking] Frame processing error:', error);
    } finally {
      tf.dispose(imageTensor);
      isProcessingRef.current = false;
    }
  }, [routeFrame]);

  // ─── Clear Text ────────────────────────────────────────

  const clearText = useCallback(() => {
    currentWordRef.current = '';
    sentenceRef.current = '';
    setCurrentWord('');
    setSentence('');
    lastRegisteredLetterRef.current = '';
    predictionHistoryRef.current = [];
    cooldownCounterRef.current = 0;
    console.log('[HandTracking] 🧹 Text cleared');
  }, []);

  // ─── Cleanup ───────────────────────────────────────────

  const cleanup = useCallback(() => {
    if (detectorRef.current) {
      detectorRef.current.dispose();
      detectorRef.current = null;
    }
    if (alphabetModelRef.current) {
      alphabetModelRef.current.dispose();
      alphabetModelRef.current = null;
    }
    if (dynamicModelRef.current) {
      dynamicModelRef.current.dispose();
      dynamicModelRef.current = null;
    }
    console.log('[HandTracking] All models disposed');
  }, []);

  useEffect(() => {
    return () => { cleanup(); };
  }, [cleanup]);

  // ─── Return ────────────────────────────────────────────

  return {
    isTfReady,
    isModelLoaded,
    landmarks,
    fps,
    currentMode,
    velocity,
    bufferSize,
    currentWord,
    sentence,
    lastPrediction,
    processFrame,
    cleanup,
    clearText,
  };
}

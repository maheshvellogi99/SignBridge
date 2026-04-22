/**
 * Hand landmark connection indices and shared types.
 *
 * These define the standard 21-point MediaPipe hand skeleton
 * topology used for drawing bone connections between joints.
 */

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/**
 * Standard MediaPipe hand skeleton connections.
 * Each pair [startIndex, endIndex] defines a bone segment.
 *
 * Topology:
 *   Thumb:   0→1→2→3→4
 *   Index:   0→5→6→7→8
 *   Middle:  5→9→10→11→12
 *   Ring:    9→13→14→15→16
 *   Pinky:   13→17→18→19→20
 *   Palm:    0→17
 */
export const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [13, 17], [17, 18], [18, 19], [19, 20],
  // Palm base
  [0, 17],
];

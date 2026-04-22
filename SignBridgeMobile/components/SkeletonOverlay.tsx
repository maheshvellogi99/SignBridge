import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Circle, Line } from 'react-native-svg';
import type { Keypoint } from '@tensorflow-models/hand-pose-detection';
import { HAND_CONNECTIONS } from '../utils/handConstants';
import { TENSOR_WIDTH, TENSOR_HEIGHT } from '../hooks/useHandTracking';

interface SkeletonOverlayProps {
  /** 21 keypoints from the hand pose detector (in tensor pixel space) */
  keypoints: Keypoint[] | null;
  /** Screen width for coordinate scaling */
  screenWidth: number;
  /** Screen height for coordinate scaling */
  screenHeight: number;
}

/**
 * Renders an SVG skeleton overlay on the camera feed.
 *
 * Coordinate Mapping:
 *   The keypoints arrive in tensor pixel space (0→256, 0→256).
 *   We scale them to screen dimensions so the skeleton aligns
 *   with the physical hand in the camera preview.
 *
 * For front camera, x is mirrored: (TENSOR_WIDTH - kp.x) to
 * match the mirrored camera preview.
 */
export const SkeletonOverlay: React.FC<SkeletonOverlayProps> = ({
  keypoints,
  screenWidth,
  screenHeight,
}) => {
  if (!keypoints || keypoints.length === 0) {
    return null;
  }

  // Scale a keypoint from tensor space → screen space
  // iOS front camera feed is already mirrored at the hardware level,
  // so the tensor keypoints are already in mirrored coordinate space.
  // No additional x-flip needed.
  const scalePoint = (kp: Keypoint) => ({
    x: (kp.x / TENSOR_WIDTH) * screenWidth,
    y: (kp.y / TENSOR_HEIGHT) * screenHeight,
  });

  return (
    <View style={styles.overlay} pointerEvents="none">
      <Svg width={screenWidth} height={screenHeight} style={styles.svg}>
        {/* Draw bone connections (blue lines) */}
        {HAND_CONNECTIONS.map(([startIdx, endIdx], index) => {
          if (startIdx >= keypoints.length || endIdx >= keypoints.length) {
            return null;
          }
          const start = scalePoint(keypoints[startIdx]);
          const end = scalePoint(keypoints[endIdx]);

          return (
            <Line
              key={`bone-${index}`}
              x1={start.x}
              y1={start.y}
              x2={end.x}
              y2={end.y}
              stroke="#007AFF"
              strokeWidth="3"
              strokeLinecap="round"
            />
          );
        })}

        {/* Draw landmark joints (green circles with dark border) */}
        {keypoints.map((kp, index) => {
          const point = scalePoint(kp);

          return (
            <Circle
              key={`joint-${index}`}
              cx={point.x}
              cy={point.y}
              r="6"
              fill="#34C759"
              stroke="#000000"
              strokeWidth="2"
            />
          );
        })}
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  svg: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
});

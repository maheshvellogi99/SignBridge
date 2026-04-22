import React, { useCallback, useRef, useEffect, useMemo } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  SafeAreaView,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import * as tf from '@tensorflow/tfjs';

import { useHandTracking, TENSOR_WIDTH, TENSOR_HEIGHT } from '../hooks/useHandTracking';
import { SkeletonOverlay } from '../components/SkeletonOverlay';
import type { RouterMode } from '../utils/routerConstants';

// Create TensorCamera by wrapping CameraView with the TFJS HOC
const TensorCamera = cameraWithTensors(CameraView);

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

const CAMERA_TEXTURE_WIDTH = 1080;
const CAMERA_TEXTURE_HEIGHT = 1920;

/**
 * Mode → Top Bar color mapping.
 */
const MODE_COLORS: Record<RouterMode | 'INITIALIZING' | 'LOADING MODEL', {
  bg: string;
  border: string;
  label: string;
}> = {
  'INITIALIZING': {
    bg: 'rgba(72, 72, 74, 0.8)',
    border: 'rgba(142, 142, 147, 0.3)',
    label: '#8E8E93',
  },
  'LOADING MODEL': {
    bg: 'rgba(72, 72, 74, 0.8)',
    border: 'rgba(142, 142, 147, 0.3)',
    label: '#8E8E93',
  },
  'WAITING': {
    bg: 'rgba(72, 72, 74, 0.7)',
    border: 'rgba(142, 142, 147, 0.3)',
    label: '#AEAEB2',
  },
  'SPELLING': {
    bg: 'rgba(48, 209, 88, 0.25)',
    border: 'rgba(48, 209, 88, 0.5)',
    label: '#30D158',
  },
  'RECORDING PHRASE': {
    bg: 'rgba(10, 132, 255, 0.25)',
    border: 'rgba(10, 132, 255, 0.5)',
    label: '#0A84FF',
  },
  'PROCESSING PHRASE': {
    bg: 'rgba(255, 159, 10, 0.25)',
    border: 'rgba(255, 159, 10, 0.5)',
    label: '#FF9F0A',
  },
};

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();

  // Hand tracking hook — manages everything
  const {
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
    clearText,
  } = useHandTracking();

  // Refs for the processing loop closure
  const isModelLoadedRef = useRef(false);
  const processFrameRef = useRef(processFrame);

  useEffect(() => {
    isModelLoadedRef.current = isModelLoaded;
  }, [isModelLoaded]);

  useEffect(() => {
    processFrameRef.current = processFrame;
  }, [processFrame]);

  // Derive display mode for top bar
  const displayMode = !isTfReady
    ? 'INITIALIZING'
    : !isModelLoaded
    ? 'LOADING MODEL'
    : currentMode;

  const modeStyle = useMemo(() => MODE_COLORS[displayMode], [displayMode]);

  /**
   * onReady callback for TensorCamera — fires ONCE.
   */
  const handleCameraStream = useCallback(
    (images: IterableIterator<tf.Tensor3D>) => {
      console.log('[CameraStream] onReady fired — starting processing loop');

      const loop = async () => {
        if (!isModelLoadedRef.current) {
          requestAnimationFrame(loop);
          return;
        }

        const imageTensor = images.next().value as tf.Tensor3D | undefined;
        if (imageTensor) {
          await processFrameRef.current(imageTensor);
        }

        requestAnimationFrame(loop);
      };

      loop();
    },
    [],
  );

  // --- Permission not yet determined ---
  if (!permission) {
    return <View style={styles.container} />;
  }

  // --- Permission not granted ---
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.permissionContainer}>
        <View style={styles.permissionCard}>
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionMessage}>
            SignBridge needs camera access to translate sign language gestures
          </Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // --- Loading screen ---
  if (!isTfReady || !isModelLoaded) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0A84FF" />
        <Text style={styles.loadingTitle}>
          {!isTfReady
            ? 'Initializing AI Engine...'
            : 'Loading Classification Models...'}
        </Text>
        <Text style={styles.loadingSubtitle}>
          Loading hand detector, alphabet model, and phrase model
        </Text>
      </View>
    );
  }

  // --- Main camera + skeleton + UI ---
  return (
    <View style={styles.container}>
      {/* Layer 1: TensorCamera */}
      <TensorCamera
        style={styles.camera}
        facing="front"
        cameraTextureWidth={CAMERA_TEXTURE_WIDTH}
        cameraTextureHeight={CAMERA_TEXTURE_HEIGHT}
        resizeWidth={TENSOR_WIDTH}
        resizeHeight={TENSOR_HEIGHT}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
        useCustomShadersToResize={true}
      />

      {/* Layer 2: Skeleton Overlay */}
      <View style={styles.skeletonLayer} pointerEvents="none">
        <SkeletonOverlay
          keypoints={landmarks}
          screenWidth={SCREEN_WIDTH}
          screenHeight={SCREEN_HEIGHT}
        />
      </View>

      {/* Layer 3: Top Header Bar */}
      <View
        style={[
          styles.topBar,
          { backgroundColor: modeStyle.bg, borderColor: modeStyle.border },
        ]}
      >
        <View>
          <Text style={[styles.modeText, { color: modeStyle.label }]}>
            {displayMode}
          </Text>
          {(currentMode === 'RECORDING PHRASE' || currentMode === 'SPELLING') && (
            <Text style={styles.velocityText}>
              v={velocity.toFixed(4)}
              {currentMode === 'RECORDING PHRASE' && ` • buf=${bufferSize}`}
            </Text>
          )}
        </View>
        <View style={styles.topBarRight}>
          {lastPrediction && (
            <Text style={styles.predictionBadge}>
              {lastPrediction.label} {(lastPrediction.confidence * 100).toFixed(0)}%
            </Text>
          )}
          {fps > 0 && <Text style={styles.fpsText}>{fps} FPS</Text>}
        </View>
      </View>

      {/* Layer 4: Bottom Control Panel */}
      <View style={styles.bottomPanel}>
        {/* Current Word */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>CURRENT WORD</Text>
          <Text style={styles.wordText}>
            {currentWord || '---'}
            {currentWord ? <Text style={styles.cursor}>|</Text> : null}
          </Text>
        </View>

        {/* Sentence */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>SENTENCE</Text>
          <Text style={styles.sentenceText}>{sentence || '---'}</Text>
        </View>

        {/* Clear Button */}
        <TouchableOpacity style={styles.clearButton} onPress={clearText}>
          <Text style={styles.clearButtonText}>Clear Text</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
    zIndex: 1,
  },
  skeletonLayer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 20,
    elevation: 20,
  },

  // Loading
  loadingContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 30,
  },
  loadingTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '600',
    marginTop: 20,
    textAlign: 'center',
    letterSpacing: 0.5,
  },
  loadingSubtitle: {
    color: '#98989D',
    fontSize: 14,
    marginTop: 8,
    textAlign: 'center',
  },

  // Permission
  permissionContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 30,
  },
  permissionCard: {
    backgroundColor: 'rgba(28, 28, 30, 0.95)',
    padding: 32,
    borderRadius: 20,
    alignItems: 'center',
    width: '100%',
    maxWidth: 360,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  permissionTitle: {
    color: '#fff',
    fontSize: 22,
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 12,
    letterSpacing: 0.5,
  },
  permissionMessage: {
    color: '#AEAEB2',
    fontSize: 16,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 28,
  },
  permissionButton: {
    backgroundColor: '#0A84FF',
    paddingHorizontal: 40,
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
    width: '100%',
  },
  permissionButtonText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '600',
    letterSpacing: 0.5,
  },

  // Top bar
  topBar: {
    position: 'absolute',
    top: 60,
    left: 16,
    right: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 16,
    borderWidth: 1,
    zIndex: 30,
    elevation: 30,
  },
  topBarRight: {
    alignItems: 'flex-end',
  },
  modeText: {
    fontSize: 15,
    fontWeight: '700',
    letterSpacing: 2,
  },
  velocityText: {
    color: '#98989D',
    fontSize: 11,
    fontWeight: '500',
    letterSpacing: 0.5,
    marginTop: 3,
  },
  predictionBadge: {
    color: '#FFD60A',
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  fpsText: {
    color: '#34C759',
    fontSize: 12,
    fontWeight: '500',
    letterSpacing: 1,
    marginTop: 2,
  },

  // Bottom panel
  bottomPanel: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(28, 28, 30, 0.85)',
    paddingHorizontal: 24,
    paddingVertical: 28,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
    zIndex: 30,
    elevation: 30,
  },
  section: {
    marginBottom: 20,
  },
  sectionLabel: {
    color: '#98989D',
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 1.5,
    marginBottom: 6,
    textTransform: 'uppercase',
  },
  wordText: {
    color: '#fff',
    fontSize: 28,
    fontWeight: '700',
    letterSpacing: 0.5,
    minHeight: 36,
  },
  cursor: {
    color: '#0A84FF',
    fontWeight: '300',
  },
  sentenceText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '400',
    lineHeight: 24,
    minHeight: 48,
  },
  clearButton: {
    backgroundColor: 'rgba(255, 59, 48, 0.9)',
    paddingHorizontal: 40,
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: 'center',
    alignSelf: 'center',
    minWidth: 140,
  },
  clearButtonText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
});

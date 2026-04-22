const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Add support for binary files like .task, .tflite
config.resolver.assetExts.push(
  // Binary model files
  'task',
  'tflite',
  'bin',
  'dat'
);

// Handle source extensions for TypeScript
config.resolver.sourceExts.push('jsx', 'js', 'ts', 'tsx', 'json');

module.exports = config;

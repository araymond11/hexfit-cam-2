const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// @tensorflow-models/pose-detection eagerly imports web-only modules.
// Stub them in React Native so Metro can bundle MoveNet without BlazePose/web backends.
config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (moduleName === '@mediapipe/pose' || moduleName === '@tensorflow/tfjs-backend-webgpu') {
    return { type: 'empty' };
  }
  return context.resolveRequest(context, moduleName, platform);
};

module.exports = config;

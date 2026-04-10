import { useCallback, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';
import * as Haptics from 'expo-haptics';
import { useRunOnJS } from 'react-native-worklets-core';
import { useFrameProcessor, runAsync, runAtTargetFps } from 'react-native-vision-camera';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as poseDetection from '@tensorflow-models/pose-detection';
import type { Keypoint } from '@tensorflow-models/pose-detection';

// Primitive captured by worklet closure - Platform.OS not available in worklet runtime
const IS_IOS = Platform.OS === 'ios';

import {
  type JumpPhase,
  type JumpResult,
  type CalibrationData,
  heightFromFlightTime,
  detectTakeoff,
  detectLanding,
} from '../utils/jumpCalc';
import {
  MIN_CONFIDENCE,
  getHipMidpoint,
  getAnkleMidpointY,
  getBodyHeightPixels,
  getHeadY,
} from '../utils/poseUtils';

function vibrateReady() {
  Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
}

/** MoveNet input resolution */
const MODEL_INPUT_SIZE = 256;
/** Number of stable frames required for calibration (~1 second at 10fps) */
const CALIBRATION_FRAMES = 10;
/** Require takeoff to persist for a couple of frames to avoid jitter-triggered false starts. */
const TAKEOFF_CONFIRM_FRAMES = 2;
/** Ignore airborne windows that are too long to be physically plausible for vertical jump tests. */
const MAX_FLIGHT_TIME_MS = 900;
/** If pose is missing for too long while airborne, abandon the sample. */
const MAX_AIRBORNE_TRACKING_GAP_MS = 250;
/** MoveNet tracking profile that trades speed for keypoint quality. */
export type PoseTrackingMode = 'fast' | 'quality';
/** Calibration can use a lower confidence floor to avoid false negatives in lower light. */
const CALIBRATION_MIN_CONFIDENCE = 0.2;
/** Avoid flickering "No person detected" on single missed frames. */
const NO_POSE_HINT_FRAMES = 3;
/** Require repeated classifications before changing hint text. */
const STABLE_HINT_FRAMES = 2;
/** Allow jump trigger/landing using hip when ankles are unstable. */
const HIP_TAKEOFF_THRESHOLD = 0.03;
const HIP_LANDING_THRESHOLD = 0.015;
/** Use custom ankle thresholds to avoid late takeoff and premature landing. */
const ANKLE_TAKEOFF_THRESHOLD = 0.04;
const ANKLE_LANDING_THRESHOLD = 0.02;
/** Require landing condition to be stable for multiple frames. */
const LANDING_CONFIRM_FRAMES = 2;
/** Ignore unrealistically short air-time samples caused by jitter. */
const MIN_FLIGHT_TIME_MS = 190;
/** Prevent immediate retriggering from post-landing jitter. */
const POST_LANDING_COOLDOWN_MS = 500;

export interface JumpDetectionReturn {
  isReady: boolean;
  isCalibrated: boolean;
  phase: JumpPhase;
  result: JumpResult | null;
  keypoints: Keypoint[];
  calibrationProgress: number;
  calibrationTargetFrames: number;
  calibrationHint: string;
  startCalibration: () => void;
  reset: () => void;
  frameProcessor: ReturnType<typeof useFrameProcessor>;
}

export function useJumpDetection(
  userHeightCm: number,
  trackingMode: PoseTrackingMode = 'fast',
): JumpDetectionReturn {
  const modelType =
    trackingMode === 'quality'
      ? poseDetection.movenet.modelType.SINGLEPOSE_THUNDER
      : poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
  // Worklet payload size scales with captureSize^2*3; keep quality mode conservative.
  const captureSize = trackingMode === 'quality' ? 128 : 96;
  const targetProcessFps = trackingMode === 'quality' ? 10 : 15;
  const smoothingAlpha = trackingMode === 'quality' ? 0.72 : 0.5;
  const takeoffConfirmFrames = TAKEOFF_CONFIRM_FRAMES;
  const landingConfirmFrames = LANDING_CONFIRM_FRAMES;
  const maxAirborneTrackingGapMs =
    trackingMode === 'quality' ? 650 : MAX_AIRBORNE_TRACKING_GAP_MS;
  const minFlightTimeMs = trackingMode === 'quality' ? 180 : MIN_FLIGHT_TIME_MS;

  const [isReady, setIsReady] = useState(false);
  const [phase, setPhaseState] = useState<JumpPhase>('IDLE');
  const [result, setResult] = useState<JumpResult | null>(null);
  const [keypoints, setKeypoints] = useState<Keypoint[]>([]);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [calibrationHint, setCalibrationHint] = useState('');

  const detector = useRef<poseDetection.PoseDetector | null>(null);
  const calibration = useRef<CalibrationData | null>(null);
  const calibrationBuf = useRef<{ ankleY: number; hipY: number; bodyH: number }[]>([]);
  const phaseRef = useRef<JumpPhase>('IDLE');
  const takeoffTime = useRef(0);
  const peakHipY = useRef(1.0);
  const takeoffConfirmCount = useRef(0);
  const landingConfirmCount = useRef(0);
  const lastTrackedAt = useRef(0);
  const cooldownUntil = useRef(0);
  const filteredHipY = useRef<number | null>(null);
  const filteredAnkleY = useRef<number | null>(null);
  const noPoseFrames = useRef(0);
  const hintCandidate = useRef('');
  const hintCandidateFrames = useRef(0);
  const appliedHint = useRef('');
  const smoothedKeypoints = useRef<Keypoint[] | null>(null);
  const isProcessing = useRef(false);
  const userHeightRef = useRef(userHeightCm);
  userHeightRef.current = userHeightCm;

  const setPhase = useCallback((p: JumpPhase) => {
    phaseRef.current = p;
    setPhaseState(p);
  }, []);

  const setStableCalibrationHint = useCallback((nextHint: string) => {
    if (nextHint === hintCandidate.current) {
      hintCandidateFrames.current += 1;
    } else {
      hintCandidate.current = nextHint;
      hintCandidateFrames.current = 1;
    }
    const required = nextHint === '' ? 1 : STABLE_HINT_FRAMES;
    if (hintCandidateFrames.current < required) return;
    if (appliedHint.current === nextHint) return;
    appliedHint.current = nextHint;
    setCalibrationHint(nextHint);
  }, []);

  // Reset state when switching tracking profile.
  useEffect(() => {
    calibration.current = null;
    calibrationBuf.current = [];
    takeoffConfirmCount.current = 0;
    landingConfirmCount.current = 0;
    lastTrackedAt.current = 0;
    cooldownUntil.current = 0;
    filteredHipY.current = null;
    filteredAnkleY.current = null;
    noPoseFrames.current = 0;
    hintCandidate.current = '';
    hintCandidateFrames.current = 0;
    appliedHint.current = '';
    smoothedKeypoints.current = null;
    setIsCalibrated(false);
    setCalibrationProgress(0);
    setCalibrationHint('');
    setKeypoints([]);
    setResult(null);
    setPhase('IDLE');
  }, [trackingMode, setPhase]);

  // Initialize TF.js backend and load MoveNet
  useEffect(() => {
    let mounted = true;
    setIsReady(false);
    (async () => {
      await tf.ready();
      const det = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType,
          enableSmoothing: true,
        },
      );
      if (mounted) {
        detector.current = det;
        setIsReady(true);
      } else {
        det.dispose();
      }
    })().catch(console.error);
    return () => {
      mounted = false;
      detector.current?.dispose();
      detector.current = null;
    };
  }, [modelType]);

  const updateJumpState = useCallback(
    (kps: Keypoint[]) => {
      const p = phaseRef.current;
      const now = Date.now();
      const minKpConfidence = p === 'CALIBRATING' ? CALIBRATION_MIN_CONFIDENCE : MIN_CONFIDENCE;
      const hip = getHipMidpoint(kps, minKpConfidence);
      const ankleY = getAnkleMidpointY(kps, minKpConfidence);
      const hasTracking = hip !== null || ankleY !== null;

      if (hasTracking) {
        lastTrackedAt.current = now;
      }

      if (p === 'CALIBRATING') {
        // Check what's missing and give targeted feedback
        const headY = getHeadY(kps, CALIBRATION_MIN_CONFIDENCE);
        const headOk = headY !== null;
        const hipsOk = hip !== null;
        const anklesOk = ankleY !== null;

        if (!headOk && !anklesOk) {
          setStableCalibrationHint('Step back - full body must be visible');
          return;
        }
        if (!anklesOk || !hipsOk) {
          setStableCalibrationHint('Feet not visible - step back or tilt camera down');
          return;
        }
        if (!headOk) {
          setStableCalibrationHint('Head not visible - improve lighting or tilt camera up');
          return;
        }

        const normHipY = hip.y / MODEL_INPUT_SIZE;
        const normAnkleY = ankleY / MODEL_INPUT_SIZE;
        const bodyH = getBodyHeightPixels(kps, CALIBRATION_MIN_CONFIDENCE);

        if (bodyH && bodyH > 20) {
          calibrationBuf.current.push({ ankleY: normAnkleY, hipY: normHipY, bodyH });
          setCalibrationProgress(calibrationBuf.current.length);
          setStableCalibrationHint('Hold still...');
        } else {
          setStableCalibrationHint('Too far away - step closer');
        }

        if (calibrationBuf.current.length >= CALIBRATION_FRAMES) {
          const frames = calibrationBuf.current;
          const n = frames.length;
          const avgAnkleY = frames.reduce((s, f) => s + f.ankleY, 0) / n;
          const avgHipY = frames.reduce((s, f) => s + f.hipY, 0) / n;
          const avgBodyH = frames.reduce((s, f) => s + f.bodyH, 0) / n;
          calibration.current = {
            pixelsPerMeter: avgBodyH / (userHeightRef.current / 100),
            baselineAnkleY: avgAnkleY,
            baselineHipY: avgHipY,
          };
          calibrationBuf.current = [];
          setIsCalibrated(true);
          setStableCalibrationHint('');
          setPhase('READY');
          vibrateReady();
        }
        return;
      }

      const normHipY = hip ? hip.y / MODEL_INPUT_SIZE : null;
      const normAnkleY = ankleY !== null ? ankleY / MODEL_INPUT_SIZE : null;

      // Low-latency smoothing for state transitions to reduce single-frame jitter.
      if (normHipY !== null) {
        filteredHipY.current =
          filteredHipY.current === null ? normHipY : filteredHipY.current * 0.6 + normHipY * 0.4;
      } else {
        filteredHipY.current = null;
      }
      if (normAnkleY !== null) {
        filteredAnkleY.current =
          filteredAnkleY.current === null
            ? normAnkleY
            : filteredAnkleY.current * 0.6 + normAnkleY * 0.4;
      } else {
        filteredAnkleY.current = null;
      }
      const detectHipY = filteredHipY.current;
      const detectAnkleY = filteredAnkleY.current;

      if (p === 'READY') {
        const cal = calibration.current;
        if (!cal) return;
        if (now < cooldownUntil.current) return;
        const ankleTakeoff =
          detectAnkleY !== null &&
          detectTakeoff(detectAnkleY, cal.baselineAnkleY, ANKLE_TAKEOFF_THRESHOLD);
        const hipTakeoff =
          detectHipY !== null && cal.baselineHipY - detectHipY > HIP_TAKEOFF_THRESHOLD;
        if (ankleTakeoff || hipTakeoff) {
          takeoffConfirmCount.current += 1;
          if (takeoffConfirmCount.current >= takeoffConfirmFrames) {
            takeoffTime.current = now;
            peakHipY.current = detectHipY ?? cal.baselineHipY;
            landingConfirmCount.current = 0;
            setPhase('AIRBORNE');
            takeoffConfirmCount.current = 0;
          }
        } else {
          takeoffConfirmCount.current = 0;
        }
        return;
      }

      if (p === 'AIRBORNE') {
        const cal = calibration.current!;
        if (!hasTracking && now - lastTrackedAt.current > maxAirborneTrackingGapMs) {
          // Lost pose too long while airborne -> invalidate this attempt.
          cooldownUntil.current = now + POST_LANDING_COOLDOWN_MS;
          landingConfirmCount.current = 0;
          setPhase('READY');
          return;
        }

        if (detectHipY !== null && detectHipY < peakHipY.current) peakHipY.current = detectHipY;
        const flightTimeMs = now - takeoffTime.current;
        if (flightTimeMs > MAX_FLIGHT_TIME_MS) {
          // Late/failed landing detection would inflate height massively; drop sample.
          cooldownUntil.current = now + POST_LANDING_COOLDOWN_MS;
          landingConfirmCount.current = 0;
          setPhase('READY');
          return;
        }
        const ankleLanding =
          detectAnkleY !== null &&
          detectLanding(detectAnkleY, cal.baselineAnkleY, ANKLE_LANDING_THRESHOLD);
        const hipLanding =
          detectHipY !== null && detectHipY >= cal.baselineHipY - HIP_LANDING_THRESHOLD;

        if (ankleLanding || hipLanding) {
          landingConfirmCount.current += 1;
        } else {
          landingConfirmCount.current = 0;
        }

        if (
          landingConfirmCount.current >= landingConfirmFrames &&
          flightTimeMs > minFlightTimeMs
        ) {
          const heightCm = heightFromFlightTime(flightTimeMs);
          setResult({
            airtimeMs: flightTimeMs,
            airtimeSec: flightTimeMs / 1000,
            heightCm,
            quality: 'MEDIUM',
          });
          cooldownUntil.current = now + POST_LANDING_COOLDOWN_MS;
          landingConfirmCount.current = 0;
          setPhase('LANDED');
          // Auto-return to READY after 3 seconds to allow next jump
          setTimeout(() => setPhase('READY'), 3000);
        }
      }
    },
    [
      landingConfirmFrames,
      maxAirborneTrackingGapMs,
      minFlightTimeMs,
      setPhase,
      setStableCalibrationHint,
      takeoffConfirmFrames,
    ],
  );

  const onFrameCapture = useCallback(
    // rgbData is a number[] of CAPTURE_SIZExCAPTURE_SIZEx3 RGB values,
    // downsampled with letterbox padding in the worklet.
    // lbScale/lbOffsetX/lbOffsetY describe the letterbox transform in model space.
    async (rgbData: number[], _lbScale: number, lbOffsetX: number, lbOffsetY: number) => {
      if (!detector.current || isProcessing.current) return;
      isProcessing.current = true;
      try {
        const rgb = new Uint8Array(rgbData);
        const small = tf.tensor3d(rgb, [captureSize, captureSize, 3]);
        const tensor = tf.image.resizeBilinear(small, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
        tf.dispose(small);

        const poses = await detector.current.estimatePoses(
          tensor as unknown as HTMLVideoElement,
        );
        tf.dispose(tensor);

        if (poses.length === 0) {
          noPoseFrames.current += 1;
          if (noPoseFrames.current < NO_POSE_HINT_FRAMES) return;
          setKeypoints([]);
          smoothedKeypoints.current = null;
          if (phaseRef.current === 'CALIBRATING') {
            setStableCalibrationHint('No person detected - improve lighting and center full body');
          }
          return;
        }
        noPoseFrames.current = 0;

        // Un-letterbox: remove padding offset so keypoints map to the
        // content area within MODEL_INPUT_SIZE.
        const contentW = MODEL_INPUT_SIZE - 2 * lbOffsetX;
        const contentH = MODEL_INPUT_SIZE - 2 * lbOffsetY;
        if (contentW <= 0 || contentH <= 0) return;
        const clamp = (v: number) => Math.max(0, Math.min(MODEL_INPUT_SIZE, v));
        const kps = poses[0].keypoints.map((kp) => ({
          ...kp,
          x: clamp(((kp.x - lbOffsetX) / contentW) * MODEL_INPUT_SIZE),
          y: clamp(((kp.y - lbOffsetY) / contentH) * MODEL_INPUT_SIZE),
        }));
        const prev = smoothedKeypoints.current;
        let next = kps;
        if (prev && prev.length === kps.length) {
          const keep = smoothingAlpha;
          const incoming = 1 - keep;
          next = kps.map((kp, idx) => {
            const prevKp = prev[idx];
            if (!prevKp) return kp;
            const kpScore = kp.score ?? 0;
            const prevScore = prevKp.score ?? 0;
            if (kpScore < MIN_CONFIDENCE || prevScore < MIN_CONFIDENCE) return kp;
            return {
              ...kp,
              x: prevKp.x * keep + kp.x * incoming,
              y: prevKp.y * keep + kp.y * incoming,
            };
          });
        }
        smoothedKeypoints.current = next;
        setKeypoints(next);
        // Use unsmoothed keypoints for state transitions (takeoff/landing) to avoid lag.
        updateJumpState(kps);
      } catch (e) {
        console.warn('Pose detection error:', e);
      } finally {
        isProcessing.current = false;
      }
    },
    [captureSize, setStableCalibrationHint, smoothingAlpha, updateJumpState],
  );

  // useRunOnJS creates a worklet-callable wrapper that runs onFrameCapture on the JS thread
  const onFrameCaptureWorklet = useRunOnJS(onFrameCapture, [onFrameCapture]);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      runAtTargetFps(targetProcessFps, () => {
        runAsync(frame, () => {
          'worklet';
          // ArrayBuffer cannot cross the worklet->JS boundary via useRunOnJS.
          // Downsample with aspect-ratio-preserving padding (letterbox).
          // TF.js will resize to 256x256 on the JS side.
          const buffer = frame.toArrayBuffer();
          const src = new Uint8Array(buffer);
          const outSize = captureSize;
          const srcW = frame.width;
          const srcH = frame.height;
          const srcBytesPerRow = frame.bytesPerRow;
          const orientation = frame.orientation;
          const swapAxes = orientation === 'landscape-left' || orientation === 'landscape-right';
          const orientedW = swapAxes ? srcH : srcW;
          const orientedH = swapAxes ? srcW : srcH;
          // Letterbox: scale uniformly, pad the shorter dimension with black
          const scale = Math.min(outSize / orientedW, outSize / orientedH);
          const scaledW = Math.floor(orientedW * scale);
          const scaledH = Math.floor(orientedH * scale);
          const offsetX = Math.floor((outSize - scaledW) / 2);
          const offsetY = Math.floor((outSize - scaledH) / 2);
          const rgb = new Array(outSize * outSize * 3).fill(0); // black padding
          for (let y = 0; y < scaledH; y++) {
            for (let x = 0; x < scaledW; x++) {
              const uprightX = Math.min(orientedW - 1, Math.floor(x / scale));
              const uprightY = Math.min(orientedH - 1, Math.floor(y / scale));

              let srcX = uprightX;
              let srcY = uprightY;
              if (orientation === 'portrait-upside-down') {
                srcX = srcW - 1 - uprightX;
                srcY = srcH - 1 - uprightY;
              } else if (orientation === 'landscape-left') {
                srcX = uprightY;
                srcY = srcH - 1 - uprightX;
              } else if (orientation === 'landscape-right') {
                srcX = srcW - 1 - uprightY;
                srcY = uprightX;
              }

              const si = srcY * srcBytesPerRow + srcX * 4;
              if (si + 2 >= src.length) continue;
              const di = ((y + offsetY) * outSize + (x + offsetX)) * 3;
              if (IS_IOS) {
                // BGRA -> RGB
                rgb[di] = src[si + 2];
                rgb[di + 1] = src[si + 1];
                rgb[di + 2] = src[si];
              } else {
                // RGBA -> RGB
                rgb[di] = src[si];
                rgb[di + 1] = src[si + 1];
                rgb[di + 2] = src[si + 2];
              }
            }
          }
          // Pass letterbox params so JS side can un-letterbox keypoints
          const lbScale = scale * (MODEL_INPUT_SIZE / outSize); // scale from camera pixels to model space
          const lbOffsetX = offsetX * (MODEL_INPUT_SIZE / outSize);
          const lbOffsetY = offsetY * (MODEL_INPUT_SIZE / outSize);
          onFrameCaptureWorklet(rgb, lbScale, lbOffsetX, lbOffsetY);
        });
      });
    },
    [captureSize, onFrameCaptureWorklet, targetProcessFps],
  );

  const startCalibration = useCallback(() => {
    calibrationBuf.current = [];
    takeoffConfirmCount.current = 0;
    landingConfirmCount.current = 0;
    lastTrackedAt.current = 0;
    cooldownUntil.current = 0;
    filteredHipY.current = null;
    filteredAnkleY.current = null;
    noPoseFrames.current = 0;
    hintCandidate.current = '';
    hintCandidateFrames.current = 0;
    appliedHint.current = '';
    smoothedKeypoints.current = null;
    setKeypoints([]);
    setResult(null);
    setCalibrationProgress(0);
    setCalibrationHint('');
    setPhase('CALIBRATING');
  }, [setPhase]);

  const reset = useCallback(() => {
    calibration.current = null;
    calibrationBuf.current = [];
    takeoffConfirmCount.current = 0;
    landingConfirmCount.current = 0;
    lastTrackedAt.current = 0;
    cooldownUntil.current = 0;
    filteredHipY.current = null;
    filteredAnkleY.current = null;
    noPoseFrames.current = 0;
    hintCandidate.current = '';
    hintCandidateFrames.current = 0;
    appliedHint.current = '';
    smoothedKeypoints.current = null;
    setIsCalibrated(false);
    setKeypoints([]);
    setResult(null);
    setCalibrationProgress(0);
    setCalibrationHint('');
    setPhase('IDLE');
  }, [setPhase]);

  return {
    isReady,
    isCalibrated,
    phase,
    result,
    keypoints,
    calibrationProgress,
    calibrationTargetFrames: CALIBRATION_FRAMES,
    calibrationHint,
    startCalibration,
    reset,
    frameProcessor,
  };
}

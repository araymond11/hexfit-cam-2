import { KP } from './poseUtils.ts';
import {
  type JumpAnalysisDebug,
  type JumpAnalysisResult,
  type JumpBaselineDebug,
  type JumpContactPhase,
  type JumpInvalidReason,
  type JumpLandmarkFrame,
  type JumpPersonCountSummary,
  type JumpPhaseSample,
  type JumpQuality,
  type JumpTimingMode,
  heightFromFlightTime,
} from './jumpCalc.ts';

const MIN_CONFIDENCE = 0.2;
const CALIBRATION_WINDOW_MS = 750;
const TAKEOFF_CONFIRM_FRAMES = 2;
const LANDING_CONFIRM_FRAMES = 2;
const ADAPTIVE_ONE_FRAME_MIN_FPS = 180;
const ADAPTIVE_TAKEOFF_MIN_CONFIDENCE = 0.72;
const ADAPTIVE_LANDING_MIN_CONFIDENCE = 0.7;
const MIN_FULL_BODY_VISIBLE_RATIO = 0.6;
const MAX_TAKEOFF_GAP_FRAMES = 4;
const MIN_FLIGHT_MS = 180;
const MAX_FLIGHT_MS = 900;
const MAX_CALIBRATION_CENTER_RANGE = 0.12;
const MAX_ALLOWED_HORIZONTAL_DRIFT = 0.18;
const MAX_CALIBRATION_NOISE = 0.025;
const MAX_ATTEMPT_SEARCH_AFTER_CALIBRATION_MS = 6000;
const ATTEMPT_WINDOW_PRE_ROLL_MS = 300;
const ATTEMPT_WINDOW_POST_ROLL_MS = 800;
const TOE_CONTACT_THRESHOLD = 0.012;
const GROUND_CONTACT_THRESHOLD = 0.018;
const TOE_FALLBACK_CONTACT_THRESHOLD = 0.016;
const TOE_CLEAR_THRESHOLD = 0.028;
const GROUND_CLEAR_THRESHOLD = 0.032;
const HIP_CLEAR_THRESHOLD = 0.015;
const MIN_FEET_VISIBLE_RATIO = 0.45;

interface AnalysisOptions {
  videoDurationMs?: number;
  videoFps?: number;
  sampleFps?: number;
  playbackVideoFps?: number;
  playbackSampleFps?: number;
  playbackDurationMs?: number;
  captureDurationMs?: number;
  timingMode?: JumpTimingMode;
  timingConfidence?: number;
  captureTimeScale?: number;
  hasTimeSegments?: boolean;
  usedOriginalAsset?: boolean;
  usedPlaybackAsset?: boolean;
  originalDurationMs?: number | null;
  timebaseSource?: string;
  personCountSummary?: JumpPersonCountSummary;
  minConfidence?: number;
}

interface FrameSignals {
  frameIndex: number;
  timestampMs: number;
  captureTimestampMs: number;
  avgConfidence: number;
  personCount: number;
  faceVisible: boolean;
  fullBodyVisible: boolean;
  feetVisible: boolean;
  leftToeY: number | null;
  rightToeY: number | null;
  leftHeelY: number | null;
  rightHeelY: number | null;
  leftKneeY: number | null;
  rightKneeY: number | null;
  leftAnkleY: number | null;
  rightAnkleY: number | null;
  hipY: number | null;
  shoulderY: number | null;
  torso: number | null;
  centerX: number | null;
  /** Pixel-based floor band Y from native module. */
  floorBandY: number | null;
  /** Floor detection confidence from native module. */
  floorConfidence: number;
  /** Pixel-based contact score for the left foot from native module. */
  nativeLeftContactScore: number | null;
  /** Pixel-based contact score for the right foot from native module. */
  nativeRightContactScore: number | null;
}

interface Baseline {
  leftToeY: number;
  rightToeY: number;
  leftGroundY: number;
  rightGroundY: number;
  leftAnkleY: number;
  rightAnkleY: number;
  hipY: number;
  torso: number;
  centerX: number;
  calibrationStartMs: number;
  calibrationEndMs: number;
  stability: number;
}

interface CalibrationCandidate {
  startIndex: number;
  endIndex: number;
  startMs: number;
  endMs: number;
  frames: FrameSignals[];
  baseline: Baseline;
}

type FootContactState = 'CONTACT' | 'CLEAR' | 'UNCERTAIN';

interface FootContactInfo {
  state: FootContactState;
  contact: boolean | null;
  confidence: number;
  toeLift: number | null;
  groundLift: number | null;
}

interface AttemptEvaluation {
  baseline: Baseline;
  takeoffMs: number | null;
  landingMs: number | null;
  takeoffCaptureMs: number | null;
  landingCaptureMs: number | null;
  phaseTimeline: JumpPhaseSample[];
  invalidReason?: JumpInvalidReason;
  maxHorizontalDrift: number;
  uncertaintyRatio: number;
  averageContactReliability: number;
  fullBodyVisibleRatio: number;
  feetVisibleRatio: number;
  peakHipLift: number;
  peakGroundClearance: number;
  attemptWindowStartMs: number;
  attemptWindowEndMs: number;
  selectionScore: number;
  /** Toe Y of the last contact frame before takeoff (landing baseline). */
  takeoffLeftToeY: number | null;
  takeoffRightToeY: number | null;
}

export interface JumpAttemptWindowSuggestion {
  startMs: number;
  endMs: number;
  calibrationStartMs: number;
  calibrationEndMs: number;
  takeoffMs: number | null;
  landingMs: number | null;
  selectionScore: number;
}

function clamp01(value: number): number {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdDev(values: number[]): number {
  if (values.length <= 1) return 0;
  const mean = average(values);
  const variance = average(values.map((value) => (value - mean) ** 2));
  return Math.sqrt(variance);
}

function pointOrNull(
  frame: JumpLandmarkFrame,
  index: number,
  minConfidence: number,
): { x: number; y: number; score: number } | null {
  const point = frame.keypoints[index];
  if (!point || point.score < minConfidence) return null;
  return point;
}

function midpointY(
  a: { y: number } | null,
  b: { y: number } | null,
): number | null {
  if (a && b) return (a.y + b.y) / 2;
  if (a) return a.y;
  if (b) return b.y;
  return null;
}

function midpointX(
  a: { x: number } | null,
  b: { x: number } | null,
): number | null {
  if (a && b) return (a.x + b.x) / 2;
  if (a) return a.x;
  if (b) return b.x;
  return null;
}

function midpointPoint(
  a: { x: number; y: number } | null,
  b: { x: number; y: number } | null,
): { x: number; y: number } | null {
  if (a && b) return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  if (a) return a;
  if (b) return b;
  return null;
}

function last<T>(values: T[]): T | undefined {
  return values[values.length - 1];
}

function bestFootGroundY(
  toeY: number | null,
  heelY: number | null,
  ankleY: number | null,
): number | null {
  const values = [toeY, heelY, ankleY].filter((value): value is number => value !== null);
  if (values.length === 0) return null;
  return Math.max(...values);
}

function medianNullable(values: Array<number | null>, index: number): number | null {
  const nearby: number[] = [];
  for (let offset = -1; offset <= 1; offset += 1) {
    const value = values[index + offset];
    if (typeof value === 'number' && Number.isFinite(value)) {
      nearby.push(value);
    }
  }
  return nearby.length > 0 ? median(nearby) : null;
}

function applyMedianFilter(samples: FrameSignals[]): FrameSignals[] {
  const leftToeSeries = samples.map((sample) => sample.leftToeY);
  const rightToeSeries = samples.map((sample) => sample.rightToeY);
  const leftHeelSeries = samples.map((sample) => sample.leftHeelY);
  const rightHeelSeries = samples.map((sample) => sample.rightHeelY);
  const leftSeries = samples.map((sample) => sample.leftAnkleY);
  const rightSeries = samples.map((sample) => sample.rightAnkleY);
  const hipSeries = samples.map((sample) => sample.hipY);
  const torsoSeries = samples.map((sample) => sample.torso);
  const centerSeries = samples.map((sample) => sample.centerX);

  return samples.map((sample, index) => ({
    ...sample,
    leftToeY: medianNullable(leftToeSeries, index),
    rightToeY: medianNullable(rightToeSeries, index),
    leftHeelY: medianNullable(leftHeelSeries, index),
    rightHeelY: medianNullable(rightHeelSeries, index),
    leftAnkleY: medianNullable(leftSeries, index),
    rightAnkleY: medianNullable(rightSeries, index),
    hipY: medianNullable(hipSeries, index),
    torso: medianNullable(torsoSeries, index),
    centerX: medianNullable(centerSeries, index),
  }));
}

function emaStep(current: number | null, prev: number | null, alpha: number): number | null {
  if (current === null) return null;
  if (prev === null) return current;
  return alpha * current + (1 - alpha) * prev;
}

function applyEma(samples: FrameSignals[], alpha = 1): FrameSignals[] {
  let leftToePrev: number | null = null;
  let rightToePrev: number | null = null;
  let leftHeelPrev: number | null = null;
  let rightHeelPrev: number | null = null;
  let leftPrev: number | null = null;
  let rightPrev: number | null = null;
  let hipPrev: number | null = null;
  let torsoPrev: number | null = null;
  let centerPrev: number | null = null;

  return samples.map((sample) => {
    const nextLeftToe = emaStep(sample.leftToeY, leftToePrev, alpha);
    const nextRightToe = emaStep(sample.rightToeY, rightToePrev, alpha);
    const nextLeftHeel = emaStep(sample.leftHeelY, leftHeelPrev, alpha);
    const nextRightHeel = emaStep(sample.rightHeelY, rightHeelPrev, alpha);
    const nextLeft = emaStep(sample.leftAnkleY, leftPrev, alpha);
    const nextRight = emaStep(sample.rightAnkleY, rightPrev, alpha);
    const nextHip = emaStep(sample.hipY, hipPrev, alpha);
    const nextTorso = emaStep(sample.torso, torsoPrev, alpha);
    const nextCenter = emaStep(sample.centerX, centerPrev, alpha);

    leftToePrev = nextLeftToe;
    rightToePrev = nextRightToe;
    leftHeelPrev = nextLeftHeel;
    rightHeelPrev = nextRightHeel;
    leftPrev = nextLeft;
    rightPrev = nextRight;
    hipPrev = nextHip;
    torsoPrev = nextTorso;
    centerPrev = nextCenter;

    return {
      ...sample,
      leftToeY: nextLeftToe,
      rightToeY: nextRightToe,
      leftHeelY: nextLeftHeel,
      rightHeelY: nextRightHeel,
      leftAnkleY: nextLeft,
      rightAnkleY: nextRight,
      hipY: nextHip,
      torso: nextTorso,
      centerX: nextCenter,
    };
  });
}

function toSignals(frame: JumpLandmarkFrame, minConfidence: number): FrameSignals {
  const footConfidence = Math.max(0.12, minConfidence - 0.08);
  const ankleConfidence = Math.max(0.16, minConfidence - 0.04);
  const faceConfidence = Math.max(0.18, minConfidence - 0.06);
  const nose = pointOrNull(frame, KP.NOSE, faceConfidence);
  const le = pointOrNull(frame, KP.LEFT_EYE, faceConfidence);
  const re = pointOrNull(frame, KP.RIGHT_EYE, faceConfidence);
  const lEar = pointOrNull(frame, KP.LEFT_EAR, faceConfidence);
  const rEar = pointOrNull(frame, KP.RIGHT_EAR, faceConfidence);
  const ls = pointOrNull(frame, KP.LEFT_SHOULDER, minConfidence);
  const rs = pointOrNull(frame, KP.RIGHT_SHOULDER, minConfidence);
  const lh = pointOrNull(frame, KP.LEFT_HIP, minConfidence);
  const rh = pointOrNull(frame, KP.RIGHT_HIP, minConfidence);
  const lk = pointOrNull(frame, KP.LEFT_KNEE, minConfidence);
  const rk = pointOrNull(frame, KP.RIGHT_KNEE, minConfidence);
  const la = pointOrNull(frame, KP.LEFT_ANKLE, ankleConfidence);
  const ra = pointOrNull(frame, KP.RIGHT_ANKLE, ankleConfidence);
  const lHeel = pointOrNull(frame, KP.LEFT_HEEL, footConfidence);
  const rHeel = pointOrNull(frame, KP.RIGHT_HEEL, footConfidence);
  const lfi = pointOrNull(frame, KP.LEFT_FOOT_INDEX, footConfidence);
  const rfi = pointOrNull(frame, KP.RIGHT_FOOT_INDEX, footConfidence);

  const shoulderY = midpointY(ls, rs);
  const hipY = midpointY(lh, rh);
  const torso =
    shoulderY !== null && hipY !== null ? Math.max(0.001, hipY - shoulderY) : null;
  const upperCenter = midpointPoint(ls, rs);
  const lowerCenter = midpointPoint(lh, rh);
  const ankleCenter = midpointPoint(
    la ? { x: la.x, y: la.y } : null,
    ra ? { x: ra.x, y: ra.y } : null,
  );
  const centerX = midpointX(
    upperCenter ?? lowerCenter ?? ankleCenter,
    lowerCenter ?? ankleCenter ?? upperCenter,
  );
  const faceVisible = [nose, le, re, lEar, rEar].some((point) => point !== null);
  const leftToeY = lfi?.y ?? null;
  const rightToeY = rfi?.y ?? null;
  const leftHeelY = lHeel?.y ?? null;
  const rightHeelY = rHeel?.y ?? null;
  const leftKneeY = lk?.y ?? null;
  const rightKneeY = rk?.y ?? null;
  const leftAnkleY = la?.y ?? null;
  const rightAnkleY = ra?.y ?? null;
  const feetVisible =
    leftToeY !== null || rightToeY !== null ||
    leftHeelY !== null || rightHeelY !== null ||
    leftAnkleY !== null || rightAnkleY !== null;
  const bestFootY = Math.max(
    leftToeY ?? 0,
    rightToeY ?? 0,
    leftHeelY ?? 0,
    rightHeelY ?? 0,
    leftAnkleY ?? 0,
    rightAnkleY ?? 0,
  );
  const fullBodyVisible =
    faceVisible &&
    shoulderY !== null &&
    hipY !== null &&
    leftKneeY !== null &&
    rightKneeY !== null &&
    feetVisible &&
    shoulderY > 0.02 &&
    bestFootY < 0.995;

  return {
    frameIndex: frame.frameIndex,
    timestampMs: frame.timestampMs,
    captureTimestampMs: frame.captureTimestampMs ?? frame.timestampMs,
    avgConfidence: frame.avgConfidence,
    personCount: frame.personCount ?? 1,
    faceVisible,
    fullBodyVisible,
    feetVisible,
    leftToeY,
    rightToeY,
    leftHeelY,
    rightHeelY,
    leftKneeY,
    rightKneeY,
    leftAnkleY,
    rightAnkleY,
    hipY,
    shoulderY,
    torso,
    centerX,
    floorBandY: typeof frame.floorBandY === 'number' ? frame.floorBandY : null,
    floorConfidence: frame.floorConfidence ?? 0,
    nativeLeftContactScore: typeof frame.leftContactScore === 'number' ? frame.leftContactScore : null,
    nativeRightContactScore: typeof frame.rightContactScore === 'number' ? frame.rightContactScore : null,
  };
}

function collectCalibrationFrames(window: FrameSignals[]): FrameSignals[] {
  return window.filter((sample) => sample.fullBodyVisible);
}

function buildBaselineFromFrames(
  calibrationFrames: FrameSignals[],
  startMs: number,
  endMs: number,
): Baseline | null {
  const leftToeValues = calibrationFrames
    .map((s) => s.leftToeY)
    .filter((v): v is number => v !== null);
  const rightToeValues = calibrationFrames
    .map((s) => s.rightToeY)
    .filter((v): v is number => v !== null);
  const leftGroundValues = calibrationFrames
    .map((s) => bestFootGroundY(s.leftToeY, s.leftHeelY, s.leftAnkleY))
    .filter((v): v is number => v !== null);
  const rightGroundValues = calibrationFrames
    .map((s) => bestFootGroundY(s.rightToeY, s.rightHeelY, s.rightAnkleY))
    .filter((v): v is number => v !== null);
  const leftAnkleValues = calibrationFrames
    .map((s) => s.leftAnkleY ?? s.rightAnkleY)
    .filter((v): v is number => v !== null);
  const rightAnkleValues = calibrationFrames
    .map((s) => s.rightAnkleY ?? s.leftAnkleY)
    .filter((v): v is number => v !== null);
  const hipValues = calibrationFrames
    .map((sample) => sample.hipY)
    .filter((value): value is number => value !== null);
  const torsoValues = calibrationFrames
    .map((sample) => sample.torso)
    .filter((value): value is number => value !== null);
  const centerValues = calibrationFrames
    .map((sample) => sample.centerX)
    .filter((value): value is number => value !== null);

  if (
    leftGroundValues.length + rightGroundValues.length < 6 ||
    hipValues.length < 6 ||
    torsoValues.length < 6
  ) {
    return null;
  }

  const torso = median(torsoValues);
  const bestFootSeries = calibrationFrames.map((s) => {
    const left = bestFootGroundY(s.leftToeY, s.leftHeelY, s.leftAnkleY);
    const right = bestFootGroundY(s.rightToeY, s.rightHeelY, s.rightAnkleY);
    if (left === null || right === null) return null;
    return (left + right) / 2;
  }).filter((v): v is number => v !== null);
  const stability = bestFootSeries.length > 0
    ? stdDev(bestFootSeries) / Math.max(0.001, torso)
    : 1;

  if (centerValues.length >= 6) {
    const centerRange = Math.max(...centerValues) - Math.min(...centerValues);
    if (centerRange > MAX_CALIBRATION_CENTER_RANGE) {
      return null;
    }
  }

  if (stability > MAX_CALIBRATION_NOISE) {
    return null;
  }

  const leftToeBaseline = leftToeValues.length > 0 ? median(leftToeValues) : median(leftGroundValues);
  const rightToeBaseline = rightToeValues.length > 0 ? median(rightToeValues) : median(rightGroundValues);
  const leftGroundBaseline = median(leftGroundValues);
  const rightGroundBaseline = median(rightGroundValues);

  return {
    leftToeY: leftToeBaseline,
    rightToeY: rightToeBaseline,
    leftGroundY: leftGroundBaseline,
    rightGroundY: rightGroundBaseline,
    leftAnkleY: leftAnkleValues.length > 0 ? median(leftAnkleValues) : leftToeBaseline,
    rightAnkleY: rightAnkleValues.length > 0 ? median(rightAnkleValues) : rightToeBaseline,
    hipY: median(hipValues),
    torso,
    centerX: centerValues.length > 0 ? median(centerValues) : 0.5,
    calibrationStartMs: startMs,
    calibrationEndMs: endMs,
    stability,
  };
}

function collectCalibrationCandidates(samples: FrameSignals[]): CalibrationCandidate[] {
  const candidates: CalibrationCandidate[] = [];

  for (let startIndex = 0; startIndex < samples.length; startIndex += 1) {
    const startSample = samples[startIndex];
    if (!startSample.fullBodyVisible) {
      continue;
    }

    const windowEndLimit = startSample.timestampMs + CALIBRATION_WINDOW_MS;
    const window: FrameSignals[] = [];
    let candidateEndIndex = startIndex;

    for (let index = startIndex; index < samples.length; index += 1) {
      const sample = samples[index];
      if (sample.timestampMs > windowEndLimit) {
        break;
      }
      window.push(sample);
      if (sample.fullBodyVisible) {
        candidateEndIndex = index;
      }
    }

    const calibrationFrames = collectCalibrationFrames(window);
    const endSample = last(calibrationFrames);
    if (calibrationFrames.length < 6 || !endSample) {
      continue;
    }

    const baseline = buildBaselineFromFrames(
      calibrationFrames,
      calibrationFrames[0].timestampMs,
      endSample.timestampMs,
    );
    if (!baseline) {
      continue;
    }

    candidates.push({
      startIndex,
      endIndex: candidateEndIndex,
      startMs: calibrationFrames[0].timestampMs,
      endMs: endSample.timestampMs,
      frames: calibrationFrames,
      baseline,
    });

    startIndex = Math.max(startIndex, candidateEndIndex - 1);
  }

  return candidates;
}

function emptyDebug(
  analyzedFrameCount: number,
  videoDurationMs: number,
  videoFps: number,
  sampleFps: number,
  playbackVideoFps?: number,
  playbackSampleFps?: number,
  playbackDurationMs?: number,
  captureDurationMs?: number,
  timingMode?: JumpTimingMode,
  timingConfidence?: number,
  captureTimeScale?: number,
  hasTimeSegments?: boolean,
  usedOriginalAsset?: boolean,
  usedPlaybackAsset?: boolean,
  originalDurationMs?: number | null,
  timebaseSource?: string,
): JumpAnalysisDebug {
  const baseline: JumpBaselineDebug = {
    leftToeY: 0,
    rightToeY: 0,
    leftAnkleY: 0,
    rightAnkleY: 0,
    hipY: 0,
    torso: 0,
    centerX: 0,
  };

  return {
    calibrationEndMs: CALIBRATION_WINDOW_MS,
    baseline,
    analyzedFrameCount,
    videoDurationMs,
    videoFps,
    sampleFps,
    playbackVideoFps,
    playbackSampleFps,
    playbackDurationMs,
    captureDurationMs,
    slowMotionScaleFactor:
      playbackVideoFps && videoFps && videoFps > 0 ? playbackVideoFps / videoFps : undefined,
    timingMode,
    timingConfidence,
    timingTrusted: timingMode !== 'timing_ambiguous',
    captureTimeScale,
    hasTimeSegments,
    usedOriginalAsset,
    usedPlaybackAsset,
    originalDurationMs,
    timebaseSource,
    averageConfidence: 0,
    uncertaintyRatio: 1,
    fullBodyVisibleRatio: 0,
    feetVisibleRatio: 0,
    maxHorizontalDrift: 0,
    calibrationStability: 0,
  };
}

function invalidSummary(reason: JumpInvalidReason): string {
  switch (reason) {
    case 'NO_PERSON':
      return 'No athlete was detected in the recorded clip.';
    case 'MULTIPLE_PEOPLE':
      return 'More than one person was visible, so the jump was rejected.';
    case 'BODY_NOT_FULLY_VISIBLE':
      return 'Keep the face, torso, hips, and both feet visible for the entire jump.';
    case 'FEET_NOT_VISIBLE':
      return 'Feet were not visible clearly enough to time takeoff and landing.';
    case 'NO_STABLE_CALIBRATION':
      return 'The standing calibration window was too unstable to establish a ground baseline.';
    case 'NO_TAKEOFF':
      return 'Takeoff was not detected after calibration.';
    case 'NO_LANDING':
      return 'Landing was not detected after takeoff.';
    case 'AIRTIME_OUT_OF_RANGE':
      return 'Airtime was outside the expected range for a valid vertical jump.';
    case 'EXCESS_HORIZONTAL_MOTION':
      return 'Too much horizontal motion was detected during the attempt.';
    case 'TIMING_AMBIGUOUS':
      return 'The imported video timebase was ambiguous, so physical airtime could not be trusted.';
    case 'EVENT_ORDER_INVALID':
      return 'Detected landing occurred before takeoff, so the event sequence was rejected.';
  }
}

function isTrustedTimingMode(timingMode: JumpTimingMode | undefined): boolean {
  return timingMode !== 'timing_ambiguous';
}

function scoreQuality(
  averageConfidence: number,
  uncertaintyRatio: number,
  visibilityRatio: number,
  fpsScore: number,
  calibrationStability: number,
  contactReliability: number,
  flags: string[],
): JumpQuality {
  const confidenceScore = clamp01((averageConfidence - 0.22) / 0.5);
  const uncertaintyScore = clamp01(1 - uncertaintyRatio / 0.25);
  const visibilityScore = clamp01((visibilityRatio - 0.55) / 0.4);
  const stabilityScore = clamp01(1 - calibrationStability / 0.03);
  const contactScore = clamp01((contactReliability - 0.35) / 0.45);
  const total =
    confidenceScore * 0.24 +
    uncertaintyScore * 0.2 +
    visibilityScore * 0.16 +
    fpsScore * 0.12 +
    stabilityScore * 0.1 +
    contactScore * 0.18;

  if (confidenceScore < 0.35) flags.push('LOW_CONFIDENCE');
  if (uncertaintyRatio > 0.12) flags.push('UNCERTAIN_TRACKING');
  if (visibilityRatio < 0.75) flags.push('PARTIAL_FEET_VISIBILITY');
  if (fpsScore < 0.75) flags.push('LOW_SAMPLE_FPS');
  if (calibrationStability > 0.018) flags.push('CALIBRATION_NOISE');
  if (contactReliability < 0.55) flags.push('WEAK_TOE_CONTACT_SIGNAL');

  if (total >= 0.72) return 'HIGH';
  if (total >= 0.45) return 'MEDIUM';
  return 'LOW';
}

function buildResult(
  debug: JumpAnalysisDebug,
  phaseTimeline: JumpPhaseSample[],
  quality: JumpQuality,
  qualityFlags: string[],
  invalidReason?: JumpInvalidReason,
  takeoffMs: number | null = null,
  landingMs: number | null = null,
  takeoffPhysicalMs: number | null = null,
  landingPhysicalMs: number | null = null,
  flightMs: number | null = null,
  heightCm: number | null = null,
  summary?: string,
): JumpAnalysisResult {
  return {
    takeoffMs,
    landingMs,
    takeoffPhysicalMs,
    landingPhysicalMs,
    flightMs,
    heightCm,
    phaseTimeline,
    quality,
    invalidReason,
    qualityFlags: qualityFlags.length > 0 ? qualityFlags : undefined,
    summary:
      summary ??
      (invalidReason
        ? invalidSummary(invalidReason)
        : `Takeoff at ${takeoffMs?.toFixed(1)} ms playback, landing at ${landingMs?.toFixed(1)} ms playback.`),
    debug,
  };
}

function classifyFootContact(
  toeY: number | null,
  heelY: number | null,
  ankleY: number | null,
  baselineToeY: number,
  baselineGroundY: number,
  torsoScale: number,
  hipLift: number | null,
  nativeContactScore: number | null = null,
): FootContactInfo {
  const footGroundY = bestFootGroundY(toeY, heelY, ankleY);
  const toeLift = toeY === null ? null : (baselineToeY - toeY) / torsoScale;
  const groundLift = footGroundY === null ? null : (baselineGroundY - footGroundY) / torsoScale;

  const toeNearGround = toeLift !== null && toeLift <= TOE_CONTACT_THRESHOLD;
  const groundNear = groundLift !== null && groundLift <= GROUND_CONTACT_THRESHOLD;
  const toeNearFallback = toeLift !== null && toeLift <= TOE_FALLBACK_CONTACT_THRESHOLD;
  const toeClear = toeLift !== null && toeLift >= TOE_CLEAR_THRESHOLD;
  const groundClear = groundLift !== null && groundLift >= GROUND_CLEAR_THRESHOLD;
  const hipClear = hipLift !== null && hipLift >= HIP_CLEAR_THRESHOLD;

  // Native pixel-based score: >0.6 strongly suggests contact, <0.3 suggests clear
  const hasNativeScore = nativeContactScore !== null && Number.isFinite(nativeContactScore);
  const nativeSupportsContact = hasNativeScore && nativeContactScore! > 0.6;
  const nativeSupportsClear = hasNativeScore && nativeContactScore! < 0.3;

  // Geometric says CONTACT
  if ((toeNearGround && (groundNear || groundLift === null)) || (groundNear && toeNearFallback)) {
    const toeScore =
      toeLift === null ? 0.45 : clamp01(1 - Math.max(0, toeLift) / Math.max(TOE_CONTACT_THRESHOLD, 0.001));
    const groundScore =
      groundLift === null
        ? 0.55
        : clamp01(1 - Math.max(0, groundLift) / Math.max(GROUND_CONTACT_THRESHOLD, 0.001));
    let geoConfidence = Math.max(0.45, (toeScore * 0.65) + (groundScore * 0.35));

    // Blend with native score: if native agrees, boost confidence; if disagrees, reduce
    if (hasNativeScore) {
      const nativeWeight = 0.35;
      geoConfidence = geoConfidence * (1 - nativeWeight) + nativeContactScore! * nativeWeight;
    }

    return {
      state: 'CONTACT',
      contact: true,
      confidence: geoConfidence,
      toeLift,
      groundLift,
    };
  }

  if (toeY === null && groundNear) {
    let conf = 0.42;
    if (hasNativeScore) {
      conf = conf * 0.65 + nativeContactScore! * 0.35;
    }
    return {
      state: 'CONTACT',
      contact: true,
      confidence: conf,
      toeLift,
      groundLift,
    };
  }

  // Native says CONTACT but geometry is ambiguous — promote to CONTACT
  if (nativeSupportsContact && !toeClear && !groundClear) {
    return {
      state: 'CONTACT',
      contact: true,
      confidence: Math.max(0.4, nativeContactScore! * 0.6 + 0.2),
      toeLift,
      groundLift,
    };
  }

  // Geometric says CLEAR
  if ((toeClear && groundClear) || (groundClear && hipClear)) {
    const toeScore =
      toeLift === null
        ? 0.35
        : clamp01((toeLift - TOE_CONTACT_THRESHOLD) / Math.max(TOE_CLEAR_THRESHOLD - TOE_CONTACT_THRESHOLD, 0.001));
    const groundScore =
      groundLift === null
        ? 0.35
        : clamp01((groundLift - GROUND_CONTACT_THRESHOLD) / Math.max(GROUND_CLEAR_THRESHOLD - GROUND_CONTACT_THRESHOLD, 0.001));
    let geoConfidence = Math.max(0.4, (toeScore * 0.6) + (groundScore * 0.4));

    // Blend: if native also says clear, boost; if native says contact, reduce
    if (hasNativeScore) {
      const nativeWeight = 0.3;
      const nativeClearScore = 1 - nativeContactScore!;
      geoConfidence = geoConfidence * (1 - nativeWeight) + nativeClearScore * nativeWeight;
    }

    return {
      state: 'CLEAR',
      contact: false,
      confidence: geoConfidence,
      toeLift,
      groundLift,
    };
  }

  // Native says CLEAR but geometry is ambiguous — promote to CLEAR
  if (nativeSupportsClear && hipClear) {
    return {
      state: 'CLEAR',
      contact: false,
      confidence: Math.max(0.38, (1 - nativeContactScore!) * 0.55 + 0.2),
      toeLift,
      groundLift,
    };
  }

  return {
    state: 'UNCERTAIN',
    contact: null,
    confidence: 0.2,
    toeLift,
    groundLift,
  };
}

function requiredTakeoffConfirmFrames(
  sampleFps: number,
  leftContact: FootContactInfo,
  rightContact: FootContactInfo,
): number {
  if (leftContact.contact !== false || rightContact.contact !== false) {
    return TAKEOFF_CONFIRM_FRAMES;
  }

  return resolveAdaptiveConfirmFrames(
    sampleFps,
    Math.min(leftContact.confidence, rightContact.confidence),
    ADAPTIVE_TAKEOFF_MIN_CONFIDENCE,
    TAKEOFF_CONFIRM_FRAMES,
  );
}

function requiredLandingConfirmFrames(
  sampleFps: number,
  leftContact: FootContactInfo,
  rightContact: FootContactInfo,
): number {
  const strongestContactConfidence = Math.max(
    leftContact.contact === true ? leftContact.confidence : 0,
    rightContact.contact === true ? rightContact.confidence : 0,
  );

  return resolveAdaptiveConfirmFrames(
    sampleFps,
    strongestContactConfidence,
    ADAPTIVE_LANDING_MIN_CONFIDENCE,
    LANDING_CONFIRM_FRAMES,
  );
}

export function resolveAdaptiveConfirmFrames(
  sampleFps: number,
  contactConfidence: number,
  minConfidenceForOneFrame: number,
  fallbackFrames: number,
): number {
  if (
    sampleFps >= ADAPTIVE_ONE_FRAME_MIN_FPS &&
    contactConfidence >= minConfidenceForOneFrame
  ) {
    return 1;
  }

  return fallbackFrames;
}

function selectionScoreForAttempt(
  attempt: AttemptEvaluation,
  averageConfidence: number,
): number {
  const validFlight =
    attempt.takeoffCaptureMs !== null &&
    attempt.landingCaptureMs !== null &&
    attempt.landingCaptureMs - attempt.takeoffCaptureMs >= MIN_FLIGHT_MS &&
    attempt.landingCaptureMs - attempt.takeoffCaptureMs <= MAX_FLIGHT_MS;

  return (
    (attempt.takeoffMs !== null ? 40 : 0) +
    (attempt.landingMs !== null ? 40 : 0) +
    (validFlight ? 50 : 0) +
    attempt.peakHipLift * 120 +
    attempt.peakGroundClearance * 150 +
    attempt.averageContactReliability * 20 +
    averageConfidence * 10 +
    Math.min(attempt.fullBodyVisibleRatio, attempt.feetVisibleRatio) * 15 -
    attempt.uncertaintyRatio * 25 -
    attempt.maxHorizontalDrift * 35 -
    (attempt.invalidReason ? 20 : 0)
  );
}

function evaluateAttemptCandidate(
  samples: FrameSignals[],
  candidate: CalibrationCandidate,
  averageConfidence: number,
  sampleFps: number,
): AttemptEvaluation {
  const baseline = candidate.baseline;
  const evaluationSignals = samples.filter(
    (sample) =>
      sample.timestampMs >= Math.max(0, baseline.calibrationStartMs - ATTEMPT_WINDOW_PRE_ROLL_MS) &&
      sample.timestampMs <= baseline.calibrationEndMs + MAX_ATTEMPT_SEARCH_AFTER_CALIBRATION_MS,
  );

  let takeoffCandidateStart: number | null = null;
  let takeoffCandidateStartCaptureMs: number | null = null;
  let landingCandidateStart: number | null = null;
  let landingCandidateStartCaptureMs: number | null = null;
  let takeoffConfirm = 0;
  let landingConfirm = 0;
  let takeoffMs: number | null = null;
  let landingMs: number | null = null;
  let takeoffCaptureMs: number | null = null;
  let landingCaptureMs: number | null = null;
  let lastContactMs: number | null = null;
  let lastContactCaptureMs: number | null = null;
  // Track toe/ground Y of the last contact frame before takeoff.
  // Used as the landing baseline to compensate for body drift during the jump.
  let lastContactLeftToeY: number | null = null;
  let lastContactRightToeY: number | null = null;
  let lastContactLeftGroundY: number | null = null;
  let lastContactRightGroundY: number | null = null;
  let airborne = false;
  let uncertainFrames = 0;
  let visibleFrames = 0;
  let feetVisibleFrames = 0;
  let trackingGap = 0;
  let maxHorizontalDrift = 0;
  let contactConfidenceValues: number[] = [];
  let peakHipLift = 0;
  let peakGroundClearance = 0;
  let stablePostLandingFrames = 0;

  const torsoScale = Math.max(0.08, baseline.torso);
  const phaseTimeline: JumpPhaseSample[] = [];

  for (const signal of evaluationSignals) {
    const horizontalDrift =
      signal.centerX === null ? null : Math.abs(signal.centerX - baseline.centerX);
    if (horizontalDrift !== null) {
      maxHorizontalDrift = Math.max(maxHorizontalDrift, horizontalDrift);
    }

    if (signal.fullBodyVisible) visibleFrames += 1;
    if (signal.feetVisible) feetVisibleFrames += 1;

    const hipLift =
      signal.hipY === null ? null : (baseline.hipY - signal.hipY) / torsoScale;
    const leftContact = classifyFootContact(
      signal.leftToeY,
      signal.leftHeelY,
      signal.leftAnkleY,
      baseline.leftToeY,
      baseline.leftGroundY,
      torsoScale,
      hipLift,
      signal.nativeLeftContactScore,
    );
    const rightContact = classifyFootContact(
      signal.rightToeY,
      signal.rightHeelY,
      signal.rightAnkleY,
      baseline.rightToeY,
      baseline.rightGroundY,
      torsoScale,
      hipLift,
      signal.nativeRightContactScore,
    );

    const leftLift = leftContact.groundLift;
    const rightLift = rightContact.groundLift;
    const anyContact = leftContact.contact === true || rightContact.contact === true;
    const bothClear = leftContact.contact === false && rightContact.contact === false;
    let phase: JumpContactPhase = 'UNCERTAIN';

    if (leftContact.contact !== null) contactConfidenceValues.push(leftContact.confidence);
    if (rightContact.contact !== null) contactConfidenceValues.push(rightContact.confidence);
    if (hipLift !== null) peakHipLift = Math.max(peakHipLift, hipLift);
    if (leftLift !== null) peakGroundClearance = Math.max(peakGroundClearance, leftLift);
    if (rightLift !== null) peakGroundClearance = Math.max(peakGroundClearance, rightLift);

    if (signal.timestampMs < baseline.calibrationEndMs) {
      phase = anyContact ? 'GROUND_CONTACT' : 'UNCERTAIN';
      if (anyContact) {
        lastContactMs = signal.timestampMs;
        lastContactCaptureMs = signal.captureTimestampMs;
        lastContactLeftToeY = signal.leftToeY;
        lastContactRightToeY = signal.rightToeY;
        lastContactLeftGroundY = bestFootGroundY(signal.leftToeY, signal.leftHeelY, signal.leftAnkleY);
        lastContactRightGroundY = bestFootGroundY(signal.rightToeY, signal.rightHeelY, signal.rightAnkleY);
      }
    } else if (!airborne) {
      if (anyContact) {
        phase = 'GROUND_CONTACT';
        lastContactMs = signal.timestampMs;
        lastContactCaptureMs = signal.captureTimestampMs;
        lastContactLeftToeY = signal.leftToeY;
        lastContactRightToeY = signal.rightToeY;
        lastContactLeftGroundY = bestFootGroundY(signal.leftToeY, signal.leftHeelY, signal.leftAnkleY);
        lastContactRightGroundY = bestFootGroundY(signal.rightToeY, signal.rightHeelY, signal.rightAnkleY);
        takeoffConfirm = 0;
        trackingGap = 0;
        takeoffCandidateStart = null;
        takeoffCandidateStartCaptureMs = null;
      } else if (bothClear) {
        phase = 'AIRBORNE';
        trackingGap = 0;
        takeoffCandidateStart ??= signal.timestampMs;
        takeoffCandidateStartCaptureMs ??= signal.captureTimestampMs;
        takeoffConfirm += 1;
        if (takeoffConfirm >= requiredTakeoffConfirmFrames(sampleFps, leftContact, rightContact)) {
          takeoffMs = takeoffCandidateStart;
          takeoffCaptureMs = takeoffCandidateStartCaptureMs;
          airborne = true;
          stablePostLandingFrames = 0;
        }
      } else {
        phase = 'UNCERTAIN';
        uncertainFrames += 1;
        trackingGap += 1;
        if (takeoffConfirm > 0 && trackingGap <= MAX_TAKEOFF_GAP_FRAMES) {
          phase = 'AIRBORNE';
        } else {
          takeoffConfirm = 0;
          trackingGap = 0;
          takeoffCandidateStart = null;
          takeoffCandidateStartCaptureMs = null;
        }
      }
    } else {
      if (anyContact) {
        phase = 'GROUND_CONTACT';
        landingCandidateStart ??= signal.timestampMs;
        landingCandidateStartCaptureMs ??= signal.captureTimestampMs;
        landingConfirm += 1;
        if (
          landingConfirm >= requiredLandingConfirmFrames(sampleFps, leftContact, rightContact) &&
          landingMs === null
        ) {
          landingMs = landingCandidateStart;
          landingCaptureMs = landingCandidateStartCaptureMs;
          airborne = false;
          stablePostLandingFrames = 1;
        } else if (landingMs !== null) {
          stablePostLandingFrames += 1;
        }
      } else if (bothClear) {
        phase = 'AIRBORNE';
        landingConfirm = 0;
        landingCandidateStart = null;
        landingCandidateStartCaptureMs = null;
      } else {
        phase = 'UNCERTAIN';
        uncertainFrames += 1;
      }
    }

    phaseTimeline.push({
      frameIndex: signal.frameIndex,
      timestampMs: signal.timestampMs,
      phase,
      leftLift,
      rightLift,
      hipLift,
      horizontalDrift,
      leftToeContact: leftContact.contact,
      rightToeContact: rightContact.contact,
      leftContactConfidence: leftContact.contact === null ? null : leftContact.confidence,
      rightContactConfidence: rightContact.contact === null ? null : rightContact.confidence,
      leftToeY: signal.leftToeY,
      rightToeY: signal.rightToeY,
      leftHeelY: signal.leftHeelY,
      rightHeelY: signal.rightHeelY,
      leftAnkleY: signal.leftAnkleY,
      rightAnkleY: signal.rightAnkleY,
      floorBandY: signal.floorBandY,
      floorConfidence: signal.floorConfidence,
      poseConfidence: signal.avgConfidence,
    });

    if (landingMs !== null && stablePostLandingFrames >= LANDING_CONFIRM_FRAMES + 2) {
      break;
    }
  }

  // --- Adaptive symmetric refinement for takeoff & landing timestamps ---
  // The fixed contact/clear thresholds create a wide dead zone (~34ms at 240fps).
  // Instead, use an adaptive threshold based on the peak foot clearance during
  // flight. The SAME threshold is used for both events so the foot is at the same
  // height at takeoff and landing — systematic errors cancel out.
  if (takeoffMs !== null && landingMs !== null) {
    // 1. Compute peak min-foot-lift during the flight phase.
    let peakMinLift = 0;
    for (const sample of phaseTimeline) {
      if (sample.timestampMs < takeoffMs || sample.timestampMs > landingMs) continue;
      const minLift = Math.min(
        sample.leftLift ?? -Infinity,
        sample.rightLift ?? -Infinity,
      );
      if (Number.isFinite(minLift) && minLift > peakMinLift) {
        peakMinLift = minLift;
      }
    }

    // 2. Adaptive threshold: 40% of peak, clamped to [0.035, 0.080].
    const refineThreshold = Math.max(0.035, Math.min(peakMinLift * 0.40, 0.080));

    // Only refine if the peak clearance is well above the threshold (real jump).
    if (peakMinLift > refineThreshold * 1.5) {
      const captureTimeByFrame = new Map<number, number>();
      for (const sig of evaluationSignals) {
        captureTimeByFrame.set(sig.frameIndex, sig.captureTimestampMs);
      }

      const origTakeoffMs = takeoffMs;
      const origTakeoffCaptureMs = takeoffCaptureMs;
      const origLandingMs = landingMs;
      const origLandingCaptureMs = landingCaptureMs;

      // 3. Refine takeoff: find UPWARD crossing of refineThreshold.
      //    Search from before lastContactMs to well after takeoffCandidateStart.
      for (let i = 1; i < phaseTimeline.length; i++) {
        const prev = phaseTimeline[i - 1];
        const curr = phaseTimeline[i];
        if (curr.timestampMs < (lastContactMs ?? takeoffMs) - 20) continue;
        if (prev.timestampMs > takeoffMs + 100) break;

        const prevMinLift = Math.min(prev.leftLift ?? Infinity, prev.rightLift ?? Infinity);
        const currMinLift = Math.min(curr.leftLift ?? Infinity, curr.rightLift ?? Infinity);

        if (
          Number.isFinite(prevMinLift) && Number.isFinite(currMinLift) &&
          prevMinLift < refineThreshold && currMinLift >= refineThreshold
        ) {
          const range = currMinLift - prevMinLift;
          if (range > 0) {
            const ratio = (refineThreshold - prevMinLift) / range;
            takeoffMs = prev.timestampMs + ratio * (curr.timestampMs - prev.timestampMs);
            const pc = captureTimeByFrame.get(prev.frameIndex);
            const cc = captureTimeByFrame.get(curr.frameIndex);
            if (pc !== undefined && cc !== undefined) {
              takeoffCaptureMs = pc + ratio * (cc - pc);
            }
          }
          break;
        }
      }

      // 4. Refine landing: find DOWNWARD crossing of refineThreshold.
      //    Search near the detected landing, working forward from mid-flight.
      let landingRefined = false;
      for (let i = phaseTimeline.length - 1; i >= 1; i--) {
        const prev = phaseTimeline[i - 1];
        const curr = phaseTimeline[i];
        if (curr.timestampMs < origLandingMs - 100) break;
        if (prev.timestampMs > origLandingMs + 20) continue;

        const prevMinLift = Math.min(prev.leftLift ?? Infinity, prev.rightLift ?? Infinity);
        const currMinLift = Math.min(curr.leftLift ?? Infinity, curr.rightLift ?? Infinity);

        if (
          Number.isFinite(prevMinLift) && Number.isFinite(currMinLift) &&
          prevMinLift >= refineThreshold && currMinLift < refineThreshold
        ) {
          const range = prevMinLift - currMinLift;
          if (range > 0) {
            const ratio = (prevMinLift - refineThreshold) / range;
            landingMs = prev.timestampMs + ratio * (curr.timestampMs - prev.timestampMs);
            const pc = captureTimeByFrame.get(prev.frameIndex);
            const cc = captureTimeByFrame.get(curr.frameIndex);
            if (pc !== undefined && cc !== undefined) {
              landingCaptureMs = pc + ratio * (cc - pc);
            }
          }
          landingRefined = true;
          break;
        }
      }

      // 5. Safety: if refined flight time is out of range, revert to originals.
      const refinedFlight = (landingCaptureMs ?? landingMs)! - (takeoffCaptureMs ?? takeoffMs)!;
      if (refinedFlight < MIN_FLIGHT_MS || refinedFlight > MAX_FLIGHT_MS) {
        takeoffMs = origTakeoffMs;
        takeoffCaptureMs = origTakeoffCaptureMs;
        landingMs = origLandingMs;
        landingCaptureMs = origLandingCaptureMs;
      }
    }
  }

  const analyzedFrames = Math.max(phaseTimeline.length, 1);
  const uncertaintyRatio = uncertainFrames / analyzedFrames;
  const averageContactReliability =
    contactConfidenceValues.length > 0 ? average(contactConfidenceValues) : 0;
  const fullBodyVisibleRatio = visibleFrames / analyzedFrames;
  const feetVisibleRatio = feetVisibleFrames / analyzedFrames;

  let invalidReason: JumpInvalidReason | undefined;
  if (fullBodyVisibleRatio < MIN_FULL_BODY_VISIBLE_RATIO) {
    invalidReason = 'BODY_NOT_FULLY_VISIBLE';
  } else if (feetVisibleRatio < MIN_FEET_VISIBLE_RATIO) {
    invalidReason = 'FEET_NOT_VISIBLE';
  } else if (maxHorizontalDrift > MAX_ALLOWED_HORIZONTAL_DRIFT) {
    invalidReason = 'EXCESS_HORIZONTAL_MOTION';
  } else if (takeoffMs === null) {
    invalidReason = 'NO_TAKEOFF';
  } else if (landingMs === null) {
    invalidReason = 'NO_LANDING';
  } else if (landingMs <= takeoffMs) {
    invalidReason = 'EVENT_ORDER_INVALID';
  } else if (
    takeoffCaptureMs !== null &&
    landingCaptureMs !== null &&
    landingCaptureMs <= takeoffCaptureMs
  ) {
    invalidReason = 'EVENT_ORDER_INVALID';
  } else {
    const flightMs = (landingCaptureMs ?? landingMs) - (takeoffCaptureMs ?? takeoffMs);
    if (flightMs < MIN_FLIGHT_MS || flightMs > MAX_FLIGHT_MS) {
      invalidReason = 'AIRTIME_OUT_OF_RANGE';
    }
  }

  const attemptWindowStartMs = Math.max(0, baseline.calibrationStartMs - ATTEMPT_WINDOW_PRE_ROLL_MS);
  const attemptWindowEndMs = Math.min(
    last(evaluationSignals)?.timestampMs ?? baseline.calibrationEndMs,
    (landingMs ?? last(evaluationSignals)?.timestampMs ?? baseline.calibrationEndMs) + ATTEMPT_WINDOW_POST_ROLL_MS,
  );

  const evaluation: AttemptEvaluation = {
    baseline,
    takeoffMs,
    landingMs,
    takeoffCaptureMs,
    landingCaptureMs,
    phaseTimeline,
    invalidReason,
    maxHorizontalDrift,
    uncertaintyRatio,
    averageContactReliability,
    fullBodyVisibleRatio,
    feetVisibleRatio,
    peakHipLift,
    peakGroundClearance,
    attemptWindowStartMs,
    attemptWindowEndMs,
    selectionScore: 0,
    takeoffLeftToeY: lastContactLeftToeY,
    takeoffRightToeY: lastContactRightToeY,
  };

  evaluation.selectionScore = selectionScoreForAttempt(evaluation, averageConfidence);
  return evaluation;
}

function selectBestAttemptEvaluation(
  samples: FrameSignals[],
  averageConfidence: number,
  sampleFps: number,
): {
  evaluation: AttemptEvaluation | null;
  candidateCount: number;
} {
  const candidates = collectCalibrationCandidates(samples);
  let bestValid: AttemptEvaluation | null = null;
  let bestAny: AttemptEvaluation | null = null;

  for (const candidate of candidates) {
    const evaluation = evaluateAttemptCandidate(samples, candidate, averageConfidence, sampleFps);
    if (!bestAny || evaluation.selectionScore > bestAny.selectionScore) {
      bestAny = evaluation;
    }
    if (!evaluation.invalidReason && (!bestValid || evaluation.selectionScore > bestValid.selectionScore)) {
      bestValid = evaluation;
    }
  }

  return {
    evaluation: bestValid ?? bestAny,
    candidateCount: candidates.length,
  };
}

function buildDebugFromAttempt(
  attempt: AttemptEvaluation,
  analyzedFrameCount: number,
  videoDurationMs: number,
  videoFps: number,
  sampleFps: number,
  playbackVideoFps: number | undefined,
  playbackSampleFps: number | undefined,
  playbackDurationMs: number | undefined,
  captureDurationMs: number | undefined,
  timingMode: JumpTimingMode | undefined,
  timingConfidence: number | undefined,
  captureTimeScale: number | undefined,
  hasTimeSegments: boolean | undefined,
  usedOriginalAsset: boolean | undefined,
  usedPlaybackAsset: boolean | undefined,
  originalDurationMs: number | null | undefined,
  timebaseSource: string | undefined,
  averageConfidence: number,
): JumpAnalysisDebug {
  const baseline: JumpBaselineDebug = {
    leftToeY: attempt.baseline.leftToeY,
    rightToeY: attempt.baseline.rightToeY,
    takeoffLeftToeY: attempt.takeoffLeftToeY ?? undefined,
    takeoffRightToeY: attempt.takeoffRightToeY ?? undefined,
    leftAnkleY: attempt.baseline.leftAnkleY,
    rightAnkleY: attempt.baseline.rightAnkleY,
    hipY: attempt.baseline.hipY,
    torso: attempt.baseline.torso,
    centerX: attempt.baseline.centerX,
  };

  return {
    calibrationStartMs: attempt.baseline.calibrationStartMs,
    calibrationEndMs: attempt.baseline.calibrationEndMs,
    attemptWindowStartMs: attempt.attemptWindowStartMs,
    attemptWindowEndMs: attempt.attemptWindowEndMs,
    baseline,
    analyzedFrameCount,
    videoDurationMs,
    videoFps,
    sampleFps,
    playbackVideoFps,
    playbackSampleFps,
    playbackDurationMs,
    captureDurationMs,
    slowMotionScaleFactor:
      playbackVideoFps && videoFps > 0 ? playbackVideoFps / videoFps : undefined,
    timingMode,
    timingConfidence,
    timingTrusted: isTrustedTimingMode(timingMode),
    captureTimeScale,
    hasTimeSegments,
    usedOriginalAsset,
    usedPlaybackAsset,
    originalDurationMs,
    timebaseSource,
    averageConfidence,
    uncertaintyRatio: attempt.uncertaintyRatio,
    fullBodyVisibleRatio: attempt.fullBodyVisibleRatio,
    feetVisibleRatio: attempt.feetVisibleRatio,
    maxHorizontalDrift: attempt.maxHorizontalDrift,
    calibrationStability: attempt.baseline.stability,
    averageContactReliability: attempt.averageContactReliability,
    attemptSelectionScore: attempt.selectionScore,
  };
}

export function explainJumpAnalysis(result: JumpAnalysisResult): string {
  return result.summary;
}

export function suggestJumpAttemptWindow(
  frames: JumpLandmarkFrame[],
  options: AnalysisOptions = {},
): JumpAttemptWindowSuggestion | null {
  if (frames.length === 0) return null;

  const minConfidence = options.minConfidence ?? MIN_CONFIDENCE;
  const rawSignals = frames.map((frame) => toSignals(frame, minConfidence));
  const signals = applyEma(applyMedianFilter(rawSignals));
  const averageConfidence = average(signals.map((signal) => signal.avgConfidence));
  const selection = selectBestAttemptEvaluation(signals, averageConfidence, options.sampleFps ?? options.videoFps ?? 60);
  const attempt = selection.evaluation;
  if (!attempt || attempt.takeoffMs === null || attempt.landingMs === null) {
    return null;
  }

  return {
    startMs: attempt.attemptWindowStartMs,
    endMs: attempt.attemptWindowEndMs,
    calibrationStartMs: attempt.baseline.calibrationStartMs,
    calibrationEndMs: attempt.baseline.calibrationEndMs,
    takeoffMs: attempt.takeoffMs,
    landingMs: attempt.landingMs,
    selectionScore: attempt.selectionScore,
  };
}

export function analyzeJumpLandmarks(
  frames: JumpLandmarkFrame[],
  options: AnalysisOptions = {},
): JumpAnalysisResult {
  const videoDurationMs =
    options.videoDurationMs ??
    Math.max(0, ...frames.map((frame) => frame.timestampMs), 0);
  const playbackDurationMs = options.playbackDurationMs ?? videoDurationMs;
  const captureDurationMs = options.captureDurationMs ?? videoDurationMs;
  const videoFps = options.videoFps ?? options.sampleFps ?? 60;
  const sampleFps = options.sampleFps ?? options.videoFps ?? 60;
  const debugBase = emptyDebug(
    frames.length,
    videoDurationMs,
    videoFps,
    sampleFps,
    options.playbackVideoFps,
    options.playbackSampleFps,
    playbackDurationMs,
    options.captureDurationMs ?? captureDurationMs,
    options.timingMode,
    options.timingConfidence,
    options.captureTimeScale,
    options.hasTimeSegments,
    options.usedOriginalAsset,
    options.usedPlaybackAsset,
    options.originalDurationMs,
    options.timebaseSource,
  );

  if (frames.length === 0) {
    return buildResult(debugBase, [], 'LOW', ['NO_ANALYZED_FRAMES'], 'NO_PERSON');
  }

  const minConfidence = options.minConfidence ?? MIN_CONFIDENCE;
  const rawSignals = frames.map((frame) => toSignals(frame, minConfidence));
  const signals = applyEma(applyMedianFilter(rawSignals));
  const averageConfidence = average(signals.map((signal) => signal.avgConfidence));
  const globalFullBodyVisibleRatio =
    signals.filter((signal) => signal.fullBodyVisible).length / signals.length;
  const globalFeetVisibleRatio =
    signals.filter((signal) => signal.feetVisible).length / signals.length;
  const observedPersonCount =
    options.personCountSummary ??
    {
      analyzedFrames: signals.length,
      multiPersonFrames: signals.filter((signal) => signal.personCount > 1).length,
      maxPeople: Math.max(...signals.map((signal) => signal.personCount), 0),
    };

  const preFlags: string[] = [];
  const fpsScore = clamp01(Math.min(sampleFps, 60) / 60);

  if (observedPersonCount.maxPeople > 1 || observedPersonCount.multiPersonFrames > 0) {
    return buildResult(
      {
        ...debugBase,
        averageConfidence,
        fullBodyVisibleRatio: globalFullBodyVisibleRatio,
        feetVisibleRatio: globalFeetVisibleRatio,
      },
      [],
      'LOW',
      ['MULTI_PERSON_INPUT'],
      'MULTIPLE_PEOPLE',
    );
  }

  const selection = selectBestAttemptEvaluation(
    signals,
    averageConfidence,
    options.sampleFps ?? options.videoFps ?? 60,
  );
  if (!selection.evaluation) {
    const fallbackReason =
      globalFullBodyVisibleRatio < MIN_FULL_BODY_VISIBLE_RATIO
        ? 'BODY_NOT_FULLY_VISIBLE'
        : globalFeetVisibleRatio < MIN_FEET_VISIBLE_RATIO
          ? 'FEET_NOT_VISIBLE'
          : 'NO_STABLE_CALIBRATION';

    return buildResult(
      {
        ...debugBase,
        averageConfidence,
        fullBodyVisibleRatio: globalFullBodyVisibleRatio,
        feetVisibleRatio: globalFeetVisibleRatio,
      },
      [],
      'LOW',
      [
        fallbackReason === 'BODY_NOT_FULLY_VISIBLE'
          ? 'INSUFFICIENT_FULL_BODY_VISIBILITY'
          : fallbackReason === 'FEET_NOT_VISIBLE'
            ? 'INSUFFICIENT_FEET_VISIBILITY'
            : 'UNSTABLE_CALIBRATION',
      ],
      fallbackReason,
    );
  }

  const attempt = selection.evaluation;
  if (attempt.fullBodyVisibleRatio < 0.8) {
    preFlags.push('PARTIAL_FULL_BODY_VISIBILITY');
  }

  const visibilityRatio = Math.min(attempt.fullBodyVisibleRatio, attempt.feetVisibleRatio);
  const qualityFlags = [...preFlags];
  const quality = scoreQuality(
    averageConfidence,
    attempt.uncertaintyRatio,
    visibilityRatio,
    fpsScore,
    attempt.baseline.stability,
    attempt.averageContactReliability,
    qualityFlags,
  );

  const debug = buildDebugFromAttempt(
    attempt,
    frames.length,
    videoDurationMs,
    videoFps,
    sampleFps,
    options.playbackVideoFps,
    options.playbackSampleFps,
    playbackDurationMs,
    options.captureDurationMs ?? captureDurationMs,
    options.timingMode,
    options.timingConfidence,
    options.captureTimeScale,
    options.hasTimeSegments,
    options.usedOriginalAsset,
    options.usedPlaybackAsset,
    options.originalDurationMs,
    options.timebaseSource,
    averageConfidence,
  );

  if (!isTrustedTimingMode(options.timingMode)) {
    return buildResult(
      debug,
      attempt.phaseTimeline,
      'LOW',
      [...qualityFlags, 'TIMING_AMBIGUOUS'],
      'TIMING_AMBIGUOUS',
      attempt.takeoffMs,
      attempt.landingMs,
      null,
      null,
      null,
      null,
    );
  }

  if (attempt.invalidReason) {
    const invalidFlightMs =
      attempt.invalidReason === 'EVENT_ORDER_INVALID'
        ? null
        : attempt.takeoffCaptureMs !== null && attempt.landingCaptureMs !== null
          ? attempt.landingCaptureMs - attempt.takeoffCaptureMs
          : null;
    return buildResult(
      debug,
      attempt.phaseTimeline,
      attempt.invalidReason === 'EXCESS_HORIZONTAL_MOTION' ||
        attempt.invalidReason === 'AIRTIME_OUT_OF_RANGE' ||
        attempt.invalidReason === 'EVENT_ORDER_INVALID'
        ? 'LOW'
        : quality,
      attempt.invalidReason === 'EXCESS_HORIZONTAL_MOTION'
        ? [...qualityFlags, 'EXCESS_HORIZONTAL_MOTION']
        : attempt.invalidReason === 'AIRTIME_OUT_OF_RANGE'
          ? [...qualityFlags, 'AIRTIME_OUT_OF_RANGE']
          : attempt.invalidReason === 'EVENT_ORDER_INVALID'
            ? [...qualityFlags, 'EVENT_ORDER_INVALID']
          : qualityFlags,
      attempt.invalidReason,
      attempt.takeoffMs,
      attempt.landingMs,
      attempt.invalidReason === 'EVENT_ORDER_INVALID' ? null : attempt.takeoffCaptureMs,
      attempt.invalidReason === 'EVENT_ORDER_INVALID' ? null : attempt.landingCaptureMs,
      invalidFlightMs,
    );
  }

  const physicalTakeoffMs = (attempt.takeoffCaptureMs ?? attempt.takeoffMs)!;
  const physicalLandingMs = (attempt.landingCaptureMs ?? attempt.landingMs)!;
  const flightMs = physicalLandingMs - physicalTakeoffMs;
  const heightCm = heightFromFlightTime(flightMs);

  return buildResult(
    debug,
    attempt.phaseTimeline,
    quality,
    qualityFlags,
    undefined,
    attempt.takeoffMs,
    attempt.landingMs,
    physicalTakeoffMs,
    physicalLandingMs,
    flightMs,
    heightCm,
    `Detected takeoff at ${attempt.takeoffMs?.toFixed(1)} ms playback and landing at ${attempt.landingMs?.toFixed(1)} ms playback.`,
  );
}

export function findFrameAtTime(
  frames: JumpLandmarkFrame[],
  timestampMs: number,
): JumpLandmarkFrame | null {
  if (frames.length === 0) return null;
  let best = frames[0];
  let bestDelta = Math.abs(best.timestampMs - timestampMs);
  for (const frame of frames) {
    const delta = Math.abs(frame.timestampMs - timestampMs);
    if (delta < bestDelta) {
      best = frame;
      bestDelta = delta;
    }
  }
  return best;
}

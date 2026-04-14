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
  heightFromFlightTime,
} from './jumpCalc.ts';

const MIN_CONFIDENCE = 0.2;
const CALIBRATION_WINDOW_MS = 750;
const TAKEOFF_THRESHOLD = 0.04;
const LANDING_THRESHOLD = 0.02;
const TAKEOFF_CONFIRM_FRAMES = 2;
const LANDING_CONFIRM_FRAMES = 2;
const MIN_FULL_BODY_VISIBLE_RATIO = 0.6;
/** Allow this many consecutive untracked frames without resetting takeoff
 *  confirmation.  Fast upward motion commonly causes brief tracking loss
 *  that should not invalidate the takeoff signal. */
const MAX_TAKEOFF_GAP_FRAMES = 4;
const MIN_FLIGHT_MS = 180;
const MAX_FLIGHT_MS = 900;
const MAX_CALIBRATION_CENTER_RANGE = 0.12;
const MAX_ALLOWED_HORIZONTAL_DRIFT = 0.18;
const MAX_CALIBRATION_NOISE = 0.025;

interface AnalysisOptions {
  videoDurationMs?: number;
  videoFps?: number;
  sampleFps?: number;
  playbackVideoFps?: number;
  playbackSampleFps?: number;
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
  /** At least one toe (foot index) or ankle is visible. */
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
  calibrationEndMs: number;
  stability: number;
}

interface CalibrationCandidate {
  frames: FrameSignals[];
  endMs: number;
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

// Keep EMA effectively pass-through in the timing path. A slower EMA reduces
// jitter, but it also shifts touchdown later by one or more 240 fps frames.
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
  // Feet visible if at least one toe or ankle is tracked.
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
  };
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

function collectCalibrationFrames(window: FrameSignals[]): FrameSignals[] {
  return window.filter((sample) => sample.fullBodyVisible);
}

function buildBaselineFromCandidate(candidate: CalibrationCandidate): Baseline | null {
  const calibrationFrames = candidate.frames;

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
    calibrationEndMs: candidate.endMs,
    stability,
  };
}

function buildBaseline(samples: FrameSignals[]): Baseline | null {
  if (samples.length === 0) return null;

  for (let startIndex = 0; startIndex < samples.length; startIndex += 1) {
    const startSample = samples[startIndex];
    if (!startSample.fullBodyVisible) {
      continue;
    }
    const windowEndLimit = startSample.timestampMs + CALIBRATION_WINDOW_MS;
    const window = samples.filter(
      (sample) =>
        sample.timestampMs >= startSample.timestampMs &&
        sample.timestampMs <= windowEndLimit,
    );
    const calibrationFrames = collectCalibrationFrames(window);
    const endSample = last(calibrationFrames);
    if (calibrationFrames.length < 6 || !endSample) {
      continue;
    }

    const baseline = buildBaselineFromCandidate({
      frames: calibrationFrames,
      endMs: endSample.timestampMs,
    });

    if (baseline) {
      return baseline;
    }
  }

  return null;
}

function emptyDebug(
  analyzedFrameCount: number,
  videoDurationMs: number,
  videoFps: number,
  sampleFps: number,
  playbackVideoFps?: number,
  playbackSampleFps?: number,
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
    slowMotionScaleFactor:
      playbackVideoFps && videoFps && videoFps > 0 ? playbackVideoFps / videoFps : undefined,
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
  }
}

function scoreQuality(
  averageConfidence: number,
  uncertaintyRatio: number,
  visibilityRatio: number,
  fpsScore: number,
  calibrationStability: number,
  flags: string[],
): JumpQuality {
  const confidenceScore = clamp01((averageConfidence - 0.22) / 0.5);
  const uncertaintyScore = clamp01(1 - uncertaintyRatio / 0.25);
  const visibilityScore = clamp01((visibilityRatio - 0.55) / 0.4);
  const stabilityScore = clamp01(1 - calibrationStability / 0.03);
  const total =
    confidenceScore * 0.3 +
    uncertaintyScore * 0.25 +
    visibilityScore * 0.2 +
    fpsScore * 0.15 +
    stabilityScore * 0.1;

  if (confidenceScore < 0.35) flags.push('LOW_CONFIDENCE');
  if (uncertaintyRatio > 0.12) flags.push('UNCERTAIN_TRACKING');
  if (visibilityRatio < 0.75) flags.push('PARTIAL_FEET_VISIBILITY');
  if (fpsScore < 0.75) flags.push('LOW_SAMPLE_FPS');
  if (calibrationStability > 0.018) flags.push('CALIBRATION_NOISE');

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

export function explainJumpAnalysis(result: JumpAnalysisResult): string {
  return result.summary;
}

export function analyzeJumpLandmarks(
  frames: JumpLandmarkFrame[],
  options: AnalysisOptions = {},
): JumpAnalysisResult {
  const videoDurationMs =
    options.videoDurationMs ??
    Math.max(0, ...frames.map((frame) => frame.timestampMs), 0);
  const videoFps = options.videoFps ?? options.sampleFps ?? 60;
  const sampleFps = options.sampleFps ?? options.videoFps ?? 60;
  const debugBase = emptyDebug(
    frames.length,
    videoDurationMs,
    videoFps,
    sampleFps,
    options.playbackVideoFps,
    options.playbackSampleFps,
  );

  if (frames.length === 0) {
    return buildResult(debugBase, [], 'LOW', ['NO_ANALYZED_FRAMES'], 'NO_PERSON');
  }

  const minConfidence = options.minConfidence ?? MIN_CONFIDENCE;
  const rawSignals = frames.map((frame) => toSignals(frame, minConfidence));
  const signals = applyEma(applyMedianFilter(rawSignals));
  const averageConfidence = average(signals.map((signal) => signal.avgConfidence));
  const fullBodyVisibleRatio =
    signals.filter((signal) => signal.fullBodyVisible).length / signals.length;
  const feetVisibleRatio =
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
        fullBodyVisibleRatio,
        feetVisibleRatio,
      },
      [],
      'LOW',
      ['MULTI_PERSON_INPUT'],
      'MULTIPLE_PEOPLE',
    );
  }

  if (fullBodyVisibleRatio < MIN_FULL_BODY_VISIBLE_RATIO) {
    return buildResult(
      {
        ...debugBase,
        averageConfidence,
        fullBodyVisibleRatio,
        feetVisibleRatio,
      },
      [],
      'LOW',
      ['INSUFFICIENT_FULL_BODY_VISIBILITY'],
      'BODY_NOT_FULLY_VISIBLE',
    );
  }

  if (feetVisibleRatio < 0.45) {
    return buildResult(
      {
        ...debugBase,
        averageConfidence,
        fullBodyVisibleRatio,
        feetVisibleRatio,
      },
      [],
      'LOW',
      ['INSUFFICIENT_FEET_VISIBILITY'],
      'FEET_NOT_VISIBLE',
    );
  }

  const baseline = buildBaseline(signals);
  if (!baseline) {
    return buildResult(
      {
        ...debugBase,
        averageConfidence,
        fullBodyVisibleRatio,
        feetVisibleRatio,
      },
      [],
      'LOW',
      ['UNSTABLE_CALIBRATION'],
      'NO_STABLE_CALIBRATION',
    );
  }

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
  let lastGroundContactMs: number | null = null;
  let lastGroundContactCaptureMs: number | null = null;
  let takeoffCandidateLastContactMs: number | null = null;
  let takeoffCandidateLastContactCaptureMs: number | null = null;
  let airborne = false;
  let uncertainFrames = 0;
  let trackingGap = 0;
  let maxHorizontalDrift = 0;

  const phaseTimeline: JumpPhaseSample[] = [];
  const torsoScale = Math.max(0.08, baseline.torso);

  for (const signal of signals) {
    const horizontalDrift =
      signal.centerX === null ? null : Math.abs(signal.centerX - baseline.centerX);
    if (horizontalDrift !== null) {
      maxHorizontalDrift = Math.max(maxHorizontalDrift, horizontalDrift);
    }

    const leftFootY = bestFootGroundY(signal.leftToeY, signal.leftHeelY, signal.leftAnkleY);
    const rightFootY = bestFootGroundY(signal.rightToeY, signal.rightHeelY, signal.rightAnkleY);
    const leftFootBaseline = baseline.leftGroundY;
    const rightFootBaseline = baseline.rightGroundY;

    const leftLift =
      leftFootY === null ? null : (leftFootBaseline - leftFootY) / torsoScale;
    const rightLift =
      rightFootY === null ? null : (rightFootBaseline - rightFootY) / torsoScale;
    const hipLift =
      signal.hipY === null ? null : (baseline.hipY - signal.hipY) / torsoScale;

    let phase: JumpContactPhase = 'UNCERTAIN';
    const bothAirborne =
      leftLift !== null &&
      rightLift !== null &&
      leftLift > TAKEOFF_THRESHOLD &&
      rightLift > TAKEOFF_THRESHOLD;
    const oneFootGrounded =
      (leftLift !== null && leftLift <= LANDING_THRESHOLD) ||
      (rightLift !== null && rightLift <= LANDING_THRESHOLD);

    if (!signal.feetVisible) {
      uncertainFrames += 1;
      trackingGap += 1;
      if (!airborne) {
        if (takeoffConfirm > 0 && trackingGap <= MAX_TAKEOFF_GAP_FRAMES) {
          takeoffConfirm += 1;
          if (takeoffConfirm >= TAKEOFF_CONFIRM_FRAMES) {
            takeoffMs = takeoffCandidateLastContactMs ?? takeoffCandidateStart;
            takeoffCaptureMs =
              takeoffCandidateLastContactCaptureMs ?? takeoffCandidateStartCaptureMs;
            airborne = true;
          }
        } else if (trackingGap > MAX_TAKEOFF_GAP_FRAMES) {
          takeoffConfirm = 0;
          takeoffCandidateStart = null;
          takeoffCandidateStartCaptureMs = null;
          takeoffCandidateLastContactMs = null;
          takeoffCandidateLastContactCaptureMs = null;
        }
      }
      landingConfirm = 0;
      landingCandidateStart = null;
      landingCandidateStartCaptureMs = null;
      phase = 'UNCERTAIN';
    } else {
      trackingGap = 0;
      if (!airborne) {
        phase = bothAirborne ? 'AIRBORNE' : oneFootGrounded ? 'GROUND_CONTACT' : 'UNCERTAIN';
        if (signal.timestampMs >= baseline.calibrationEndMs && bothAirborne) {
          takeoffCandidateStart ??= signal.timestampMs;
          takeoffCandidateStartCaptureMs ??= signal.captureTimestampMs;
          takeoffCandidateLastContactMs ??= lastGroundContactMs ?? signal.timestampMs;
          takeoffCandidateLastContactCaptureMs ??=
            lastGroundContactCaptureMs ?? signal.captureTimestampMs;
          takeoffConfirm += 1;
          if (takeoffConfirm >= TAKEOFF_CONFIRM_FRAMES) {
            takeoffMs = takeoffCandidateLastContactMs ?? takeoffCandidateStart;
            takeoffCaptureMs =
              takeoffCandidateLastContactCaptureMs ?? takeoffCandidateStartCaptureMs;
            airborne = true;
          }
        } else if (phase === 'GROUND_CONTACT') {
          lastGroundContactMs = signal.timestampMs;
          lastGroundContactCaptureMs = signal.captureTimestampMs;
          takeoffConfirm = 0;
          takeoffCandidateStart = null;
          takeoffCandidateStartCaptureMs = null;
          takeoffCandidateLastContactMs = null;
          takeoffCandidateLastContactCaptureMs = null;
        } else {
          uncertainFrames += 1;
          takeoffConfirm = 0;
          takeoffCandidateStart = null;
          takeoffCandidateStartCaptureMs = null;
          takeoffCandidateLastContactMs = null;
          takeoffCandidateLastContactCaptureMs = null;
        }
      } else {
        if (oneFootGrounded) {
          phase = 'GROUND_CONTACT';
          landingCandidateStart ??= signal.timestampMs;
          landingCandidateStartCaptureMs ??= signal.captureTimestampMs;
          landingConfirm += 1;
          if (landingConfirm >= LANDING_CONFIRM_FRAMES && landingMs === null) {
            landingMs = landingCandidateStart;
            landingCaptureMs = landingCandidateStartCaptureMs;
            airborne = false;
          }
        } else {
          phase = bothAirborne ? 'AIRBORNE' : 'UNCERTAIN';
          if (phase === 'UNCERTAIN') {
            uncertainFrames += 1;
          }
          landingConfirm = 0;
          landingCandidateStart = null;
          landingCandidateStartCaptureMs = null;
        }
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
    });
  }

  const uncertaintyRatio = phaseTimeline.length > 0 ? uncertainFrames / phaseTimeline.length : 1;
  const qualityFlags = [...preFlags];
  const visibilityRatio = Math.min(fullBodyVisibleRatio, feetVisibleRatio);
  if (fullBodyVisibleRatio < 0.8) {
    qualityFlags.push('PARTIAL_FULL_BODY_VISIBILITY');
  }
  const quality = scoreQuality(
    averageConfidence,
    uncertaintyRatio,
    visibilityRatio,
    fpsScore,
    baseline.stability,
    qualityFlags,
  );

  const debug: JumpAnalysisDebug = {
    calibrationEndMs: baseline.calibrationEndMs,
    baseline,
    analyzedFrameCount: frames.length,
    videoDurationMs,
    videoFps,
    sampleFps,
    playbackVideoFps: options.playbackVideoFps,
    playbackSampleFps: options.playbackSampleFps,
    slowMotionScaleFactor:
      options.playbackVideoFps && videoFps > 0
        ? options.playbackVideoFps / videoFps
        : undefined,
    averageConfidence,
    uncertaintyRatio,
    fullBodyVisibleRatio,
    feetVisibleRatio,
    maxHorizontalDrift,
    calibrationStability: baseline.stability,
  };

  if (maxHorizontalDrift > MAX_ALLOWED_HORIZONTAL_DRIFT) {
    return buildResult(
      debug,
      phaseTimeline,
      'LOW',
      [...qualityFlags, 'EXCESS_HORIZONTAL_MOTION'],
      'EXCESS_HORIZONTAL_MOTION',
    );
  }

  if (takeoffMs === null) {
    return buildResult(debug, phaseTimeline, quality, qualityFlags, 'NO_TAKEOFF');
  }

  if (landingMs === null) {
    return buildResult(debug, phaseTimeline, quality, qualityFlags, 'NO_LANDING', takeoffMs);
  }

  const physicalTakeoffMs = takeoffCaptureMs ?? takeoffMs;
  const physicalLandingMs = landingCaptureMs ?? landingMs;
  const flightMs = physicalLandingMs - physicalTakeoffMs;
  if (flightMs < MIN_FLIGHT_MS || flightMs > MAX_FLIGHT_MS) {
    return buildResult(
      debug,
      phaseTimeline,
      'LOW',
      [...qualityFlags, 'AIRTIME_OUT_OF_RANGE'],
      'AIRTIME_OUT_OF_RANGE',
      takeoffMs,
      landingMs,
      physicalTakeoffMs,
      physicalLandingMs,
      flightMs,
    );
  }

  const heightCm = heightFromFlightTime(flightMs);
  return buildResult(
    debug,
    phaseTimeline,
    quality,
    qualityFlags,
    undefined,
    takeoffMs,
    landingMs,
    physicalTakeoffMs,
    physicalLandingMs,
    flightMs,
    heightCm,
    `Flight time ${flightMs.toFixed(1)} ms produced ${heightCm.toFixed(1)} cm.`,
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

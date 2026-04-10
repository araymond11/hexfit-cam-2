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
const MIN_FLIGHT_MS = 180;
const MAX_FLIGHT_MS = 900;
const MAX_CALIBRATION_CENTER_RANGE = 0.12;
const MAX_ALLOWED_HORIZONTAL_DRIFT = 0.18;
const MAX_CALIBRATION_NOISE = 0.025;

interface AnalysisOptions {
  videoDurationMs?: number;
  videoFps?: number;
  sampleFps?: number;
  personCountSummary?: JumpPersonCountSummary;
  minConfidence?: number;
}

interface FrameSignals {
  frameIndex: number;
  timestampMs: number;
  avgConfidence: number;
  personCount: number;
  fullBodyVisible: boolean;
  feetVisible: boolean;
  leftAnkleY: number | null;
  rightAnkleY: number | null;
  hipY: number | null;
  shoulderY: number | null;
  torso: number | null;
  centerX: number | null;
}

interface Baseline {
  leftAnkleY: number;
  rightAnkleY: number;
  hipY: number;
  torso: number;
  centerX: number;
  calibrationEndMs: number;
  stability: number;
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
  const leftSeries = samples.map((sample) => sample.leftAnkleY);
  const rightSeries = samples.map((sample) => sample.rightAnkleY);
  const hipSeries = samples.map((sample) => sample.hipY);
  const torsoSeries = samples.map((sample) => sample.torso);
  const centerSeries = samples.map((sample) => sample.centerX);

  return samples.map((sample, index) => ({
    ...sample,
    leftAnkleY: medianNullable(leftSeries, index),
    rightAnkleY: medianNullable(rightSeries, index),
    hipY: medianNullable(hipSeries, index),
    torso: medianNullable(torsoSeries, index),
    centerX: medianNullable(centerSeries, index),
  }));
}

function applyEma(samples: FrameSignals[], alpha = 0.58): FrameSignals[] {
  let leftPrev: number | null = null;
  let rightPrev: number | null = null;
  let hipPrev: number | null = null;
  let torsoPrev: number | null = null;
  let centerPrev: number | null = null;

  return samples.map((sample) => {
    const nextLeft =
      sample.leftAnkleY === null
        ? null
        : leftPrev === null
          ? sample.leftAnkleY
          : alpha * sample.leftAnkleY + (1 - alpha) * leftPrev;
    const nextRight =
      sample.rightAnkleY === null
        ? null
        : rightPrev === null
          ? sample.rightAnkleY
          : alpha * sample.rightAnkleY + (1 - alpha) * rightPrev;
    const nextHip =
      sample.hipY === null
        ? null
        : hipPrev === null
          ? sample.hipY
          : alpha * sample.hipY + (1 - alpha) * hipPrev;
    const nextTorso =
      sample.torso === null
        ? null
        : torsoPrev === null
          ? sample.torso
          : alpha * sample.torso + (1 - alpha) * torsoPrev;
    const nextCenter =
      sample.centerX === null
        ? null
        : centerPrev === null
          ? sample.centerX
          : alpha * sample.centerX + (1 - alpha) * centerPrev;

    leftPrev = nextLeft;
    rightPrev = nextRight;
    hipPrev = nextHip;
    torsoPrev = nextTorso;
    centerPrev = nextCenter;

    return {
      ...sample,
      leftAnkleY: nextLeft,
      rightAnkleY: nextRight,
      hipY: nextHip,
      torso: nextTorso,
      centerX: nextCenter,
    };
  });
}

function toSignals(frame: JumpLandmarkFrame, minConfidence: number): FrameSignals {
  const ankleConfidence = Math.max(0.16, minConfidence - 0.04);
  const ls = pointOrNull(frame, KP.LEFT_SHOULDER, minConfidence);
  const rs = pointOrNull(frame, KP.RIGHT_SHOULDER, minConfidence);
  const lh = pointOrNull(frame, KP.LEFT_HIP, minConfidence);
  const rh = pointOrNull(frame, KP.RIGHT_HIP, minConfidence);
  const la = pointOrNull(frame, KP.LEFT_ANKLE, ankleConfidence);
  const ra = pointOrNull(frame, KP.RIGHT_ANKLE, ankleConfidence);

  const shoulderY = midpointY(ls, rs);
  const hipY = midpointY(lh, rh);
  const torso =
    shoulderY !== null && hipY !== null ? Math.max(0.001, hipY - shoulderY) : null;
  const centerX = midpointX(midpointPoint(ls, rs), midpointPoint(lh, rh));
  const leftAnkleY = la?.y ?? null;
  const rightAnkleY = ra?.y ?? null;
  const feetVisible = leftAnkleY !== null || rightAnkleY !== null;
  const fullBodyVisible =
    shoulderY !== null &&
    hipY !== null &&
    feetVisible &&
    shoulderY > 0.02 &&
    Math.max(leftAnkleY ?? 0, rightAnkleY ?? 0) < 0.995;

  return {
    frameIndex: frame.frameIndex,
    timestampMs: frame.timestampMs,
    avgConfidence: frame.avgConfidence,
    personCount: frame.personCount ?? 1,
    fullBodyVisible,
    feetVisible,
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

function buildBaseline(samples: FrameSignals[]): Baseline | null {
  const calibrationFrames = samples.filter(
    (sample) =>
      sample.timestampMs <= CALIBRATION_WINDOW_MS &&
      sample.fullBodyVisible &&
      sample.feetVisible &&
      sample.torso !== null &&
      sample.hipY !== null,
  );

  if (calibrationFrames.length < 6) return null;

  const torsoValues = calibrationFrames.map((sample) => sample.torso!);
  const leftValues = calibrationFrames.map(
    (sample) => sample.leftAnkleY ?? sample.rightAnkleY!,
  );
  const rightValues = calibrationFrames.map(
    (sample) => sample.rightAnkleY ?? sample.leftAnkleY!,
  );
  const hipValues = calibrationFrames.map((sample) => sample.hipY!);
  const centerValues = calibrationFrames
    .map((sample) => sample.centerX)
    .filter((value): value is number => value !== null);

  if (centerValues.length < 6) return null;

  const torso = median(torsoValues);
  const averageAnkleSeries = calibrationFrames.map((sample) => {
    const left = sample.leftAnkleY ?? sample.rightAnkleY!;
    const right = sample.rightAnkleY ?? sample.leftAnkleY!;
    return (left + right) / 2;
  });
  const stability = stdDev(averageAnkleSeries) / Math.max(0.001, torso);
  const centerRange = Math.max(...centerValues) - Math.min(...centerValues);

  if (stability > MAX_CALIBRATION_NOISE || centerRange > MAX_CALIBRATION_CENTER_RANGE) {
    return null;
  }

  return {
    leftAnkleY: median(leftValues),
    rightAnkleY: median(rightValues),
    hipY: median(hipValues),
    torso,
    centerX: median(centerValues),
    calibrationEndMs: CALIBRATION_WINDOW_MS,
    stability,
  };
}

function emptyDebug(
  analyzedFrameCount: number,
  videoDurationMs: number,
  videoFps: number,
  sampleFps: number,
): JumpAnalysisDebug {
  const baseline: JumpBaselineDebug = {
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
      return 'The full body was not visible for enough of the clip.';
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
  if (visibilityRatio < 0.75) flags.push('PARTIAL_BODY_VISIBILITY');
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
  flightMs: number | null = null,
  heightCm: number | null = null,
  summary?: string,
): JumpAnalysisResult {
  return {
    takeoffMs,
    landingMs,
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
        : `Takeoff at ${takeoffMs?.toFixed(1)} ms, landing at ${landingMs?.toFixed(1)} ms.`),
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
  const debugBase = emptyDebug(frames.length, videoDurationMs, videoFps, sampleFps);

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

  if (fullBodyVisibleRatio < 0.4) {
    return buildResult(
      {
        ...debugBase,
        averageConfidence,
        fullBodyVisibleRatio,
        feetVisibleRatio,
      },
      [],
      'LOW',
      ['INSUFFICIENT_BODY_VISIBILITY'],
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
  let landingCandidateStart: number | null = null;
  let takeoffConfirm = 0;
  let landingConfirm = 0;
  let takeoffMs: number | null = null;
  let landingMs: number | null = null;
  let airborne = false;
  let uncertainFrames = 0;
  let maxHorizontalDrift = 0;

  const phaseTimeline: JumpPhaseSample[] = [];
  const torsoScale = Math.max(0.08, baseline.torso);

  for (const signal of signals) {
    const horizontalDrift =
      signal.centerX === null ? null : Math.abs(signal.centerX - baseline.centerX);
    if (horizontalDrift !== null) {
      maxHorizontalDrift = Math.max(maxHorizontalDrift, horizontalDrift);
    }

    const leftLift =
      signal.leftAnkleY === null ? null : (baseline.leftAnkleY - signal.leftAnkleY) / torsoScale;
    const rightLift =
      signal.rightAnkleY === null ? null : (baseline.rightAnkleY - signal.rightAnkleY) / torsoScale;
    const hipLift =
      signal.hipY === null ? null : (baseline.hipY - signal.hipY) / torsoScale;

    let phase: JumpContactPhase = 'UNCERTAIN';
    const bothLifted =
      leftLift !== null &&
      rightLift !== null &&
      leftLift > TAKEOFF_THRESHOLD &&
      rightLift > TAKEOFF_THRESHOLD;
    const oneFootGrounded =
      (leftLift !== null && leftLift <= LANDING_THRESHOLD) ||
      (rightLift !== null && rightLift <= LANDING_THRESHOLD);

    if (!signal.fullBodyVisible || !signal.feetVisible || signal.torso === null) {
      uncertainFrames += 1;
      if (!airborne) {
        takeoffConfirm = 0;
        takeoffCandidateStart = null;
      }
      landingConfirm = 0;
      landingCandidateStart = null;
      phase = 'UNCERTAIN';
    } else if (!airborne) {
      phase = bothLifted ? 'AIRBORNE' : 'GROUND_CONTACT';
      if (signal.timestampMs >= baseline.calibrationEndMs && bothLifted) {
        takeoffCandidateStart ??= signal.timestampMs;
        takeoffConfirm += 1;
        if (takeoffConfirm >= TAKEOFF_CONFIRM_FRAMES) {
          takeoffMs = takeoffCandidateStart;
          airborne = true;
        }
      } else {
        takeoffConfirm = 0;
        takeoffCandidateStart = null;
      }
    } else {
      if (oneFootGrounded) {
        phase = 'GROUND_CONTACT';
        landingCandidateStart ??= signal.timestampMs;
        landingConfirm += 1;
        if (landingConfirm >= LANDING_CONFIRM_FRAMES && landingMs === null) {
          landingMs = landingCandidateStart;
          airborne = false;
        }
      } else {
        phase = bothLifted ? 'AIRBORNE' : 'UNCERTAIN';
        landingConfirm = 0;
        landingCandidateStart = null;
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
  const quality = scoreQuality(
    averageConfidence,
    uncertaintyRatio,
    Math.min(fullBodyVisibleRatio, feetVisibleRatio),
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

  const flightMs = landingMs - takeoffMs;
  if (flightMs < MIN_FLIGHT_MS || flightMs > MAX_FLIGHT_MS) {
    return buildResult(
      debug,
      phaseTimeline,
      'LOW',
      [...qualityFlags, 'AIRTIME_OUT_OF_RANGE'],
      'AIRTIME_OUT_OF_RANGE',
      takeoffMs,
      landingMs,
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

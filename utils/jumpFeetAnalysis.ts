import {
  type JumpContactPhase,
  type JumpFeetAnalysisDebug,
  type JumpFeetAnalysisResult,
  type JumpFeetBaselineDebug,
  type JumpFeetPhaseSample,
  type JumpFootBox,
  type JumpFootContactFrame,
  type JumpInvalidReason,
  type JumpPersonCountSummary,
  type JumpQuality,
  heightFromFlightTime,
} from './jumpCalc.ts';

const CALIBRATION_WINDOW_MS = 750;
const TAKEOFF_LIFT_THRESHOLD = 0.2;
const LANDING_LIFT_THRESHOLD = 0.08;
const TAKEOFF_CONTACT_MAX = 0.35;
const LANDING_CONTACT_MIN = 0.55;
const TAKEOFF_CONFIRM_FRAMES = 2;
const LANDING_CONFIRM_FRAMES = 2;
const MIN_FLIGHT_MS = 120;
const MAX_FLIGHT_MS = 900;
const MAX_CALIBRATION_CENTER_RANGE = 0.22;
const MAX_ALLOWED_HORIZONTAL_DRIFT = 0.24;
const MAX_CALIBRATION_NOISE = 0.16;

interface AnalysisOptions {
  videoDurationMs?: number;
  videoFps?: number;
  sampleFps?: number;
  personCountSummary?: JumpPersonCountSummary;
}

interface FrameSignals {
  frameIndex: number;
  timestampMs: number;
  avgConfidence: number;
  personCount: number;
  leftFootBox: JumpFootBox | null;
  rightFootBox: JumpFootBox | null;
  leftBottomY: number | null;
  rightBottomY: number | null;
  leftHeight: number | null;
  rightHeight: number | null;
  leftContactScore: number | null;
  rightContactScore: number | null;
  floorY: number | null;
  centerX: number | null;
  feetVisible: boolean;
  dualFeetVisible: boolean;
}

interface Baseline {
  leftBottomY: number;
  rightBottomY: number;
  floorY: number;
  leftFootHeight: number;
  rightFootHeight: number;
  centerX: number;
  calibrationEndMs: number;
  stability: number;
}

function clamp01(value: number): number {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function stdDev(values: number[]): number {
  if (values.length <= 1) return 0;
  const mean = average(values);
  const variance = average(values.map((value) => (value - mean) ** 2));
  return Math.sqrt(variance);
}

function centerXFromBoxes(left: JumpFootBox | null, right: JumpFootBox | null): number | null {
  const leftCenter = left ? left.x + left.width / 2 : null;
  const rightCenter = right ? right.x + right.width / 2 : null;
  if (leftCenter !== null && rightCenter !== null) return (leftCenter + rightCenter) / 2;
  if (leftCenter !== null) return leftCenter;
  if (rightCenter !== null) return rightCenter;
  return null;
}

function boxHeight(box: JumpFootBox | null): number | null {
  if (!box || !Number.isFinite(box.height) || box.height <= 0) return null;
  return box.height;
}

function toSignals(frame: JumpFootContactFrame): FrameSignals {
  const leftBottomY = frame.leftFootBottomY;
  const rightBottomY = frame.rightFootBottomY;
  const leftHeight = boxHeight(frame.leftFootBox);
  const rightHeight = boxHeight(frame.rightFootBox);
  const feetVisible = leftBottomY !== null || rightBottomY !== null;
  const dualFeetVisible = leftBottomY !== null && rightBottomY !== null;

  return {
    frameIndex: frame.frameIndex,
    timestampMs: frame.timestampMs,
    avgConfidence: frame.avgConfidence,
    personCount: frame.personCount ?? 1,
    leftFootBox: frame.leftFootBox,
    rightFootBox: frame.rightFootBox,
    leftBottomY,
    rightBottomY,
    leftHeight,
    rightHeight,
    leftContactScore: frame.leftContactScore,
    rightContactScore: frame.rightContactScore,
    floorY: frame.floorY,
    centerX: centerXFromBoxes(frame.leftFootBox, frame.rightFootBox),
    feetVisible,
    dualFeetVisible,
  };
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
  const leftBottoms = samples.map((sample) => sample.leftBottomY);
  const rightBottoms = samples.map((sample) => sample.rightBottomY);
  const leftHeights = samples.map((sample) => sample.leftHeight);
  const rightHeights = samples.map((sample) => sample.rightHeight);
  const leftContacts = samples.map((sample) => sample.leftContactScore);
  const rightContacts = samples.map((sample) => sample.rightContactScore);
  const floorSeries = samples.map((sample) => sample.floorY);
  const centerSeries = samples.map((sample) => sample.centerX);

  return samples.map((sample, index) => ({
    ...sample,
    leftBottomY: medianNullable(leftBottoms, index),
    rightBottomY: medianNullable(rightBottoms, index),
    leftHeight: medianNullable(leftHeights, index),
    rightHeight: medianNullable(rightHeights, index),
    leftContactScore: medianNullable(leftContacts, index),
    rightContactScore: medianNullable(rightContacts, index),
    floorY: medianNullable(floorSeries, index),
    centerX: medianNullable(centerSeries, index),
  }));
}

function emaStep(current: number | null, prev: number | null, alpha: number): number | null {
  if (current === null) return null;
  if (prev === null) return current;
  return alpha * current + (1 - alpha) * prev;
}

function applyEma(samples: FrameSignals[], alpha = 0.55): FrameSignals[] {
  let leftBottomPrev: number | null = null;
  let rightBottomPrev: number | null = null;
  let leftHeightPrev: number | null = null;
  let rightHeightPrev: number | null = null;
  let leftContactPrev: number | null = null;
  let rightContactPrev: number | null = null;
  let floorPrev: number | null = null;
  let centerPrev: number | null = null;

  return samples.map((sample) => {
    const nextLeftBottom = emaStep(sample.leftBottomY, leftBottomPrev, alpha);
    const nextRightBottom = emaStep(sample.rightBottomY, rightBottomPrev, alpha);
    const nextLeftHeight = emaStep(sample.leftHeight, leftHeightPrev, alpha);
    const nextRightHeight = emaStep(sample.rightHeight, rightHeightPrev, alpha);
    const nextLeftContact = emaStep(sample.leftContactScore, leftContactPrev, alpha);
    const nextRightContact = emaStep(sample.rightContactScore, rightContactPrev, alpha);
    const nextFloor = emaStep(sample.floorY, floorPrev, alpha);
    const nextCenter = emaStep(sample.centerX, centerPrev, alpha);

    leftBottomPrev = nextLeftBottom;
    rightBottomPrev = nextRightBottom;
    leftHeightPrev = nextLeftHeight;
    rightHeightPrev = nextRightHeight;
    leftContactPrev = nextLeftContact;
    rightContactPrev = nextRightContact;
    floorPrev = nextFloor;
    centerPrev = nextCenter;

    return {
      ...sample,
      leftBottomY: nextLeftBottom,
      rightBottomY: nextRightBottom,
      leftHeight: nextLeftHeight,
      rightHeight: nextRightHeight,
      leftContactScore: nextLeftContact,
      rightContactScore: nextRightContact,
      floorY: nextFloor,
      centerX: nextCenter,
    };
  });
}

function buildBaseline(samples: FrameSignals[]): Baseline | null {
  const calibrationFrames = samples.filter(
    (sample) => sample.timestampMs <= CALIBRATION_WINDOW_MS && sample.feetVisible,
  );

  if (calibrationFrames.length < 6) return null;

  const leftBottoms = calibrationFrames
    .map((sample) => sample.leftBottomY)
    .filter((value): value is number => value !== null);
  const rightBottoms = calibrationFrames
    .map((sample) => sample.rightBottomY)
    .filter((value): value is number => value !== null);
  const leftHeights = calibrationFrames
    .map((sample) => sample.leftHeight)
    .filter((value): value is number => value !== null && value > 0.005);
  const rightHeights = calibrationFrames
    .map((sample) => sample.rightHeight)
    .filter((value): value is number => value !== null && value > 0.005);
  const floorYs = calibrationFrames
    .map((sample) => sample.floorY)
    .filter((value): value is number => value !== null);
  const centers = calibrationFrames
    .map((sample) => sample.centerX)
    .filter((value): value is number => value !== null);

  if (leftBottoms.length + rightBottoms.length < 8) return null;

  const leftBottomY = leftBottoms.length > 0 ? median(leftBottoms) : median(rightBottoms);
  const rightBottomY = rightBottoms.length > 0 ? median(rightBottoms) : median(leftBottoms);
  const leftFootHeight = leftHeights.length > 0 ? median(leftHeights) : 0.05;
  const rightFootHeight = rightHeights.length > 0 ? median(rightHeights) : 0.05;
  const avgFootHeight = Math.max(0.01, (leftFootHeight + rightFootHeight) / 2);
  const avgBottomSeries = calibrationFrames
    .map((sample) => {
      const left = sample.leftBottomY ?? sample.rightBottomY;
      const right = sample.rightBottomY ?? sample.leftBottomY;
      if (left === null || right === null) return null;
      return (left + right) / 2;
    })
    .filter((value): value is number => value !== null);
  const stability = avgBottomSeries.length > 1 ? stdDev(avgBottomSeries) / avgFootHeight : 1;

  if (centers.length >= 6) {
    const centerRange = Math.max(...centers) - Math.min(...centers);
    if (centerRange > MAX_CALIBRATION_CENTER_RANGE) {
      return null;
    }
  }

  if (stability > MAX_CALIBRATION_NOISE) {
    return null;
  }

  return {
    leftBottomY,
    rightBottomY,
    floorY:
      floorYs.length > 0
        ? median(floorYs)
        : Math.max(leftBottomY, rightBottomY),
    leftFootHeight,
    rightFootHeight,
    centerX: centers.length > 0 ? median(centers) : 0.5,
    calibrationEndMs: CALIBRATION_WINDOW_MS,
    stability,
  };
}

function emptyDebug(
  analyzedFrameCount: number,
  videoDurationMs: number,
  videoFps: number,
  sampleFps: number,
): JumpFeetAnalysisDebug {
  const baseline: JumpFeetBaselineDebug = {
    leftBottomY: 0,
    rightBottomY: 0,
    floorY: 0,
    leftFootHeight: 0,
    rightFootHeight: 0,
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
    feetVisibleRatio: 0,
    dualFootVisibleRatio: 0,
    maxHorizontalDrift: 0,
    calibrationStability: 0,
    averageContactReliability: 0,
  };
}

function invalidSummary(reason: JumpInvalidReason): string {
  switch (reason) {
    case 'NO_PERSON':
      return 'Aucun pied detecte dans la video importee.';
    case 'MULTIPLE_PEOPLE':
      return 'Plus d une personne a ete detectee dans la video.';
    case 'BODY_NOT_FULLY_VISIBLE':
      return 'Le cadrage actuel ne correspond pas au mode feet-only.';
    case 'FEET_NOT_VISIBLE':
      return 'Le detecteur n a pas reussi a suivre les deux pieds sur assez de frames pour mesurer le saut de facon fiable.';
    case 'NO_STABLE_CALIBRATION':
      return 'Le debut de video est trop instable pour etablir une ligne de contact au sol.';
    case 'NO_TAKEOFF':
      return 'Le decollage n a pas ete detecte.';
    case 'NO_LANDING':
      return 'L atterrissage n a pas ete detecte.';
    case 'AIRTIME_OUT_OF_RANGE':
      return 'Le temps de vol detecte est hors plage attendue.';
    case 'EXCESS_HORIZONTAL_MOTION':
      return 'Trop de deplacement horizontal a ete detecte pour une mesure fiable.';
  }
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
  const confidenceScore = clamp01((averageConfidence - 0.22) / 0.45);
  const uncertaintyScore = clamp01(1 - uncertaintyRatio / 0.3);
  const visibilityScore = clamp01((visibilityRatio - 0.55) / 0.35);
  const stabilityScore = clamp01(1 - calibrationStability / 0.3);
  const contactScore = clamp01((contactReliability - 0.35) / 0.45);

  const total =
    confidenceScore * 0.22 +
    uncertaintyScore * 0.24 +
    visibilityScore * 0.22 +
    fpsScore * 0.14 +
    stabilityScore * 0.08 +
    contactScore * 0.1;

  if (confidenceScore < 0.35) flags.push('LOW_CONFIDENCE');
  if (uncertaintyRatio > 0.18) flags.push('UNCERTAIN_CONTACT_TRACKING');
  if (visibilityRatio < 0.75) flags.push('PARTIAL_FEET_VISIBILITY');
  if (fpsScore < 0.75) flags.push('LOW_SAMPLE_FPS');
  if (calibrationStability > 0.2) flags.push('CALIBRATION_NOISE');
  if (contactReliability < 0.45) flags.push('WEAK_CONTACT_SIGNAL');

  if (total >= 0.72) return 'HIGH';
  if (total >= 0.45) return 'MEDIUM';
  return 'LOW';
}

function buildResult(
  debug: JumpFeetAnalysisDebug,
  phaseTimeline: JumpFeetPhaseSample[],
  quality: JumpQuality,
  qualityFlags: string[],
  invalidReason?: JumpInvalidReason,
  takeoffMs: number | null = null,
  landingMs: number | null = null,
  flightMs: number | null = null,
  heightCm: number | null = null,
  summary?: string,
): JumpFeetAnalysisResult {
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

function contactReliability(scores: Array<number | null>): number {
  const defined = scores.filter((value): value is number => value !== null);
  if (defined.length === 0) return 0;
  return average(defined.map((value) => Math.abs(value - 0.5) * 2));
}

export function explainJumpFeetAnalysis(result: JumpFeetAnalysisResult): string {
  return result.summary;
}

export function analyzeJumpFootContacts(
  frames: JumpFootContactFrame[],
  options: AnalysisOptions = {},
): JumpFeetAnalysisResult {
  const videoDurationMs =
    options.videoDurationMs ??
    Math.max(0, ...frames.map((frame) => frame.timestampMs), 0);
  const videoFps = options.videoFps ?? options.sampleFps ?? 60;
  const sampleFps = options.sampleFps ?? options.videoFps ?? 60;
  const debugBase = emptyDebug(frames.length, videoDurationMs, videoFps, sampleFps);

  if (frames.length === 0) {
    return buildResult(debugBase, [], 'LOW', ['NO_ANALYZED_FRAMES'], 'NO_PERSON');
  }

  const rawSignals = frames.map(toSignals);
  const signals = applyEma(applyMedianFilter(rawSignals));
  const averageConfidence = average(signals.map((signal) => signal.avgConfidence));
  const feetVisibleRatio =
    signals.filter((signal) => signal.feetVisible).length / signals.length;
  const dualFootVisibleRatio =
    signals.filter((signal) => signal.dualFeetVisible).length / signals.length;
  const observedPersonCount =
    options.personCountSummary ?? {
      analyzedFrames: signals.length,
      multiPersonFrames: signals.filter((signal) => signal.personCount > 1).length,
      maxPeople: Math.max(...signals.map((signal) => signal.personCount), 0),
    };

  const preFlags: string[] = [];
  const fpsScore = clamp01(Math.min(sampleFps, 240) / 240);
  const averageContactReliability = contactReliability(
    signals.flatMap((signal) => [signal.leftContactScore, signal.rightContactScore]),
  );

  if (observedPersonCount.maxPeople > 1 || observedPersonCount.multiPersonFrames > 0) {
    return buildResult(
      {
        ...debugBase,
        averageConfidence,
        feetVisibleRatio,
        dualFootVisibleRatio,
        averageContactReliability,
      },
      [],
      'LOW',
      ['MULTI_PERSON_INPUT'],
      'MULTIPLE_PEOPLE',
    );
  }

  if (feetVisibleRatio < 0.55 || dualFootVisibleRatio < 0.35) {
    return buildResult(
      {
        ...debugBase,
        averageConfidence,
        feetVisibleRatio,
        dualFootVisibleRatio,
        averageContactReliability,
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
        feetVisibleRatio,
        dualFootVisibleRatio,
        averageContactReliability,
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
  let lastGroundContactMs: number | null = null;
  let takeoffCandidateLastContactMs: number | null = null;
  let airborne = false;
  let uncertainFrames = 0;
  let maxHorizontalDrift = 0;

  const phaseTimeline: JumpFeetPhaseSample[] = [];

  for (const signal of signals) {
    const horizontalDrift =
      signal.centerX === null ? null : Math.abs(signal.centerX - baseline.centerX);
    if (horizontalDrift !== null) {
      maxHorizontalDrift = Math.max(maxHorizontalDrift, horizontalDrift);
    }

    const leftLift =
      signal.leftBottomY === null
        ? null
        : (baseline.leftBottomY - signal.leftBottomY) / Math.max(0.01, baseline.leftFootHeight);
    const rightLift =
      signal.rightBottomY === null
        ? null
        : (baseline.rightBottomY - signal.rightBottomY) / Math.max(0.01, baseline.rightFootHeight);

    const leftGrounded =
      signal.leftBottomY !== null &&
      ((signal.leftContactScore !== null && signal.leftContactScore >= LANDING_CONTACT_MIN) ||
        (leftLift !== null && leftLift <= LANDING_LIFT_THRESHOLD));
    const rightGrounded =
      signal.rightBottomY !== null &&
      ((signal.rightContactScore !== null && signal.rightContactScore >= LANDING_CONTACT_MIN) ||
        (rightLift !== null && rightLift <= LANDING_LIFT_THRESHOLD));
    const leftClear =
      signal.leftBottomY !== null &&
      ((signal.leftContactScore !== null && signal.leftContactScore <= TAKEOFF_CONTACT_MAX) ||
        (leftLift !== null && leftLift > TAKEOFF_LIFT_THRESHOLD));
    const rightClear =
      signal.rightBottomY !== null &&
      ((signal.rightContactScore !== null && signal.rightContactScore <= TAKEOFF_CONTACT_MAX) ||
        (rightLift !== null && rightLift > TAKEOFF_LIFT_THRESHOLD));

    let phase: JumpContactPhase = 'UNCERTAIN';

    if (!signal.feetVisible) {
      uncertainFrames += 1;
      takeoffConfirm = 0;
      landingConfirm = 0;
      takeoffCandidateStart = null;
      takeoffCandidateLastContactMs = null;
      landingCandidateStart = null;
      phase = 'UNCERTAIN';
    } else if (!airborne) {
      const bothAirborne = signal.dualFeetVisible && leftClear && rightClear;
      phase = bothAirborne ? 'AIRBORNE' : leftGrounded || rightGrounded ? 'GROUND_CONTACT' : 'UNCERTAIN';
      if (signal.timestampMs >= baseline.calibrationEndMs && bothAirborne) {
        takeoffCandidateStart ??= signal.timestampMs;
        takeoffCandidateLastContactMs ??= lastGroundContactMs ?? signal.timestampMs;
        takeoffConfirm += 1;
        if (takeoffConfirm >= TAKEOFF_CONFIRM_FRAMES) {
          takeoffMs = takeoffCandidateLastContactMs ?? takeoffCandidateStart;
          airborne = true;
        }
      } else if (phase !== 'GROUND_CONTACT') {
        uncertainFrames += 1;
        takeoffConfirm = 0;
        takeoffCandidateStart = null;
        takeoffCandidateLastContactMs = null;
      } else {
        lastGroundContactMs = signal.timestampMs;
        takeoffConfirm = 0;
        takeoffCandidateStart = null;
        takeoffCandidateLastContactMs = null;
      }
    } else {
      if (leftGrounded || rightGrounded) {
        phase = 'GROUND_CONTACT';
        landingCandidateStart ??= signal.timestampMs;
        landingConfirm += 1;
        if (landingConfirm >= LANDING_CONFIRM_FRAMES && landingMs === null) {
          landingMs = landingCandidateStart;
          airborne = false;
        }
      } else {
        phase = leftClear || rightClear ? 'AIRBORNE' : 'UNCERTAIN';
        if (phase === 'UNCERTAIN') {
          uncertainFrames += 1;
        }
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
      leftContactScore: signal.leftContactScore,
      rightContactScore: signal.rightContactScore,
      horizontalDrift,
    });
  }

  const uncertaintyRatio = phaseTimeline.length > 0 ? uncertainFrames / phaseTimeline.length : 1;
  const qualityFlags = [...preFlags];
  const quality = scoreQuality(
    averageConfidence,
    uncertaintyRatio,
    Math.min(feetVisibleRatio, dualFootVisibleRatio + 0.15),
    fpsScore,
    baseline.stability,
    averageContactReliability,
    qualityFlags,
  );

  const debug: JumpFeetAnalysisDebug = {
    calibrationEndMs: baseline.calibrationEndMs,
    baseline,
    analyzedFrameCount: frames.length,
    videoDurationMs,
    videoFps,
    sampleFps,
    averageConfidence,
    uncertaintyRatio,
    feetVisibleRatio,
    dualFootVisibleRatio,
    maxHorizontalDrift,
    calibrationStability: baseline.stability,
    averageContactReliability,
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
    `Last takeoff contact at ${takeoffMs.toFixed(1)} ms, first landing contact at ${landingMs.toFixed(1)} ms, flight time ${flightMs.toFixed(1)} ms, height ${heightCm.toFixed(1)} cm.`,
  );
}

export function findFootFrameAtTime(
  frames: JumpFootContactFrame[],
  timestampMs: number,
): JumpFootContactFrame | null {
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

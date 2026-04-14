export const GRAVITY_MS2 = 9.81;

export type JumpPhase =
  | 'IDLE'
  | 'CALIBRATING'
  | 'READY'
  | 'AIRBORNE'
  | 'LANDED'
  | 'INVALID_SETUP';

export type JumpQuality = 'LOW' | 'MEDIUM' | 'HIGH';

export type JumpInvalidReason =
  | 'NO_PERSON'
  | 'MULTIPLE_PEOPLE'
  | 'BODY_NOT_FULLY_VISIBLE'
  | 'FEET_NOT_VISIBLE'
  | 'NO_STABLE_CALIBRATION'
  | 'NO_TAKEOFF'
  | 'NO_LANDING'
  | 'AIRTIME_OUT_OF_RANGE'
  | 'EXCESS_HORIZONTAL_MOTION';

export type JumpContactPhase = 'GROUND_CONTACT' | 'AIRBORNE' | 'UNCERTAIN';

export interface JumpResult {
  airtimeMs: number;
  airtimeSec: number;
  heightCm: number;
  quality: JumpQuality;
  qualityFlags?: string[];
}

export interface JumpClip {
  uri: string;
  assetId?: string | null;
  fps: number;
  durationMs: number;
  width: number;
  height: number;
  recordedAt: string;
}

export interface JumpKeypoint {
  x: number;
  y: number;
  score: number;
  name?: string;
}

export interface JumpLandmarkFrame {
  frameIndex: number;
  timestampMs: number;
  captureTimestampMs?: number;
  keypoints: JumpKeypoint[];
  avgConfidence: number;
  personCount?: number;
}

export interface JumpFootBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface JumpFootContactFrame {
  frameIndex: number;
  timestampMs: number;
  leftFootBox: JumpFootBox | null;
  rightFootBox: JumpFootBox | null;
  leftFootBottomY: number | null;
  rightFootBottomY: number | null;
  leftContactScore: number | null;
  rightContactScore: number | null;
  floorY: number | null;
  avgConfidence: number;
  personCount?: number;
}

export interface JumpPersonCountSummary {
  analyzedFrames: number;
  multiPersonFrames: number;
  maxPeople: number;
}

export interface JumpVideoAnalysisOptions {
  sampleFps?: number;
  maxFrames?: number;
  minConfidence?: number;
  analysisStartMs?: number;
  analysisEndMs?: number;
}

export interface JumpVideoNativeResult {
  frames: JumpLandmarkFrame[];
  videoFps: number;
  sampleFps: number;
  playbackVideoFps?: number;
  playbackSampleFps?: number;
  videoDurationMs: number;
  personCountSummary: JumpPersonCountSummary;
}

export interface JumpFootVideoNativeResult {
  frames: JumpFootContactFrame[];
  videoFps: number;
  sampleFps: number;
  videoDurationMs: number;
  personCountSummary: JumpPersonCountSummary;
}

export interface JumpPhaseSample {
  frameIndex: number;
  timestampMs: number;
  phase: JumpContactPhase;
  leftLift: number | null;
  rightLift: number | null;
  hipLift: number | null;
  horizontalDrift: number | null;
}

export interface JumpBaselineDebug {
  leftToeY: number;
  rightToeY: number;
  leftAnkleY: number;
  rightAnkleY: number;
  hipY: number;
  torso: number;
  centerX: number;
}

export interface JumpAnalysisDebug {
  calibrationEndMs: number;
  baseline: JumpBaselineDebug;
  analyzedFrameCount: number;
  videoDurationMs: number;
  videoFps: number;
  sampleFps: number;
  playbackVideoFps?: number;
  playbackSampleFps?: number;
  slowMotionScaleFactor?: number;
  averageConfidence: number;
  uncertaintyRatio: number;
  fullBodyVisibleRatio: number;
  feetVisibleRatio: number;
  maxHorizontalDrift: number;
  calibrationStability: number;
}

export interface JumpAnalysisResult {
  takeoffMs: number | null;
  landingMs: number | null;
  takeoffPhysicalMs: number | null;
  landingPhysicalMs: number | null;
  flightMs: number | null;
  heightCm: number | null;
  phaseTimeline: JumpPhaseSample[];
  quality: JumpQuality;
  invalidReason?: JumpInvalidReason;
  qualityFlags?: string[];
  summary: string;
  debug: JumpAnalysisDebug;
}

export interface JumpFeetPhaseSample {
  frameIndex: number;
  timestampMs: number;
  phase: JumpContactPhase;
  leftLift: number | null;
  rightLift: number | null;
  leftContactScore: number | null;
  rightContactScore: number | null;
  horizontalDrift: number | null;
}

export interface JumpFeetBaselineDebug {
  leftBottomY: number;
  rightBottomY: number;
  floorY: number;
  leftFootHeight: number;
  rightFootHeight: number;
  centerX: number;
}

export interface JumpFeetAnalysisDebug {
  calibrationEndMs: number;
  baseline: JumpFeetBaselineDebug;
  analyzedFrameCount: number;
  videoDurationMs: number;
  videoFps: number;
  sampleFps: number;
  averageConfidence: number;
  uncertaintyRatio: number;
  feetVisibleRatio: number;
  dualFootVisibleRatio: number;
  maxHorizontalDrift: number;
  calibrationStability: number;
  averageContactReliability: number;
}

export interface JumpFeetAnalysisResult {
  takeoffMs: number | null;
  landingMs: number | null;
  flightMs: number | null;
  heightCm: number | null;
  phaseTimeline: JumpFeetPhaseSample[];
  quality: JumpQuality;
  invalidReason?: JumpInvalidReason;
  qualityFlags?: string[];
  summary: string;
  debug: JumpFeetAnalysisDebug;
}

export interface CalibrationData {
  pixelsPerMeter: number;
  /** Normalized [0,1] ankle Y in model-input coordinates */
  baselineAnkleY: number;
  /** Normalized [0,1] hip Y in model-input coordinates */
  baselineHipY: number;
}

/** Calculates jump height in cm from flight time using h = g*t^2/8 */
export function heightFromFlightTime(flightTimeMs: number): number {
  const t = flightTimeMs / 1000;
  return (GRAVITY_MS2 * t * t) / 8 * 100;
}

/** @deprecated Legacy helper kept for backward compatibility during migration. */
/** Calculates jump height in cm from hip pixel displacement */
export function heightFromDisplacement(pixelDisplacement: number, pixelsPerMeter: number): number {
  if (pixelsPerMeter <= 0) return 0;
  return (pixelDisplacement / pixelsPerMeter) * 100;
}

/**
 * Returns true when ankles have lifted above baseline (takeoff detected).
 * Y increases downward - a smaller ankleY means higher position.
 *
 * Takeoff threshold MUST be larger than landing threshold to create hysteresis.
 * Previous values (takeoff=0.04, landing=0.05) overlapped, causing landing to
 * fire immediately after takeoff and producing ~0ms flight time.
 */
export function detectTakeoff(
  normAnkleY: number,
  baselineAnkleY: number,
  threshold = 0.06,
): boolean {
  return baselineAnkleY - normAnkleY > threshold;
}

/**
 * Returns true when ankles have returned close to baseline (landing detected).
 * Landing threshold (0.03) must be strictly less than takeoff threshold (0.06)
 * so landing cannot fire in the same window as takeoff.
 */
export function detectLanding(
  normAnkleY: number,
  baselineAnkleY: number,
  threshold = 0.03,
): boolean {
  return normAnkleY >= baselineAnkleY - threshold;
}

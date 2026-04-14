import { useCallback, useMemo, useState } from 'react';
import { Platform } from 'react-native';
import { runAsync, runAtTargetFps, useFrameProcessor } from 'react-native-vision-camera';
import { useRunOnJS, useSharedValue } from 'react-native-worklets-core';

import { type JumpResult } from '@/utils/jumpCalc';

const IS_IOS = Platform.OS === 'ios';

const ANALYSIS_WIDTH = 160;
const ANALYSIS_HEIGHT = 120;
const TARGET_PROCESS_FPS = 60;
const CALIBRATION_DURATION_MS = 1200;
const CALIBRATION_MIN_FRAMES = 40;
const CALIBRATION_STABLE_SETUP_FRAMES = 10;
const TAKEOFF_CONFIRM_FRAMES = 2;
const LANDING_CONFIRM_FRAMES = 3;
const MIN_AIRTIME_MS = 180;
const MAX_AIRTIME_MS = 900;
const MIN_AIRBORNE_BEFORE_LANDING_MS = 100;
const COOLDOWN_MS = 500;
const LANDED_HOLD_MS = 1200;
const TAKEOFF_CONTACT_THRESHOLD = 0.42;
const TAKEOFF_LIFT_THRESHOLD = 0.32;
const LANDING_CONTACT_THRESHOLD = 0.64;
const READY_ARM_MIN_MS = 450;
const READY_ARM_STABLE_FRAMES = 5;
const READY_ARM_CONTACT_MIN = 0.64;
const READY_ARM_LIFT_MAX = 0.26;
const READY_ARM_ACTIVE_WINDOW_MS = 1400;
const READY_INVALID_SETUP_FRAMES = 10;
const GRAVITY_M_S2 = 9.81;

type Region = {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
};

function toRegion(nx0: number, nx1: number, ny0: number, ny1: number): Region {
  return {
    x0: Math.floor(nx0 * ANALYSIS_WIDTH),
    x1: Math.ceil(nx1 * ANALYSIS_WIDTH),
    y0: Math.floor(ny0 * ANALYSIS_HEIGHT),
    y1: Math.ceil(ny1 * ANALYSIS_HEIGHT),
  };
}

const FULL_REGION = toRegion(0, 1, 0, 1);
const GUIDE_REGION = toRegion(0.31, 0.69, 0.12, 0.93);
const GUIDE_LEFT_REGION = toRegion(0.31, 0.5, 0.12, 0.93);
const GUIDE_RIGHT_REGION = toRegion(0.5, 0.69, 0.12, 0.93);
const FEET_REGION = toRegion(0.3, 0.7, 0.66, 0.97);
const LIFT_REGION = toRegion(0.3, 0.7, 0.7, 0.9);
const CONTACT_REGION = toRegion(0.3, 0.7, 0.86, 0.98);
const FLOOR_REGION = toRegion(0.28, 0.72, 0.94, 0.997);

export type AirtimeStatus =
  | 'IDLE'
  | 'CALIBRATING'
  | 'READY'
  | 'AIRBORNE'
  | 'LANDED'
  | 'INVALID_SETUP';

export interface AirtimeChecklist {
  feetVisible: boolean;
  floorVisible: boolean;
  centered: boolean;
  lightingOk: boolean;
}

interface ChecklistMetrics {
  checklist: AirtimeChecklist;
  globalMean: number;
  guideMean: number;
  outerMotion: number;
  contactDiffPrev: number;
  liftDiffPrev: number;
}

interface CalibrationCounters {
  frames: number;
  centeredPass: number;
  feetVisiblePass: number;
  floorVisiblePass: number;
  lightingPass: number;
}

interface WorkletState {
  status: AirtimeStatus;
  nowMs: number;
  lastRawTs: number;
  hasPrev: boolean;
  gray: Uint8Array;
  prevGray: Uint8Array;
  baselineGray: Uint8Array;
  baselineAccum: Float32Array;
  calibrationStartMs: number;
  calibrationFrames: number;
  calibrationSetupStableFrames: number;
  calibrationCounters: CalibrationCounters;
  contactNoiseSum: number;
  liftNoiseSum: number;
  contactScale: number;
  liftScale: number;
  setupChecklist: AirtimeChecklist;
  currentChecklist: AirtimeChecklist;
  backlitSeen: boolean;
  takeoffConfirm: number;
  landingConfirm: number;
  takeoffMs: number;
  readySinceMs: number;
  readyStableFrames: number;
  readyArmed: boolean;
  readyArmedAtMs: number;
  setupInvalidFrames: number;
  takeoffMargin: number;
  landingMargin: number;
  airborneDroppedFrames: number;
  landedAtMs: number;
  cooldownUntilMs: number;
  dtEma: number;
  dtJitterEma: number;
  fpsEma: number;
  lastResetCommand: number;
  lastCalibrateCommand: number;
  readyEmitted: boolean;
  lastGuidance: string;
  lastChecklistMask: number;
  lastStatus: AirtimeStatus | '';
}

const DEFAULT_CHECKLIST: AirtimeChecklist = {
  feetVisible: false,
  floorVisible: false,
  centered: false,
  lightingOk: false,
};

export interface UseAirtimePocReturn {
  isReady: boolean;
  status: AirtimeStatus;
  guidance: string;
  checklist: AirtimeChecklist;
  result: JumpResult | null;
  startCalibration: () => void;
  reset: () => void;
  frameProcessor: ReturnType<typeof useFrameProcessor>;
}

function clamp01(value: number): number {
  'worklet';
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function createCounters(): CalibrationCounters {
  'worklet';
  return {
    frames: 0,
    centeredPass: 0,
    feetVisiblePass: 0,
    floorVisiblePass: 0,
    lightingPass: 0,
  };
}

function createWorkletState(): WorkletState {
  'worklet';
  const size = ANALYSIS_WIDTH * ANALYSIS_HEIGHT;
  return {
    status: 'IDLE',
    nowMs: 0,
    lastRawTs: 0,
    hasPrev: false,
    gray: new Uint8Array(size),
    prevGray: new Uint8Array(size),
    baselineGray: new Uint8Array(size),
    baselineAccum: new Float32Array(size),
    calibrationStartMs: 0,
    calibrationFrames: 0,
    calibrationSetupStableFrames: 0,
    calibrationCounters: createCounters(),
    contactNoiseSum: 0,
    liftNoiseSum: 0,
    contactScale: 24,
    liftScale: 18,
    setupChecklist: { ...DEFAULT_CHECKLIST },
    currentChecklist: { ...DEFAULT_CHECKLIST },
    backlitSeen: false,
    takeoffConfirm: 0,
    landingConfirm: 0,
    takeoffMs: 0,
    readySinceMs: 0,
    readyStableFrames: 0,
    readyArmed: false,
    readyArmedAtMs: 0,
    setupInvalidFrames: 0,
    takeoffMargin: 0,
    landingMargin: 0,
    airborneDroppedFrames: 0,
    landedAtMs: 0,
    cooldownUntilMs: 0,
    dtEma: 0,
    dtJitterEma: 0,
    fpsEma: 0,
    lastResetCommand: 0,
    lastCalibrateCommand: 0,
    readyEmitted: false,
    lastGuidance: '',
    lastChecklistMask: -1,
    lastStatus: '',
  };
}

function resetTrackingState(state: WorkletState): void {
  'worklet';
  state.status = 'IDLE';
  state.hasPrev = false;
  state.calibrationStartMs = 0;
  state.calibrationFrames = 0;
  state.calibrationSetupStableFrames = 0;
  state.calibrationCounters = createCounters();
  state.contactNoiseSum = 0;
  state.liftNoiseSum = 0;
  state.contactScale = 24;
  state.liftScale = 18;
  state.setupChecklist = { ...DEFAULT_CHECKLIST };
  state.currentChecklist = { ...DEFAULT_CHECKLIST };
  state.backlitSeen = false;
  state.takeoffConfirm = 0;
  state.landingConfirm = 0;
  state.takeoffMs = 0;
  state.readySinceMs = 0;
  state.readyStableFrames = 0;
  state.readyArmed = false;
  state.readyArmedAtMs = 0;
  state.setupInvalidFrames = 0;
  state.takeoffMargin = 0;
  state.landingMargin = 0;
  state.airborneDroppedFrames = 0;
  state.landedAtMs = 0;
  state.cooldownUntilMs = 0;
  state.lastGuidance = '';
  state.lastChecklistMask = -1;
  state.lastStatus = '';
  state.baselineAccum.fill(0);
  state.baselineGray.fill(0);
  state.prevGray.fill(0);
}

function beginCalibration(state: WorkletState): void {
  'worklet';
  resetTrackingState(state);
  state.status = 'CALIBRATING';
  state.calibrationStartMs = state.nowMs;
}

function enterReadyState(state: WorkletState): void {
  'worklet';
  state.status = 'READY';
  state.takeoffConfirm = 0;
  state.landingConfirm = 0;
  state.readySinceMs = state.nowMs;
  state.readyStableFrames = 0;
  state.readyArmed = false;
  state.readyArmedAtMs = 0;
  state.setupInvalidFrames = 0;
}

function resetCalibrationCollection(state: WorkletState): void {
  'worklet';
  state.calibrationFrames = 0;
  state.calibrationCounters = createCounters();
  state.contactNoiseSum = 0;
  state.liftNoiseSum = 0;
  state.baselineAccum.fill(0);
}

function statsInRegion(gray: Uint8Array, region: Region): { mean: number; std: number } {
  'worklet';
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = region.y0; y < region.y1; y++) {
    const rowStart = y * ANALYSIS_WIDTH;
    for (let x = region.x0; x < region.x1; x++) {
      const v = gray[rowStart + x];
      sum += v;
      sumSq += v * v;
      count += 1;
    }
  }
  if (count === 0) {
    return { mean: 0, std: 0 };
  }
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  return { mean, std: Math.sqrt(variance) };
}

function meanAbsDiffRegion(
  current: Uint8Array,
  previous: Uint8Array,
  region: Region,
  stride = 1,
): number {
  'worklet';
  let sum = 0;
  let count = 0;
  for (let y = region.y0; y < region.y1; y += stride) {
    const rowStart = y * ANALYSIS_WIDTH;
    for (let x = region.x0; x < region.x1; x += stride) {
      const i = rowStart + x;
      sum += Math.abs(current[i] - previous[i]);
      count += 1;
    }
  }
  return count > 0 ? sum / count : 0;
}

function meanAbsDiffOutside(
  current: Uint8Array,
  previous: Uint8Array,
  innerRegion: Region,
  stride = 2,
): number {
  'worklet';
  let sum = 0;
  let count = 0;
  for (let y = 0; y < ANALYSIS_HEIGHT; y += stride) {
    const rowStart = y * ANALYSIS_WIDTH;
    for (let x = 0; x < ANALYSIS_WIDTH; x += stride) {
      const inside =
        x >= innerRegion.x0 &&
        x < innerRegion.x1 &&
        y >= innerRegion.y0 &&
        y < innerRegion.y1;
      if (inside) continue;
      const i = rowStart + x;
      sum += Math.abs(current[i] - previous[i]);
      count += 1;
    }
  }
  return count > 0 ? sum / count : 0;
}

function checklistMask(checklist: AirtimeChecklist): number {
  'worklet';
  return (
    (checklist.feetVisible ? 1 : 0) |
    (checklist.floorVisible ? 2 : 0) |
    (checklist.centered ? 4 : 0) |
    (checklist.lightingOk ? 8 : 0)
  );
}

function checklistFromMetrics(
  gray: Uint8Array,
  prevGray: Uint8Array,
  hasPrev: boolean,
): ChecklistMetrics {
  'worklet';
  const globalStats = statsInRegion(gray, FULL_REGION);
  const guideStats = statsInRegion(gray, GUIDE_REGION);
  const leftGuide = statsInRegion(gray, GUIDE_LEFT_REGION);
  const rightGuide = statsInRegion(gray, GUIDE_RIGHT_REGION);
  const feetStats = statsInRegion(gray, FEET_REGION);
  const floorStats = statsInRegion(gray, FLOOR_REGION);

  const centeredBalance =
    Math.min(leftGuide.std, rightGuide.std) / (Math.max(leftGuide.std, rightGuide.std) + 1e-6);
  const centered =
    guideStats.std > 10 &&
    centeredBalance > 0.32 &&
    leftGuide.std > 4 &&
    rightGuide.std > 4;

  const contrastFeetFloor = Math.abs(feetStats.mean - floorStats.mean);
  const feetVisible = feetStats.std > 7 && contrastFeetFloor > 5;
  const floorVisible =
    floorStats.mean > 18 &&
    floorStats.mean < 245 &&
    (floorStats.std > 1.2 || contrastFeetFloor > 5);

  const backlit = globalStats.mean > 75 && guideStats.mean < globalStats.mean - 18;
  const lightingOk =
    globalStats.mean >= 40 && globalStats.mean <= 215 && globalStats.std >= 16 && !backlit;

  const outerMotion = hasPrev
    ? meanAbsDiffOutside(gray, prevGray, GUIDE_REGION, 2)
    : 0;

  const contactDiffPrev = hasPrev
    ? meanAbsDiffRegion(gray, prevGray, CONTACT_REGION, 1)
    : 0;
  const liftDiffPrev = hasPrev
    ? meanAbsDiffRegion(gray, prevGray, LIFT_REGION, 1)
    : 0;

  return {
    checklist: {
      feetVisible,
      floorVisible,
      centered,
      lightingOk,
    },
    globalMean: globalStats.mean,
    guideMean: guideStats.mean,
    outerMotion,
    contactDiffPrev,
    liftDiffPrev,
  };
}

function invalidSetupGuidance(checklist: AirtimeChecklist): string {
  'worklet';
  if (!checklist.centered) {
    return 'Move back so full body and floor are visible';
  }
  if (!checklist.floorVisible) {
    return 'Tilt camera slightly down';
  }
  if (!checklist.lightingOk) {
    return 'Add front light and avoid a bright window behind you';
  }
  if (!checklist.feetVisible) {
    return 'Increase contrast with the floor (shoes/socks/background)';
  }
  return 'Setup is unstable. Reposition and calibrate again';
}

function pushUnique(flags: string[], flag: string): void {
  'worklet';
  for (let i = 0; i < flags.length; i++) {
    if (flags[i] === flag) return;
  }
  flags.push(flag);
}

function qualityFromSignals(
  airtimeMs: number,
  state: WorkletState,
  checklist: AirtimeChecklist,
  backlitSeen: boolean,
): JumpResult {
  'worklet';
  const flags: string[] = [];

  if (!checklist.lightingOk) {
    pushUnique(flags, backlitSeen ? 'BACKLIT' : 'LOW_LIGHT');
  }
  if (!checklist.feetVisible) {
    pushUnique(flags, 'WEAK_FEET_CONTRAST');
  }
  if (state.fpsEma > 0 && state.fpsEma < 28) {
    pushUnique(flags, 'LOW_FPS');
  }
  if (state.airborneDroppedFrames > 1) {
    pushUnique(flags, 'LOW_FPS');
  }

  const setupScore =
    (Number(checklist.centered) +
      Number(checklist.feetVisible) +
      Number(checklist.floorVisible) +
      Number(checklist.lightingOk)) /
    4;

  const fpsScore =
    state.fpsEma > 0
      ? clamp01((state.fpsEma - 20) / 35) * clamp01(1 - state.dtJitterEma / 8)
      : 0.5;

  const transitionScore = clamp01((state.takeoffMargin + state.landingMargin) / 0.5);
  const dropPenalty = Math.min(0.35, state.airborneDroppedFrames * 0.08);

  const weighted = 0.45 * setupScore + 0.3 * fpsScore + 0.25 * transitionScore - dropPenalty;

  let quality: JumpResult['quality'] = 'LOW';
  if (weighted >= 0.72) quality = 'HIGH';
  else if (weighted >= 0.48) quality = 'MEDIUM';

  return {
    airtimeMs: Math.round(airtimeMs),
    airtimeSec: airtimeMs / 1000,
    heightCm: (GRAVITY_M_S2 * (airtimeMs / 1000) * (airtimeMs / 1000) * 100) / 8,
    quality,
    qualityFlags: flags.length > 0 ? flags : undefined,
  };
}

function toRawFrameDeltaMs(rawDelta: number): number {
  'worklet';
  if (rawDelta <= 0) return 0;
  // iOS uses milliseconds, Android camera timestamps are often nanoseconds.
  if (rawDelta > 10000) return rawDelta / 1_000_000;
  // Some runtimes can expose seconds.
  if (rawDelta < 1) return rawDelta * 1000;
  return rawDelta;
}

function updateTimestamp(state: WorkletState, timestamp: number): number {
  'worklet';
  if (state.lastRawTs <= 0) {
    state.lastRawTs = timestamp;
    return 0;
  }
  const rawDelta = timestamp - state.lastRawTs;
  state.lastRawTs = timestamp;
  let dtMs = toRawFrameDeltaMs(rawDelta);
  if (!Number.isFinite(dtMs) || dtMs < 0 || dtMs > 300) {
    dtMs = state.dtEma > 0 ? state.dtEma : 1000 / 30;
  }
  state.nowMs += dtMs;
  if (state.dtEma <= 0) {
    state.dtEma = dtMs;
    state.dtJitterEma = 0;
  } else {
    const jitter = Math.abs(dtMs - state.dtEma);
    state.dtEma = state.dtEma * 0.9 + dtMs * 0.1;
    state.dtJitterEma = state.dtJitterEma * 0.85 + jitter * 0.15;
  }
  state.fpsEma = state.dtEma > 0 ? 1000 / state.dtEma : 0;
  return dtMs;
}

function fillGrayscalePlane(
  gray: Uint8Array,
  frame: {
    width: number;
    height: number;
    bytesPerRow: number;
    orientation: string;
    toArrayBuffer: () => ArrayBuffer;
  },
): void {
  'worklet';
  const buffer = frame.toArrayBuffer();
  const src = new Uint8Array(buffer);

  const srcW = frame.width;
  const srcH = frame.height;
  const swapAxes =
    frame.orientation === 'landscape-left' || frame.orientation === 'landscape-right';
  const orientedW = swapAxes ? srcH : srcW;
  const orientedH = swapAxes ? srcW : srcH;

  for (let y = 0; y < ANALYSIS_HEIGHT; y++) {
    const uprightY = Math.min(
      orientedH - 1,
      Math.max(0, Math.floor(((y + 0.5) * orientedH) / ANALYSIS_HEIGHT)),
    );
    for (let x = 0; x < ANALYSIS_WIDTH; x++) {
      const uprightX = Math.min(
        orientedW - 1,
        Math.max(0, Math.floor(((x + 0.5) * orientedW) / ANALYSIS_WIDTH)),
      );

      let srcX = uprightX;
      let srcY = uprightY;

      if (frame.orientation === 'portrait-upside-down') {
        srcX = srcW - 1 - uprightX;
        srcY = srcH - 1 - uprightY;
      } else if (frame.orientation === 'landscape-left') {
        srcX = uprightY;
        srcY = srcH - 1 - uprightX;
      } else if (frame.orientation === 'landscape-right') {
        srcX = srcW - 1 - uprightY;
        srcY = uprightX;
      }

      srcX = Math.min(srcW - 1, Math.max(0, srcX));
      srcY = Math.min(srcH - 1, Math.max(0, srcY));

      const srcIndex = srcY * frame.bytesPerRow + srcX * 4;
      const outIndex = y * ANALYSIS_WIDTH + x;

      if (srcIndex + 2 >= src.length) {
        gray[outIndex] = 0;
        continue;
      }

      let r = 0;
      let g = 0;
      let b = 0;
      if (IS_IOS) {
        // BGRA -> RGB
        b = src[srcIndex];
        g = src[srcIndex + 1];
        r = src[srcIndex + 2];
      } else {
        // RGBA -> RGB
        r = src[srcIndex];
        g = src[srcIndex + 1];
        b = src[srcIndex + 2];
      }

      gray[outIndex] = (r * 77 + g * 150 + b * 29) >> 8;
    }
  }
}

function isChecklistPassing(checklist: AirtimeChecklist): boolean {
  'worklet';
  return (
    checklist.centered &&
    checklist.feetVisible &&
    checklist.floorVisible &&
    checklist.lightingOk
  );
}

export function useAirtimePoc(): UseAirtimePocReturn {
  const [isReady, setIsReady] = useState(false);
  const [status, setStatus] = useState<AirtimeStatus>('IDLE');
  const [guidance, setGuidance] = useState('Tap Calibrate to begin');
  const [checklist, setChecklist] = useState<AirtimeChecklist>(DEFAULT_CHECKLIST);
  const [result, setResult] = useState<JumpResult | null>(null);

  const resetCommand = useSharedValue(0);
  const calibrateCommand = useSharedValue(0);
  const instanceId = useMemo(() => Math.floor(Math.random() * 1_000_000_000), []);

  const onReady = useCallback(() => {
    setIsReady(true);
  }, []);

  const onStatus = useCallback((next: AirtimeStatus) => {
    setStatus(next);
  }, []);

  const onGuidance = useCallback((next: string) => {
    setGuidance(next);
  }, []);

  const onChecklist = useCallback((next: AirtimeChecklist) => {
    setChecklist(next);
  }, []);

  const onResult = useCallback((next: JumpResult) => {
    setResult(next);
  }, []);

  const onReadyWorklet = useRunOnJS(onReady, [onReady]);
  const onStatusWorklet = useRunOnJS(onStatus, [onStatus]);
  const onGuidanceWorklet = useRunOnJS(onGuidance, [onGuidance]);
  const onChecklistWorklet = useRunOnJS(onChecklist, [onChecklist]);
  const onResultWorklet = useRunOnJS(onResult, [onResult]);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';

      runAtTargetFps(TARGET_PROCESS_FPS, () => {
        runAsync(frame, () => {
          'worklet';

          const globalStore = globalThis as unknown as { [key: string]: WorkletState | undefined };
          const stateKey = `__hexfitAirtimePocState_${instanceId}`;
          let state = globalStore[stateKey];
          if (!state) {
            state = createWorkletState();
            globalStore[stateKey] = state;
          }

          const emitStatus = (next: AirtimeStatus) => {
            if (state.lastStatus === next) return;
            state.lastStatus = next;
            onStatusWorklet(next);
          };

          const emitGuidance = (next: string) => {
            if (state.lastGuidance === next) return;
            state.lastGuidance = next;
            onGuidanceWorklet(next);
          };

          const emitChecklist = (next: AirtimeChecklist) => {
            const mask = checklistMask(next);
            if (mask === state.lastChecklistMask) return;
            state.lastChecklistMask = mask;
            onChecklistWorklet(next);
          };

          if (!state.readyEmitted) {
            state.readyEmitted = true;
            onReadyWorklet();
          }

          if (resetCommand.value !== state.lastResetCommand) {
            state.lastResetCommand = resetCommand.value;
            resetTrackingState(state);
            emitStatus('IDLE');
            emitGuidance('Tap Calibrate to begin');
            emitChecklist(DEFAULT_CHECKLIST);
            return;
          }

          const dtMs = updateTimestamp(state, frame.timestamp);
          fillGrayscalePlane(state.gray, frame);

          const metrics = checklistFromMetrics(state.gray, state.prevGray, state.hasPrev);
          state.currentChecklist = metrics.checklist;
          state.backlitSeen =
            state.backlitSeen ||
            (metrics.globalMean > 70 &&
              metrics.guideMean < metrics.globalMean - 18);

          emitChecklist(metrics.checklist);
          const setupNowValid = isChecklistPassing(metrics.checklist);

          if (calibrateCommand.value !== state.lastCalibrateCommand) {
            state.lastCalibrateCommand = calibrateCommand.value;
            beginCalibration(state);
            emitStatus('CALIBRATING');
            emitGuidance('Calibrating... stand still');
          }

          if (state.status === 'IDLE') {
            emitStatus('IDLE');
            emitGuidance('Tap Calibrate to begin');
          }

          if (state.status === 'CALIBRATING') {
            if (!setupNowValid) {
              if (
                state.calibrationSetupStableFrames > 0 ||
                state.calibrationFrames > 0
              ) {
                state.calibrationSetupStableFrames = 0;
                resetCalibrationCollection(state);
              }

              if (!metrics.checklist.centered) {
                emitGuidance('Move back so full body and floor are visible');
              } else if (!metrics.checklist.floorVisible) {
                emitGuidance('Tilt camera slightly down');
              } else if (!metrics.checklist.lightingOk) {
                emitGuidance('Add front light and avoid a bright window behind you');
              } else if (!metrics.checklist.feetVisible) {
                emitGuidance('Increase contrast with the floor (shoes/socks/background)');
              } else {
                emitGuidance('Move into frame, then hold still');
              }
            } else {
              if (state.calibrationSetupStableFrames < CALIBRATION_STABLE_SETUP_FRAMES) {
                state.calibrationSetupStableFrames += 1;
                emitGuidance(
                  `Hold still... preparing (${state.calibrationSetupStableFrames}/${CALIBRATION_STABLE_SETUP_FRAMES})`,
                );
              } else {
                if (state.calibrationFrames === 0) {
                  state.calibrationStartMs = state.nowMs;
                }

                state.calibrationFrames += 1;
                state.calibrationCounters.frames += 1;
                for (let i = 0; i < state.baselineAccum.length; i++) {
                  state.baselineAccum[i] += state.gray[i];
                }

                if (metrics.checklist.centered) state.calibrationCounters.centeredPass += 1;
                if (metrics.checklist.feetVisible) state.calibrationCounters.feetVisiblePass += 1;
                if (metrics.checklist.floorVisible) state.calibrationCounters.floorVisiblePass += 1;
                if (metrics.checklist.lightingOk) state.calibrationCounters.lightingPass += 1;

                if (state.hasPrev) {
                  state.contactNoiseSum += metrics.contactDiffPrev;
                  state.liftNoiseSum += metrics.liftDiffPrev;
                }

                emitGuidance('Calibrating... hold still');

                const elapsedMs = state.nowMs - state.calibrationStartMs;
                const canFinish =
                  elapsedMs >= CALIBRATION_DURATION_MS &&
                  state.calibrationFrames >= CALIBRATION_MIN_FRAMES;

                if (canFinish) {
                  const frames = Math.max(1, state.calibrationFrames);
                  for (let i = 0; i < state.baselineGray.length; i++) {
                    state.baselineGray[i] = Math.round(state.baselineAccum[i] / frames);
                  }

                  const passRatio = {
                    centered: state.calibrationCounters.centeredPass / frames,
                    feetVisible: state.calibrationCounters.feetVisiblePass / frames,
                    floorVisible: state.calibrationCounters.floorVisiblePass / frames,
                    lightingOk: state.calibrationCounters.lightingPass / frames,
                  };

                  state.setupChecklist = {
                    centered: passRatio.centered >= 0.65,
                    feetVisible: passRatio.feetVisible >= 0.65,
                    floorVisible: passRatio.floorVisible >= 0.65,
                    lightingOk: passRatio.lightingOk >= 0.65,
                  };

                  state.contactScale = Math.max(
                    18,
                    (state.contactNoiseSum / Math.max(1, frames - 1)) * 6 + 10,
                  );
                  state.liftScale = Math.max(
                    15,
                    (state.liftNoiseSum / Math.max(1, frames - 1)) * 5 + 8,
                  );

                  if (isChecklistPassing(state.setupChecklist)) {
                    enterReadyState(state);
                    emitStatus('READY');
                    emitGuidance('Ready - jump!');
                    emitChecklist(state.setupChecklist);
                  } else {
                    state.calibrationSetupStableFrames = 0;
                    resetCalibrationCollection(state);
                    emitGuidance('Setup changed. Hold still and calibrating will restart');
                  }
                }
              }
            }
          }

          if (state.status === 'INVALID_SETUP') {
            if (setupNowValid) {
              emitGuidance('Setup looks good. Tap Calibrate');
            } else {
              emitGuidance(invalidSetupGuidance(metrics.checklist));
            }
          }

          if (state.status === 'READY') {
            emitStatus('READY');

            if (!setupNowValid) {
              state.setupInvalidFrames += 1;
              state.takeoffConfirm = 0;
              state.readyStableFrames = 0;
              state.readyArmed = false;
              state.readyArmedAtMs = 0;

              if (state.setupInvalidFrames >= READY_INVALID_SETUP_FRAMES) {
                state.status = 'INVALID_SETUP';
                emitStatus('INVALID_SETUP');
                emitGuidance(invalidSetupGuidance(metrics.checklist));
              } else {
                emitGuidance(invalidSetupGuidance(metrics.checklist));
              }
            } else {
              state.setupInvalidFrames = Math.max(0, state.setupInvalidFrames - 2);

              const contactMad = meanAbsDiffRegion(state.gray, state.baselineGray, CONTACT_REGION, 1);
              const liftMad = meanAbsDiffRegion(state.gray, state.baselineGray, LIFT_REGION, 1);

              const contactScore = clamp01(1 - contactMad / state.contactScale);
              const liftScore = clamp01((liftMad - 4) / state.liftScale);
              const stabilityScore = clamp01(1 - metrics.outerMotion / 8);
              const stableReadyPose =
                contactScore > READY_ARM_CONTACT_MIN &&
                liftScore < READY_ARM_LIFT_MAX;

              if (!state.readyArmed) {
                if (stableReadyPose) {
                  state.readyStableFrames = Math.min(
                    READY_ARM_STABLE_FRAMES + 6,
                    state.readyStableFrames + 1,
                  );
                } else {
                  state.readyStableFrames = Math.max(0, state.readyStableFrames - 1);
                }

                const canArm =
                  state.nowMs - state.readySinceMs >= READY_ARM_MIN_MS &&
                  state.readyStableFrames >= READY_ARM_STABLE_FRAMES;
                if (canArm) {
                  state.readyArmed = true;
                  state.readyArmedAtMs = state.nowMs;
                }
              } else if (state.nowMs - state.readyArmedAtMs > READY_ARM_ACTIVE_WINDOW_MS) {
                // If user waits too long, require a short re-arm but avoid full reset.
                state.readyArmed = false;
                state.readyArmedAtMs = 0;
                state.readyStableFrames = Math.max(0, READY_ARM_STABLE_FRAMES - 2);
              }

              const readyArmed = state.readyArmed;

              if (state.nowMs < state.cooldownUntilMs) {
                state.takeoffConfirm = 0;
                emitGuidance(readyArmed ? 'Jump now' : 'Hold still to arm detection, then jump');
              } else if (!readyArmed) {
                state.takeoffConfirm = 0;
                emitGuidance('Hold still to arm detection, then jump');
              } else {
                const takeoffDetected =
                  contactScore < TAKEOFF_CONTACT_THRESHOLD &&
                  liftScore > TAKEOFF_LIFT_THRESHOLD &&
                  stabilityScore > 0.08;

                if (takeoffDetected) {
                  state.takeoffConfirm += 1;
                  if (state.takeoffConfirm >= TAKEOFF_CONFIRM_FRAMES) {
                    state.status = 'AIRBORNE';
                    state.takeoffMs = state.nowMs;
                    state.readyArmed = false;
                    state.readyArmedAtMs = 0;
                    state.takeoffMargin = clamp01(
                      ((TAKEOFF_CONTACT_THRESHOLD - contactScore) +
                        (liftScore - TAKEOFF_LIFT_THRESHOLD)) /
                        2,
                    );
                    state.landingConfirm = 0;
                    state.airborneDroppedFrames = 0;
                    emitStatus('AIRBORNE');
                    emitGuidance('Airborne');
                    state.takeoffConfirm = 0;
                  }
                } else {
                  state.takeoffConfirm = 0;
                  emitGuidance('Jump now');
                }
              }
            }
          }

          if (state.status === 'AIRBORNE') {
            emitStatus('AIRBORNE');
            emitGuidance('Airborne');

            const contactMad = meanAbsDiffRegion(state.gray, state.baselineGray, CONTACT_REGION, 1);
            const contactScore = clamp01(1 - contactMad / state.contactScale);
            const airborneMs = state.nowMs - state.takeoffMs;

            if (state.dtEma > 0 && dtMs > state.dtEma * 1.7) {
              state.airborneDroppedFrames += 1;
            }

            if (airborneMs > MAX_AIRTIME_MS) {
              enterReadyState(state);
              state.cooldownUntilMs = state.nowMs + COOLDOWN_MS;
              state.landingConfirm = 0;
              emitStatus('READY');
              emitGuidance('Jump not captured reliably. Try again.');
            } else {
              if (airborneMs < MIN_AIRBORNE_BEFORE_LANDING_MS) {
                state.landingConfirm = 0;
              } else if (contactScore > LANDING_CONTACT_THRESHOLD) {
                state.landingConfirm += 1;
              } else {
                state.landingConfirm = 0;
              }

              if (state.landingConfirm >= LANDING_CONFIRM_FRAMES) {
                const airtimeMs = airborneMs;
                state.landingMargin = clamp01(contactScore - LANDING_CONTACT_THRESHOLD);
                state.landingConfirm = 0;
                state.cooldownUntilMs = state.nowMs + COOLDOWN_MS;

                if (airtimeMs >= MIN_AIRTIME_MS && airtimeMs <= MAX_AIRTIME_MS) {
                  const resultPayload = qualityFromSignals(
                    airtimeMs,
                    state,
                    state.setupChecklist,
                    state.backlitSeen,
                  );
                  state.status = 'LANDED';
                  state.landedAtMs = state.nowMs;
                  emitStatus('LANDED');
                  emitGuidance('Landed - result captured');
                  onResultWorklet(resultPayload);
                } else {
                  enterReadyState(state);
                  emitStatus('READY');
                  emitGuidance('Jump not captured reliably. Try again.');
                }
              }
            }
          }

          if (state.status === 'LANDED') {
            emitStatus('LANDED');
            if (state.nowMs - state.landedAtMs >= LANDED_HOLD_MS) {
              enterReadyState(state);
              emitStatus('READY');
              emitGuidance('Ready - jump!');
            }
          }

          state.prevGray.set(state.gray);
          state.hasPrev = true;
        });
      });
    },
    [
      calibrateCommand,
      instanceId,
      onChecklistWorklet,
      onGuidanceWorklet,
      onReadyWorklet,
      onResultWorklet,
      onStatusWorklet,
      resetCommand,
    ],
  );

  const startCalibration = useCallback(() => {
    setResult(null);
    setStatus('CALIBRATING');
    setGuidance('Calibrating... stand still');
    calibrateCommand.value += 1;
  }, [calibrateCommand]);

  const reset = useCallback(() => {
    setResult(null);
    setStatus('IDLE');
    setGuidance('Tap Calibrate to begin');
    setChecklist(DEFAULT_CHECKLIST);
    resetCommand.value += 1;
  }, [resetCommand]);

  return {
    isReady,
    status,
    guidance,
    checklist,
    result,
    startCalibration,
    reset,
    frameProcessor,
  };
}

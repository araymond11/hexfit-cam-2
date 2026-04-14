import type { Keypoint } from '@tensorflow-models/pose-detection';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';
import { runAsync, runAtTargetFps, useFrameProcessor } from 'react-native-vision-camera';
import { useRunOnJS } from 'react-native-worklets-core';

import { type JumpResult } from '@/utils/jumpCalc';
import { KP, MIN_CONFIDENCE } from '@/utils/poseUtils';

const IS_IOS = Platform.OS === 'ios';
const MODEL_INPUT_SIZE = 256;
const CAPTURE_SIZE = 160;
const TARGET_PROCESS_FPS = 15;

const SETUP_LOCK_FRAMES = 14;
const BASELINE_FRAMES = 24;
const MIN_BASELINE_TORSO = 0.12;
const MAX_BASELINE_TORSO = 0.55;

const READY_ARM_MIN_MS = 350;
const READY_ARM_FRAMES = 6;
const READY_ARM_ACTIVE_WINDOW_MS = 2400;

const TAKEOFF_CONFIRM_FRAMES = 2;
const LANDING_CONFIRM_FRAMES = 2;
const MIN_AIRTIME_MS = 180;
const MAX_AIRTIME_MS = 900;
const MIN_AIRBORNE_BEFORE_LANDING_MS = 120;
const COOLDOWN_MS = 500;
const LANDED_HOLD_MS = 1200;

const TAKEOFF_UP_THRESHOLD = 0.055;
const TAKEOFF_HIP_THRESHOLD = 0.04;
const TAKEOFF_VEL_THRESHOLD = 0.45;
const TAKEOFF_CONTACT_MAX = 0.68;
const LANDING_UP_THRESHOLD = 0.08;
const LANDING_CONTACT_MIN = 0.64;
const LANDING_VEL_MAX = 1.6;
const COUNTERMOVE_THRESHOLD = 0.02;
const COUNTERMOVE_WINDOW_MS = 850;

const INVALID_SETUP_FRAMES = 4;
const RECOVER_SETUP_FRAMES = 10;
const MAX_TRACKING_LOSS_MS = 550;

const GRAVITY_M_S2 = 9.81;

type InternalStatus = 'IDLE' | 'CALIBRATING' | 'READY' | 'AIRBORNE' | 'LANDED' | 'INVALID_SETUP';

export type AirtimeStatus = InternalStatus;

export interface AirtimeChecklist {
  feetVisible: boolean;
  floorVisible: boolean;
  centered: boolean;
  lightingOk: boolean;
}

export interface UseJumpAirtimeV2Return {
  isReady: boolean;
  isArmed: boolean;
  status: AirtimeStatus;
  guidance: string;
  checklist: AirtimeChecklist;
  result: JumpResult | null;
  startCalibration: () => void;
  reset: () => void;
  frameProcessor: ReturnType<typeof useFrameProcessor>;
}

interface PoseSignals {
  ankleY: number;
  hipY: number;
  shoulderY: number;
  centerX: number;
  torso: number;
  anklesVisible: boolean;
  shouldersVisible: boolean;
  hipsVisible: boolean;
  fullBodyVisible: boolean;
  avgConfidence: number;
}

interface Baseline {
  ankleY: number;
  hipY: number;
  torso: number;
}

interface Sample {
  ts: number;
  ankleY: number;
  hipY: number;
  torso: number;
  centerX: number;
  avgConfidence: number;
  luminanceMean: number;
  fullBodyVisible: boolean;
  checklist: AirtimeChecklist;
}

interface PreviousSample {
  ts: number;
  ankleY: number;
  hipY: number;
  up: number;
  hipUp: number;
}

const DEFAULT_CHECKLIST: AirtimeChecklist = {
  feetVisible: false,
  floorVisible: false,
  centered: false,
  lightingOk: false,
};

function clamp01(v: number): number {
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
  return sorted[mid];
}

function reasonFromChecklist(checklist: AirtimeChecklist): string {
  if (!checklist.centered) return 'Move back so full body and floor are visible';
  if (!checklist.floorVisible) return 'Tilt camera slightly down';
  if (!checklist.feetVisible) return 'Keep both feet visible and clear from floor';
  if (!checklist.lightingOk) return 'Add front light and avoid bright backlight';
  return 'Adjust setup and hold still';
}

function reasonFromSetup(checklist: AirtimeChecklist, fullBodyVisible: boolean): string {
  if (!fullBodyVisible) return 'Move back so full body and floor are visible';
  return reasonFromChecklist(checklist);
}

function sampleLuminance(rgbData: number[]): { mean: number; std: number } {
  let sum = 0;
  let sumSq = 0;
  let count = 0;

  const stridePixels = 12;
  const stride = stridePixels * 3;

  for (let i = 0; i < rgbData.length; i += stride) {
    const r = rgbData[i] ?? 0;
    const g = rgbData[i + 1] ?? 0;
    const b = rgbData[i + 2] ?? 0;
    const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    sum += y;
    sumSq += y * y;
    count += 1;
  }

  if (count === 0) return { mean: 0, std: 0 };
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  return { mean, std: Math.sqrt(variance) };
}

function midpointY(a: Keypoint | undefined, b: Keypoint | undefined, minConf: number): number | null {
  const aOk = !!a && (a.score ?? 0) >= minConf;
  const bOk = !!b && (b.score ?? 0) >= minConf;
  if (aOk && bOk) return (a!.y + b!.y) / 2;
  if (aOk) return a!.y;
  if (bOk) return b!.y;
  return null;
}

function midpointX(a: Keypoint | undefined, b: Keypoint | undefined, minConf: number): number | null {
  const aOk = !!a && (a.score ?? 0) >= minConf;
  const bOk = !!b && (b.score ?? 0) >= minConf;
  if (aOk && bOk) return (a!.x + b!.x) / 2;
  if (aOk) return a!.x;
  if (bOk) return b!.x;
  return null;
}

function buildPoseSignals(kps: Keypoint[]): PoseSignals | null {
  const coreMinConf = Math.max(0.2, MIN_CONFIDENCE - 0.04);
  const ankleMinConf = Math.max(0.14, MIN_CONFIDENCE - 0.12);
  const faceMinConf = Math.max(0.12, MIN_CONFIDENCE - 0.14);

  const ls = kps[KP.LEFT_SHOULDER];
  const rs = kps[KP.RIGHT_SHOULDER];
  const lh = kps[KP.LEFT_HIP];
  const rh = kps[KP.RIGHT_HIP];
  const la = kps[KP.LEFT_ANKLE];
  const ra = kps[KP.RIGHT_ANKLE];
  const nose = kps[KP.NOSE];

  const shoulderY = midpointY(ls, rs, coreMinConf);
  const hipY = midpointY(lh, rh, coreMinConf);
  const ankleY = midpointY(la, ra, ankleMinConf);
  const shoulderX = midpointX(ls, rs, coreMinConf);
  const hipX = midpointX(lh, rh, coreMinConf);

  if (shoulderY === null || hipY === null || ankleY === null || shoulderX === null || hipX === null) {
    return null;
  }

  const torso = Math.max(1, hipY - shoulderY) / MODEL_INPUT_SIZE;
  const centerX = (shoulderX + hipX) / (2 * MODEL_INPUT_SIZE);

  const shouldersVisible = (ls?.score ?? 0) >= coreMinConf || (rs?.score ?? 0) >= coreMinConf;
  const hipsVisible = (lh?.score ?? 0) >= coreMinConf || (rh?.score ?? 0) >= coreMinConf;
  const anklesVisible = (la?.score ?? 0) >= ankleMinConf || (ra?.score ?? 0) >= ankleMinConf;

  const noseVisible = (nose?.score ?? 0) >= faceMinConf;
  const fullBodyVisible =
    shouldersVisible &&
    hipsVisible &&
    anklesVisible &&
    shoulderY / MODEL_INPUT_SIZE > 0.02 &&
    ankleY / MODEL_INPUT_SIZE < 0.995 &&
    (noseVisible || torso > 0.11);

  const tracked = [ls, rs, lh, rh, la, ra, nose];
  const validScores = tracked
    .map((kp) => kp?.score ?? 0)
    .filter((s) => s > 0);
  const avgConfidence =
    validScores.length > 0
      ? validScores.reduce((a, b) => a + b, 0) / validScores.length
      : 0;

  return {
    ankleY: ankleY / MODEL_INPUT_SIZE,
    hipY: hipY / MODEL_INPUT_SIZE,
    shoulderY: shoulderY / MODEL_INPUT_SIZE,
    centerX,
    torso,
    anklesVisible,
    shouldersVisible,
    hipsVisible,
    fullBodyVisible,
    avgConfidence,
  };
}

function buildChecklist(signals: PoseSignals, lumMean: number, lumStd: number): AirtimeChecklist {
  const centered = signals.centerX > 0.28 && signals.centerX < 0.72;
  const feetVisible = signals.anklesVisible && signals.ankleY < 0.995;
  const floorVisible = signals.anklesVisible && signals.ankleY < 0.985;
  const lightingOk =
    signals.avgConfidence >= 0.26 && lumMean > 20 && lumMean < 236 && lumStd > 5;

  return { centered, feetVisible, floorVisible, lightingOk };
}

function computeContact(sample: Sample, baseline: Baseline): number {
  const delta = Math.abs(sample.ankleY - baseline.ankleY) / Math.max(0.05, baseline.torso);
  return clamp01(1 - delta / 0.09);
}

function createResult(
  airtimeMs: number,
  avgConfidence: number,
  fpsEma: number,
  hadTrackingLoss: boolean,
  checklist: AirtimeChecklist,
): JumpResult {
  const flags: string[] = [];

  if (avgConfidence < 0.42) flags.push('LOW_CONFIDENCE');
  if (fpsEma > 0 && fpsEma < 12) flags.push('LOW_FPS');
  if (hadTrackingLoss) flags.push('TRACK_LOSS');
  if (!checklist.lightingOk) flags.push('LOW_LIGHT');

  const setupScore =
    (Number(checklist.centered) +
      Number(checklist.feetVisible) +
      Number(checklist.floorVisible) +
      Number(checklist.lightingOk)) /
    4;

  const confScore = clamp01((avgConfidence - 0.25) / 0.45);
  const fpsScore = fpsEma > 0 ? clamp01((fpsEma - 8) / 12) : 0.5;
  const weighted = 0.45 * setupScore + 0.35 * confScore + 0.2 * fpsScore - (hadTrackingLoss ? 0.2 : 0);

  let quality: JumpResult['quality'] = 'LOW';
  if (weighted >= 0.72) quality = 'HIGH';
  else if (weighted >= 0.48) quality = 'MEDIUM';

  const tSec = airtimeMs / 1000;
  const heightCm = (GRAVITY_M_S2 * tSec * tSec * 100) / 8;

  return {
    airtimeMs: Math.round(airtimeMs),
    airtimeSec: tSec,
    heightCm,
    quality,
    qualityFlags: flags.length > 0 ? flags : undefined,
  };
}

function toRawFrameDeltaMs(rawDelta: number): number {
  if (rawDelta <= 0) return 0;
  if (rawDelta > 10000) return rawDelta / 1_000_000;
  if (rawDelta < 1) return rawDelta * 1000;
  return rawDelta;
}

export function useJumpAirtimeV2(): UseJumpAirtimeV2Return {
  const [isReady, setIsReady] = useState(false);
  const [isArmed, setIsArmed] = useState(false);
  const [status, setStatus] = useState<AirtimeStatus>('IDLE');
  const [guidance, setGuidance] = useState('Tap Calibrate to begin');
  const [checklist, setChecklist] = useState<AirtimeChecklist>(DEFAULT_CHECKLIST);
  const [result, setResult] = useState<JumpResult | null>(null);

  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const isProcessingRef = useRef(false);
  const statusRef = useRef<AirtimeStatus>('IDLE');

  const baselineRef = useRef<Baseline | null>(null);
  const baselineBufRef = useRef<Sample[]>([]);
  const calibrationSetupStableFramesRef = useRef(0);

  const previousRef = useRef<PreviousSample | null>(null);

  const armStableFramesRef = useRef(0);
  const armedRef = useRef(false);
  const armedAtRef = useRef(0);
  const readyEnteredAtRef = useRef(0);
  const counterSeenAtRef = useRef(0);

  const takeoffConfirmRef = useRef(0);
  const landingConfirmRef = useRef(0);
  const takeoffTsRef = useRef(0);
  const cooldownUntilRef = useRef(0);
  const landedAtRef = useRef(0);

  const invalidSetupFramesRef = useRef(0);
  const recoverFramesRef = useRef(0);

  const lastRawTsRef = useRef(0);
  const nowMsRef = useRef(0);
  const dtEmaRef = useRef(0);
  const fpsEmaRef = useRef(0);

  const lastGoodTrackingTsRef = useRef(0);
  const trackingLossDuringAirRef = useRef(false);

  const checklistRef = useRef<AirtimeChecklist>(DEFAULT_CHECKLIST);
  const avgConfidenceRef = useRef(0);

  const setPhase = useCallback((next: AirtimeStatus) => {
    statusRef.current = next;
    setStatus(next);
  }, []);

  const setArmedState = useCallback((next: boolean) => {
    armedRef.current = next;
    setIsArmed(next);
  }, []);

  const resetTrackingState = useCallback(() => {
    baselineRef.current = null;
    baselineBufRef.current = [];
    calibrationSetupStableFramesRef.current = 0;
    previousRef.current = null;

    armStableFramesRef.current = 0;
    setArmedState(false);
    armedAtRef.current = 0;
    readyEnteredAtRef.current = 0;
    counterSeenAtRef.current = 0;

    takeoffConfirmRef.current = 0;
    landingConfirmRef.current = 0;
    takeoffTsRef.current = 0;
    cooldownUntilRef.current = 0;
    landedAtRef.current = 0;

    invalidSetupFramesRef.current = 0;
    recoverFramesRef.current = 0;

    lastRawTsRef.current = 0;
    nowMsRef.current = 0;
    dtEmaRef.current = 0;
    fpsEmaRef.current = 0;

    lastGoodTrackingTsRef.current = 0;
    trackingLossDuringAirRef.current = false;

    checklistRef.current = DEFAULT_CHECKLIST;
    avgConfidenceRef.current = 0;
  }, [setArmedState]);

  const updateNowMs = useCallback((frameTimestamp: number): number => {
    const lastRawTs = lastRawTsRef.current;
    if (lastRawTs <= 0) {
      lastRawTsRef.current = frameTimestamp;
      return 0;
    }

    const rawDelta = frameTimestamp - lastRawTs;
    lastRawTsRef.current = frameTimestamp;

    let dtMs = toRawFrameDeltaMs(rawDelta);
    if (!Number.isFinite(dtMs) || dtMs <= 0 || dtMs > 300) {
      dtMs = dtEmaRef.current > 0 ? dtEmaRef.current : 1000 / 30;
    }

    nowMsRef.current += dtMs;

    if (dtEmaRef.current <= 0) {
      dtEmaRef.current = dtMs;
    } else {
      dtEmaRef.current = dtEmaRef.current * 0.9 + dtMs * 0.1;
    }
    fpsEmaRef.current = dtEmaRef.current > 0 ? 1000 / dtEmaRef.current : 0;

    return dtMs;
  }, []);

  const updateChecklist = useCallback((next: AirtimeChecklist) => {
    checklistRef.current = next;
    setChecklist(next);
  }, []);

  const enterReady = useCallback(() => {
    setPhase('READY');
    armStableFramesRef.current = 0;
    setArmedState(false);
    armedAtRef.current = 0;
    readyEnteredAtRef.current = nowMsRef.current;
    counterSeenAtRef.current = 0;
    takeoffConfirmRef.current = 0;
    landingConfirmRef.current = 0;
    invalidSetupFramesRef.current = 0;
    recoverFramesRef.current = 0;
  }, [setArmedState, setPhase]);

  const processSample = useCallback(
    (sample: Sample) => {
      const phase = statusRef.current;
      const checklistNow = sample.checklist;
      updateChecklist(checklistNow);
      avgConfidenceRef.current = sample.avgConfidence;
      const setupValid =
        sample.fullBodyVisible &&
        checklistNow.centered &&
        checklistNow.feetVisible &&
        checklistNow.floorVisible;

      if (phase === 'IDLE') {
        setGuidance('Tap Calibrate to begin');
        return;
      }

      if (phase === 'CALIBRATING') {
        if (!setupValid) {
          calibrationSetupStableFramesRef.current = 0;
          baselineBufRef.current = [];
          setGuidance(reasonFromSetup(checklistNow, sample.fullBodyVisible));
          return;
        }

        if (calibrationSetupStableFramesRef.current < SETUP_LOCK_FRAMES) {
          calibrationSetupStableFramesRef.current += 1;
          setGuidance(
            `Hold still... locking setup (${calibrationSetupStableFramesRef.current}/${SETUP_LOCK_FRAMES})`,
          );
          return;
        }

        if (sample.torso < MIN_BASELINE_TORSO || sample.torso > MAX_BASELINE_TORSO) {
          setGuidance('Move to a normal distance from the camera');
          baselineBufRef.current = [];
          return;
        }

        baselineBufRef.current.push(sample);
        if (baselineBufRef.current.length > BASELINE_FRAMES) {
          baselineBufRef.current.shift();
        }

        setGuidance(`Calibrating... hold still (${baselineBufRef.current.length}/${BASELINE_FRAMES})`);

        if (baselineBufRef.current.length < BASELINE_FRAMES) return;

        const baselineSamples = baselineBufRef.current;
        baselineRef.current = {
          ankleY: median(baselineSamples.map((s) => s.ankleY)),
          hipY: median(baselineSamples.map((s) => s.hipY)),
          torso: median(baselineSamples.map((s) => s.torso)),
        };
        previousRef.current = {
          ts: sample.ts,
          ankleY: sample.ankleY,
          hipY: sample.hipY,
          up: 0,
          hipUp: 0,
        };

        enterReady();
        setGuidance('Hold still to arm detection, then jump');
        return;
      }

      if (phase === 'INVALID_SETUP') {
        if (setupValid) {
          recoverFramesRef.current += 1;
          if (recoverFramesRef.current >= RECOVER_SETUP_FRAMES) {
            recoverFramesRef.current = 0;
            enterReady();
            setGuidance('Hold still to arm detection, then jump');
          } else {
            setGuidance('Setup looks better. Hold position...');
          }
        } else {
          recoverFramesRef.current = 0;
          setGuidance(reasonFromSetup(checklistNow, sample.fullBodyVisible));
        }
        return;
      }

      if (phase === 'LANDED') {
        if (sample.ts - landedAtRef.current > LANDED_HOLD_MS) {
          enterReady();
          setGuidance('Hold still to arm detection, then jump');
        }
        return;
      }

      const baseline = baselineRef.current;
      if (!baseline) {
        setPhase('CALIBRATING');
        setGuidance('Baseline missing. Tap Calibrate');
        return;
      }

      const previous = previousRef.current;
      const dtSec = previous
        ? Math.max(0.016, Math.min(0.12, (sample.ts - previous.ts) / 1000))
        : 1 / 30;

      const up = (baseline.ankleY - sample.ankleY) / Math.max(0.05, baseline.torso);
      const hipUp = (baseline.hipY - sample.hipY) / Math.max(0.05, baseline.torso);
      const ankleVel = previous
        ? (previous.ankleY - sample.ankleY) / Math.max(0.05, baseline.torso) / dtSec
        : 0;
      const hipVel = previous
        ? (previous.hipY - sample.hipY) / Math.max(0.05, baseline.torso) / dtSec
        : 0;
      const contact = computeContact(sample, baseline);

      previousRef.current = {
        ts: sample.ts,
        ankleY: sample.ankleY,
        hipY: sample.hipY,
        up,
        hipUp,
      };

      if (!setupValid && phase !== 'AIRBORNE') {
        invalidSetupFramesRef.current += 1;
        setArmedState(false);
        armStableFramesRef.current = 0;
        takeoffConfirmRef.current = 0;

        if (invalidSetupFramesRef.current >= INVALID_SETUP_FRAMES) {
          setPhase('INVALID_SETUP');
          setGuidance(reasonFromSetup(checklistNow, sample.fullBodyVisible));
        } else {
          setGuidance(reasonFromSetup(checklistNow, sample.fullBodyVisible));
        }
        return;
      }

      invalidSetupFramesRef.current = Math.max(0, invalidSetupFramesRef.current - 1);

      if (phase === 'READY') {
        if (sample.ts < cooldownUntilRef.current) {
          setGuidance('Ready - settle, then jump');
          return;
        }

          const stablePose =
          contact > 0.78 &&
          Math.abs(ankleVel) < 0.9 &&
          Math.abs(hipVel) < 0.9;

        if (!armedRef.current) {
          if (stablePose) {
            armStableFramesRef.current += 1;
          } else {
            armStableFramesRef.current = Math.max(0, armStableFramesRef.current - 1);
          }

          const canArm =
            sample.ts - readyEnteredAtRef.current >= READY_ARM_MIN_MS &&
            armStableFramesRef.current >= READY_ARM_FRAMES;
          if (canArm) {
            setArmedState(true);
            armedAtRef.current = sample.ts;
            setGuidance('Jump now');
          } else {
            setGuidance('Hold still to arm detection, then jump');
          }
          return;
        }

        if (sample.ts - armedAtRef.current > READY_ARM_ACTIVE_WINDOW_MS) {
          setArmedState(false);
          armStableFramesRef.current = Math.max(0, READY_ARM_FRAMES - 2);
          setGuidance('Hold still to arm detection, then jump');
          return;
        }

        const down = (sample.ankleY - baseline.ankleY) / Math.max(0.05, baseline.torso);
        if (down > COUNTERMOVE_THRESHOLD) {
          counterSeenAtRef.current = sample.ts;
        }

        const inCounterWindow =
          counterSeenAtRef.current > 0 &&
          sample.ts - counterSeenAtRef.current <= COUNTERMOVE_WINDOW_MS;

        const strongRise = up > TAKEOFF_UP_THRESHOLD || hipUp > TAKEOFF_HIP_THRESHOLD;
        const risingVelocity =
          ankleVel > TAKEOFF_VEL_THRESHOLD ||
          hipVel > TAKEOFF_VEL_THRESHOLD * 0.75;
        const takeoffCandidate =
          armedRef.current &&
          contact < TAKEOFF_CONTACT_MAX &&
          strongRise &&
          (risingVelocity || up > TAKEOFF_UP_THRESHOLD + 0.03) &&
          (inCounterWindow || up > TAKEOFF_UP_THRESHOLD + 0.02);

        if (takeoffCandidate) {
          takeoffConfirmRef.current += 1;
          if (takeoffConfirmRef.current >= TAKEOFF_CONFIRM_FRAMES) {
            setPhase('AIRBORNE');
            setGuidance('Airborne');
            takeoffTsRef.current = sample.ts;
            landingConfirmRef.current = 0;
            trackingLossDuringAirRef.current = false;
            setArmedState(false);
            takeoffConfirmRef.current = 0;
          }
        } else {
          takeoffConfirmRef.current = 0;
          setGuidance('Jump now');
        }
        return;
      }

      if (phase === 'AIRBORNE') {
        const airborneMs = sample.ts - takeoffTsRef.current;

        if (airborneMs > MAX_AIRTIME_MS) {
          cooldownUntilRef.current = sample.ts + COOLDOWN_MS;
          enterReady();
          setGuidance('Jump not captured reliably. Try again.');
          return;
        }

        if (airborneMs < MIN_AIRBORNE_BEFORE_LANDING_MS) {
          setGuidance('Airborne');
          return;
        }

        const landingCandidate =
          (contact > LANDING_CONTACT_MIN &&
            up < LANDING_UP_THRESHOLD &&
            Math.abs(ankleVel) < LANDING_VEL_MAX) ||
          (contact > LANDING_CONTACT_MIN + 0.1 &&
            up < LANDING_UP_THRESHOLD + 0.04);

        if (landingCandidate) {
          landingConfirmRef.current += 1;
        } else {
          landingConfirmRef.current = 0;
        }

        if (landingConfirmRef.current >= LANDING_CONFIRM_FRAMES) {
          const airtimeMs = sample.ts - takeoffTsRef.current;
          landingConfirmRef.current = 0;
          cooldownUntilRef.current = sample.ts + COOLDOWN_MS;

          if (airtimeMs >= MIN_AIRTIME_MS && airtimeMs <= MAX_AIRTIME_MS) {
            const resultPayload = createResult(
              airtimeMs,
              avgConfidenceRef.current,
              fpsEmaRef.current,
              trackingLossDuringAirRef.current,
              checklistRef.current,
            );
            setResult(resultPayload);
            landedAtRef.current = sample.ts;
            setPhase('LANDED');
            setGuidance('Landed - result captured');
          } else {
            enterReady();
            setGuidance('Jump not captured reliably. Try again.');
          }
        }
      }
    },
    [enterReady, setArmedState, setPhase, updateChecklist],
  );

  const onFrameCapture = useCallback(
    async (
      rgbData: number[],
      lbOffsetX: number,
      lbOffsetY: number,
      frameTimestamp: number,
    ) => {
      if (!detectorRef.current || isProcessingRef.current) return;
      isProcessingRef.current = true;

      try {
        updateNowMs(frameTimestamp);
        const nowMs = nowMsRef.current;

        const rgb = new Uint8Array(rgbData);
        const small = tf.tensor3d(rgb, [CAPTURE_SIZE, CAPTURE_SIZE, 3]);
        const tensor = tf.image.resizeBilinear(small, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
        tf.dispose(small);

        const poses = await detectorRef.current.estimatePoses(
          tensor as unknown as HTMLVideoElement,
        );
        tf.dispose(tensor);

        if (poses.length === 0) {
          updateChecklist(DEFAULT_CHECKLIST);
          avgConfidenceRef.current = 0;

          if (statusRef.current === 'CALIBRATING') {
            setGuidance('No person detected - step into guide and hold still');
          } else if (statusRef.current === 'READY' || statusRef.current === 'INVALID_SETUP') {
            invalidSetupFramesRef.current += 1;
            if (invalidSetupFramesRef.current >= INVALID_SETUP_FRAMES) {
              setPhase('INVALID_SETUP');
              setGuidance('Step into guide so full body and feet are visible');
            }
          } else if (statusRef.current === 'AIRBORNE') {
            if (lastGoodTrackingTsRef.current > 0 && nowMs - lastGoodTrackingTsRef.current > MAX_TRACKING_LOSS_MS) {
              trackingLossDuringAirRef.current = true;
              const airborneMs = nowMs - takeoffTsRef.current;
              if (airborneMs > MAX_AIRTIME_MS + 220) {
                cooldownUntilRef.current = nowMs + COOLDOWN_MS;
                enterReady();
                setGuidance('Jump not captured reliably. Try again.');
              } else {
                setGuidance('Airborne - recovering tracking...');
              }
            }
          }
          return;
        }

        lastGoodTrackingTsRef.current = nowMs;

        const kps = poses[0].keypoints;
        const contentW = MODEL_INPUT_SIZE - 2 * lbOffsetX;
        const contentH = MODEL_INPUT_SIZE - 2 * lbOffsetY;

        if (contentW <= 0 || contentH <= 0) return;

        const clamp = (v: number) => Math.max(0, Math.min(MODEL_INPUT_SIZE, v));
        const unletterboxed = kps.map((kp) => ({
          ...kp,
          x: clamp(((kp.x - lbOffsetX) / contentW) * MODEL_INPUT_SIZE),
          y: clamp(((kp.y - lbOffsetY) / contentH) * MODEL_INPUT_SIZE),
        }));

        const signals = buildPoseSignals(unletterboxed);
        if (!signals) {
          updateChecklist(DEFAULT_CHECKLIST);
          if (statusRef.current === 'CALIBRATING') {
            setGuidance('Step back - full body and ankles must be visible');
          } else if (statusRef.current === 'READY' || statusRef.current === 'INVALID_SETUP') {
            invalidSetupFramesRef.current += 1;
            if (invalidSetupFramesRef.current >= INVALID_SETUP_FRAMES) {
              setPhase('INVALID_SETUP');
              setGuidance('Step into guide so full body and feet are visible');
            }
          } else if (statusRef.current === 'AIRBORNE') {
            trackingLossDuringAirRef.current = true;
            setGuidance('Airborne - recovering tracking...');
          }
          return;
        }

        const lum = sampleLuminance(rgbData);
        const checklistNow = buildChecklist(signals, lum.mean, lum.std);

        const sample: Sample = {
          ts: nowMs,
          ankleY: signals.ankleY,
          hipY: signals.hipY,
          torso: signals.torso,
          centerX: signals.centerX,
          avgConfidence: signals.avgConfidence,
          luminanceMean: lum.mean,
          fullBodyVisible: signals.fullBodyVisible,
          checklist: checklistNow,
        };

        processSample(sample);
      } catch (error) {
        console.warn('useJumpAirtimeV2 frame processing error:', error);
      } finally {
        isProcessingRef.current = false;
      }
    },
    [enterReady, processSample, setPhase, updateChecklist, updateNowMs],
  );

  const onFrameCaptureWorklet = useRunOnJS(onFrameCapture, [onFrameCapture]);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      runAtTargetFps(TARGET_PROCESS_FPS, () => {
        runAsync(frame, () => {
          'worklet';

          const buffer = frame.toArrayBuffer();
          const src = new Uint8Array(buffer);
          const outSize = CAPTURE_SIZE;

          const srcW = frame.width;
          const srcH = frame.height;
          const srcBytesPerRow = frame.bytesPerRow;
          const orientation = frame.orientation;
          const swapAxes = orientation === 'landscape-left' || orientation === 'landscape-right';
          const orientedW = swapAxes ? srcH : srcW;
          const orientedH = swapAxes ? srcW : srcH;

          const scale = Math.min(outSize / orientedW, outSize / orientedH);
          const scaledW = Math.floor(orientedW * scale);
          const scaledH = Math.floor(orientedH * scale);
          const offsetX = Math.floor((outSize - scaledW) / 2);
          const offsetY = Math.floor((outSize - scaledH) / 2);

          const rgb = new Array(outSize * outSize * 3).fill(0);

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
                rgb[di] = src[si + 2];
                rgb[di + 1] = src[si + 1];
                rgb[di + 2] = src[si];
              } else {
                rgb[di] = src[si];
                rgb[di + 1] = src[si + 1];
                rgb[di + 2] = src[si + 2];
              }
            }
          }

          const lbOffsetX = offsetX * (MODEL_INPUT_SIZE / outSize);
          const lbOffsetY = offsetY * (MODEL_INPUT_SIZE / outSize);

          onFrameCaptureWorklet(rgb, lbOffsetX, lbOffsetY, frame.timestamp);
        });
      });
    },
    [onFrameCaptureWorklet],
  );

  useEffect(() => {
    let mounted = true;
    setIsReady(false);

    (async () => {
      await tf.ready();
      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true,
        },
      );

      if (mounted) {
        detectorRef.current = detector;
        setIsReady(true);
      } else {
        detector.dispose();
      }
    })().catch((error) => {
      console.warn('useJumpAirtimeV2 initialization error:', error);
    });

    return () => {
      mounted = false;
      detectorRef.current?.dispose();
      detectorRef.current = null;
    };
  }, []);

  const startCalibration = useCallback(() => {
    resetTrackingState();
    setResult(null);
    setPhase('CALIBRATING');
    setGuidance('Move into position and hold still for calibration');
  }, [resetTrackingState, setPhase]);

  const reset = useCallback(() => {
    resetTrackingState();
    setResult(null);
    setPhase('IDLE');
    setGuidance('Tap Calibrate to begin');
    setChecklist(DEFAULT_CHECKLIST);
  }, [resetTrackingState, setPhase]);

  return {
    isReady,
    isArmed,
    status,
    guidance,
    checklist,
    result,
    startCalibration,
    reset,
    frameProcessor,
  };
}

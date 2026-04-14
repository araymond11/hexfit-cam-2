import assert from 'node:assert/strict';
import test from 'node:test';

import {
  analyzeJumpLandmarks,
  resolveAdaptiveConfirmFrames,
  suggestJumpAttemptWindow,
} from '../utils/jumpAnalysis.ts';
import { heightFromFlightTime, type JumpKeypoint, type JumpLandmarkFrame } from '../utils/jumpCalc.ts';
import { KP } from '../utils/poseUtils.ts';

interface FrameConfig {
  centerX?: number;
  shoulderY?: number;
  hipY?: number;
  leftKneeY?: number;
  rightKneeY?: number;
  leftAnkleY?: number;
  rightAnkleY?: number;
  leftHeelY?: number;
  rightHeelY?: number;
  leftToeY?: number;
  rightToeY?: number;
  faceVisible?: boolean;
  score?: number;
  personCount?: number;
}

function emptyKeypoints(): JumpKeypoint[] {
  return Array.from({ length: 33 }, (_, index) => ({
    x: 0.5,
    y: 1.15,
    score: 0,
    name: `kp-${index}`,
  }));
}

function setPoint(
  keypoints: JumpKeypoint[],
  index: number,
  x: number,
  y: number,
  score: number,
) {
  keypoints[index] = { x, y, score };
}

function makeFrame(frameIndex: number, timestampMs: number, config: FrameConfig = {}): JumpLandmarkFrame {
  const {
    centerX = 0.5,
    shoulderY = 0.29,
    hipY = 0.54,
    leftKneeY = 0.72,
    rightKneeY = 0.72,
    leftAnkleY = 0.865,
    rightAnkleY = 0.865,
    leftHeelY = 0.895,
    rightHeelY = 0.895,
    leftToeY = 0.905,
    rightToeY = 0.905,
    faceVisible = true,
    score = 0.96,
    personCount = 1,
  } = config;

  const keypoints = emptyKeypoints();
  const leftX = centerX - 0.1;
  const rightX = centerX + 0.1;

  if (faceVisible) {
    setPoint(keypoints, KP.NOSE, centerX, 0.12, score);
    setPoint(keypoints, KP.LEFT_EYE, centerX - 0.03, 0.11, score);
    setPoint(keypoints, KP.RIGHT_EYE, centerX + 0.03, 0.11, score);
    setPoint(keypoints, KP.LEFT_EAR, centerX - 0.06, 0.13, score);
    setPoint(keypoints, KP.RIGHT_EAR, centerX + 0.06, 0.13, score);
  }

  setPoint(keypoints, KP.LEFT_SHOULDER, centerX - 0.08, shoulderY, score);
  setPoint(keypoints, KP.RIGHT_SHOULDER, centerX + 0.08, shoulderY, score);
  setPoint(keypoints, KP.LEFT_HIP, centerX - 0.06, hipY, score);
  setPoint(keypoints, KP.RIGHT_HIP, centerX + 0.06, hipY, score);
  setPoint(keypoints, KP.LEFT_KNEE, leftX, leftKneeY, score);
  setPoint(keypoints, KP.RIGHT_KNEE, rightX, rightKneeY, score);
  setPoint(keypoints, KP.LEFT_ANKLE, leftX, leftAnkleY, score);
  setPoint(keypoints, KP.RIGHT_ANKLE, rightX, rightAnkleY, score);
  setPoint(keypoints, KP.LEFT_HEEL, leftX - 0.015, leftHeelY, score);
  setPoint(keypoints, KP.RIGHT_HEEL, rightX + 0.015, rightHeelY, score);
  setPoint(keypoints, KP.LEFT_FOOT_INDEX, leftX - 0.005, leftToeY, score);
  setPoint(keypoints, KP.RIGHT_FOOT_INDEX, rightX + 0.005, rightToeY, score);

  return {
    frameIndex,
    timestampMs,
    keypoints,
    avgConfidence: score,
    personCount,
  };
}

function makeValidFlightFrames(): JumpLandmarkFrame[] {
  const timestamps = [
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
  ];

  const sequence = [
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.895, rightHeelY: 0.895 },
    { leftToeY: 0.904, rightToeY: 0.906, leftHeelY: 0.894, rightHeelY: 0.896 },
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.895, rightHeelY: 0.894 },
    { leftToeY: 0.906, rightToeY: 0.905, leftHeelY: 0.895, rightHeelY: 0.895 },
    { leftToeY: 0.905, rightToeY: 0.904, leftHeelY: 0.894, rightHeelY: 0.894 },
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.895, rightHeelY: 0.895 },
    { leftToeY: 0.904, rightToeY: 0.905, leftHeelY: 0.894, rightHeelY: 0.895 },
    { leftToeY: 0.905, rightToeY: 0.906, leftHeelY: 0.895, rightHeelY: 0.896 },
    { leftToeY: 0.845, rightToeY: 0.846, leftHeelY: 0.852, rightHeelY: 0.853, leftAnkleY: 0.81, rightAnkleY: 0.81 },
    { leftToeY: 0.825, rightToeY: 0.824, leftHeelY: 0.832, rightHeelY: 0.833, leftAnkleY: 0.79, rightAnkleY: 0.79 },
    { leftToeY: 0.817, rightToeY: 0.816, leftHeelY: 0.824, rightHeelY: 0.825, leftAnkleY: 0.782, rightAnkleY: 0.782 },
    { leftToeY: 0.821, rightToeY: 0.82, leftHeelY: 0.828, rightHeelY: 0.829, leftAnkleY: 0.786, rightAnkleY: 0.786 },
    { leftToeY: 0.842, rightToeY: 0.841, leftHeelY: 0.849, rightHeelY: 0.85, leftAnkleY: 0.808, rightAnkleY: 0.808 },
    { leftToeY: 0.904, rightToeY: 0.905, leftHeelY: 0.895, rightHeelY: 0.895 },
    { leftToeY: 0.905, rightToeY: 0.906, leftHeelY: 0.895, rightHeelY: 0.896 },
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.894, rightHeelY: 0.895 },
  ];

  return timestamps.map((timestamp, index) =>
    makeFrame(index, timestamp, sequence[index]),
  );
}

function makeDelayedCalibrationFrames(): JumpLandmarkFrame[] {
  const movingPrefix = [
    makeFrame(0, 0, { faceVisible: false }),
    makeFrame(1, 100, { faceVisible: false }),
    makeFrame(2, 200, { faceVisible: false }),
    makeFrame(3, 300, { faceVisible: false }),
    makeFrame(4, 400, { faceVisible: false }),
    makeFrame(5, 500, { faceVisible: false }),
    makeFrame(6, 600, { faceVisible: false }),
    makeFrame(7, 700, { faceVisible: false }),
  ];

  const shifted = makeValidFlightFrames().map((frame, index) => ({
    ...frame,
    frameIndex: movingPrefix.length + index,
    timestampMs: frame.timestampMs + 800,
  }));

  return [...movingPrefix, ...shifted];
}

function makeToeContactPriorityFrames(): JumpLandmarkFrame[] {
  const timestamps = [
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
  ];

  const sequence = [
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.895, rightHeelY: 0.895 },
    { leftToeY: 0.905, rightToeY: 0.904, leftHeelY: 0.895, rightHeelY: 0.894 },
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.894, rightHeelY: 0.895 },
    { leftToeY: 0.904, rightToeY: 0.905, leftHeelY: 0.894, rightHeelY: 0.895 },
    { leftToeY: 0.905, rightToeY: 0.906, leftHeelY: 0.895, rightHeelY: 0.896 },
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.895, rightHeelY: 0.895 },
    { leftToeY: 0.904, rightToeY: 0.905, leftHeelY: 0.894, rightHeelY: 0.895 },
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.895, rightHeelY: 0.895 },
    { leftToeY: 0.905, rightToeY: 0.904, leftHeelY: 0.872, rightHeelY: 0.871, leftAnkleY: 0.84, rightAnkleY: 0.84 },
    { leftToeY: 0.856, rightToeY: 0.855, leftHeelY: 0.842, rightHeelY: 0.841, leftAnkleY: 0.812, rightAnkleY: 0.812 },
    { leftToeY: 0.834, rightToeY: 0.833, leftHeelY: 0.82, rightHeelY: 0.819, leftAnkleY: 0.79, rightAnkleY: 0.79 },
    { leftToeY: 0.846, rightToeY: 0.845, leftHeelY: 0.832, rightHeelY: 0.831, leftAnkleY: 0.802, rightAnkleY: 0.802 },
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.894, rightHeelY: 0.894, leftAnkleY: 0.865, rightAnkleY: 0.865 },
    { leftToeY: 0.905, rightToeY: 0.905, leftHeelY: 0.895, rightHeelY: 0.895 },
    { leftToeY: 0.905, rightToeY: 0.904, leftHeelY: 0.895, rightHeelY: 0.894 },
  ];

  return timestamps.map((timestamp, index) => makeFrame(index, timestamp, sequence[index]));
}

function makeLongClipWithOneJump(): JumpLandmarkFrame[] {
  const prefix = Array.from({ length: 25 }, (_, index) =>
    makeFrame(index, index * 200, {
      faceVisible: false,
      centerX: 0.2 + (index % 5) * 0.08,
    }),
  );

  const shiftedJump = makeValidFlightFrames().map((frame, index) => ({
    ...frame,
    frameIndex: prefix.length + index,
    timestampMs: frame.timestampMs + 5000,
  }));

  const suffixStart = shiftedJump[shiftedJump.length - 1].timestampMs + 200;
  const suffix = Array.from({ length: 15 }, (_, index) =>
    makeFrame(prefix.length + shiftedJump.length + index, suffixStart + index * 200, {
      centerX: 0.5 + (index % 4) * 0.05,
    }),
  );

  return [...prefix, ...shiftedJump, ...suffix];
}

test('heightFromFlightTime converts airtime to centimeters', () => {
  assert.ok(Math.abs(heightFromFlightTime(500) - 30.65625) < 0.001);
});

test('landmark fixture yields deterministic takeoff and landing timestamps', () => {
  const result = analyzeJumpLandmarks(makeValidFlightFrames(), {
    videoDurationMs: 1600,
    sampleFps: 240,
    videoFps: 240,
  });

  assert.equal(result.invalidReason, undefined);
  assert.equal(result.takeoffMs, 700);
  assert.equal(result.landingMs, 1300);
  assert.equal(result.flightMs, 600);
  assert.ok(result.heightCm !== null);
  assert.ok(Math.abs((result.heightCm ?? 0) - heightFromFlightTime(600)) < 0.001);
});

test('timing analysis tolerates small toe jitter', () => {
  const frames = makeValidFlightFrames().map((frame, index) => {
    if (index === 9) {
      return makeFrame(frame.frameIndex, frame.timestampMs, {
        leftToeY: 0.836,
        rightToeY: 0.832,
        leftHeelY: 0.844,
        rightHeelY: 0.839,
        leftAnkleY: 0.8,
        rightAnkleY: 0.797,
      });
    }
    if (index === 13) {
      return makeFrame(frame.frameIndex, frame.timestampMs, {
        leftToeY: 0.903,
        rightToeY: 0.907,
        leftHeelY: 0.894,
        rightHeelY: 0.897,
      });
    }
    return frame;
  });

  const result = analyzeJumpLandmarks(frames, {
    videoDurationMs: 1600,
    sampleFps: 240,
    videoFps: 240,
  });

  assert.equal(result.invalidReason, undefined);
  assert.equal(result.takeoffMs, 700);
  assert.equal(result.landingMs, 1300);
});

test('calibration can be found later in a long clip before the jump', () => {
  const result = analyzeJumpLandmarks(makeDelayedCalibrationFrames(), {
    videoDurationMs: 2400,
    sampleFps: 240,
    videoFps: 240,
  });

  assert.equal(result.invalidReason, undefined);
  assert.equal(result.takeoffMs, 1500);
  assert.equal(result.landingMs, 2100);
  assert.equal(result.debug.calibrationEndMs, 1500);
});

test('takeoff is anchored to the last toe-contact frame before the clear sequence', () => {
  const result = analyzeJumpLandmarks(makeToeContactPriorityFrames(), {
    videoDurationMs: 1500,
    sampleFps: 240,
    videoFps: 240,
  });

  assert.equal(result.invalidReason, undefined);
  assert.equal(result.takeoffMs, 800);
  assert.equal(result.landingMs, 1200);
  assert.equal(result.flightMs, 400);
});

test('attempt window suggestion isolates the single jump inside a long clip', () => {
  const frames = makeLongClipWithOneJump();
  const suggestion = suggestJumpAttemptWindow(frames, {
    videoDurationMs: 9000,
    sampleFps: 30,
    videoFps: 30,
  });

  assert.ok(suggestion);
  assert.equal(suggestion?.takeoffMs, 5700);
  assert.equal(suggestion?.landingMs, 6300);
  assert.ok((suggestion?.startMs ?? 0) < 5700);
  assert.ok((suggestion?.endMs ?? 0) > 6300);
});

test('adaptive confirmation uses one frame only for high-fps strong signals', () => {
  assert.equal(resolveAdaptiveConfirmFrames(240, 0.8, 0.7, 2), 1);
  assert.equal(resolveAdaptiveConfirmFrames(240, 0.6, 0.7, 2), 2);
  assert.equal(resolveAdaptiveConfirmFrames(60, 0.9, 0.7, 2), 2);
});

test('timing ambiguous suppresses physical height even when playback events are found', () => {
  const result = analyzeJumpLandmarks(makeValidFlightFrames(), {
    videoDurationMs: 1600,
    playbackDurationMs: 1600,
    captureDurationMs: 1600,
    sampleFps: 240,
    videoFps: 240,
    timingMode: 'timing_ambiguous',
    timingConfidence: 0.2,
    timebaseSource: 'photos_segments_mismatch',
  });

  assert.equal(result.invalidReason, 'TIMING_AMBIGUOUS');
  assert.equal(result.takeoffMs, 700);
  assert.equal(result.landingMs, 1300);
  assert.equal(result.flightMs, null);
  assert.equal(result.heightCm, null);
});

test('event order invalid is rejected instead of surfacing negative flight time', () => {
  const frames = makeValidFlightFrames().map((frame) => ({
    ...frame,
    captureTimestampMs: 1600 - frame.timestampMs,
  }));

  const result = analyzeJumpLandmarks(frames, {
    videoDurationMs: 1600,
    playbackDurationMs: 1600,
    captureDurationMs: 1600,
    sampleFps: 240,
    videoFps: 240,
    timingMode: 'playback_is_physical',
    timingConfidence: 0.98,
    timebaseSource: 'playback_asset_flat_hfr',
  });

  assert.equal(result.invalidReason, 'EVENT_ORDER_INVALID');
  assert.equal(result.takeoffMs, 700);
  assert.equal(result.landingMs, 1300);
  assert.equal(result.takeoffPhysicalMs, null);
  assert.equal(result.landingPhysicalMs, null);
  assert.equal(result.flightMs, null);
  assert.equal(result.heightCm, null);
});

test('missing face or upper body is rejected before event detection', () => {
  const frames = makeValidFlightFrames().map((frame) =>
    makeFrame(frame.frameIndex, frame.timestampMs, {
      faceVisible: false,
      leftToeY: frame.keypoints[KP.LEFT_FOOT_INDEX].y,
      rightToeY: frame.keypoints[KP.RIGHT_FOOT_INDEX].y,
      leftHeelY: frame.keypoints[KP.LEFT_HEEL].y,
      rightHeelY: frame.keypoints[KP.RIGHT_HEEL].y,
      leftAnkleY: frame.keypoints[KP.LEFT_ANKLE].y,
      rightAnkleY: frame.keypoints[KP.RIGHT_ANKLE].y,
      hipY: frame.keypoints[KP.LEFT_HIP].y,
      shoulderY: frame.keypoints[KP.LEFT_SHOULDER].y,
    }),
  );

  const result = analyzeJumpLandmarks(frames, {
    videoDurationMs: 1600,
    sampleFps: 240,
    videoFps: 240,
  });

  assert.equal(result.invalidReason, 'BODY_NOT_FULLY_VISIBLE');
});

test('unstable calibration is rejected before event detection', () => {
  const frames = makeValidFlightFrames().map((frame, index) => {
    if (frame.timestampMs <= 700) {
      const leftToeY = index % 2 === 0 ? 0.87 : 0.94;
      const rightToeY = index % 2 === 0 ? 0.872 : 0.942;
      return makeFrame(frame.frameIndex, frame.timestampMs, {
        leftToeY,
        rightToeY,
        leftHeelY: leftToeY - 0.01,
        rightHeelY: rightToeY - 0.01,
      });
    }
    return frame;
  });

  const result = analyzeJumpLandmarks(frames, {
    videoDurationMs: 1600,
    sampleFps: 240,
    videoFps: 240,
  });

  assert.equal(result.invalidReason, 'NO_STABLE_CALIBRATION');
});

test('missing landing is rejected after takeoff', () => {
  const frames = makeValidFlightFrames().map((frame) =>
    frame.timestampMs >= 1300
      ? makeFrame(frame.frameIndex, frame.timestampMs, {
          leftToeY: 0.82,
          rightToeY: 0.819,
          leftHeelY: 0.828,
          rightHeelY: 0.827,
          leftAnkleY: 0.786,
          rightAnkleY: 0.785,
        })
      : frame,
  );

  const result = analyzeJumpLandmarks(frames, {
    videoDurationMs: 1600,
    sampleFps: 240,
    videoFps: 240,
  });

  assert.equal(result.takeoffMs, 700);
  assert.equal(result.invalidReason, 'NO_LANDING');
});

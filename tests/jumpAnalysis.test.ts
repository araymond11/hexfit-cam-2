import assert from 'node:assert/strict';
import test from 'node:test';

import { analyzeJumpLandmarks } from '../utils/jumpAnalysis.ts';
import { heightFromFlightTime, type JumpLandmarkFrame, type JumpKeypoint } from '../utils/jumpCalc.ts';
import { KP } from '../utils/poseUtils.ts';

function makeKeypoint(x: number, y: number, score = 0): JumpKeypoint {
  return { x, y, score };
}

function makeFrame(
  frameIndex: number,
  timestampMs: number,
  leftAnkleY: number,
  rightAnkleY: number,
  centerX = 0.5,
  personCount = 1,
): JumpLandmarkFrame {
  const keypoints = Array.from({ length: 17 }, () => makeKeypoint(0, 0, 0));

  keypoints[KP.NOSE] = makeKeypoint(centerX, 0.12, 0.98);
  keypoints[KP.LEFT_EYE] = makeKeypoint(centerX - 0.015, 0.11, 0.97);
  keypoints[KP.RIGHT_EYE] = makeKeypoint(centerX + 0.015, 0.11, 0.97);
  keypoints[KP.LEFT_EAR] = makeKeypoint(centerX - 0.03, 0.12, 0.95);
  keypoints[KP.RIGHT_EAR] = makeKeypoint(centerX + 0.03, 0.12, 0.95);
  keypoints[KP.LEFT_SHOULDER] = makeKeypoint(centerX - 0.08, 0.25, 0.98);
  keypoints[KP.RIGHT_SHOULDER] = makeKeypoint(centerX + 0.08, 0.25, 0.98);
  keypoints[KP.LEFT_ELBOW] = makeKeypoint(centerX - 0.12, 0.4, 0.96);
  keypoints[KP.RIGHT_ELBOW] = makeKeypoint(centerX + 0.12, 0.4, 0.96);
  keypoints[KP.LEFT_WRIST] = makeKeypoint(centerX - 0.14, 0.55, 0.95);
  keypoints[KP.RIGHT_WRIST] = makeKeypoint(centerX + 0.14, 0.55, 0.95);
  keypoints[KP.LEFT_HIP] = makeKeypoint(centerX - 0.06, 0.55, 0.99);
  keypoints[KP.RIGHT_HIP] = makeKeypoint(centerX + 0.06, 0.55, 0.99);
  keypoints[KP.LEFT_KNEE] = makeKeypoint(centerX - 0.055, 0.72, 0.98);
  keypoints[KP.RIGHT_KNEE] = makeKeypoint(centerX + 0.055, 0.72, 0.98);
  keypoints[KP.LEFT_ANKLE] = makeKeypoint(centerX - 0.05, leftAnkleY, 0.99);
  keypoints[KP.RIGHT_ANKLE] = makeKeypoint(centerX + 0.05, rightAnkleY, 0.99);

  return {
    frameIndex,
    timestampMs,
    keypoints,
    avgConfidence: 0.97,
    personCount,
  };
}

function makeValidFlightFrames(): JumpLandmarkFrame[] {
  const timestamps = [
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
  ];
  const anklePairs: Array<[number, number]> = [
    [0.86, 0.862],
    [0.859, 0.861],
    [0.861, 0.86],
    [0.858, 0.859],
    [0.86, 0.861],
    [0.859, 0.86],
    [0.86, 0.861],
    [0.859, 0.86],
    [0.82, 0.821],
    [0.802, 0.804],
    [0.79, 0.792],
    [0.792, 0.793],
    [0.804, 0.806],
    [0.95, 0.951],
    [0.95, 0.95],
    [0.951, 0.952],
  ];

  return timestamps.map((timestamp, index) =>
    makeFrame(index, timestamp, anklePairs[index][0], anklePairs[index][1]),
  );
}

test('heightFromFlightTime converts airtime to centimeters', () => {
  assert.ok(Math.abs(heightFromFlightTime(500) - 30.645625) < 0.001);
});

test('fixture timeline yields deterministic takeoff and landing timestamps', () => {
  const result = analyzeJumpLandmarks(makeValidFlightFrames(), {
    videoDurationMs: 1600,
    sampleFps: 60,
    videoFps: 60,
  });

  assert.equal(result.invalidReason, undefined);
  assert.equal(result.takeoffMs, 800);
  assert.equal(result.landingMs, 1300);
  assert.equal(result.flightMs, 500);
  assert.ok(result.heightCm !== null);
  assert.ok(Math.abs((result.heightCm ?? 0) - heightFromFlightTime(500)) < 0.001);
});

test('median plus EMA smoothing tolerates small ankle jitter', () => {
  const frames = makeValidFlightFrames().map((frame, index) => {
    if (index === 9) {
      return makeFrame(frame.frameIndex, frame.timestampMs, 0.81, 0.816);
    }
    if (index === 13) {
      return makeFrame(frame.frameIndex, frame.timestampMs, 0.948, 0.952);
    }
    return frame;
  });

  const result = analyzeJumpLandmarks(frames, {
    videoDurationMs: 1600,
    sampleFps: 60,
    videoFps: 60,
  });

  assert.equal(result.invalidReason, undefined);
  assert.equal(result.takeoffMs, 800);
  assert.equal(result.landingMs, 1300);
});

test('unstable calibration is rejected before event detection', () => {
  const frames = makeValidFlightFrames().map((frame, index) => {
    if (frame.timestampMs <= 700) {
      const ankleY = index % 2 === 0 ? 0.8 : 0.93;
      return makeFrame(frame.frameIndex, frame.timestampMs, ankleY, ankleY + 0.002);
    }
    return frame;
  });

  const result = analyzeJumpLandmarks(frames, {
    videoDurationMs: 1600,
    sampleFps: 60,
    videoFps: 60,
  });

  assert.equal(result.invalidReason, 'NO_STABLE_CALIBRATION');
});

test('missing landing is rejected after takeoff', () => {
  const frames = makeValidFlightFrames().map((frame) =>
    frame.timestampMs >= 1300
      ? makeFrame(frame.frameIndex, frame.timestampMs, 0.79, 0.791)
      : frame,
  );

  const result = analyzeJumpLandmarks(frames, {
    videoDurationMs: 1600,
    sampleFps: 60,
    videoFps: 60,
  });

  assert.equal(result.takeoffMs, 800);
  assert.equal(result.invalidReason, 'NO_LANDING');
});

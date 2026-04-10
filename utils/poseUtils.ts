import type { Keypoint } from '@tensorflow-models/pose-detection';

/** MoveNet 17-keypoint indices */
export const KP = {
  NOSE: 0,
  LEFT_EYE: 1,
  RIGHT_EYE: 2,
  LEFT_EAR: 3,
  RIGHT_EAR: 4,
  LEFT_SHOULDER: 5,
  RIGHT_SHOULDER: 6,
  LEFT_ELBOW: 7,
  RIGHT_ELBOW: 8,
  LEFT_WRIST: 9,
  RIGHT_WRIST: 10,
  LEFT_HIP: 11,
  RIGHT_HIP: 12,
  LEFT_KNEE: 13,
  RIGHT_KNEE: 14,
  LEFT_ANKLE: 15,
  RIGHT_ANKLE: 16,
} as const;

/** Pairs of keypoint indices defining skeleton line segments */
export const SKELETON_CONNECTIONS: [number, number][] = [
  [KP.NOSE, KP.LEFT_SHOULDER],
  [KP.NOSE, KP.RIGHT_SHOULDER],
  [KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER],
  [KP.LEFT_SHOULDER, KP.LEFT_ELBOW],
  [KP.LEFT_ELBOW, KP.LEFT_WRIST],
  [KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW],
  [KP.RIGHT_ELBOW, KP.RIGHT_WRIST],
  [KP.LEFT_SHOULDER, KP.LEFT_HIP],
  [KP.RIGHT_SHOULDER, KP.RIGHT_HIP],
  [KP.LEFT_HIP, KP.RIGHT_HIP],
  [KP.LEFT_HIP, KP.LEFT_KNEE],
  [KP.LEFT_KNEE, KP.LEFT_ANKLE],
  [KP.RIGHT_HIP, KP.RIGHT_KNEE],
  [KP.RIGHT_KNEE, KP.RIGHT_ANKLE],
];

export const MIN_CONFIDENCE = 0.3;

/** Returns the best available hip position (midpoint of both, or whichever is visible). */
export function getHipMidpoint(
  kps: Keypoint[],
  minConfidence: number = MIN_CONFIDENCE,
): { x: number; y: number } | null {
  const lh = kps[KP.LEFT_HIP];
  const rh = kps[KP.RIGHT_HIP];
  const lhOk = lh && (lh.score ?? 0) >= minConfidence;
  const rhOk = rh && (rh.score ?? 0) >= minConfidence;
  if (lhOk && rhOk) return { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 };
  if (lhOk) return { x: lh.x, y: lh.y };
  if (rhOk) return { x: rh.x, y: rh.y };
  return null;
}

/** Returns the best available ankle Y (average of both, or whichever is visible). */
export function getAnkleMidpointY(
  kps: Keypoint[],
  minConfidence: number = MIN_CONFIDENCE,
): number | null {
  const la = kps[KP.LEFT_ANKLE];
  const ra = kps[KP.RIGHT_ANKLE];
  const laOk = la && (la.score ?? 0) >= minConfidence;
  const raOk = ra && (ra.score ?? 0) >= minConfidence;
  if (laOk && raOk) return (la.y + ra.y) / 2;
  if (laOk) return la.y;
  if (raOk) return ra.y;
  return null;
}

/**
 * Returns an estimated head Y for calibration:
 * prefer face keypoints, fallback to shoulder/hip-based estimate if face is occluded.
 */
export function getHeadY(
  kps: Keypoint[],
  minConfidence: number = MIN_CONFIDENCE,
): number | null {
  const faceIndices = [KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE, KP.LEFT_EAR, KP.RIGHT_EAR] as const;
  let bestFaceY: number | null = null;
  for (const idx of faceIndices) {
    const kp = kps[idx];
    if (!kp || (kp.score ?? 0) < minConfidence) continue;
    bestFaceY = bestFaceY === null ? kp.y : Math.min(bestFaceY, kp.y);
  }
  if (bestFaceY !== null) return bestFaceY;

  const ls = kps[KP.LEFT_SHOULDER];
  const rs = kps[KP.RIGHT_SHOULDER];
  const lsOk = ls && (ls.score ?? 0) >= minConfidence;
  const rsOk = rs && (rs.score ?? 0) >= minConfidence;
  if (!lsOk && !rsOk) return null;
  const shoulderY = lsOk && rsOk ? (ls.y + rs.y) / 2 : (lsOk ? ls.y : rs.y);

  // Estimate head above shoulders using torso length when hips are visible.
  const hip = getHipMidpoint(kps, minConfidence);
  if (hip) {
    const torso = Math.max(12, hip.y - shoulderY);
    return shoulderY - torso * 0.65;
  }
  return shoulderY - 20;
}

/**
 * Returns body height in pixels (estimated head to best ankle) in model-input coordinates.
 * Used to compute pixels-per-meter during calibration.
 */
export function getBodyHeightPixels(
  kps: Keypoint[],
  minConfidence: number = MIN_CONFIDENCE,
): number | null {
  const headY = getHeadY(kps, minConfidence);
  if (headY === null) return null;
  const ankleY = getAnkleMidpointY(kps, minConfidence);
  if (ankleY === null) return null;
  return ankleY - headY;
}

import Svg, { Circle, Line } from 'react-native-svg';

import { KP, SKELETON_CONNECTIONS } from '@/utils/poseUtils';
import type { JumpAnalysisResult, JumpKeypoint, JumpLandmarkFrame } from '@/utils/jumpCalc';

interface JumpDebugOverlayProps {
  frame: JumpLandmarkFrame | null;
  analysis: JumpAnalysisResult | null;
}

const MIN_OVERLAY_SCORE = 0.2;

function isTracked(point: JumpKeypoint | undefined): point is JumpKeypoint {
  return !!point && Number.isFinite(point.x) && Number.isFinite(point.y) && point.score >= MIN_OVERLAY_SCORE;
}

function pointColor(index: number): string {
  if (index === KP.LEFT_FOOT_INDEX || index === KP.RIGHT_FOOT_INDEX) {
    return 'rgba(34,197,94,0.98)';
  }
  if (index === KP.LEFT_HEEL || index === KP.RIGHT_HEEL) {
    return 'rgba(56,189,248,0.98)';
  }
  if (index === KP.LEFT_ANKLE || index === KP.RIGHT_ANKLE) {
    return 'rgba(245,158,11,0.98)';
  }
  return 'rgba(248,250,252,0.92)';
}

export function JumpDebugOverlay({ frame, analysis }: JumpDebugOverlayProps) {
  if (!frame || !analysis) return null;

  const toeBaselineY =
    ((analysis.debug.baseline.leftToeY + analysis.debug.baseline.rightToeY) / 2) * 100;

  return (
    <Svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
      <Line
        x1="0"
        x2="100"
        y1={String(toeBaselineY)}
        y2={String(toeBaselineY)}
        stroke="rgba(59,130,246,0.85)"
        strokeWidth="0.7"
        strokeDasharray="3 2"
      />

      {SKELETON_CONNECTIONS.map(([start, end]) => {
        const a = frame.keypoints[start];
        const b = frame.keypoints[end];
        if (!isTracked(a) || !isTracked(b)) return null;
        return (
          <Line
            key={`${start}-${end}`}
            x1={String(a.x * 100)}
            y1={String(a.y * 100)}
            x2={String(b.x * 100)}
            y2={String(b.y * 100)}
            stroke="rgba(148,163,184,0.78)"
            strokeWidth="0.65"
          />
        );
      })}

      {frame.keypoints.map((point, index) => {
        if (!isTracked(point)) return null;
        const radius =
          index === KP.LEFT_FOOT_INDEX || index === KP.RIGHT_FOOT_INDEX
            ? 1.5
            : index === KP.LEFT_HEEL || index === KP.RIGHT_HEEL
              ? 1.3
              : 0.95;

        return (
          <Circle
            key={`kp-${index}`}
            cx={String(point.x * 100)}
            cy={String(point.y * 100)}
            r={String(radius)}
            fill={pointColor(index)}
          />
        );
      })}
    </Svg>
  );
}

import Svg, { Circle, Line } from 'react-native-svg';

import { SKELETON_CONNECTIONS } from '@/utils/poseUtils';
import type { JumpAnalysisResult, JumpLandmarkFrame } from '@/utils/jumpCalc';

interface JumpDebugOverlayProps {
  frame: JumpLandmarkFrame | null;
  analysis: JumpAnalysisResult | null;
}

const MIN_DRAW_SCORE = 0.18;

export function JumpDebugOverlay({ frame, analysis }: JumpDebugOverlayProps) {
  if (!frame || !analysis) return null;

  const baselineY =
    ((analysis.debug.baseline.leftAnkleY + analysis.debug.baseline.rightAnkleY) / 2) * 100;
  const hipBaselineY = analysis.debug.baseline.hipY * 100;

  return (
    <Svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
      <Line
        x1="0"
        x2="100"
        y1={String(baselineY)}
        y2={String(baselineY)}
        stroke="rgba(34,197,94,0.9)"
        strokeWidth="0.7"
        strokeDasharray="2 2"
      />
      <Line
        x1="0"
        x2="100"
        y1={String(hipBaselineY)}
        y2={String(hipBaselineY)}
        stroke="rgba(59,130,246,0.85)"
        strokeWidth="0.55"
        strokeDasharray="2 2"
      />

      {SKELETON_CONNECTIONS.map(([fromIndex, toIndex]) => {
        const from = frame.keypoints[fromIndex];
        const to = frame.keypoints[toIndex];
        if (!from || !to) return null;
        if (from.score < MIN_DRAW_SCORE || to.score < MIN_DRAW_SCORE) return null;

        return (
          <Line
            key={`${fromIndex}-${toIndex}`}
            x1={String(from.x * 100)}
            y1={String(from.y * 100)}
            x2={String(to.x * 100)}
            y2={String(to.y * 100)}
            stroke="rgba(255,255,255,0.85)"
            strokeWidth="0.55"
          />
        );
      })}

      {frame.keypoints.map((keypoint, index) => {
        if (keypoint.score < MIN_DRAW_SCORE) return null;
        return (
          <Circle
            key={`${index}-${keypoint.name ?? 'kp'}`}
            cx={String(keypoint.x * 100)}
            cy={String(keypoint.y * 100)}
            r="1.15"
            fill="rgba(251,191,36,0.95)"
          />
        );
      })}
    </Svg>
  );
}

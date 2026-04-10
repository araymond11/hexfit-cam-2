import { requireOptionalNativeModule } from 'expo-modules-core';

import type {
  JumpClip,
  JumpVideoAnalysisOptions,
  JumpVideoNativeResult,
} from '../../utils/jumpCalc';

interface JumpVideoAnalysisNativeModule {
  analyze(clip: JumpClip, options?: JumpVideoAnalysisOptions): Promise<JumpVideoNativeResult>;
}

const nativeModule =
  requireOptionalNativeModule<JumpVideoAnalysisNativeModule>('JumpVideoAnalysis');

export async function analyzeRecordedJumpVideo(
  clip: JumpClip,
  options: JumpVideoAnalysisOptions = {},
): Promise<JumpVideoNativeResult> {
  if (!nativeModule) {
    throw new Error(
      'JumpVideoAnalysis native module is unavailable. Rebuild the iOS development client after pod install to use recorded analysis.',
    );
  }

  return nativeModule.analyze(clip, options);
}

export function isJumpVideoAnalysisAvailable(): boolean {
  return nativeModule != null;
}

import { useIsFocused } from '@react-navigation/native';
import { useEvent } from 'expo';
import * as ImagePicker from 'expo-image-picker';
import { VideoView, useVideoPlayer } from 'expo-video';
import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  LayoutChangeEvent,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  Camera,
  type CameraCaptureError,
  type CameraDevice,
  type CameraDeviceFormat,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';

import { JumpDebugOverlay } from '@/components/jump-debug-overlay';
import {
  analyzeRecordedJumpVideo,
  isJumpVideoAnalysisAvailable,
} from '@/modules/jump-video-analysis';
import {
  analyzeJumpLandmarks,
  explainJumpAnalysis,
  findFrameAtTime,
  suggestJumpAttemptWindow,
} from '@/utils/jumpAnalysis';
import {
  heightFromFlightTime,
  type JumpClip,
  type JumpAnalysisResult,
  type JumpLandmarkFrame,
  type JumpPhaseSample,
  type JumpVideoNativeResult,
} from '@/utils/jumpCalc';

type JumpStage =
  | 'SETUP'
  | 'COUNTDOWN'
  | 'RECORDING'
  | 'REVIEW'
  | 'ANALYZING'
  | 'RESULT_DEBUG';

const Spacing = {
  one: 4,
  two: 8,
  three: 16,
  four: 24,
};

const COUNTDOWN_SEQUENCE = ['Hold still', '3', '2', '1'] as const;
const HOLD_STILL_MS = 1000;
const RECORDING_HARD_CAP_MS = 10000;
const MIN_ANALYSIS_WINDOW_MS = 1600;
const MAX_STANDARD_ANALYSIS_WINDOW_MS = 12000;
const MAX_SLOW_MO_PLAYBACK_WINDOW_MS = 30000;

interface ManualJumpEventSelection {
  frameIndex: number;
  playbackMs: number;
  physicalMs: number;
}

const SETUP_COPY = [
  'Import an iPhone 240 fps slow-motion video when possible.',
  'Keep the full body and face visible from head to feet for the whole jump.',
  'Use a fixed camera. One athlete only. Stay still briefly before takeoff.',
  'Front or slight 3/4 view is acceptable as long as both feet stay visible.',
];

function normalizedResolution(format: CameraDeviceFormat): { longSide: number; shortSide: number } {
  const longSide = Math.max(format.videoWidth, format.videoHeight);
  const shortSide = Math.min(format.videoWidth, format.videoHeight);
  return { longSide, shortSide };
}

function selectFpsFirstFormat(device: CameraDevice | undefined): CameraDeviceFormat | undefined {
  if (!device) return undefined;
  const formats = device.formats;
  if (formats.length === 0) return undefined;

  const sortByTarget = (
    input: CameraDeviceFormat[],
    targetLong: number,
    targetShort: number,
  ) => {
    return [...input].sort((a, b) => {
      const aRes = normalizedResolution(a);
      const bRes = normalizedResolution(b);
      const aDelta = Math.abs(aRes.longSide - targetLong) + Math.abs(aRes.shortSide - targetShort);
      const bDelta = Math.abs(bRes.longSide - targetLong) + Math.abs(bRes.shortSide - targetShort);
      if (aDelta !== bDelta) return aDelta - bDelta;
      if (a.maxFps !== b.maxFps) return b.maxFps - a.maxFps;
      return aRes.longSide * aRes.shortSide - bRes.longSide * bRes.shortSide;
    });
  };

  // Prefer 240 fps slow-motion formats first (iPhone back camera)
  const slowMo = formats.filter((format) => format.maxFps >= 240);
  const slowMo720 = slowMo.filter((format) => {
    const res = normalizedResolution(format);
    return res.longSide >= 1280 && res.shortSide >= 720;
  });
  if (slowMo720.length > 0) {
    return sortByTarget(slowMo720, 1280, 720)[0];
  }
  if (slowMo.length > 0) {
    return sortByTarget(slowMo, 1280, 720)[0];
  }

  // Fallback: 120 fps formats
  const highFps = formats.filter((format) => format.maxFps >= 120);
  const highFps720 = highFps.filter((format) => {
    const res = normalizedResolution(format);
    return res.longSide >= 1280 && res.shortSide >= 720;
  });
  if (highFps720.length > 0) {
    return sortByTarget(highFps720, 1280, 720)[0];
  }
  if (highFps.length > 0) {
    return sortByTarget(highFps, 1280, 720)[0];
  }

  // Fallback: 60 fps formats
  const sixtyFps = formats.filter((format) => format.maxFps >= 60);
  const atLeast1080 = sixtyFps.filter((format) => {
    const res = normalizedResolution(format);
    return res.longSide >= 1920 && res.shortSide >= 1080;
  });
  if (atLeast1080.length > 0) {
    return sortByTarget(atLeast1080, 1920, 1080)[0];
  }

  const atLeast720 = sixtyFps.filter((format) => {
    const res = normalizedResolution(format);
    return res.longSide >= 1280 && res.shortSide >= 720;
  });
  if (atLeast720.length > 0) {
    return sortByTarget(atLeast720, 1280, 720)[0];
  }

  const atLeast30 = formats
    .filter((format) => format.maxFps >= 30)
    .sort((a, b) => {
      if (a.maxFps !== b.maxFps) return b.maxFps - a.maxFps;
      const aRes = normalizedResolution(a);
      const bRes = normalizedResolution(b);
      return bRes.longSide * bRes.shortSide - aRes.longSide * aRes.shortSide;
    });
  if (atLeast30.length > 0) return atLeast30[0];

  return [...formats].sort((a, b) => b.maxFps - a.maxFps)[0];
}

function normalizeMediaUri(input: string): string {
  if (!input) return input;
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(input)) {
    return input;
  }
  return input.startsWith('/') ? `file://${input}` : input;
}

function formatMs(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return `${value.toFixed(1)} ms`;
}

function formatDetailedTime(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—';
  const roundedMs = Math.abs(value - Math.round(value)) < 0.05 ? `${Math.round(value)}` : value.toFixed(1);
  return `${roundedMs}ms (${(value / 1000).toFixed(3)}s)`;
}

function formatFramePrecision(fps: number | null | undefined): string {
  if (!fps || !Number.isFinite(fps) || fps <= 0) return '—';
  return `±${(1000 / fps).toFixed(2)} ms`;
}

function formatSignedNumber(value: number | null, digits = 1): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

function formatSignedMilliseconds(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return `${formatSignedNumber(value, 1)} ms`;
}

function formatSignedFrames(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return `${formatSignedNumber(value, 2)} frames`;
}

function formatSignedCentimeters(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return `${formatSignedNumber(value, 2)} cm`;
}

function formatTimingMode(value: string | null | undefined): string {
  switch (value) {
    case 'playback_is_physical':
      return 'Playback is physical';
    case 'segment_mapped_slow_motion':
      return 'Segment-mapped slow motion';
    case 'global_scaled_slow_motion':
      return 'Globally scaled slow motion';
    case 'timing_ambiguous':
      return 'Timing ambiguous';
    default:
      return '—';
  }
}

function formatTimebaseSource(value: string | null | undefined): string {
  if (!value) return '—';
  return value
    .split('_')
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(' ');
}

function formatPercent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

function manualSelectionFromFrame(frame: JumpLandmarkFrame): ManualJumpEventSelection {
  return {
    frameIndex: frame.frameIndex,
    playbackMs: frame.timestampMs,
    physicalMs: frame.captureTimestampMs ?? frame.timestampMs,
  };
}

function clampAnalysisStart(valueMs: number, endMs: number, durationMs: number): number {
  return Math.max(0, Math.min(valueMs, Math.max(0, Math.min(endMs - MIN_ANALYSIS_WINDOW_MS, durationMs))));
}

function clampAnalysisEnd(valueMs: number, startMs: number, durationMs: number): number {
  return Math.min(durationMs, Math.max(valueMs, Math.min(durationMs, startMs + MIN_ANALYSIS_WINDOW_MS)));
}

function phaseColor(phase: JumpPhaseSample['phase'] | null): string {
  switch (phase) {
    case 'GROUND_CONTACT':
      return '#22c55e';
    case 'AIRBORNE':
      return '#3b82f6';
    case 'UNCERTAIN':
      return '#f59e0b';
    default:
      return '#ffffff';
  }
}

function stageLabel(stage: JumpStage): string {
  switch (stage) {
    case 'SETUP':
      return 'Setup';
    case 'COUNTDOWN':
      return 'Countdown';
    case 'RECORDING':
      return 'Recording';
    case 'REVIEW':
      return 'Review';
    case 'ANALYZING':
      return 'Analyzing';
    case 'RESULT_DEBUG':
      return 'Result + Debug';
  }
}

function findPhaseSampleAtTime(
  samples: JumpPhaseSample[],
  timestampMs: number,
): JumpPhaseSample | null {
  if (samples.length === 0) return null;
  let best = samples[0];
  let bestDelta = Math.abs(best.timestampMs - timestampMs);
  for (const sample of samples) {
    const delta = Math.abs(sample.timestampMs - timestampMs);
    if (delta < bestDelta) {
      best = sample;
      bestDelta = delta;
    }
  }
  return best;
}

function summaryColor(result: JumpAnalysisResult | null): string {
  if (!result) return '#ffffff';
  return result.invalidReason ? '#f97316' : '#22c55e';
}

function formatQualityFlag(flag: string): string {
  switch (flag) {
    case 'INSUFFICIENT_FULL_BODY_VISIBILITY':
      return 'Full body or face not visible enough';
    case 'INSUFFICIENT_FEET_VISIBILITY':
      return 'Feet tracking insufficient';
    case 'UNSTABLE_CALIBRATION':
      return 'Calibration unstable';
    case 'LOW_CONFIDENCE':
      return 'Low landmark confidence';
    case 'UNCERTAIN_TRACKING':
      return 'Tracking uncertain';
    case 'PARTIAL_FEET_VISIBILITY':
      return 'Partial feet visibility';
    case 'PARTIAL_FULL_BODY_VISIBILITY':
      return 'Partial full-body visibility';
    case 'LOW_SAMPLE_FPS':
      return 'Low sample FPS';
    case 'CALIBRATION_NOISE':
      return 'Calibration noise';
    case 'WEAK_TOE_CONTACT_SIGNAL':
      return 'Weak toe-contact signal';
    case 'EXCESS_HORIZONTAL_MOTION':
      return 'Excess horizontal motion';
    case 'AIRTIME_OUT_OF_RANGE':
      return 'Airtime out of range';
    case 'TIMING_AMBIGUOUS':
      return 'Timing ambiguous';
    case 'EVENT_ORDER_INVALID':
      return 'Event order invalid';
    case 'MULTI_PERSON_INPUT':
      return 'Multiple people detected';
    case 'NO_ANALYZED_FRAMES':
      return 'No analyzed frames';
    default:
      return flag.replaceAll('_', ' ').toLowerCase();
  }
}

function statusCopy(
  stage: JumpStage,
  countdownStep: number,
  recordingElapsedMs: number,
  error: string | null,
): string {
  if (error) return error;
  if (stage === 'COUNTDOWN') {
    return COUNTDOWN_SEQUENCE[countdownStep] ?? 'Recording...';
  }
  if (stage === 'RECORDING') {
    return `Recording ${(recordingElapsedMs / 1000).toFixed(1)} s / 10.0 s`;
  }
  if (stage === 'ANALYZING') {
    return 'Decoding frames, selecting the jump attempt, running ML Kit accurate pose detection, and timing toe-off/touch-down.';
  }
  if (stage === 'REVIEW') {
    return 'Review the clip. Range markers are optional for debug; long clips can auto-focus on the jump attempt during analysis.';
  }
  if (stage === 'RESULT_DEBUG') {
    return 'Review the result. Pose overlay is optional if you need to inspect landmarks.';
  }
  return 'Import a 240 fps iPhone clip or record a reference clip, then run pose analysis.';
}

interface ReviewPlayerProps {
  clip: JumpClip;
  nativeResult: JumpVideoNativeResult | null;
  analysis: JumpAnalysisResult | null;
  manualTakeoff: ManualJumpEventSelection | null;
  manualLanding: ManualJumpEventSelection | null;
  analysisStartMs: number;
  analysisEndMs: number;
  showDebugOverlay: boolean;
  debugMode: boolean;
  onToggleDebugOverlay: () => void;
  onSetAnalysisStart: (timeMs: number) => void;
  onSetAnalysisEnd: (timeMs: number) => void;
  onResetAnalysisRange: () => void;
  onSetManualTakeoff: (frame: JumpLandmarkFrame) => void;
  onSetManualLanding: (frame: JumpLandmarkFrame) => void;
  onClearManualReview: () => void;
}

function ReviewPlayer({
  clip,
  nativeResult,
  analysis,
  manualTakeoff,
  manualLanding,
  analysisStartMs,
  analysisEndMs,
  showDebugOverlay,
  debugMode,
  onToggleDebugOverlay,
  onSetAnalysisStart,
  onSetAnalysisEnd,
  onResetAnalysisRange,
  onSetManualTakeoff,
  onSetManualLanding,
  onClearManualReview,
}: ReviewPlayerProps) {
  const [timelineWidth, setTimelineWidth] = useState(0);
  const [shellSize, setShellSize] = useState({ width: 0, height: 0 });
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const player = useVideoPlayer(clip.uri, (instance) => {
    instance.loop = false;
    instance.muted = true;
    instance.timeUpdateEventInterval = 1 / 30;
  });
  const { currentTime } = useEvent(player, 'timeUpdate', {
    currentTime: 0,
    currentLiveTimestamp: null,
    currentOffsetFromLive: null,
    bufferedPosition: 0,
  });
  const { isPlaying } = useEvent(player, 'playingChange', { isPlaying: false });
  const currentMs = currentTime * 1000;
  const durationMs = Math.max(nativeResult?.videoDurationMs ?? clip.durationMs, 1);
  const aspectRatio = clip.width > 0 && clip.height > 0 ? clip.width / clip.height : 9 / 16;
  const maxPlayerHeight = Math.min(Math.max(windowHeight * 0.5, 320), 520);
  const estimatedPlayerWidth = Math.max(windowWidth - Spacing.three * 2, 240);
  const playerHeight = Math.min(estimatedPlayerWidth / aspectRatio, maxPlayerHeight);
  const playbackSampleFps = Math.max(
    nativeResult?.playbackSampleFps ?? nativeResult?.sampleFps ?? clip.fps ?? 30,
    1,
  );
  const currentFrame = useMemo(
    () => (nativeResult ? findFrameAtTime(nativeResult.frames, currentMs) : null),
    [nativeResult, currentMs],
  );
  const currentPhase = useMemo(
    () => (analysis ? findPhaseSampleAtTime(analysis.phaseTimeline, currentMs) : null),
    [analysis, currentMs],
  );
  const videoFrame = useMemo(() => {
    const containerWidth = shellSize.width || estimatedPlayerWidth;
    const containerHeight = shellSize.height || playerHeight;
    const containerRatio = containerWidth / Math.max(containerHeight, 1);

    if (containerRatio > aspectRatio) {
      const width = containerHeight * aspectRatio;
      return {
        left: (containerWidth - width) / 2,
        top: 0,
        width,
        height: containerHeight,
      };
    }

    const height = containerWidth / aspectRatio;
    return {
      left: 0,
      top: (containerHeight - height) / 2,
      width: containerWidth,
      height,
    };
  }, [aspectRatio, estimatedPlayerWidth, playerHeight, shellSize.height, shellSize.width]);

  const seekToRatio = useCallback(
    (ratio: number) => {
      player.pause();
      player.currentTime = Math.max(0, Math.min(1, ratio)) * (durationMs / 1000);
    },
    [durationMs, player],
  );
  const seekToMs = useCallback(
    (timeMs: number | null | undefined) => {
      if (timeMs === null || timeMs === undefined || !Number.isFinite(timeMs)) return;
      seekToRatio(timeMs / durationMs);
    },
    [durationMs, seekToRatio],
  );

  const onTimelinePress = useCallback(
    (event: { nativeEvent: { locationX: number } }) => {
      if (!timelineWidth) return;
      seekToRatio(event.nativeEvent.locationX / timelineWidth);
    },
    [seekToRatio, timelineWidth],
  );

  const onTimelineLayout = useCallback((event: LayoutChangeEvent) => {
    setTimelineWidth(event.nativeEvent.layout.width);
  }, []);
  const onPlayerShellLayout = useCallback((event: LayoutChangeEvent) => {
    setShellSize({
      width: event.nativeEvent.layout.width,
      height: event.nativeEvent.layout.height,
    });
  }, []);
  const stepFrame = useCallback(
    (direction: -1 | 1) => {
      player.pause();
      const nextTime = Math.max(0, currentTime + direction / playbackSampleFps);
      player.currentTime = nextTime;
    },
    [currentTime, playbackSampleFps, player],
  );

  const takeoffRatio =
    analysis?.takeoffMs !== null && analysis?.takeoffMs !== undefined
      ? analysis.takeoffMs / durationMs
      : null;
  const landingRatio =
    analysis?.landingMs !== null && analysis?.landingMs !== undefined
      ? analysis.landingMs / durationMs
      : null;
  const manualTakeoffRatio =
    manualTakeoff && Number.isFinite(manualTakeoff.playbackMs)
      ? manualTakeoff.playbackMs / durationMs
      : null;
  const manualLandingRatio =
    manualLanding && Number.isFinite(manualLanding.playbackMs)
      ? manualLanding.playbackMs / durationMs
      : null;
  const rangeStartRatio = Math.max(0, Math.min(1, analysisStartMs / durationMs));
  const rangeEndRatio = Math.max(0, Math.min(1, analysisEndMs / durationMs));
  const rangeWidthRatio = Math.max(0, rangeEndRatio - rangeStartRatio);
  const isDefaultRange =
    Math.abs(analysisStartMs) < 0.5 && Math.abs(analysisEndMs - durationMs) < 0.5;
  const hasAlgoTakeoff =
    analysis?.takeoffMs !== null && analysis?.takeoffMs !== undefined;
  const hasAlgoLanding =
    analysis?.landingMs !== null && analysis?.landingMs !== undefined;

  return (
    <View style={styles.playerSection}>
      <View style={[styles.playerShell, { height: playerHeight }]} onLayout={onPlayerShellLayout}>
        <VideoView
          player={player}
          style={StyleSheet.absoluteFill}
          contentFit="contain"
          nativeControls={false}
        />
        <View
          style={[
            styles.videoContentFrame,
            {
              left: videoFrame.left,
              top: videoFrame.top,
              width: videoFrame.width,
              height: videoFrame.height,
            },
          ]}>
          <View style={StyleSheet.absoluteFill}>
            {nativeResult && analysis && showDebugOverlay && (
              <JumpDebugOverlay frame={currentFrame} analysis={analysis} />
            )}
          </View>
        </View>
        <View style={styles.playerMetaBadge}>
          <Text style={styles.playerMetaText}>{formatMs(currentMs)}</Text>
          <Text style={[styles.playerMetaText, { color: phaseColor(currentPhase?.phase ?? null) }]}>
            {currentPhase?.phase ?? 'RAW_CLIP'}
          </Text>
        </View>
      </View>

      <Pressable style={styles.timeline} onLayout={onTimelineLayout} onPress={onTimelinePress}>
        <View
          style={[
            styles.timelineRange,
            {
              left: `${rangeStartRatio * 100}%`,
              width: `${Math.max(rangeWidthRatio, 0.004) * 100}%`,
            },
          ]}
        />
        <View style={[styles.timelineProgress, { width: `${(currentMs / durationMs) * 100}%` }]} />
        <View style={[styles.timelineMarker, styles.rangeStartMarker, { left: `${rangeStartRatio * 100}%` }]} />
        <View style={[styles.timelineMarker, styles.rangeEndMarker, { left: `${rangeEndRatio * 100}%` }]} />
        {takeoffRatio !== null && (
          <View style={[styles.timelineMarker, styles.takeoffMarker, { left: `${takeoffRatio * 100}%` }]} />
        )}
        {landingRatio !== null && (
          <View style={[styles.timelineMarker, styles.landingMarker, { left: `${landingRatio * 100}%` }]} />
        )}
        {manualTakeoffRatio !== null && (
          <View
            style={[styles.timelineMarker, styles.manualTakeoffMarker, { left: `${manualTakeoffRatio * 100}%` }]}
          />
        )}
        {manualLandingRatio !== null && (
          <View
            style={[styles.timelineMarker, styles.manualLandingMarker, { left: `${manualLandingRatio * 100}%` }]}
          />
        )}
      </Pressable>

      <View style={styles.playerControls}>
        <TouchableOpacity style={styles.playerControlButton} onPress={() => stepFrame(-1)}>
          <Text style={styles.playerControlText}>-1 frame</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.playerControlButton, styles.playerControlPrimary]}
          onPress={() => {
            if (isPlaying) {
              player.pause();
            } else {
              player.play();
            }
          }}>
          <Text style={styles.playerControlText}>{isPlaying ? 'Pause' : 'Play'}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.playerControlButton} onPress={() => stepFrame(1)}>
          <Text style={styles.playerControlText}>+1 frame</Text>
        </TouchableOpacity>
      </View>

      {debugMode && (
        <View style={styles.analysisRangeCard}>
          <Text style={styles.referenceTitle}>Analysis Range</Text>
          <Text style={styles.referenceText}>
            The highlighted range is the current analysis window. You can still set start/end
            manually for debug, but long clips can be narrowed automatically during analysis.
          </Text>
          {clip.assetId && clip.fps >= 120 && (
            <Text style={styles.referenceText}>
              Slow-motion playback ranges can be longer here because the physical jump still lasts far
              less time than the clip playback.
            </Text>
          )}
          <View style={styles.analysisRangeStats}>
            <Text style={styles.analysisRangeStat}>Start: {formatDetailedTime(analysisStartMs)}</Text>
            <Text style={styles.analysisRangeStat}>End: {formatDetailedTime(analysisEndMs)}</Text>
            <Text style={styles.analysisRangeStat}>
              Window: {formatDetailedTime(Math.max(analysisEndMs - analysisStartMs, 0))}
            </Text>
          </View>
          <View style={styles.playerControls}>
            <TouchableOpacity
              style={[styles.playerControlButton, styles.rangeActionButton]}
              onPress={() => onSetAnalysisStart(currentMs)}>
              <Text style={styles.playerControlText}>Set Start</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.playerControlButton, styles.rangeActionButton]}
              onPress={() => onSetAnalysisEnd(currentMs)}>
              <Text style={styles.playerControlText}>Set End</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.playerControlButton, isDefaultRange && styles.btnDisabled]}
              onPress={onResetAnalysisRange}
              disabled={isDefaultRange}>
              <Text style={styles.playerControlText}>Reset Range</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {debugMode && nativeResult && (
        <View style={styles.eventReviewCard}>
          <Text style={styles.referenceTitle}>Event Review</Text>
          <Text style={styles.referenceText}>
            Use frame-by-frame stepping, jump to the algorithm markers, and save manual takeoff and
            landing frames from the current frame.
          </Text>
          <Text style={styles.analysisRangeStat}>
            Current frame: {currentFrame ? `#${currentFrame.frameIndex}` : '—'} at {formatDetailedTime(currentFrame?.timestampMs ?? null)}
          </Text>
          {currentFrame && (
            <Text style={styles.analysisRangeStat}>
              Current physical time: {formatDetailedTime(currentFrame.captureTimestampMs ?? currentFrame.timestampMs)}
            </Text>
          )}
          {manualTakeoff && (
            <Text style={styles.analysisRangeStat}>
              Manual takeoff marker: frame #{manualTakeoff.frameIndex} at {formatDetailedTime(manualTakeoff.playbackMs)}
            </Text>
          )}
          {manualLanding && (
            <Text style={styles.analysisRangeStat}>
              Manual landing marker: frame #{manualLanding.frameIndex} at {formatDetailedTime(manualLanding.playbackMs)}
            </Text>
          )}
          {currentPhase && (
            <>
              <Text style={styles.analysisRangeStat}>
                Left toe contact: {currentPhase.leftToeContact === null ? 'uncertain' : currentPhase.leftToeContact ? 'contact' : 'clear'}
              </Text>
              <Text style={styles.analysisRangeStat}>
                Right toe contact: {currentPhase.rightToeContact === null ? 'uncertain' : currentPhase.rightToeContact ? 'contact' : 'clear'}
              </Text>
              <Text style={styles.analysisRangeStat}>
                Contact confidence: L {currentPhase.leftContactConfidence !== null && currentPhase.leftContactConfidence !== undefined ? currentPhase.leftContactConfidence.toFixed(2) : '—'} / R {currentPhase.rightContactConfidence !== null && currentPhase.rightContactConfidence !== undefined ? currentPhase.rightContactConfidence.toFixed(2) : '—'}
              </Text>
            </>
          )}
          <View style={styles.playerControls}>
            <TouchableOpacity
              style={[styles.playerControlButton, !hasAlgoTakeoff && styles.btnDisabled]}
              onPress={() => seekToMs(analysis?.takeoffMs)}
              disabled={!hasAlgoTakeoff}>
              <Text style={styles.playerControlText}>Go Algo Takeoff</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.playerControlButton, !hasAlgoLanding && styles.btnDisabled]}
              onPress={() => seekToMs(analysis?.landingMs)}
              disabled={!hasAlgoLanding}>
              <Text style={styles.playerControlText}>Go Algo Landing</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.playerControls}>
            <TouchableOpacity
              style={[styles.playerControlButton, styles.manualTakeoffButton, !currentFrame && styles.btnDisabled]}
              onPress={() => currentFrame && onSetManualTakeoff(currentFrame)}
              disabled={!currentFrame}>
              <Text style={styles.playerControlText}>Mark Manual Takeoff</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.playerControlButton, styles.manualLandingButton, !currentFrame && styles.btnDisabled]}
              onPress={() => currentFrame && onSetManualLanding(currentFrame)}
              disabled={!currentFrame}>
              <Text style={styles.playerControlText}>Mark Manual Landing</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.playerControls}>
            <TouchableOpacity
              style={[styles.playerControlButton, !manualTakeoff && styles.btnDisabled]}
              onPress={() => seekToMs(manualTakeoff?.playbackMs)}
              disabled={!manualTakeoff}>
              <Text style={styles.playerControlText}>Go Manual Takeoff</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.playerControlButton, !manualLanding && styles.btnDisabled]}
              onPress={() => seekToMs(manualLanding?.playbackMs)}
              disabled={!manualLanding}>
              <Text style={styles.playerControlText}>Go Manual Landing</Text>
            </TouchableOpacity>
          </View>
          <TouchableOpacity
            style={[
              styles.floorModeButton,
              styles.clearManualButton,
              !manualTakeoff && !manualLanding && styles.btnDisabled,
            ]}
            onPress={onClearManualReview}
            disabled={!manualTakeoff && !manualLanding}>
            <Text style={styles.floorModeButtonText}>Clear Manual Review</Text>
          </TouchableOpacity>
        </View>
      )}

      {debugMode && (
        <View style={styles.floorCalibrationCard}>
          <Text style={styles.referenceTitle}>ML Kit Accurate Overlay</Text>
          <Text style={styles.referenceText}>
            The overlay draws ML Kit landmarks over the current frame. Use it to confirm the face,
            shoulders, hips, knees, ankles, heels, and toes stay tracked through takeoff and landing.
          </Text>
          <View style={styles.floorAdjustRow}>
            {analysis && (
              <TouchableOpacity style={styles.floorModeButton} onPress={onToggleDebugOverlay}>
                <Text style={styles.floorModeButtonText}>
                  {showDebugOverlay ? 'Hide Pose Overlay' : 'Show Pose Overlay'}
                </Text>
              </TouchableOpacity>
            )}
          </View>
          {showDebugOverlay && analysis && (
            <Text style={styles.referenceText}>
              Blue dashed line: average toe baseline from the standing calibration window.
            </Text>
          )}
        </View>
      )}
    </View>
  );
}

export default function JumpDetectorScreen() {
  const isFocused = useIsFocused();
  const insets = useSafeAreaInsets();
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const cameraRef = useRef<Camera>(null);
  const countdownTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const recordingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const recordingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const recordingActiveRef = useRef(false);
  const recordingStartedAtRef = useRef<number | null>(null);

  const [stage, setStage] = useState<JumpStage>('SETUP');
  const [countdownStep, setCountdownStep] = useState(0);
  const [recordingElapsedMs, setRecordingElapsedMs] = useState(0);
  const [clip, setClip] = useState<JumpClip | null>(null);
  const [nativeResult, setNativeResult] = useState<JumpVideoNativeResult | null>(null);
  const [analysis, setAnalysis] = useState<JumpAnalysisResult | null>(null);
  const [manualTakeoff, setManualTakeoff] = useState<ManualJumpEventSelection | null>(null);
  const [manualLanding, setManualLanding] = useState<ManualJumpEventSelection | null>(null);
  const [analysisStartMs, setAnalysisStartMs] = useState(0);
  const [analysisEndMs, setAnalysisEndMs] = useState(0);
  const [showDebugOverlay, setShowDebugOverlay] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const format = useMemo(() => selectFpsFirstFormat(device), [device]);
  const captureFps = useMemo(() => {
    if (!format) return 240;
    return format.maxFps;
  }, [format]);
  const nativeAnalysisAvailable = useMemo(() => isJumpVideoAnalysisAvailable(), []);

  const clearCountdown = useCallback(() => {
    if (countdownTimeoutRef.current) {
      clearTimeout(countdownTimeoutRef.current);
      countdownTimeoutRef.current = null;
    }
  }, []);

  const clearRecordingTimers = useCallback(() => {
    if (recordingTimeoutRef.current) {
      clearTimeout(recordingTimeoutRef.current);
      recordingTimeoutRef.current = null;
    }
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
  }, []);

  const resetSession = useCallback(() => {
    clearCountdown();
    clearRecordingTimers();
    recordingActiveRef.current = false;
    recordingStartedAtRef.current = null;
    setCountdownStep(0);
    setRecordingElapsedMs(0);
    setClip(null);
    setNativeResult(null);
    setAnalysis(null);
    setManualTakeoff(null);
    setManualLanding(null);
    setAnalysisStartMs(0);
    setAnalysisEndMs(0);
    setShowDebugOverlay(false);
    setError(null);
    setStage('SETUP');
  }, [clearCountdown, clearRecordingTimers]);

  const stopRecording = useCallback(async () => {
    if (!recordingActiveRef.current || !cameraRef.current) return;
    try {
      await cameraRef.current.stopRecording();
    } catch (recordingError) {
      const errorCode = (recordingError as CameraCaptureError)?.code;
      if (errorCode !== 'capture/no-recording-in-progress') {
        setError((recordingError as Error).message);
      }
    }
  }, []);

  const beginRecording = useCallback(() => {
    if (!cameraRef.current) {
      setError('Camera is not ready.');
      setStage('SETUP');
      return;
    }

    clearCountdown();
    clearRecordingTimers();
    setError(null);
    setStage('RECORDING');
    setRecordingElapsedMs(0);
    recordingStartedAtRef.current = Date.now();
    recordingActiveRef.current = true;

    cameraRef.current.startRecording({
      fileType: 'mov',
      videoCodec: 'h264',
      onRecordingError: (recordingError) => {
        clearRecordingTimers();
        recordingActiveRef.current = false;
        recordingStartedAtRef.current = null;
        if (recordingError.code !== 'capture/recording-canceled') {
          setError(recordingError.message);
        }
        setStage('SETUP');
      },
      onRecordingFinished: (video) => {
        clearRecordingTimers();
        recordingActiveRef.current = false;
        recordingStartedAtRef.current = null;
        const nextClip: JumpClip = {
          uri: normalizeMediaUri(video.path),
          fps: captureFps,
          durationMs: Math.round(video.duration * 1000),
          width: video.width,
          height: video.height,
          recordedAt: new Date().toISOString(),
        };
        startTransition(() => {
          setClip(nextClip);
          setNativeResult(null);
          setAnalysis(null);
          setManualTakeoff(null);
          setManualLanding(null);
          setAnalysisStartMs(0);
          setAnalysisEndMs(nextClip.durationMs);
          setShowDebugOverlay(false);
          setError(null);
          setStage('REVIEW');
        });
      },
    });

    recordingTimeoutRef.current = setTimeout(() => {
      void stopRecording();
    }, RECORDING_HARD_CAP_MS);

    recordingIntervalRef.current = setInterval(() => {
      if (!recordingStartedAtRef.current) return;
      setRecordingElapsedMs(Date.now() - recordingStartedAtRef.current);
    }, 100);
  }, [captureFps, clearCountdown, clearRecordingTimers, stopRecording]);

  const startSession = useCallback(() => {
    setError(null);
    setClip(null);
    setNativeResult(null);
    setAnalysis(null);
    setManualTakeoff(null);
    setManualLanding(null);
    setAnalysisStartMs(0);
    setAnalysisEndMs(0);
    setCountdownStep(0);
    setRecordingElapsedMs(0);
    setStage('COUNTDOWN');
  }, []);

  const analyzeClip = useCallback(async () => {
    if (!clip) return;

    const durationMs = Math.max(clip.durationMs, 1);
    const clampedStartMs = clampAnalysisStart(analysisStartMs, analysisEndMs, durationMs);
    const clampedEndMs = clampAnalysisEnd(analysisEndMs, clampedStartMs, durationMs);
    const analysisWindowMs = clampedEndMs - clampedStartMs;
    const maxSinglePassWindowMs =
      clip.assetId && clip.fps >= 120
        ? MAX_SLOW_MO_PLAYBACK_WINDOW_MS
        : MAX_STANDARD_ANALYSIS_WINDOW_MS;

    if (analysisWindowMs < MIN_ANALYSIS_WINDOW_MS) {
      setError('Select a longer analysis range so the clip includes the still standing phase and landing.');
      setStage('REVIEW');
      return;
    }

    setError(null);
    setManualTakeoff(null);
    setManualLanding(null);
    setStage('ANALYZING');

    try {
      let resolvedStartMs = clampedStartMs;
      let resolvedEndMs = clampedEndMs;

      if (analysisWindowMs > maxSinglePassWindowMs) {
        const coarseSampleFps = Math.max(24, Math.min(30, clip.fps));
        const coarseMaxFrames = Math.ceil((coarseSampleFps * analysisWindowMs) / 1000) + 24;
        const coarseNative = await analyzeRecordedJumpVideo(clip, {
          sampleFps: coarseSampleFps,
          maxFrames: coarseMaxFrames,
          analysisStartMs: clampedStartMs,
          analysisEndMs: clampedEndMs,
        });
        const suggestedWindow = suggestJumpAttemptWindow(coarseNative.frames, {
          videoDurationMs: coarseNative.videoDurationMs,
          videoFps: coarseNative.videoFps,
          sampleFps: coarseNative.sampleFps,
          playbackVideoFps: coarseNative.playbackVideoFps,
          playbackSampleFps: coarseNative.playbackSampleFps,
          personCountSummary: coarseNative.personCountSummary,
        });

        if (!suggestedWindow) {
          setError(
            'Automatic jump-range detection could not isolate a single attempt. Tighten the range manually around one jump, then re-run analysis.',
          );
          setStage('REVIEW');
          return;
        }

        resolvedStartMs = clampAnalysisStart(suggestedWindow.startMs, suggestedWindow.endMs, durationMs);
        resolvedEndMs = clampAnalysisEnd(suggestedWindow.endMs, resolvedStartMs, durationMs);
      }

      const requestedSampleFps = clip.fps > 60 ? clip.fps : 60;
      const requestedMaxFrames = Math.ceil((requestedSampleFps * (resolvedEndMs - resolvedStartMs)) / 1000) + 12;
      const native = await analyzeRecordedJumpVideo(clip, {
        sampleFps: requestedSampleFps,
        maxFrames: requestedMaxFrames,
        analysisStartMs: resolvedStartMs,
        analysisEndMs: resolvedEndMs,
      });
      const nextAnalysis = analyzeJumpLandmarks(native.frames, {
        videoDurationMs: native.videoDurationMs,
        videoFps: native.videoFps,
        sampleFps: native.sampleFps,
        playbackVideoFps: native.playbackVideoFps,
        playbackSampleFps: native.playbackSampleFps,
        playbackDurationMs: native.playbackDurationMs,
        captureDurationMs: native.captureDurationMs,
        timingMode: native.timingMode,
        timingConfidence: native.timingConfidence,
        captureTimeScale: native.captureTimeScale,
        hasTimeSegments: native.hasTimeSegments,
        usedOriginalAsset: native.usedOriginalAsset,
        usedPlaybackAsset: native.usedPlaybackAsset,
        originalDurationMs: native.originalDurationMs,
        timebaseSource: native.timebaseSource,
        personCountSummary: native.personCountSummary,
      });

      startTransition(() => {
        setNativeResult(native);
        setAnalysis(nextAnalysis);
        setAnalysisStartMs(resolvedStartMs);
        setAnalysisEndMs(resolvedEndMs);
        setStage('RESULT_DEBUG');
      });
    } catch (analysisError) {
      setError((analysisError as Error).message);
      setStage('REVIEW');
    }
  }, [analysisEndMs, analysisStartMs, clip]);

  const toggleDebugOverlay = useCallback(() => {
    setShowDebugOverlay((value) => !value);
  }, []);

  const updateAnalysisRange = useCallback(
    (nextStartMs: number, nextEndMs: number) => {
      if (!clip) return;
      const durationMs = Math.max(clip.durationMs, 1);
      const clampedStartMs = clampAnalysisStart(nextStartMs, nextEndMs, durationMs);
      const clampedEndMs = clampAnalysisEnd(nextEndMs, clampedStartMs, durationMs);

      setAnalysisStartMs(clampedStartMs);
      setAnalysisEndMs(clampedEndMs);
      setNativeResult(null);
      setAnalysis(null);
      setManualTakeoff(null);
      setManualLanding(null);
      setShowDebugOverlay(false);
      setError(null);
      setStage('REVIEW');
    },
    [clip],
  );

  const setRangeStartAt = useCallback(
    (timeMs: number) => {
      if (!clip) return;
      updateAnalysisRange(timeMs, analysisEndMs || clip.durationMs);
    },
    [analysisEndMs, clip, updateAnalysisRange],
  );

  const setRangeEndAt = useCallback(
    (timeMs: number) => {
      if (!clip) return;
      updateAnalysisRange(analysisStartMs, timeMs);
    },
    [analysisStartMs, clip, updateAnalysisRange],
  );

  const resetAnalysisRange = useCallback(() => {
    if (!clip) return;
    updateAnalysisRange(0, clip.durationMs);
  }, [clip, updateAnalysisRange]);

  const setManualTakeoffFromFrame = useCallback((frame: JumpLandmarkFrame) => {
    setManualTakeoff(manualSelectionFromFrame(frame));
  }, []);

  const setManualLandingFromFrame = useCallback((frame: JumpLandmarkFrame) => {
    setManualLanding(manualSelectionFromFrame(frame));
  }, []);

  const clearManualReview = useCallback(() => {
    setManualTakeoff(null);
    setManualLanding(null);
  }, []);

  const importVideo = useCallback(async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission required', 'Photo library access is needed to import videos.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['videos'],
      quality: 1,
      preferredAssetRepresentationMode:
        ImagePicker.UIImagePickerPreferredAssetRepresentationMode.Current,
      videoExportPreset: ImagePicker.VideoExportPreset.Passthrough,
    });

    if (result.canceled || result.assets.length === 0) return;

    const asset = result.assets[0];
    const uri = normalizeMediaUri(asset.uri);
    const width = asset.width ?? 0;
    const height = asset.height ?? 0;
    const durationMs = Math.round(asset.duration ?? 0);

    // Default to 240 fps for imported slow-motion videos.
    // The native analyzer reads the actual fps from the file metadata.
    const importedClip: JumpClip = {
      uri,
      assetId: asset.assetId ?? null,
      fps: 240,
      durationMs,
      width,
      height,
      recordedAt: new Date().toISOString(),
    };

    startTransition(() => {
      setClip(importedClip);
      setNativeResult(null);
      setAnalysis(null);
      setManualTakeoff(null);
      setManualLanding(null);
      setAnalysisStartMs(0);
      setAnalysisEndMs(importedClip.durationMs);
      setShowDebugOverlay(false);
      setError(null);
      setStage('REVIEW');
    });
  }, []);

  useEffect(() => {
    if (stage !== 'COUNTDOWN') return;
    if (countdownStep >= COUNTDOWN_SEQUENCE.length) {
      beginRecording();
      return;
    }

    const delay = countdownStep === 0 ? HOLD_STILL_MS : 1000;
    countdownTimeoutRef.current = setTimeout(() => {
      setCountdownStep((value) => value + 1);
    }, delay);

    return clearCountdown;
  }, [beginRecording, clearCountdown, countdownStep, stage]);

  useEffect(() => {
    return () => {
      clearCountdown();
      clearRecordingTimers();
    };
  }, [clearCountdown, clearRecordingTimers]);

  useEffect(() => {
    if (!isFocused && recordingActiveRef.current) {
      const cancelPromise = cameraRef.current?.cancelRecording();
      cancelPromise?.catch(() => undefined);
    }
  }, [isFocused]);

  const manualReview = useMemo(() => {
    if (!manualTakeoff || !manualLanding) return null;
    if (manualLanding.physicalMs <= manualTakeoff.physicalMs) return null;

    const flightPhysicalMs = manualLanding.physicalMs - manualTakeoff.physicalMs;
    const timingTrusted = analysis?.debug.timingTrusted !== false;
    const heightCm = timingTrusted ? heightFromFlightTime(flightPhysicalMs) : null;
    const playbackFps =
      nativeResult?.playbackSampleFps ?? nativeResult?.sampleFps ?? clip?.fps ?? null;
    const msPerPlaybackFrame =
      playbackFps && playbackFps > 0 ? 1000 / playbackFps : null;
    const takeoffDeltaMs =
      analysis?.takeoffMs !== null && analysis?.takeoffMs !== undefined
        ? manualTakeoff.playbackMs - analysis.takeoffMs
        : null;
    const landingDeltaMs =
      analysis?.landingMs !== null && analysis?.landingMs !== undefined
        ? manualLanding.playbackMs - analysis.landingMs
        : null;
    const takeoffDeltaFrames =
      takeoffDeltaMs !== null && msPerPlaybackFrame ? takeoffDeltaMs / msPerPlaybackFrame : null;
    const landingDeltaFrames =
      landingDeltaMs !== null && msPerPlaybackFrame ? landingDeltaMs / msPerPlaybackFrame : null;
    const flightDeltaMs =
      analysis?.flightMs !== null && analysis?.flightMs !== undefined
        ? flightPhysicalMs - analysis.flightMs
        : null;
    const heightDeltaCm =
      heightCm !== null && analysis?.heightCm !== null && analysis?.heightCm !== undefined
        ? heightCm - analysis.heightCm
        : null;

    return {
      flightPhysicalMs,
      heightCm,
      takeoffDeltaMs,
      landingDeltaMs,
      takeoffDeltaFrames,
      landingDeltaFrames,
      flightDeltaMs,
      heightDeltaCm,
      timingTrusted,
    };
  }, [analysis, clip?.fps, manualLanding, manualTakeoff, nativeResult]);
  const manualReviewHasInvalidOrder =
    manualTakeoff !== null &&
    manualLanding !== null &&
    manualLanding.physicalMs <= manualTakeoff.physicalMs;

  if (Platform.OS !== 'ios') {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.permText}>Recorded jump analysis is implemented on iOS first.</Text>
      </SafeAreaView>
    );
  }

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.permText}>Camera access is required to record and analyze jumps.</Text>
        <TouchableOpacity style={[styles.btn, styles.btnPrimary]} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Camera Access</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  if (!device || !format) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.permText}>No compatible rear camera format was found.</Text>
      </SafeAreaView>
    );
  }

  const showCamera = stage === 'SETUP' || stage === 'COUNTDOWN' || stage === 'RECORDING';
  const showRecordedReview = stage === 'REVIEW' || stage === 'ANALYZING' || stage === 'RESULT_DEBUG';
  const canStart =
    nativeAnalysisAvailable && stage === 'SETUP' && isFocused && device !== undefined && format !== undefined;
  const statusCard = (
    <View style={styles.statusCard}>
      <View style={styles.statusRow}>
        <Text style={styles.stageText}>{stageLabel(stage)}</Text>
        <Text style={styles.stageSubText}>
          {showRecordedReview && nativeResult
            ? `${(nativeResult.playbackSampleFps ?? nativeResult.sampleFps).toFixed(0)} fps playback / ${nativeResult.videoFps.toFixed(0)} fps effective`
            : showRecordedReview && clip
              ? `${clip.fps} fps source`
              : `${captureFps} fps capture target`}
        </Text>
      </View>
      <Text style={[styles.statusHeadline, error && styles.errorText]}>
        {statusCopy(stage, countdownStep, recordingElapsedMs, error)}
      </Text>
      {!nativeAnalysisAvailable && (
        <Text style={styles.warningText}>
          Native analyzer unavailable. Rebuild the iOS development client so the local ML Kit module
          is included.
        </Text>
      )}
    </View>
  );
  const reviewActionButtons = clip ? (
    <View style={styles.btnRow}>
      <TouchableOpacity
        style={[
          styles.btn,
          styles.btnPrimary,
          (stage === 'ANALYZING' || !nativeAnalysisAvailable) && styles.btnDisabled,
        ]}
        onPress={() => void analyzeClip()}
        disabled={stage === 'ANALYZING' || !nativeAnalysisAvailable}>
        <Text style={styles.btnText}>{analysis ? 'Re-analyze Video' : 'Analyze Video'}</Text>
      </TouchableOpacity>
      <TouchableOpacity style={[styles.btn, styles.btnSecondary]} onPress={resetSession}>
        <Text style={styles.btnText}>Retake</Text>
      </TouchableOpacity>
    </View>
  ) : null;
  const reviewContent = clip ? (
    <View style={styles.reviewCards}>
      <ReviewPlayer
        clip={clip}
        nativeResult={nativeResult}
        analysis={analysis}
        manualTakeoff={manualTakeoff}
        manualLanding={manualLanding}
        analysisStartMs={analysisStartMs}
        analysisEndMs={analysisEndMs}
        showDebugOverlay={showDebugOverlay}
        debugMode={debugMode}
        onToggleDebugOverlay={toggleDebugOverlay}
        onSetAnalysisStart={setRangeStartAt}
        onSetAnalysisEnd={setRangeEndAt}
        onResetAnalysisRange={resetAnalysisRange}
        onSetManualTakeoff={setManualTakeoffFromFrame}
        onSetManualLanding={setManualLandingFromFrame}
        onClearManualReview={clearManualReview}
      />

      <View style={styles.resultCard}>
        <Text style={[styles.resultHeadline, { color: summaryColor(analysis) }]}>
          {analysis?.invalidReason ? 'Invalid attempt' : analysis ? 'Measured jump' : 'Recorded clip'}
        </Text>
        <Text style={styles.resultLine}>
          Takeoff playback: {formatDetailedTime(analysis?.takeoffMs ?? null)}
        </Text>
        <Text style={styles.resultLine}>
          Landing playback: {formatDetailedTime(analysis?.landingMs ?? null)}
        </Text>
        <Text style={styles.resultLine}>
          Takeoff physical: {formatDetailedTime(analysis?.takeoffPhysicalMs ?? null)}
        </Text>
        <Text style={styles.resultLine}>
          Landing physical: {formatDetailedTime(analysis?.landingPhysicalMs ?? null)}
        </Text>
        <Text style={styles.resultLine}>Flight physical: {formatDetailedTime(analysis?.flightMs ?? null)}</Text>
        <Text style={styles.resultLine}>
          Timing mode: {formatTimingMode(analysis?.debug.timingMode)}
        </Text>
        <Text style={styles.resultLine}>
          Timing confidence: {formatPercent(analysis?.debug.timingConfidence ?? null)}
        </Text>
        <Text style={styles.resultLine}>
          Playback duration: {formatDetailedTime(analysis?.debug.playbackDurationMs ?? nativeResult?.playbackDurationMs ?? clip.durationMs)}
        </Text>
        <Text style={styles.resultLine}>
          Physical duration: {formatDetailedTime(analysis?.debug.captureDurationMs ?? nativeResult?.captureDurationMs ?? null)}
        </Text>
        <Text style={styles.resultLine}>
          Timebase source: {formatTimebaseSource(analysis?.debug.timebaseSource)}
        </Text>
        <Text style={styles.resultLine}>
          Analysis range: {formatDetailedTime(analysisStartMs)} → {formatDetailedTime(analysisEndMs)}
        </Text>
        <Text style={styles.resultLine}>
          Detected attempt window:{' '}
          {analysis?.debug.attemptWindowStartMs !== null && analysis?.debug.attemptWindowStartMs !== undefined
            ? `${formatDetailedTime(analysis.debug.attemptWindowStartMs)} → ${formatDetailedTime(analysis.debug.attemptWindowEndMs ?? null)}`
            : '—'}
        </Text>
        <Text style={styles.resultLine}>
          Height:{' '}
          {analysis?.heightCm !== null && analysis?.heightCm !== undefined
            ? `${analysis.heightCm.toFixed(1)} cm`
            : '—'}
        </Text>
        <Text style={styles.resultLine}>Quality: {analysis?.quality ?? '—'}</Text>
        <Text style={styles.resultLine}>
          Precision: {formatFramePrecision(nativeResult?.sampleFps ?? clip.fps)}
        </Text>
        <Text style={styles.resultSummary}>
          {analysis
            ? explainJumpAnalysis(analysis)
            : 'Run analysis to extract full-body landmarks and event timings.'}
        </Text>
        {analysis?.invalidReason === 'TIMING_AMBIGUOUS' && (
          <Text style={styles.resultSummary}>
            Physical jump height is intentionally suppressed until the imported clip’s timebase can
            be trusted.
          </Text>
        )}
        {clip.assetId &&
          nativeResult &&
          (nativeResult.playbackVideoFps ?? nativeResult.videoFps) <= 60 &&
          nativeResult.videoFps <= 60 &&
          clip.fps >= 240 && (
          <Text style={styles.resultSummary}>
            The imported Photos asset still resolved to a {nativeResult.videoFps.toFixed(0)} fps
            playback file. Re-import the original slow-motion asset from Photos without editing if
            height remains invalid.
          </Text>
        )}
        {analysis?.invalidReason === 'FEET_NOT_VISIBLE' && (
          <Text style={styles.resultSummary}>
            Detector coverage: {((analysis.debug.feetVisibleRatio ?? 0) * 100).toFixed(0)}% of
            frames with the feet tracked.
          </Text>
        )}
        {analysis?.invalidReason === 'BODY_NOT_FULLY_VISIBLE' && (
          <Text style={styles.resultSummary}>
            Full-body coverage: {((analysis.debug.fullBodyVisibleRatio ?? 0) * 100).toFixed(0)}%
            of frames with face, torso, hips, and feet all visible.
          </Text>
        )}
        {analysis?.qualityFlags && (
          <Text style={styles.resultFlags}>
            Flags: {analysis.qualityFlags.map(formatQualityFlag).join(', ')}
          </Text>
        )}
      </View>

      <TouchableOpacity
        style={styles.debugToggleButton}
        onPress={() => setDebugMode((v) => !v)}>
        <Text style={styles.debugToggleText}>
          {debugMode ? 'Hide Debug Tools' : 'Show Debug Tools'}
        </Text>
      </TouchableOpacity>

      {debugMode && (
        <View style={styles.debugCard}>
          <Text style={styles.referenceTitle}>Manual Review</Text>
          <Text style={styles.referenceText}>
            Mark the exact current frame for takeoff and landing, then compare the manual result to
            the algorithm output.
          </Text>
          <Text style={styles.referenceText}>
            Manual takeoff: {manualTakeoff ? `frame #${manualTakeoff.frameIndex} at ${formatDetailedTime(manualTakeoff.playbackMs)}` : '—'}
          </Text>
          <Text style={styles.referenceText}>
            Manual landing: {manualLanding ? `frame #${manualLanding.frameIndex} at ${formatDetailedTime(manualLanding.playbackMs)}` : '—'}
          </Text>
          {manualReview ? (
            <>
              <Text style={styles.referenceText}>
                Manual takeoff physical: {formatDetailedTime(manualTakeoff?.physicalMs ?? null)}
              </Text>
              <Text style={styles.referenceText}>
                Manual landing physical: {formatDetailedTime(manualLanding?.physicalMs ?? null)}
              </Text>
              <Text style={styles.referenceText}>
                Manual flight physical: {formatDetailedTime(manualReview.flightPhysicalMs)}
              </Text>
              <Text style={styles.referenceText}>
                Manual height:{' '}
                {manualReview.heightCm !== null ? `${manualReview.heightCm.toFixed(1)} cm` : '—'}
              </Text>
              {!manualReview.timingTrusted && (
                <Text style={styles.referenceText}>
                  Manual physical timing is unsafe because this clip’s timebase is ambiguous.
                </Text>
              )}
              <Text style={styles.referenceText}>
                Takeoff delta: {formatSignedMilliseconds(manualReview.takeoffDeltaMs)} / {formatSignedFrames(manualReview.takeoffDeltaFrames)}
              </Text>
              <Text style={styles.referenceText}>
                Landing delta: {formatSignedMilliseconds(manualReview.landingDeltaMs)} / {formatSignedFrames(manualReview.landingDeltaFrames)}
              </Text>
              <Text style={styles.referenceText}>
                Flight delta: {formatSignedMilliseconds(manualReview.flightDeltaMs)}
              </Text>
              <Text style={styles.referenceText}>
                Height delta: {formatSignedCentimeters(manualReview.heightDeltaCm)}
              </Text>
            </>
          ) : manualReviewHasInvalidOrder ? (
            <Text style={styles.referenceText}>
              Manual landing must be after manual takeoff. Re-mark one of the events from the player.
            </Text>
          ) : (
            <Text style={styles.referenceText}>
              Mark both manual events from the player to compute manual flight time, manual height,
              and deltas versus the algorithm.
            </Text>
          )}
        </View>
      )}

      {debugMode && analysis && (
        <View style={styles.debugCard}>
          <Text style={styles.referenceTitle}>Debug Metrics</Text>
          <Text style={styles.referenceText}>
            Frames: {analysis?.debug.analyzedFrameCount ?? nativeResult?.frames.length ?? 0}
          </Text>
          {analysis?.debug.calibrationStartMs !== null &&
            analysis?.debug.calibrationStartMs !== undefined && (
            <Text style={styles.referenceText}>
              Calibration start: {formatDetailedTime(analysis.debug.calibrationStartMs)}
            </Text>
          )}
          <Text style={styles.referenceText}>
            Calibration end: {formatDetailedTime(analysis?.debug.calibrationEndMs ?? null)}
          </Text>
          {analysis?.debug.attemptWindowStartMs !== null &&
            analysis?.debug.attemptWindowStartMs !== undefined && (
            <Text style={styles.referenceText}>
              Auto attempt window:{' '}
              {formatDetailedTime(analysis.debug.attemptWindowStartMs)} → {formatDetailedTime(analysis.debug.attemptWindowEndMs ?? null)}
            </Text>
          )}
          <Text style={styles.referenceText}>
            Sample FPS: {(analysis?.debug.sampleFps ?? nativeResult?.sampleFps ?? clip.fps).toFixed(1)}
          </Text>
          {nativeResult?.playbackSampleFps && (
            <Text style={styles.referenceText}>
              Playback FPS: {nativeResult.playbackSampleFps.toFixed(1)}
            </Text>
          )}
          <Text style={styles.referenceText}>
            Timing mode: {formatTimingMode(analysis?.debug.timingMode)}
          </Text>
          <Text style={styles.referenceText}>
            Timing confidence: {formatPercent(analysis?.debug.timingConfidence ?? null)}
          </Text>
          <Text style={styles.referenceText}>
            Playback duration: {formatDetailedTime(analysis?.debug.playbackDurationMs ?? null)}
          </Text>
          <Text style={styles.referenceText}>
            Physical duration: {formatDetailedTime(analysis?.debug.captureDurationMs ?? null)}
          </Text>
          <Text style={styles.referenceText}>
            Timebase source: {formatTimebaseSource(analysis?.debug.timebaseSource)}
          </Text>
          {analysis?.debug.slowMotionScaleFactor && (
            <Text style={styles.referenceText}>
              Slow-motion scale factor:{' '}
              {(1 / analysis.debug.slowMotionScaleFactor).toFixed(2)}x capture vs playback
            </Text>
          )}
          <Text style={styles.referenceText}>
            Frame interval: {formatFramePrecision(analysis?.debug.sampleFps ?? nativeResult?.sampleFps ?? clip.fps)}
          </Text>
          <Text style={styles.referenceText}>
            Avg confidence: {(analysis?.debug.averageConfidence ?? 0).toFixed(2)}
          </Text>
          <Text style={styles.referenceText}>
            Full body visible: {((analysis?.debug.fullBodyVisibleRatio ?? 0) * 100).toFixed(0)}%
          </Text>
          <Text style={styles.referenceText}>
            Feet visible: {((analysis?.debug.feetVisibleRatio ?? 0) * 100).toFixed(0)}%
          </Text>
          <Text style={styles.referenceText}>
            Horizontal drift: {((analysis?.debug.maxHorizontalDrift ?? 0) * 100).toFixed(1)}%
          </Text>
          <Text style={styles.referenceText}>
            Calibration stability: {((analysis?.debug.calibrationStability ?? 0) * 100).toFixed(2)}%
          </Text>
          {analysis?.debug.averageContactReliability !== null &&
            analysis?.debug.averageContactReliability !== undefined && (
            <Text style={styles.referenceText}>
              Contact reliability: {(analysis.debug.averageContactReliability * 100).toFixed(0)}%
            </Text>
          )}
          {analysis?.debug.attemptSelectionScore !== null &&
            analysis?.debug.attemptSelectionScore !== undefined && (
            <Text style={styles.referenceText}>
              Attempt score: {analysis.debug.attemptSelectionScore.toFixed(1)}
            </Text>
          )}
        </View>
      )}
    </View>
  ) : null;

  return (
    <View style={styles.container}>
      {showCamera ? (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={isFocused}
          format={format}
          fps={captureFps}
          video
          audio={false}
        />
      ) : (
        <View style={styles.playerBackdrop} />
      )}

      {showCamera && <View pointerEvents="none" style={styles.guideBox} />}

      {!showRecordedReview && <SafeAreaView edges={['top']} style={styles.topOverlay}>{statusCard}</SafeAreaView>}

      {stage === 'COUNTDOWN' && (
        <View pointerEvents="none" style={styles.countdownOverlay}>
          <Text style={styles.countdownLabel}>
            {COUNTDOWN_SEQUENCE[Math.min(countdownStep, COUNTDOWN_SEQUENCE.length - 1)]}
          </Text>
        </View>
      )}

      {!showRecordedReview && (
        <View style={[styles.bottomOverlay, { paddingBottom: insets.bottom + Spacing.three }]}>
          {(stage === 'SETUP' || stage === 'COUNTDOWN' || stage === 'RECORDING') && (
            <>
              <View style={styles.instructionsCard}>
                {SETUP_COPY.map((line) => (
                  <Text key={line} style={styles.instructionText}>
                    {line}
                  </Text>
                ))}
              </View>
              <View style={styles.referenceCard}>
                <Text style={styles.referenceTitle}>Capture Contract</Text>
                <Text style={styles.referenceText}>iPhone slow-motion preferred. Use fixed support if recording in-app.</Text>
                <Text style={styles.referenceText}>Keep the face, hips, knees, ankles, and both feet visible for the whole jump.</Text>
              </View>
              <View style={styles.btnRow}>
                <TouchableOpacity
                  style={[styles.btn, styles.btnPrimary, !canStart && styles.btnDisabled]}
                  onPress={startSession}
                  disabled={!canStart}>
                  <Text style={styles.btnText}>Start Jump Session</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.btn, styles.btnSecondary]}
                  onPress={stage === 'RECORDING' ? () => void stopRecording() : resetSession}>
                  <Text style={styles.btnText}>{stage === 'RECORDING' ? 'Stop Early' : 'Reset'}</Text>
                </TouchableOpacity>
              </View>
              {stage === 'SETUP' && (
                <TouchableOpacity
                  style={[styles.btn, styles.btnImport]}
                  onPress={() => void importVideo()}>
                  <Text style={styles.btnText}>Import Full-Body 240 fps Video</Text>
                </TouchableOpacity>
              )}
            </>
          )}
        </View>
      )}

      {showRecordedReview && (
        <SafeAreaView edges={['top']} style={styles.reviewLayout}>
          <View style={styles.topOverlay}>{statusCard}</View>
          <ScrollView
            style={styles.reviewScroll}
            contentContainerStyle={[
              styles.reviewScrollContent,
              { paddingBottom: insets.bottom + 116 },
            ]}
            showsVerticalScrollIndicator={false}>
            {reviewContent}
          </ScrollView>
          {reviewActionButtons && (
            <SafeAreaView
              edges={['bottom']}
              style={[styles.reviewActionBar, { paddingBottom: Spacing.three }]}>
              {reviewActionButtons}
            </SafeAreaView>
          )}
          {stage === 'ANALYZING' && (
            <View style={styles.analyzingOverlay}>
              <ActivityIndicator size="large" color="#ffffff" />
            </View>
          )}
        </SafeAreaView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#050816',
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.three,
    paddingHorizontal: Spacing.four,
    backgroundColor: '#050816',
  },
  permText: {
    color: '#ffffff',
    textAlign: 'center',
    fontSize: 16,
    lineHeight: 22,
  },
  topOverlay: {
    paddingHorizontal: Spacing.three,
    paddingTop: Spacing.three,
  },
  statusCard: {
    backgroundColor: 'rgba(5,8,22,0.78)',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: 'rgba(148,163,184,0.25)',
    padding: Spacing.three,
    gap: Spacing.two,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: Spacing.two,
  },
  stageText: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: '700',
  },
  stageSubText: {
    color: '#94a3b8',
    fontSize: 13,
    fontWeight: '600',
  },
  statusHeadline: {
    color: '#e2e8f0',
    fontSize: 15,
    lineHeight: 20,
    fontWeight: '500',
  },
  warningText: {
    color: '#fbbf24',
    fontSize: 13,
    lineHeight: 18,
  },
  errorText: {
    color: '#fca5a5',
  },
  guideBox: {
    position: 'absolute',
    top: '14%',
    left: '13%',
    right: '13%',
    height: '74%',
    borderRadius: 24,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.85)',
    backgroundColor: 'rgba(255,255,255,0.05)',
  },
  countdownOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  countdownLabel: {
    color: '#ffffff',
    fontSize: 56,
    fontWeight: '800',
    backgroundColor: 'rgba(5,8,22,0.62)',
    overflow: 'hidden',
    paddingHorizontal: 28,
    paddingVertical: 18,
    borderRadius: 28,
  },
  bottomOverlay: {
    marginTop: 'auto',
    paddingHorizontal: Spacing.three,
    gap: Spacing.three,
  },
  reviewLayout: {
    flex: 1,
  },
  reviewScroll: {
    flex: 1,
  },
  reviewScrollContent: {
    paddingHorizontal: Spacing.three,
    paddingTop: Spacing.three,
  },
  reviewCards: {
    gap: Spacing.three,
  },
  instructionsCard: {
    backgroundColor: 'rgba(5,8,22,0.82)',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(148,163,184,0.25)',
    padding: Spacing.three,
    gap: Spacing.one,
  },
  instructionText: {
    color: '#ffffff',
    fontSize: 14,
    lineHeight: 19,
  },
  referenceCard: {
    backgroundColor: 'rgba(15,23,42,0.9)',
    borderRadius: 18,
    padding: Spacing.three,
    gap: Spacing.one,
  },
  referenceTitle: {
    color: '#f8fafc',
    fontSize: 15,
    fontWeight: '700',
  },
  referenceText: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 18,
  },
  btnRow: {
    flexDirection: 'row',
    gap: Spacing.two,
  },
  btn: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 16,
    minHeight: 52,
    paddingHorizontal: Spacing.three,
  },
  btnPrimary: {
    backgroundColor: '#0f766e',
  },
  btnSecondary: {
    backgroundColor: 'rgba(15,23,42,0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148,163,184,0.25)',
  },
  btnImport: {
    backgroundColor: 'rgba(99,102,241,0.85)',
  },
  btnDisabled: {
    opacity: 0.45,
  },
  btnText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '700',
  },
  playerBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#050816',
  },
  playerSection: {
    gap: Spacing.two,
  },
  playerShell: {
    width: '100%',
    overflow: 'hidden',
    borderRadius: 22,
    backgroundColor: '#000000',
    borderWidth: 1,
    borderColor: 'rgba(148,163,184,0.25)',
  },
  videoContentFrame: {
    position: 'absolute',
    overflow: 'hidden',
  },
  playerMetaBadge: {
    position: 'absolute',
    top: Spacing.two,
    left: Spacing.two,
    right: Spacing.two,
    flexDirection: 'row',
    justifyContent: 'space-between',
    backgroundColor: 'rgba(5,8,22,0.74)',
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  playerMetaText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '700',
  },
  timeline: {
    height: 16,
    borderRadius: 999,
    backgroundColor: 'rgba(148,163,184,0.18)',
    overflow: 'hidden',
    position: 'relative',
  },
  timelineProgress: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    backgroundColor: '#38bdf8',
  },
  timelineRange: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    backgroundColor: 'rgba(168,85,247,0.28)',
  },
  timelineMarker: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 3,
    marginLeft: -1.5,
  },
  rangeStartMarker: {
    backgroundColor: '#a78bfa',
  },
  rangeEndMarker: {
    backgroundColor: '#e879f9',
  },
  takeoffMarker: {
    backgroundColor: '#22c55e',
  },
  landingMarker: {
    backgroundColor: '#f97316',
  },
  manualTakeoffMarker: {
    backgroundColor: '#14b8a6',
  },
  manualLandingMarker: {
    backgroundColor: '#facc15',
  },
  playerControls: {
    flexDirection: 'row',
    gap: Spacing.two,
  },
  playerControlButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 42,
    borderRadius: 14,
    backgroundColor: 'rgba(15,23,42,0.92)',
    borderWidth: 1,
    borderColor: 'rgba(148,163,184,0.25)',
  },
  playerControlPrimary: {
    backgroundColor: '#1d4ed8',
  },
  playerControlText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '700',
  },
  analysisRangeCard: {
    backgroundColor: 'rgba(15,23,42,0.92)',
    borderRadius: 18,
    padding: Spacing.three,
    gap: Spacing.two,
  },
  analysisRangeStats: {
    gap: Spacing.one,
  },
  analysisRangeStat: {
    color: '#cbd5e1',
    fontSize: 13,
    lineHeight: 18,
  },
  eventReviewCard: {
    backgroundColor: 'rgba(15,23,42,0.92)',
    borderRadius: 18,
    padding: Spacing.three,
    gap: Spacing.two,
  },
  rangeActionButton: {
    backgroundColor: 'rgba(76,29,149,0.88)',
  },
  manualTakeoffButton: {
    backgroundColor: 'rgba(13,148,136,0.88)',
  },
  manualLandingButton: {
    backgroundColor: 'rgba(202,138,4,0.88)',
  },
  clearManualButton: {
    backgroundColor: 'rgba(51,65,85,0.98)',
  },
  floorCalibrationCard: {
    backgroundColor: 'rgba(15,23,42,0.92)',
    borderRadius: 18,
    padding: Spacing.three,
    gap: Spacing.two,
  },
  floorAdjustRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
  },
  floorModeButton: {
    flex: 1,
    minHeight: 42,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 12,
    backgroundColor: 'rgba(30,41,59,0.96)',
    borderWidth: 1,
    borderColor: 'rgba(148,163,184,0.24)',
  },
  floorModeButtonText: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '700',
    textAlign: 'center',
  },
  resultCard: {
    backgroundColor: 'rgba(15,23,42,0.9)',
    borderRadius: 18,
    padding: Spacing.three,
    gap: Spacing.one,
  },
  resultHeadline: {
    fontSize: 18,
    fontWeight: '800',
  },
  resultLine: {
    color: '#e2e8f0',
    fontSize: 14,
    lineHeight: 19,
  },
  resultSummary: {
    color: '#cbd5e1',
    fontSize: 14,
    lineHeight: 20,
    marginTop: Spacing.one,
  },
  resultFlags: {
    color: '#fbbf24',
    fontSize: 13,
    lineHeight: 18,
  },
  debugToggleButton: {
    alignSelf: 'center',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 12,
    backgroundColor: 'rgba(51,65,85,0.8)',
    borderWidth: 1,
    borderColor: 'rgba(148,163,184,0.3)',
  },
  debugToggleText: {
    color: '#94a3b8',
    fontSize: 13,
    fontWeight: '600',
  },
  debugCard: {
    backgroundColor: 'rgba(15,23,42,0.9)',
    borderRadius: 18,
    padding: Spacing.three,
    gap: Spacing.one,
  },
  reviewActionBar: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: Spacing.three,
    paddingTop: Spacing.three,
    backgroundColor: 'rgba(5,8,22,0.96)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(148,163,184,0.2)',
  },
  analyzingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

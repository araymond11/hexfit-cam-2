import { useIsFocused } from '@react-navigation/native';
import { useEvent } from 'expo';
import { VideoView, useVideoPlayer } from 'expo-video';
import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
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
import { analyzeJumpLandmarks, explainJumpAnalysis, findFrameAtTime } from '@/utils/jumpAnalysis';
import type {
  JumpAnalysisResult,
  JumpClip,
  JumpContactPhase,
  JumpPhaseSample,
  JumpVideoNativeResult,
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
const RECORDING_HARD_CAP_MS = 4000;

const SETUP_COPY = [
  'Place the camera in a side view.',
  'Keep the full body, both feet, and the floor visible.',
  'Stand still through the countdown, then jump once.',
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

function normalizeFileUri(path: string): string {
  return path.startsWith('file://') ? path : `file://${path}`;
}

function formatMs(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '—';
  return `${value.toFixed(1)} ms`;
}

function phaseColor(phase: JumpContactPhase | null): string {
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
    return `Recording ${(recordingElapsedMs / 1000).toFixed(1)} s / 4.0 s`;
  }
  if (stage === 'ANALYZING') {
    return 'Decoding frames, extracting landmarks, and measuring airtime.';
  }
  if (stage === 'REVIEW') {
    return 'Scrub the clip or run analysis again.';
  }
  if (stage === 'RESULT_DEBUG') {
    return 'Inspect takeoff, landing, and frame-by-frame debug overlays.';
  }
  return 'Recorded analysis uses VisionCamera capture and native offline landmark extraction.';
}

interface ReviewPlayerProps {
  clip: JumpClip;
  nativeResult: JumpVideoNativeResult | null;
  analysis: JumpAnalysisResult | null;
}

function ReviewPlayer({ clip, nativeResult, analysis }: ReviewPlayerProps) {
  const [timelineWidth, setTimelineWidth] = useState(0);
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
  const sampleFps = Math.max(nativeResult?.sampleFps ?? clip.fps ?? 30, 1);
  const currentFrame = useMemo(
    () => (nativeResult ? findFrameAtTime(nativeResult.frames, currentMs) : null),
    [nativeResult, currentMs],
  );
  const currentPhase = useMemo(
    () => (analysis ? findPhaseSampleAtTime(analysis.phaseTimeline, currentMs) : null),
    [analysis, currentMs],
  );

  const seekToRatio = useCallback(
    (ratio: number) => {
      player.pause();
      player.currentTime = Math.max(0, Math.min(1, ratio)) * (durationMs / 1000);
    },
    [durationMs, player],
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

  const stepFrame = useCallback(
    (direction: -1 | 1) => {
      player.pause();
      const nextTime = Math.max(0, currentTime + direction / sampleFps);
      player.currentTime = nextTime;
    },
    [currentTime, player, sampleFps],
  );

  const takeoffRatio =
    analysis?.takeoffMs !== null && analysis?.takeoffMs !== undefined
      ? analysis.takeoffMs / durationMs
      : null;
  const landingRatio =
    analysis?.landingMs !== null && analysis?.landingMs !== undefined
      ? analysis.landingMs / durationMs
      : null;

  return (
    <View style={styles.playerSection}>
      <View style={[styles.playerShell, { height: playerHeight }]}>
        <VideoView
          player={player}
          style={StyleSheet.absoluteFill}
          contentFit="contain"
          nativeControls={false}
        />
        {nativeResult && analysis && (
          <View pointerEvents="none" style={StyleSheet.absoluteFill}>
            <JumpDebugOverlay frame={currentFrame} analysis={analysis} />
          </View>
        )}
        <View style={styles.playerMetaBadge}>
          <Text style={styles.playerMetaText}>{formatMs(currentMs)}</Text>
          <Text style={[styles.playerMetaText, { color: phaseColor(currentPhase?.phase ?? null) }]}>
            {currentPhase?.phase ?? 'RAW_CLIP'}
          </Text>
        </View>
      </View>

      <Pressable style={styles.timeline} onLayout={onTimelineLayout} onPress={onTimelinePress}>
        <View style={[styles.timelineProgress, { width: `${(currentMs / durationMs) * 100}%` }]} />
        {takeoffRatio !== null && (
          <View style={[styles.timelineMarker, styles.takeoffMarker, { left: `${takeoffRatio * 100}%` }]} />
        )}
        {landingRatio !== null && (
          <View style={[styles.timelineMarker, styles.landingMarker, { left: `${landingRatio * 100}%` }]} />
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
  const [error, setError] = useState<string | null>(null);

  const format = useMemo(() => selectFpsFirstFormat(device), [device]);
  const captureFps = useMemo(() => {
    if (!format) return 60;
    return Math.min(60, format.maxFps);
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
          uri: normalizeFileUri(video.path),
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
    setCountdownStep(0);
    setRecordingElapsedMs(0);
    setStage('COUNTDOWN');
  }, []);

  const analyzeClip = useCallback(async () => {
    if (!clip) return;

    setError(null);
    setStage('ANALYZING');

    try {
      const native = await analyzeRecordedJumpVideo(clip, {
        sampleFps: 60,
        maxFrames: 360,
        minConfidence: 0.2,
      });
      const nextAnalysis = analyzeJumpLandmarks(native.frames, {
        videoDurationMs: native.videoDurationMs,
        videoFps: native.videoFps,
        sampleFps: native.sampleFps,
        personCountSummary: native.personCountSummary,
        minConfidence: 0.2,
      });

      startTransition(() => {
        setNativeResult(native);
        setAnalysis(nextAnalysis);
        setStage('RESULT_DEBUG');
      });
    } catch (analysisError) {
      setError((analysisError as Error).message);
      setStage('REVIEW');
    }
  }, [clip]);

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
        <Text style={styles.stageSubText}>{captureFps} fps capture target</Text>
      </View>
      <Text style={[styles.statusHeadline, error && styles.errorText]}>
        {statusCopy(stage, countdownStep, recordingElapsedMs, error)}
      </Text>
      {!nativeAnalysisAvailable && (
        <Text style={styles.warningText}>
          Native analyzer unavailable. Rebuild the iOS development client so the local Expo module
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
        <Text style={styles.btnText}>{analysis ? 'Re-analyze Clip' : 'Analyze Clip'}</Text>
      </TouchableOpacity>
      <TouchableOpacity style={[styles.btn, styles.btnSecondary]} onPress={resetSession}>
        <Text style={styles.btnText}>Retake</Text>
      </TouchableOpacity>
    </View>
  ) : null;
  const reviewContent = clip ? (
    <View style={styles.reviewCards}>
      <ReviewPlayer clip={clip} nativeResult={nativeResult} analysis={analysis} />

      <View style={styles.resultCard}>
        <Text style={[styles.resultHeadline, { color: summaryColor(analysis) }]}>
          {analysis?.invalidReason ? 'Invalid attempt' : analysis ? 'Measured jump' : 'Recorded clip'}
        </Text>
        <Text style={styles.resultLine}>Takeoff: {formatMs(analysis?.takeoffMs ?? null)}</Text>
        <Text style={styles.resultLine}>Landing: {formatMs(analysis?.landingMs ?? null)}</Text>
        <Text style={styles.resultLine}>Flight: {formatMs(analysis?.flightMs ?? null)}</Text>
        <Text style={styles.resultLine}>
          Height:{' '}
          {analysis?.heightCm !== null && analysis?.heightCm !== undefined
            ? `${analysis.heightCm.toFixed(1)} cm`
            : '—'}
        </Text>
        <Text style={styles.resultLine}>Quality: {analysis?.quality ?? '—'}</Text>
        <Text style={styles.resultSummary}>
          {analysis
            ? explainJumpAnalysis(analysis)
            : 'Run analysis to extract landmarks and event timings.'}
        </Text>
        {analysis?.qualityFlags && (
          <Text style={styles.resultFlags}>Flags: {analysis.qualityFlags.join(', ')}</Text>
        )}
      </View>

      {(analysis || nativeResult) && (
        <View style={styles.debugCard}>
          <Text style={styles.referenceTitle}>Debug Metrics</Text>
          <Text style={styles.referenceText}>
            Frames: {analysis?.debug.analyzedFrameCount ?? nativeResult?.frames.length ?? 0}
          </Text>
          <Text style={styles.referenceText}>
            Sample FPS: {(analysis?.debug.sampleFps ?? nativeResult?.sampleFps ?? clip.fps).toFixed(1)}
          </Text>
          <Text style={styles.referenceText}>
            Avg confidence: {(analysis?.debug.averageConfidence ?? 0).toFixed(2)}
          </Text>
          <Text style={styles.referenceText}>
            Feet visible: {((analysis?.debug.feetVisibleRatio ?? 0) * 100).toFixed(0)}%
          </Text>
          <Text style={styles.referenceText}>
            Full body visible: {((analysis?.debug.fullBodyVisibleRatio ?? 0) * 100).toFixed(0)}%
          </Text>
          <Text style={styles.referenceText}>
            Horizontal drift: {((analysis?.debug.maxHorizontalDrift ?? 0) * 100).toFixed(1)}%
          </Text>
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
                <Text style={styles.referenceText}>Rear camera. Side profile. Fixed support.</Text>
                <Text style={styles.referenceText}>Full body, feet, and floor visible throughout.</Text>
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
  timelineMarker: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 3,
    marginLeft: -1.5,
  },
  takeoffMarker: {
    backgroundColor: '#22c55e',
  },
  landingMarker: {
    backgroundColor: '#f97316',
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

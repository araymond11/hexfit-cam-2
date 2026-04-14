import AVFoundation
import ExpoModulesCore
import MLKitPoseDetectionAccurate
import MLKitVision
import Photos
import UIKit

public final class JumpVideoAnalysisModule: Module {
  public func definition() -> ModuleDefinition {
    Name("JumpVideoAnalysis")

    AsyncFunction("analyze") { (clip: [String: Any], options: [String: Any]?) throws -> [String: Any] in
      let analyzer = try JumpVideoAnalyzer(clip: clip, options: options ?? [:])
      return try analyzer.analyze()
    }
  }
}

private final class JumpVideoAnalyzer {
  private let clipURL: URL
  private let assetLocalIdentifier: String?
  private let requestedSampleFps: Double
  private let maxFramesOverride: Int?
  private let clipDurationMs: Double
  private let requestedAnalysisStartMs: Double?
  private let requestedAnalysisEndMs: Double?

  private static let defaultSampleFps = 60.0
  private static let defaultMaxFrames = 720
  private static let keypointDefinitions: [(name: String, type: PoseLandmarkType?)] = [
    ("nose", .nose),
    ("leftEyeInner", .leftEyeInner),
    ("leftEye", .leftEye),
    ("leftEyeOuter", .leftEyeOuter),
    ("rightEyeInner", .rightEyeInner),
    ("rightEye", .rightEye),
    ("rightEyeOuter", .rightEyeOuter),
    ("leftEar", .leftEar),
    ("rightEar", .rightEar),
    ("leftMouth", .mouthLeft),
    ("rightMouth", .mouthRight),
    ("leftShoulder", .leftShoulder),
    ("rightShoulder", .rightShoulder),
    ("leftElbow", .leftElbow),
    ("rightElbow", .rightElbow),
    ("leftWrist", .leftWrist),
    ("rightWrist", .rightWrist),
    ("leftPinky", .leftPinkyFinger),
    ("rightPinky", .rightPinkyFinger),
    ("leftIndex", .leftIndexFinger),
    ("rightIndex", .rightIndexFinger),
    ("leftThumb", .leftThumb),
    ("rightThumb", .rightThumb),
    ("leftHip", .leftHip),
    ("rightHip", .rightHip),
    ("leftKnee", .leftKnee),
    ("rightKnee", .rightKnee),
    ("leftAnkle", .leftAnkle),
    ("rightAnkle", .rightAnkle),
    ("leftHeel", .leftHeel),
    ("rightHeel", .rightHeel),
    ("leftFootIndex", .leftToe),
    ("rightFootIndex", .rightToe),
  ]

  /// A single source→target time mapping segment from an AVComposition.
  private struct TimeSegment {
    let sourceStartSec: Double
    let sourceDurationSec: Double
    let targetStartSec: Double
    let targetDurationSec: Double

    /// Converts a playback timestamp (in seconds) that falls within this segment
    /// to the corresponding physical (source) timestamp.
    func sourceSeconds(forTarget targetSec: Double) -> Double {
      guard targetDurationSec > 0 else { return sourceStartSec }
      let offset = targetSec - targetStartSec
      let ratio = sourceDurationSec / targetDurationSec
      return sourceStartSec + offset * ratio
    }
  }

  private struct SourceTiming {
    let playbackDurationMs: Double
    let playbackVideoFps: Double
    let playbackSampleFps: Double
    let captureDurationMs: Double
    let captureVideoFps: Double
    let captureSampleFps: Double
    let captureTimeScale: Double
    /// Per-section time mapping from an AVComposition (slow-motion).
    /// When present, this is used instead of the global captureTimeScale.
    let timeSegments: [TimeSegment]?
    let timingMode: String
    let timingConfidence: Double
    let timingTrusted: Bool
    let hasTimeSegments: Bool
    let usedOriginalAsset: Bool
    let usedPlaybackAsset: Bool
    let originalDurationMs: Double?
    let timebaseSource: String
  }

  /// Converts a playback timestamp to a physical (capture) timestamp using
  /// per-segment time mapping when available, or the global scale as fallback.
  private static func physicalTimestampMs(
    playbackMs: Double,
    segments: [TimeSegment]?,
    fallbackScale: Double
  ) -> Double {
    guard let segments = segments, !segments.isEmpty else {
      return playbackMs * fallbackScale
    }
    let playSec = playbackMs / 1000.0
    for segment in segments {
      let targetEnd = segment.targetStartSec + segment.targetDurationSec
      if playSec >= segment.targetStartSec - 0.001 && playSec <= targetEnd + 0.001 {
        return segment.sourceSeconds(forTarget: playSec) * 1000.0
      }
    }
    // Outside any known segment — use global scale
    return playbackMs * fallbackScale
  }

  init(clip: [String: Any], options: [String: Any]) throws {
    guard let uri = clip["uri"] as? String, !uri.isEmpty else {
      throw JumpVideoAnalysisError.invalidClip("Missing clip uri.")
    }

    clipURL = JumpVideoAnalyzer.parseURL(from: uri)
    assetLocalIdentifier = clip["assetId"] as? String
    requestedSampleFps = max(1, options["sampleFps"] as? Double ?? JumpVideoAnalyzer.defaultSampleFps)
    if let maxFrames = options["maxFrames"] as? Int {
      maxFramesOverride = max(1, maxFrames)
    } else {
      maxFramesOverride = nil
    }
    clipDurationMs = clip["durationMs"] as? Double ?? 0
    requestedAnalysisStartMs = JumpVideoAnalyzer.clampOptionalMilliseconds(options["analysisStartMs"])
    requestedAnalysisEndMs = JumpVideoAnalyzer.clampOptionalMilliseconds(options["analysisEndMs"])
  }

  func analyze() throws -> [String: Any] {
    let asset = AVURLAsset(url: clipURL)
    guard let track = asset.tracks(withMediaType: .video).first else {
      throw JumpVideoAnalysisError.invalidClip("No video track was found in the selected clip.")
    }

    let sourceTiming = resolveSourceTiming(playbackAsset: asset, playbackTrack: track)
    let effectiveDurationMs = sourceTiming.playbackDurationMs
    let analysisStartMs = max(0, min(requestedAnalysisStartMs ?? 0, sourceTiming.playbackDurationMs))
    let unclampedAnalysisEndMs = requestedAnalysisEndMs ?? sourceTiming.playbackDurationMs
    let analysisEndMs = max(
      analysisStartMs + 1,
      min(unclampedAnalysisEndMs, sourceTiming.playbackDurationMs)
    )

    let reader = try AVAssetReader(asset: asset)
    let analysisStartTime = CMTime(seconds: analysisStartMs / 1000.0, preferredTimescale: 600)
    let analysisEndTime = CMTime(seconds: analysisEndMs / 1000.0, preferredTimescale: 600)
    reader.timeRange = CMTimeRange(
      start: analysisStartTime,
      duration: CMTimeSubtract(analysisEndTime, analysisStartTime)
    )
    let outputSettings: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    ]
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
    output.alwaysCopiesSampleData = false

    guard reader.canAdd(output) else {
      throw JumpVideoAnalysisError.readFailed("Unable to attach a video track reader.")
    }

    reader.add(output)
    guard reader.startReading() else {
      throw JumpVideoAnalysisError.readFailed(reader.error?.localizedDescription ?? "Reader failed to start.")
    }

    let detectorOptions = AccuratePoseDetectorOptions()
    // V2 keeps offline decoding, but runs ML Kit in stream mode so sequential
    // frames benefit from temporal stabilization before JS derives toe contact.
    detectorOptions.detectorMode = .stream
    let detector = PoseDetector.poseDetector(options: detectorOptions)

    let playbackSampleFps = sourceTiming.playbackSampleFps
    let sampleIntervalMs = 1000.0 / playbackSampleFps
    let analysisDurationMs = max(analysisEndMs - analysisStartMs, sampleIntervalMs)
    let maxFrames = maxFramesOverride
      ?? max(
        JumpVideoAnalyzer.defaultMaxFrames,
        Int(ceil(analysisDurationMs / sampleIntervalMs)) + 8
      )
    let orientation = JumpVideoAnalyzer.imageOrientation(for: track.preferredTransform)

    var frames: [[String: Any]] = []
    var decodedFrameCount = 0
    var lastAcceptedTimestampMs = -Double.infinity
    var multiPersonFrames = 0
    var maxPeople = 0

    while let sampleBuffer = output.copyNextSampleBuffer() {
      let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
      let presentationTimestampMs = CMTimeGetSeconds(presentationTime) * 1000.0
      let fallbackTimestampMs =
        analysisStartMs + Double(decodedFrameCount) / max(sourceTiming.playbackVideoFps, 1) * 1000.0
      let timestampMs =
        presentationTimestampMs.isFinite && presentationTimestampMs >= 0
        ? presentationTimestampMs
        : fallbackTimestampMs
      let captureTimestampMs = JumpVideoAnalyzer.physicalTimestampMs(
        playbackMs: timestampMs,
        segments: sourceTiming.timeSegments,
        fallbackScale: sourceTiming.captureTimeScale
      )
      decodedFrameCount += 1

      if !timestampMs.isFinite || timestampMs < 0 {
        continue
      }

      if timestampMs + 0.5 < analysisStartMs {
        continue
      }

      if timestampMs > analysisEndMs + 0.5 {
        break
      }

      if timestampMs < lastAcceptedTimestampMs + sampleIntervalMs {
        continue
      }

      let frameIndex = frames.count
      let frameDictionary = try JumpVideoAnalyzer.analyzeFrame(
        sampleBuffer: sampleBuffer,
        frameIndex: frameIndex,
        timestampMs: timestampMs,
        captureTimestampMs: captureTimestampMs,
        detector: detector,
        orientation: orientation
      )

      let personCount = frameDictionary["personCount"] as? Int ?? 0
      if personCount > 1 {
        multiPersonFrames += 1
      }
      maxPeople = max(maxPeople, personCount)

      frames.append(frameDictionary)
      lastAcceptedTimestampMs = timestampMs

      if frames.count >= maxFrames {
        break
      }
    }

    if reader.status == .failed {
      throw JumpVideoAnalysisError.readFailed(reader.error?.localizedDescription ?? "Reader failed while decoding frames.")
    }

    return [
      "frames": frames,
      "videoFps": sourceTiming.captureVideoFps,
      "sampleFps": sourceTiming.captureSampleFps,
      "playbackVideoFps": sourceTiming.playbackVideoFps,
      "playbackSampleFps": sourceTiming.playbackSampleFps,
      "videoDurationMs": effectiveDurationMs,
      "playbackDurationMs": sourceTiming.playbackDurationMs,
      "captureDurationMs": sourceTiming.captureDurationMs,
      "timingMode": sourceTiming.timingMode,
      "timingConfidence": sourceTiming.timingConfidence,
      "captureTimeScale": sourceTiming.captureTimeScale,
      "hasTimeSegments": sourceTiming.hasTimeSegments,
      "usedOriginalAsset": sourceTiming.usedOriginalAsset,
      "usedPlaybackAsset": sourceTiming.usedPlaybackAsset,
      "originalDurationMs": sourceTiming.originalDurationMs as Any,
      "timebaseSource": sourceTiming.timebaseSource,
      "personCountSummary": [
        "analyzedFrames": frames.count,
        "multiPersonFrames": multiPersonFrames,
        "maxPeople": maxPeople,
      ],
    ]
  }

  private func resolveSourceTiming(playbackAsset: AVAsset, playbackTrack: AVAssetTrack) -> SourceTiming {
    let playbackDurationMs = max(
      clipDurationMs,
      JumpVideoAnalyzer.assetDurationMs(playbackAsset)
    )
    let playbackVideoFps = JumpVideoAnalyzer.estimatedTrackFps(
      track: playbackTrack,
      fallback: requestedSampleFps
    )
    let playbackSampleFps = min(requestedSampleFps, max(playbackVideoFps, 1))

    /// Extract TimeSegment array from an AVComposition's video track.
    func extractTimeSegments(from asset: AVAsset) -> [TimeSegment]? {
      guard let composition = asset as? AVComposition else { return nil }
      for track in composition.tracks(withMediaType: .video) {
        var segments: [TimeSegment] = []
        for segment in track.segments {
          let srcStart = CMTimeGetSeconds(segment.timeMapping.source.start)
          let srcDur   = CMTimeGetSeconds(segment.timeMapping.source.duration)
          let tgtStart = CMTimeGetSeconds(segment.timeMapping.target.start)
          let tgtDur   = CMTimeGetSeconds(segment.timeMapping.target.duration)
          guard srcDur > 0 && tgtDur > 0 else { continue }
          segments.append(TimeSegment(
            sourceStartSec: srcStart,
            sourceDurationSec: srcDur,
            targetStartSec: tgtStart,
            targetDurationSec: tgtDur
          ))
        }
        if !segments.isEmpty { return segments }
      }
      return nil
    }

    func sourceDurationMs(from segments: [TimeSegment]) -> Double {
      let end = segments.map { $0.sourceStartSec + $0.sourceDurationSec }.max() ?? 0
      return max(end * 1000.0, 0)
    }

    func buildTiming(
      timingMode: String,
      timingConfidence: Double,
      captureDurationMs: Double,
      captureVideoFps: Double,
      captureTimeScale: Double,
      timeSegments: [TimeSegment]?,
      usedOriginalAsset: Bool,
      originalDurationMs: Double?,
      timebaseSource: String
    ) -> SourceTiming {
      let safeCaptureTimeScale = min(max(captureTimeScale, 0.01), 1)
      let safeCaptureVideoFps = max(captureVideoFps, 1)
      let captureSampleFps = max(
        1,
        min(
          safeCaptureVideoFps,
          playbackSampleFps / max(safeCaptureTimeScale, 0.01)
        )
      )
      return SourceTiming(
        playbackDurationMs: playbackDurationMs,
        playbackVideoFps: playbackVideoFps,
        playbackSampleFps: playbackSampleFps,
        captureDurationMs: max(captureDurationMs, 0),
        captureVideoFps: safeCaptureVideoFps,
        captureSampleFps: captureSampleFps,
        captureTimeScale: safeCaptureTimeScale,
        timeSegments: timeSegments,
        timingMode: timingMode,
        timingConfidence: timingConfidence,
        timingTrusted: timingMode != "timing_ambiguous",
        hasTimeSegments: !(timeSegments?.isEmpty ?? true),
        usedOriginalAsset: usedOriginalAsset,
        usedPlaybackAsset: true,
        originalDurationMs: originalDurationMs,
        timebaseSource: timebaseSource
      )
    }

    // Fetch PHAsset once (used by multiple strategies)
    let photoAsset: PHAsset? = {
      guard let id = assetLocalIdentifier, !id.isEmpty else { return nil }
      return JumpVideoAnalyzer.photoAsset(withLocalIdentifier: id)
    }()

    let playbackSegments = extractTimeSegments(from: playbackAsset)
    if let playbackSegments, !playbackSegments.isEmpty {
      let captureDurationMs = sourceDurationMs(from: playbackSegments)
      let minRatio = playbackSegments.map { $0.sourceDurationSec / $0.targetDurationSec }.min() ?? 1
      let captureTimeScale = min(max(minRatio, 0.01), 1)
      let captureVideoFps = max(playbackVideoFps / captureTimeScale, playbackVideoFps)
      return buildTiming(
        timingMode: "segment_mapped_slow_motion",
        timingConfidence: 1,
        captureDurationMs: captureDurationMs > 0 ? captureDurationMs : playbackDurationMs * captureTimeScale,
        captureVideoFps: captureVideoFps,
        captureTimeScale: captureTimeScale,
        timeSegments: playbackSegments,
        usedOriginalAsset: false,
        originalDurationMs: nil,
        timebaseSource: "playback_asset_segments"
      )
    }

    if playbackVideoFps >= 120 {
      return buildTiming(
        timingMode: "playback_is_physical",
        timingConfidence: 0.98,
        captureDurationMs: playbackDurationMs,
        captureVideoFps: playbackVideoFps,
        captureTimeScale: 1,
        timeSegments: nil,
        usedOriginalAsset: false,
        originalDurationMs: nil,
        timebaseSource: "playback_asset_flat_hfr"
      )
    }

    // ---------------------------------------------------------------------------
    // Strategy 1: PHAsset — request both .original (for capture FPS) and
    // .current (for per-segment time mapping on slow-motion compositions).
    // ---------------------------------------------------------------------------
    if let photoAsset {
      let originalAsset = JumpVideoAnalyzer.requestVideoAsset(for: photoAsset, version: .original)
      let currentAsset  = JumpVideoAnalyzer.requestVideoAsset(for: photoAsset, version: .current)

      let currentSegments = currentAsset.flatMap { extractTimeSegments(from: $0) }
      let currentDurationMs = currentAsset.map(JumpVideoAnalyzer.assetDurationMs)
      let originalDurationMs = originalAsset.map(JumpVideoAnalyzer.assetDurationMs)
      let originalTrack = originalAsset?.tracks(withMediaType: .video).first
      let originalVideoFps = originalTrack.map {
        JumpVideoAnalyzer.estimatedTrackFps(track: $0, fallback: playbackVideoFps)
      }
      let currentMatchesPlayback =
        currentDurationMs.map { abs($0 - playbackDurationMs) <= max(120, playbackDurationMs * 0.05) } ?? false

      if let currentSegments, !currentSegments.isEmpty, currentMatchesPlayback {
        let captureDurationMs = sourceDurationMs(from: currentSegments)
        let minRatio = currentSegments.map { $0.sourceDurationSec / $0.targetDurationSec }.min() ?? 1
        let captureTimeScale = min(max(minRatio, 0.01), 1)
        let captureVideoFps = max(
          originalVideoFps ?? (playbackVideoFps / captureTimeScale),
          playbackVideoFps / captureTimeScale
        )
        return buildTiming(
          timingMode: "segment_mapped_slow_motion",
          timingConfidence: 0.99,
          captureDurationMs: captureDurationMs > 0 ? captureDurationMs : (originalDurationMs ?? (playbackDurationMs * captureTimeScale)),
          captureVideoFps: captureVideoFps,
          captureTimeScale: captureTimeScale,
          timeSegments: currentSegments,
          usedOriginalAsset: originalTrack != nil,
          originalDurationMs: originalDurationMs,
          timebaseSource: "photos_current_segments"
        )
      }

      if let originalDurationMs,
         let originalVideoFps,
         playbackVideoFps <= 60,
         originalVideoFps >= 120,
         playbackDurationMs > 0
      {
        let durationScale = originalDurationMs / playbackDurationMs
        let looksRetimed = durationScale > 0.05 && durationScale < 0.9

        if looksRetimed {
          let captureVideoFps = max(originalVideoFps, playbackVideoFps / max(durationScale, 0.01))
          return buildTiming(
            timingMode: "global_scaled_slow_motion",
            timingConfidence: 0.82,
            captureDurationMs: originalDurationMs,
            captureVideoFps: captureVideoFps,
            captureTimeScale: durationScale,
            timeSegments: nil,
            usedOriginalAsset: true,
            originalDurationMs: originalDurationMs,
            timebaseSource: "photos_original_duration_scale"
          )
        }
      }

      if let currentSegments, !currentSegments.isEmpty, !currentMatchesPlayback {
        return buildTiming(
          timingMode: "timing_ambiguous",
          timingConfidence: 0.2,
          captureDurationMs: playbackDurationMs,
          captureVideoFps: playbackVideoFps,
          captureTimeScale: 1,
          timeSegments: nil,
          usedOriginalAsset: originalTrack != nil,
          originalDurationMs: originalDurationMs,
          timebaseSource: "photos_segments_mismatch"
        )
      }
    }

    return buildTiming(
      timingMode: "playback_is_physical",
      timingConfidence: playbackVideoFps >= 60 ? 0.9 : 0.75,
      captureDurationMs: playbackDurationMs,
      captureVideoFps: playbackVideoFps,
      captureTimeScale: 1,
      timeSegments: nil,
      usedOriginalAsset: false,
      originalDurationMs: nil,
      timebaseSource: playbackVideoFps >= 60 ? "playback_asset_default" : "playback_asset_flat_standard"
    )
  }

  private static func analyzeFrame(
    sampleBuffer: CMSampleBuffer,
    frameIndex: Int,
    timestampMs: Double,
    captureTimestampMs: Double,
    detector: PoseDetector,
    orientation: UIImage.Orientation
  ) throws -> [String: Any] {
    let image = VisionImage(buffer: sampleBuffer)
    image.orientation = orientation

    let poses: [Pose]
    do {
      poses = try detector.results(in: image)
    } catch {
      return emptyFrame(frameIndex: frameIndex, timestampMs: timestampMs, personCount: 0)
    }

    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      return emptyFrame(frameIndex: frameIndex, timestampMs: timestampMs, personCount: poses.count)
    }

    let rawWidth = Double(CVPixelBufferGetWidth(pixelBuffer))
    let rawHeight = Double(CVPixelBufferGetHeight(pixelBuffer))

    let primaryPose = selectPrimaryPose(from: poses)
    let keypoints = makeKeypoints(from: primaryPose, rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation)
    let avgConfidence =
      keypoints.isEmpty
      ? 0.0
      : keypoints.reduce(0.0) { $0 + ((($1["score"] as? Double) ?? 0.0)) } / Double(keypoints.count)

    let contactAnalysis = analyzeFloorContact(
      pixelBuffer: pixelBuffer,
      pose: primaryPose,
      rawWidth: rawWidth,
      rawHeight: rawHeight,
      orientation: orientation
    )

    var result: [String: Any] = [
      "frameIndex": frameIndex,
      "timestampMs": timestampMs,
      "captureTimestampMs": captureTimestampMs,
      "keypoints": keypoints,
      "avgConfidence": avgConfidence,
      "personCount": poses.count,
    ]
    result.merge(contactAnalysis) { _, new in new }
    return result
  }

  private static func emptyFrame(
    frameIndex: Int,
    timestampMs: Double,
    personCount: Int
  ) -> [String: Any] {
    let keypoints = keypointDefinitions.map { definition in
      [
        "x": 0.0,
        "y": 0.0,
        "score": 0.0,
        "name": definition.name,
      ] as [String: Any]
    }

    return [
      "frameIndex": frameIndex,
      "timestampMs": timestampMs,
      "captureTimestampMs": timestampMs,
      "keypoints": keypoints,
      "avgConfidence": 0.0,
      "personCount": personCount,
      "floorBandY": NSNull(),
      "floorConfidence": 0.0,
      "leftFootBox": NSNull(),
      "rightFootBox": NSNull(),
      "leftFootBottomY": NSNull(),
      "rightFootBottomY": NSNull(),
      "leftContactScore": NSNull(),
      "rightContactScore": NSNull(),
    ]
  }

  private static func clampOptionalMilliseconds(_ value: Any?) -> Double? {
    guard let number = value as? NSNumber else {
      return nil
    }

    let milliseconds = number.doubleValue
    guard milliseconds.isFinite else {
      return nil
    }

    return max(0, milliseconds)
  }

  private static func photoAsset(withLocalIdentifier localIdentifier: String) -> PHAsset? {
    let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil)
    return fetchResult.firstObject
  }

  private static func estimatedTrackFps(track: AVAssetTrack, fallback: Double) -> Double {
    let nominalFps = track.nominalFrameRate > 0 ? Double(track.nominalFrameRate) : fallback
    let minDur = track.minFrameDuration
    let minFrameFps = (minDur.value > 0 && minDur.timescale > 0)
      ? Double(minDur.timescale) / Double(minDur.value)
      : 0
    return max(nominalFps, minFrameFps, 1)
  }

  private static func assetDurationMs(_ asset: AVAsset) -> Double {
    let durationMs = CMTimeGetSeconds(asset.duration) * 1000.0
    return durationMs.isFinite ? max(durationMs, 0) : 0
  }

  private static func requestVideoAsset(
    for asset: PHAsset,
    version: PHVideoRequestOptionsVersion
  ) -> AVAsset? {
    let options = PHVideoRequestOptions()
    options.version = version
    options.deliveryMode = .highQualityFormat
    options.isNetworkAccessAllowed = true

    let semaphore = DispatchSemaphore(value: 0)
    var resolvedAsset: AVAsset?

    PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
      resolvedAsset = avAsset
      semaphore.signal()
    }

    _ = semaphore.wait(timeout: .now() + 30)
    return resolvedAsset
  }

  private static func selectPrimaryPose(from poses: [Pose]) -> Pose? {
    guard !poses.isEmpty else {
      return nil
    }

    return poses.max { lhs, rhs in
      poseScore(lhs) < poseScore(rhs)
    }
  }

  private static func poseScore(_ pose: Pose) -> Double {
    let emphasis: [PoseLandmarkType] = [
      .nose,
      .leftShoulder,
      .rightShoulder,
      .leftHip,
      .rightHip,
      .leftKnee,
      .rightKnee,
      .leftAnkle,
      .rightAnkle,
      .leftToe,
      .rightToe,
    ]

    let scores = emphasis.map { Double(pose.landmark(ofType: $0).inFrameLikelihood) }
    return scores.reduce(0, +) / Double(scores.count)
  }

  /// Converts a raw pixel-buffer coordinate to normalised display-space [0,1].
  ///
  /// ML Kit always returns landmark positions in the raw pixel-buffer coordinate
  /// system regardless of the VisionImage orientation hint.  We must apply the
  /// inverse of the display rotation so that (x=0,y=0) is the top-left corner
  /// of the frame as seen by the user.
  ///
  /// Rotation conventions (UIImage.Orientation):
  ///   .right  – buffer stored landscape, displayed portrait (home button right /
  ///             standard iPhone back-camera portrait recording).
  ///             Raw X axis → display Y (inverted), raw Y axis → display X.
  ///   .left   – buffer stored landscape, displayed portrait (home button left).
  ///             Raw X axis → display Y, raw Y axis → display X (inverted).
  ///   .down   – buffer stored upside-down.
  ///   .up     – buffer already in display orientation (no rotation needed).
  private static func normalizeToDisplaySpace(
    rawX: Double,
    rawY: Double,
    rawWidth: Double,
    rawHeight: Double,
    orientation: UIImage.Orientation
  ) -> (x: Double, y: Double) {
    let w = max(rawWidth, 1)
    let h = max(rawHeight, 1)
    // Rotation conventions verified against CGAffineTransform.preferredTransform:
    //
    // .left  → preferredTransform (0,-1, 1,0, 0,W): display_x = rawY/H, display_y = (W-rawX)/W
    //   iPhone back-camera portrait (most common slow-mo case).
    //   Raw right-column (high rawX) = scene top = small display_y.
    //
    // .right → preferredTransform (0, 1,-1,0, H,0): display_x = (H-rawY)/H, display_y = rawX/W
    //   iPhone front-camera portrait or landscape-right recording.
    //   Raw left-column (low rawX) = scene top = small display_y.
    switch orientation {
    case .left, .leftMirrored:
      return (rawY / h, (w - rawX) / w)
    case .right, .rightMirrored:
      return ((h - rawY) / h, rawX / w)
    case .down, .downMirrored:
      return ((w - rawX) / w, (h - rawY) / h)
    default:
      // .up / .upMirrored – no rotation needed
      return (rawX / w, rawY / h)
    }
  }

  // MARK: - Display-to-Raw Coordinate Conversion (inverse of normalizeToDisplaySpace)

  private static func displayToRawSpace(
    displayX: Double,
    displayY: Double,
    rawWidth: Double,
    rawHeight: Double,
    orientation: UIImage.Orientation
  ) -> (x: Double, y: Double) {
    let w = max(rawWidth, 1)
    let h = max(rawHeight, 1)
    switch orientation {
    case .left, .leftMirrored:
      return (x: w * (1.0 - displayY), y: h * displayX)
    case .right, .rightMirrored:
      return (x: w * displayY, y: h * (1.0 - displayX))
    case .down, .downMirrored:
      return (x: w * (1.0 - displayX), y: h * (1.0 - displayY))
    default:
      return (x: w * displayX, y: h * displayY)
    }
  }

  // MARK: - Pixel-Based Floor Detection & Contact Analysis

  /// Samples average luminance in a small patch around the given raw-buffer coordinate.
  private static func sampleLuminance(
    baseAddress: UnsafeRawPointer,
    bytesPerRow: Int,
    bufferWidth: Int,
    bufferHeight: Int,
    rawX: Int,
    rawY: Int,
    radius: Int = 2
  ) -> Double {
    var sum = 0.0
    var count = 0
    for dy in -radius...radius {
      let y = rawY + dy
      guard y >= 0, y < bufferHeight else { continue }
      for dx in -radius...radius {
        let x = rawX + dx
        guard x >= 0, x < bufferWidth else { continue }
        let offset = y * bytesPerRow + x * 4
        let b = Double(baseAddress.load(fromByteOffset: offset, as: UInt8.self))
        let g = Double(baseAddress.load(fromByteOffset: offset + 1, as: UInt8.self))
        let r = Double(baseAddress.load(fromByteOffset: offset + 2, as: UInt8.self))
        sum += 0.299 * r + 0.587 * g + 0.114 * b
        count += 1
      }
    }
    return count > 0 ? sum / Double(count) : 128.0
  }

  /// Detects the floor band by scanning for strong horizontal edges near the foot positions.
  /// Returns the estimated floor Y in normalised display coordinates and a confidence score.
  private static func detectFloorBand(
    baseAddress: UnsafeRawPointer,
    bytesPerRow: Int,
    bufferWidth: Int,
    bufferHeight: Int,
    rawWidth: Double,
    rawHeight: Double,
    orientation: UIImage.Orientation,
    footPositions: [(x: Double, y: Double)]
  ) -> (y: Double, confidence: Double) {
    guard !footPositions.isEmpty else {
      return (0.9, 0.0)
    }

    let maxFootY = footPositions.map(\.y).max() ?? 0.9
    let avgFootX = footPositions.map(\.x).reduce(0, +) / Double(footPositions.count)
    let scanStartY = max(0, maxFootY - 0.04)
    let scanEndY = min(1, maxFootY + 0.04)
    let numRows = 24
    let numCols = 7
    let step = (scanEndY - scanStartY) / Double(numRows)

    var bestGradient = 0.0
    var bestY = maxFootY

    for row in 0..<numRows {
      let displayY = scanStartY + Double(row) * step
      let displayYNext = displayY + step
      var totalGradient = 0.0

      for col in 0..<numCols {
        let ct = Double(col) / Double(numCols - 1)
        let displayX = max(0.05, min(0.95, avgFootX - 0.12 + ct * 0.24))

        let (rx1, ry1) = displayToRawSpace(
          displayX: displayX, displayY: displayY,
          rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation
        )
        let (rx2, ry2) = displayToRawSpace(
          displayX: displayX, displayY: displayYNext,
          rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation
        )

        let l1 = sampleLuminance(
          baseAddress: baseAddress, bytesPerRow: bytesPerRow,
          bufferWidth: bufferWidth, bufferHeight: bufferHeight,
          rawX: Int(rx1), rawY: Int(ry1), radius: 2
        )
        let l2 = sampleLuminance(
          baseAddress: baseAddress, bytesPerRow: bytesPerRow,
          bufferWidth: bufferWidth, bufferHeight: bufferHeight,
          rawX: Int(rx2), rawY: Int(ry2), radius: 2
        )

        totalGradient += abs(l1 - l2)
      }

      let avgGradient = totalGradient / Double(numCols)
      if avgGradient > bestGradient {
        bestGradient = avgGradient
        bestY = displayY + step / 2
      }
    }

    let confidence = min(1.0, bestGradient / 25.0)
    return (bestY, confidence)
  }

  /// Scans pixels vertically at the foot position to find the visual bottom of the foot
  /// and computes a contact score based on proximity to the floor band.
  private static func computeFootContact(
    baseAddress: UnsafeRawPointer,
    bytesPerRow: Int,
    bufferWidth: Int,
    bufferHeight: Int,
    rawWidth: Double,
    rawHeight: Double,
    orientation: UIImage.Orientation,
    footDisplayX: Double,
    footDisplayBottomY: Double,
    floorDisplayY: Double
  ) -> (visualBottomY: Double, score: Double) {
    let searchStartY = max(0, footDisplayBottomY - 0.025)
    let searchEndY = min(1.0, floorDisplayY + 0.015)
    guard searchEndY > searchStartY else {
      return (footDisplayBottomY, 0.5)
    }

    let numSamples = 16
    let step = (searchEndY - searchStartY) / Double(numSamples - 1)

    var luminances: [Double] = []
    for i in 0..<numSamples {
      let displayY = searchStartY + Double(i) * step
      let (rx, ry) = displayToRawSpace(
        displayX: footDisplayX, displayY: displayY,
        rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation
      )
      let lum = sampleLuminance(
        baseAddress: baseAddress, bytesPerRow: bytesPerRow,
        bufferWidth: bufferWidth, bufferHeight: bufferHeight,
        rawX: Int(rx), rawY: Int(ry), radius: 3
      )
      luminances.append(lum)
    }

    // Find the maximum gradient (strongest edge = where foot ends)
    var maxGrad = 0.0
    var maxIdx = numSamples - 1
    for i in 1..<numSamples {
      let grad = abs(luminances[i] - luminances[i - 1])
      if grad > maxGrad {
        maxGrad = grad
        maxIdx = i
      }
    }

    let bottomT = Double(maxIdx) / Double(numSamples - 1)
    let visualBottomY = searchStartY + bottomT * (searchEndY - searchStartY)

    // Contact score: proximity of visual foot bottom to floor band
    let distance = abs(visualBottomY - floorDisplayY)
    let proximityScore = max(0, 1.0 - distance / 0.025)
    let edgeScore = min(1.0, maxGrad / 20.0)
    let score = proximityScore * 0.7 + edgeScore * 0.3

    return (visualBottomY, score)
  }

  /// Builds a normalised bounding box around a foot from its landmarks.
  private static func buildFootBox(
    toeDisplay: (x: Double, y: Double),
    heelDisplay: (x: Double, y: Double),
    ankleDisplay: (x: Double, y: Double),
    likelihood: Double
  ) -> [String: Any]? {
    guard likelihood > 0.3 else { return nil }

    let minX = min(toeDisplay.x, heelDisplay.x, ankleDisplay.x)
    let maxX = max(toeDisplay.x, heelDisplay.x, ankleDisplay.x)
    let minY = min(toeDisplay.y, heelDisplay.y, ankleDisplay.y)
    let maxY = max(toeDisplay.y, heelDisplay.y, ankleDisplay.y)

    let padX = max((maxX - minX) * 0.3, 0.01)
    let padY = max((maxY - minY) * 0.15, 0.005)

    let x = max(0, minX - padX)
    let y = max(0, minY - padY)
    let w = min(1 - x, maxX - minX + 2 * padX)
    let h = min(1 - y, maxY - minY + 2 * padY)

    return ["x": x, "y": y, "width": w, "height": h]
  }

  /// Orchestrates pixel-based floor detection and per-foot contact scoring.
  /// Returns a dictionary with floorBandY, floorConfidence, foot boxes, and contact scores.
  private static func analyzeFloorContact(
    pixelBuffer: CVPixelBuffer,
    pose: Pose?,
    rawWidth: Double,
    rawHeight: Double,
    orientation: UIImage.Orientation
  ) -> [String: Any] {
    let nullResult: [String: Any] = [
      "floorBandY": NSNull(),
      "floorConfidence": 0.0,
      "leftFootBox": NSNull(),
      "rightFootBox": NSNull(),
      "leftFootBottomY": NSNull(),
      "rightFootBottomY": NSNull(),
      "leftContactScore": NSNull(),
      "rightContactScore": NSNull(),
    ]

    guard let pose = pose else { return nullResult }

    let leftToe = pose.landmark(ofType: .leftToe)
    let leftHeel = pose.landmark(ofType: .leftHeel)
    let leftAnkle = pose.landmark(ofType: .leftAnkle)
    let rightToe = pose.landmark(ofType: .rightToe)
    let rightHeel = pose.landmark(ofType: .rightHeel)
    let rightAnkle = pose.landmark(ofType: .rightAnkle)

    let leftToeDisp = normalizeToDisplaySpace(
      rawX: Double(leftToe.position.x), rawY: Double(leftToe.position.y),
      rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation
    )
    let leftHeelDisp = normalizeToDisplaySpace(
      rawX: Double(leftHeel.position.x), rawY: Double(leftHeel.position.y),
      rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation
    )
    let leftAnkleDisp = normalizeToDisplaySpace(
      rawX: Double(leftAnkle.position.x), rawY: Double(leftAnkle.position.y),
      rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation
    )
    let rightToeDisp = normalizeToDisplaySpace(
      rawX: Double(rightToe.position.x), rawY: Double(rightToe.position.y),
      rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation
    )
    let rightHeelDisp = normalizeToDisplaySpace(
      rawX: Double(rightHeel.position.x), rawY: Double(rightHeel.position.y),
      rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation
    )
    let rightAnkleDisp = normalizeToDisplaySpace(
      rawX: Double(rightAnkle.position.x), rawY: Double(rightAnkle.position.y),
      rawWidth: rawWidth, rawHeight: rawHeight, orientation: orientation
    )

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
      return nullResult
    }

    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let bufferWidth = Int(rawWidth)
    let bufferHeight = Int(rawHeight)

    let footPositions = [
      (x: leftToeDisp.x, y: leftToeDisp.y),
      (x: leftHeelDisp.x, y: leftHeelDisp.y),
      (x: rightToeDisp.x, y: rightToeDisp.y),
      (x: rightHeelDisp.x, y: rightHeelDisp.y),
    ]

    let floor = detectFloorBand(
      baseAddress: baseAddress, bytesPerRow: bytesPerRow,
      bufferWidth: bufferWidth, bufferHeight: bufferHeight,
      rawWidth: rawWidth, rawHeight: rawHeight,
      orientation: orientation, footPositions: footPositions
    )

    let leftLikelihood = Double(min(
      leftToe.inFrameLikelihood, leftHeel.inFrameLikelihood, leftAnkle.inFrameLikelihood
    ))
    let rightLikelihood = Double(min(
      rightToe.inFrameLikelihood, rightHeel.inFrameLikelihood, rightAnkle.inFrameLikelihood
    ))

    let leftBox = buildFootBox(
      toeDisplay: leftToeDisp, heelDisplay: leftHeelDisp,
      ankleDisplay: leftAnkleDisp, likelihood: leftLikelihood
    )
    let rightBox = buildFootBox(
      toeDisplay: rightToeDisp, heelDisplay: rightHeelDisp,
      ankleDisplay: rightAnkleDisp, likelihood: rightLikelihood
    )

    let leftBottomY = max(leftToeDisp.y, leftHeelDisp.y)
    let rightBottomY = max(rightToeDisp.y, rightHeelDisp.y)

    var result: [String: Any] = [
      "floorBandY": floor.y,
      "floorConfidence": floor.confidence,
    ]

    result["leftFootBox"] = leftBox as Any? ?? NSNull()
    result["rightFootBox"] = rightBox as Any? ?? NSNull()

    if leftLikelihood > 0.3 {
      let leftContact = computeFootContact(
        baseAddress: baseAddress, bytesPerRow: bytesPerRow,
        bufferWidth: bufferWidth, bufferHeight: bufferHeight,
        rawWidth: rawWidth, rawHeight: rawHeight,
        orientation: orientation,
        footDisplayX: leftToeDisp.x,
        footDisplayBottomY: leftBottomY,
        floorDisplayY: floor.y
      )
      result["leftFootBottomY"] = leftContact.visualBottomY
      result["leftContactScore"] = leftContact.score
    } else {
      result["leftFootBottomY"] = NSNull()
      result["leftContactScore"] = NSNull()
    }

    if rightLikelihood > 0.3 {
      let rightContact = computeFootContact(
        baseAddress: baseAddress, bytesPerRow: bytesPerRow,
        bufferWidth: bufferWidth, bufferHeight: bufferHeight,
        rawWidth: rawWidth, rawHeight: rawHeight,
        orientation: orientation,
        footDisplayX: rightToeDisp.x,
        footDisplayBottomY: rightBottomY,
        floorDisplayY: floor.y
      )
      result["rightFootBottomY"] = rightContact.visualBottomY
      result["rightContactScore"] = rightContact.score
    } else {
      result["rightFootBottomY"] = NSNull()
      result["rightContactScore"] = NSNull()
    }

    return result
  }

  private static func makeKeypoints(
    from pose: Pose?,
    rawWidth: Double,
    rawHeight: Double,
    orientation: UIImage.Orientation
  ) -> [[String: Any]] {
    guard let pose else {
      return keypointDefinitions.map { definition in
        [
          "x": 0.0,
          "y": 0.0,
          "score": 0.0,
          "name": definition.name,
        ] as [String: Any]
      }
    }

    return keypointDefinitions.map { definition in
      guard let type = definition.type else {
        return [
          "x": 0.0,
          "y": 0.0,
          "score": 0.0,
          "name": definition.name,
        ] as [String: Any]
      }

      let landmark = pose.landmark(ofType: type)
      let point = landmark.position
      let (normX, normY) = normalizeToDisplaySpace(
        rawX: Double(point.x),
        rawY: Double(point.y),
        rawWidth: rawWidth,
        rawHeight: rawHeight,
        orientation: orientation
      )
      return [
        "x": normX,
        "y": normY,
        "score": Double(landmark.inFrameLikelihood),
        "name": definition.name,
      ] as [String: Any]
    }
  }

  private static func imageOrientation(for transform: CGAffineTransform) -> UIImage.Orientation {
    if approximately(transform.a, 0) &&
      approximately(transform.b, 1) &&
      approximately(transform.c, -1) &&
      approximately(transform.d, 0)
    {
      return .right
    }

    if approximately(transform.a, 0) &&
      approximately(transform.b, -1) &&
      approximately(transform.c, 1) &&
      approximately(transform.d, 0)
    {
      return .left
    }

    if approximately(transform.a, 1) &&
      approximately(transform.b, 0) &&
      approximately(transform.c, 0) &&
      approximately(transform.d, 1)
    {
      return .up
    }

    if approximately(transform.a, -1) &&
      approximately(transform.b, 0) &&
      approximately(transform.c, 0) &&
      approximately(transform.d, -1)
    {
      return .down
    }

    return .up
  }

  private static func approximately(_ lhs: CGFloat, _ rhs: CGFloat, tolerance: CGFloat = 0.001) -> Bool {
    abs(lhs - rhs) <= tolerance
  }

  private static func parseURL(from uri: String) -> URL {
    if let url = URL(string: uri), url.scheme != nil {
      return url
    }
    return URL(fileURLWithPath: uri)
  }
}

private enum JumpVideoAnalysisError: LocalizedError {
  case invalidClip(String)
  case readFailed(String)

  var errorDescription: String? {
    switch self {
    case let .invalidClip(message):
      return message
    case let .readFailed(message):
      return message
    }
  }
}

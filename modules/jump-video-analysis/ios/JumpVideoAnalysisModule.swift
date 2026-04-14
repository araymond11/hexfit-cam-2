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

  private struct SourceTiming {
    let playbackDurationMs: Double
    let playbackVideoFps: Double
    let playbackSampleFps: Double
    let captureDurationMs: Double
    let captureVideoFps: Double
    let captureSampleFps: Double
    let captureTimeScale: Double
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
    detectorOptions.detectorMode = .singleImage
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
      let captureTimestampMs = timestampMs * sourceTiming.captureTimeScale
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

    guard
      let assetLocalIdentifier,
      !assetLocalIdentifier.isEmpty,
      let photoAsset = JumpVideoAnalyzer.photoAsset(withLocalIdentifier: assetLocalIdentifier),
      let originalAsset = JumpVideoAnalyzer.requestVideoAsset(for: photoAsset, version: .original),
      let originalTrack = originalAsset.tracks(withMediaType: .video).first
    else {
      return SourceTiming(
        playbackDurationMs: playbackDurationMs,
        playbackVideoFps: playbackVideoFps,
        playbackSampleFps: playbackSampleFps,
        captureDurationMs: playbackDurationMs,
        captureVideoFps: playbackVideoFps,
        captureSampleFps: playbackSampleFps,
        captureTimeScale: 1
      )
    }

    let originalDurationMs = JumpVideoAnalyzer.assetDurationMs(originalAsset)
    let durationScale =
      playbackDurationMs > 0 && originalDurationMs > 0
      ? originalDurationMs / playbackDurationMs
      : 1
    let captureTimeScale = min(max(durationScale, 0.01), 1)
    let originalVideoFps = JumpVideoAnalyzer.estimatedTrackFps(
      track: originalTrack,
      fallback: playbackVideoFps / max(captureTimeScale, 0.01)
    )
    let derivedCaptureSampleFps = playbackSampleFps / max(captureTimeScale, 0.01)
    let captureVideoFps = max(originalVideoFps, playbackVideoFps / max(captureTimeScale, 0.01))
    let captureSampleFps = max(1, min(captureVideoFps, derivedCaptureSampleFps))

    return SourceTiming(
      playbackDurationMs: playbackDurationMs,
      playbackVideoFps: playbackVideoFps,
      playbackSampleFps: playbackSampleFps,
      captureDurationMs: originalDurationMs > 0 ? originalDurationMs : playbackDurationMs,
      captureVideoFps: captureVideoFps,
      captureSampleFps: captureSampleFps,
      captureTimeScale: captureTimeScale
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

    return [
      "frameIndex": frameIndex,
      "timestampMs": timestampMs,
      "captureTimestampMs": captureTimestampMs,
      "keypoints": keypoints,
      "avgConfidence": avgConfidence,
      "personCount": poses.count,
    ]
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

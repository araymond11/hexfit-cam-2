import AVFoundation
import ExpoModulesCore
import ImageIO
import Vision

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
  private let requestedSampleFps: Double
  private let maxFrames: Int
  private let clipDurationMs: Double

  init(clip: [String: Any], options: [String: Any]) throws {
    guard let uri = clip["uri"] as? String, !uri.isEmpty else {
      throw JumpVideoAnalysisError.invalidClip("Missing clip uri.")
    }

    self.clipURL = JumpVideoAnalyzer.parseURL(from: uri)
    self.requestedSampleFps = max(1, options["sampleFps"] as? Double ?? 60)
    self.maxFrames = max(1, options["maxFrames"] as? Int ?? 360)
    self.clipDurationMs = clip["durationMs"] as? Double ?? 0
  }

  func analyze() throws -> [String: Any] {
    let asset = AVURLAsset(url: clipURL)
    guard let track = asset.tracks(withMediaType: .video).first else {
      throw JumpVideoAnalysisError.invalidClip("No video track was found in the recorded clip.")
    }

    let reader = try AVAssetReader(asset: asset)
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

    let videoFps = track.nominalFrameRate > 0 ? Double(track.nominalFrameRate) : requestedSampleFps
    let sampleFps = min(requestedSampleFps, max(videoFps, 1))
    let sampleIntervalMs = 1000.0 / sampleFps
    let orientation = JumpVideoAnalyzer.videoOrientation(for: track.preferredTransform)
    let request = VNDetectHumanBodyPoseRequest()
    request.revision = VNDetectHumanBodyPoseRequestRevision1

    var analyzedFrames = 0
    var multiPersonFrames = 0
    var maxPeople = 0
    var frames: [[String: Any]] = []
    var lastAcceptedTimestampMs = -Double.infinity

    while let sampleBuffer = output.copyNextSampleBuffer() {
      let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
      let timestampMs = CMTimeGetSeconds(timestamp) * 1000
      if !timestampMs.isFinite || timestampMs < 0 {
        continue
      }

      if timestampMs < lastAcceptedTimestampMs + sampleIntervalMs {
        continue
      }

      guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        continue
      }

      let handler = VNImageRequestHandler(cvPixelBuffer: imageBuffer, orientation: orientation, options: [:])
      do {
        try handler.perform([request])
      } catch {
        throw JumpVideoAnalysisError.analysisFailed(error.localizedDescription)
      }

      let observations = request.results ?? []
      let personCount = observations.count
      if personCount > 1 {
        multiPersonFrames += 1
      }
      maxPeople = max(maxPeople, personCount)

      let framePayload = JumpVideoAnalyzer.framePayload(
        index: frames.count,
        timestampMs: timestampMs,
        observation: observations.first,
        personCount: personCount
      )

      frames.append(framePayload)
      lastAcceptedTimestampMs = timestampMs
      analyzedFrames += 1

      if analyzedFrames >= maxFrames {
        break
      }
    }

    if reader.status == .failed {
      throw JumpVideoAnalysisError.readFailed(reader.error?.localizedDescription ?? "Reader failed while decoding frames.")
    }

    let detectedDurationMs = CMTimeGetSeconds(asset.duration) * 1000

    return [
      "frames": frames,
      "videoFps": videoFps,
      "sampleFps": sampleFps,
      "videoDurationMs": max(clipDurationMs, detectedDurationMs.isFinite ? detectedDurationMs : 0),
      "personCountSummary": [
        "analyzedFrames": analyzedFrames,
        "multiPersonFrames": multiPersonFrames,
        "maxPeople": maxPeople,
      ],
    ]
  }

  private static func parseURL(from uri: String) -> URL {
    if uri.hasPrefix("file://"), let url = URL(string: uri) {
      return url
    }
    return URL(fileURLWithPath: uri)
  }

  private static func videoOrientation(for transform: CGAffineTransform) -> CGImagePropertyOrientation {
    if transform.a == 0 && transform.b == 1 && transform.c == -1 && transform.d == 0 {
      return .right
    }
    if transform.a == 0 && transform.b == -1 && transform.c == 1 && transform.d == 0 {
      return .left
    }
    if transform.a == 1 && transform.b == 0 && transform.c == 0 && transform.d == 1 {
      return .up
    }
    if transform.a == -1 && transform.b == 0 && transform.c == 0 && transform.d == -1 {
      return .down
    }
    return .up
  }

  private static func framePayload(
    index: Int,
    timestampMs: Double,
    observation: VNHumanBodyPoseObservation?,
    personCount: Int
  ) -> [String: Any] {
    let recognized = recognizedPoints(from: observation)
    let keypoints = orderedKeypoints(recognized: recognized)
    let averageConfidence = averageScore(keypoints: keypoints)

    return [
      "frameIndex": index,
      "timestampMs": timestampMs,
      "keypoints": keypoints,
      "avgConfidence": averageConfidence,
      "personCount": personCount,
    ]
  }

  private static func recognizedPoints(
    from observation: VNHumanBodyPoseObservation?
  ) -> [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint] {
    guard let observation else {
      return [:]
    }

    do {
      return try observation.recognizedPoints(.all)
    } catch {
      return [:]
    }
  }

  private static func orderedKeypoints(
    recognized: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]
  ) -> [[String: Any]] {
    let order: [(String, VNHumanBodyPoseObservation.JointName)] = [
      ("nose", .nose),
      ("leftEye", .leftEye),
      ("rightEye", .rightEye),
      ("leftEar", .leftEar),
      ("rightEar", .rightEar),
      ("leftShoulder", .leftShoulder),
      ("rightShoulder", .rightShoulder),
      ("leftElbow", .leftElbow),
      ("rightElbow", .rightElbow),
      ("leftWrist", .leftWrist),
      ("rightWrist", .rightWrist),
      ("leftHip", .leftHip),
      ("rightHip", .rightHip),
      ("leftKnee", .leftKnee),
      ("rightKnee", .rightKnee),
      ("leftAnkle", .leftAnkle),
      ("rightAnkle", .rightAnkle),
    ]

    return order.map { name, joint in
      guard let point = recognized[joint] else {
        return [
          "name": name,
          "x": 0,
          "y": 0,
          "score": 0,
        ]
      }

      return [
        "name": name,
        "x": Double(point.location.x),
        "y": 1.0 - Double(point.location.y),
        "score": Double(point.confidence),
      ]
    }
  }

  private static func averageScore(keypoints: [[String: Any]]) -> Double {
    let scores = keypoints.compactMap { $0["score"] as? Double }.filter { $0 > 0 }
    guard !scores.isEmpty else {
      return 0
    }
    return scores.reduce(0, +) / Double(scores.count)
  }
}

private enum JumpVideoAnalysisError: Error, LocalizedError {
  case invalidClip(String)
  case readFailed(String)
  case analysisFailed(String)

  var errorDescription: String? {
    switch self {
    case .invalidClip(let message), .readFailed(let message), .analysisFailed(let message):
      return message
    }
  }
}

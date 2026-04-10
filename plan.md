# POC Plan: JS-Only Airtime Detector (RNVC Worklet, No Native Code)

## Summary
Rebuild the feature as a **pure airtime detector**:
- Measure `t` (airborne duration) from camera frame timestamps.
- Compute height only with `h = (g*t^2)/8`, `g = 9.8066`.
- No skeleton dependency in v1 POC.
- No unit tests, no feature flags (per your request).

Chosen decisions:
- Detector: **Worklet Motion ROI**
- Capture: **FPS-first**
- Skeleton: **hidden for POC**

Expected airtime references (for user education/sanity):
- `0.40s` ≈ `20 cm`
- `0.50s` ≈ `31 cm`
- `0.57s` ≈ `40 cm`
- `0.64s` ≈ `50 cm`
- `<0.30s` generally tiny hop or false timing

## Public API / Type Changes
## `utils/jumpCalc.ts`
- Keep:
  - `heightFromFlightTime(flightTimeMs)`
- Update result type to airtime-first:
  - `airtimeMs: number`
  - `airtimeSec: number`
  - `heightCm: number`
  - `quality: 'LOW' | 'MEDIUM' | 'HIGH'`
  - `qualityFlags?: string[]`
- Remove displacement-based result fields from active POC path.

## New hook
Create:
- `[hooks/useAirtimePoc.ts](/Users/aray/Documents/GitHub/hexfit-cam-2/hooks/useAirtimePoc.ts)`

Return shape:
- `isReady`
- `status: 'IDLE' | 'CALIBRATING' | 'READY' | 'AIRBORNE' | 'LANDED' | 'INVALID_SETUP'`
- `guidance: string`
- `checklist: { feetVisible, floorVisible, centered, lightingOk, cameraStable }`
- `result: { airtimeMs, airtimeSec, heightCm, quality, qualityFlags } | null`
- `startCalibration()`
- `reset()`
- `frameProcessor`

## Implementation Approach (Non-Native, Performance-Optimized)
## 1) Camera format strategy (FPS-first)
In `[app/(tabs)/jump-detector.tsx](/Users/aray/Documents/GitHub/hexfit-cam-2/app/(tabs)/jump-detector.tsx)`:
- Prefer formats in this order:
  1. `1080p @ 60fps`
  2. `720p @ 60fps`
  3. highest available >= `30fps`
- Do not force max resolution if it drops fps; airtime precision needs frame cadence.

## 2) Worklet-only frame processing
In `frameProcessor`:
- Keep all heavy computation in worklet runtime.
- Do not send pixel buffers to JS.
- Send only small event payloads to JS on:
  - status change
  - guidance change
  - jump result ready

Preprocessing:
- Convert frame to small grayscale analysis plane (e.g., `160x120`) in worklet.
- Orientation/stride safe sampling (reuse existing robust mapping approach).
- ROI = bottom-center region around feet and floor under user guide rectangle.

## 3) Calibration phase (1.0–1.5s)
Collect stable baseline over N frames:
- `floorBandY` (expected contact line)
- baseline contact score
- baseline noise/lighting stats
- centered occupancy stats

Calibration pass criteria:
- subject centered enough
- feet ROI has sufficient contrast
- floor band visible
- frame-to-frame camera motion low

If fail:
- set `INVALID_SETUP` with one highest-priority guidance reason.

## 4) Airtime detection state machine (worklet)
Signals:
- `contactScore`: occupancy near floor contact band
- `liftScore`: upward displacement of lowest foreground pixels from baseline
- `stabilityScore`: local motion jitter

Transitions:
- `READY -> AIRBORNE` when:
  - `contactScore` drops below takeoff threshold
  - `liftScore` exceeds threshold
  - confirmed for 2 consecutive frames
- `AIRBORNE -> LANDED` when:
  - `contactScore` returns above landing threshold
  - confirmed for 3 consecutive frames

Timing:
- Use frame timestamp (`frame.timestamp`) for takeoff/landing.
- `airtimeMs = landingTs - takeoffTs`
- Clamp acceptance window: `180ms <= airtimeMs <= 900ms`
- Apply cooldown `500ms` after landing to stop re-trigger flicker.

## 5) Quality scoring
Derive `quality` from:
- fps stability
- setup quality (lighting/floor/feet visibility)
- detection confidence margin at takeoff/landing
- number of dropped/unknown frames during airborne

Quality flags examples:
- `LOW_LIGHT`
- `BACKLIT`
- `LOW_FPS`
- `WEAK_FEET_CONTRAST`
- `UNSTABLE_CAMERA`

## UX Plan (User-Friendly Feedback)
## Setup card before jump
Checklist rows:
- `You are centered`
- `Feet clearly visible`
- `Floor visible below feet`
- `Lighting is good`
- `Camera is stable`

Guidance priority (single message):
1. `Move back so full body and floor are visible`
2. `Tilt camera slightly down`
3. `Add front light / avoid window behind you`
4. `Increase contrast with floor (shoes/socks/background)`

## Jump flow
- `Calibrate` -> `READY` with green “Jump now”.
- On takeoff: badge `AIRBORNE`.
- On landing confirmation: show result card immediately.

## Result card
Primary:
- `Airborne Time: X.XXX s`
Secondary:
- `Derived Height: YY.Y cm`
- `Quality: LOW/MEDIUM/HIGH`

If rejected (too short/uncertain):
- `Jump not captured reliably. Try again.` + specific reason.

## File Change Plan
1. Add `[hooks/useAirtimePoc.ts](/Users/aray/Documents/GitHub/hexfit-cam-2/hooks/useAirtimePoc.ts)`.
2. Update `[app/(tabs)/jump-detector.tsx](/Users/aray/Documents/GitHub/hexfit-cam-2/app/(tabs)/jump-detector.tsx):
- use new hook
- remove skeleton drawing for POC
- add checklist + improved guidance UI
- airtime-first result card
3. Update `[utils/jumpCalc.ts](/Users/aray/Documents/GitHub/hexfit-cam-2/utils/jumpCalc.ts)` types to airtime-first result.
4. Keep old `useJumpDetection` as legacy file or remove usage entirely.

## Manual Validation (No Unit Tests)
Run manual POC matrix:
1. Good lighting, plain background, 10 jumps.
2. Backlit window condition, 10 jumps.
3. Dark socks + dark floor, 10 jumps.
4. Different camera distances (2.5m, 3m, 4m).
5. Different fps-capable devices/formats.

POC acceptance criteria:
- In good setup: >=80% jumps produce a plausible result.
- No rapid `READY <-> AIRBORNE` flicker loops.
- Typical jumps do not cluster below `0.30s` unless clearly small hops.

## Assumptions & Defaults
- iOS-first POC.
- No native plugin in this iteration.
- No unit tests and no feature flags.
- Skeleton deferred; reintroduced later as optional visual layer only (not timing-critical).

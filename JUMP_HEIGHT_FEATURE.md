# Fonctionnalité de mesure de hauteur de saut

## Vue d'ensemble

La fonctionnalité mesure la hauteur d'un saut vertical à partir d'une vidéo en utilisant la **détection de pose par ML Kit Accurate Pose Detection** (iOS, BlazePose 33 points). La hauteur n'est **pas** calculée par déplacement pixel : elle est déduite du **temps de vol** (airtime) entre le décollage et l'atterrissage.

### Formule fondamentale

```
h = g × t² / 8
```

où `g = 9.8066 m/s²` et `t` est le temps de vol en secondes (`jumpCalc.ts:213`).

Cette formule découle de la physique des projectiles : en supposant une sortie et une réception symétriques, `t_vol = 2 × t_montée`, et `h = g × t_montée² / 2 = g × (t_vol/2)² / 2 = g × t_vol² / 8`.

---

## Architecture du pipeline

```
Vidéo (fichier)
    │
    ▼
[JumpVideoAnalysisModule.swift]   ← module natif iOS (Expo Module)
    │  AVFoundation décode les frames
    │  ML Kit Accurate PoseDetector détecte les 33 landmarks par frame
    │  Normalisation coordonnées → [0,1]
    │
    ▼  JumpVideoNativeResult { frames[], videoFps, sampleFps, videoDurationMs, personCountSummary }
    │
    ▼
[analyzeJumpLandmarks()]          ← jumpAnalysis.ts
    │  Extraction des signaux par frame
    │  Filtrage (médian + EMA)
    │  Calibration (baseline au sol)
    │  Détection décollage / atterrissage
    │
    ▼  JumpAnalysisResult { takeoffMs, landingMs, flightMs, heightCm, quality, ... }
    │
    ▼
[jump-detector.tsx]               ← écran React Native
    │  Affichage résultat + overlay squelette
```

---

## 1. Extraction vidéo native (`JumpVideoAnalysisModule.swift`)

### Décodage des frames
- `AVAssetReader` lit la vidéo frame par frame en format BGRA.
- La cadence d'échantillonnage (`sampleFps`) est plafonnée à la cadence réelle de la vidéo (`videoFps`).
- Par défaut : 60 fps d'échantillonnage, 720 frames max.
- Un timestamp précis (`CMSampleBufferGetPresentationTimeStamp`) est attaché à chaque frame.

### Détection de pose (ML Kit BlazePose Accurate)
- `AccuratePoseDetectorOptions` en mode `singleImage` pour privilégier la précision sur l'analyse offline.
- 33 landmarks par frame, chacun avec coordonnées `(x, y)` absolues en pixels et un score de confiance `inFrameLikelihood`.
- Si plusieurs personnes sont détectées, la pose primaire est sélectionnée par un score pondéré sur 11 points clés (nez, épaules, hanches, genoux, chevilles, orteils).

### Normalisation
- Chaque coordonnée est divisée par la largeur/hauteur réelle de l'image (en tenant compte de l'orientation du capteur) → valeurs dans `[0, 1]`.
- L'axe Y croît **vers le bas** (convention image).

---

## 2. Extraction des signaux par frame (`toSignals()`, `jumpAnalysis.ts:231`)

Pour chaque frame, les landmarks pertinents sont extraits avec des seuils de confiance différenciés :

| Point           | Seuil de confiance |
|-----------------|--------------------|
| Orteils / pieds | `minConf - 0.08` (min 0.12) |
| Chevilles       | `minConf - 0.04` (min 0.16) |
| Visage          | `minConf - 0.06` (min 0.18) |
| Corps (épaules, hanches) | `minConf` (défaut 0.20) |

Signaux calculés par frame :
- `leftToeY`, `rightToeY` : Y des orteils (foot index)
- `leftHeelY`, `rightHeelY` : Y des talons
- `leftAnkleY`, `rightAnkleY` : Y des chevilles
- `hipY` : milieu des hanches
- `torso` : longueur torse = `hipY - shoulderY` (utilisé comme échelle de normalisation)
- `centerX` : centre horizontal de la personne (pour détecter le drift)
- `fullBodyVisible` : visage + épaules + hanches + pieds tous visibles
- `fullBodyVisible` : visage + épaules + hanches + genoux + pieds tous visibles
- `feetVisible` : au moins un point de pied/talon/cheville visible

### Sélection du meilleur point de contact au sol (`bestFootGroundY`)
Pour chaque pied, le point **le plus bas** (Y le plus grand) parmi orteil, talon et cheville est retenu. C'est ce point qui sert de référence pour détecter le contact au sol.

---

## 3. Filtrage des signaux

### Filtre médian (`applyMedianFilter`)
Appliqué en premier sur toutes les séries temporelles. Chaque valeur est remplacée par la médiane de sa valeur et de ses voisines immédiates (fenêtre de 3). Élimine les valeurs aberrantes ponctuelles dues à des erreurs de tracking.

### Filtre EMA (`applyEma`)
Appliqué après le filtre médian. `alpha = 1.0` par défaut dans le chemin principal (passage transparent, sans lissage) — commentaire dans le code : "Keep EMA effectively pass-through in the timing path" pour éviter de décaler le moment de touchdown.

---

## 4. Calibration / Baseline (`buildBaseline()`, `jumpAnalysis.ts:324`)

Une fenêtre immobile d’environ **750 ms** (`CALIBRATION_WINDOW_MS`) est recherchée automatiquement dans la vidéo pour établir la position de référence au sol. Elle n’a plus besoin d’être située au tout début du clip.

### Critères de validité
- Au moins 6 frames avec corps entier visible pendant la fenêtre de calibration.
- Au moins 6 mesures de point de pied valides (gauche + droit combinés).
- **Stabilité horizontale** : le range du centre X doit être < 0.12 (12% de la largeur). Si l'athlète bouge trop avant de sauter → `NO_STABLE_CALIBRATION`.
- **Stabilité verticale** : l'écart-type des positions de pied divisé par la longueur du torse doit être < 0.025. Filtre le bruit de tracking.

### Valeurs de baseline établies
- `leftGroundY`, `rightGroundY` : médiane du meilleur point de contact par pied
- `leftToeY`, `rightToeY` : médiane des orteils seuls (fallback sur ground si non disponible)
- `leftAnkleY`, `rightAnkleY` : médiane des chevilles
- `hipY` : médiane de la hauteur des hanches
- `torso` : médiane de la longueur torse (utilisée comme échelle)
- `centerX` : position horizontale médiane

---

## 5. Détection du décollage et de l'atterrissage

### Métriques de levée (`leftLift`, `rightLift`)

```typescript
leftLift = (baseline.leftGroundY - leftFootY) / torsoScale
```

Le **lift** est positif quand le pied monte (Y diminue). Il est normalisé par la longueur du torse pour être indépendant de la distance caméra-athlète.

### Logique de détection (machine à états)

**Décollage (`TAKEOFF_THRESHOLD = 0.04`)**
- Condition : `leftLift > 0.04 AND rightLift > 0.04` (les deux pieds en l'air)
- Confirmation requise : 2 frames consécutives (`TAKEOFF_CONFIRM_FRAMES`)
- Tolérance aux gaps : jusqu'à 4 frames sans tracking pendant la phase de décollage (`MAX_TAKEOFF_GAP_FRAMES`)
- Timestamp retenu : le dernier frame de contact au sol avant le décollage (pas le premier frame en l'air)

**Atterrissage (`LANDING_THRESHOLD = 0.02`)**
- Condition : au moins un pied avec `lift ≤ 0.02` (retour proche du sol)
- Confirmation requise : 2 frames consécutives
- Timestamp retenu : le premier frame de contact

### Hysteresis
Le seuil de décollage (0.04) est strictement supérieur au seuil d'atterrissage (0.02), créant une hysteresis qui empêche un faux atterrissage immédiatement après le décollage.

---

## 6. Validations et raisons d'invalidation

Vérifications dans l'ordre :

| Raison | Condition |
|--------|-----------|
| `NO_PERSON` | 0 frames analysées |
| `MULTIPLE_PEOPLE` | Plus d'une personne détectée |
| `BODY_NOT_FULLY_VISIBLE` | Corps entier visible < 60% des frames |
| `FEET_NOT_VISIBLE` | Pieds visibles < 45% des frames |
| `NO_STABLE_CALIBRATION` | Calibration échoue (stabilité ou frames insuffisants) |
| `EXCESS_HORIZONTAL_MOTION` | Drift horizontal max > 0.18 (18% largeur) |
| `NO_TAKEOFF` | Décollage non détecté après calibration |
| `NO_LANDING` | Atterrissage non détecté après décollage |
| `AIRTIME_OUT_OF_RANGE` | `flightMs < 180ms` ou `flightMs > 900ms` |

La plage `[180ms, 900ms]` correspond approximativement à des sauts entre ~4 cm et ~100 cm de hauteur.

---

## 7. Calcul de la hauteur

```typescript
// jumpCalc.ts:213
export function heightFromFlightTime(flightTimeMs: number): number {
  const t = flightTimeMs / 1000;
  return (GRAVITY_MS2 * t * t) / 8 * 100;  // résultat en cm
}
```

Exemple : 400 ms de vol → `9.8066 × 0.4² / 8 × 100 ≈ 19.6 cm`

---

## 8. Score de qualité

La qualité (`LOW` / `MEDIUM` / `HIGH`) est calculée par une somme pondérée :

| Facteur | Poids | Formule |
|---------|-------|---------|
| Confiance moyenne des landmarks | 30% | `clamp01((avgConf - 0.22) / 0.5)` |
| Ratio frames incertaines | 25% | `clamp01(1 - uncertaintyRatio / 0.25)` |
| Visibilité (corps + pieds) | 20% | `clamp01((ratio - 0.55) / 0.4)` |
| Score FPS (cible 60 fps) | 15% | `min(sampleFps, 60) / 60` |
| Stabilité calibration | 10% | `clamp01(1 - stability / 0.03)` |

Seuils : `≥ 0.72` → `HIGH`, `≥ 0.45` → `MEDIUM`, sinon `LOW`.

Les `qualityFlags` ajoutent des avertissements textuels sans invalider le résultat (ex: `LOW_CONFIDENCE`, `UNCERTAIN_TRACKING`, `CALIBRATION_NOISE`).

---

## 9. Timeline de phases

Chaque frame produit un `JumpPhaseSample` avec :
- `phase` : `GROUND_CONTACT` | `AIRBORNE` | `UNCERTAIN`
- `leftLift`, `rightLift` : levée normalisée par le torse
- `hipLift` : levée des hanches normalisée
- `horizontalDrift` : drift horizontal depuis la baseline

Cette timeline est utilisée par l'UI pour colorier la barre de progression vidéo (vert = sol, bleu = en l'air, orange = incertain).

---

## 10. Overlay de debug (`JumpDebugOverlay`)

Affiché en superposition sur la vidéo lors de la review :
- Squelette complet via les `SKELETON_CONNECTIONS` de `poseUtils.ts`
- Points colorés par type : orteils (vert), talons (bleu ciel), chevilles (orange), reste (blanc)
- Ligne horizontale pointillée bleue = Y de baseline des orteils

---

## 11. Statut du mode pieds seuls

Le prototype `jumpFeetAnalysis.ts` n'est plus le chemin actif. Le flux de production visé repose sur :

- vidéo corps entier
- visage visible
- ML Kit Accurate Pose Detection
- détection du temps de vol à partir des landmarks full-body

---

## Constantes clés

| Constante | Valeur | Rôle |
|-----------|--------|------|
| `CALIBRATION_WINDOW_MS` | 750 ms | Durée de la fenêtre de calibration |
| `TAKEOFF_THRESHOLD` | 0.04 | Seuil de levée pour décollage (normalisé torse) |
| `LANDING_THRESHOLD` | 0.02 | Seuil de levée pour atterrissage |
| `TAKEOFF_CONFIRM_FRAMES` | 2 | Frames consécutives pour confirmer décollage |
| `LANDING_CONFIRM_FRAMES` | 2 | Frames consécutives pour confirmer atterrissage |
| `MAX_TAKEOFF_GAP_FRAMES` | 4 | Gap de tracking toléré pendant décollage |
| `MIN_FLIGHT_MS` | 180 ms | Temps de vol minimum valide |
| `MAX_FLIGHT_MS` | 900 ms | Temps de vol maximum valide |
| `MAX_CALIBRATION_CENTER_RANGE` | 0.12 | Mouvement horizontal max pendant calibration |
| `MAX_CALIBRATION_NOISE` | 0.025 | Bruit de stabilité max pendant calibration |
| `MAX_ALLOWED_HORIZONTAL_DRIFT` | 0.18 | Drift horizontal max pendant le saut |
| `MIN_FULL_BODY_VISIBLE_RATIO` | 0.60 | Corps entier visible minimum |
| `GRAVITY_MS2` | 9.8066 m/s² | Constante gravitationnelle |

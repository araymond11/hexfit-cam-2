# Plan Technique V2: Detection Automatique de Hauteur de Saut

## Objectif

Construire une V2 robuste qui permet a l'utilisateur de:

1. filmer ou importer une video,
2. appuyer sur `Analyze Video`,
3. obtenir automatiquement:
   - le moment exact du decollage,
   - le moment exact de l'atterrissage,
   - le temps de vol,
   - la hauteur du saut.

La V2 doit rester compatible avec les videos slow-motion iPhone, mais l'objectif produit final est une experience `one-tap` sans `Set Start`, `Set End`, ni marquage manuel dans le flow principal.

## Definition exacte des evenements

La V2 doit respecter strictement ce contrat:

- `takeoff` = dernier frame ou au moins un orteil touche encore le sol avant une sequence stable sans contact
- `landing` = premier frame ou au moins un orteil touche de nouveau le sol apres la phase de vol

Important:

- le frame retourne ne doit pas etre le frame de confirmation,
- la confirmation sert seulement a eviter le bruit,
- le timestamp retenu doit etre le vrai frame de transition.

## Formule physique

La hauteur reste calculee uniquement a partir du temps de vol:

```text
h = g * t^2 / 8
```

avec:

- `g = 9.81 m/s^2`
- `t = flightTimeSeconds`

## Etat actuel et limite principale

Le pipeline actuel repose sur:

- decodage offline de la video,
- ML Kit Accurate Pose Detection full-body,
- detection d'evenements a partir des landmarks `toe`, `heel`, `ankle`,
- calibration et machine d'etat en JavaScript,
- outils de review manuelle pour benchmarker l'algo.

Ce pipeline est utile pour:

- localiser le corps,
- estimer la posture,
- fournir une premiere approximation des evenements.

Mais il reste une limite structurelle:

- un landmark `toe` n'est pas un detecteur de contact sol,
- un toe landmark peut etre bon spatialement tout en etant decale de quelques frames sur le moment exact du contact,
- la logique actuelle detecte surtout un passage `grounded -> airborne`, pas un contact orteil-sol explicite.

Donc la V2 doit garder ML Kit pour la localisation, mais ne plus laisser ML Kit decider seul du `takeoff` et du `landing`.

## Direction technique V2

La V2 doit devenir un pipeline `pose-assisted toe-contact detection`.

En pratique:

- ML Kit sert a localiser la personne et les pieds,
- un detecteur de contact local au niveau du sol decide si un orteil touche encore le sol,
- la machine d'etat derive `takeoff` et `landing` a partir de ce signal de contact.

## Architecture cible

```text
Video importee / capturee
    ->
Decodage natif frame par frame
    ->
ML Kit Accurate Pose Detection
    ->
Localisation des ROI pied gauche / pied droit
    ->
Detection automatique de la ligne/bande de sol
    ->
Detection locale de contact avant-pied / orteils
    ->
Signal par frame:
  leftToeContact
  rightToeContact
  contactConfidence
    ->
Machine d'etat:
  grounded / airborne / grounded
    ->
Takeoff / Landing
    ->
Flight time
    ->
Height
```

## Composants V2

### 1. Video ingestion

Responsabilites:

- importer ou enregistrer une video,
- conserver les timestamps de playback pour l'UI,
- reconstruire les timestamps physiques pour le calcul,
- continuer a supporter les assets slow-motion iPhone.

Contraintes:

- l'overlay doit rester aligne avec la video lue a l'ecran,
- les calculs physiques doivent utiliser le temps capture reel,
- les outils de benchmark doivent afficher playback et physical en parallele.

### 2. Pose localization

Responsabilites:

- detecter la pose complete,
- verifier qu'une seule personne est presente,
- verifier que le visage et le corps complet sont visibles dans le contrat full-body,
- fournir `toe`, `heel`, `ankle`, `knee`, `hip` pour guider la suite.

Usage en V2:

- filtrer les frames inutilisables,
- estimer la zone du pied gauche et du pied droit,
- stabiliser l'orientation et la geometrie de chaque pied,
- aider a la detection automatique du sol.

### 3. Floor detection

Responsabilites:

- detecter automatiquement une bande de sol fiable,
- la maintenir stable pendant toute la tentative,
- produire un score de confiance sur la qualite de cette estimation.

Approche recommandee:

- detection initiale pendant la fenetre immobile avant le saut,
- bande de sol dans la partie basse de l'image,
- raffinement leger pendant l'analyse mais sans drift libre,
- fallback a l'invalidation si le sol ne peut pas etre estime de facon stable.

Le but n'est pas une ligne mathematiquement parfaite.
Le but est une bande de contact exploitable pour dire si le pied intersecte encore le sol.

### 4. Foot ROI extraction

Responsabilites:

- definir une ROI pied gauche et une ROI pied droit par frame,
- suivre ces ROI pendant la tentative,
- minimiser l'influence de l'arriere-plan.

Approche recommandee:

- centre de ROI guide par `toe`, `heel`, `ankle`,
- orientation du pied estimee par la geometrie `heel -> toe`,
- ROI plus compacte sur l'avant-pied que sur l'ensemble de la jambe.

Sorties souhaitees:

- `leftFootRoi`
- `rightFootRoi`
- `leftForefootRoi`
- `rightForefootRoi`

### 5. Local toe-contact detector

Responsabilite principale:

- produire le vrai signal d'entree evenementiel.

Signal cible par frame:

- `leftToeContact: boolean | null`
- `rightToeContact: boolean | null`
- `leftContactConfidence: number`
- `rightContactConfidence: number`

Detection recommandee:

- mesurer si la zone avant-pied / orteils intersecte la bande de sol,
- utiliser une bande de contact tres fine juste au-dessus du sol,
- combiner:
  - distance verticale au sol,
  - intersection avec la bande de sol,
  - variation locale d'intensite/contour,
  - stabilite frame-to-frame.

Important:

- ML Kit localise le pied,
- le contact detector decide le contact.

### 6. Event state machine

La machine d'etat doit prendre comme entree le signal de contact, pas seulement le `toeY`.

Definition recommandee:

- `grounded` si `leftToeContact || rightToeContact`
- `airborne` si `leftToeContact == false && rightToeContact == false`

Regles:

- `takeoff`:
  - detecter une sequence de `N` frames sans contact,
  - retourner le dernier frame juste avant cette sequence ou au moins un pied etait encore en contact
- `landing`:
  - apres le takeoff, detecter une sequence de `M` frames avec retour de contact,
  - retourner le premier frame de cette sequence ou un pied retouche le sol

Recommandation initiale:

- `N = 2 ou 3 frames`
- `M = 2 ou 3 frames`

Le frame retourne doit rester le frame de transition, pas le frame de confirmation.

### 7. Physics and scoring

Responsabilites:

- calculer `flightPhysicalMs`,
- convertir en hauteur,
- evaluer la qualite du resultat,
- invalider les tentatives qui n'ont pas un signal de contact suffisamment fiable.

Resultat minimal attendu:

- `takeoffPlaybackMs`
- `landingPlaybackMs`
- `takeoffPhysicalMs`
- `landingPhysicalMs`
- `flightPhysicalMs`
- `heightCm`
- `quality`
- `qualityFlags`

## UX cible

Le flow final doit etre:

1. l'utilisateur importe ou filme une video,
2. l'app detecte automatiquement la meilleure fenetre de tentative,
3. l'app analyse la video,
4. l'app retourne la hauteur du saut.

Les outils suivants ne doivent pas rester obligatoires dans le flow final:

- `Set Start`
- `Set End`
- `Mark Manual Takeoff`
- `Mark Manual Landing`

Ces outils peuvent rester en mode interne/debug/QA, mais pas comme dependance produit.

## Strategie de transition

La V2 ne doit pas etre construite d'un coup. Elle doit passer par des etapes mesurables.

### Phase 1. Benchmark fiable

But:

- garder l'outil manuel actuel comme verite terrain,
- mesurer l'ecart entre l'algo et un humain.

Travail:

- conserver `Review Takeoff / Review Landing`,
- garder le recalcul automatique de la hauteur manuelle,
- enregistrer les deltas:
  - `takeoff delta frames`
  - `landing delta frames`
  - `flight delta ms`
  - `height delta cm`

Critere de sortie:

- benchmark stable sur un petit dataset interne.

### Phase 2. Contact detector assiste par pose

But:

- remplacer la logique purement landmarks par un vrai signal de contact.

Travail:

- detecter automatiquement la bande de sol,
- definir des ROI locales de l'avant-pied,
- produire `leftToeContact` / `rightToeContact`,
- brancher la machine d'etat sur ce signal.

Critere de sortie:

- reduction mesurable des deltas algo vs manuel.

### Phase 3. Auto-segmentation de la tentative

But:

- supprimer `Set Start` / `Set End` du flow principal.

Travail:

- detecter automatiquement une fenetre candidate contenant:
  - phase immobile,
  - impulsion,
  - phase airborne,
  - atterrissage,
  - court retour stable
- analyser seulement cette plage.

Critere de sortie:

- videos longues traitables sans intervention manuelle.

### Phase 4. One-tap product mode

But:

- masquer les outils debug au grand public.

Travail:

- conserver les outils manuels derriere un mode interne,
- flow principal:
  - import
  - analyze
  - result

Critere de sortie:

- zero etape manuelle pour l'utilisateur final sur une video valide.

## Donnees par frame recommandees

La V2 devrait idealement exposer une structure de frame de ce type:

```ts
type JumpContactFrame = {
  frameIndex: number;
  playbackTimestampMs: number;
  captureTimestampMs: number | null;
  leftToeContact: boolean | null;
  rightToeContact: boolean | null;
  leftContactConfidence: number;
  rightContactConfidence: number;
  leftToeY: number | null;
  rightToeY: number | null;
  leftHeelY: number | null;
  rightHeelY: number | null;
  leftAnkleY: number | null;
  rightAnkleY: number | null;
  floorBandY: number | null;
  floorConfidence: number;
  poseConfidence: number;
};
```

Cette structure garde assez d'information pour:

- deboguer l'algo,
- recalculer les evenements,
- comparer algo et manuel,
- tracer les signaux utiles.

## Critere de qualite

Un resultat ne doit etre considere robuste que si:

- la personne est suivie correctement pendant la tentative,
- les pieds sont visibles pendant la phase critique,
- la bande de sol est estimee de facon stable,
- le signal de contact a une confiance suffisante,
- l'ecart algo vs manuel devient faible sur le dataset de benchmark.

## Cibles de precision

Cibles recommandees pour une video de bonne qualite:

- erreur moyenne takeoff: `<= 2 frames`
- erreur moyenne landing: `<= 2 frames`
- erreur moyenne flight time: `<= 10 ms`
- erreur moyenne hauteur: `<= 1 cm` a `<= 2 cm`

Pour des videos difficiles:

- l'app peut sortir un resultat `LOW quality`,
- ou invalider proprement la tentative,
- mais elle ne doit pas retourner une valeur arbitraire avec une fausse precision.

## Dataset de benchmark recommande

Construire un petit dataset interne avec:

- 20 a 50 videos,
- differentes hauteurs de saut,
- plusieurs conditions de lumiere,
- plusieurs fonds,
- annotation manuelle du `takeoff`,
- annotation manuelle du `landing`.

Pour chaque clip:

- conserver le resultat algo,
- conserver le resultat manuel,
- comparer l'erreur en frames, ms et cm.

Ce dataset devient la reference avant toute declaration de "mode automatique".

## Choix d'angle de camera

L'angle final doit etre choisi sur la base du benchmark, pas seulement par intuition.

Options a comparer:

- vue de face
- vue legerement 3/4
- vue plus basse avec sol bien visible

Critere de choix:

- angle qui minimise l'erreur sur le moment exact du dernier/premier contact orteil-sol.

## Risques techniques principaux

1. Le landmark `toe` est bon pour localiser le pied mais insuffisant pour dater seul le contact exact.
2. La ligne de sol peut etre mal estimee si le contraste est mauvais.
3. Les videos slow-motion iPhone peuvent separer temps playback et temps physique, ce qui doit rester explicitement gere.
4. Une logique trop agressive de confirmation peut decaler les evenements de plusieurs frames.
5. Des outils debug trop visibles peuvent devenir une fausse solution produit si on ne les retire jamais du flow principal.

## Definition de succes de la V2

La V2 sera consideree reussie si:

- l'utilisateur peut donner une video valide a l'app,
- l'app trouve automatiquement la tentative,
- l'app detecte:
  - le dernier frame de contact avant l'envol,
  - le premier frame de contact a l'atterrissage,
- l'app calcule la hauteur automatiquement,
- le benchmark montre un faible ecart par rapport au marquage manuel,
- les outils manuels restent reserves au debug et ne sont plus necessaires pour l'utilisateur final.


# halj — jam génératif local

Une appli qui écoute la guitare en train d'être jouée et génère en live un
accompagnement : un **drone dark accordé sur la tonalité détectée** et une
**couche rythmique kick/percu réactive au jeu**.

Tout tourne en local, aucun traitement cloud.

Implémentation de la note de cadrage [`note-de-cadrage-jam-guitare.pdf`](note-de-cadrage-jam-guitare.pdf).

```
guitare ──▶ analyse Python ──OSC──▶ synthèse SuperCollider ──▶ drone + rythme
            chroma / tonalité              drones, percus
            attaques / énergie             faible latence
```

## Démarrage rapide

```bash
pip install -e ".[audio]"      # + sounddevice pour la capture micro
halj devices                   # repérer l'entrée à utiliser
halj analyze --source mic      # phase 1 : voir ce que l'analyse entend
```

Sans guitare sous la main, `--source synth` fait tourner la chaîne complète sur
une guitare synthétique interne :

```bash
halj analyze --source synth
```

Pour le son, ouvrir [`supercollider/jam.scd`](supercollider/jam.scd) dans
l'IDE SuperCollider, évaluer le bloc entier, puis :

```bash
halj jam --source mic          # drone + rythme
```

## Les commandes suivent la roadmap

| Commande | Étape | Ce que ça fait |
|---|---|---|
| `halj analyze` | phase 1 | analyse seule, affichage terminal, **aucun son généré** |
| `halj drone` | phase 2 | drone accordé sur la tonalité, sans rythme |
| `halj jam` | phase 3 | drone + couche rythmique |
| `halj mock` | — | émet le protocole OSC **sans analyse**, pour régler le patch SC |
| `halj patterns` | — | table des motifs euclidiens par palier de densité |
| `halj devices` | — | liste les entrées audio |

Options utiles : `--source mic|file|synth`, `--input prise.wav`, `--bpm`,
`--osc-port`, `--duration`, `--config config.json`, `--no-pacing` (analyse un
fichier à pleine vitesse au lieu du temps réel).

`halj mock` répond directement au risque identifié au cadrage — l'overhead du
setup cross-process. Il permet de développer tout le patch SuperCollider sans
micro, sans guitare et sans que l'analyse soit finie.

## Ce que fait l'analyse

| Brique | Approche | Fichier |
|---|---|---|
| Chroma | projection log-fréquentielle du spectre sur 12 classes | `analysis/chroma.py` |
| Tonalité | Krumhansl-Schmuckler + lissage + hystérésis | `analysis/key.py` |
| Attaques | flux spectral normalisé, seuil adaptatif | `analysis/onset.py` |
| Énergie | RMS lissé + débit d'attaques → densité | `analysis/energy.py` |
| Rythme | motifs euclidiens (Bjorklund) pilotés par la densité | `generative/` |

**Un seul pipeline chroma gère accords et notes seules.** Le chroma replie les
octaves, donc une note isolée et l'accord qui la contient nourrissent le même
espace à 12 dimensions : pas besoin de détecter dans quel mode on joue.

**Le lissage temporel n'est pas optionnel.** Deux mécanismes se cumulent : le
chroma est lissé (EMA + fenêtre glissante de 2 s), et la décision de changer de
tonalité passe par une hystérésis — la nouvelle tonalité doit battre celle en
place d'une marge nette *et* tenir plusieurs trames. Sans ça la tonalité flicke
à chaque trame et le drone devient inécoutable.

**Onset detection plutôt que beat tracking**, comme tranché au cadrage : plus
robuste, déjà organique, moins risqué à développer.

### Deux réglages qui portent tout le reste

Ils sont calibrés sur un banc de signaux synthétiques (24 accords, 24 lignes
monodiques), et les tests les protègent :

- **`key.spectral_tilt`** (compensation en 1/f²). Le 5ᵉ harmonique d'une corde
  est une tierce majeure au-dessus de sa fondamentale : sans compensation, un
  chroma entend un do majeur comme un mi mineur. Mesuré : **8/48 sans
  compensation, 47/48 avec**. Corollaire contre-intuitif — la compression
  logarithmique du spectre, pourtant d'usage courant pour un chroma, *dégrade*
  la détection de moitié ici, puisqu'elle remonte précisément les partiels
  aigus qu'on cherche à atténuer.
- **`onset.delta` / `onset.floor`**. Le flux spectral est normalisé par
  l'énergie des deux trames, donc borné dans [0, 1] : une attaque vaut 0.6 à
  1.0, le sustain d'un accord tourne autour de 0.10, et une note *tenue* tend
  vers 0 quel que soit son niveau. Sans cette normalisation, l'ondulation
  d'interférence d'une note tenue suffit à franchir n'importe quel seuil.

Mesures actuelles sur le banc synthétique : **23/24** accords isolés reconnus
(tonique + mode) ; les progressions Am-F-C-G, C-F-G-C et Em-C-G-D résolvent sur
la bonne tonique ; une attaque détectée par grattage. L'analyse tourne à
**~30× le temps réel** sur cette machine, ce qui laisse de la marge sous le
budget de latence visé (< 20-30 ms).

> Ces chiffres viennent de guitare **synthétique** (Karplus-Strong). Ils
> valident la mécanique et attrapent les régressions, ils ne remplacent pas la
> calibration sur prise réelle — que le cadrage identifie déjà comme un point
> de vigilance. `halj analyze --source file --input prise.wav` est fait pour ça.

## Protocole OSC

Python envoie, SuperCollider reçoit (port 57120, préfixe `/jam` configurable) :

| Adresse | Arguments | Rôle |
|---|---|---|
| `/jam/drone` | `amp` `glide` | démarre / reconfigure le drone |
| `/jam/key` | `tonic` (0-11) `mode` (`maj`/`min`) `root_hz` `confidence` | tonalité détectée |
| `/jam/energy` | `level` `density` | niveau de jeu et densité rythmique |
| `/jam/onset` | `strength` | une attaque vient d'être détectée |
| `/jam/kick` / `/jam/perc` | `velocity` `step` | impact de la grille |
| `/jam/stop` | — | extinction propre |

Les messages d'état ne sont réémis que lorsqu'ils changent : inonder le réseau
local à chaque trame ajouterait de la gigue aux messages qui comptent.

## Configuration

Tous les réglages vivent dans `halj/config.py` (dataclasses figées, commentées
une par une). Pour surcharger, un JSON partiel suffit :

```json
{ "rhythm": { "bpm": 96.0, "steps": 16 }, "drone": { "amp": 0.4 } }
```

```bash
halj jam --config mon-setup.json
```

## Dépendances

Le cœur ne dépend que de **NumPy** et **python-osc** : toute la DSP est écrite
en NumPy, donc l'analyse tourne et se teste sans rien installer de lourd.

- `pip install -e ".[audio]"` → `sounddevice`, pour la capture micro
- `pip install -e ".[mir]"` → `librosa` et `aubio`, sélectionnables via
  `--chroma-backend librosa` et `--onset-backend aubio`
- `scipy`, s'il est présent, accélère le filtrage IIR (sinon, repli pur Python)

`aubio` tient le temps réel sans problème (~x59 mesuré) et constitue une
alternative viable pour les attaques. `librosa`, en revanche, mesure **~x0.2 du
temps réel** sur le chroma CQT : il sert à arbitrer la qualité hors ligne sur
une prise enregistrée, pas à jouer en direct.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

218 tests, ~12 s, sans matériel audio. Ils couvrent les motifs euclidiens
(contre les valeurs de référence connues : tresillo, cinquillo…), la détection
de tonalité, l'hystérésis, la détection d'attaques (jeu doux/fort, note tenue,
rumble, temps mort), les filtres, le protocole OSC sur une vraie socket UDP, et
la chaîne complète du signal jusqu'aux messages.

## Hors scope v0

Conformément au cadrage : beat tracking causal, suivi mélodique note à note,
transcription d'accords fine, interface graphique évoluée. Le tempo de la
grille est donc un paramètre fixe (`--bpm`) — c'est la densité de jeu, pas
l'horloge, qui fait vivre le rythme.

Deux limites à connaître, au-delà de ce périmètre :

- Les impacts rythmiques sont datés côté Python et envoyés en UDP, ce qui
  introduit une gigue de l'ordre de la milliseconde sur la grille. Acceptable
  au v0 ; si ça s'entend, la parade est de faire tourner l'horloge dans
  SuperCollider et de ne lui envoyer que la densité.
- La tonique et son relatif restent proches au sens des profils K-S. La marge
  d'hystérésis les sépare en pratique, mais une progression réellement
  ambiguë (Am-F-C-G) peut légitimement basculer entre les deux.

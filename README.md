# halj — jam génératif local

Une appli qui écoute la guitare en train d'être jouée et génère en live un
accompagnement : un **drone dark qui déroule une grille d'accords** ancrée sur
la tonalité détectée, et une **couche rythmique kick/percu réactive au jeu**.

Elle sait aussi **s'ignorer elle-même** : joué sur haut-parleur, un téléphone
réentend son propre drone, et sans précaution c'est lui — pas l'instrument —
qui finit par décider de la tonalité.

Tout tourne en local, aucun traitement cloud.

Implémentation de la note de cadrage [`note-de-cadrage-jam-guitare.pdf`](note-de-cadrage-jam-guitare.pdf).

Deux versions, même chaîne d'analyse :

```
guitare ──▶ analyse Python ──OSC──▶ synthèse SuperCollider ──▶ accords + rythme
            chroma / tonalité              drone à trois voix
            attaques / énergie             percus, faible latence
            auto-écoute                 ▲
            grille d'accords ───────────┘

guitare ──▶ page web autonome ──▶ accords + rythme        (web/index.html)
            Web Audio, zéro dépendance, rien à installer
```

## Version web

`web/index.html` fait tourner toute la chaîne dans le navigateur — l'alternative
Web Audio envisagée au cadrage. Un seul fichier, aucune dépendance, aucun
serveur de traitement : ouvrez, jouez.

```bash
cd web && python -m http.server 8000   # le micro exige https ou localhost
```

puis <http://localhost:8000>. Le bouton **Démo** joue une guitare de synthèse
interne et fonctionne partout, même sans micro — l'équivalent de `--source synth`.

Deux réglages décident du comportement :

- **écoute : casque / haut-parleur.** Sur haut-parleur, la page retire de son
  analyse ce qu'elle vient de jouer *et* laisse le navigateur annuler l'écho —
  il dispose du signal de sortie et de l'horloge matérielle, ce que la page n'a
  pas. Au casque, l'annulateur reste coupé : conçu pour la voix, il écraserait
  la dynamique de l'instrument.
- **initiative.** À gauche, l'appli suit la tonalité détectée comme un
  accompagnateur discret. À droite, elle déroule sa grille et ne cède plus. Au
  milieu, elle propose et vous laisse la détourner.

### Publier sur GitHub Pages

Pages sert en HTTPS, donc le micro y fonctionne : c'est le bon hébergement pour
cette appli. Le workflow [`.github/workflows/pages.yml`](.github/workflows/pages.yml)
publie `web/` à la racine du site, sur `https://<utilisateur>.github.io/HAL25/`.

Il active Pages lui-même au premier passage (`enablement: true`) : rien à
régler au préalable. Il se déclenche sur chaque push vers `main` touchant
`web/`, ou à la demande via `Actions` → `Run workflow`.

Deux causes d'échec à connaître :

- **`Get Pages site failed … Not Found`** : Pages n'est pas activé et l'action
  n'a pas le droit de l'activer. Basculer `Settings` → `Pages` → Source sur
  **GitHub Actions** règle le problème définitivement.
- **rien ne se déclenche** : le workflow doit être sur `main`. Depuis une
  branche de travail, ni le push ni `Run workflow` ne le proposent.

La version web reprend les constantes calibrées et les algorithmes du paquet
Python, et `tests/test_web.py` le vérifie dans un vrai Chromium : motifs
euclidiens, corrélations K-S, projection du chroma, plancher d'auto-écoute,
grilles d'accords et point de bascule entre mener et suivre sont confrontés aux
**mêmes valeurs de référence** que la version Python. C'est ce qui empêche les
deux implémentations de diverger en silence.

Deux différences assumées, imposées par le contexte navigateur :

- le pas d'analyse est celui de `requestAnimationFrame` (~16 ms) et non un hop
  fixe de 5.8 ms ; toutes les constantes de temps sont donc exprimées en
  secondes et converties avec le `dt` réellement mesuré, ce qui rend l'analyse
  indépendante de la cadence d'affichage ;
- le rythme est planifié sur l'horloge audio avec anticipation, donc posé à
  l'échantillon près — la gigue signalée comme limite connue de la version
  Python (datation côté Python + UDP) disparaît ici.

Cette planification par anticipation sert une deuxième fois : l'appli connaît
la date exacte de chaque impact qu'elle va émettre, et peut donc prévenir son
analyse de ne pas le prendre pour une attaque du joueur. La conversion se fait
avec l'écart mesuré entre l'horloge audio et celle de l'analyse, plus la
latence de sortie déclarée par le navigateur (`outputLatency`).

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
| `halj drone` | phase 2 | drone accordé, grille d'accords, sans rythme |
| `halj jam` | phase 3 | drone + couche rythmique |
| `halj mock` | — | émet le protocole OSC **sans analyse** (accords compris), pour régler le patch SC |
| `halj patterns` | — | table des motifs euclidiens par palier de densité |
| `halj devices` | — | liste les entrées audio |

Options utiles : `--source mic|file|synth`, `--input prise.wav`, `--bpm`,
`--agency 0..1`, `--osc-port`, `--duration`, `--config config.json`,
`--no-harmony`, `--no-self-listen`, `--no-pacing` (analyse un fichier à pleine
vitesse au lieu du temps réel).

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
| Auto-écoute | plancher glissant : ce qui ne bouge pas, c'est nous | `analysis/selfmask.py` |
| Rythme | motifs euclidiens (Bjorklund) pilotés par la densité | `generative/` |
| Accords | grille en degrés, avancée à la mesure, cède si le jeu insiste | `generative/harmony.py` |

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

## L'appli s'ignore elle-même

Posé sur la table, un téléphone entend son propre haut-parleur mieux qu'il
n'entend l'instrument. La boucle se referme alors deux fois : le drone nourrit
le chroma, le chroma confirme la tonalité du drone, et la tonalité se fige sur
elle-même quoi que joue l'instrumentiste ; et le niveau ne redescend jamais,
donc la densité non plus, donc la couche rythmique tourne toute seule et
entretient le niveau.

Tout part d'**une seule mesure** : la part de ce qu'on entend qui n'est pas
nous. Le niveau réinjecté n'est pas modélisable — haut-parleur, pièce,
distance — donc il n'est pas modélisé, il est **mesuré**. Un plancher glissant
apprend, bin par bin, le niveau de ce qui ne bouge pas ; ce qui le dépasse est
ce que quelqu'un vient d'ajouter. De là découlent, sans autre estimateur, le
spectre nettoyé qui nourrit le chroma, le niveau de jeu, et le moment où
l'analyse peut de nouveau se prononcer sur une tonalité.

Trois choix portent le résultat, et chacun vient d'une mesure :

- **le plancher descend lentement, pas instantanément.** La tentation serait
  d'en faire un suiveur de minimum. Mais un drone *bat* : deux partiels voisins
  se renforcent et s'annulent tour à tour, et un minimum se pose au creux du
  battement. Mesuré sur un drone à trois voix — creux 4.7, moyenne 15.0 : le
  minimum sous-estime d'un facteur trois ce qu'il faut retirer.
- **on retire exactement le plancher, ni plus ni moins.** À 0.9 on laisserait
  fuir 10 % du drone sur *tous* ses partiels, assez pour qu'il continue à
  dicter sa tonalité ; au-delà de 1.0 on mordrait sur le jeu. Ce qui dépasse le
  plancher est, par construction, ce que le joueur a ajouté — y compris quand
  il joue la note du drone, ce qui arrive tout le temps puisque c'est sa
  tonique.
- **le plancher continue de monter pendant le jeu.** Ce n'est pas une
  négligence : bin par bin, une ligne mélodique se déplace — une note tient une
  demi-seconde là où le plancher met deux secondes à monter, elle est partie
  avant d'avoir compté. Conditionner cette montée à « personne ne joue »
  fabriquerait au contraire un verrou : plancher figé, donc résidu élevé, donc
  « ça joue », donc plancher figé — mesuré, une ligne jouée 4 dB sous le drone
  disparaissait complètement au bout de sept secondes. Un sursis après chaque
  attaque protège le seul cas où l'argument tombe : une note réellement tenue.

Les impacts rythmiques sont traités à part, parce qu'on connaît leur date à
l'avance : autour de chacun, le seuil de détection d'attaque est **relevé**
plutôt que coupé. Une vraie attaque par-dessus le kick passe encore, le kick
seul non — couper franchement mangerait toutes les attaques jouées sur le
temps, c'est-à-dire exactement celles qu'on veut entendre.

**Ce que ça donne.** Cas extrême du banc de test — drone quatre décibels
*au-dessus* de l'instrument, tenu en permanence, même bande de fréquences :

| | sans auto-écoute | avec |
|---|---|---|
| masse du chroma prise par les trois notes du drone | 63 % | **13 %** |
| masse du chroma laissée à la ligne jouée | 34 % | **74 %** |
| niveau de jeu, micro n'entendant que l'appli | 0.60 | **0.001** |
| densité, micro n'entendant que l'appli | 0.33 | **0.000** |

Au casque, il n'y a rien à retirer : le plancher tombe à zéro, la correction
disparaît d'elle-même. Aucun réglage à faire pour passer d'un cas à l'autre.

**Ce que ça ne fait pas.** Ce n'est pas un annulateur d'écho : sans signal de
référence aligné à l'échantillon, on retire un *niveau*, pas une forme d'onde.
Le résidu suffit encore, parfois, à faire pencher une corrélation K-S entre
deux tonalités voisines. C'est pourquoi la version web active *aussi*
l'annulateur du navigateur en mode haut-parleur — lui a le signal de sortie et
l'horloge matérielle — et pourquoi l'autre moitié de la réponse est ailleurs :
une appli qui tient sa propre grille d'accords ne dépend plus d'entendre juste
à chaque instant.

## L'appli propose, au lieu de seulement suivre

Un accompagnement qui ne fait que suivre finit par tourner en rond. Le
directeur harmonique (`generative/harmony.py`) renverse le rapport : il tient
une grille écrite en degrés — donc transposable telle quelle dans la tonalité
détectée — l'avance à la mesure, et ne s'en détourne que si le jeu insiste.

Un seul réglage, `agency` (`--agency`, ou le curseur *initiative* dans la page) :

| valeur | comportement |
|---|---|
| `0` | suiveur pur : l'accord est la triade de la tonalité détectée, comme avant ce module |
| `0.5` | conversation : l'appli déroule sa grille, mais cède dès que le jeu désigne clairement un autre accord |
| `1` | meneur : la grille tient bon, quoi que joue l'instrumentiste |

Le vote du joueur est lu dans le chroma accumulé depuis le dernier changement :
la masse tombant dans l'accord, plus un bonus sur la fondamentale. Ce bonus
n'est pas cosmétique — La mineur et Do majeur partagent deux notes sur trois,
c'est la fondamentale qui les sépare.

Trois détails font la différence entre une grille et une grille jouable :

- **on ne se réancre pas sur un degré de sa propre grille.** Ancre en La
  mineur, le joueur passe sur Fa, le détecteur annonce « Fa majeur » — mais Fa
  est le VI<sup>e</sup> degré : rien n'a bougé, c'est la grille qui fonctionne.
  Sans ce filtre, l'ancre suivait chaque accord et la grille repartait de zéro
  toutes les deux mesures.
- **le drone glisse au plus court, sans dériver.** Prendre l'octave la plus
  proche à chaque accord est un cliquet : mesuré sur Am-F-C-G, le drone perdait
  une octave et demie en deux tours et finissait sous le seuil d'audition d'un
  haut-parleur de téléphone. Un rappel vers le registre nominal borne l'écart à
  une octave.
- **le changement s'annonce.** Le dernier pas avant un nouvel accord est
  accentué, et la page affiche l'accord suivant avec le temps qu'il reste : une
  grille qui bouge sans prévenir ne se joue pas.

## Protocole OSC

Python envoie, SuperCollider reçoit (port 57120, préfixe `/jam` configurable) :

| Adresse | Arguments | Rôle |
|---|---|---|
| `/jam/drone` | `amp` `glide` | démarre / reconfigure le drone |
| `/jam/key` | `tonic` (0-11) `mode` (`maj`/`min`) `root_hz` `confidence` | tonalité détectée |
| `/jam/chord` | `root` (0-11) `quality` `root_hz` `decision` | accord tenu par le drone, et qui l'a décidé |
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
{
  "rhythm": { "bpm": 96.0, "steps": 16 },
  "drone": { "amp": 0.4 },
  "harmony": { "agency": 0.8, "bars_per_chord": 2 },
  "self_listen": { "enabled": true, "play_gate": 0.04 }
}
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

308 tests, ~23 s, sans matériel audio. Ils couvrent les motifs euclidiens
(contre les valeurs de référence connues : tresillo, cinquillo…), la détection
de tonalité, l'hystérésis, la détection d'attaques (jeu doux/fort, note tenue,
rumble, temps mort), les filtres, le protocole OSC sur une vraie socket UDP, et
la chaîne complète du signal jusqu'aux messages.

L'auto-écoute et la grille d'accords sont testées sur ce qu'elles changent, pas
sur leur mécanique : une ligne mélodique rejouée par-dessus un drone plus fort
qu'elle doit rendre au joueur la majorité du chroma, un drone seul ne doit plus
faire vivre le rythme, et le point de bascule entre mener et suivre doit tomber
au même endroit des deux côtés du portage.

Les tests de parité web (`tests/test_web.py`) pilotent un Chromium sans
interface ; ils sont ignorés si Playwright n'est pas installé :

```bash
pip install playwright && playwright install chromium
```

## Hors scope v0

Conformément au cadrage : beat tracking causal, suivi mélodique note à note,
transcription d'accords fine, interface graphique évoluée. Le tempo de la
grille est donc un paramètre fixe (`--bpm`) — c'est la densité de jeu, pas
l'horloge, qui fait vivre le rythme.

Trois limites à connaître, au-delà de ce périmètre :

- Les impacts rythmiques sont datés côté Python et envoyés en UDP, ce qui
  introduit une gigue de l'ordre de la milliseconde sur la grille. Acceptable
  au v0 ; si ça s'entend, la parade est de faire tourner l'horloge dans
  SuperCollider et de ne lui envoyer que la densité.
- La tonique et son relatif restent proches au sens des profils K-S. La marge
  d'hystérésis les sépare en pratique, mais une progression réellement
  ambiguë (Am-F-C-G) peut légitimement basculer entre les deux.
- L'auto-écoute retire un *niveau*, pas une forme d'onde : elle n'a pas de
  signal de référence aligné à l'échantillon. Sur haut-parleur, un résidu
  subsiste et peut encore faire pencher une corrélation K-S entre deux
  tonalités voisines. Si l'appli ne vous entend plus, la parade est de baisser
  le drone plutôt que de monter le volume.

"""Retire de l'analyse ce que l'appli vient elle-meme de jouer.

Le probleme apparait des qu'on joue sur haut-parleur — un telephone pose sur la
table, typiquement. Le micro reentend le drone et les percussions emis a
l'instant, et la boucle se referme deux fois :

* le drone nourrit le chroma, le chroma confirme la tonalite du drone, et la
  tonalite se fige sur elle-meme quoi que joue l'instrumentiste ;
* le niveau ne redescend jamais, donc la densite non plus, donc la couche
  rythmique tourne toute seule et entretient le niveau.

Tout part d'une seule mesure : **la part de ce qu'on entend qui n'est pas
nous**. Le niveau reinjecte n'est pas modelisable — il depend du haut-parleur,
de la piece, de la distance — donc il n'est pas modelise, il est mesure. Un
plancher glissant apprend, bin par bin, le niveau de ce qui ne bouge pas ; ce
qui le depasse est ce que quelqu'un vient d'ajouter.

De cette mesure decoulent, sans autre estimateur :

* le spectre nettoye qui nourrit le chroma — le plancher est retire sur les
  bins ou vit le drone, dont les partiels sont connus a la frequence pres ;
  ailleurs, rien n'est touche, pour ne pas mordre sur une note tenue par
  l'instrumentiste ;
* le niveau de jeu, qui est le niveau mesure multiplie par cette part, avec un
  portail franc en dessous d'un seuil : tant que le micro n'entend que l'appli,
  le niveau de jeu est nul, pas "un peu" ;
* le droit qu'a le plancher de continuer a monter, et le moment ou l'analyse
  peut recommencer a se prononcer sur une tonalite.

Au casque, il n'y a rien a retirer : le plancher tombe a zero, la part "pas
nous" vaut 1, et toute la correction disparait d'elle-meme. Aucun reglage a
faire pour passer d'un cas a l'autre.

**Les impacts rythmiques** sont traites a part, parce qu'on connait leur date a
l'avance : autour de chacun, le seuil de detection d'attaque est *releve*
plutot que coupe. Une vraie attaque par-dessus le kick passe encore, le kick
seul non. Couper franchement mangerait toutes les attaques jouees sur le temps,
c'est-a-dire exactement celles qu'on veut entendre.

Ils ne polluent pas le chroma, et c'est gratuit : le kick vit sous `fmin_hz`
(55 Hz environ) et la percu au-dessus de `fmax_hz` (bande 2.4-4.2 kHz). Seul le
drone tombe dans la bande analysee.

**Ce que ca ne fait pas.** Ce n'est pas un annulateur d'echo : sans signal de
reference aligne a l'echantillon, on retire un *niveau*, pas une forme d'onde.
Mesure sur le cas extreme — drone 4 dB au-dessus de l'instrument, meme bande —
les trois notes du drone tombent de 63 % a 13 % de la masse du chroma et la
ligne jouee monte de 34 % a 74 %. Le residu suffit encore, parfois, a faire
pencher une correlation de Krumhansl-Schmuckler entre deux tonalites voisines.
D'ou l'autre moitie de la reponse, dans `generative/harmony.py` : une appli qui
tient sa propre grille d'accords ne depend plus d'entendre juste a chaque
instant.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..config import SelfListenConfig


# Inertie de la moyenne courante au demarrage, en secondes : sans elle, la
# toute premiere trame fixerait le plancher a elle seule, et si le joueur a
# deja commence, c'est lui qu'on prendrait pour le fond. Exprimee en temps et
# non en trames, pour que le portage navigateur — dont le pas d'analyse est
# celui de l'affichage — se comporte a l'identique.
STARTUP_INERTIA_S = 0.2


class FloorFollower:
    """Suiveur du plancher d'energie : montee lente et conditionnelle, descente lente.

    C'est l'outil qui separe le stationnaire du transitoire sans rien savoir du
    signal : le drone tient une note pendant des secondes, une phrase jouee
    bouge. Le plancher colle donc au drone et ignore le jeu.

    Deux constantes de temps, et le choix de chacune se justifie :

    * la *montee* est conditionnee par l'appelant (`allow_rise`), qui la coupe
      juste apres une attaque. Bin par bin, une ligne melodique se deplace trop
      vite pour etre apprise, mais une note reellement *tenue* finirait par
      l'etre — et on soustrairait alors l'instrumentiste ;
    * la *descente* est libre mais lente. La tentation serait de la rendre
      instantanee — un vrai suiveur de minimum — mais un drone bat : deux
      partiels voisins produisent une oscillation de quelques hertz, et un
      minimum se pose au creux du battement. Mesure sur un drone a trois voix :
      creux 4.7, moyenne 15.0 ; le minimum sous-estime le niveau tenu d'un
      facteur trois, et ce qu'on ne retire pas, le chroma l'entend encore.
      Une descente en ~1 s enjambe les battements tout en suivant un vrai
      changement d'accord (le glissando du drone dure 1.8 s).
    """

    def __init__(self, rise_tau_s: float, fall_tau_s: float):
        self.rise_tau_s = max(rise_tau_s, 1e-6)
        self.fall_tau_s = max(fall_tau_s, 1e-6)
        self.value: np.ndarray | None = None
        self._elapsed = 0.0

    def reset(self) -> None:
        self.value = None
        self._elapsed = 0.0

    def update(
        self, observed: np.ndarray, dt: float, allow_rise: bool = True
    ) -> np.ndarray:
        observed = np.asarray(observed, dtype=np.float64)
        if self.value is None or self.value.shape != observed.shape:
            self.value = observed.copy()
            return self.value
        dt = max(dt, 0.0)
        rising = observed > self.value
        rise = 0.0
        if allow_rise:
            self._elapsed += dt
            # Au demarrage, l'exponentielle mettrait plusieurs secondes a
            # rejoindre un niveau qu'une moyenne courante donne tout de suite —
            # et pendant ces secondes-la, l'appli entend son propre drone et
            # peut se verrouiller dessus. On prend donc la moyenne cumulee tant
            # qu'elle converge plus vite que la constante de temps.
            rise = max(
                1.0 - np.exp(-dt / self.rise_tau_s),
                dt / (self._elapsed + STARTUP_INERTIA_S),
            )
        alpha = np.where(rising, rise, 1.0 - np.exp(-dt / self.fall_tau_s))
        self.value = self.value + alpha * (observed - self.value)
        return self.value


def drone_partials(
    root_hz: float,
    intervals: tuple[int, ...] | list[int],
    count: int,
    fmax_hz: float,
) -> list[float]:
    """Frequences reellement occupees par le drone pour un accord donne.

    Le drone empile un sub une octave sous la fondamentale, puis une voix par
    note de l'accord. Chaque voix est modelisee avec ses harmoniques : les
    nappes de dents de scie en ont par construction, et meme une voix sinus
    revient distordue d'un petit haut-parleur de telephone.
    """
    if root_hz <= 0:
        return []
    voices = [root_hz / 2.0] + [
        root_hz * (2.0 ** (interval / 12.0)) for interval in intervals
    ]
    partials: list[float] = []
    for voice in voices:
        for harmonic in range(1, max(1, count) + 1):
            freq = voice * harmonic
            if freq > fmax_hz:
                break
            partials.append(freq)
    return partials


class SelfListen:
    """Etat de l'auto-ecoute : le peigne du drone, le plancher, les impacts emis.

    Un seul objet porte les trois corrections, parce qu'elles partagent la meme
    information : ce que l'appli est en train de jouer.
    """

    def __init__(
        self,
        config: SelfListenConfig,
        samplerate: int,
        frame_size: int,
        fmin_hz: float,
        fmax_hz: float,
    ):
        self.config = config
        self.bin_hz = samplerate / frame_size
        self.n_bins = frame_size // 2 + 1
        self.fmin_hz = fmin_hz
        self.fmax_hz = fmax_hz
        freqs = np.arange(self.n_bins) * self.bin_hz
        # La part "pas nous" se mesure dans la bande reellement analysee : hors
        # d'elle, le souffle du micro dominerait le rapport et conclurait
        # toujours que quelqu'un joue.
        self._analysed = (freqs >= fmin_hz) & (freqs <= fmax_hz)

        self._spectrum_floor = FloorFollower(
            config.floor_rise_tau_s, config.floor_fall_tau_s
        )
        self._mask = np.zeros(self.n_bins, dtype=bool)
        self._mask_key: tuple[int, ...] = ()
        self._smoothed: np.ndarray | None = None
        self._hits: deque[float] = deque()
        self._elapsed_s = 0.0
        # Part de l'energie analysee qui n'est pas celle de l'appli, lissee.
        # C'est la mesure centrale du module : tout en decoule.
        self.play_ratio = 1.0
        self.playing = True
        self.quiet = True
        self._settled = False
        self._frozen_until = -float("inf")
        self._time_s = 0.0
        self.removed_ratio = 0.0
        # Tant que l'appli n'a rien declare jouer, toutes les corrections sont
        # inertes : en mode analyse seule il n'y a rien a retirer, et la chaine
        # doit se comporter exactement comme avant ce module.
        self.active = False

    # -- ce que l'appli joue ------------------------------------------

    def set_drone(
        self, root_hz: float, intervals: tuple[int, ...] | list[int] = (0,)
    ) -> None:
        """Declare l'accord tenu par le drone ; reconstruit le masque si besoin."""
        partials = drone_partials(
            root_hz, intervals, self.config.partials, self.fmax_hz
        )
        self.active = True
        key = tuple(int(round(f)) for f in partials)
        if key == self._mask_key:
            return
        self._mask_key = key
        self._mask = self._build_mask(partials)

    def _build_mask(self, partials: list[float]) -> np.ndarray:
        mask = np.zeros(self.n_bins, dtype=bool)
        ratio = 2.0 ** (self.config.band_semitones / 12.0) - 1.0
        for freq in partials:
            center = freq / self.bin_hz
            # Au moins deux bins de chaque cote : c'est la largeur du lobe
            # principal de la fenetre de Hann, donc la fuite spectrale d'un
            # partiel fort, meme quand la bande en demi-tons est plus etroite
            # qu'un bin dans le grave.
            half = max(2, int(round(freq * ratio / self.bin_hz)))
            low = max(0, int(round(center)) - half)
            high = min(self.n_bins - 1, int(round(center)) + half)
            if low <= high:
                mask[low : high + 1] = True
        return mask

    @property
    def warming(self) -> bool:
        """Le plancher n'a pas encore trouve le niveau de l'appli.

        Pendant ce temps, l'appli s'entend elle-meme sans savoir encore se
        retirer : ce qu'elle analyse alors ne dit rien du jeu, et surtout pas
        une tonalite — c'est exactement la fenetre ou elle s'accorderait sur son
        propre drone pour ne plus jamais en bouger.

        La fin de la mise en place est *mesuree*, pas decomptee : elle arrive
        quand la part "pas nous" retombe sous le portail. Le delai ne sert que
        de garde-fou, pour le cas ou l'instrumentiste joue deja au demarrage et
        ou cette retombee n'arrive jamais.
        """
        return (
            self.config.enabled
            and self.active
            and not self._settled
            and self._elapsed_s <= self.config.startup_s
        )

    def note_hit(self, time_s: float) -> None:
        """Signale un impact rythmique emis, date sur l'horloge de l'analyse."""
        self._hits.append(time_s + self.config.hit_latency_s)

    def note_onset(self, time_s: float) -> None:
        """Signale une attaque detectee : elle gele la mesure du plancher.

        Une note tenue ne bouge pas assez pour que la part "pas nous" reste
        haute ; l'attaque, elle, prouve qu'on joue. Elle vaut donc un sursis.
        """
        self._frozen_until = time_s + self.config.freeze_after_onset_s

    def reset(self) -> None:
        self._spectrum_floor.reset()
        self._smoothed = None
        self._mask[:] = False
        self._mask_key = ()
        self._hits.clear()
        self._elapsed_s = 0.0
        self.play_ratio = 1.0
        self.playing = True
        self.quiet = True
        self._settled = False
        self._frozen_until = -float("inf")
        self._time_s = 0.0
        self.removed_ratio = 0.0
        self.active = False

    # -- corrections ---------------------------------------------------

    def clean_spectrum(self, magnitudes: np.ndarray, dt: float) -> np.ndarray:
        """Retire le plancher mesure sur les bins du drone, et mesure le reste.

        Un seul passage rend les trois informations dont le reste de la chaine
        a besoin : le spectre nettoye qui nourrit le chroma, la part de
        l'energie qui n'est pas celle de l'appli (`play_ratio`), et de la le
        droit qu'a le plancher de continuer a monter.
        """
        magnitudes = np.asarray(magnitudes, dtype=np.float64)
        if not (self.config.enabled and self.active):
            self.removed_ratio = 0.0
            self.play_ratio, self.playing, self.quiet = 1.0, True, False
            return magnitudes

        self._elapsed_s += dt
        # Le plancher suit — et le chroma recoit — le spectre *lisse*. Un drone
        # bat : ses partiels voisins se renforcent et s'annulent tour a tour, et
        # compare trame a trame il semble aller et venir en permanence. Moyenne
        # sur quelques dixiemes de seconde, il redevient ce qu'il est, immobile,
        # pendant qu'une ligne jouee continue de se deplacer.
        if self._smoothed is None or self._smoothed.shape != magnitudes.shape:
            self._smoothed = magnitudes.copy()
        else:
            beta = 1.0 - float(np.exp(-max(dt, 0.0) / self.config.play_smooth_tau_s))
            self._smoothed = self._smoothed + beta * (magnitudes - self._smoothed)
        # Le plancher est suivi sur tout le spectre, pas seulement sous le
        # masque : quand le drone change d'accord, le masque se deplace sur des
        # bins dont on connait deja le plancher.
        floor = self._spectrum_floor.update(self._smoothed, dt, allow_rise=self.quiet)

        # La soustraction, elle, se limite aux partiels du drone. Ailleurs, ce
        # qui ne bouge pas pourrait etre une note tenue par l'instrumentiste,
        # et on n'a aucune raison d'y toucher.
        cleaned = self._smoothed.copy()
        removed = np.minimum(
            self.config.subtraction * floor[self._mask], self._smoothed[self._mask]
        )
        cleaned[self._mask] = self._smoothed[self._mask] - removed
        total = float(self._smoothed.sum())
        self.removed_ratio = float(removed.sum()) / total if total > 1e-12 else 0.0

        # Pour *mesurer* qui fait du bruit, on retire le plancher partout —
        # pas seulement sous le masque. Ce calcul ne nourrit aucun son et ne
        # peut donc abimer aucun jeu : il n'a pas besoin de la prudence qui
        # limite la soustraction du chroma aux seuls partiels du drone. Sans
        # cela, la fuite spectrale du drone hors de son peigne suffit a faire
        # croire en permanence que quelqu'un joue.
        analysed = self._analysed
        heard = float(self._smoothed[analysed].sum())
        others = float(
            np.maximum(
                self._smoothed[analysed] - self.config.subtraction * floor[analysed],
                0.0,
            ).sum()
        )
        self.play_ratio = others / heard if heard > 1e-12 else 1.0

        self.playing = self.play_ratio > self.config.play_gate
        if not self.playing:
            # La mesure est retombee sous le portail : le plancher a trouve le
            # niveau de l'appli, on sait de nouveau ecouter.
            self._settled = True
        # Le plancher continue de monter *pendant* le jeu, et ce n'est pas une
        # negligence : bin par bin, une ligne melodique se deplace — une note
        # tient un demi-seconde la ou le plancher met deux secondes a monter,
        # elle est partie avant d'avoir compte. Seul l'immobile est appris.
        # Conditionner cette montee a "personne ne joue" fabriquerait au
        # contraire un verrou : plancher fige, donc residu eleve, donc "ca
        # joue", donc plancher fige. Le sursis apres une attaque suffit, et
        # protege le seul cas ou l'argument tombe — une note reellement tenue.
        self.quiet = self._time_s >= self._frozen_until
        return cleaned

    def clean_rms(self, rms: float) -> float:
        """Le niveau de ce qui est *joue* : le niveau mesure, moins notre part.

        Pas de second estimateur ici : `play_ratio` dit deja quelle fraction de
        l'energie analysee n'est pas la notre, et c'est exactement le facteur a
        appliquer. Un portail franc en dessous du seuil, sinon le battement du
        drone entretiendrait a lui seul la densite, donc le rythme, donc le
        battement.
        """
        if not (self.config.enabled and self.active):
            return rms
        return rms * self.play_ratio if self.playing else 0.0

    def threshold_scale(self, time_s: float) -> float:
        """Facteur applique au seuil d'attaque : > 1 autour d'un impact emis."""
        self._time_s = time_s
        if not (self.config.enabled and self.active):
            return 1.0
        horizon = time_s - self.config.hit_post_s
        while self._hits and self._hits[0] < horizon:
            self._hits.popleft()
        for hit in self._hits:
            if hit - self.config.hit_pre_s <= time_s <= hit + self.config.hit_post_s:
                return self.config.hit_threshold_boost
        return 1.0

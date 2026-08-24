"""Configuration centrale du pipeline.

Un seul objet passe de la capture audio jusqu'au scheduler : ca evite les
constantes eparpillees et rend les tests reproductibles (on instancie un
Config, on l'injecte, on assert).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioConfig:
    samplerate: int = 44100
    # hop court = onsets reactifs. 256 @ 44.1k -> 5.8 ms de grain temporel.
    hop_size: int = 256
    channels: int = 1
    device: int | str | None = None
    # taille du bloc rendu par le driver ; multiple de hop_size.
    block_size: int = 512


@dataclass(frozen=True)
class KeyConfig:
    """Detection de tonalite : chroma -> Krumhansl-Schmuckler -> lissage."""

    # Fenetre longue : a 44.1k, 8192 echantillons donnent ~5.4 Hz de resolution,
    # necessaire pour separer les notes graves de la guitare (E2 82 / F2 87 Hz).
    frame_size: int = 8192
    # Le chroma n'a pas besoin du grain temporel de l'onset : 1 trame sur 8
    # (~46 ms) suffit et divise d'autant le cout CPU.
    decimation: int = 8
    # fenetre glissante d'accumulation du chroma, en secondes.
    window_s: float = 2.0
    # constante de temps du lissage exponentiel du vecteur chroma.
    chroma_tau_s: float = 0.35
    # bornes de la bande utile guitare (E2 ~82 Hz -> harmoniques aigues).
    fmin_hz: float = 73.4  # D2, une case sous le Mi grave en drop
    fmax_hz: float = 2093.0  # C7
    bins_per_octave: int = 36  # 3 bins/demi-ton : tolere un accordage flottant
    # Pente de compensation spectrale, en 1/f^tilt. C'est le reglage le plus
    # sensible de toute la detection : une corde de guitare a un 5e harmonique
    # qui est une tierce majeure au-dessus de la fondamentale, si bien qu'un
    # chroma non compense entend un do majeur comme un mi mineur. Mesure sur
    # un banc de 24 accords + 24 lignes monodiques : 8/48 sans compensation,
    # 47/48 a partir de 1.5. Le palier s'etend jusqu'a 3.0.
    spectral_tilt: float = 2.0
    # Largeur des gaussiennes de projection, en multiples de l'ecart entre
    # bins. Au-dela de ~1.5 les classes voisines commencent a se melanger.
    bin_width_scale: float = 1.3
    # Hysterese : la nouvelle tonalite doit battre la courante de cette marge
    # (en score K-S normalise) pendant `hold_frames` trames consecutives.
    switch_margin: float = 0.06
    hold_frames: int = 12
    # En dessous de ce score, on considere la tonalite comme non fiable.
    min_confidence: float = 0.55
    # En dessous de ce RMS, on ne nourrit pas le chroma (silence / souffle).
    silence_rms: float = 0.004


@dataclass(frozen=True)
class OnsetConfig:
    """Detection d'attaques par flux spectral a seuil adaptatif."""

    # Fenetre courte : on veut la reactivite, pas la resolution frequentielle.
    frame_size: int = 1024
    # mediane glissante servant de plancher au seuil, en secondes.
    median_window_s: float = 0.35
    # Facteur multiplicatif au-dessus de la mediane locale. Le flux etant
    # normalise dans [0, 1], ces seuils sont absolus et transposables : une
    # attaque vaut 0.6 a 1.0, le sustain d'un accord tourne autour de 0.10.
    # Le comptage d'attaques est identique entre delta 4.5 et 6 sur le banc
    # de signaux de test — signe que le reglage n'est pas sur une arete.
    delta: float = 5.0
    # Plancher absolu : c'est lui qui tranche pour une attaque isolee dans le
    # silence, ou la mediane locale vaut 0 et ne dit rien.
    floor: float = 0.25
    # Seuil propre au backend aubio, dont l'echelle n'a rien a voir avec
    # `delta` (fonction de detection differente, deja normalisee).
    aubio_threshold: float = 0.4
    # temps mort apres une attaque (bruit de mediator, double-attaque).
    min_interval_s: float = 0.045
    # passe-haut avant flux spectral : coupe les rumbles et le bruit de manche.
    highpass_hz: float = 120.0


@dataclass(frozen=True)
class EnergyConfig:
    """Energie de jeu -> densite rythmique."""

    # constantes de temps asymetriques : monte vite, redescend lentement.
    attack_tau_s: float = 0.12
    release_tau_s: float = 1.6
    # RMS mappe sur [0, 1] entre ces deux bornes (echelle dB).
    floor_db: float = -48.0
    ceil_db: float = -12.0
    # part de la densite venant du debit d'attaques (le reste vient du RMS).
    onset_rate_weight: float = 0.45
    # debit d'attaques considere comme "plein regime", en attaques/seconde.
    onset_rate_full: float = 6.0


@dataclass(frozen=True)
class RhythmConfig:
    """Grille euclidienne pilotee par la densite."""

    bpm: float = 84.0
    steps: int = 16
    # nombre d'impacts kick interpole entre ces bornes selon la densite.
    kick_pulses_min: int = 2
    kick_pulses_max: int = 7
    perc_pulses_min: int = 0
    perc_pulses_max: int = 11
    # rotations fixes : decale les motifs pour eviter kick et perc a l'unisson.
    kick_rotation: int = 0
    perc_rotation: int = 3
    # en dessous de cette densite, la couche rythmique se tait.
    gate_density: float = 0.08


@dataclass(frozen=True)
class DroneConfig:
    amp: float = 0.35
    # glissando entre deux tonalites, en secondes (pas de saut brutal).
    glide_s: float = 1.8
    # octave de la fondamentale du drone (2 = E2/C2 territoire basse).
    octave: int = 2


@dataclass(frozen=True)
class SelfListenConfig:
    """Ce que l'appli retire de son analyse parce qu'elle vient de le jouer.

    Sur haut-parleur — un telephone pose sur la table — le micro reentend le
    drone et les percussions emises a l'instant. Sans precaution la boucle se
    referme : le drone nourrit le chroma, le chroma confirme la tonalite du
    drone, et la tonalite se fige sur elle-meme quoi que joue l'instrumentiste.
    """

    enabled: bool = True
    # Demi-largeur du masque autour d'un partiel du drone, en demi-tons. Seuls
    # ces bins sont corriges : une note du joueur qui ne tombe pas sur un
    # partiel du drone traverse l'analyse intacte.
    band_semitones: float = 0.6
    # Harmoniques modelisees par oscillateur : au-dela, la dent de scie est
    # sous le plancher de bruit du haut-parleur.
    partials: int = 10
    # Suiveur de plancher : c'est lui qui *mesure* le niveau reinjecte, qu'aucun
    # modele ne peut predire (haut-parleur, piece, distance). Il ne monte que
    # quand personne ne joue, avec cette constante de temps.
    floor_rise_tau_s: float = 2.0
    # Descente du plancher. Lente a dessein : un drone bat (deux partiels
    # voisins oscillent a quelques hertz) et une descente instantanee — un vrai
    # suiveur de minimum — se poserait au creux du battement, sous-estimant le
    # niveau tenu d'un facteur trois. Doit rester assez vive pour suivre un
    # changement d'accord, dont le glissando dure 1.8 s.
    floor_fall_tau_s: float = 0.8
    # Fraction du plancher retiree. 1.0 retire exactement ce qui a ete mesure,
    # et rien de plus : ce qui depasse le plancher est, par construction, ce
    # que le joueur a ajoute. En dessous, on laisse fuir une part constante du
    # drone — a 0.9, c'est 10 % du drone sur *tous* ses partiels, assez pour
    # qu'il continue a dicter sa tonalite. Au-dessus, on mord sur le jeu.
    subtraction: float = 1.0
    # Lissage de la part "pas nous" du spectre, en secondes.
    play_smooth_tau_s: float = 0.4
    # En dessous de cette part, on considere que le micro n'entend que l'appli.
    # Mesure sur un drone tenu : moins de 1 % lui survit ; une ligne melodique
    # jouee 10 dB *sous* ce drone en laisse encore 8 a 10 %.
    play_gate: float = 0.04
    # Garde-fou sur la mise en place du plancher, en secondes. Normalement elle
    # se termine d'elle-meme, quand la part "pas nous" retombe sous le portail ;
    # ce plafond ne sert qu'au cas ou l'instrumentiste joue deja au demarrage,
    # ou cette retombee n'arrive jamais.
    startup_s: float = 3.0
    # Le plancher ne monte pas non plus dans la seconde qui suit une attaque :
    # une note tenue ne doit pas etre prise pour le fond.
    freeze_after_onset_s: float = 0.8
    # Fenetre autour d'un impact rythmique emis, ou le seuil d'attaque est
    # releve (et non coupe) : une vraie attaque par-dessus le kick passe encore.
    hit_pre_s: float = 0.02
    hit_post_s: float = 0.09
    hit_threshold_boost: float = 3.0
    # Latence sortie -> micro : le temps que l'impact sorte du haut-parleur et
    # revienne. Sur telephone, la chaine audio ajoute typiquement 20-60 ms.
    hit_latency_s: float = 0.03


@dataclass(frozen=True)
class HarmonyConfig:
    """Agency : l'appli propose une grille d'accords au lieu de subir la tonalite.

    Un accompagnement qui ne fait que suivre finit par tourner en rond — et,
    sur haut-parleur, par se suivre lui-meme. Ici l'appli tient une grille,
    l'avance a la mesure, et n'en change que si le joueur insiste vraiment.
    """

    enabled: bool = True
    # 0 = suiveur pur (l'accord est la tonique detectee, comportement d'avant),
    # 1 = meneur (l'appli deroule sa grille et ne cede qu'a une insistance nette).
    agency: float = 0.5
    bars_per_chord: int = 1
    # Nombre de tours de grille avant d'en proposer une autre.
    renew_after_loops: int = 2
    # Marge dont le jeu doit battre l'accord prevu pour detourner la grille,
    # interpolee entre ces deux bornes selon `agency`. La borne haute depasse
    # l'ecart maximal possible entre deux accords (1.5, tout le chroma sur la
    # fondamentale d'un seul) : a `agency` = 1, la grille ne cede plus du tout.
    follow_margin_min: float = 0.0
    follow_margin_max: float = 1.5
    # Poids supplementaire de la fondamentale dans le vote du joueur : sans lui
    # Am et C, qui partagent deux notes sur trois, seraient indiscernables.
    root_weight: float = 0.5
    # En dessous de cette masse dans l'accord, le vote du joueur est ignore.
    min_vote: float = 0.35


@dataclass(frozen=True)
class ConversationConfig:
    """A qui est le tour — et ce que l'appli fait quand c'est le sien.

    Dans une jam, le lead circule, et il circule par la place qu'on laisse. Un
    accompagnement a initiative fixe suit bien ou mene bien, mais il ne repond
    pas : on joue devant lui, pas avec lui.
    """

    enabled: bool = True
    # Fenetre sur laquelle se juge la place prise par l'instrumentiste, en
    # montee et en descente. L'asymetrie dit tout du comportement voulu : on
    # entend en une seconde que quelqu'un relance, alors qu'il faut plusieurs
    # secondes de blanc avant de conclure qu'il laisse la place.
    presence_attack_s: float = 0.6
    presence_release_s: float = 2.5
    # Delai avant que la mesure de presence veuille dire quelque chose. Sans
    # lui, l'appli prendrait la main dans les premieres secondes — la moyenne
    # part de zero, donc la place semble libre alors que personne n'a encore
    # eu le temps d'y jouer.
    engage_s: float = 4.0
    # Les seuils sont *relatifs* a ce que l'instrumentiste produit d'habitude,
    # jamais absolus. Une ligne melodique sur un petit instrument a cordes, un
    # telephone a deux metres, un accompagnement gratte au mediateur : la meme
    # presence de jeu couvre plus d'un ordre de grandeur. Un seuil fixe
    # conclurait "il ne joue plus" chez l'un et "il n'arrete jamais" chez
    # l'autre. La reference est la presence typique, tenue par un suiveur qui
    # monte vite et redescend en une minute.
    reference_release_s: float = 60.0
    # Plancher de cette reference : sinon, apres un long silence, le moindre
    # souffle passerait pour du jeu.
    reference_floor: float = 0.05
    # Fraction de la presence typique en dessous de laquelle la place est
    # consideree libre — un partenaire entreprenant s'y autorise plus tot.
    free_ratio_min: float = 0.20
    free_ratio_max: float = 0.55
    # Largeur de l'hysterese, en fraction elle aussi. Sans elle, le lead
    # clignoterait a chaque note.
    hysteresis_ratio: float = 0.25
    # Delai avant de prendre la main, et avant de la rendre. L'asymetrie est le
    # coeur du comportement : on met plusieurs secondes a s'autoriser a mener,
    # et moins d'une a se ranger quand l'autre relance. Un partenaire qui
    # n'ecoute pas est pire qu'un partenaire muet.
    take_s_min: float = 1.2
    take_s_max: float = 5.0
    give_s_min: float = 0.4
    give_s_max: float = 1.6
    # En dessous / au-dessus de ces valeurs d'initiative, le lead ne circule
    # plus : le curseur au minimum donne un pur accompagnateur, au maximum un
    # meneur qui ne rend jamais la main.
    never_leads_below: float = 0.02
    always_leads_above: float = 0.98
    # Initiative harmonique effective selon le tour : l'appli tient sa grille
    # quand elle mene, se laisse detourner quand elle accompagne.
    lead_agency: float = 0.85
    follow_agency: float = 0.30
    # Plancher de densite rythmique quand l'appli accompagne. C'est lui qui
    # empeche la percu d'abandonner des qu'on respire : la couche rythmique est
    # censee motiver le jeu, si elle s'eteint elle motive un silence.
    follow_density: float = 0.20
    # ...et quand elle mene. Une relance qui n'insiste pas n'est pas une relance.
    lead_density_min: float = 0.40
    lead_density_max: float = 0.70
    # Phrase de reponse : l'appli joue quelques notes de l'accord en cours
    # quand elle a la main. Une mesure sur deux — a chaque mesure, ce n'est
    # plus une reponse, c'est du bavardage.
    respond: bool = True
    respond_every_bars: int = 2
    # Registre de la reponse. Median a dessein : c'est ce qu'un haut-parleur de
    # telephone reproduit vraiment, alors qu'il ne rend presque rien du drone.
    response_octave: int = 4
    # Duree d'une note de reponse, et donc de la pause d'analyse qu'elle impose.
    response_note_s: float = 0.35


@dataclass(frozen=True)
class OscConfig:
    host: str = "127.0.0.1"
    port: int = 57120  # sclang, pas scsynth (57110)
    prefix: str = "/jam"


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    key: KeyConfig = field(default_factory=KeyConfig)
    onset: OnsetConfig = field(default_factory=OnsetConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    self_listen: SelfListenConfig = field(default_factory=SelfListenConfig)
    harmony: HarmonyConfig = field(default_factory=HarmonyConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    osc: OscConfig = field(default_factory=OscConfig)

    @property
    def hop_s(self) -> float:
        """Duree d'un hop en secondes : l'unite de temps de tout le pipeline."""
        return self.audio.hop_size / self.audio.samplerate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        sections = {
            "audio": AudioConfig,
            "key": KeyConfig,
            "onset": OnsetConfig,
            "energy": EnergyConfig,
            "rhythm": RhythmConfig,
            "drone": DroneConfig,
            "self_listen": SelfListenConfig,
            "harmony": HarmonyConfig,
            "conversation": ConversationConfig,
            "osc": OscConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sections.items():
            if name in data:
                kwargs[name] = klass(**data[name])
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"sections inconnues dans la config : {sorted(unknown)}")
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, **sections: Any) -> "Config":
        """Retourne une copie avec des sections remplacees (Config est frozen)."""
        return replace(self, **sections)

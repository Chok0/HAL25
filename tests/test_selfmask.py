"""L'appli doit s'ignorer elle-meme quand elle joue sur haut-parleur.

Le scenario a couvrir est celui du telephone pose sur la table : le micro
reentend le drone, et sans precaution c'est lui — pas l'instrumentiste — qui
finit par decider de la tonalite.
"""

import numpy as np
import pytest

from halj.analysis.engine import AnalysisEngine
from halj.analysis.selfmask import FloorFollower, SelfListen, drone_partials
from halj.config import Config, SelfListenConfig
from halj.notes import Key, pitch_class_to_hz


@pytest.fixture
def guard(config):
    return SelfListen(
        config.self_listen,
        config.audio.samplerate,
        config.key.frame_size,
        config.key.fmin_hz,
        config.key.fmax_hz,
    )


# --- suiveur de plancher ---------------------------------------------


def test_le_plancher_monte_vite_au_demarrage_puis_lentement():
    """Deux regimes : trouver le plancher tout de suite, le corriger doucement.

    Sans acceleration initiale, l'exponentielle mettrait plusieurs secondes a
    rejoindre le niveau du drone — et pendant ce temps l'appli s'entend
    elle-meme et peut se verrouiller sur sa propre tonalite.
    """
    floor = FloorFollower(rise_tau_s=2.0, fall_tau_s=0.8)
    floor.update(np.array([0.0]), dt=0.05)
    assert floor.update(np.array([1.0]), dt=0.05)[0] < 0.5  # jamais d'un bond

    for _ in range(20):  # 1 s
        floor.update(np.array([1.0]), dt=0.05)
    assert floor.value[0] > 0.75  # deja au bon ordre de grandeur

    # Une fois lance, il faut plusieurs secondes pour bouger d'autant.
    depart = floor.value[0]
    for _ in range(4):  # 0.2 s
        floor.update(np.array([2.0]), dt=0.05)
    assert floor.value[0] - depart < 0.2

    for _ in range(200):
        floor.update(np.array([1.0]), dt=0.05)
    assert floor.value[0] == pytest.approx(1.0, abs=0.02)


def test_le_plancher_gele_pendant_le_jeu():
    """Pendant le jeu, le plancher reste sur ce qu'on a mesure au silence."""
    floor = FloorFollower(rise_tau_s=2.0, fall_tau_s=0.8)
    floor.update(np.array([0.1]), dt=0.05)
    for _ in range(200):
        floor.update(np.array([1.0]), dt=0.05, allow_rise=False)
    assert floor.value[0] == pytest.approx(0.1)


def test_le_plancher_redescend_meme_gele():
    """Si l'appli baisse le son, le plancher doit suivre sans attendre un silence."""
    floor = FloorFollower(rise_tau_s=2.0, fall_tau_s=0.4)
    for _ in range(200):
        floor.update(np.array([1.0]), dt=0.05)
    for _ in range(60):  # 3 s de descente
        floor.update(np.array([0.2]), dt=0.05, allow_rise=False)
    assert floor.value[0] == pytest.approx(0.2, abs=0.01)


def test_le_plancher_enjambe_les_battements_du_drone():
    """Le point dur : deux partiels voisins font battre le niveau a 2 Hz.

    Un suiveur de minimum se poserait au creux, donc sous-estimerait de moitie
    ce qu'il faut retirer — et le drone continuerait a s'entendre.
    """
    floor = FloorFollower(rise_tau_s=2.0, fall_tau_s=0.8)
    dt = 0.046
    creux, moyenne = [], []
    for index in range(600):
        t = index * dt
        niveau = 15.0 + 9.0 * np.cos(2 * np.pi * 2.0 * t)  # bat entre 6 et 24
        floor.update(np.array([niveau]), dt)
        if t > 20.0:
            creux.append(niveau)
            moyenne.append(floor.value[0])
    assert min(creux) < 7.0  # le signal descend bien tres bas
    assert np.mean(moyenne) > 12.0  # le plancher, lui, tient la moyenne


# --- modele du drone --------------------------------------------------


def test_les_partiels_couvrent_le_sub_et_l_accord():
    partiels = drone_partials(110.0, (0, 3, 7), count=4, fmax_hz=2093.0)
    assert min(partiels) == 55.0  # sub, une octave sous la fondamentale
    assert 110.0 in partiels
    assert any(abs(f - 130.81) < 0.1 for f in partiels)  # tierce mineure
    assert any(abs(f - 164.81) < 0.1 for f in partiels)  # quinte


def test_les_partiels_s_arretent_a_la_bande_analysee():
    partiels = drone_partials(110.0, (0,), count=100, fmax_hz=2093.0)
    assert max(partiels) <= 2093.0
    assert len(partiels) < 200  # la serie est coupee, pas juste filtree


def test_le_masque_couvre_le_drone_et_pas_le_reste(guard, config):
    guard.set_drone(110.0, (0, 4, 7))
    bin_hz = config.audio.samplerate / config.key.frame_size

    def masque(freq):
        return bool(guard._mask[int(round(freq / bin_hz))])

    assert masque(55.0) and masque(110.0) and masque(220.0)  # sub et harmoniques
    assert masque(138.59) and masque(164.81)  # tierce majeure et quinte
    # Entre le sub et la fondamentale, le drone n'a rien : le jeu y passe.
    assert not masque(82.41)


# --- soustraction -----------------------------------------------------


def test_sans_drone_declare_rien_n_est_touche(guard):
    """En analyse seule, l'appli ne joue rien : la chaine doit etre inchangee."""
    spectre = np.linspace(0.1, 1.0, guard.n_bins)
    assert np.array_equal(guard.clean_spectrum(spectre, dt=0.05), spectre)
    assert guard.clean_rms(0.5) == 0.5
    assert guard.threshold_scale(1.0) == 1.0
    assert not guard.warming


def test_le_drone_est_retire_et_le_jeu_survit(guard, config):
    """Le point dur : retirer le drone sans mordre sur la note jouee.

    Y compris quand le joueur tombe sur la note du drone, ce qui arrive tout le
    temps — c'est sa tonique.
    """
    bin_hz = config.audio.samplerate / config.key.frame_size
    guard.set_drone(110.0, (0,))
    drone_bin = int(round(110.0 / bin_hz))
    note_bin = int(round(587.33 / bin_hz))  # re5, hors des partiels du drone

    drone = np.zeros(guard.n_bins)
    drone[drone_bin] = 1.0
    for _ in range(400):  # 20 s de drone seul : le plancher a le temps de monter
        propre = guard.clean_spectrum(drone, dt=0.05)
    assert guard.play_ratio < 0.02  # il ne reste rien de ce que l'appli joue
    assert not guard.playing and not guard.warming
    assert propre[drone_bin] < 0.05  # le drone a disparu du spectre analyse

    joue = drone.copy()
    joue[drone_bin] += 0.5  # le joueur ajoute de l'energie sur la meme classe
    joue[note_bin] = 0.8
    for index in range(10):  # une demi-seconde de jeu
        guard.note_onset(index * 0.05)  # comme le ferait le detecteur d'attaques
        propre = guard.clean_spectrum(joue, dt=0.05)

    # Sur le bin du drone, ce qui reste est ce que le joueur y a ajoute : le
    # drone lui-meme est parti, son surplus non.
    assert propre[drone_bin] > 0.2
    assert propre[note_bin] > 0.4  # et la note hors du drone traverse intacte
    assert guard.playing


def test_le_niveau_de_jeu_est_le_niveau_moins_notre_part(guard):
    """Le niveau joue derive de la meme mesure : pas de second estimateur."""
    guard.set_drone(110.0, (0,))
    spectre = np.zeros(guard.n_bins)
    spectre[int(round(110.0 / guard.bin_hz))] = 1.0
    for _ in range(400):
        guard.clean_spectrum(spectre, dt=0.05)
    assert guard.clean_rms(0.02) == 0.0  # portail ferme : personne ne joue

    joue = spectre.copy()
    joue[int(round(587.33 / guard.bin_hz))] = 2.0
    for _ in range(20):
        guard.clean_spectrum(joue, dt=0.05)
    assert 0.0 < guard.clean_rms(0.06) < 0.06


# --- fenetre autour des impacts ---------------------------------------


def test_le_seuil_est_releve_autour_d_un_impact_emis(guard, config):
    latence = config.self_listen.hit_latency_s
    guard.set_drone(110.0, (0,))
    guard.note_hit(1.0)

    assert guard.threshold_scale(0.90) == 1.0  # avant l'impact
    assert guard.threshold_scale(1.0 + latence) == config.self_listen.hit_threshold_boost
    assert guard.threshold_scale(1.0 + latence + 0.05) > 1.0
    assert guard.threshold_scale(1.0 + latence + 0.5) == 1.0  # bien apres


def test_les_impacts_passes_sont_oublies(guard):
    """La file d'impacts ne doit pas grandir indefiniment sur une longue session."""
    guard.set_drone(110.0, (0,))
    for time_s in range(200):
        guard.note_hit(float(time_s))
    guard.threshold_scale(500.0)
    assert len(guard._hits) <= 2


# --- la boucle complete -----------------------------------------------


def _corde(freq, duree_s, samplerate, amp=1.0):
    """Note pincee : quelques harmoniques et une decroissance exponentielle."""
    t = np.arange(int(duree_s * samplerate)) / samplerate
    signal = sum(np.sin(2 * np.pi * freq * n * t) / n for n in range(1, 6))
    return amp * signal * np.exp(-3.0 * t)


def _melodie(notes, samplerate, note_s=0.5, amp=0.05):
    return np.concatenate([_corde(f, note_s, samplerate, amp) for f in notes])


def _nappe(freq, intervals, duree_s, samplerate, amp=0.08):
    """Ce que le haut-parleur du telephone renvoie dans son propre micro."""
    t = np.arange(int(duree_s * samplerate)) / samplerate
    out = np.zeros_like(t)
    for interval in intervals:
        voix = freq * 2 ** (interval / 12)
        for harmonic in range(1, 7):
            out += np.sin(2 * np.pi * voix * harmonic * t) / harmonic
    return amp * out / len(intervals)


GAMME_DE_DO = [
    261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25,
    493.88, 440.0, 392.0, 349.23, 329.63, 293.66,
]


def _chroma_moyen(frames, dernieres=40):
    vecteurs = [f.chroma for f in frames if f.chroma is not None][-dernieres:]
    return np.mean(vecteurs, axis=0)


def _masse(chroma, classes):
    return float(sum(chroma[c] for c in classes))


def test_le_drone_ne_prend_plus_le_dessus_sur_la_melodie(config):
    """Le scenario du telephone : le drone, plus fort que l'instrument, tenu.

    On mesure ou va le chroma — c'est lui qui decide de la tonalite. Sans
    auto-ecoute, les trois notes du drone raflent les deux tiers de la masse et
    la ligne jouee n'existe plus. Avec, le rapport s'inverse.
    """
    sr = config.audio.samplerate
    melodie = _melodie(GAMME_DE_DO, sr)
    fa_diese = pitch_class_to_hz(6, 3)
    nappe = _nappe(fa_diese, (0, 4, 7), 6.0 + len(melodie) / sr, sr)

    def chroma(self_listen: SelfListenConfig):
        engine = AnalysisEngine(config.with_overrides(self_listen=self_listen))
        engine.set_self_drone(fa_diese, (0, 4, 7))
        signal = nappe.copy()
        signal[len(signal) - len(melodie) :] += melodie
        return _chroma_moyen(engine.process_block(signal))

    accord_du_drone = (6, 10, 1)  # fa diese, la diese, do diese
    gamme_jouee = (0, 2, 4, 5, 7, 9, 11)

    sans = chroma(SelfListenConfig(enabled=False))
    avec = chroma(config.self_listen)

    assert _masse(sans, accord_du_drone) > 0.6  # le drone occupe le chroma
    assert _masse(sans, gamme_jouee) < 0.4
    assert _masse(avec, accord_du_drone) < 0.2  # il n'en reste presque rien
    assert _masse(avec, gamme_jouee) > 0.7  # la ligne jouee a repris la place


def test_le_drone_seul_ne_fait_plus_vivre_le_rythme(config):
    """L'autre moitie du symptome, et la plus audible.

    Micro qui n'entend que le haut-parleur, personne ne joue : sans
    auto-ecoute, le niveau se tient a 0.6 et la densite a 0.33, donc la couche
    rythmique tourne toute seule, donc le micro l'entend, etc.
    """
    sr = config.audio.samplerate
    la_mineur = pitch_class_to_hz(9, 2)
    nappe = _nappe(la_mineur, (0, 3, 7), 12.0, sr)

    def session(self_listen: SelfListenConfig):
        engine = AnalysisEngine(config.with_overrides(self_listen=self_listen))
        engine.set_self_drone(la_mineur, (0, 3, 7))
        return engine.process_block(nappe)[-1], engine

    sans, _ = session(SelfListenConfig(enabled=False))
    avec, engine = session(config.self_listen)

    assert sans.level > 0.5 and sans.density > config.rhythm.gate_density
    assert avec.level < 0.01
    assert avec.density < config.rhythm.gate_density  # la grille se tait
    assert avec.play_rms == 0.0
    assert engine.self_listen.play_ratio < 0.02


def test_les_impacts_emis_ne_gonflent_pas_la_densite(config):
    """Boucle numero deux : le rythme s'entend, donc s'excite lui-meme."""
    sr = config.audio.samplerate

    def densite_finale(declare: bool):
        engine = AnalysisEngine(config)
        engine.set_self_drone(110.0, (0, 3, 7))
        signal = np.zeros(int(6.0 * sr))
        for time_s in np.arange(0.5, 6.0, 0.25):  # une percu tous les 250 ms
            claquement = _corde(2400.0, 0.06, sr, amp=0.3)
            debut = int(time_s * sr)
            signal[debut : debut + len(claquement)] += claquement
            if declare:
                # L'appli connait la date d'emission ; `note_hit` y ajoute le
                # temps de faire l'aller-retour par le haut-parleur.
                engine.note_self_hit(time_s - config.self_listen.hit_latency_s)
        return engine.process_block(signal)[-1].density

    assert densite_finale(declare=True) < densite_finale(declare=False)

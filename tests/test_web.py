"""Verifie que le portage navigateur donne les memes resultats que Python.

Le risque d'un portage, c'est la derive silencieuse : deux implementations qui
divergent sans que personne ne s'en apercoive. Ces tests confrontent donc le
JavaScript aux *memes* valeurs de reference que la version Python — motifs
euclidiens connus, profils K-S ideaux, seuils de detection — dans un vrai
Chromium.

Ils sont ignores si Playwright ou son navigateur ne sont pas installes.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from halj.analysis.key import KS_MAJOR, KS_MINOR, estimate_key
from halj.generative.euclidean import euclidean_pattern, pulses_for_density

PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"


@pytest.fixture(scope="module")
def page():
    playwright = pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    chromium_path = "/opt/pw-browsers/chromium"
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                executable_path=chromium_path if Path(chromium_path).exists() else None,
                args=["--autoplay-policy=no-user-gesture-required"],
            )
        except Exception as exc:  # pragma: no cover - depend de l'environnement
            pytest.skip(f"Chromium indisponible : {exc}")
        page = browser.new_page()
        erreurs = []
        page.on("pageerror", lambda e: erreurs.append(str(e)))
        page.goto(PAGE.as_uri())
        page.wait_for_function("window.HALJ !== undefined")
        assert not erreurs, f"erreurs JS au chargement : {erreurs}"
        yield page
        browser.close()


def js(page, expression):
    return page.evaluate(expression)


# --- parite des motifs euclidiens ------------------------------------


@pytest.mark.parametrize(
    "steps, pulses",
    [(8, 3), (8, 5), (16, 4), (13, 5), (16, 6), (12, 7), (9, 4), (8, 0), (8, 8)],
)
def test_les_motifs_sont_identiques_a_python(page, steps, pulses):
    attendu = euclidean_pattern(steps, pulses)
    obtenu = js(page, f"HALJ.euclideanPattern({steps}, {pulses})")
    assert obtenu == attendu


@pytest.mark.parametrize("rotation", [0, 1, 3, 7, 15])
def test_les_rotations_sont_identiques_a_python(page, rotation):
    attendu = euclidean_pattern(16, 5, rotation)
    obtenu = js(page, f"HALJ.euclideanPattern(16, 5, {rotation})")
    assert obtenu == attendu


def test_la_conversion_densite_impulsions_est_identique(page):
    attendu = [pulses_for_density(percent / 100, 2, 7) for percent in range(0, 101, 5)]
    obtenu = js(
        page,
        "Array.from({length: 21}, (_, i) => HALJ.pulsesForDensity(i * 5 / 100, 2, 7))",
    )
    assert obtenu == attendu


def test_les_motifs_des_deux_voix_correspondent(page):
    from halj.config import RhythmConfig
    from halj.generative.scheduler import RhythmScheduler

    for density in (0.0, 0.25, 0.5, 0.75, 1.0):
        scheduler = RhythmScheduler(RhythmConfig())
        scheduler.set_density(density)
        for voice in ("kick", "perc"):
            obtenu = js(
                page,
                f"HALJ.patternForVoice('{voice}', {density}, HALJ.CONFIG.rhythm)",
            )
            assert obtenu == scheduler.pattern(voice), f"{voice} @ densite {density}"


# --- parite de la detection de tonalite ------------------------------


def test_les_profils_ks_sont_les_memes(page):
    assert js(page, "HALJ.KS_MAJOR") == pytest.approx(list(KS_MAJOR))
    assert js(page, "HALJ.KS_MINOR") == pytest.approx(list(KS_MINOR))


@pytest.mark.parametrize("tonic", range(12))
def test_les_tonalites_ideales_sont_reconnues(page, tonic):
    for profile, mode in ((KS_MAJOR, "maj"), (KS_MINOR, "min")):
        rolled = np.roll(profile, tonic)
        chroma = (rolled / rolled.sum()).tolist()
        obtenu = js(page, f"HALJ.estimateKey({json.dumps(chroma)})")
        assert (obtenu["tonic"], obtenu["mode"]) == (tonic, mode)


def test_les_correlations_sont_numeriquement_identiques(page):
    rng = np.random.default_rng(4)
    chroma = rng.random(12)
    chroma /= chroma.sum()

    attendu = estimate_key(chroma)
    obtenu = js(page, f"HALJ.estimateKey({json.dumps(chroma.tolist())})")

    assert obtenu["tonic"] == attendu.key.tonic
    assert obtenu["mode"] == attendu.key.mode
    assert obtenu["confidence"] == pytest.approx(attendu.confidence, abs=1e-9)


def test_un_chroma_vide_ne_donne_rien(page):
    assert js(page, "HALJ.estimateKey(new Array(12).fill(0))") is None


def test_la_fondamentale_du_drone_est_la_meme(page):
    from halj.notes import Key

    for tonic in range(12):
        attendu = Key(tonic, "min").root_hz(octave=2)
        obtenu = js(page, f"HALJ.rootHz({{tonic: {tonic}, mode: 'min'}}, 2)")
        assert obtenu == pytest.approx(attendu, rel=1e-9)


# --- parite du chroma ------------------------------------------------


def test_le_banc_de_filtres_projette_comme_python(page, config):
    """Un spectre synthetique doit donner le meme chroma des deux cotes."""
    from halj.analysis.chroma import ChromaExtractor

    extractor = ChromaExtractor(config.audio, config.key)
    samplerate = config.audio.samplerate
    frame = config.key.frame_size
    n_bins = frame // 2 + 1
    bin_hz = samplerate / frame

    # Spectre artificiel : trois pics purs sur la, do, mi (accord de la mineur).
    spectrum = np.zeros(n_bins)
    for freq in (220.0, 261.63, 329.63, 440.0):
        spectrum[int(round(freq / bin_hz))] = 1.0

    attendu = extractor._filterbank @ spectrum
    attendu = attendu / attendu.sum()

    obtenu = js(
        page,
        f"""(() => {{
            const bank = HALJ.buildChromaFilterbank({n_bins}, {bin_hz}, HALJ.CONFIG.key);
            const spectrum = new Float32Array({n_bins});
            {"".join(
                f"spectrum[{int(round(f / bin_hz))}] = 1.0;"
                for f in (220.0, 261.63, 329.63, 440.0)
            )}
            return Array.from(HALJ.chromaFromMagnitudes(bank, spectrum));
        }})()""",
    )
    assert obtenu == pytest.approx(attendu.tolist(), abs=1e-6)
    # et la tonalite qui en decoule est bien la mineur
    assert int(np.argmax(obtenu)) == 9


# --- parite des seuils -----------------------------------------------


def test_les_constantes_critiques_sont_alignees(page, config):
    """Les réglages calibrés ne doivent pas diverger entre les deux versions."""
    cfg = js(page, "HALJ.CONFIG")
    assert cfg["key"]["spectralTilt"] == config.key.spectral_tilt
    assert cfg["key"]["binWidthScale"] == config.key.bin_width_scale
    assert cfg["key"]["fminHz"] == config.key.fmin_hz
    assert cfg["key"]["fmaxHz"] == config.key.fmax_hz
    assert cfg["key"]["binsPerOctave"] == config.key.bins_per_octave
    assert cfg["key"]["minConfidence"] == config.key.min_confidence
    assert cfg["key"]["switchMarginKS"] == config.key.switch_margin
    assert cfg["onset"]["delta"] == config.onset.delta
    assert cfg["onset"]["floor"] == config.onset.floor
    assert cfg["onset"]["minIntervalS"] == config.onset.min_interval_s
    assert cfg["onset"]["highpassHz"] == config.onset.highpass_hz
    assert cfg["energy"]["onsetRateWeight"] == config.energy.onset_rate_weight
    assert cfg["rhythm"]["bpm"] == config.rhythm.bpm
    assert cfg["rhythm"]["steps"] == config.rhythm.steps
    assert cfg["rhythm"]["gateDensity"] == config.rhythm.gate_density


def test_le_flux_d_onset_reste_borne(page):
    """La normalisation du flux garantit un ratio dans [0, 1]."""
    valeurs = js(
        page,
        """(() => {
            const det = new HALJ.OnsetDetector(HALJ.CONFIG.onset);
            const out = [];
            for (let trame = 0; trame < 40; trame++) {
                const mags = new Float32Array(512);
                // trame 20 : irruption brutale d'energie (attaque depuis rien)
                const amp = trame < 20 ? 1e-6 : 1.0;
                for (let i = 0; i < mags.length; i++) mags[i] = amp * (1 + 0.1 * Math.sin(i));
                det.process(mags, trame * 0.016, 0.016);
                out.push(det.flux);
            }
            return out;
        })()""",
    )
    assert all(0.0 <= v <= 1.0 for v in valeurs)
    assert max(valeurs) > 0.9  # l'attaque depuis le silence sature a 1


def test_une_note_tenue_ne_declenche_pas(page):
    """Le point dur : un spectre stable ne doit produire aucune attaque."""
    onsets = js(
        page,
        """(() => {
            const det = new HALJ.OnsetDetector(HALJ.CONFIG.onset);
            let count = 0;
            for (let trame = 0; trame < 120; trame++) {
                const mags = new Float32Array(512);
                for (let i = 0; i < mags.length; i++) {
                    // niveau stable, avec une ondulation d'interference realiste
                    mags[i] = 1 + 0.05 * Math.sin(i * 0.7 + trame * 0.9);
                }
                if (det.process(mags, trame * 0.016, 0.016)) count++;
            }
            return count;
        })()""",
    )
    assert onsets == 0


def test_la_densite_suit_le_niveau(page):
    densites = js(
        page,
        """(() => {
            return [0.001, 0.01, 0.05, 0.2, 0.8].map((amp) => {
                const e = new HALJ.EnergyTracker(HALJ.CONFIG.energy);
                for (let i = 0; i < 400; i++) e.update(amp, 0, 0.016);
                return e.density();
            });
        })()""",
    )
    assert densites == sorted(densites)
    assert densites[0] < 0.05 < densites[-1]


# --- parite de l'auto-ecoute ------------------------------------------


def test_le_plancher_reagit_comme_en_python(page, config):
    """Meme signal, meme plancher : c'est la mesure qui pilote tout le reste."""
    from halj.analysis.selfmask import FloorFollower

    rise, fall, dt = 2.0, 0.8, 0.046
    niveaux = [0.0] + [1.0] * 60 + [0.2] * 30

    attendu = FloorFollower(rise, fall)
    for niveau in niveaux:
        attendu.update(np.array([niveau]), dt)

    obtenu = js(
        page,
        f"""(() => {{
            const f = new HALJ.FloorFollower({rise}, {fall});
            for (const v of {niveaux}) f.update([v], {dt});
            return f.value[0];
        }})()""",
    )
    assert obtenu == pytest.approx(float(attendu.value[0]), rel=1e-9)


def test_le_gel_du_plancher_est_le_meme(page):
    from halj.analysis.selfmask import FloorFollower

    attendu = FloorFollower(2.0, 0.8)
    for _ in range(50):
        attendu.update(np.array([0.1]), 0.046)
    for _ in range(50):
        attendu.update(np.array([1.0]), 0.046, allow_rise=False)

    obtenu = js(
        page,
        """(() => {
            const f = new HALJ.FloorFollower(2.0, 0.8);
            for (let i = 0; i < 50; i++) f.update([0.1], 0.046);
            for (let i = 0; i < 50; i++) f.update([1.0], 0.046, false);
            return f.value[0];
        })()""",
    )
    assert obtenu == pytest.approx(float(attendu.value[0]), rel=1e-9)


def test_les_partiels_du_drone_sont_les_memes(page):
    from halj.analysis.selfmask import drone_partials

    attendu = drone_partials(110.0, (0, 3, 7), 6, 2093.0)
    obtenu = js(page, "HALJ.dronePartials(110, [0, 3, 7], 6, 2093)")
    assert obtenu == pytest.approx(attendu, rel=1e-9)


def test_le_drone_disparait_du_spectre_analyse(page, config):
    """Le resultat qui compte : apres mesure, l'appli ne s'entend plus."""
    resultat = js(
        page,
        """(() => {
            const bins = 512, binHz = 5.383;
            const guard = new HALJ.SelfListen(
                HALJ.CONFIG.selfListen, binHz, bins, 73.4, 2093);
            guard.setDrone(110, [0, 3, 7]);
            const drone = new Float64Array(bins);
            for (const f of HALJ.dronePartials(110, [0, 3, 7], 10, 2093)) {
                drone[Math.round(f / binHz)] += 1.0;
            }
            for (let i = 0; i < 400; i++) guard.cleanSpectrum(drone, 0.046);
            const propre = guard.cleanSpectrum(drone, 0.046);
            let reste = 0, total = 0;
            for (let i = 0; i < bins; i++) { reste += propre[i]; total += drone[i]; }
            return { reste: reste / total, ratio: guard.playRatio,
                     joue: guard.playing, niveau: guard.cleanRms(0.05) };
        })()""",
    )
    assert resultat["reste"] < 0.05  # le drone a disparu du spectre analyse
    assert resultat["ratio"] < config.self_listen.play_gate
    assert resultat["joue"] is False
    assert resultat["niveau"] == 0.0  # donc plus de niveau de jeu, donc plus de rythme


def test_le_seuil_d_attaque_est_releve_autour_d_un_impact(page, config):
    valeurs = js(
        page,
        """(() => {
            const guard = new HALJ.SelfListen(
                HALJ.CONFIG.selfListen, 5.383, 64, 73.4, 2093);
            guard.setDrone(110, [0]);
            guard.noteHit(1.0);
            const l = HALJ.CONFIG.selfListen.hitLatencyS;
            return [guard.thresholdScale(0.9), guard.thresholdScale(1.0 + l),
                    guard.thresholdScale(1.0 + l + 0.5)];
        })()""",
    )
    assert valeurs == [1.0, config.self_listen.hit_threshold_boost, 1.0]


# --- parite de la grille d'accords ------------------------------------


def test_les_grilles_sont_les_memes(page):
    from halj.generative.harmony import PROGRESSIONS, progression_for
    from halj.notes import Key

    assert js(page, "HALJ.PROGRESSIONS") == {
        mode: [list(grille) for grille in grilles]
        for mode, grilles in PROGRESSIONS.items()
    }
    for mode in ("min", "maj"):
        for index in range(4):
            attendu = [c.name for c in progression_for(Key(9, mode), index)]
            obtenu = js(
                page,
                f"HALJ.progressionFor({{tonic: 9, mode: '{mode}'}}, {index})"
                ".map(HALJ.chordName)",
            )
            assert obtenu == attendu


def test_le_vote_du_joueur_est_calcule_pareil(page):
    from halj.generative.harmony import Chord, chord_score

    chroma = [0.30, 0.02, 0.04, 0.03, 0.18, 0.05, 0.02, 0.16, 0.03, 0.11, 0.02, 0.04]
    for root, quality in ((9, "min"), (0, "maj"), (5, "maj"), (7, "maj")):
        attendu = chord_score(Chord(root, quality), chroma)
        obtenu = js(
            page,
            f"HALJ.chordScore({{root: {root}, quality: '{quality}'}}, {chroma})",
        )
        assert obtenu == pytest.approx(attendu, rel=1e-9)


def test_la_conduite_des_voix_est_la_meme(page):
    from halj.generative.harmony import voiced_root_hz

    precedente_py, sequence = None, []
    for _ in range(3):
        for pitch_class in (9, 5, 0, 7):
            precedente_py = voiced_root_hz(pitch_class, 2, precedente_py)
            sequence.append(precedente_py)

    obtenu = js(
        page,
        """(() => {
            let precedente = null;
            const out = [];
            for (let tour = 0; tour < 3; tour++) {
                for (const pc of [9, 5, 0, 7]) {
                    precedente = HALJ.voicedRootHz(pc, 2, precedente);
                    out.push(precedente);
                }
            }
            return out;
        })()""",
    )
    assert obtenu == pytest.approx(sequence, rel=1e-9)


def test_le_directeur_deroule_la_meme_grille(page):
    from halj.config import HarmonyConfig
    from halj.generative.harmony import HarmonyDirector
    from halj.notes import Key

    directeur = HarmonyDirector(HarmonyConfig(agency=1.0), steps_per_bar=16)
    directeur.observe_key(Key(9, "min"))
    attendu = [directeur.on_step(step).chord.name for step in (16, 32, 48, 64, 80)]

    obtenu = js(
        page,
        """(() => {
            const cfg = Object.assign({}, HALJ.CONFIG.harmony, {agency: 1.0});
            const d = new HALJ.HarmonyDirector(cfg, 16);
            d.observeKey({tonic: 9, mode: 'min'}, 0);
            return [16, 32, 48, 64, 80].map((s) => HALJ.chordName(d.onStep(s).chord));
        })()""",
    )
    assert obtenu == attendu


def test_le_directeur_cede_au_meme_moment(page):
    """Le point de bascule entre mener et suivre doit etre identique des deux cotes."""
    from halj.config import HarmonyConfig
    from halj.generative.harmony import HarmonyDirector
    from halj.notes import Key

    vote = [0.0] * 12
    for pitch_class in (7, 7, 11, 2):  # sol majeur, fondamentale appuyee
        vote[pitch_class] += 0.25

    decisions = []
    for agency in (0.0, 0.25, 0.5, 0.75, 1.0):
        directeur = HarmonyDirector(HarmonyConfig(agency=agency), steps_per_bar=16)
        directeur.observe_key(Key(9, "min"))
        directeur.observe_chroma(vote)
        decisions.append(directeur.on_step(16).decision)

    obtenu = js(
        page,
        f"""(() => {{
            return [0.0, 0.25, 0.5, 0.75, 1.0].map((agency) => {{
                const cfg = Object.assign({{}}, HALJ.CONFIG.harmony, {{agency}});
                const d = new HALJ.HarmonyDirector(cfg, 16);
                d.observeKey({{tonic: 9, mode: 'min'}}, 0);
                d.observeChroma({vote});
                return d.onStep(16).decision;
            }});
        }})()""",
    )
    assert obtenu == decisions


def test_les_constantes_d_auto_ecoute_sont_alignees(page, config):
    cfg = js(page, "HALJ.CONFIG")
    self_listen, harmony = cfg["selfListen"], cfg["harmony"]
    assert self_listen["subtraction"] == config.self_listen.subtraction
    assert self_listen["playGate"] == config.self_listen.play_gate
    assert self_listen["floorRiseTauS"] == config.self_listen.floor_rise_tau_s
    assert self_listen["floorFallTauS"] == config.self_listen.floor_fall_tau_s
    assert self_listen["playSmoothTauS"] == config.self_listen.play_smooth_tau_s
    assert self_listen["bandSemitones"] == config.self_listen.band_semitones
    assert self_listen["partials"] == config.self_listen.partials
    assert self_listen["hitThresholdBoost"] == config.self_listen.hit_threshold_boost
    assert self_listen["startupS"] == config.self_listen.startup_s
    assert harmony["agency"] == config.harmony.agency
    assert harmony["rootWeight"] == config.harmony.root_weight
    assert harmony["followMarginMax"] == config.harmony.follow_margin_max
    assert harmony["minVote"] == config.harmony.min_vote
    assert harmony["barsPerChord"] == config.harmony.bars_per_chord

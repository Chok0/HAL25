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

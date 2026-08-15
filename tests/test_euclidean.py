import pytest

from halj.generative.euclidean import euclidean_pattern, pulses_for_density


def as_string(pattern):
    return "".join("x" if step else "." for step in pattern)


@pytest.mark.parametrize(
    "steps, pulses, expected",
    [
        (8, 3, "x..x..x."),  # tresillo
        (8, 5, "x.xx.xx."),  # cinquillo
        (16, 4, "x...x...x...x..."),  # quatre au sol
        (13, 5, "x..x.x..x.x.."),  # motif impair (Macedoine)
        (8, 0, "........"),
        (8, 8, "xxxxxxxx"),
        (1, 1, "x"),
    ],
)
def test_motifs_de_reference(steps, pulses, expected):
    assert as_string(euclidean_pattern(steps, pulses)) == expected


@pytest.mark.parametrize("steps, pulses", [(16, 5), (12, 7), (9, 4), (32, 13)])
def test_le_nombre_d_impulsions_est_respecte(steps, pulses):
    assert sum(euclidean_pattern(steps, pulses)) == pulses


@pytest.mark.parametrize("steps, pulses", [(16, 5), (12, 7), (8, 3)])
def test_la_rotation_preserve_le_contenu(steps, pulses):
    base = euclidean_pattern(steps, pulses)
    for rotation in range(steps):
        rotated = euclidean_pattern(steps, pulses, rotation)
        assert sum(rotated) == pulses
        assert rotated == base[rotation:] + base[:rotation]


def test_la_rotation_est_circulaire():
    assert euclidean_pattern(8, 3, 8) == euclidean_pattern(8, 3, 0)


def test_repartition_la_plus_uniforme_possible():
    # Un motif euclidien n'admet au plus que deux longueurs d'intervalle entre
    # impulsions consecutives, et elles sont adjacentes : c'est sa definition.
    pattern = euclidean_pattern(16, 6)
    positions = [i for i, step in enumerate(pattern) if step]
    gaps = [
        (positions[(i + 1) % len(positions)] - positions[i]) % 16
        for i in range(len(positions))
    ]
    assert len(set(gaps)) <= 2
    assert max(gaps) - min(gaps) <= 1


@pytest.mark.parametrize("steps, pulses", [(8, 9), (8, -1), (0, 0)])
def test_parametres_invalides(steps, pulses):
    with pytest.raises(ValueError):
        euclidean_pattern(steps, pulses)


def test_densite_vers_impulsions():
    assert pulses_for_density(0.0, 2, 7) == 2
    assert pulses_for_density(1.0, 2, 7) == 7
    assert pulses_for_density(0.5, 2, 8) == 5
    # les densites hors bornes sont clampees, pas rejetees
    assert pulses_for_density(-3.0, 2, 7) == 2
    assert pulses_for_density(42.0, 2, 7) == 7


def test_la_densite_est_monotone():
    previous = -1
    for percent in range(0, 101):
        pulses = pulses_for_density(percent / 100, 0, 11)
        assert pulses >= previous
        previous = pulses

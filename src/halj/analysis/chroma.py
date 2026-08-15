"""Extraction du vecteur chroma (12 classes de hauteur).

Un seul pipeline chroma gere accords et notes seules : c'est un choix deja
tranche dans la note de cadrage. Le chroma replie toutes les octaves, donc une
note isolee et l'accord qui la contient nourrissent le meme espace a 12
dimensions — pas besoin de savoir si le guitariste joue mono ou poly.

Implementation par defaut : projection log-frequentielle du spectre STFT sur
les 12 classes, en pur NumPy (aucune dependance lourde). `librosa` est utilise
si disponible et explicitement demande, via `ChromaExtractor.create`.
"""

from __future__ import annotations

import numpy as np

from ..config import AudioConfig, KeyConfig


class ChromaExtractor:
    """Spectre magnitude -> vecteur chroma normalise (12,).

    La matrice de projection est precalculee une fois : le calcul par trame se
    reduit a une compression logarithmique puis un produit matrice-vecteur.
    """

    def __init__(self, audio: AudioConfig, key: KeyConfig):
        self.audio = audio
        self.key = key
        self.frame_size = key.frame_size
        self._window = np.hanning(self.frame_size).astype(np.float64)
        self._filterbank = self._build_filterbank()

    # -- construction --------------------------------------------------

    def _build_filterbank(self) -> np.ndarray:
        """Matrice (12, n_bins) projetant les bins FFT sur les classes.

        Chaque bin FFT est place sur l'echelle des demi-tons, puis reparti sur
        les classes voisines par une fenetre gaussienne. La largeur decoule de
        `bins_per_octave` : plus il y a de bins par octave, plus la reponse est
        etroite, donc plus on tolere un accordage legerement flottant sans
        etaler l'energie sur le demi-ton voisin.
        """
        n_bins = self.frame_size // 2 + 1
        freqs = np.fft.rfftfreq(self.frame_size, 1.0 / self.audio.samplerate)

        fb = np.zeros((12, n_bins), dtype=np.float64)
        band = (freqs >= self.key.fmin_hz) & (freqs <= self.key.fmax_hz)
        usable = np.where(band)[0]
        if usable.size == 0:
            raise ValueError(
                "aucun bin FFT dans [fmin_hz, fmax_hz] : "
                "augmenter frame_size ou elargir la bande"
            )

        # position en demi-tons depuis C0, puis distance a chaque classe.
        midi = 69.0 + 12.0 * np.log2(freqs[usable] / 440.0)
        semitone_width = 12.0 / self.key.bins_per_octave  # en demi-tons
        sigma = max(semitone_width, 0.25) * self.key.bin_width_scale

        for pitch_class in range(12):
            # distance circulaire au demi-ton de la classe, dans [-6, 6].
            dist = (midi - pitch_class + 6.0) % 12.0 - 6.0
            weights = np.exp(-0.5 * (dist / sigma) ** 2)
            fb[pitch_class, usable] = weights

        # Compensation spectrale en 1/f^tilt : sans elle, les harmoniques
        # aigues d'une corde grave pesent autant que sa fondamentale et
        # deplacent la tonalite percue vers la tierce (cf. `spectral_tilt`).
        tilt = np.ones(n_bins, dtype=np.float64)
        tilt[usable] = (
            self.key.fmin_hz / np.maximum(freqs[usable], 1e-9)
        ) ** self.key.spectral_tilt
        fb *= tilt

        norms = fb.sum(axis=1, keepdims=True)
        return fb / np.maximum(norms, 1e-12)

    # -- calcul --------------------------------------------------------

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.process(frame)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Retourne le chroma (12,) d'une trame temporelle.

        Le vecteur est normalise en norme L1 ; s'il est nul (silence), il est
        retourne tel quel et l'appelant decide quoi en faire.
        """
        if frame.shape[0] != self.frame_size:
            raise ValueError(
                f"trame de {frame.shape[0]} echantillons, "
                f"{self.frame_size} attendus"
            )
        # Pas de compression logarithmique ici, contrairement a l'usage
        # courant : elle remonte les partiels faibles, donc exactement les
        # harmoniques aigues que la compensation en 1/f^tilt cherche a
        # attenuer. Mesure a l'appui, elle degrade la detection de moitie.
        spectrum = np.abs(np.fft.rfft(frame.astype(np.float64) * self._window))
        chroma = self._filterbank @ spectrum
        total = chroma.sum()
        if total <= 1e-12:
            return np.zeros(12, dtype=np.float64)
        return chroma / total

    # -- backends optionnels -------------------------------------------

    @staticmethod
    def create(audio: AudioConfig, key: KeyConfig, backend: str = "numpy"):
        """Fabrique l'extracteur : "numpy" (defaut) ou "librosa" (CQT)."""
        if backend == "numpy":
            return ChromaExtractor(audio, key)
        if backend == "librosa":
            return LibrosaChromaExtractor(audio, key)
        raise ValueError(f"backend chroma inconnu : {backend}")


class LibrosaChromaExtractor(ChromaExtractor):
    """Variante CQT via librosa : meilleure resolution dans le grave.

    Reservee a l'analyse hors ligne d'une prise : mesuree autour de x0.2 du
    temps reel, soit ~150 fois plus lente que le backend NumPy, elle ne peut
    pas tenir le direct. Son interet est d'arbitrer la qualite du backend par
    defaut sur du materiel reel, pas de le remplacer.
    """

    def __init__(self, audio: AudioConfig, key: KeyConfig):
        super().__init__(audio, key)
        import librosa  # import tardif : dependance optionnelle

        self._librosa = librosa
        self._n_octaves = int(np.ceil(np.log2(key.fmax_hz / key.fmin_hz)))
        self._n_cqt_bins = self._n_octaves * key.bins_per_octave
        self._bin_freqs = librosa.cqt_frequencies(
            n_bins=self._n_cqt_bins,
            fmin=key.fmin_hz,
            bins_per_octave=key.bins_per_octave,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] != self.frame_size:
            raise ValueError(
                f"trame de {frame.shape[0]} echantillons, "
                f"{self.frame_size} attendus"
            )
        cqt = np.abs(
            self._librosa.cqt(
                y=frame.astype(np.float32),
                sr=self.audio.samplerate,
                hop_length=self.frame_size,
                fmin=self.key.fmin_hz,
                n_bins=self._n_cqt_bins,
                bins_per_octave=self.key.bins_per_octave,
            )
        )
        if cqt.size == 0:
            return np.zeros(12, dtype=np.float64)
        # Meme choix que le backend NumPy : pas de compression logarithmique,
        # mais une compensation en 1/f^tilt. La CQT ayant des bins a largeur
        # relative constante, la compensation porte sur la frequence centrale
        # de chaque bin.
        magnitudes = cqt.mean(axis=1) * (self.key.fmin_hz / self._bin_freqs) ** (
            self.key.spectral_tilt
        )
        per_semitone = self.key.bins_per_octave // 12
        chroma = np.zeros(12, dtype=np.float64)
        for idx, value in enumerate(magnitudes):
            semitone = (idx // per_semitone) % 12
            # bin 0 = fmin_hz : on realigne sur C.
            offset = int(round(self._librosa.hz_to_midi(self.key.fmin_hz))) % 12
            chroma[(semitone + offset) % 12] += value
        total = chroma.sum()
        if total <= 1e-12:
            return np.zeros(12, dtype=np.float64)
        return chroma / total

import librosa
import numpy as np
from librosa import feature
from dataclasses import dataclass, field

"""
audio/analyzer.py
=================
AudioAnalyzer — single class that owns all audio loading, feature extraction,
and per-frame data packaging for the music visualizer pipeline.

Combines:
  - Clean OOP structure (class AudioAnalyzer, dataclass AudioFrame)
  - Log-scaled spectrogram  (matches human hearing; bass stays visible)
  - 5 perceptual band energies (sub-bass / bass / mid / high-mid / treble)
  - Beat detection           (binary pulse, perfect for lightning triggers)
  - EMA smoothing            (prevents terrain jitter on transients)
  - Native sample rate       (no quality loss from resampling)
  - Array length alignment   (all arrays guaranteed to be same length)
"""



# ---------------------------------------------------------------------------
# AudioFrame — one object per video frame, passed to terrain + lightning
# ---------------------------------------------------------------------------

@dataclass
class AudioFrame:
    index:        int                        # frame number
    t:            float                      # timestamp in seconds
    frequencies:  np.ndarray                 # log-scaled spectrum (n_bins,)
    rms:          float                      # overall amplitude [0, 1]
    onset:        float                      # onset strength    [0, 1]
    is_beat:      bool                       # True on drum hits
    bands: dict = field(default_factory=dict)
    # bands keys: 'sub_bass', 'bass', 'mid', 'high_mid', 'treble' — each [0, 1]

    @property
    def low(self)      -> float: return self.bands.get("sub_bass", 0.0) * 0.5 \
                                      + self.bands.get("bass",    0.0) * 0.5
    @property
    def mid(self)      -> float: return self.bands.get("mid",     0.0)
    @property
    def high(self)     -> float: return self.bands.get("high_mid",0.0) * 0.5 \
                                      + self.bands.get("treble",  0.0) * 0.5


# ---------------------------------------------------------------------------
# AudioAnalyzer
# ---------------------------------------------------------------------------

class AudioAnalyzer:
    """
    Loads an audio file and extracts per-frame features aligned to a
    target video frame rate.

    Usage
    -----
        analyzer = AudioAnalyzer("song.mp3", fps=60)
        analyzer.load()
        analyzer.extract_features()

        for frame in analyzer.frames:
            print(frame.is_beat, frame.low, frame.high)

        # Or access raw arrays directly:
        print(analyzer.spectrogram.shape)   # (num_frames, n_bins)
        print(analyzer.band_energies.keys())
    """

    # Perceptual frequency bands (Hz) — used to drive separate visual parameters
    BAND_RANGES = {
        "sub_bass":  (20,   80),
        "bass":      (80,   250),
        "mid":       (250,  2000),
        "high_mid":  (2000, 6000),
        "treble":    (6000, 20000),
    }

    def __init__(
        self,
        file_path: str,
        fps:       int   = 60,
        n_fft:     int   = 2048,
        ema_alpha: float = 0.3,
    ):
        """
        Parameters
        ----------
        file_path : str
            Path to audio file (mp3 / wav / flac / ogg …).
        fps : int
            Target video frame rate.  Determines hop_length so every
            returned frame aligns to exactly one video frame.
        n_fft : int
            FFT window size.  2048 → 1025 frequency bins (good default).
        ema_alpha : float
            Smoothing strength for per-band EMA.
            0.3 = light smoothing (recommended for visualizers).
            0.1 = heavy smoothing.  1.0 = no smoothing (raw / jittery).
        """
        self.file_path = file_path
        self.fps       = fps
        self.n_fft     = n_fft
        self.ema_alpha = ema_alpha

        # Populated by load()
        self.y:  np.ndarray | None = None
        self.sr: int | None        = None

        # Populated by extract_features()
        self.spectrogram:   np.ndarray | None       = None   # (frames, bins)
        self.band_energies: dict[str, np.ndarray]   = {}     # each (frames,)
        self.rms:           np.ndarray | None        = None   # (frames,)
        self.onset_env:     np.ndarray | None        = None   # (frames,)
        self.beat_array:    np.ndarray | None        = None   # (frames,) binary
        self.frames:        list[AudioFrame]         = []
        self.num_frames:    int                      = 0

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------

    def load(self) -> "AudioAnalyzer":
        """
        Load the audio file at the native sample rate (avoids quality loss
        from resampling to librosa's default 22050 Hz).
        Returns self for method chaining: analyzer.load().extract_features()
        """
        print(f"[AudioAnalyzer] Loading: {self.file_path}")
        self.y, self.sr = librosa.load(self.file_path, sr=None)
        duration = len(self.y) / self.sr
        print(f"[AudioAnalyzer] Sample rate: {self.sr} Hz | Duration: {duration:.1f}s")
        return self

    # ------------------------------------------------------------------
    # Step 2 — Extract features
    # ------------------------------------------------------------------

    def extract_features(self) -> "AudioAnalyzer":
        """
        Run all analysis passes and build self.frames (list of AudioFrame).
        Must be called after load().
        Returns self for method chaining.
        """
        if self.y is None:
            raise RuntimeError("Call load() before extract_features().")

        hop_length = int(self.sr / self.fps)

        self._compute_spectrogram(hop_length)
        self._compute_band_energies(hop_length)
        self._compute_rms(hop_length)
        self._compute_onset_and_beats(hop_length)
        self._build_frames()

        print(f"[AudioAnalyzer] Done. {self.num_frames} frames | "
              f"{int(self.beat_array.sum())} beats detected")
        return self

    # ------------------------------------------------------------------
    # Public convenience — get a single frame safely
    # ------------------------------------------------------------------

    def get_frame(self, index: int) -> AudioFrame:
        """Return the AudioFrame for a given index (clamped to valid range)."""
        return self.frames[max(0, min(index, self.num_frames - 1))]

    def frame_at_time(self, t: float) -> AudioFrame:
        """Return the AudioFrame closest to a timestamp in seconds."""
        return self.get_frame(int(t * self.fps))

    # ------------------------------------------------------------------
    # Private analysis passes
    # ------------------------------------------------------------------

    def _compute_spectrogram(self, hop_length: int) -> None:
        """
        STFT → log-amplitude spectrogram.

        Why log scale?
        Raw STFT amplitudes span several orders of magnitude; high-frequency
        bins with tiny amplitudes become invisible next to loud low bins.
        Converting to dB (log scale) compresses this range so bass and treble
        features both produce visible terrain height, matching how human
        hearing perceives loudness.
        """
        print("[AudioAnalyzer] Computing spectrogram …")
        stft        = librosa.stft(self.y, n_fft=self.n_fft, hop_length=hop_length)
        magnitude   = np.abs(stft)

        # ref=np.max normalises loudest bin to 0 dB
        log_spec    = librosa.amplitude_to_db(magnitude, ref=np.max)
        log_spec   -= log_spec.min()          # shift to [0, ∞)

        # Transpose: (bins, frames) → (frames, bins)
        self.spectrogram = log_spec.T.astype(np.float32)
        self.num_frames  = self.spectrogram.shape[0]

    def _compute_band_energies(self, hop_length: int) -> None:
        """
        Collapse the full spectrogram into 5 perceptual bands.

        Each band becomes a (num_frames,) array normalised to [0, 1] and
        EMA-smoothed.  These drive separate visual parameters:
            sub_bass  → major terrain heaves, camera shake on kick
            bass      → general terrain height
            mid       → mid-ground colour temperature
            high_mid  → lightning spawn rate
            treble    → particle flicker / emissive intensity
        """
        # Re-compute magnitude on the raw STFT for band energy (not log-scaled)
        stft      = librosa.stft(self.y, n_fft=self.n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)           # (bins, frames)
        freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        for name, (lo, hi) in self.BAND_RANGES.items():
            mask = (freq_bins >= lo) & (freq_bins <= hi)
            if mask.sum() == 0:
                self.band_energies[name] = np.zeros(self.num_frames, dtype=np.float32)
                continue

            raw = magnitude[mask, :].mean(axis=0).astype(np.float32)
            raw = self._align_length(raw, self.num_frames)
            raw = self._smooth_ema(raw, alpha=self.ema_alpha)

            peak = raw.max()
            self.band_energies[name] = raw / peak if peak > 0 else raw

    def _compute_rms(self, hop_length: int) -> None:
        """
        Root-mean-square energy — overall loudness per frame.
        Normalised to [0, 1] and EMA-smoothed.
        """
        rms = librosa.feature.rms(y=self.y, hop_length=hop_length)[0]
        rms = self._align_length(rms.astype(np.float32), self.num_frames)
        rms = self._smooth_ema(rms, alpha=self.ema_alpha)
        peak = rms.max()
        self.rms = rms / peak if peak > 0 else rms

    def _compute_onset_and_beats(self, hop_length: int) -> None:
        """
        Onset strength envelope + binary beat frame array.

        onset_env  — continuous [0, 1] signal; use for smooth reactions
                     (e.g. bloom intensity, terrain glow)
        beat_array — 1.0 on beat frames, 0.0 elsewhere; use for instant
                     triggers (e.g. lightning spawn, camera flash)
        """
        print("[AudioAnalyzer] Detecting beats …")
        raw_onset = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=hop_length
        )
        raw_onset      = self._align_length(raw_onset.astype(np.float32), self.num_frames)
        self.onset_env = raw_onset / (raw_onset.max() + 1e-6)

        _, beat_indices = librosa.beat.beat_track(
            y=self.y, sr=self.sr, hop_length=hop_length
        )
        self.beat_array = np.zeros(self.num_frames, dtype=np.float32)
        valid = beat_indices[beat_indices < self.num_frames]
        self.beat_array[valid] = 1.0

    def _build_frames(self) -> None:
        """Assemble all per-frame arrays into AudioFrame objects."""
        self.frames = []
        for i in range(self.num_frames):
            self.frames.append(AudioFrame(
                index       = i,
                t           = i / self.fps,
                frequencies = self.spectrogram[i],
                rms         = float(self.rms[i]),
                onset       = float(self.onset_env[i]),
                is_beat     = bool(self.beat_array[i]),
                bands       = {k: float(v[i]) for k, v in self.band_energies.items()},
            ))

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_ema(arr: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Exponential Moving Average.  Each output frame blends the new raw
        value with the previous smoothed value.

        alpha = 1.0 → no smoothing (raw values, jittery terrain)
        alpha = 0.3 → light smoothing (recommended)
        alpha = 0.1 → heavy smoothing (very fluid, slow to react)
        """
        out    = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _align_length(arr: np.ndarray, target: int) -> np.ndarray:
        """Trim or zero-pad a 1-D array to exactly target elements."""
        if len(arr) >= target:
            return arr[:target]
        return np.pad(arr, (0, target - len(arr)))


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_song.mp3"

    analyzer = AudioAnalyzer(path, fps=60).load().extract_features()

    print(f"\nSpectrogram shape : {analyzer.spectrogram.shape}")
    print(f"Band keys         : {list(analyzer.band_energies.keys())}")
    print(f"Total frames      : {analyzer.num_frames}")

    f = analyzer.get_frame(100)
    print(f"\nFrame 100 — t={f.t:.2f}s  rms={f.rms:.3f}  "
          f"beat={f.is_beat}  low={f.low:.3f}  mid={f.mid:.3f}  high={f.high:.3f}")
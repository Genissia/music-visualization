"""
audio_process.py
----------------
Handles all audio analysis for the music visualizer.

Improvements over v1:
  - Log-scaled spectrogram  → matches how human hearing works (bass feels as
    "big" as treble rather than being swamped by high-frequency bins).
  - Per-band energy arrays  → sub-bass / bass / mid / high-mid / treble each
    drive separate visual parameters instead of everything using one flat spectrum.
  - Beat / onset detection  → gives us a sharp 0-or-1 pulse every time the
    drums hit, perfect for flashing the peak colour or a camera shake.
  - Smoothed energy history → exponential moving average prevents the terrain
    from jittering frame-to-frame on transients.
"""

import librosa
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_audio_frames(audio_path: str, fps: int = 60, n_fft: int = 2048):
    """
    Load an audio file and extract per-frame frequency + beat data.

    Parameters
    ----------
    audio_path : str
        Path to any audio format librosa supports (mp3, wav, flac, ogg …).
    fps : int
        Target video frame rate.  The hop_length is derived from this so
        every returned row corresponds to exactly one video frame.
    n_fft : int
        FFT window size.  2048 gives 1025 frequency bins at any sample rate.

    Returns
    -------
    spectrogram_frames : np.ndarray, shape (num_frames, n_fft//2 + 1)
        Log-amplitude spectrogram, one row per video frame.
    band_energies : dict[str, np.ndarray]
        Per-frame scalar energy for five perceptual frequency bands.
        Keys: 'sub_bass', 'bass', 'mid', 'high_mid', 'treble'
    beat_frames : np.ndarray, shape (num_frames,)
        Binary array – 1.0 on beat frames, 0.0 elsewhere.
    onset_env : np.ndarray, shape (num_frames,)
        Continuous onset-strength envelope, normalised to [0, 1].
    sr : int
        Sample rate of the loaded audio.
    """
    print(f"[audio] Loading: {audio_path}")
    # sr=None keeps the file's native sample rate (avoids quality loss from resampling)
    y, sr = librosa.load(audio_path, sr=None)
    print(f"[audio] Sample rate: {sr} Hz  |  Duration: {len(y)/sr:.1f}s")

    # Each video frame covers exactly (sr / fps) audio samples
    hop_length = int(sr / fps)

    # ------------------------------------------------------------------
    # 1. Short-Time Fourier Transform → magnitude spectrogram
    # ------------------------------------------------------------------
    print("[audio] Computing STFT …")
    stft_data   = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude   = np.abs(stft_data)          # complex → real amplitude

    # Convert to decibels (log scale).  ref=np.max normalises the loudest
    # bin to 0 dB so quieter passages still have visible terrain features.
    log_spec    = librosa.amplitude_to_db(magnitude, ref=np.max)

    # Shift to [0, ∞) so heights are never negative
    log_spec   -= log_spec.min()

    # Rows = frequencies, cols = time → transpose so rows = video frames
    spectrogram_frames = log_spec.T.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Per-band energy  (one scalar per frame per band)
    # ------------------------------------------------------------------
    # Frequency resolution of each bin
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Define perceptual bands in Hz
    band_ranges = {
        "sub_bass":  (20,   80),
        "bass":      (80,   250),
        "mid":       (250,  2000),
        "high_mid":  (2000, 6000),
        "treble":    (6000, 20000),
    }

    band_energies = {}
    for name, (lo, hi) in band_ranges.items():
        # Boolean mask selects the bins that fall inside this band
        mask = (freq_bins >= lo) & (freq_bins <= hi)
        if mask.sum() == 0:
            # Edge case: band falls outside the representable frequency range
            band_energies[name] = np.zeros(spectrogram_frames.shape[0], dtype=np.float32)
            continue

        # Mean energy across the band bins, then smooth with a small EMA
        raw_energy = magnitude[mask, :].mean(axis=0).astype(np.float32)  # shape: (num_frames,)
        band_energies[name] = _smooth_ema(raw_energy, alpha=0.3)

    # Normalise each band to [0, 1] independently so they're easy to use
    # as multipliers in terrain_gen without extra scaling there.
    for name in band_energies:
        e = band_energies[name]
        peak = e.max()
        if peak > 0:
            band_energies[name] = e / peak

    # ------------------------------------------------------------------
    # 3. Beat / onset detection
    # ------------------------------------------------------------------
    print("[audio] Detecting beats …")
    # onset_envelope measures how much "new energy" appears each frame
    onset_env_raw = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Align onset envelope length with spectrogram frames
    num_frames    = spectrogram_frames.shape[0]
    onset_env_raw = _align_length(onset_env_raw, num_frames)

    # Normalise onset envelope
    onset_env = onset_env_raw / (onset_env_raw.max() + 1e-6)

    # Detect beat frame indices using librosa's beat tracker
    _, beat_indices = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    # Convert to a binary frame array
    beat_frames = np.zeros(num_frames, dtype=np.float32)
    valid_beats = beat_indices[beat_indices < num_frames]
    beat_frames[valid_beats] = 1.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"[audio] Done!  frames={num_frames}  bins={spectrogram_frames.shape[1]}")
    print(f"[audio] Beats detected: {int(beat_frames.sum())}")

    return spectrogram_frames, band_energies, beat_frames, onset_env, sr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _smooth_ema(arr: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Exponential Moving Average smoothing.

    alpha=1.0 → no smoothing (raw values)
    alpha=0.1 → very heavy smoothing (slow response)
    alpha=0.3 → light smoothing (recommended for visualizers)
    """
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        # Each output frame blends the new value with the previous output
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _align_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad a 1-D array to exactly target_len elements."""
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)))


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "acdc_Thunderstruck-[AudioTrimmer.com].mp3"

    try:
        spec, bands, beats, onset, sr = extract_audio_frames(audio_file, fps=60)
        print(f"\nSpectrogram shape : {spec.shape}")
        print(f"Band energy keys  : {list(bands.keys())}")
        print(f"Beat frames total : {int(beats.sum())}")
        print(f"Onset env range   : [{onset.min():.3f}, {onset.max():.3f}]")
    except FileNotFoundError:
        print(f"File not found: '{audio_file}'.  Pass your audio file as an argument.")
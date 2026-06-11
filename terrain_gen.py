"""
terrain_gen.py
--------------
GPU-accelerated 3D terrain generation driven by audio analysis.

Improvements over v1 & v2:
  - 100% GPU Noise Generation → Replaced CPU-bound Perlin noise with CuPy-native
    Fractional Brownian Motion (fBm) using layered sine/cosine waves.
  - Zero Host-to-Device Bottlenecks → No more .tolist() loops. The entire mesh
    is calculated simultaneously on the GPU in microseconds.
  - Multi-octave noise layers → a "macro" landscape layer plus a "detail"
    layer create richer terrain using fast matrix math.
  - Per-band terrain shaping  → sub-bass lifts broad mountain ranges, treble
    adds fine surface ripples, beats cause a sharp terrain "pop".
  - Temporal smoothing        → heights lerp from the previous frame so the
    terrain morphs smoothly rather than jumping between frames.
"""

import numpy as np
import cupy as cp
import time

# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------
# Previous frame's Y height map for temporal smoothing
_prev_Y = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_terrain_frame(
    audio_frame:   np.ndarray,
    band_energies: dict,
    beat_pulse:    float,
    grid_size:     int   = 100,
    noise_scale:   float = 0.05,
    amplitude:     float = 5.0,
    frame_index:   int   = 0,
    smooth_alpha:  float = 0.4,
) -> tuple:
    """
    Build one frame of the 3-D terrain mesh entirely on the GPU.
    """
    global _prev_Y

    # ------------------------------------------------------------------
    # 1. Base XZ grid  (−1 … +1 on both axes, centred at origin)
    # ------------------------------------------------------------------
    x = cp.linspace(-1, 1, grid_size)
    z = cp.linspace(-1, 1, grid_size)
    X, Z = cp.meshgrid(x, z)

    # ------------------------------------------------------------------
    # 2. Audio-driven frequency ridges
    # ------------------------------------------------------------------
    original_bins  = np.linspace(0, 1, len(audio_frame))
    target_bins    = np.linspace(0, 1, grid_size)
    resized_audio  = np.interp(target_bins, original_bins, audio_frame)

    resized_audio  = np.log1p(resized_audio)

    # Tile so each frequency "ridge" runs along the whole Z direction
    Y_audio = cp.tile(cp.array(resized_audio, dtype=cp.float32), (grid_size, 1))

    # Scale by amplitude and additionally boost with bass energy
    bass_boost = 1.0 + band_energies.get("bass", 0.0) * 0.6
    Y_audio   *= amplitude * bass_boost

    # ------------------------------------------------------------------
    # 3. Procedural GPU Noise (two layers, pure matrix math)
    # ------------------------------------------------------------------
    time_offset = frame_index * 0.008   # controls scrolling speed
    
    # Calculate effective scale to map your old 0..grid_size scale to the new -1..1 grid
    effective_scale = (grid_size * noise_scale) / 2.0

    # Macro layer (2 octaves for broad landscape shapes)
    Y_macro  = _get_gpu_fractal_noise(X, Z, time_offset, effective_scale * 1.0, octaves=2)
    
    # Detail layer (6 octaves for fine surface texture)
    Y_detail = _get_gpu_fractal_noise(X, Z, time_offset * 1.5, effective_scale * 3.5, octaves=6)

    # Weight the two layers
    treble_boost = 1.0 + band_energies.get("treble", 0.0) * 0.8
    Y_noise = Y_macro * 2.5 + Y_detail * 0.8 * treble_boost

    # ------------------------------------------------------------------
    # 4. Beat-driven terrain pop
    # ------------------------------------------------------------------
    if beat_pulse > 0.5:
        cx, cz     = 0.0, 0.0                  # centre of the thump
        dist_sq    = (X - cx) ** 2 + (Z - cz) ** 2
        beat_bump  = cp.exp(-dist_sq / 0.3) * band_energies.get("sub_bass", 0.5) * 3.0
        Y_noise   += beat_bump

    # ------------------------------------------------------------------
    # 5. Combine layers → raw height map
    # ------------------------------------------------------------------
    Y_raw = Y_audio + Y_noise

    # ------------------------------------------------------------------
    # 6. Temporal smoothing  (blend with previous frame's heights)
    # ------------------------------------------------------------------
    if _prev_Y is not None and _prev_Y.shape == Y_raw.shape:
        Y = smooth_alpha * Y_raw + (1.0 - smooth_alpha) * _prev_Y
    else:
        Y = Y_raw

    # Store for next frame
    _prev_Y = Y

    return X, Y, Z


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_gpu_fractal_noise(
    X: cp.ndarray, 
    Z: cp.ndarray, 
    time_offset: float, 
    scale: float, 
    octaves: int
) -> cp.ndarray:
    """
    Generates GPU-accelerated procedural noise using trigonometric Fractional 
    Brownian Motion (fBm). Replaces CPU Perlin Noise entirely.
    """
    Y = cp.zeros_like(X)
    amp = 1.0
    freq = scale
    
    for i in range(octaves):
        # Non-linear phase shifts for an organic, non-repeating look
        phase_x = time_offset * (1.0 + i * 0.5)
        phase_z = time_offset * (0.8 + i * 0.4)
        
        # Cross-combine sine and cosine waves across the meshgrid
        term1 = cp.sin(X * freq + phase_x)
        term2 = cp.cos(Z * freq + phase_z)
        term3 = cp.sin((X + Z) * freq * 0.7 - time_offset)
        
        Y += (term1 * term2 + term3 * 0.3) * amp
        
        # Standard fBm fractal iteration: half the amplitude, double the frequency
        amp *= 0.5
        freq *= 2.0
        
    return Y * 0.5


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from audio_process import extract_audio_frames
    import sys

    audio_file = sys.argv[1] if len(sys.argv) > 1 else "test_song.mp3"

    if not cp.cuda.is_available():
        print("No NVIDIA GPU found – CuPy requires CUDA.")
        raise SystemExit(1)

    print("GPU detected.  Running ultra-fast terrain generation test …")
    try:
        spec, bands, beats, onset, sr = extract_audio_frames(audio_file, fps=60)

        # Simulate per-frame band energy dicts
        band_frame_0 = {k: float(v[0]) for k, v in bands.items()}
        band_frame_100 = {k: float(v[100]) for k, v in bands.items()}

        t0 = time.time()
        X0, Y0, Z0   = create_terrain_frame(spec[0],   band_frame_0,   beats[0],   frame_index=0)
        X1, Y1, Z1   = create_terrain_frame(spec[100], band_frame_100, beats[100], frame_index=100)
        elapsed = time.time() - t0

        print(f"Two frames generated in {elapsed:.4f}s")
        print(f"Height diff (scrolling check): {float(cp.sum(cp.abs(Y0 - Y1))):.2f}")
    except Exception as exc:
        print(f"Test failed: {exc}")
        raise
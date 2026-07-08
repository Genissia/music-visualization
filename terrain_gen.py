"""
terrain_gen.py
--------------
GPU-accelerated 3D terrain generation driven by audio analysis.

ARCHITECTURE:
  - Two mountain WALLS running along the Z-axis (into the screen)
  - Left wall:  X ≈ -1   (always positive Y, rises upward)
  - Right wall: X ≈ +1   (always positive Y, rises upward)
  - Flat road:  X ≈  0   (Y = 0, driver drives here)
  - Mountains scroll toward camera along Z so road rushes at you
  - Y is ALWAYS >= 0 (no underground terrain, no mirror line)
"""

import numpy as np

try:
    import cupy as cp
    _device_count = cp.cuda.runtime.getDeviceCount()
    if _device_count == 0:
        raise RuntimeError("No CUDA devices available.")
    print("[terrain] Using NVIDIA CUDA acceleration (CuPy).")
except Exception:
    import numpy as cp
    print("[terrain] CUDA missing or AMD/Intel detected. Falling back to CPU (NumPy).")

_prev_Y = None


def create_terrain_frame(
    audio_frame:   np.ndarray,
    band_energies: dict,
    beat_pulse:    float,
    grid_size:     int   = 100,
    noise_scale:   float = 0.05,
    amplitude:     float = 4.0,
    frame_index:   int   = 0,
    smooth_alpha:  float = 0.4,
) -> tuple:
    """
    Builds one frame of the 3D canyon terrain.

    Layout (top-down view):
        X = -1.5  [LEFT WALL — mountain]
        X = -0.3  [left road edge]
        X =  0.0  [road center — camera is here]
        X = +0.3  [right road edge]
        X = +1.5  [RIGHT WALL — mountain]

        Z = +1  →  near (just behind camera)
        Z = -6  →  far  (deep into screen)
    """
    global _prev_Y

    # ----------------------------------------------------------------
    # 1. Grid — wider X range so walls are clearly off to the sides
    #    X: -1.5 to +1.5  (road at center, mountains at edges)
    #    Z:  -1  to  +1   (renderer maps this to world -6 … +1.5)
    # ----------------------------------------------------------------
    x = cp.linspace(-1.5, 1.5, grid_size)
    z = cp.linspace(-1.0, 1.0, grid_size)
    X, Z = cp.meshgrid(x, z)

    # ----------------------------------------------------------------
    # 2. Audio band multipliers
    # ----------------------------------------------------------------
    bass_boost   = 1.0 + band_energies.get("bass",    0.0) * 2.0
    mid_boost    = 1.0 + band_energies.get("mid",     0.0) * 1.2
    treble_boost = 1.0 + band_energies.get("treble",  0.0) * 0.8
    sub_bass     = band_energies.get("sub_bass", 0.0)

    # ----------------------------------------------------------------
    # 3. Wall profile — Gaussian peaks centered at X = ±wall_center
    #
    #    KEY FIX: instead of multiplicative noise (which goes negative),
    #    we build height as:
    #        wall_profile(X) × positive_noise(Z)
    #
    #    wall_profile is always >= 0  (Gaussian bell curve)
    #    positive_noise is always >= 0  (abs of sine waves)
    #
    #    This guarantees Y >= 0 everywhere → no mirror line.
    # ----------------------------------------------------------------
    wall_center = 1.35       # X position of wall center (left = -1.1, right = +1.1)
    wall_width  = 0.22      # Gaussian sigma — controls how wide each wall is

    # Gaussian bell: 1.0 at the wall center, falls to ~0 at the road
    left_profile  = cp.exp(-((X + wall_center) ** 2) / (2 * wall_width ** 2))
    right_profile = cp.exp(-((X - wall_center) ** 2) / (2 * wall_width ** 2))
    wall_profile  = left_profile + right_profile   # combined: two bumps

    # ----------------------------------------------------------------
    # 4. Scrolling Z noise — always positive
    #
    #    abs(sin(...)) gives a rippled mountain silhouette that is
    #    always >= 0, scrolling along Z as the frame advances.
    #    This is what makes the mountains animate with the music.
    # ----------------------------------------------------------------
    time_offset = frame_index * 0.018
    freq_base   = (grid_size * noise_scale) / 2.0

    # Macro layer: broad mountain peaks driven by bass
    z_macro  = (
        cp.abs(cp.sin(Z * freq_base * 0.9  - time_offset)) * 0.6 +
        cp.abs(cp.cos(Z * freq_base * 0.5  - time_offset * 0.7)) * 0.4
    ) * bass_boost

    # Detail layer: rocky surface texture driven by mid + treble
    z_detail = (
        cp.abs(cp.sin(Z * freq_base * 2.1  - time_offset * 1.4)) * 0.3 +
        cp.abs(cp.sin(Z * freq_base * 3.7  - time_offset * 1.9)) * 0.2
    ) * mid_boost * treble_boost

    # Combined height noise — always positive
    z_noise = (z_macro + z_detail) * amplitude

    # ----------------------------------------------------------------
    # 5. Final height = wall profile × z noise
    #
    #    wall_profile = 0 at road center → road stays flat
    #    wall_profile = 1 at wall center → full mountain height
    #    z_noise >= 0 always             → Y >= 0 always (no mirror)
    # ----------------------------------------------------------------
    Y = wall_profile * z_noise

    # ----------------------------------------------------------------
    # 6. Beat shockwave — pulse that radiates from road center
    #    Only adds height to the walls (wall_profile mask applied)
    # ----------------------------------------------------------------
    if beat_pulse > 0.5:
        dist_z    = cp.abs(Z)                     # distance along road
        shockwave = cp.exp(-dist_z / 0.4) * sub_bass * 2.5
        Y        += wall_profile * shockwave

    # ----------------------------------------------------------------
    # 7. Temporal smoothing — blend with previous frame to reduce jitter
    # ----------------------------------------------------------------
    if _prev_Y is not None and _prev_Y.shape == Y.shape:
        Y = smooth_alpha * Y + (1.0 - smooth_alpha) * _prev_Y
    _prev_Y = Y

    return X, Y, Z
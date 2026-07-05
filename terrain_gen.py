"""
terrain_gen.py
--------------
GPU-accelerated 3D terrain generation driven by audio analysis.

Updated for "Ultra-Smooth Wide Diorama":
  - smooth_alpha dropped to 0.08 for extremely slow, heavy fluid dynamics.
  - Base amplitude lowered slightly to prevent audio spikes from exploding 
    out of the camera frame.
"""

import numpy as np
import cupy as cp
import time

_prev_Y = None

def create_terrain_frame(
    audio_frame:   np.ndarray,
    band_energies: dict,
    beat_pulse:    float,
    grid_size:     int   = 100,
    noise_scale:   float = 0.05,
    amplitude:     float = 4.0,  # Lowered from 5.0 to keep peaks controlled
    frame_index:   int   = 0,
    smooth_alpha:  float = 0.08, # Dropped to 0.08 for maximum liquid smoothness
) -> tuple:
    
    global _prev_Y

    # 1. Base 1:1 Square Grid
    x = cp.linspace(-1, 1, grid_size)
    z = cp.linspace(-1, 1, grid_size)
    X, Z = cp.meshgrid(x, z)

    # 2. Audio-driven frequency ridges
    original_bins  = np.linspace(0, 1, len(audio_frame))
    target_bins    = np.linspace(0, 1, grid_size)
    resized_audio  = np.interp(target_bins, original_bins, audio_frame)

    resized_audio  = np.log1p(resized_audio)

    Y_audio = cp.tile(cp.array(resized_audio, dtype=cp.float32), (grid_size, 1))

    # Reduced bass_boost multiplier so the kick drum doesn't blast out of frame
    bass_boost = 1.0 + band_energies.get("bass", 0.0) * 0.4
    Y_audio   *= amplitude * bass_boost

    # 3. Procedural GPU Noise
    time_offset = frame_index * 0.008   
    effective_scale = (grid_size * noise_scale) / 2.0

    Y_macro  = _get_gpu_fractal_noise(X, Z, time_offset, effective_scale * 1.0, octaves=2)
    Y_detail = _get_gpu_fractal_noise(X, Z, time_offset * 1.5, effective_scale * 3.5, octaves=6)

    treble_boost = 1.0 + band_energies.get("treble", 0.0) * 0.8
    Y_noise = Y_macro * 2.5 + Y_detail * 0.8 * treble_boost

    # 4. Smoothened Beat-driven terrain pop
    if beat_pulse > 0.5:
        cx, cz     = 0.0, 0.0
        dist_sq    = (X - cx) ** 2 + (Z - cz) ** 2
        
        # Softened beat pop to match the new fluid look
        beat_bump  = cp.exp(-dist_sq / 1.5) * band_energies.get("sub_bass", 0.5) * 1.2
        Y_noise   += beat_bump

    # 5. Combine layers
    Y_raw = Y_audio + Y_noise

    # 6. Temporal smoothing (liquid blending)
    if _prev_Y is not None and _prev_Y.shape == Y_raw.shape:
        Y = smooth_alpha * Y_raw + (1.0 - smooth_alpha) * _prev_Y
    else:
        Y = Y_raw

    _prev_Y = Y

    return X, Y, Z


def _get_gpu_fractal_noise(X: cp.ndarray, Z: cp.ndarray, time_offset: float, scale: float, octaves: int) -> cp.ndarray:
    Y = cp.zeros_like(X)
    amp = 1.0
    freq = scale
    
    for i in range(octaves):
        phase_x = time_offset * (1.0 + i * 0.5)
        phase_z = time_offset * (0.8 + i * 0.4)
        
        term1 = cp.sin(X * freq + phase_x)
        term2 = cp.cos(Z * freq + phase_z)
        term3 = cp.sin((X + Z) * freq * 0.7 - time_offset)
        
        Y += (term1 * term2 + term3 * 0.3) * amp
        
        amp *= 0.5
        freq *= 2.0
        
    return Y * 0.5
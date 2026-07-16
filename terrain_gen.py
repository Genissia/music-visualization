"""
terrain_gen.py
--------------
Builds SLOT CANYON corridor geometry (not a heightmap).

ARCHITECTURAL CHANGE
--------------------
Old approach (heightmap): grid over (X, Z), audio drives Y (height).
    → Can only make hills viewed from above.
    → Vertical walls collapse to slivers. Cannot make a slot canyon.

New approach (corridor): grid over (Z, Y), audio drives X (wall displacement).
    → Each wall is a vertical surface running down the road.
    → Z = how far down the road    (parameter)
      Y = how high up the wall     (parameter)
      X = how far the wall bulges  (AUDIO-DRIVEN)
    → Camera sits INSIDE the corridor at low Y, looking down -Z.

Returns two meshes: left wall and right wall, each shape (N_Z, N_Y, 3).
They are completely separate surfaces — never connected.
"""

import numpy as np

try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("No CUDA devices")
    print("[terrain] Using NVIDIA CUDA (CuPy).")
except Exception:
    import numpy as cp
    print("[terrain] Falling back to CPU (NumPy).")


# ---------------------------------------------------------------------------
# Canyon shape constants — tune these to change the canyon's character
# ---------------------------------------------------------------------------
Z_NEAR       =   2.0    # road starts just behind the camera
Z_FAR        = -18.0    # road vanishes into the distance
WALL_HEIGHT  =   8.0    # walls fit inside the shot and top rims are visible

GAP_BOTTOM   =   0.70   # half-width at the floor — the narrow lane you drive
GAP_TOP      =   1.60   # half-width high up — only a slight lean outward,
                        # so the wall edges stay near-VERTICAL on screen

BULGE_AUDIO  =   0.90   # how far audio pushes the walls in/out
BULGE_NOISE  =   0.35   # baseline organic waviness (independent of audio)

_prev_L = None
_prev_R = None


def create_terrain_frame(
    audio_frame:   np.ndarray,
    band_energies: dict,
    beat_pulse:    float,
    grid_size:     int   = 150,
    n_height:      int   = 40,
    frame_index:   int   = 0,
    smooth_alpha:  float = 0.35,
    **_ignored,
) -> tuple:
    """
    Returns
    -------
    left_wall  : ndarray (n_z, n_y, 3)  — vertices of the left canyon wall
    right_wall : ndarray (n_z, n_y, 3)  — vertices of the right canyon wall
    """
    global _prev_L, _prev_R

    n_z = grid_size
    n_y = n_height

    # ------------------------------------------------------------------
    # Parameter grid.
    #   zi : 0 (near, at camera) → 1 (far, vanishing point)
    #   yi : 0 (canyon floor)    → 1 (canyon rim)
    # ------------------------------------------------------------------
    zi = cp.linspace(0.0, 1.0, n_z).reshape(n_z, 1)   # column vector
    yi = cp.linspace(0.0, 1.0, n_y).reshape(1, n_y)   # row vector

    Zg = cp.broadcast_to(zi, (n_z, n_y))
    Yg = cp.broadcast_to(yi, (n_z, n_y))

    # World Z (depth) and world Y (height) — these are pure parameters,
    # they do NOT depend on audio. Only X does.
    world_z = Z_NEAR + (Z_FAR - Z_NEAR) * Zg
    world_y = WALL_HEIGHT * Yg

    # ------------------------------------------------------------------
    # Audio → wall displacement
    # ------------------------------------------------------------------
    bass   = band_energies.get("bass",     0.0)
    sub    = band_energies.get("sub_bass", 0.0)
    mid    = band_energies.get("mid",      0.0)
    treble = band_energies.get("treble",   0.0)

    # scroll term: makes the wall shapes travel toward the camera (speed up)
    t = frame_index * 0.06

    # Depth phase — the wall carves in and out as you travel down the road.
    # Multiple frequencies give the eroded, layered look of real slot canyons.
    zf = Zg * 14.0    # spatial frequency along the road

    carve_left = (
        cp.sin(zf * 1.00 + t * 1.00) * (0.45 + bass   * 0.9) +
        cp.sin(zf * 2.30 + t * 1.45) * (0.25 + mid    * 0.5) +
        cp.sin(zf * 4.70 + t * 2.10) * (0.12 + treble * 0.3)
    )

    # Right wall uses different phase offsets so the two walls are
    # independent — they never mirror each other exactly.
    carve_right = (
        cp.sin(zf * 1.00 + t * 1.00 + 2.1) * (0.45 + bass   * 0.9) +
        cp.sin(zf * 2.30 + t * 1.45 + 4.3) * (0.25 + mid    * 0.5) +
        cp.sin(zf * 4.70 + t * 2.10 + 1.7) * (0.12 + treble * 0.3)
    )

    # Height phase — walls also undulate as they rise, which is what gives
    # slot canyons their smooth sculpted ripples rather than flat slabs.
    height_ripple_L = cp.sin(Yg * 6.0 + zf * 0.8) * 0.30
    height_ripple_R = cp.sin(Yg * 6.0 + zf * 0.8 + 3.1) * 0.30

    # ------------------------------------------------------------------
    # Base gap: canyon is narrow at the floor and flares open at the rim.
    # (Look at the reference photo — the walls lean away as they go up.)
    # ------------------------------------------------------------------
    base_gap = GAP_BOTTOM + (GAP_TOP - GAP_BOTTOM) * (Yg ** 1.4)

    # Displacement is stronger higher up (floor stays tight, rim is wild)
    bulge_scale = 0.25 + 0.75 * Yg

    # Procedural high-frequency rough rock textures (multi-octave sines)
    rock_noise_L = (
        cp.sin(Zg * 35.0 + Yg * 18.0 + t * 0.3) * 0.04 +
        cp.sin(Zg * 67.0 - Yg * 32.0 - t * 0.1) * 0.02 +
        cp.sin(Zg * 115.0 + Yg * 57.0) * 0.01
    )
    rock_noise_R = (
        cp.sin(Zg * 35.0 + Yg * 18.0 + t * 0.3 + 1.5) * 0.04 +
        cp.sin(Zg * 67.0 - Yg * 32.0 - t * 0.1 + 2.8) * 0.02 +
        cp.sin(Zg * 115.0 + Yg * 57.0 + 4.1) * 0.01
    )

    disp_L = (carve_left  * BULGE_AUDIO + height_ripple_L * BULGE_NOISE + rock_noise_L) * bulge_scale
    disp_R = (carve_right * BULGE_AUDIO + height_ripple_R * BULGE_NOISE + rock_noise_R) * bulge_scale

    # Beat: walls punch inward briefly, squeezing the corridor
    if beat_pulse > 0.5:
        squeeze = sub * 0.45 * bulge_scale
        disp_L += squeeze
        disp_R += squeeze

    # Left wall sits at NEGATIVE x, right wall at POSITIVE x.
    # Never allow a wall to cross the centre line — clamp keeps the road open.
    left_x  = -cp.maximum(base_gap + disp_L, 0.35)
    right_x =  cp.maximum(base_gap + disp_R, 0.35)

    # ------------------------------------------------------------------
    # Temporal smoothing on the X displacement only
    # ------------------------------------------------------------------
    if _prev_L is not None and _prev_L.shape == left_x.shape:
        left_x  = smooth_alpha * left_x  + (1.0 - smooth_alpha) * _prev_L
        right_x = smooth_alpha * right_x + (1.0 - smooth_alpha) * _prev_R
    _prev_L, _prev_R = left_x, right_x

    # ------------------------------------------------------------------
    # Pack positions into (n_z, n_y, 3)
    # ------------------------------------------------------------------
    left_pos  = cp.stack([left_x,  world_y, world_z], axis=-1).astype(cp.float32)
    right_pos = cp.stack([right_x, world_y, world_z], axis=-1).astype(cp.float32)

    # ------------------------------------------------------------------
    # ANALYTIC NORMALS  (critical!)
    #
    # We must NOT let the fragment shader derive normals with dFdx/dFdy.
    # When the camera looks *down* the corridor, the walls are at a
    # grazing angle to the view; the two screen-space derivatives become
    # nearly parallel, their cross product collapses toward zero, and
    # normalize() returns NaN -> the wall renders BLACK.
    #
    # Instead we compute the true surface normal from the parametric
    # tangents:   n = normalize( dP/dz  x  dP/dy )
    # This is exact, view-independent, and never degenerates.
    # ------------------------------------------------------------------
    left_nrm  = _surface_normals(left_pos,  inward=+1.0)   # left wall faces +X
    right_nrm = _surface_normals(right_pos, inward=-1.0)   # right wall faces -X

    # Interleave position+normal: (n_z, n_y, 6)
    left_wall  = cp.concatenate([left_pos,  left_nrm],  axis=-1).astype(cp.float32)
    right_wall = cp.concatenate([right_pos, right_nrm], axis=-1).astype(cp.float32)

    # ------------------------------------------------------------------
    # CANYON FLOOR
    #
    # Without this the frame is black below the horizon: the camera sits
    # at y=1.6 but the walls only START at y=0, so there is simply no
    # geometry beneath the eye line. That is the horizontal "x-axis cut".
    #
    # The floor is a strip that spans from the left wall's base to the
    # right wall's base at every point down the road, so it always meets
    # the walls exactly — no gaps, no matter how the walls move.
    # ------------------------------------------------------------------
    floor_lx = left_x[:, 0]     # wall base X at each z  (n_z,)
    floor_rx = right_x[:, 0]

    n_fx = 12                   # lateral resolution across the floor
    u = cp.linspace(0.0, 1.0, n_fx).reshape(1, n_fx)          # 0=left, 1=right

    floor_x = floor_lx.reshape(n_z, 1) + \
              (floor_rx - floor_lx).reshape(n_z, 1) * u        # (n_z, n_fx)
    floor_z = world_z[:, 0].reshape(n_z, 1) * cp.ones((1, n_fx))

    # Gentle undulation so the floor isn't a mirror-flat plane
    floor_y = cp.abs(cp.sin(floor_z * 0.6 + t * 0.5)) * 0.10 \
            + cp.abs(cp.cos(floor_x * 1.3)) * 0.05

    floor_pos = cp.stack([floor_x, floor_y, floor_z], axis=-1).astype(cp.float32)
    floor_nrm = _surface_normals(floor_pos, inward=0.0, up=True)

    floor = cp.concatenate([floor_pos, floor_nrm], axis=-1).astype(cp.float32)

    return left_wall, right_wall, floor


def _surface_normals(P, inward: float, up: bool = False):
    """
    P : (n, m, 3) vertex positions
    Returns (n, m, 3) unit normals facing the camera.

    inward : for WALLS, +1 = left wall (normal points +X toward the road),
             -1 = right wall (normal points -X).
    up     : for the FLOOR, force the normal to point +Y (skywards).
    """
    # Tangent along the road (axis 0 = z) and up the wall (axis 1 = y)
    t_z = cp.gradient(P, axis=0)
    t_y = cp.gradient(P, axis=1)

    n = cp.cross(t_z, t_y)

    length = cp.sqrt((n ** 2).sum(axis=-1, keepdims=True))
    n = n / cp.maximum(length, 1e-8)

    if up:
        # Floor: normal must point skywards, never down into the ground.
        flip = cp.sign(n[..., 1:2])
        flip = cp.where(flip == 0, 1.0, flip)
        return n * flip

    # Walls: X component points toward the corridor centre, so both walls
    # are lit from inside rather than from within the rock.
    flip = cp.sign(n[..., 0:1] * inward)
    flip = cp.where(flip == 0, 1.0, flip)
    return n * flip
"""
renderer.py
-----------
ModernGL headless 3-D terrain renderer with a proper cinematic camera system.

Camera philosophy (the fix):
  The previous camera started at cam_y=0.5 with radius=4.0, meaning the
  camera was practically INSIDE the terrain waves — all you saw was chaos.
  The new camera uses a three-act structure that always keeps the full
  landscape in view:

    Act 1 (0%–30%)  : "Establishing shot"
      High and far back (like a drone reveal).  You can see the entire
      terrain laid out below you.  The camera slowly descends toward a
      comfortable 3/4 top-down angle.

    Act 2 (30%–70%) : "Immersive cruise"
      Camera locks to a low-angle perspective shot — similar to flying
      just above the terrain at a 25-30° tilt.  This is the "music video"
      angle where you actually see the ridges rising and falling with the
      music.  Horizontal orbit is SLOW so the landscape scrolls past you
      rather than spinning wildly.

    Act 3 (70%–100%): "Pull-back reveal"
      Camera rises and zooms out again for a grand ending wide shot.

  All three acts blend smoothly using hermite (smoothstep) curves so there
  are no jarring jumps between them.

Other changes in this file:
  - FOV widened from 45° → 60°  so more terrain is visible at any position.
  - lookAt target is slightly in FRONT of the camera, not at world origin,
    so the camera always looks toward the "horizon" of the terrain rather
    than back at the centre point.
  - VERT_SCALE raised 0.035 → 0.055 so the terrain peaks are taller and
    the height variation is actually visible from a distance.
  - Fog start pushed farther out so the full terrain width is clear.
"""

import moderngl
import numpy as np
import glm
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """
    Hermite smooth-step: maps x from [edge0, edge1] → [0, 1] with ease-in/out.
    Used everywhere we need a smooth blend between two states.
    Returns 0 below edge0, 1 above edge1, smooth curve in between.
    """
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-9)))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by factor t (0-1)."""
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Camera system
# ---------------------------------------------------------------------------

def _compute_camera(frame_index, total_frames, beat_pulse, rng, shake_intensity):
    # Static camera
    cam_pos = glm.vec3(0.0, 1.5, 9.5)
    look_at = glm.vec3(0.0, 0.6, 0.0)

    return cam_pos, look_at
    # ------------------------------------------------------------------
    # Beat camera shake — a short, sharp upward kick on every beat.
    # We only shake the camera UP (never sideways or down) so it reads
    # as a percussive impact rather than random noise.
    # ------------------------------------------------------------------
    if beat_pulse > 0.5:
        # Upward-biased shake: feels like the bass is pushing the camera
        shake_y = abs(rng.uniform(0, shake_intensity))
        shake_x = rng.uniform(-shake_intensity * 0.3, shake_intensity * 0.3)
        shake_z = rng.uniform(-shake_intensity * 0.3, shake_intensity * 0.3)
        cam_x  += shake_x
        cam_y  += shake_y
        cam_z  += shake_z

    cam_pos = glm.vec3(cam_x, cam_y, cam_z)
    return cam_pos, look_at


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class TerrainRenderer:
    """
    Renders 3-D terrain meshes to PNG images using off-screen OpenGL.

    Usage
    -----
    renderer = TerrainRenderer(grid_size=150)
    img = renderer.render_frame(X, Y, Z, frame_index=i, total_frames=N,
                                band_energies=bands, beat_pulse=beat,
                                frame_path="out/frame_000001.png")
    """

    # ------------------------------------------------------------------
    # TWEAK PANEL — safe to change these numbers without reading the rest
    # ------------------------------------------------------------------
    
    
    #RENDER_WIDTH    = 1920
    #RENDER_HEIGHT   = 1080

    RENDER_WIDTH    = 720
    RENDER_HEIGHT   = 720


    # How much the Y values from terrain_gen are exaggerated vertically.
    # Raised from 0.035 to 0.055 so ridges are clearly visible from distance.
    VERT_SCALE      = 0.040

    # Field of view in degrees.  Wider (60°) shows more terrain than the
    # original 45° — better for a landscape visualizer.
    FOV_DEGREES = 45.0

    # Beat camera shake — how many world-units the camera jolts on a beat
    SHAKE_INTENSITY = 0.0

    # Fog — terrain beyond FOG_FAR fades to the background colour.
    # Pushed out far so the full terrain width is always visible.
    FOG_NEAR        = 5.0
    FOG_FAR         = 18.0

    # Bloom post-process
    BLOOM_THRESHOLD = 200    # pixel brightness (0–255) above which glow applies
    BLOOM_RADIUS    = 12     # Gaussian blur radius for the glow halo
    BLOOM_STRENGTH  = 0.45   # additive blend factor (0 = no bloom, 1 = full)
    # ------------------------------------------------------------------

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        w, h = self.RENDER_WIDTH, self.RENDER_HEIGHT

        # ------------------------------------------------------------------
        # 1. Headless OpenGL context
        # DEPTH_TEST ensures closer triangles occlude farther ones correctly.
        # ------------------------------------------------------------------
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        # ------------------------------------------------------------------
        # 2. Off-screen framebuffer
        # We render into a texture (not a window) and read the pixels back
        # to save as a PNG.  The depth renderbuffer is required for DEPTH_TEST.
        # ------------------------------------------------------------------
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)),
        )

        # ------------------------------------------------------------------
        # 3. GLSL shader program
        # ------------------------------------------------------------------
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;       // raw mesh vertex from terrain_gen

                uniform mat4 mvp;          // model-view-projection matrix
                uniform float vert_scale;  // vertical exaggeration multiplier

                out float v_height;  // scaled Y — drives colour in fragment shader
                out vec3  v_pos;     // scaled world position — used for fog & normals

                void main() {
                    // Scale the Y (height) axis so peaks are clearly visible.
                    // X and Z stay at terrain_gen's -1..+1 range.
                    vec3 pos = vec3(in_position.x,
                                   in_position.y * vert_scale,
                                   in_position.z);

                    gl_Position = mvp * vec4(pos, 1.0);

                    // Pass to fragment shader
                    v_height = pos.y;
                    v_pos    = pos;
                }
            """,
            fragment_shader="""
                #version 330

                in float v_height;   // scaled vertex height
                in vec3  v_pos;      // world-space position

                out vec4 fragColor;

                // ---- lighting uniforms ----
                uniform vec3  light_dir;   // world-space sun direction
                uniform vec3  cam_pos;     // camera position (for specular)
                uniform float ambient;     // minimum brightness in shadows

                // ---- colour palette (animated each frame from CPU) ----
                uniform vec3 col_valley;   // colour at y ≈ 0  (lowest terrain)
                uniform vec3 col_mid;      // colour at y ≈ 0.4
                uniform vec3 col_peak;     // colour at y > 0.4 (highest peaks)

                // ---- fog uniforms ----
                uniform vec3  fog_color;
                uniform float fog_near;
                uniform float fog_far;

                void main() {
                    // ---- Flat shading normal from screen-space derivatives ----
                    // dFdx/dFdy give us how the position changes between adjacent
                    // pixels.  Their cross product points perpendicular to the face.
                    vec3 normal = normalize(cross(dFdx(v_pos), dFdy(v_pos)));

                    // ---- Diffuse (Lambertian) lighting ----
                    vec3  L    = normalize(light_dir);
                    float diff = max(dot(normal, L), 0.0);

                    // ---- Specular (Phong) highlight ----
                    // Makes tall peaks catch the light with a bright glint.
                    vec3  V    = normalize(cam_pos - v_pos);
                    vec3  R    = reflect(-L, normal);
                    float spec = pow(max(dot(V, R), 0.0), 32.0) * 0.4;

                    float lighting = ambient + diff + spec;

                    // ---- Height-based colour gradient ----
                    // valley_color → mid_color between y=0 and y=0.4
                    // mid_color    → peak_color  above y=0.4
                    vec3 color;
                    if (v_height < 0.4) {
                        float t = clamp(v_height * 2.5, 0.0, 1.0);
                        color = mix(col_valley, col_mid, t);
                    } else {
                        float t = clamp((v_height - 0.4) * 1.6, 0.0, 1.0);
                        color = mix(col_mid, col_peak, t);
                    }

                    vec3 lit = color * lighting * 1.2;

                    // ---- Atmospheric depth fog ----
                    // Linearly blend lit colour toward fog_color with distance.
                    // This softens the far edge of the terrain so it doesn't
                    // look like it just "cuts off".
                    float dist  = length(v_pos);
                    float fog_t = clamp((dist - fog_near) / (fog_far - fog_near),
                                        0.0, 1.0);
                    vec3 final  = mix(lit, fog_color, fog_t);

                    fragColor = vec4(final, 1.0);
                }
            """,
        )

        # ------------------------------------------------------------------
        # 4. Vertex buffer (pre-allocated, overwritten every frame)
        #    Index buffer (built once — mesh topology never changes)
        # ------------------------------------------------------------------
        # Reserve enough bytes for grid_size² vertices × 3 floats × 4 bytes
        self.vbo = self.ctx.buffer(reserve=grid_size * grid_size * 12)

        # Triangulate the quad grid.  Each quad → 2 triangles → 6 indices.
        indices = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                tl = i * grid_size + j        # top-left vertex index
                tr = tl + 1                   # top-right
                bl = (i + 1) * grid_size + j  # bottom-left
                br = bl + 1                   # bottom-right
                # Two triangles per quad (counter-clockwise winding)
                indices.extend([tl, bl, tr,   # triangle 1
                                 tr, bl, br])  # triangle 2

        self.ibo = self.ctx.buffer(np.array(indices, dtype="i4").tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "3f", "in_position")], self.ibo
        )

        # ------------------------------------------------------------------
        # 5. Static shader uniforms (values that never change per frame)
        # ------------------------------------------------------------------
        self.prog["vert_scale"].value = self.VERT_SCALE

        # Sun is upper-right-front — creates a clear shadow on the far slopes
        # so depth is readable even without fog.
        self.prog["light_dir"].value  = (1.5, 3.0, 1.0)

        # Fog colour should match the clear colour exactly so the fade is seamless
        self.prog["fog_color"].value  = (0.01, 0.01, 0.02)
        self.prog["fog_near"].value   = self.FOG_NEAR
        self.prog["fog_far"].value    = self.FOG_FAR

        # Ambient raised slightly from v1 (0.28→0.32) so valley details don't
        # get completely lost in shadow when the camera is low.
        self.prog["ambient"].value    = 0.32

        # Seeded RNG for reproducible beat shake offsets
        self._rng = np.random.default_rng(seed=0)

    # ------------------------------------------------------------------
    # Per-frame render entry point
    # ------------------------------------------------------------------

    def render_frame(
        self,
        X_gpu,
        Y_gpu,
        Z_gpu,
        frame_index:   int,
        total_frames:  int   = 1,
        band_energies: dict  = None,
        beat_pulse:    float = 0.0,
        frame_path:    str   = None,
    ) -> Image.Image:
        """
        Render a single frame of terrain to a PIL Image.

        Parameters
        ----------
        X_gpu, Y_gpu, Z_gpu : cp.ndarray, shape (grid_size, grid_size)
            Mesh vertex positions from terrain_gen (on the GPU).
        frame_index   : Current frame number (drives camera & palette).
        total_frames  : Total frame count (used to compute progress 0→1).
        band_energies : Dict of per-band scalar energies for colour animation.
        beat_pulse    : 1.0 on a beat frame, 0.0 otherwise (camera shake).
        frame_path    : If given, the PNG is saved to this path.

        Returns
        -------
        PIL.Image.Image (RGB, RENDER_WIDTH × RENDER_HEIGHT)
        """
        if band_energies is None:
            band_energies = {}

        progress = frame_index / max(total_frames, 1)  # 0.0 → 1.0

        # ------------------------------------------------------------------
        # A. Compute camera position and target via the three-act system
        # ------------------------------------------------------------------
        cam_pos, look_at = _compute_camera(
            frame_index, total_frames, beat_pulse, self._rng, self.SHAKE_INTENSITY
        )

        # Upload camera position to the shader (needed for specular calculation)
        self.prog["cam_pos"].value = (cam_pos.x, cam_pos.y, cam_pos.z)

        # Build the view matrix (camera transform) and projection matrix
        view = glm.lookAt(cam_pos, look_at, glm.vec3(0, 1, 0))
        proj = glm.perspective(
            glm.radians(self.FOV_DEGREES),
            self.RENDER_WIDTH / self.RENDER_HEIGHT,
            0.1,    # near clip plane
            100.0,  # far clip plane
        )
        # Combined MVP = Projection × View  (no separate model transform;
        # terrain_gen already centres the mesh at the origin)
        mvp = proj * view
        self.prog["mvp"].write(np.array(mvp, dtype="f4").tobytes())

        # ------------------------------------------------------------------
        # B. Animate colour palette based on song progress
        #
        # Three-act palette to match the camera:
        #   Act 1 (cool blue)  → Act 2 (vivid purple/teal) → Act 3 (warm gold)
        # This gives the video a visual narrative arc beyond just moving terrain.
        # ------------------------------------------------------------------
        t_warm = _smoothstep(0.35, 0.65, progress)  # 0=cool, 1=warm

        cool_valley = np.array([0.05, 0.05, 0.25])   # deep navy
        cool_mid    = np.array([0.30, 0.60, 0.80])   # bright teal (more readable from far)
        cool_peak   = np.array([0.90, 0.95, 1.00])   # near-white ice

        warm_valley = np.array([0.15, 0.02, 0.02])   # dark crimson
        warm_mid    = np.array([0.80, 0.40, 0.05])   # orange
        warm_peak   = np.array([1.00, 0.95, 0.30])   # bright gold

        valley = cool_valley * (1 - t_warm) + warm_valley * t_warm
        mid    = cool_mid    * (1 - t_warm) + warm_mid    * t_warm
        peak   = cool_peak   * (1 - t_warm) + warm_peak   * t_warm

        # Louder mid-range audio nudges the peak colour toward white —
        # loud drops feel more intense because the peaks "blow out"
        mid_energy = band_energies.get("mid", 0.0)
        peak = peak + (1.0 - peak) * mid_energy * 0.25

        self.prog["col_valley"].value = tuple(valley.clip(0, 1))
        self.prog["col_mid"].value    = tuple(mid.clip(0, 1))
        self.prog["col_peak"].value   = tuple(peak.clip(0, 1))

        # ------------------------------------------------------------------
        # C. Draw the terrain mesh
        # ------------------------------------------------------------------
        self.fbo.use()
        # Clear to near-black so the fog colour blends correctly at the edges
        self.fbo.clear(0.01, 0.01, 0.02, 1.0)

        # Copy vertex data from GPU (CuPy) → CPU → ModernGL buffer
        # This is the one unavoidable GPU→CPU transfer per frame.
        X = X_gpu.get()
        Y = Y_gpu.get()
        Z = Z_gpu.get()
        vertices = np.stack([X, Y, Z], axis=-1).astype("f4").tobytes()
        self.vbo.write(vertices)

        # Issue the draw call — renders all triangles in the index buffer
        self.vao.render(moderngl.TRIANGLES)

        # ------------------------------------------------------------------
        # D. Read pixels from framebuffer → PIL Image
        # OpenGL stores pixels bottom-row-first; we flip vertically to fix it.
        # ------------------------------------------------------------------
        raw = self.fbo.color_attachments[0].read()
        img = Image.frombytes("RGBA", (self.RENDER_WIDTH, self.RENDER_HEIGHT), raw)
        img = img.convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)

        # ------------------------------------------------------------------
        # E. Post-process: bloom glow on bright pixels
        # ------------------------------------------------------------------
        img = self._apply_bloom(img)

        # ------------------------------------------------------------------
        # F. Optional save to disk
        # ------------------------------------------------------------------
        if frame_path:
            img.save(frame_path)

        return img

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _apply_bloom(self, img: Image.Image) -> Image.Image:
        """
        Bloom glow effect:
          1. Extract pixels brighter than BLOOM_THRESHOLD.
          2. Blur them with a wide Gaussian (simulates lens scatter).
          3. Add the blurred glow additively back onto the original image.
        The result makes bright terrain peaks appear to emit light.
        """
        arr = np.array(img, dtype=np.float32)

        # Keep only the bright pixels (per-pixel max across RGB channels)
        bright_mask  = arr.max(axis=2, keepdims=True) > self.BLOOM_THRESHOLD
        bright_layer = arr * bright_mask

        # Blur the isolated bright pixels
        bright_img  = Image.fromarray(np.clip(bright_layer, 0, 255).astype(np.uint8))
        blurred     = bright_img.filter(ImageFilter.GaussianBlur(radius=self.BLOOM_RADIUS))
        blurred_arr = np.array(blurred, dtype=np.float32)

        # Additive composite: original + (glow × strength)
        bloomed = np.clip(arr + blurred_arr * self.BLOOM_STRENGTH, 0, 255).astype(np.uint8)
        return Image.fromarray(bloomed)
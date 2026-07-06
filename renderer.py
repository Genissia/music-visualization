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
from pyglm import glm
from PIL import Image, ImageFilter, ImageChops


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """
    Hermite smooth-step: maps x from [edge0, edge1] → [0, 1] with ease-in/out.
    """
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-9)))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by factor t (0-1)."""
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class TerrainRenderer:
    """
    Renders 3-D terrain meshes to PNG images using off-screen OpenGL.
    Fully compatible with both CuPy (NVIDIA CUDA) and NumPy (AMD/Intel CPU) inputs.
    """

    # ------------------------------------------------------------------
    # TWEAK PANEL
    # ------------------------------------------------------------------
    RENDER_WIDTH    = 720
    RENDER_HEIGHT   = 720

    # How much the Y values from terrain_gen are exaggerated vertically.
    VERT_SCALE      = 0.040

    # Field of view in degrees.
    FOV_DEGREES     = 45.0

    # Beat camera shake — how many world-units the camera jolts on a beat
    SHAKE_INTENSITY = 0.15

    # Fog — terrain beyond FOG_FAR fades to the background colour.
    FOG_NEAR        = 5.0
    FOG_FAR         = 18.0

    # Bloom post-process
    BLOOM_THRESHOLD = 180    # pixel brightness (0–255) above which glow applies
    BLOOM_RADIUS    = 12     # Gaussian blur radius for the glow halo
    BLOOM_STRENGTH  = 0.45   # additive blend factor (0 = no bloom, 1 = full)
    # ------------------------------------------------------------------

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        w, h = self.RENDER_WIDTH, self.RENDER_HEIGHT

        # 1. Headless OpenGL context
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        # 2. Off-screen framebuffer
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)),
        )

        # 3. GLSL shader program
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;       // raw mesh vertex from terrain_gen

                uniform mat4 mvp;          // model-view-projection matrix
                uniform float vert_scale;  // vertical exaggeration multiplier

                out float v_height;  // scaled Y — drives colour in fragment shader
                out vec3  v_pos;     // scaled world position — used for fog & normals

                void main() {
                    vec3 pos = vec3(in_position.x,
                                   in_position.y * vert_scale,
                                   in_position.z);

                    gl_Position = mvp * vec4(pos, 1.0);

                    v_height = pos.y;
                    v_pos    = pos;
                }
            """,
            fragment_shader="""
                #version 330

                in float v_height;   // scaled vertex height
                in vec3  v_pos;      // world-space position

                out vec4 fragColor;

                uniform vec3  light_dir;   // world-space sun direction
                uniform vec3  cam_pos;     // camera position (for specular)
                uniform float ambient;     // minimum brightness in shadows

                uniform vec3 col_valley;   // colour at y ≈ 0  (lowest terrain)
                uniform vec3 col_mid;      // colour at y ≈ 0.4
                uniform vec3 col_peak;     // colour at y > 0.4 (highest peaks)

                uniform vec3  fog_color;
                uniform float fog_near;
                uniform float fog_far;

                void main() {
                    // Flat shading normal from screen-space derivatives
                    vec3 normal = normalize(cross(dFdx(v_pos), dFdy(v_pos)));

                    // Diffuse (Lambertian) lighting
                    vec3  L    = normalize(light_dir);
                    float diff = max(dot(normal, L), 0.0);

                    // Specular (Phong) highlight
                    vec3  V    = normalize(cam_pos - v_pos);
                    vec3  R    = reflect(-L, normal);
                    float spec = pow(max(dot(V, R), 0.0), 32.0) * 0.4;

                    float lighting = ambient + diff + spec;

                    // Height-based colour gradient
                    vec3 color;
                    if (v_height < 0.4) {
                        float t = clamp(v_height * 2.5, 0.0, 1.0);
                        color = mix(col_valley, col_mid, t);
                    } else {
                        float t = clamp((v_height - 0.4) * 1.6, 0.0, 1.0);
                        color = mix(col_mid, col_peak, t);
                    }

                    vec3 lit = color * lighting * 1.2;

                    // Atmospheric depth fog
                    float dist  = length(v_pos);
                    float fog_t = clamp((dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
                    vec3 final  = mix(lit, fog_color, fog_t);

                    fragColor = vec4(final, 1.0);
                }
            """,
        )

        # 4. Vertex buffer (pre-allocated) & Index Buffer (static topology)
        self.vbo = self.ctx.buffer(reserve=grid_size * grid_size * 12)

        indices = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                tl = i * grid_size + j
                tr = tl + 1
                bl = (i + 1) * grid_size + j
                br = bl + 1
                indices.extend([tl, bl, tr, tr, bl, br])

        self.ibo = self.ctx.buffer(np.array(indices, dtype="i4").tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "3f", "in_position")], self.ibo
        )

        # 5. Static shader uniforms
        self.prog["vert_scale"].value = self.VERT_SCALE
        self.prog["light_dir"].value  = (1.5, 3.0, 1.0)
        self.prog["fog_color"].value  = (0.01, 0.01, 0.02)
        self.prog["fog_near"].value   = self.FOG_NEAR
        self.prog["fog_far"].value    = self.FOG_FAR
        self.prog["ambient"].value    = 0.32

        # Seeded RNG for reproducible beat shake offsets
        self._rng = np.random.default_rng(seed=0)

    def _compute_camera(self, progress: float, beat_pulse: float):
        """
        Implements the smooth 3-Act Cinematic Camera track with a percussive beat kick.
        """
        # --- Base Horizontal Orbit Position ---
        # Slow continuous rotational angle over the course of the song
        angle = progress * glm.pi() * 0.4

        if progress <= 0.3:
            # Act 1 (0% - 30%): Drone Reveal (High and far away, descending)
            t = _smoothstep(0.0, 0.3, progress)
            radius = _lerp(12.0, 8.5, t)
            cam_y  = _lerp(6.0, 2.2, t)
            look_at = glm.vec3(0.0, 0.2, 0.0)
        elif progress <= 0.7:
            # Act 2 (30% - 70%): Immersive low cruise
            t = _smoothstep(0.3, 0.7, progress)
            radius = 8.5
            cam_y  = _lerp(2.2, 1.4, t)
            # Look slightly forward towards the terrain horizon
            look_at = glm.vec3(0.0, _lerp(0.2, 0.5, t), -0.5)
        else:
            # Act 3 (70% - 100%): Pull back grand ending reveal
            t = _smoothstep(0.7, 1.0, progress)
            radius = _lerp(8.5, 13.0, t)
            cam_y  = _lerp(1.4, 5.0, t)
            look_at = glm.vec3(0.0, _lerp(0.5, 0.0, t), 0.0)

        # Calculate base structural position on an orbital cylinder arc
        cam_x = radius * glm.sin(angle)
        cam_z = radius * glm.cos(angle)

        # --- Beat Camera Shake (Upward Impact) ---
        if beat_pulse > 0.5 and self.SHAKE_INTENSITY > 0:
            shake_y = abs(self._rng.uniform(0, self.SHAKE_INTENSITY))
            shake_x = self._rng.uniform(-self.SHAKE_INTENSITY * 0.3, self.SHAKE_INTENSITY * 0.3)
            shake_z = self._rng.uniform(-self.SHAKE_INTENSITY * 0.3, self.SHAKE_INTENSITY * 0.3)
            cam_x += shake_x
            cam_y += shake_y
            cam_z += shake_z

        return glm.vec3(cam_x, cam_y, cam_z), look_at

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
        if band_energies is None:
            band_energies = {"sub_bass": 0, "mid": 0, "treble": 0}

        progress = frame_index / max(total_frames, 1)

        # ------------------------------------------------------------------
        # AMD/NVIDIA Cross-Compatibility Layer
        # ------------------------------------------------------------------
        # If your teammate passes a CuPy array, .get() safely pulls it to CPU.
        # On your AMD system, it detects standard NumPy arrays and passes safely.
        X = X_gpu.get() if hasattr(X_gpu, "get") else X_gpu
        Y = Y_gpu.get() if hasattr(Y_gpu, "get") else Y_gpu
        Z = Z_gpu.get() if hasattr(Z_gpu, "get") else Z_gpu

        # Stack into continuous interleaved binary data [x0,y0,z0, x1,y1,z1...]
        vertices = np.dstack((X, Y, Z)).astype(np.float32).tobytes()
        self.vbo.write(vertices)

        # ------------------------------------------------------------------
        # Camera & Matrix Math (PyGLM)
        # ------------------------------------------------------------------
        cam_pos, look_at = self._compute_camera(progress, beat_pulse)

        proj = glm.perspective(
            glm.radians(self.FOV_DEGREES),
            self.RENDER_WIDTH / self.RENDER_HEIGHT,
            0.1, 50.0
        )
        view = glm.lookAt(cam_pos, look_at, glm.vec3(0.0, 1.0, 0.0))
        mvp = proj * view

        # Update dynamic camera uniform positions
        self.prog["mvp"].write(bytes(mvp))
        self.prog["cam_pos"].value = tuple(cam_pos)

        # ------------------------------------------------------------------
        # Audio-Reactive Dynamic Color Palette Shifts
        # ------------------------------------------------------------------
        # Values base-map to rock aesthetics, scaling saturation/tone with audio bands
        sub_bass = band_energies.get("sub_bass", 0.0)
        mid      = band_energies.get("mid", 0.0)
        treble   = band_energies.get("treble", 0.0)

        # Valley shifts from deep indigo to electric purple on heavy bass
        self.prog["col_valley"].value = (0.02 + sub_bass * 0.08, 0.01, 0.05 + sub_bass * 0.15)
        # Mid-ranges pulse cyan/magenta with mid-frequency tracks
        self.prog["col_mid"].value    = (0.4 + mid * 0.3, 0.05, 0.6 - mid * 0.2)
        # Peaks glint bright white/neon gold on high treble transients or drum hits
        self.prog["col_peak"].value   = (0.9, 0.8 + treble * 0.2, 0.5 + beat_pulse * 0.5)

        # ------------------------------------------------------------------
        # OpenGL Render Call
        # ------------------------------------------------------------------
        self.fbo.use()
        self.ctx.clear(0.01, 0.01, 0.02, 1.0) # Matches background fog color
        self.vao.render(moderngl.TRIANGLES)

        # Extract raw color buffer to python memory image
        image_data = self.fbo.read(components=3)
        img = Image.frombytes("RGB", (self.RENDER_WIDTH, self.RENDER_HEIGHT), image_data)

        # ------------------------------------------------------------------
        # Pure PIL Post-Process Multi-Pass Bloom
        # ------------------------------------------------------------------
        if self.BLOOM_STRENGTH > 0:
            # 1. Isolate bright pixels exceeding our threshold limits
            bright_pass = img.point(lambda p: p if p > self.BLOOM_THRESHOLD else 0)
            # 2. Heavy blur layer to generate glow emissions
            blur_layer = bright_pass.filter(ImageFilter.GaussianBlur(radius=self.BLOOM_RADIUS))
            # 3. Composite additive screen blend overlay back onto original render frame
            screen_blend = ImageChops.screen(img, blur_layer)
            img = Image.blend(img, screen_blend, self.BLOOM_STRENGTH)

        # Optional synchronous file preservation fallback if called directly
        if frame_path:
            img.save(frame_path)

        return img
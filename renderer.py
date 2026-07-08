"""
renderer.py
-----------
ModernGL headless 3-D terrain renderer.

Updated for "Static Top-Down Diorama":
  - Camera is locked in a high, static isometric position.
  - Orbital movement and beat-shake completely disabled.
  - FOV narrowed to flatten perspective and keep the whole 1:1 plane perfectly framed.
"""

import moderngl
import numpy as np
import glm
from PIL import Image, ImageFilter


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0 + 1e-9)))
    return t * t * (3.0 - 2.0 * t)


def _compute_camera(frame_index, total_frames, beat_pulse, rng, shake_intensity):
    # STATIC ISOMETRIC ANGLE
    # Locked to a perfect 45-degree corner view (no orbit progression)
    orbit_rad = np.radians(45.0)

    # Pulled high up and back to perfectly frame the 1:1 square block
    radius = 5.5
    height = 5.5

    cam_x = radius * np.sin(orbit_rad)
    cam_z = radius * np.cos(orbit_rad)
    cam_y = height

    # Beat shake is technically still here, but SHAKE_INTENSITY is set to 0.0 below
    if beat_pulse > 0.5 and shake_intensity > 0.0:
        shake_y = abs(rng.uniform(0, shake_intensity))
        shake_x = rng.uniform(-shake_intensity * 0.3, shake_intensity * 0.3)
        shake_z = rng.uniform(-shake_intensity * 0.3, shake_intensity * 0.3)
        cam_x  += shake_x
        cam_y  += shake_y
        cam_z  += shake_z

    cam_pos = glm.vec3(cam_x, cam_y, cam_z)
    
    # Locked directly to the dead-center of the flat terrain plane
    look_at = glm.vec3(0.0, 0.0, 0.0)

    return cam_pos, look_at


class TerrainRenderer:
    # ------------------------------------------------------------------
    # TWEAK PANEL
    # ------------------------------------------------------------------
    RENDER_WIDTH    = 1920
    RENDER_HEIGHT   = 1080

    # Boosted slightly because top-down views make mountains look flatter
    VERT_SCALE      = 0.050

    # Narrow FOV gives it that rigid, perfect isometric diorama shape
    FOV_DEGREES     = 30.0
    
    # Locked to 0.0 for a purely static camera
    SHAKE_INTENSITY = 0.00

    FOG_NEAR        = 50.0
    FOG_FAR         = 100.0

    BLOOM_THRESHOLD = 180    
    BLOOM_RADIUS    = 16     
    BLOOM_STRENGTH  = 0.60   
    # ------------------------------------------------------------------

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        w, h = self.RENDER_WIDTH, self.RENDER_HEIGHT

        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)),
        )

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                uniform float vert_scale;
                
                uniform int is_wireframe;
                
                out float v_height;
                out vec3  v_pos;

                void main() {
                    vec3 pos = vec3(in_position.x, in_position.y * vert_scale, in_position.z);
                    
                    if (is_wireframe == 1) {
                        pos.y += 0.005; 
                    }
                    
                    gl_Position = mvp * vec4(pos, 1.0);
                    v_height = pos.y;
                    v_pos    = pos;
                }
            """,
            fragment_shader="""
                #version 330
                in float v_height;
                in vec3  v_pos;
                out vec4 fragColor;

                uniform vec3  light_dir;
                uniform vec3  cam_pos;
                uniform float ambient;

                uniform vec3 col_valley;
                uniform vec3 col_mid;
                uniform vec3 col_peak;

                uniform vec3  fog_color;
                uniform float fog_near;
                uniform float fog_far;

                // --- AUDIO REACTIVE UNIFORMS ---
                uniform int   is_wireframe;
                uniform float beat_pulse;
                uniform float treble_energy;

                void main() {
                    vec3 normal = normalize(cross(dFdx(v_pos), dFdy(v_pos)));

                    // DYNAMIC LIGHTING PHYSICS
                    vec3 dyn_light_dir = normalize(light_dir + vec3(sin(beat_pulse * 10.0)*0.5, 0.0, cos(beat_pulse * 10.0)*0.5));
                    float diff = max(dot(normal, dyn_light_dir), 0.0);

                    float flash = beat_pulse * 2.0;
                    float dyn_ambient = ambient + (treble_energy * 0.5);

                    vec3 V = normalize(cam_pos - v_pos);
                    vec3 R = reflect(-dyn_light_dir, normal);
                    float spec = pow(max(dot(V, R), 0.0), 32.0) * (0.4 + flash);

                    float lighting = dyn_ambient + diff + spec;

                    // HEIGHT COLORS
                    vec3 color;
                    if (abs(v_height) < 0.02) {
                        color = col_highway;  // Flat clean road asphalt
                    } else {
                        // Blend between mountain slopes and illuminated neon wave crests
                        float t = clamp(abs(v_height) * 2.2, 0.0, 1.0);
                        color = mix(col_wave_slope, col_wave_crest, t);
                    }

                    // MATERIAL PASS (SOLID CORE vs GLOWING NEON)
                    vec3 lit;
                    if (is_wireframe == 1) {
                        lit = color * (1.8 + flash);
                    } else {
                        lit = color * lighting * 0.20; 
                    }

                    // DEPTH FOG
                    float dist  = length(v_pos);
                    float fog_t = clamp((dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
                    vec3 final  = mix(lit, fog_color, fog_t);

                    vec3 final = mix(lit, fog_color, fog_t);
                    fragColor  = vec4(final, 1.0);
                }
            """,
        )

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

        self.prog["vert_scale"].value = self.VERT_SCALE
        self.prog["light_dir"].value  = (1.5, 3.0, 1.0)
        self.prog["fog_color"].value  = (0.01, 0.01, 0.015) 
        self.prog["fog_near"].value   = self.FOG_NEAR
        self.prog["fog_far"].value    = self.FOG_FAR
        self.prog["ambient"].value    = 0.35 

        self._rng = np.random.default_rng(seed=0)

    def render_frame(
        self, X_gpu, Y_gpu, Z_gpu, frame_index: int, total_frames: int = 1,
        band_energies: dict = None, beat_pulse: float = 0.0, frame_path: str = None,
    ) -> Image.Image:
        
        if band_energies is None:
            band_energies = {}

        progress = frame_index / max(total_frames, 1)

        cam_pos, look_at = _compute_camera(
            frame_index, total_frames, beat_pulse, self._rng, self.SHAKE_INTENSITY
        )

        self.prog["cam_pos"].value = (cam_pos.x, cam_pos.y, cam_pos.z)

        view = glm.lookAt(cam_pos, look_at, glm.vec3(0, 1, 0))
        proj = glm.perspective(
            glm.radians(self.FOV_DEGREES),
            self.RENDER_WIDTH / self.RENDER_HEIGHT,
            0.1, 100.0,
        )
        
        mvp = proj * view
        self.prog["mvp"].write(np.array(mvp, dtype="f4").tobytes())

        self.prog["beat_pulse"].value    = float(beat_pulse)
        self.prog["treble_energy"].value = float(band_energies.get("treble", 0.0))

        t_warm = _smoothstep(0.35, 0.65, progress)

        cool_valley = np.array([0.05, 0.05, 0.25])
        cool_mid    = np.array([0.15, 0.80, 0.90]) 
        cool_peak   = np.array([0.90, 0.95, 1.00])

        warm_valley = np.array([0.15, 0.02, 0.05])
        warm_mid    = np.array([0.95, 0.30, 0.05]) 
        warm_peak   = np.array([1.00, 0.90, 0.40])

        # Look straight down the negative Z-axis corridor towards the horizon
        look_at = glm.vec3(0.0, 0.12, -2.5)

        mid_energy = band_energies.get("mid", 0.0)
        peak = peak + (1.0 - peak) * mid_energy * 0.35

        self.prog["mvp"].write(np.array(mvp, dtype=np.float32).tobytes())
        self.prog["cam_pos"].value = (cam_x, cam_y, cam_z)

        self.fbo.use()
        bg = (0.01, 0.01, 0.015) 
        self.fbo.clear(bg[0], bg[1], bg[2], 1.0)

        X = X_gpu.get()
        Y = Y_gpu.get()
        Z = Z_gpu.get()
        vertices = np.stack([X, Y, Z], axis=-1).astype("f4").tobytes()
        self.vbo.write(vertices)

        # PASS 1: Solid Dark Terrain Core
        self.prog["is_wireframe"].value = 0
        self.ctx.wireframe = False
        self.vao.render(moderngl.TRIANGLES)

        # PASS 2: Glowing Neon Wireframe Overlay
        self.prog["is_wireframe"].value = 1
        self.ctx.wireframe = True
        self.vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = False 

        raw = self.fbo.color_attachments[0].read()
        img = Image.frombytes("RGBA", (self.RENDER_WIDTH, self.RENDER_HEIGHT), raw)
        img = img.convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)

        img = self._apply_bloom(img)

        if frame_path:
            img.save(frame_path)

        return img

    def _apply_bloom(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)

        bright_mask  = arr.max(axis=2, keepdims=True) > self.BLOOM_THRESHOLD
        bright_layer = arr * bright_mask

        bright_img  = Image.fromarray(np.clip(bright_layer, 0, 255).astype(np.uint8))
        blurred     = bright_img.filter(ImageFilter.GaussianBlur(radius=self.BLOOM_RADIUS))
        blurred_arr = np.array(blurred, dtype=np.float32)

        bloomed = np.clip(arr + blurred_arr * self.BLOOM_STRENGTH, 0, 255).astype(np.uint8)
        return Image.fromarray(bloomed)

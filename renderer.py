"""
renderer.py
-----------
Renders the slot-canyon corridor: two independent wall surfaces with the
camera flying between them.

ARCHITECTURAL CHANGE
--------------------
Previously this drew ONE heightmap mesh. It now draws TWO wall meshes
(left + right), each with its own VBO/VAO, sharing one shader program.

Vertex data arrives already in WORLD SPACE from terrain_gen — the vertex
shader no longer rescales anything. That was the source of the endless
"tune the magic numbers" loop: scales were being applied in two places.
"""

import moderngl
import numpy as np
from pyglm import glm
from PIL import Image, ImageFilter, ImageChops, ImageDraw


class TerrainRenderer:

    RENDER_WIDTH  = 1920
    RENDER_HEIGHT = 1080

    FOV_DEGREES = 70.0

    # Fog must span the full road depth (Z_NEAR=2 → Z_FAR=-18, so ~20 units)
    FOG_NEAR = 6.0
    FOG_FAR  = 17.0

    BLOOM_THRESHOLD = 120
    BLOOM_RADIUS    = 7
    BLOOM_STRENGTH  = 0.45

    # Camera lives INSIDE the canyon, WELL PAST THE ENTRANCE.
    #
    # CRITICAL: CAM_Z must be several units *inside* the corridor, not at
    # its mouth. At the mouth (z ~ +1) the nearby walls fall outside the
    # view frustum sideways and the mesh collapses to ~4% screen coverage
    # (this was the "thin slivers" bug). A few units in → ~45% coverage,
    # i.e. walls filling both sides of the frame.
    CAM_HEIGHT = 1.6          # low in the slot, like sitting in a car
    CAM_Z      = -4.0         # inside the corridor  (was +1.2 = at the mouth)
    LOOK_Y     = 3.0          # gaze level-ish; walls run off the top of frame
    LOOK_Z     = -16.0

    def __init__(self, grid_size: int, n_height: int = 40):
        self.grid_size = grid_size
        self.n_height  = n_height
        w, h = self.RENDER_WIDTH, self.RENDER_HEIGHT

        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Walls are one-sided surfaces; we want to see them from inside,
        # so culling stays OFF (both faces render).
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)),
        )

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                // Vertices arrive in WORLD SPACE already — no rescaling here.
                in vec3 in_position;
                in vec3 in_normal;     // analytic normal from terrain_gen

                uniform mat4 mvp;

                out vec3  v_nrm;
                out vec3  v_pos;
                out float v_height;   // 0 at canyon floor, 1 at rim
                out float v_depth;    // 0 near camera, 1 far away

                uniform float wall_height;
                uniform float z_near;
                uniform float z_far;

                void main() {
                    vec3 pos = in_position;

                    gl_Position = mvp * vec4(pos, 1.0);

                    v_pos    = pos;
                    v_nrm    = normalize(in_normal);
                    v_height = clamp(pos.y / wall_height, 0.0, 1.0);
                    v_depth  = clamp((pos.z - z_near) / (z_far - z_near), 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                in vec3  v_pos;
                in vec3  v_nrm;
                in float v_height;
                in float v_depth;

                out vec4 fragColor;

                uniform vec3  light_dir;
                uniform vec3  cam_pos;
                uniform float ambient;

                uniform vec3 col_floor;   // deep in the canyon — darkest
                uniform vec3 col_mid;     // mid-wall
                uniform vec3 col_rim;     // top of the wall, catches the light

                uniform vec3  fog_color;
                uniform float fog_near;
                uniform float fog_far;

                void main() {
                    // Analytic normal from terrain_gen.
                    // (dFdx/dFdy derivatives collapse to zero at grazing
                    //  angles down the corridor and produce NaN -> black.)
                    vec3 normal = normalize(v_nrm);

                    vec3  L    = normalize(light_dir);
                    float diff = max(dot(normal, L), 0.0) * 0.7 + 0.3;

                    vec3  V    = normalize(cam_pos - v_pos);
                    vec3  R    = reflect(-L, normal);
                    float spec = pow(max(dot(V, R), 0.0), 24.0) * 0.35;

                    float lighting = ambient + diff * 0.8 + spec;

                    // Vertical colour gradient: dark floor → bright rim.
                    // This is what sells the "light pouring in from above"
                    // look of the reference photograph.
                    vec3 color;
                    if (v_height < 0.55) {
                        color = mix(col_floor, col_mid, v_height / 0.55);
                    } else {
                        color = mix(col_mid, col_rim, (v_height - 0.55) / 0.45);
                    }

                    // Ambient occlusion: the deeper into the slot, the darker.
                    float ao = 0.60 + 0.40 * pow(v_height, 0.6);
                    color *= ao;

                    vec3 lit = color * lighting;

                    float dist  = length(v_pos - cam_pos);
                    float fog_t = clamp((dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
                    vec3 final  = mix(lit, fog_color, fog_t);

                    fragColor = vec4(final, 1.0);
                }
            """,
        )

        # ------------------------------------------------------------------
        # Index buffer for ONE wall: an (n_z x n_y) quad grid.
        # Both walls share the same topology, so one IBO serves both.
        # ------------------------------------------------------------------
        nz, ny = grid_size, n_height
        idx = []
        for i in range(nz - 1):
            for j in range(ny - 1):
                a = i * ny + j
                b = a + 1
                c = (i + 1) * ny + j
                d = c + 1
                idx.extend([a, c, b,  b, c, d])

        self.ibo = self.ctx.buffer(np.array(idx, dtype="i4").tobytes())

        # Two separate VBOs — one per wall. This is what keeps the walls
        # disconnected: there is no triangle bridging them.
        # 6 floats per vertex now: position(3) + normal(3)
        nbytes = nz * ny * 6 * 4
        self.vbo_left  = self.ctx.buffer(reserve=nbytes)
        self.vbo_right = self.ctx.buffer(reserve=nbytes)

        self.vao_left = self.ctx.vertex_array(
            self.prog, [(self.vbo_left, "3f 3f", "in_position", "in_normal")], self.ibo
        )
        self.vao_right = self.ctx.vertex_array(
            self.prog, [(self.vbo_right, "3f 3f", "in_position", "in_normal")], self.ibo
        )

        # ------------------------------------------------------------------
        # CANYON FLOOR — its own mesh. Without it, everything below the
        # camera's eye line is empty space and renders black (the
        # horizontal "x-axis cut" across the middle of the frame).
        # ------------------------------------------------------------------
        self.n_floor_x = 12
        fx = self.n_floor_x
        fidx = []
        for i in range(nz - 1):
            for j in range(fx - 1):
                a = i * fx + j
                b = a + 1
                c = (i + 1) * fx + j
                d = c + 1
                fidx.extend([a, c, b,  b, c, d])
        self.ibo_floor = self.ctx.buffer(np.array(fidx, dtype="i4").tobytes())
        self.vbo_floor = self.ctx.buffer(reserve=nz * fx * 6 * 4)
        self.vao_floor = self.ctx.vertex_array(
            self.prog, [(self.vbo_floor, "3f 3f", "in_position", "in_normal")],
            self.ibo_floor
        )

        self._set("fog_near",    self.FOG_NEAR)
        self._set("fog_far",     self.FOG_FAR)
        self._set("wall_height", 5.5)   # colour-ramp reference (matches WALL_HEIGHT in terrain_gen.py)
        self._set("z_near",      2.0)
        self._set("z_far",     -18.0)

        self._rng = np.random.default_rng(seed=999)

        # Beat-responsive wave color state
        self.beat_glow = 0.5
        self.last_frame_was_beat = False

    # ------------------------------------------------------------------
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> tuple:
        """Convert HSV color space to RGB color space."""
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i %= 6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)
        return (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    def _set(self, name, value):
        if name in self.prog:
            self.prog[name].value = value

    # ------------------------------------------------------------------
    def render_frame(
        self,
        left_wall,
        right_wall,
        floor,
        frame_index:   int,
        total_frames:  int   = 1,
        band_energies: dict  = None,
        beat_pulse:    float = 0.0,
        frame_path:    str   = None,
    ) -> Image.Image:

        if band_energies is None:
            band_energies = {}

        # GPU → CPU if needed
        L = left_wall.get()  if hasattr(left_wall,  "get") else left_wall
        R = right_wall.get() if hasattr(right_wall, "get") else right_wall
        F = floor.get()      if hasattr(floor,      "get") else floor

        self.vbo_left.write(np.ascontiguousarray(L, dtype="f4").tobytes())
        self.vbo_right.write(np.ascontiguousarray(R, dtype="f4").tobytes())
        self.vbo_floor.write(np.ascontiguousarray(F, dtype="f4").tobytes())

        sub    = float(band_energies.get("sub_bass", 0.0))
        mid    = float(band_energies.get("mid",      0.0))
        treble = float(band_energies.get("treble",   0.0))

        # ----------------------------------------------------------------
        # Lightning
        # ----------------------------------------------------------------
        is_lightning = (
            (beat_pulse > 0.5 and self._rng.random() > 0.60) or
            (treble > 0.88    and self._rng.random() > 0.75)
        )

        # Smoothly decay the beat glow factor
        self.beat_glow *= 0.90  # decay rate

        # Spike the glow on a beat
        if beat_pulse > 0.5:
            if not self.last_frame_was_beat:
                self.beat_glow = 1.0
                self.last_frame_was_beat = True
        else:
            self.last_frame_was_beat = False

        if is_lightning:
            ambient_val = 1.05
            self._set("light_dir", (
                self._rng.uniform(-1.5, 1.5), 2.0, self._rng.uniform(-1.5, 1.5)
            ))
        else:
            ambient_val = 0.50 + mid * 0.15 + self.beat_glow * 0.40
            self._set("light_dir", (1.0, 1.5, -0.8))    # sideways+up: vertical walls catch it

        sky = (0.02, 0.01, 0.04)                    # background behind the canyon

        # Base hue shifts slowly over time (takes ~20 seconds to cycle the full color wheel)
        base_hue = (frame_index * 0.0008) % 1.0

        # Calculate dynamic RGB colors based on base_hue and beat_glow
        # When a beat hits, the rim glows (boost brightness and lower saturation slightly for a hot glowing neon core look)
        rim_glow_sat = 0.85 - self.beat_glow * 0.45
        rim_glow_val = 1.0 + self.beat_glow * 0.60
        r_rim, g_rim, b_rim = self._hsv_to_rgb(base_hue, rim_glow_sat, rim_glow_val)
        rim_intensity = 0.85 + treble * 0.15
        col_rim = (r_rim * rim_intensity, g_rim * rim_intensity, b_rim * rim_intensity)

        # Mid-tone also glows and is shifted
        mid_glow_val = 0.6 + self.beat_glow * 0.40
        r_mid, g_mid, b_mid = self._hsv_to_rgb((base_hue - 0.1) % 1.0, 0.9, mid_glow_val)
        col_mid = (r_mid, g_mid, b_mid)

        # Fog glow (light pouring down the slot) matches the dynamic wall color
        r_fog, g_fog, b_fog = self._hsv_to_rgb(base_hue, 0.5, 0.8)
        fog_glow = (r_fog, g_fog * (0.8 + treble * 0.2), b_fog)

        # Floor color matches the sky background for a seamless look
        col_floor = sky

        self._set("ambient",   ambient_val)
        self._set("fog_color", fog_glow)
        self._set("col_floor", col_floor)
        self._set("col_mid",   col_mid)
        self._set("col_rim",   col_rim)

        # ----------------------------------------------------------------
        # Camera — inside the canyon, low, looking down the corridor
        # ----------------------------------------------------------------
        tt   = frame_index * 0.05
        vx   = np.sin(tt * 1.7) * 0.05
        vy   = np.cos(tt * 2.3) * 0.03
        drop = beat_pulse * -0.06

        cam = glm.vec3(vx, self.CAM_HEIGHT + vy + drop, self.CAM_Z)
        tgt = glm.vec3(vx * 0.5, self.LOOK_Y, self.LOOK_Z)

        proj = glm.perspective(
            glm.radians(self.FOV_DEGREES),
            self.RENDER_WIDTH / self.RENDER_HEIGHT,
            0.05, 60.0
        )
        view = glm.lookAt(cam, tgt, glm.vec3(0, 1, 0))
        mvp  = proj * view

        if "mvp" in self.prog:
            # NOTE: glm stores matrices COLUMN-major. np.array(mvp).tobytes()
            # serialises them row-major, which uploads the TRANSPOSE and
            # silently destroys the projection (the floor collapses onto the
            # horizon and the frame gets cut in half). The .T restores the
            # correct memory order.
            self.prog["mvp"].write(np.array(mvp, dtype="f4").T.tobytes())
        self._set("cam_pos", (cam.x, cam.y, cam.z))

        # ----------------------------------------------------------------
        # Draw BOTH walls
        # ----------------------------------------------------------------
        self.fbo.use()
        self.ctx.clear(sky[0], sky[1], sky[2], 1.0)
        self.vao_left.render(moderngl.TRIANGLES)
        self.vao_right.render(moderngl.TRIANGLES)
        self.vao_floor.render(moderngl.TRIANGLES)

        raw = self.fbo.read(components=3)
        img = Image.frombytes("RGB", (self.RENDER_WIDTH, self.RENDER_HEIGHT), raw)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # ----------------------------------------------------------------
        # Lightning bolt in the sky slot above
        # ----------------------------------------------------------------
        if is_lightning:
            d  = ImageDraw.Draw(img)
            cx = self._rng.uniform(self.RENDER_WIDTH * 0.40, self.RENDER_WIDTH * 0.60)
            cy = 0.0
            pts = [(cx, cy)]
            while cy < self.RENDER_HEIGHT * 0.45:
                cy += self._rng.uniform(14, 28)
                cx += self._rng.uniform(-26, 26)
                pts.append((cx, cy))
            d.line(pts, fill=(170, 220, 255), width=5)
            d.line(pts, fill=(255, 255, 255), width=2)

        # ----------------------------------------------------------------
        # Bloom
        # ----------------------------------------------------------------
        if self.BLOOM_STRENGTH > 0:
            bright = img.point(lambda p: p if p > self.BLOOM_THRESHOLD else 0)
            blur   = bright.filter(ImageFilter.GaussianBlur(radius=self.BLOOM_RADIUS))
            img    = Image.blend(img, ImageChops.screen(img, blur), self.BLOOM_STRENGTH)

        if frame_path:
            img.save(frame_path)

        return img
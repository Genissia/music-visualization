"""
renderer.py
-----------
Driver POV canyon road — camera raised and pulled back so you see
the full canyon walls on left/right with the road stretching ahead.
"""

import moderngl
import numpy as np
from pyglm import glm
from PIL import Image, ImageFilter, ImageChops, ImageDraw


class TerrainRenderer:

    RENDER_WIDTH  = 720
    RENDER_HEIGHT = 720

    FOV_DEGREES = 75.0      # wide enough to see both walls comfortably

    FOG_NEAR = 2.0
    FOG_FAR  = 8.5

    BLOOM_THRESHOLD = 110
    BLOOM_RADIUS    = 8
    BLOOM_STRENGTH  = 0.45

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

                uniform mat4  mvp;
                uniform float amp_bass;
                uniform float min_z;
                uniform float max_z;

                out float v_height;
                out vec3  v_pos;

                void main() {
                    // X: left/right walls — scale outward so canyon is wide
                    float world_x = in_position.x * 2.8;

                    // Y: height, always >= 0, amplify with bass
                    float world_y = in_position.y * (1.0 + amp_bass * 0.35);

                    // Z: map grid [-1,+1] → world [-7.0, +3.5]
                    // +3.5 = behind camera (cam_z=2.5), -7.0 = deep ahead
                    float norm_z  = (in_position.z - min_z) / (max_z - min_z + 1e-6);
                    float world_z = mix(-7.0, 3.5, norm_z);

                    vec3 pos    = vec3(world_x, world_y, world_z);
                    gl_Position = mvp * vec4(pos, 1.0);
                    v_height    = world_y;
                    v_pos       = pos;
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

                uniform vec3 col_road;
                uniform vec3 col_wall_base;
                uniform vec3 col_wall_peak;

                uniform vec3  fog_color;
                uniform float fog_near;
                uniform float fog_far;

                void main() {
                    vec3 normal = normalize(cross(dFdx(v_pos), dFdy(v_pos)));

                    vec3  L    = normalize(light_dir);
                    float diff = max(dot(normal, L), 0.0);

                    vec3  V    = normalize(cam_pos - v_pos);
                    vec3  R    = reflect(-L, normal);
                    float spec = pow(max(dot(V, R), 0.0), 32.0) * 0.5;

                    float lighting = ambient + (diff + spec) * 1.4;

                    vec3 color;
                    if (v_height < 0.04) {
                        color = col_road;
                    } else {
                        float t = clamp(v_height * 0.5, 0.0, 1.0);
                        color = mix(col_wall_base, col_wall_peak, t);
                    }

                    vec3 lit  = color * lighting;
                    float dist  = length(v_pos - cam_pos);
                    float fog_t = clamp((dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
                    vec3 final  = mix(lit, fog_color, fog_t);

                    fragColor = vec4(final, 1.0);
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

        self._set_uniform("fog_near",  self.FOG_NEAR)
        self._set_uniform("fog_far",   self.FOG_FAR)
        self._set_uniform("light_dir", (1.0, 4.0, -1.0))
        self._rng = np.random.default_rng(seed=999)

    def _set_uniform(self, name, value):
        if name in self.prog:
            self.prog[name].value = value

    def render_frame(
        self,
        X_gpu, Y_gpu, Z_gpu,
        frame_index:   int,
        total_frames:  int   = 1,
        band_energies: dict  = None,
        beat_pulse:    float = 0.0,
        frame_path:    str   = None,
    ) -> Image.Image:

        if band_energies is None:
            band_energies = {}

        progress = frame_index / max(total_frames, 1)

        X = X_gpu.get() if hasattr(X_gpu, "get") else X_gpu
        Y = Y_gpu.get() if hasattr(Y_gpu, "get") else Y_gpu
        Z = Z_gpu.get() if hasattr(Z_gpu, "get") else Z_gpu

        vertices = np.dstack((X, Y, Z)).astype(np.float32).tobytes()
        self.vbo.write(vertices)

        self._set_uniform("min_z", float(Z.min()))
        self._set_uniform("max_z", float(Z.max()))

        sub_bass = float(band_energies.get("sub_bass", 0.0))
        mid      = float(band_energies.get("mid",      0.0))
        treble   = float(band_energies.get("treble",   0.0))

        self._set_uniform("amp_bass",   sub_bass)
        self._set_uniform("amp_mid",    mid)
        self._set_uniform("amp_treble", treble)

        is_lightning = (
            (beat_pulse > 0.84 and self._rng.random() > 0.35) or
            (treble > 0.88     and self._rng.random() > 0.60)
        )

        if is_lightning:
            ambient_val = self._rng.uniform(0.90, 1.30)
            fog_color   = (0.65, 0.72, 0.92)
            self._set_uniform("light_dir", (
                self._rng.uniform(-3.0, 3.0), 6.0,
                self._rng.uniform(-3.0, 3.0)
            ))
        else:
            ambient_val = 0.30 + mid * 0.15
            fog_color   = (0.01, 0.005, 0.02)
            self._set_uniform("light_dir", (1.0, 4.0, -1.0))

        self._set_uniform("ambient",   ambient_val)
        self._set_uniform("fog_color", fog_color)

        self._set_uniform("col_road",      (0.04, 0.04, 0.05))
        self._set_uniform("col_wall_base", (0.24, 0.02, 0.40))
        self._set_uniform("col_wall_peak", (
            0.65 + mid * 0.25, 0.15,
            0.98 + beat_pulse * 0.02
        ))

        # ----------------------------------------------------------------
        # Camera — raised higher, pulled back further
        #
        # Key changes from v3:
        #   cam_y: 0.18 → 0.9   (higher up so we look DOWN at the road,
        #                         not from ground level INTO the wall base)
        #   cam_z: 2.10 → 2.5   (slightly further back for more FOV coverage)
        #   look_at y: 0.12 → 0.0  (look at road surface, not wall mid)
        #   look_at z: -2.5 → -3.5 (look further down the road)
        # ----------------------------------------------------------------
        t            = progress * 40.0
        vibration_x  = np.sin(t * 85.0) * 0.0015
        vibration_y  = np.cos(t * 95.0) * 0.0010
        chassis_drop = beat_pulse * -0.015

        cam_x = 0.0 + vibration_x
        cam_y = 1.4  + vibration_y + chassis_drop   # raised from 0.18
        cam_z = 2.5                                  # pulled back from 2.10

        look_at = glm.vec3(0.0, 0.0, -3.5)          # look at road surface ahead

        proj = glm.perspective(
            glm.radians(self.FOV_DEGREES),
            self.RENDER_WIDTH / self.RENDER_HEIGHT,
            0.01, 18.0
        )
        view = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            look_at,
            glm.vec3(0.0, 1.0, 0.0)
        )
        mvp = proj * view

        if "mvp" in self.prog:
            self.prog["mvp"].write(np.array(mvp, dtype=np.float32).tobytes())
        self._set_uniform("cam_pos", (cam_x, cam_y, cam_z))

        self.fbo.use()
        self.ctx.clear(fog_color[0], fog_color[1], fog_color[2], 1.0)
        self.vao.render(moderngl.TRIANGLES)

        image_data = self.fbo.read(components=3)
        img = Image.frombytes("RGB", (self.RENDER_WIDTH, self.RENDER_HEIGHT), image_data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Lightning bolt overlay
        if is_lightning:
            draw    = ImageDraw.Draw(img)
            start_x = self._rng.uniform(self.RENDER_WIDTH * 0.35, self.RENDER_WIDTH * 0.65)
            cx, cy  = start_x, 0
            bolt    = [(cx, cy)]
            while cy < self.RENDER_HEIGHT * 0.55:
                cy += self._rng.uniform(15, 30)
                cx += self._rng.uniform(-40, 40)
                bolt.append((cx, cy))
            draw.line(bolt, fill=(160, 215, 255), width=6)
            draw.line(bolt, fill=(255, 255, 255), width=2)
            if self._rng.random() > 0.45 and len(bolt) > 4:
                bx, by = bolt[len(bolt) // 2]
                branch = [(bx, by)]
                for _ in range(3):
                    by += self._rng.uniform(12, 24)
                    bx += self._rng.uniform(-30, 30)
                    branch.append((bx, by))
                draw.line(branch, fill=(130, 195, 255), width=3)
                draw.line(branch, fill=(255, 255, 255), width=1)

        # Bloom
        if self.BLOOM_STRENGTH > 0:
            bright_pass  = img.point(lambda p: p if p > self.BLOOM_THRESHOLD else 0)
            blur_layer   = bright_pass.filter(ImageFilter.GaussianBlur(radius=self.BLOOM_RADIUS))
            screen_blend = ImageChops.screen(img, blur_layer)
            img          = Image.blend(img, screen_blend, self.BLOOM_STRENGTH)

        if frame_path:
            img.save(frame_path)

        return img
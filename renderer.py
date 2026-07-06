import moderngl
import numpy as np
from pyglm import glm
from PIL import Image, ImageFilter, ImageChops, ImageDraw


class TerrainRenderer:
    """
    Renders a 3D musical highway pass flanked by harmonic wave mountains.
    Directly simulates wave interference patterns inspired by image_ed6e44.png,
    propagating them over time to mimic continuous forward driving motion.
    """

    # ------------------------------------------------------------------
    # CINEMATIC WAVE-HIGHWAY PANEL
    # ------------------------------------------------------------------
    RENDER_WIDTH = 720
    RENDER_HEIGHT = 720

    BASE_WAVE_HEIGHT = 0.350  # Fundamental baseline height of the wave crests
    HIGHWAY_WIDTH = 0.280  # Size of the flat driving lane before waves start rising
    FOV_DEGREES = 65.0  # Field-of-view tracking

    # Fog clipping boundaries
    FOG_NEAR = 0.1
    FOG_FAR = 6.0

    # Post-process bloom configuration
    BLOOM_THRESHOLD = 110
    BLOOM_RADIUS = 8
    BLOOM_STRENGTH = 0.45

    # ------------------------------------------------------------------

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        w, h = self.RENDER_WIDTH, self.RENDER_HEIGHT

        # 1. Initialize standalone headless graphics context
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST)

        # 2. Allocation of render framebuffers
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((w, h)),
        )

        # 3. Harmonic Wave-Synthesis Shader Pipeline
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;       // Raw data arrays from terrain_gen

                uniform mat4  mvp;          
                uniform float wave_time;   // Continuous phase offset simulating car speed

                // Audio frequency modifiers
                uniform float amp_bass;    
                uniform float amp_mid;     
                uniform float amp_treble;  

                // Data normalization boundaries
                uniform float min_x;
                uniform float max_x;
                uniform float min_z;
                uniform float max_z;
                uniform float highway_width;
                uniform float base_height;

                out float v_height;        // Drives the neon lighting gradients
                out vec3  v_pos;           

                void main() {
                    // Normalize the coordinates cleanly down to [-1.0, 1.0] bounding boxes
                    float norm_x = -1.0 + 2.0 * ((in_position.x - min_x) / (max_x - min_x + 1e-6));
                    float norm_z = -1.0 + 2.0 * ((in_position.z - min_z) / (max_z - min_z + 1e-6));

                    // Isolate the central highway lane axis
                    float distance_from_center = abs(norm_x);
                    float mountain_ramp        = smoothstep(highway_width, 0.90, distance_from_center);

                    // Procedural Wave Synthesis Layer (Directly replicates harmonics in image_ed6e44.png)
                    // Wave 1: The deep fundamental low-frequency swell (Bass driven)
                    float wave1 = sin(norm_z * 5.0 - wave_time * 2.0) * (1.0 + amp_bass * 1.5);

                    // Wave 2: The secondary medium harmonic overlay ripples (Mid driven)
                    float wave2 = sin(norm_z * 14.0 + wave_time * 4.5) * 0.30 * (1.0 + amp_mid * 2.0);

                    // Wave 3: The high-frequency micro-vibrations across the crest ridges (Treble driven)
                    float wave3 = cos(norm_x * 12.0) * sin(norm_z * 32.0 - wave_time * 7.0) * 0.08 * (1.0 + amp_treble * 2.5);

                    // Composite all mathematical waves together into a unified mountain ridge structure
                    float composite_wave = (wave1 + wave2 + wave3) * base_height;

                    // Enforce the highway: Flatten the middle area completely, ramp up waves to the sides
                    float final_height = composite_wave * mountain_ramp;

                    // Reconstruct localized world coordinates
                    vec3 localized_pos = vec3(norm_x * 2.0, final_height, norm_z * 2.8);

                    gl_Position = mvp * vec4(localized_pos, 1.0);
                    v_height    = final_height;
                    v_pos       = localized_pos;
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

                uniform vec3 col_highway;  
                uniform vec3 col_wave_slope;     
                uniform vec3 col_wave_crest;     

                uniform vec3  fog_color;   
                uniform float fog_near;
                uniform float fog_far;

                void main() {
                    // Generate crisp low-poly geometric shading normal planes
                    vec3 normal = normalize(cross(dFdx(v_pos), dFdy(v_pos)));

                    // Calculate basic diffuse directional illumination
                    vec3  L    = normalize(light_dir);
                    float diff = max(dot(normal, L), 0.0);

                    // High specularity multiplier to make the waves look wet and metallic
                    vec3  V    = normalize(cam_pos - v_pos);
                    vec3  R    = reflect(-L, normal);
                    float spec = pow(max(dot(V, R), 0.0), 32.0) * 0.50;

                    float lighting = ambient + (diff + spec) * 1.4;

                    // Dynamic wave layer color mapping
                    vec3 color;
                    if (abs(v_height) < 0.02) {
                        color = col_highway;  // Flat clean road asphalt
                    } else {
                        // Blend between mountain slopes and illuminated neon wave crests
                        float t = clamp(abs(v_height) * 2.2, 0.0, 1.0);
                        color = mix(col_wave_slope, col_wave_crest, t);
                    }

                    vec3 lit = color * lighting;

                    // Apply depth fog boundary blending
                    float dist  = length(v_pos - cam_pos);
                    float fog_t = clamp((dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
                    fog_t       = min(fog_t, 0.82); 

                    vec3 final = mix(lit, fog_color, fog_t);
                    fragColor  = vec4(final, 1.0);
                }
            """,
        )

        # 4. Generate index array layout map
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

        # Set uniform constants
        self.prog["fog_near"].value = self.FOG_NEAR
        self.prog["fog_far"].value = self.FOG_FAR
        self.prog["highway_width"].value = self.HIGHWAY_WIDTH
        self.prog["base_height"].value = self.BASE_WAVE_HEIGHT

        # Initialize global random tracking
        self._rng = np.random.default_rng(seed=999)

    def render_frame(
            self,
            X_gpu,
            Y_gpu,
            Z_gpu,
            frame_index: int,
            total_frames: int = 1,
            band_energies: dict = None,
            beat_pulse: float = 0.0,
            frame_path: str = None,
    ) -> Image.Image:
        if band_energies is None:
            band_energies = {"sub_bass": 0, "mid": 0, "treble": 0}

        progress = frame_index / max(total_frames, 1)

        # Safely capture arrays out of target device contexts
        X = X_gpu.get() if hasattr(X_gpu, "get") else X_gpu
        Y = Y_gpu.get() if hasattr(Y_gpu, "get") else Y_gpu
        Z = Z_gpu.get() if hasattr(Z_gpu, "get") else Z_gpu

        vertices = np.dstack((X, Y, Z)).astype(np.float32).tobytes()
        self.vbo.write(vertices)

        # Update dynamic grid tracking boundaries
        self.prog["min_x"].value = float(X.min())
        self.prog["max_x"].value = float(X.max())
        self.prog["min_z"].value = float(Z.min())
        self.prog["max_z"].value = float(Z.max())

        # ------------------------------------------------------------------
        # Audio Intensity Modifiers
        # ------------------------------------------------------------------
        sub_bass = float(band_energies.get("sub_bass", 0.0))
        mid = float(band_energies.get("mid", 0.0))
        treble = float(band_energies.get("treble", 0.0))

        self.prog["amp_bass"].value = sub_bass
        self.prog["amp_mid"].value = mid
        self.prog["amp_treble"].value = treble

        # Move the waves back along the Z-axis over time to simulate car speed
        self.prog["wave_time"].value = progress * 60.0

        # ------------------------------------------------------------------
        # Storm State Machine Engine
        # ------------------------------------------------------------------
        is_lightning = (beat_pulse > 0.84 and self._rng.random() > 0.35) or (
                    treble > 0.88 and self._rng.random() > 0.60)

        if is_lightning:
            ambient_brightness = self._rng.uniform(0.90, 1.30)
            sky_fog_color = (0.65, 0.72, 0.92)  # Blinding lightning flash sky
            self.prog["light_dir"].value = (self._rng.uniform(-3.0, 3.0), 6.0, self._rng.uniform(-3.0, 3.0))
        else:
            ambient_brightness = 0.30 + (mid * 0.15)
            sky_fog_color = (0.01, 0.005, 0.02)  # Dark synthwave thunderstorm void
            self.prog["light_dir"].value = (0.4, 3.0, -1.0)

        self.prog["ambient"].value = ambient_brightness
        self.prog["fog_color"].value = sky_fog_color

        # ------------------------------------------------------------------
        # Color Palettes (Sleek High-Contrast Wave Highway)
        # ------------------------------------------------------------------
        self.prog["col_highway"].value = (0.04, 0.04, 0.05)  # Pitch black asphalt floor center
        self.prog["col_wave_slope"].value = (0.24, 0.02, 0.40)  # Rich purple deep wave valleys
        self.prog["col_wave_crest"].value = (0.65 + mid * 0.25, 0.15,
                                             0.98 + beat_pulse * 0.02)  # High-intensity glowing peaks

        # ------------------------------------------------------------------
        # Fixed Highway Center Cockpit Camera Tracking
        # ------------------------------------------------------------------
        t = progress * 40.0
        vibration_x = np.sin(t * 85.0) * 0.0012
        vibration_y = np.cos(t * 95.0) * 0.0008
        chassis_drop = beat_pulse * -0.010

        # Position camera dead-center down the lane looking forward
        cam_x = 0.0 + vibration_x
        cam_y = 0.18 + vibration_y + chassis_drop  # Positioned right over the highway floor
        cam_z = 2.10  # Locked at the entry boundary looking forward

        # Look straight down the negative Z-axis corridor towards the horizon
        look_at = glm.vec3(0.0, 0.12, -2.5)

        proj = glm.perspective(glm.radians(self.FOV_DEGREES), self.RENDER_WIDTH / self.RENDER_HEIGHT, 0.01, 10.0)
        view = glm.lookAt(glm.vec3(cam_x, cam_y, cam_z), look_at, glm.vec3(0.0, 1.0, 0.0))
        mvp = proj * view

        self.prog["mvp"].write(np.array(mvp, dtype=np.float32).tobytes())
        self.prog["cam_pos"].value = (cam_x, cam_y, cam_z)

        # ------------------------------------------------------------------
        # Execute Scene Frame Render Pass
        # ------------------------------------------------------------------
        self.fbo.use()
        self.ctx.clear(sky_fog_color[0], sky_fog_color[1], sky_fog_color[2], 1.0)
        self.vao.render(moderngl.TRIANGLES)

        image_data = self.fbo.read(components=3)
        img = Image.frombytes("RGB", (self.RENDER_WIDTH, self.RENDER_HEIGHT), image_data)

        # ------------------------------------------------------------------
        # 2D Procedural Lightning Vector Overlay
        # ------------------------------------------------------------------
        if is_lightning:
            overlay_canvas = ImageDraw.Draw(img)

            # Select lightning strike coordinates along sky boundaries
            start_x = self._rng.uniform(self.RENDER_WIDTH * 0.30, self.RENDER_WIDTH * 0.70)
            cur_x, cur_y = start_x, 0

            bolt_points = [(cur_x, cur_y)]

            # Form multi-segment zig-zag paths stretching toward the horizon
            while cur_y < self.RENDER_HEIGHT * 0.42:
                cur_y += self._rng.uniform(15, 30)
                cur_x += self._rng.uniform(-35, 35)
                bolt_points.append((cur_x, cur_y))

            # Draw outer atmospheric glow layer
            overlay_canvas.line(bolt_points, fill=(160, 215, 255), width=6)
            # Draw intense white plasma core filament
            overlay_canvas.line(bolt_points, fill=(255, 255, 255), width=2)

            # Generate random secondary branches fork paths
            if self._rng.random() > 0.45 and len(bolt_points) > 4:
                branch_origin = bolt_points[len(bolt_points) // 2]
                bx, by = branch_origin
                branch_points = [(bx, by)]

                for _ in range(3):
                    by += self._rng.uniform(12, 24)
                    bx += self._rng.uniform(-25, 25)
                    branch_points.append((bx, by))

                overlay_canvas.line(branch_points, fill=(130, 195, 255), width=3)
                overlay_canvas.line(branch_points, fill=(255, 255, 255), width=1)

        # ------------------------------------------------------------------
        # Post-Process Glow Bloom Pipeline
        # ------------------------------------------------------------------
        if self.BLOOM_STRENGTH > 0:
            bright_pass = img.point(lambda p: p if p > self.BLOOM_THRESHOLD else 0)
            blur_layer = bright_pass.filter(ImageFilter.GaussianBlur(radius=self.BLOOM_RADIUS))
            screen_blend = ImageChops.screen(img, blur_layer)
            img = Image.blend(img, screen_blend, self.BLOOM_STRENGTH)

        if frame_path:
            img.save(frame_path)

        return img
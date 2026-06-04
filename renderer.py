import moderngl
import numpy as np
import glm
from PIL import Image

class TerrainRenderer:
    def __init__(self, grid_size, width=1920, height=1080):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
        # 1. Create a Headless Context
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # 2. Setup the Framebuffer
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((width, height))
        )
        
        # 3. Write our GLSL Shaders
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                out float height;
                void main() {
                    // We scale down the Y axis slightly so the 190+ height fits on screen
                    vec3 pos = vec3(in_position.x, in_position.y * 0.02, in_position.z);
                    gl_Position = mvp * vec4(pos, 1.0);
                    height = pos.y;
                }
            ''',
            fragment_shader='''
                #version 330
                in float height;
                out vec4 fragColor;
                void main() {
                    vec3 base_color = vec3(0.1, 0.1, 0.2);
                    vec3 peak_color = vec3(1.0, 0.3, 0.0);
                    vec3 final_color = mix(base_color, peak_color, clamp(height, 0.0, 1.0));
                    fragColor = vec4(final_color, 1.0);
                }
            '''
        )

        # 4. Initialize Buffers with dummy data
        # We will update the VBO every frame, but keep the IBO static
        self.vbo = self.ctx.buffer(reserve=grid_size * grid_size * 12) # 3 floats * 4 bytes
        
        indices = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                top_left = i * grid_size + j
                top_right = top_left + 1
                bottom_left = (i + 1) * grid_size + j
                bottom_right = bottom_left + 1
                indices.extend([top_left, bottom_left, top_right, top_right, bottom_left, bottom_right])
        
        self.ibo = self.ctx.buffer(np.array(indices, dtype='i4').tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f', 'in_position')], self.ibo)

        # 5. Setup Camera
        view = glm.lookAt(glm.vec3(0, 3, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        proj = glm.perspective(glm.radians(45.0), width / height, 0.1, 100.0)
        self.mvp = proj * view
        self.prog['mvp'].write(np.array(self.mvp, dtype='f4').tobytes())

    def render_frame(self, X_gpu, Y_gpu, Z_gpu, frame_path=None):
        self.fbo.use()
        self.fbo.clear(0.05, 0.05, 0.08, 1.0)

        # Update Vertex Buffer
        X, Y, Z = X_gpu.get(), Y_gpu.get(), Z_gpu.get()
        vertices = np.stack([X, Y, Z], axis=-1).astype('f4').tobytes()
        self.vbo.write(vertices)

        # Render
        self.vao.render(moderngl.TRIANGLES)

        # Read result
        image_data = self.fbo.color_attachments[0].read()
        img = Image.frombytes('RGBA', (self.width, self.height), image_data)
        img = img.convert('RGB').transpose(Image.FLIP_TOP_BOTTOM)
        
        if frame_path:
            img.save(frame_path)
        
        return img

# --- Test Block ---
if __name__ == "__main__":
    from audio_process import extract_audio_frames
    from terrain_gen import create_terrain_frame
    
    print("Extracting audio...")
    frames, sr = extract_audio_frames("test_song.mp3")
    
    renderer = TerrainRenderer(grid_size=100)
    
    print("Generating terrain...")
    X, Y, Z = create_terrain_frame(frames[1000], grid_size=100)
    
    print("Rendering 3D mesh...")
    renderer.render_frame(X, Y, Z, frame_path='test_frame.png')

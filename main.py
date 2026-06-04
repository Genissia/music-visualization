import os
import shutil
import subprocess
from audio_process import extract_audio_frames
from terrain_gen import create_terrain_frame
from renderer import TerrainRenderer
import time

def generate_video(audio_path, output_video="output.mp4", fps=60, grid_size=150):
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # 1. Extract Audio Data
    audio_frames, sr = extract_audio_frames(audio_path, fps=fps)
    num_frames = audio_frames.shape[0]
    
    # 2. Initialize Renderer
    renderer = TerrainRenderer(grid_size=grid_size)
    
    print(f"Starting render of {num_frames} frames...")
    start_time = time.time()
    
    # 3. Render Loop
    for i in range(num_frames):
        # Progress update every 100 frames
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Rendering frame {i}/{num_frames} ({i/num_frames*100:.1f}%) - Elapsed: {elapsed:.2f}s")
        
        # Generate Terrain for this frame
        X, Y, Z = create_terrain_frame(audio_frames[i], grid_size=grid_size)
        
        # Render and Save
        frame_filename = os.path.join(temp_dir, f"frame_{i:06d}.png")
        renderer.render_frame(X, Y, Z, frame_path=frame_filename)
    
    total_render_time = time.time() - start_time
    print(f"\nRender complete! Total time: {total_render_time:.2f}s")
    
    # 4. Assemble Video with FFmpeg
    print("Assembling video with FFmpeg...")
    
    # Use ffmpeg-python if available, or call subprocess
    # Command: ffmpeg -y -framerate 60 -i temp_frames/frame_%06d.png -i audio.mp3 -c:v libx264 -pix_fmt yuv420p output.mp4
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%06d.png'),
        '-i', audio_path,
        '-c:v', 'libx264',
        '-shortest',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video successfully saved to {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg execution: {e}")
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure it is installed and in your PATH.")

    # Cleanup (Optional)
    # shutil.rmtree(temp_dir)

if __name__ == "__main__":
    AUDIO_FILE = "test_song.mp3"
    if os.path.exists(AUDIO_FILE):
        generate_video(AUDIO_FILE)
    else:
        print(f"Audio file '{AUDIO_FILE}' not found. Please provide a song to visualize.")

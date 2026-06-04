import numpy as np
import cupy as cp
from perlin_noise import PerlinNoise
import time

# We initialize the noise generator globally
noise_gen = PerlinNoise(octaves=3, seed=42)
_cached_noise = None

def get_noise_matrix(grid_size, noise_scale):
    global _cached_noise
    if _cached_noise is not None and _cached_noise.shape == (grid_size, grid_size):
        return _cached_noise
    
    print(f"Generating static {grid_size}x{grid_size} noise matrix...")
    noise_matrix = np.zeros((grid_size, grid_size), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            noise_matrix[i][j] = noise_gen([i * noise_scale, j * noise_scale])
    
    _cached_noise = cp.array(noise_matrix)
    return _cached_noise

def create_terrain_frame(audio_frame, grid_size=100, noise_scale=0.05, amplitude=5.0):
    # 1. Base Grid (X, Z plane)
    x = cp.linspace(-1, 1, grid_size)
    z = cp.linspace(-1, 1, grid_size)
    X, Z = cp.meshgrid(x, z)

    # 2. Map audio frequencies to the grid
    original_bins = np.linspace(0, 1, len(audio_frame))
    target_bins = np.linspace(0, 1, grid_size)
    resized_audio = np.interp(target_bins, original_bins, audio_frame)
    
    Y_audio = cp.tile(cp.array(resized_audio), (grid_size, 1)) * amplitude

    # 3. Apply Cached Perlin Noise
    Y_noise = get_noise_matrix(grid_size, noise_scale) * 2.0 

    # 4. Final Terrain Height
    Y = Y_audio + Y_noise

    return X, Y, Z

# --- Test Block ---
if __name__ == "__main__":
    from audio_process import extract_audio_frames
    
    # 1. Grab just the first few frames from your song to test
    sample_audio = "test_song.mp3" # Ensure this matches your file name
    print("Extracting test frames...")
    frames, sr = extract_audio_frames(sample_audio)
    
    # 2. Pick a single frame of audio data (e.g., frame 1000 where music might be playing)
    test_frame = frames[1000]
    
    print("\nSending data to GPU to build terrain...")
    start_time = time.time()
    
    # 3. Generate the 3D coordinates
    X, Y, Z = create_terrain_frame(test_frame, grid_size=100)
    
    end_time = time.time()
    
    print(f"Success! Terrain generated in {end_time - start_time:.4f} seconds.")
    print(f"X, Y, Z matrices generated on GPU with shape: {X.shape}")
    print(f"Maximum terrain height (Y) for this frame: {float(cp.max(Y)):.2f}")
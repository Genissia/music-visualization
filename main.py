import os
import subprocess
import time
import argparse

import numpy as np

from audio_process import extract_audio_frames
from terrain_gen   import create_terrain_frame
from renderer      import TerrainRenderer


def _progress_bar(current: int, total: int, bar_width: int = 30) -> str:
    frac   = current / max(total, 1)
    filled = int(bar_width * frac)
    bar    = "█" * filled + "░" * (bar_width - filled)
    return f"[{bar}] {current}/{total} ({frac*100:.1f}%)"


def generate_video(
    audio_path:   str,
    output_video: str = "output.mp4",
    fps:          int = 60,
    grid_size:    int = 150,
):
    # ------------------------------------------------------------------
    # 1. Audio analysis
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 / 3  –  Audio Analysis")
    print("=" * 60)

    spectrogram, band_energies, beat_frames, onset_env, sr = extract_audio_frames(
        audio_path, fps=fps
    )
    num_frames = spectrogram.shape[0]

    print(f"\n  Frames  : {num_frames}")
    print(f"  Duration: {num_frames / fps:.1f}s  at {fps} FPS")

    # ------------------------------------------------------------------
    # 2. Renderer initialisation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 / 3  –  Renderer Initialisation")
    print("=" * 60)

    renderer = TerrainRenderer(grid_size=grid_size)
    print("  ModernGL headless context ready.")

    # ------------------------------------------------------------------
    # 3. FFmpeg Live Pipe Setup
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 / 3  –  Cinematic Stream Render (In-Memory)")
    print("=" * 60)

    # We configure FFmpeg to read raw video bytes directly from standard input (stdin)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",                  # Input format is raw pixels
        "-pix_fmt", "rgb24",               # 3 channels (Red, Green, Blue), 8 bits each
        "-s", f"{renderer.RENDER_WIDTH}x{renderer.RENDER_HEIGHT}", # Target resolution
        "-framerate", str(fps),            # Video frame rate
        "-i", "pipe:0",                    # Read video frames from Python memory stream (stdin)
        "-i", audio_path,                  # Second input source is the audio file
        "-c:v",     "libx264",
        "-preset",  "fast",
        "-crf",     "18",
        "-pix_fmt", "yuv420p",
        "-c:a",     "aac",
        "-b:a",     "320k",
        "-ac",      "2",
        "-shortest",
        output_video,
    ]

    try:
        # Launch FFmpeg as an active asynchronous background process
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print("  ✓ FFmpeg process pipe established successfully.")
    except FileNotFoundError:
        print("\n  ✗ FFmpeg not found. Install it and add it to your PATH.")
        return

    start_time = time.time()

    # ------------------------------------------------------------------
    # Core Stream Loop
    # ------------------------------------------------------------------
    for i in range(num_frames):
        frame_spec   = spectrogram[i]
        frame_bands  = {k: float(v[i]) for k, v in band_energies.items()}
        frame_beat   = float(beat_frames[i])

        # Generate terrain coordinates
        X, Y, Z = create_terrain_frame(
            audio_frame   = frame_spec,
            band_energies = frame_bands,
            beat_pulse    = frame_beat,
            grid_size     = grid_size,
            frame_index   = i,
        )

        # Render frame (Returns a PIL Image object)
        img = renderer.render_frame(
            X_gpu         = X,
            Y_gpu         = Y,
            Z_gpu         = Z,
            frame_index   = i,
            total_frames  = num_frames,
            band_energies = frame_bands,
            beat_pulse    = frame_beat,
            frame_path    = None,
        )

        # --- THE MAGIC TRICK ---
        # Convert the PIL image directly into a sequential stream of binary RGB bytes
        raw_rgb_bytes = img.tobytes()

        # Shove the raw bytes down the standard input pipe directly into FFmpeg
        ffmpeg_process.stdin.write(raw_rgb_bytes)

        # Progress tracking
        if i % 100 == 0 or i == num_frames - 1:
            elapsed   = time.time() - start_time
            speed     = (i + 1) / elapsed if elapsed > 0 else 0
            eta       = (num_frames - i - 1) / speed if speed > 0 else 0
            bar       = _progress_bar(i + 1, num_frames)
            print(f"  {bar}  |  {speed:.1f} fps  |  ETA {eta:.0f}s")

    # Clean close: Tell FFmpeg we are completely out of video frames
    print("\n  Closing memory streams and rendering final video file...")
    ffmpeg_process.stdin.close()

    # Wait for FFmpeg to finish packaging the audio track and video into the MP4 container
    _, stderr = ffmpeg_process.communicate()

    total_render_time = time.time() - start_time

    if ffmpeg_process.returncode == 0:
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n  ✓ SUCCESS: '{output_video}' ({size_mb:.1f} MB)")
        print(f"  Total render time: {total_render_time:.1f}s ({num_frames / total_render_time:.1f} fps average)")
    else:
        print(f"\n  ✗ FFmpeg compilation failed:\n{stderr.decode()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Landscape Visualizer")
    parser.add_argument("audio", nargs="?", default="test_song1.mp3")
    parser.add_argument("-o", "--output", default="visual_landscape.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--grid", type=int, default=150)
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file '{args.audio}' not found.")
        raise SystemExit(1)

    generate_video(
        audio_path   = args.audio,
        output_video = args.output,
        fps          = args.fps,
        grid_size    = args.grid,
    )
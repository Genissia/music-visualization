"""
Entry point and pipeline orchestrator for the Music Landscape Visualizer.
"""

import os
import subprocess
import time
import argparse

import numpy as np

from audio_process import extract_audio_frames
from terrain_gen import create_terrain_frame
from renderer import TerrainRenderer

def _progress_bar(current: int, total: int, bar_width: int = 30) -> str:
    """Return progress bar string using standard ASCII characters."""
    frac   = current / max(total, 1)
    filled = int(bar_width * frac)
    bar    = "#" * filled + "-" * (bar_width - filled)
    return f"[{bar}] {current}/{total} ({frac*100:.1f}%)"


def generate_video(
        audio_path: str,
        output_video: str = "output.mp4",
        fps: int = 60,
        grid_size: int = 150,
):
    """
    Execute the visualization pipeline: audio extraction, terrain generation,
    frame rendering, and video assembly.
    """
    temp_dir = "temp_frames"

    # Resolve unique output filename to avoid overwriting existing files
    original_output = output_video
    base, ext = os.path.splitext(output_video)
    counter = 1
    while os.path.exists(output_video):
        output_video = f"{base}_{counter}{ext}"
        counter += 1

    if output_video != original_output:
        print(f"  Note: '{original_output}' already exists. Output will be saved to '{output_video}'.\n")

    # Create temporary directory for frame sequence
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print("=" * 60)
    print("STEP 1 / 4  -  Audio Analysis")
    print("=" * 60)

    # Extract audio spectrogram, band energies, and beat frames
    spectrogram, band_energies, beat_frames, onset_env, sr = extract_audio_frames(
        audio_path, fps=fps
    )
    num_frames = spectrogram.shape[0]

    print(f"\n  Frames  : {num_frames}")
    print(f"  Duration: {num_frames / fps:.1f}s  at {fps} FPS")

    print("\n" + "=" * 60)
    print("STEP 2 / 4  -  Renderer Initialisation")
    print("=" * 60)

    # Initialize ModernGL 3D terrain renderer
    renderer = TerrainRenderer(grid_size=grid_size)
    print("  ModernGL headless context ready.")

    print("\n" + "=" * 60)
    print("STEP 3 / 4  -  Cinematic Render")
    print("=" * 60)
    print(f"  Grid: {grid_size}x{grid_size}  |  FPS target: {fps}")
    print()

    start_time = time.time()

    # Save rendered frames asynchronously via thread pool
    save_executor = ThreadPoolExecutor(max_workers=workers)

    def _save_image(img: Image.Image, path: str):
        """Save PIL image to disk."""
        img.save(path)

    futures = []  # Tracks asynchronous image write tasks

    # ------------------------------------------------------------------
    # Core In-Memory Stream Loop
    # ------------------------------------------------------------------
    for i in range(num_frames):
        # Extract audio features for the current frame
        frame_spec   = spectrogram[i]
        frame_bands  = {k: float(v[i]) for k, v in band_energies.items()}
        frame_beat   = float(beat_frames[i])

        # Generate terrain mesh on the GPU
        X, Y, Z = create_terrain_frame(
            audio_frame=frame_spec,
            band_energies=frame_bands,
            beat_pulse=frame_beat,
            grid_size=grid_size,
            frame_index=i,
        )

        # Render 3D frame using ModernGL
        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
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

        # Queue image write task
        futures.append(save_executor.submit(_save_image, img, frame_path))

        # Print progress status every 100 frames
        if i % 100 == 0 or i == num_frames - 1:
            elapsed   = time.time() - start_time
            speed     = (i + 1) / elapsed if elapsed > 0 else 0
            eta       = (num_frames - i - 1) / speed if speed > 0 else 0
            bar       = _progress_bar(i + 1, num_frames)
            print(f"  {bar}  |  {speed:.1f} fps  |  ETA {eta:.0f}s")

    # Wait for all background file writes to finish
    print("\n  Waiting for frame writes to complete ...")
    for f in futures:
        f.result()
    save_executor.shutdown(wait=False)

    total_render_time = time.time() - start_time
    print(f"\n  [SUCCESS] Render complete in {total_render_time:.1f}s  "
          f"({num_frames / total_render_time:.1f} fps average)")

    print("\n" + "=" * 60)
    print("STEP 4 / 4  -  Video Assembly (FFmpeg)")
    print("=" * 60)

    # Assemble output MP4 video using FFmpeg
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%06d.png"),
        "-i", audio_path,
        # Specify H.264 video encoding parameters
        "-c:v",     "libx264",
        "-preset",  "fast",
        "-crf",     "18",
        "-pix_fmt", "yuv420p",
        # Specify AAC audio encoding parameters
        "-c:a",     "aac",
        "-b:a",     "320k",
        "-ac",      "2",
        # Truncate output to match the shorter input stream
        "-shortest",
        output_video,
    ]

    try:
        print("  Running FFmpeg ...")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n  [SUCCESS] Created: '{output_video}'  ({size_mb:.1f} MB)")
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] FFmpeg error:\n{e.stderr.decode()}")
    except FileNotFoundError:
        print("\n  [ERROR] FFmpeg not found. Install it and add it to your PATH.")

    # Remove temporary frame directory to free disk space
    shutil.rmtree("temp_frames", ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Music Landscape Visualizer - generates a 3-D terrain video synced to audio."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default="test_song1.mp3",
        help="Path to the input audio file (default: test_song1.mp3)",
    )
    parser.add_argument(
        "-o", "--output",
        default="visual_landscape.mp4",
        help="Output video filename (default: visual_landscape.mp4)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Video frame rate (default: 60)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=150,
        help="Terrain grid resolution N (N x N mesh, default: 150)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size for parallel PNG saving (default: 4)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file '{args.audio}' not found.")
        raise SystemExit(1)

    generate_video(
        audio_path=args.audio,
        output_video=args.output,
        fps=args.fps,
        grid_size=args.grid,
    )
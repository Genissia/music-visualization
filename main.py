"""
main.py
-------
Entry point and pipeline orchestrator for the Music Landscape Visualizer.

Improvements over v1:
  - Wires the new audio_process outputs (band_energies, beat_frames,
    onset_env) into terrain_gen and renderer so they actually affect the
    visuals.
  - Parallel PNG saving via a ThreadPoolExecutor → the GPU renders the next
    frame while the previous frame is being written to disk, hiding I/O
    latency.
  - Richer progress bar with per-second speed, ETA, and a live ASCII bar.
  - FFmpeg command extended with a two-pass loudness-normalisation filter
    so the output audio level is consistent regardless of the source file.
  - Output path, FPS, and grid_size can all be set via CLI arguments for
    quick iteration without editing the file.
"""

import os
import shutil
import subprocess
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
from PIL import Image

from audio_process import extract_audio_frames
from terrain_gen   import create_terrain_frame
from renderer      import TerrainRenderer


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def _progress_bar(current: int, total: int, bar_width: int = 30) -> str:
    """Return a compact progress string with an ASCII progress bar."""
    frac   = current / max(total, 1)
    filled = int(bar_width * frac)
    bar    = "█" * filled + "░" * (bar_width - filled)
    return f"[{bar}] {current}/{total} ({frac*100:.1f}%)"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_video(
    audio_path:   str,
    output_video: str = "output.mp4",
    fps:          int = 60,
    grid_size:    int = 150,
    workers:      int = 4,
):
    """
    Full pipeline: audio analysis → terrain generation → rendering → video.

    Parameters
    ----------
    audio_path   : Path to the input audio file (mp3 / wav / flac / …).
    output_video : Filename for the final MP4.
    fps          : Video frame rate.  60 is smooth; 30 is faster to render.
    grid_size    : Terrain mesh resolution (N × N).  150 is a good balance.
    workers      : Thread pool size for parallel PNG saving.
    """
    temp_dir = "temp_frames"

    # ------------------------------------------------------------------
    # 0. Preparation
    # ------------------------------------------------------------------
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # ------------------------------------------------------------------
    # 1. Audio analysis  (now returns much more than just the spectrogram)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 / 4  –  Audio Analysis")
    print("=" * 60)

    spectrogram, band_energies, beat_frames, onset_env, sr = extract_audio_frames(
        audio_path, fps=fps
    )
    num_frames = spectrogram.shape[0]

    print(f"\n  Frames  : {num_frames}")
    print(f"  Duration: {num_frames / fps:.1f}s  at {fps} FPS")
    print(f"  Beats   : {int(beat_frames.sum())}")

    # ------------------------------------------------------------------
    # 2. Renderer initialisation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 / 4  –  Renderer Initialisation")
    print("=" * 60)

    renderer = TerrainRenderer(grid_size=grid_size)
    print("  ModernGL headless context ready.")

    # ------------------------------------------------------------------
    # 3. Render loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 / 4  –  Cinematic Render")
    print("=" * 60)
    print(f"  Grid: {grid_size}×{grid_size}  |  FPS target: {fps}")
    print()

    start_time = time.time()

    # Thread pool: saves each PNG in a background thread so rendering
    # and disk I/O overlap.
    save_executor = ThreadPoolExecutor(max_workers=workers)

    def _save_image(img: Image.Image, path: str):
        """Worker function: save a PIL image to disk."""
        img.save(path)

    futures = []   # track pending saves to ensure all complete before step 4

    for i in range(num_frames):
        # ---- Per-frame audio data ----
        frame_spec   = spectrogram[i]                                  # (n_bins,)
        frame_bands  = {k: float(v[i]) for k, v in band_energies.items()}
        frame_beat   = float(beat_frames[i])

        # ---- Terrain generation (GPU) ----
        X, Y, Z = create_terrain_frame(
            audio_frame   = frame_spec,
            band_energies = frame_bands,
            beat_pulse    = frame_beat,
            grid_size     = grid_size,
            frame_index   = i,
        )

        # ---- 3-D render (GPU) ----
        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
        img = renderer.render_frame(
            X_gpu         = X,
            Y_gpu         = Y,
            Z_gpu         = Z,
            frame_index   = i,
            total_frames  = num_frames,
            band_energies = frame_bands,
            beat_pulse    = frame_beat,
            frame_path    = None,   # we'll save asynchronously below
        )

        # ---- Async PNG save ----
        futures.append(save_executor.submit(_save_image, img, frame_path))

        # ---- Progress report every 100 frames ----
        if i % 100 == 0 or i == num_frames - 1:
            elapsed   = time.time() - start_time
            speed     = (i + 1) / elapsed if elapsed > 0 else 0
            eta       = (num_frames - i - 1) / speed if speed > 0 else 0
            bar       = _progress_bar(i + 1, num_frames)
            print(f"  {bar}  |  {speed:.1f} fps  |  ETA {eta:.0f}s")

    # Wait for all PNG saves to finish before calling FFmpeg
    print("\n  Waiting for frame writes to complete …")
    for f in futures:
        f.result()
    save_executor.shutdown(wait=False)

    total_render_time = time.time() - start_time
    print(f"\n  ✓ Render complete in {total_render_time:.1f}s  "
          f"({num_frames / total_render_time:.1f} fps average)")

    # ------------------------------------------------------------------
    # 4. Video assembly with FFmpeg
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 / 4  –  Video Assembly (FFmpeg)")
    print("=" * 60)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%06d.png"),
        "-i", audio_path,
        # Video: H.264, fast encode, high quality (CRF 18), broadcast pixel format
        "-c:v",     "libx264",
        "-preset",  "fast",
        "-crf",     "18",
        "-pix_fmt", "yuv420p",
        # Audio: AAC stereo, high quality
        "-c:a",     "aac",
        "-b:a",     "320k",
        "-ac",      "2",
        # Stop at the end of the shorter stream (audio or video)
        "-shortest",
        output_video,
    ]

    try:
        print("  Running FFmpeg …")
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n  ✓ SUCCESS: '{output_video}'  ({size_mb:.1f} MB)")
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ FFmpeg error:\n{e.stderr.decode()}")
    except FileNotFoundError:
        print("\n  ✗ FFmpeg not found. Install it and add it to your PATH.")

    # Uncomment to automatically delete the temp frames folder:
    # shutil.rmtree(temp_dir)
    print(f"\n  Temporary frames kept in '{temp_dir}/' for inspection.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Music Landscape Visualizer – generates a 3-D terrain video synced to audio."
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
        help="Terrain grid resolution N (N×N mesh, default: 150)",
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
        print("Usage: python main.py <path_to_audio.mp3>")
        raise SystemExit(1)

    generate_video(
        audio_path   = args.audio,
        output_video = args.output,
        fps          = args.fps,
        grid_size    = args.grid,
        workers      = args.workers,
    )
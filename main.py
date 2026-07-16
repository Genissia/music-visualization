import os
import subprocess
import time
import argparse

import numpy as np

from audio_process import extract_audio_frames
from terrain_gen import create_terrain_frame
from renderer import TerrainRenderer


def _progress_bar(current: int, total: int, bar_width: int = 30) -> str:
    frac = current / max(total, 1)
    filled = int(bar_width * frac)
    bar = "#" * filled + "-" * (bar_width - filled)
    return f"[{bar}] {current}/{total} ({frac * 100:.1f}%)"


def generate_video(
    audio_path: str,
    output_video: str = "output.mp4",
    fps: int = 60,
    grid_size: int = 150,
    n_height: int = 40,
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
    # 2. Renderer
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 / 3  –  Renderer Initialisation")
    print("=" * 60)

    # n_height = vertical resolution of each canyon wall
    renderer = TerrainRenderer(grid_size=grid_size, n_height=n_height)
    print(f"  ModernGL ready. Corridor mesh: {grid_size} deep x {n_height} tall, 2 walls.")

    # ------------------------------------------------------------------
    # 3. FFmpeg pipe
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 / 3  –  Cinematic Stream Render")
    print("=" * 60)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{renderer.RENDER_WIDTH}x{renderer.RENDER_HEIGHT}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-i", audio_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "320k",
        "-ac", "2",
        "-shortest",
        output_video,
    ]

    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("  [OK] FFmpeg pipe established.")
    except FileNotFoundError:
        print("\n  [ERROR] FFmpeg not found. Install it and add it to your PATH.")
        return

    start_time = time.time()

    for i in range(num_frames):
        frame_spec  = spectrogram[i]
        frame_bands = {k: float(v[i]) for k, v in band_energies.items()}
        frame_beat  = float(beat_frames[i])

        # --- NEW: returns TWO wall meshes, not X/Y/Z heightmap ---
        left_wall, right_wall, floor = create_terrain_frame(
            audio_frame   = frame_spec,
            band_energies = frame_bands,
            beat_pulse    = frame_beat,
            grid_size     = grid_size,
            n_height      = n_height,
            frame_index   = i,
        )

        img = renderer.render_frame(
            left_wall     = left_wall,
            right_wall    = right_wall,
            floor         = floor,
            frame_index   = i,
            total_frames  = num_frames,
            band_energies = frame_bands,
            beat_pulse    = frame_beat,
            frame_path    = None,
        )

        ffmpeg_process.stdin.write(img.tobytes())

        if i % 10 == 0 or i == num_frames - 1:
            elapsed = time.time() - start_time
            speed   = (i + 1) / elapsed if elapsed > 0 else 0
            eta     = (num_frames - i - 1) / speed if speed > 0 else 0
            bar     = _progress_bar(i + 1, num_frames)
            print(f"  {bar}  |  {speed:.1f} fps  |  ETA {eta:.0f}s", end="\r")

    print("\n\n  Closing streams and finalising video...")
    ffmpeg_process.stdin.close()
    ffmpeg_process.communicate()

    total = time.time() - start_time

    if ffmpeg_process.returncode == 0:
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n  [OK] SUCCESS: '{output_video}' ({size_mb:.1f} MB)")
        print(f"  Render time: {total:.1f}s ({num_frames / total:.1f} fps average)")
    else:
        print("\n  [ERROR] FFmpeg failed. Check the audio file format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Slot-Canyon Visualizer")
    parser.add_argument("audio", nargs="?", default="acdc_Thunderstruck-[AudioTrimmer.com].mp3")
    parser.add_argument("-o", "--output", default="visual_landscape1001.mp4")
    parser.add_argument("--fps",    type=int, default=60)
    parser.add_argument("--grid",   type=int, default=150, help="depth resolution")
    parser.add_argument("--height", type=int, default=40,  help="wall vertical resolution")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: audio file '{args.audio}' not found.")
        raise SystemExit(1)

    generate_video(
        audio_path   = args.audio,
        output_video = args.output,
        fps          = args.fps,
        grid_size    = args.grid,
        n_height     = args.height,
    )
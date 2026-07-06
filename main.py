import os
import subprocess
import time
import argparse

import numpy as np

from audio_process import extract_audio_frames
from terrain_gen import create_terrain_frame
from renderer import TerrainRenderer


def _progress_bar(current: int, total: int, bar_width: int = 30) -> str:
    """Generates a clean, visual text progress bar for the console."""
    frac = current / max(total, 1)
    filled = int(bar_width * frac)
    bar = "█" * filled + "░" * (bar_width - filled)
    return f"[{bar}] {current}/{total} ({frac * 100:.1f}%)"


def generate_video(
        audio_path: str,
        output_video: str = "output.mp4",
        fps: int = 60,
        grid_size: int = 150,
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

    # Configure FFmpeg to accept raw byte buffers via standard input (stdin)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",  # Input format is uncompressed raw pixels
        "-pix_fmt", "rgb24",  # 3 channels (RGB), 8 bits per channel
        "-s", f"{renderer.RENDER_WIDTH}x{renderer.RENDER_HEIGHT}",
        "-framerate", str(fps),  # Video frame rate matches audio sync
        "-i", "pipe:0",  # Read video frames directly from Python RAM
        "-i", audio_path,  # Second input source is our audio file
        "-c:v", "libx264",  # H.264 video codec
        "-preset", "fast",
        "-crf", "18",  # High-quality visually lossless ceiling
        "-pix_fmt", "yuv420p",  # Ensures max compatibility with media players
        "-c:a", "aac",  # Compress audio stream to AAC
        "-b:a", "320k",  # High-fidelity audio bitrate
        "-ac", "2",  # Stereo channel output
        "-shortest",  # Terminate clip when the shorter stream (audio) ends
        output_video,
    ]

    try:
        # Launch FFmpeg. stderr is directed to DEVNULL to avoid OS pipe deadlock buffer limitations.
        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("  ✓ FFmpeg process pipe established successfully.")
    except FileNotFoundError:
        print("\n  ✗ FFmpeg not found. Install it and add it to your PATH.")
        return

    start_time = time.time()

    # ------------------------------------------------------------------
    # Core In-Memory Stream Loop
    # ------------------------------------------------------------------
    for i in range(num_frames):
        frame_spec = spectrogram[i]
        frame_bands = {k: float(v[i]) for k, v in band_energies.items()}
        frame_beat = float(beat_frames[i])

        # Generate terrain coordinates
        X, Y, Z = create_terrain_frame(
            audio_frame=frame_spec,
            band_energies=frame_bands,
            beat_pulse=frame_beat,
            grid_size=grid_size,
            frame_index=i,
        )

        # Render frame to a PIL Image object
        img = renderer.render_frame(
            X_gpu=X,
            Y_gpu=Y,
            Z_gpu=Z,
            frame_index=i,
            total_frames=num_frames,
            band_energies=frame_bands,
            beat_pulse=frame_beat,
            frame_path=None,
        )

        # Convert the PIL image structure into raw, linear RGB bytes
        raw_rgb_bytes = img.tobytes()

        # Inject the frame bytes straight into the background FFmpeg stream
        ffmpeg_process.stdin.write(raw_rgb_bytes)

        # Console Progress Tracking
        if i % 10 == 0 or i == num_frames - 1:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_frames - i - 1) / speed if speed > 0 else 0
            bar = _progress_bar(i + 1, num_frames)
            print(f"  {bar}  |  {speed:.1f} fps  |  ETA {eta:.0f}s", end="\r")

    # Clean close: Inform FFmpeg that no more video bytes are coming
    print("\n\n  Closing memory streams and rendering final video file...")
    ffmpeg_process.stdin.close()

    # Wait for the background compiler to finish containerizing the video and audio tracks
    ffmpeg_process.communicate()

    total_render_time = time.time() - start_time

    if ffmpeg_process.returncode == 0:
        size_mb = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n  ✓ SUCCESS: '{output_video}' ({size_mb:.1f} MB)")
        print(f"  Total render time: {total_render_time:.1f}s ({num_frames / total_render_time:.1f} fps average)")
    else:
        print(f"\n  ✗ FFmpeg compilation failed. Verify your source audio file formats or encoder path variables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Landscape Visualizer")
    parser.add_argument("audio", nargs="?", default="acdc_Thunderstruck.mp3")
    parser.add_argument("-o", "--output", default="visual_landscape.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--grid", type=int, default=150)
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
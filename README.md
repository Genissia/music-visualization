# Music Landscape Visualizer

A Python-based tool that generates a 3D landscape visualization of an audio file. The terrain height is dynamically modulated by the audio's frequency spectrum, creating a "dancing landscape" synced to the music.

## Features
- **Audio Analysis:** Uses `librosa` to perform Short-Time Fourier Transform (STFT) for frequency extraction.
- **GPU-Accelerated Terrain:** Uses `CuPy` and `NumPy` for high-performance 3D mesh generation.
- **Headless Rendering:** Uses `ModernGL` for efficient, GPU-based 3D rendering without requiring a window.
- **Video Assembly:** Automatically combines rendered frames and audio into a final `.mp4` video using FFmpeg.

## Prerequisites

### 1. Python 3.8+
Ensure you have Python installed.

### 2. FFmpeg
FFmpeg is required to assemble the video.
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin` folder to your system's `PATH`.
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

### 3. NVIDIA GPU & CUDA (Recommended)
This project uses `CuPy` for GPU-accelerated calculations. You should have an NVIDIA GPU and the corresponding CUDA Toolkit installed. 

If you do not have an NVIDIA GPU, you may need to modify `terrain_gen.py` to use `NumPy` instead of `CuPy` (though performance will be significantly lower).

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd music-visualization
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `cupy` fails to install, check the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) to install the version matching your CUDA toolkit (e.g., `pip install cupy-cuda12x`).*

## Usage

1. **Prepare your audio file:**
   Place your audio file (e.g., `test_song.mp3`) in the project root directory.

2. **Configure the script:**
   Open `main.py` and ensure the `AUDIO_FILE` variable points to your file:
   ```python
   AUDIO_FILE = "test_song.mp3"
   ```

3. **Run the visualizer:**
   ```bash
   python main.py
   ```

4. **Find your video:**
   Once the process completes, the final video will be saved as `output.mp4`.

## Project Structure
- `main.py`: Main entry point and video assembly logic.
- `audio_process.py`: Handles audio loading and frequency extraction.
- `terrain_gen.py`: Generates the 3D terrain height map using GPU.
- `renderer.py`: ModernGL-based 3D renderer.
- `requirements.txt`: List of Python dependencies.
- `temp_frames/`: Temporary directory for rendered image frames (created during runtime).

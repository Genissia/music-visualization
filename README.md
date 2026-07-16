# Music Corridor Visualizer

A Python-based 3D slot-canyon music visualizer.

## How to Run

A pre-installed Python virtual environment (`rock-env`) is already set up in this directory.

### 1. Run the Visualizer
To run the visualizer with the default audio file:
```bash
rock-env\Scripts\python main.py
```
The final video will be saved in the directory.

### 2. Run with a Custom Song
To run with a custom song and output file name:
```bash
rock-env\Scripts\python main.py <path_to_audio_file> -o <output_video_name.mp4>
```

### 3. Optional Settings
You can customize the generation using these optional flags:
* `--fps`: Set the target video framerate (default: `60`)
* `--grid`: Depth resolution of the corridor (default: `150`)
* `--height`: Wall vertical resolution (default: `40`)

Example:
```bash
rock-env\Scripts\python main.py my_song.mp3 -o output.mp4 --fps 60 --grid 150
```

# New Version: Cinematic Contrast & Dynamic Camera Path

This update focuses on making the terrain "pop" against the dark background and adds a sophisticated camera path that evolves throughout the song.

## Major Changes

### 1. High-Contrast Terrain Lighting (`renderer.py`)
- **Vibrant Palette:** The colors have been shifted from muted tones to a high-contrast palette:
    - **Valleys:** Deep midnight blue.
    - **Mid-tones:** Vibrant lavender/purple.
    - **Peaks:** Bright golden white.
- **Enhanced Lighting:** Increased the `ambient` light and added a brightness multiplier (`* 1.2`) to the final fragment shader color to ensure the landscape stands out sharply against the pitch-black background.

### 2. Cinematic Camera Transition (`renderer.py`)
- **Dynamic Elevation:** The camera now starts at **eye-level** (low to the ground) to show the scale of the peaks and slowly rises to a **bird's-eye view** (high overhead) to reveal the entire landscape as the song progresses.
- **TWEAKABLE Section:** Look for the `--- CAMERA CONTROLS (EDIT HERE) ---` block in `renderer.py`. I have isolated the `radius`, `angle`, and `cam_y` variables with clear comments so you can easily change the height or rotation speed.

### 3. Synchronization Improvements (`main.py`)
- The `main.py` loop now passes the `total_frames` count to the renderer, allowing the camera transition to be perfectly timed to the length of your song automatically.

## How to Edit Further
Check the **`TWEAKABLE`** tags in the code:
- **`renderer.py` (Vertex Shader):** Adjust `0.035` to change how tall the music mountains are.
- **`renderer.py` (Fragment Shader):** Adjust the `vec3` color values to change the landscape's appearance.
- **`renderer.py` (Camera Math):** Adjust the `cam_y` start/end values to change the camera's path.

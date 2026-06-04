import librosa
import numpy as np
from librosa import feature


class AudioAnalyzer:
    def __init__(self, file_path: str, fps: int = 60):
        """
        Loads the audio track and handles extracting frame-by-frame
        features matched to the video's frames-per-second.
        """
        self.file_path = file_path
        self.fps = fps

        # Load audio (y = audio time series, sr = sampling rate)
        self.y, self.sr = librosa.load(self.file_path)

    def extract_features(self) -> dict:
        """
        Extracts mathematical features per video frame.
        """
        # Calculate hop_length to sync audio analysis frames with video frames
        # if sr=22050 and fps=60, hop_length ≈ 367 samples per frame
        hop_length = int(self.sr / self.fps)

        # STFT for the frequency height map
        stft_matrix = np.abs(librosa.stft(self.y, hop_length=hop_length))
        # Transpose so the matrix is structured as (frame_index, frequency_bin)
        frequency_frames = stft_matrix.T

        # RMS for general energy/amplitude (landscape scaling)
        rms = librosa.feature.rms(y=self.y, hop_length=hop_length)[0]

        # Onset Strength for beat detection for timing lightning flashes
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=hop_length)

        return {
            "frequencies": frequency_frames,
            "rms": rms,
            "onset": onset_env,
            "total_frames": len(frequency_frames)
        }
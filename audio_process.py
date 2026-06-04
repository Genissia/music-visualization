import librosa
import numpy as np

def extract_audio_frames(audio_path, fps=60, n_fft=2048):
    print(f"Loading audio file: {audio_path}...")
    
    # Load the audio file (sr=None preserves the original sample rate)
    y, sr = librosa.load(audio_path, sr=None)
    
    print(f"Original Sample Rate: {sr} Hz")
    
    # The hop_length determines how many audio samples we skip between each frame.
    # To sync perfectly with a 60 FPS video, we divide the sample rate by the FPS.
    hop_length = int(sr / fps)
    
    print("Performing Short-Time Fourier Transform (STFT)...")
    
    # Perform the STFT. This converts our audio from the time domain to the frequency domain.
    stft_data = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # The raw STFT returns complex numbers. We use np.abs() to get the magnitude (amplitude/volume).
    spectrogram = np.abs(stft_data)
    
    # By default, rows are frequencies and columns are time (frames).
    # We transpose (.T) it so that rows = video frames and columns = frequency bins.
    spectrogram_frames = spectrogram.T
    
    print(f"Extraction complete!")
    print(f"Total Video Frames Generated: {spectrogram_frames.shape[0]}")
    print(f"Frequency Bins per Frame: {spectrogram_frames.shape[1]}")
    
    return spectrogram_frames, sr

# --- Test Block ---
if __name__ == "__main__":
    # To test this, place a sample rock song in your folder and rename the string below.
    # Make sure you have a file named 'test_song.wav' or 'test_song.mp3' in your directory!
    
    sample_audio = "test_song.mp3"  # Change this to the name of your actual audio file
    
    try:
        frames, sample_rate = extract_audio_frames(sample_audio, fps=60)
        print("\nSuccess! The audio data is ready for the GPU.")
    except FileNotFoundError:
        print(f"\nError: Could not find '{sample_audio}'. Please add an audio file to your folder to test.")
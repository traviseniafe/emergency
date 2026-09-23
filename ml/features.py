# Turn an audio file into a picture (a spectrogram) that a neural network can learn from.

# Every clip, whatever its original length or sample rate, ends up as a fixed-size
# array: SAMPLE_RATE and DURATION_SECONDS decide its shape, so every clip produces
# an array of the exact same dimensions, which a neural network requires.

import librosa
import numpy as np

SAMPLE_RATE = 16_000       # samples per second we standardise every clip to
DURATION_SECONDS = 4.0     # every clip is cropped or padded to this length
N_MELS = 64                # number of mel frequency bands (rows in the output)

SAMPLES_PER_CLIP = int(SAMPLE_RATE * DURATION_SECONDS)

# Load an audio file as a single-channel (mono) waveform at SAMPLE_RATE.
# librosa.load handles the conversion for us: it reads whatever the file's
# original sample rate and channel count are, and resamples/mixes it down.
    
def load_audio(path) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return waveform

# Make every waveform exactly SAMPLES_PER_CLIP samples long.
# Longer clips (SESA has some up to 33 seconds) are cropped to the first
# DURATION_SECONDS. Shorter clips are padded with silence (zeros) at the end.

def fix_length(waveform: np.ndarray) -> np.ndarray:
    if len(waveform) > SAMPLES_PER_CLIP:
        return waveform[:SAMPLES_PER_CLIP]
    if len(waveform) < SAMPLES_PER_CLIP:
        padding = SAMPLES_PER_CLIP - len(waveform)
        return np.pad(waveform, (0, padding))
    return waveform

# Convert a waveform into a log-scaled mel spectrogram.
# The result is a 2D array: N_MELS rows (frequency bands, low to high) by
# a number of time steps (left to right). Think of it as a grayscale image
# of the sound, which is exactly how a CNN will treat it.

def to_log_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS)
    # Loudness is naturally logarithmic (that's what decibels are), so we
    # convert to a log (decibel) scale, which is what makes quiet sounds
    # like a distant siren visible next to loud ones like a gunshot.
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

# The full pipeline: file on disk -> fixed-size log-mel spectrogram.
def audio_file_to_features(path) -> np.ndarray:
    waveform = load_audio(path)
    waveform = fix_length(waveform)
    return to_log_mel_spectrogram(waveform)
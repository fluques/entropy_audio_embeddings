import numpy as np
from .entropy import get_bark_bands_entropy
import librosa

def multiband_entropy(y, sr=44100, n_fft=2048, window_size=2048, hop_length=2048):
    """
    Computes the multiband entropy for an audio signal.

    The function segments the audio signal into frames, computes the STFT for each
    frame, and then calculates the Gaussian entropy for 24 critical (Bark) bands.

    Args:
        y (np.ndarray): The input audio time series.
        sr (int): The sampling rate of the audio signal.
        n_fft (int): The FFT window size.
        window_size (int): The size of the analysis window for framing.
        hop_length (int): The hop length between consecutive frames.

    Returns:
        np.ndarray: A 2D array where each row is the entropy vector for a frame
                    and each column corresponds to a Bark band.
    """
    res = []
    frames = librosa.util.frame(y, frame_length=window_size, hop_length=hop_length, axis=0)
    for w_frame in frames:
        h_frames=(w_frame*np.hanning(window_size))
        S = librosa.stft(h_frames,n_fft=n_fft, center=False)
        res.append(get_bark_bands_entropy(S, sr, n_fft))
    return np.array(res)

def get_embeddings(file_path,sr=44100, n_fft=2048, window_size=2048, hop_length=512, shingle_size=10, shingle_step=2):
    """
    Generates embeddings for an audio file in wav format using multiband entropy.

    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate. Default is 44100.
        n_fft (int): FFT window size. Default is 2048.
        window_size (int): Window size for framing. Default is 2048.
        hop_length (int): Hop length for framing. Default is 512.
        shingle_size (int): Size of the shingle. Default is 10.
        shingle_step (int): Step size for the shingle. Default is 2.

    Returns:
        numpy.ndarray: A 2D array of embeddings.
    """
    y, sr = librosa.load(file_path, sr=sr) 
    y =librosa.util.normalize(y)
    audio_segments = []
    y=np.trim_zeros(y, 'f')
    embeddings = []
    chunk_size = sr * shingle_size
    chunk_hoop = sr * shingle_step
    for i in range(0,len(y),chunk_hoop):
        chunk = y[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
        entropies = multiband_entropy(chunk, sr=sr, n_fft=n_fft, window_size=window_size, hop_length=hop_length)
        entropies = np.mean(entropies,axis=0)
        entropies = normalize(entropies)
        embeddings.append(entropies)
        
    return np.vstack(embeddings)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
            return v
    return v / norm

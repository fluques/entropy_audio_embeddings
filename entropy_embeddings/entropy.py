import librosa
import numpy as np

def gaussian_entropy_2d_complex(complex_array):
    """Calculates the differential entropy of a 2D Gaussian distribution.

    This function treats the real and imaginary parts of the input complex
    numbers as two dimensions of a dataset. It then computes the covariance
    matrix and uses it to calculate the entropy of the corresponding
    multivariate Gaussian distribution.

    Args:
        complex_array (np.ndarray): An array of complex numbers.

    Returns:
        float: The calculated Gaussian entropy. Returns 0 if the covariance
               matrix is singular.
    """
    data = complex_array.flatten()
    real_part = data.real
    imag_part = data.imag
    X = np.vstack((real_part, imag_part)).T
    cov = np.cov(X,rowvar=False)
    if np.linalg.matrix_rank(cov) < cov.shape[0]:
        return 0 # Or handle singularity
    n = cov.shape[0]
    sign, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov)
    entropy = 0.5 * logdet
    return entropy

def get_bark_bands_entropy(frames, sr, n_fft):
    """
    Computes the entropy for 24 Bark scale bands from the given STFT frames.

    This function maps the FFT frequency bins to 24 critical bands (Bark scale)
    and calculates the Gaussian entropy for the complex coefficients within
    each band.

    Args:
        frames (np.ndarray): The STFT complex frames.
        sr (int): The sampling rate of the audio signal.
        n_fft (int): The FFT window size used to compute the STFT.

    Returns:
        np.ndarray: A 1D array of size 24 containing the entropy values for
            each Bark band.
    """
    entropies=np.zeros(24)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    first_freq = freqs >= 0 
    last_freq = freqs > 15500 
    
    first_freq_index = np.argmax(first_freq)
    last_freq_index = np.argmax(last_freq)
    
    bark_bands_hertz = [100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500]
    bark = 0
    n2=n1=first_freq_index;
    
    for i in range(first_freq_index, last_freq_index+1):
        f=freqs[i]
        n2+=1        
        if(bark_bands_hertz[bark]<= f):
            entropies[bark]=gaussian_entropy_2d_complex(frames[n1:n2])
            bark+=1
            n1=n2

    return entropies
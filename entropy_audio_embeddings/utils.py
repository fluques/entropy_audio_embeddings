import numpy as np
from pydub import AudioSegment
from .embeddings import get_embeddings


def multiband_entropy_from_mp3(filename,sr=44100, n_fft=2048, window_size=2048, hop_length=512, shingle_size=10, shingle_step=2):
    wavfile_path =  convert_mp3_to_wav(filename)
    entropies = get_embeddings(wavfile_path,sr=sr, n_fft=n_fft, window_size=window_size, hop_length=hop_length, shingle_size=shingle_size, shingle_step=shingle_step)
    return entropies


def convert_mp3_to_wav(file_path):
    output_file = file_path+".wav"
    print(output_file)
    print("Converting mp3 to wav")
    try:
        sound = AudioSegment.from_mp3(file_path)
        sound.set_frame_rate(44100)
        sound.export(output_file, format="wav")
        print(f"Successfully converted {file_path} to {output_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")
    return output_file
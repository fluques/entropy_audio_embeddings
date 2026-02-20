from entropy_audio_embeddings import multiband_entropy_from_mp3
from pathlib import Path





def embeddings_from_mp3(filename):
    print("Getting embeddings from mp3 file")
    entropies = multiband_entropy_from_mp3(filename, sr=44100, n_fft=2048, window_size=2048, hop_length=2048, shingle_size=10, shingle_step=2)
    return entropies



def main():
    print("Getting embeddings from mp3 file")
    entropies =embeddings_from_mp3(str(Path("resources/test.mp3")))
    print("Results :")
    print(f"Shape of entropies: {entropies.shape}")
    print(f"Entropies : {entropies}")
    print("Done")





if __name__ == "__main__":
    main()




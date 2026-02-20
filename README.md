# Audio entropy embeddings

**entropy_embeddings** provides the tools to get a low dimensional metric space mapping for audio files. The interface gives a set of tools to input an mp3 file and as output a numpy 2d matrix is generated. The embeddings are robust for time shifting and noise evironments also it discrmineates between versions e.g. music recordings. With the low dimension representation task like audio retrieval search by excerpt are solved.


## Installation
librosa>=0.11.0 is required.
pydub>=0.25.1 is required.
numpy>= 2.4.2 is required.

```
$ pip install entropy-embeddings
```

## Usage
```
$ python3 example.py
```

For example:

```
from entropy_embeddings import multiband_entropy_from_mp3
from pathlib import Path

print("Getting embeddings from mp3 file")
entropies = multiband_entropy_from_mp3(filename, sr=44100, n_fft=2048, window_size=2048, hop_length=2048, shingle_size=10, shingle_step=2)
print("Done getting embeddings from mp3 file")
```


## Results
<pre>
------ Entropies ------
numpy array:
shape: 24, t
t = time 
</pre>



## Cite
[1] Camarena-Ibarrola, A., Chávez, E., & Tellez, E. S. (2009, November). Robust radio broadcast monitoring using a multi-band spectral entropy signature. In Iberoamerican Congress on Pattern Recognition (pp. 587-594). Berlin, Heidelberg: Springer Berlin Heidelberg.

[2] Camarena-Ibarrola, A., Luque, F., & Chavez, E. (2017, November). Speaker identification through spectral entropy analysis. In 2017 IEEE international autumn meeting on power, electronics and computing (ROPEC) (pp. 1-6). IEEE.
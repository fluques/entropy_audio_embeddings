import hashlib
import os
import urllib.request

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()



setuptools.setup(
    name="entropy-audio-embeddings", # Replace with your own username
    version="0.1.0",
    author="Fernando Luque",
    author_email="ing.fernando.luqueg@gmail.com",
    description="entropy-audio-embeddings: audio mapping to low dimensional metric space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fluques/entropy_audio_embeddings",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'librosa', 'pydub'],
    python_requires='>=3.10',
)

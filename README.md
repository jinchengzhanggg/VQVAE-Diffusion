# VQVAE-Diffusion
The official implementation of "COMPOSER STYLE-SPECIFIC SYMBOLIC MUSIC GENERATION USING VECTOR QUANTIZED DISCRETE DIFFUSION MODELS"

## Overview
We propose to combine a vector quantized variational autoencoder (VQVAE) and discrete diffusion models for the generation of symbolic music with desired composer styles. The trained VQ-VAE can represent symbolic music as a sequence of indexes that correspond to specific entries in a learned codebook. Subsequently, a discrete diffusion model is used to model the VQ-VAE’s discrete latent space. The diffusion model is trained to generate intermediate music sequences consisting of codebook indexes, which are then decoded to symbolic music using the VQ-VAE’s decoder.

## Dataset
Transform the MIDI files from the MAESTRO dataset into pianorolls and then use the trained VQ-VAE to encode the pianorolls into sequences of music tokens for training the vector quantized diffusion model.

## Training
Run python Vqvae/train.py first to train the vqvae model. 
Run python VQDiffusion/train.py to train the discrete diffusion model. 

## Inference
To generate music samples with specified composer styles:
```
python VQDiffusion/generate_midi.py
```
## Citation
If this work is helpful for you, please consider citing:

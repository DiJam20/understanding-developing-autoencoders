# Developing Autoencoder

This repository contains the code for my Bachelor's thesis:  
**Effects of Incrementally Increasing Latent Dimensionality in Autoencoders**

## Abstract

The Developing Autoencoder (Dev-AE) is a variant of the standard autoencoder that incrementally adds neurons to its latent space during training, loosely inspired by the increase in dimensionality in neural cortices during development. Comparison of the Dev-AE to conventional autoencoder (AE) models aims to provide an understanding of the effects of the developing training method. The dynamic growth reveals significant findings: neurons introduced at different stages learn disentangled features and capture distinct frequencies; there is a hierarchy of importance, where reconstruction is driven by earlier neurons, while later neurons largely influence classification; and the sparsity observed in the latent layer extends to the hidden layers. These results demonstrate that the small, biologically motivated change leads to disentangled, sparse representations, greatly boosting model interpretability.

## Requirements

Python 3.10.12
PyTorch 2.1.1

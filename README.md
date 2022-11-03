# Deformable Temporal Convolutional Networks (DTCN)
This repository  provides training and evalution scripts for the DTCN speech separation model described in the paper "Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation".

As baseline TCN schema is also provided along with tools for estimating computational efficiency.

This recipe is a fork of the WHAMandWHAMR recipe in the SpeechBrain library (required, see below). For more help and information on any SpeechBrain related issues:
 * https://speechbrain.github.io/
 * https://github.com/speechbrain/speechbrain

# Data and models
Data:
 * WHAMR
 * WSJ0-2Mix

Models:
 * Deformable Temporal Convolutional Networks
 * Temporal Convolutional Networks (Conv-TasNet without skip connections)

# Running basic script
First install required packages
```
git clone https://github.com/jwr1995/dc1d.git
cd dc1d
pip install -r requirements.txt
```
Then to run basic training of a DTCN model do
```
cd separation
python train.py hparams/deformable/dtcn-whamr.yaml
```

# Paper
Please cite the following paper if you make use of any of this codebase:
```
@misc{ravenscroft2022dtcn,
  doi = {10.48550/ARXIV.2210.15305},
  url = {https://arxiv.org/abs/2210.15305},
  author = {Ravenscroft, William and Goetze, Stefan and Hain, Thomas},
  title = {Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
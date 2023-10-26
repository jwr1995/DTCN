# Deformable Temporal Convolutional Networks (DTCN)

Work on this repository is moving to https://github.com/jwr1995/PubSep

This repository  provides training and evalution scripts for the DTCN speech separation model described in the paper "Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation" - https://arxiv.org/pdf/2210.15305.pdf.

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
First install SRMRpy and remaining required packages
```
git clone https://github.com/jfsantos/SRMRpy.git
cd SRMRpy
python setup.py install

pip install -r requirements.txt
```
Then to run basic training of a DTCN model firstly change the ```data_folder``` hyperparameter in the ```separation/hparams/deformable/dtcn-whamr.yaml``` folder. Then run
```
cd separation
HPARAMS=hparams/deformable/dtcn-whamr.yaml
python train.py $HPARAMS
```
or if you wish to use multi GPU (recommended) run
```
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py $HPARAMS --distributed_launch --distributed_backend='nccl' 

```
replacing ```NGPU``` with the desired number of GPUs to use.
In order to use dynamic mixing you will also need to change the ```base_folder_dm``` and ```rir_path``` hyperparameters, refer to https://github.com/speechbrain/speechbrain/blob/develop/recipes/WHAMandWHAMR/separation/README.md for more info on setting up dynamic mixing in SpeechBrain recipes.

# Known issues
 * The main issue at present is mixed precision training with ```autocast``` enabled. The reason for this is unknown. At present we do not recommend trying to use this functionality.

# Paper
Please cite the following paper if you make use of any of this codebase:
```
@INPROCEEDINGS{dtcn23,
  author={Ravenscroft, William and Goetze, Stefan and Hain, Thomas},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Deformable Temporal Convolutional Networks for Monaural Noisy Reverberant Speech Separation}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095230}}
```

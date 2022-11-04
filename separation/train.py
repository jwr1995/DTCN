#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on WHAM! and WHAMR!
datasets. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-wham.yaml --data_folder /your_path/wham_original
> python train.py hparams/sepformer-whamr.yaml --data_folder /your_path/whamr

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures.

Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
sys.path.append("../")
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_utils import batch_pad_right
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging

# Define training procedure
class Separation(sb.Brain):

    def device_check(self, module, variable):
        if "_device" in module.__dict__:
            return variable.to(module._device)
        else:
            return variable.to('cuda:0') if torch.cuda.is_available() else variable.to('cpu')

    def force_to_self_devices(self):
        if "_device" in self.hparams.Encoder.__dict__:
                self.hparams.Encoder.to(self.hparams.Encoder._device)
        if "_device" in self.hparams.MaskNet.__dict__:
                self.hparams.MaskNet.to(self.hparams.MaskNet._device)
        if "_device" in self.hparams.Decoder.__dict__:
                self.hparams.Decoder.to(self.hparams.Decoder._device)

    def compute_forward(
        self, 
        mix, 
        targets, 
        stage, 
        noise=None, 
        rirs=None, 
        extended_features=None,
        store_intermediates=False,
        profiler=False
        ):
        """Forward computations from the mixture to the separated signals."""
        
        if profiler:
            from thop import profile
            from src.macs import sb_ops_dict

        # Unpack lists and put tensors in the right device
        if not profiler:
            mix, mix_lens = mix
            mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)
            
            # Convert targets to tensor
            targets = torch.cat(
                [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
                dim=-1,
            ).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN and not profiler:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)

                    if "whamr" in self.hparams.data_folder:
                        targets_rev = [
                            self.hparams.reverb(targets[:, :, i], None)
                            for i in range(self.hparams.num_spks)
                        ]
                        try:
                            targets_rev = torch.stack(targets_rev, dim=-1)
                            mix = targets_rev.sum(-1)
                        except:
                            min_len = min([targets_rev[i].shape[-1] for i in range(len(targets_rev))])
                            targets_rev = [targets_rev[i][:,:min_len] for i in range(len(targets_rev))]
                            targets_rev = torch.stack(targets_rev, dim=-1)
                            mix = targets_rev.sum(-1)
                    else:
                        mix = targets.sum(-1)

                    if "wham" in self.hparams.data_folder:
                        noise = noise.to(self.device)
                        len_noise = noise.shape[1]
                        len_mix = mix.shape[1]
                        min_len = min(len_noise, len_mix)

                        # add the noise
                        mix = mix[:, :min_len] + noise[:, :min_len]
                    else:
                        min_len = mix.shape[1]

                        mix = mix[:, :min_len]

                    # fix the length of targets also
                    targets = targets[:, :min_len, :]

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        mix = mix.squeeze(-1)

        if self.hparams.Encoder == None:
            return mix.unsqueeze(-1), targets

        mix = self.device_check(self.hparams.Encoder, mix)

        mix_w = self.hparams.Encoder(mix)

        if not self.hparams.MaskNet == None:
            mix_w = self.device_check(self.hparams.MaskNet, mix_w)

        if store_intermediates:
            self._intermediates = {}
            self._intermediates["encoded_mix"] = mix_w.detach().cpu().numpy()

        if profiler==True:
            prof_macs = {}
            prof_params = {}
            key="encoder"
            print(mix.shape)
            prof_macs[key], prof_params[key] = profile(
                self.hparams.Encoder, 
                inputs=(mix, ), 
                custom_ops=sb_ops_dict,
                verbose=False,
                report_missing=True
                )

        if not self.hparams.MaskNet == None:
            mix_w = self.device_check(self.hparams.MaskNet, mix_w)

            if not extended_features == None:
                B, N, K = mix_w.shape
                B, num_extra_feats = extended_features.shape
                feat_tensor = extended_features.repeat((K,1)).reshape(B,num_extra_feats,K)            
                try:
                    est_mask = self.hparams.MaskNet(mix_w, feat_tensor, intermediates=store_intermediates)
                except:
                    est_mask = self.hparams.MaskNet(mix_w)
            else:
                est_mask = self.hparams.MaskNet(mix_w)
            # print(est_mask)
            if store_intermediates:
                if isinstance(est_mask,tuple):
                    self._intermediates["tcn"] = est_mask[1]
                    est_mask = est_mask[0]
                self._intermediates["est_masks"] = est_mask.detach().cpu().numpy()

        if profiler==True:
            key="masknet"
            print(mix_w.shape)
            prof_macs[key], prof_params[key] = profile(
                self.hparams.MaskNet, 
                inputs=(mix_w, ), 
                custom_ops=sb_ops_dict,
                verbose=False,
                report_missing=True
                )

        if not self.hparams.Decoder == None:
            mix_w = self.device_check(self.hparams.Decoder, mix_w)
            est_mask = self.device_check(self.hparams.Decoder, est_mask)
               
            mix_w = torch.stack([mix_w] * self.hparams.num_spks)
            sep_h = mix_w * est_mask

            if store_intermediates:
                self._intermediates["masked_mixtures"] = sep_h.detach().cpu().numpy()
            
            if profiler==True:
                key="decoder"
                prof_macs[key], prof_params[key] = profile(
                    self.hparams.Decoder, 
                    inputs=(sep_h[0], ), 
                    custom_ops=sb_ops_dict,
                    verbose=False,
                    report_missing=True
                    )
                return prof_macs, prof_params
            # Decoding
            est_source = torch.cat(
                [
                    self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                    for i in range(self.hparams.num_spks)
                ],
                dim=-1,
            )

            if store_intermediates:
                    self._intermediates["masked_mixtures"] = sep_h.detach().cpu().numpy()

            # T changed after conv1d in encoder, fix it here
            T_origin = mix.size(1)
            T_est = est_source.size(1)
            if T_origin > T_est:
                est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
            else:
                est_source = est_source[:, :T_origin, :]
            if profiler:
                return prof_macs, prof_params
            return (est_source.to('cuda:0'), targets.to('cuda:0')) if torch.cuda.is_available() else (est_source.to('cpu'), targets.to('cpu'))
        
    def get_intermediates(self):
        return self._intermediates
    
    def get_intermediate(self,key):
        return self._intermediates[key]

    def match_len(self, tensors, target_length):
        sigs = torch.cat(
            [
                (
                    torch.cat(
                    [
                        t.reshape(1,-1),
                        torch.zeros((1,target_length-t.shape[-1])).to(self.device)
                    ],
                    dim=-1
                    )
                    if t.shape[-1] < target_length
                    else t[:target_length].reshape(1,-1)

                ) for i, t in enumerate(tensors)
            ],
            dim=0
        )

        return sigs
    def compute_objectives(self, predictions, targets):
        """Computes the si-snr loss"""
        if self.hparams.loss.func == sb.nnet.losses.cal_si_snr:
            return self.hparams.loss(torch.moveaxis(targets,1,0), torch.moveaxis(predictions,1,0))
        else:
            return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        
        if not "encode_rirs" in self.hparams.__dict__.keys():
            self.hparams.encode_rirs = False # sanity check for old hparams 
        if not "mask_rirs" in self.hparams.__dict__.keys():
            self.hparams.mask_rirs = False # sanity check for old hparams 
        if not "sad" in self.hparams.__dict__.keys():
            self.hparams.sad = False

        mixture = batch.mix_sig
        targets = [batch.s1_sig]
        if self.hparams.encode_rirs: 
            rirs = [batch.s1_rir_sig]
        else:
            rirs = None

        if self.hparams.num_spks == 2:
            targets.append(batch.s2_sig)
            if self.hparams.encode_rirs:  
                rirs.append(batch.s2_rir_sig)
        
        if "wham" in self.hparams.data_folder:
            noise = batch.noise_sig[0]

        extra_feats = None # required for unimplemented functionality

        if self.hparams.auto_mix_prec:
            with autocast():
                predictions, targets = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN, noise, extended_features=extra_feats, rirs=rirs
                )
                loss = self.compute_objectives(predictions, targets)
                og_loss = loss
                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()
           
            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                logger.info("\nLoss: "+str(loss))
                raise RuntimeError("Bad loss")
                loss.data = torch.tensor(0).to(self.device)
        else:
            if "wham" in self.hparams.data_folder:
                predictions, targets = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN, noise, extended_features=extra_feats, rirs=rirs
                )
            else:
                predictions, targets = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN, extended_features=extra_feats, rirs=rirs
                )
            loss = self.compute_objectives(predictions, targets)
            og_loss = loss
            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()
           
            if (
                loss.mean() < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm, error_if_nonfinite=True
                    )
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                # logger.info("\nLoss: "+str(og_loss))
                # logger.info("\nPredictions: "+str(predictions))
                # raise RuntimeError("Bad loss")
                loss.data = torch.tensor(0).to(self.device)
            
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""

        if not "encode_rirs" in self.hparams.__dict__.keys():
            self.hparams.encode_rirs = False # sanity check for old hparams 
        if not "mask_rirs" in self.hparams.__dict__.keys():
            self.hparams.mask_rirs = False # sanity check for old hparams 
        if not "sad" in self.hparams.__dict__.keys():
            self.hparams.sad = False

        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig]
        if self.hparams.encode_rirs: 
            rirs = [batch.s1_rir_sig]
        else:
            rirs = None

        if self.hparams.num_spks == 2:
            targets.append(batch.s2_sig)
            if self.hparams.encode_rirs:  
                rirs.append(batch.s2_rir_sig)

        
        extra_feats = None

        basenames = batch.basename

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage, extended_features=extra_feats, rirs=rirs)
            loss = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions, basenames)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions, basenames)

        return loss.detach()

    def update_average(self, loss, avg_loss):
            """Update running average of the loss.

            Arguments
            ---------
            loss : torch.tensor
                detached loss, a single float value.
            avg_loss : float
                current running average.

            Returns
            -------
            avg_loss : float
                The average loss.
            """
            loss = torch.Tensor.float(loss).mean()
            if torch.isfinite(loss):
                avg_loss -= avg_loss / self.step
                avg_loss += float(loss) / self.step
            return avg_loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length withing the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data, non_standard=False):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources
        from src.measures import PESQ, STOI, SRMR

        pesq_measure = PESQ(self.hparams.sample_rate)
        stoi_measure = STOI(self.hparams.sample_rate)
        estoi_measure = STOI(self.hparams.sample_rate,extended=True)
        srmr_measure = SRMR(self.hparams.sample_rate)

        # Create folders where to store audio
        if non_standard:
            save_file = os.path.join(self.hparams.output_folder, "ext_test_results.csv")
        else:
            save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        all_pesqs = []
        all_pesqs_i = []
        all_stois = []
        all_stois_i = []
        all_estois = []
        all_estois_i = []
        all_srmrs = []
        all_srmrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i","pesq",
            "pesq_i","stoi","stoi_i","estoi","estoi_i","srmr","srmr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **(self.hparams.test_dataloader_opts if "test_dataloader_opts" 
            in list(self.hparams.__dict__) else self.hparams.dataloader_opts)
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()
                        
            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    
                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig]
                    if self.hparams.encode_rirs: 
                        rirs = [batch.s1_rir_sig]
                    else:
                        rirs = None

                    extra_feats = None

                    if self.hparams.num_spks > 1:
                        targets.append(batch.s2_sig)
                        if self.hparams.encode_rirs:  
                            rirs.append(batch.s2_rir_sig)

                    if self.hparams.num_spks > 2:
                        targets.append(batch.s3_sig)
                    
                    extra_feats = None

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST, extended_features=extra_feats, rirs=rirs
                        )
                    
                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)
                    
                    #Compute PESQ
                    pesq = pesq_measure.pesq_measure_with_pit(targets,predictions)
                    
                    #Compute STOI
                    stoi = stoi_measure.stoi_measure_with_pit(targets,predictions)
                    
                    #Compute extended STOI
                    estoi = estoi_measure.stoi_measure_with_pit(targets,predictions)

                    #Compute SRMR
                    srmr = srmr_measure.srmr_measure_with_pit(predictions)
                    
                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    #Compute PESQ improvement
                    pesq_baseline = pesq_measure.pesq_measure_with_pit(targets,mixture_signal)
                    pesq_i = pesq - pesq_baseline

                    #Compute STOI improvement
                    stoi_baseline = stoi_measure.stoi_measure_with_pit(targets,mixture_signal)
                    stoi_i = stoi - stoi_baseline

                    #Compute extended STOI improvement
                    estoi_baseline = estoi_measure.stoi_measure_with_pit(targets,mixture_signal)
                    estoi_i = estoi - estoi_baseline

                    #Compute SRMR improvement
                    srmr_baseline = srmr_measure.srmr_measure_with_pit(mixture_signal)
                    srmr_i = srmr - srmr_baseline
                    
                    if self.hparams.num_spks==1:
                        targets = torch.repeat_interleave(targets,2,dim=-1)
                        predictions = torch.repeat_interleave(predictions,2,dim=-1)
                        mixture_signal = torch.repeat_interleave(mixture_signal,2,dim=-1)

                    try:
                        # Compute SDR
                        sdr = np.zeros((targets.shape[0],targets.shape[-1]))
                        for i, (target, prediction) in enumerate(zip(targets,predictions)):
                            sdr_value, _, _, _ = bss_eval_sources(
                                target.t().cpu().numpy(),
                                prediction.t().detach().cpu().numpy(),
                            )
                            sdr[i] = sdr_value

                        sdr_baseline = np.zeros((targets.shape[0],targets.shape[-1]))
                        for i, (target, mix) in enumerate(zip(targets,mixture_signal)):
                            sdr_baseline_value, _, _, _ = bss_eval_sources(
                                target.t().cpu().numpy(),
                                mix.t().detach().cpu().numpy(),
                            )
                            sdr_baseline[i] = sdr_baseline_value
                    except ValueError as e:
                        print(e)
                        print("tragets", targets.shape,
                                "predictions", predictions.shape, 
                                "mixture",mixture_signal.shape,
                                "sdr shape", sdr.shape)
                        return

                    sdr_i = sdr.mean() - sdr_baseline.mean()
                    
                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.mean().item(),
                        "si-snr_i": -sisnr_i.mean().item(),
                        "pesq": pesq.mean().item(),
                        "pesq_i": pesq_i.mean().item(),
                        "stoi": stoi.mean().item(),
                        "stoi_i": stoi_i.mean().item(),
                        "estoi": estoi.mean().item(),
                        "estoi_i": estoi_i.mean().item(),
                        "srmr": srmr.mean().item(),
                        "srmr_i": srmr_i.mean().item(),
                    }
                    
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.mean().item())
                    all_sisnrs_i.append(-sisnr_i.mean().item())
                    all_pesqs.append(pesq.mean().item())
                    all_pesqs_i.append(pesq_i.mean().item())
                    all_stois.append(stoi.mean().item())
                    all_stois_i.append(stoi_i.mean().item())
                    all_estois.append(estoi.mean().item())
                    all_estois_i.append(estoi_i.mean().item())
                    all_srmrs.append(srmr.mean().item())
                    all_srmrs_i.append(srmr_i.mean().item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                    "pesq": np.array(all_pesqs).mean(),
                    "pesq_i": np.array(all_pesqs_i).mean(),
                    "stoi": np.array(all_stois).mean(),
                    "stoi_i": np.array(all_stois_i).mean(),
                    "estoi": np.array(all_estois).mean(),
                    "estoi_i": np.array(all_estois_i).mean(),
                    "srmr": np.array(all_srmrs).mean(),
                    "srmr_i": np.array(all_srmrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean Δ SISNR is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean Δ SDR is {}".format(np.array(all_sdrs_i).mean()))
        logger.info("Mean PESQ is {}".format(np.array(all_pesqs).mean()))
        logger.info("Mean Δ PESQ is {}".format(np.array(all_pesqs_i).mean()))
        logger.info("Mean STOI is {}".format(np.array(all_stois).mean()))
        logger.info("Mean Δ STOI is {}".format(np.array(all_stois_i).mean()))
        logger.info("Mean ESTOI is {}".format(np.array(all_estois).mean()))
        logger.info("Mean Δ ESTOI is {}".format(np.array(all_estois_i).mean()))
        logger.info("Mean SRMR is {}".format(np.array(all_srmrs).mean()))
        logger.info("Mean Δ SRMR is {}".format(np.array(all_srmrs_i).mean()))

        logger.info("Generated .csv row:")
        csv_row = [
            np.array(all_sisnrs).mean(),
            np.array(all_sisnrs_i).mean(),
            np.array(all_sdrs).mean(),
            np.array(all_sdrs_i).mean(),
            np.array(all_pesqs).mean(),
            np.array(all_pesqs_i).mean(),
            np.array(all_stois).mean(),
            np.array(all_stois_i).mean(),
            np.array(all_estois).mean(),
            np.array(all_estois_i).mean(),
            np.array(all_srmrs).mean(),
            np.array(all_srmrs_i).mean()
            ]
        csv_row = [str(entry) for entry in csv_row]
        csv_row = ",".join(csv_row)
        logger.info(csv_row)

    def save_audio(self, snt_id, mixture, targets, predictions, basenames):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "{}_s{}hat.wav".format(basenames[0], ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "{}_s{}.wav".format(basenames[0], ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "{}.wav".format(basenames[0]))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )

    def save_modules(self, path=None):
        pretrained_modules = {}
        if not self.hparams.Encoder == None:
            pretrained_modules["encoder"] = self.hparams.Encoder.state_dict()
        if not self.hparams.Decoder == None:
            pretrained_modules["decoder"] = self.hparams.Decoder.state_dict()
        if not self.hparams.MaskNet == None:
            pretrained_modules["masknet"] = self.hparams.MaskNet.state_dict()
        if path == None:
            torch.save(pretrained_modules,os.path.join(self.hparams.output_folder,"modules.pt"))
        else:
            torch.save(pretrained_modules,path)
    
    def load_modules(
        self, 
        path=None,
        encoder=True,
        masknet=True,
        decoder=True,
        device='cuda'):
        if path == None:
            pretrained_modules=torch.load(os.path.join(self.hparams.output_folder,"modules.pt"), map_location=torch.device(device))
        else:
            pretrained_modules=torch.load(path, map_location=torch.device(device))

        if encoder and ("encoder" in pretrained_modules.keys()):
            self.hparams.Encoder.load_state_dict(pretrained_modules["encoder"]) 
        if decoder and ("decoder" in pretrained_modules.keys()):
            self.hparams.Decoder.load_state_dict(pretrained_modules["decoder"]) 
        if masknet and ("masknet" in pretrained_modules.keys()):
            self.hparams.MaskNet.load_state_dict(pretrained_modules["masknet"]) 

def dataio_prep(hparams, extended=False):
    """Creates data processing pipeline"""

    if not extended:
        # 1. Define datasets
        train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["train_data"],
            replacements={"data_root": hparams["data_folder"]},
        )

        valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["valid_data"],
            replacements={"data_root": hparams["data_folder"]},
        )

        test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["test_data"],
            replacements={"data_root": hparams["data_folder"]},
        )
        datasets = [train_data, valid_data, test_data]
    else:
        ext_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["ext_data"],
            replacements={"data_root": hparams["ext_data_folder"]},
        )
        datasets = [ext_data]

    # 2. Provide audio pipelines

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        return mix_sig

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("basename")
    def file_pipeline_basename(mix_wav):
        return os.path.basename(mix_wav).replace(".wav","")

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig

    if hparams["num_spks"] > 1:
        @sb.utils.data_pipeline.takes("s2_wav")
        @sb.utils.data_pipeline.provides("s2_sig")
        def audio_pipeline_s2(s2_wav):
            s2_sig = sb.dataio.dataio.read_audio(s2_wav)
            return s2_sig
        
    if "encode_rirs" in hparams.keys():
        if hparams["encode_rirs"]:
            @sb.utils.data_pipeline.takes("s1_rir")
            @sb.utils.data_pipeline.provides("s1_rir_sig")
            def audio_pipeline_s1_rir(s1_rir):
                s1_rir_sig = sb.dataio.dataio.read_audio(s1_rir)
                return s1_rir_sig
            if hparams["num_spks"] > 1:
                @sb.utils.data_pipeline.takes("s2_rir")
                @sb.utils.data_pipeline.provides("s2_rir_sig")
                def audio_pipeline_s2_rir(s2_rir):
                    s2_rir_sig = sb.dataio.dataio.read_audio(s2_rir)
                    return s2_rir_sig
    
    if "wham" in hparams["data_folder"]:
        @sb.utils.data_pipeline.takes("noise_wav")
        @sb.utils.data_pipeline.provides("noise_sig")
        def audio_pipeline_noise(noise_wav):
            noise_sig = sb.dataio.dataio.read_audio(noise_wav)
            return noise_sig

        @sb.utils.data_pipeline.takes("t60")
        @sb.utils.data_pipeline.provides("t60_val")
        def data_pipeline_t60(t60):
            t60_val = float(t60) if not t60=='' else None
            return t60_val 
        
        @sb.utils.data_pipeline.takes("room_size")
        @sb.utils.data_pipeline.provides("room_size_val")
        def data_pipeline_room_size(room_size):
            room_size_val = float(room_size) if not room_size=='' else None
            return room_size_val
        
        @sb.utils.data_pipeline.takes("snr")
        @sb.utils.data_pipeline.provides("snr_val")
        def data_pipeline_snr(snr):
            snr_val = float(snr) if not snr=='' else None
            return snr_val
        
        @sb.utils.data_pipeline.takes("room_x")
        @sb.utils.data_pipeline.provides("room_x_val")
        def data_pipeline_room_x(room_x):
            room_x_val = float(room_x) if not room_x=='' else None
            return room_x_val

        @sb.utils.data_pipeline.takes("room_y")
        @sb.utils.data_pipeline.provides("room_y_val")
        def data_pipeline_room_y(room_y):
            room_y_val = float(room_y) if not room_y=='' else None
            return room_y_val
        
        @sb.utils.data_pipeline.takes("room_z")
        @sb.utils.data_pipeline.provides("room_z_val")
        def data_pipeline_room_z(room_z):
            room_z_val = float(room_z) if not room_z=='' else None
            return room_z_val

        @sb.utils.data_pipeline.takes("micL_x")
        @sb.utils.data_pipeline.provides("micL_x_val")
        def data_pipeline_micL_x(micL_x):
            micL_x_val = float(micL_x) if not micL_x=='' else None
            return micL_x_val

        @sb.utils.data_pipeline.takes("micL_y")
        @sb.utils.data_pipeline.provides("micL_y_val")
        def data_pipeline_micL_y(micL_y):
            micL_y_val = float(micL_y) if not micL_y=='' else None
            return micL_y_val

        @sb.utils.data_pipeline.takes("micR_x")
        @sb.utils.data_pipeline.provides("micR_x_val")
        def data_pipeline_micR_x(micR_x):
            micR_x_val = float(micR_x) if not micR_x=='' else None
            return micR_x_val

        @sb.utils.data_pipeline.takes("micR_y")
        @sb.utils.data_pipeline.provides("micR_y_val")
        def data_pipeline_micR_y(micR_y):
            micR_y_val = float(micR_y) if not micR_y=='' else None
            return micR_y_val
        
        @sb.utils.data_pipeline.takes("mic_z")
        @sb.utils.data_pipeline.provides("mic_z_val")
        def data_pipeline_mic_z(mic_z):
            mic_z_val = float(mic_z) if not mic_z=='' else None
            return mic_z_val

        @sb.utils.data_pipeline.takes("s1_x")
        @sb.utils.data_pipeline.provides("s1_x_val")
        def data_pipeline_s1_x(s1_x):
            s1_x_val = float(s1_x) if not s1_x=='' else None
            return s1_x_val
        
        @sb.utils.data_pipeline.takes("s1_y")
        @sb.utils.data_pipeline.provides("s1_y_val")
        def data_pipeline_s1_y(s1_y):
            s1_y_val = float(s1_y) if not s1_y=='' else None
            return s1_y_val

        @sb.utils.data_pipeline.takes("s1_z")
        @sb.utils.data_pipeline.provides("s1_z_val")
        def data_pipeline_s1_z(s1_z):
            s1_z_val = float(s1_z) if not s1_z=='' else None
            return s1_z_val

        @sb.utils.data_pipeline.takes("s2_x")
        @sb.utils.data_pipeline.provides("s2_x_val")
        def data_pipeline_s2_x(s2_x):
            s1_z_val = float(s2_x) if not s2_x=='' else None
            return s1_z_val

        @sb.utils.data_pipeline.takes("s2_y")
        @sb.utils.data_pipeline.provides("s2_y_val")
        def data_pipeline_s2_y(s2_y):
            s2_y_val = float(s2_y) if not s2_y=='' else None
            return s2_y_val

        @sb.utils.data_pipeline.takes("s2_z")
        @sb.utils.data_pipeline.provides("s2_z_val")
        def data_pipeline_s2_z(s2_z):
            s2_z_val = float(s2_z) if not s2_z=='' else None
            return s2_z_val

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, file_pipeline_basename)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    
    if hparams["num_spks"] > 1:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)

    if "wham" in hparams["data_folder"]:
        print("Using the WHAM! noise in the data pipeline")
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noise)

    if "creation_path" in hparams.keys() and "wham_noise_path" in hparams.keys():
        sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_t60)
        sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_room_size)
        sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_snr)
        extra_keys = ["t60_val","room_size_val","snr_val"]
        print("Added extra keys to data pipeline")
        if "meta_dump" in hparams.keys():
            if hparams["meta_dump"]:
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_room_x) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_room_y) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_room_z) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_micL_x) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_micL_y) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_micR_x) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_micR_y) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_mic_z) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_s1_x) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_s1_y)
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_s1_z) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_s2_x) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_s2_y) 
                sb.dataio.dataset.add_dynamic_item(datasets, data_pipeline_s2_z)

                extra_keys = extra_keys + [
                    "room_x_val",
                    "room_y_val",
                    "room_z_val",
                    "micL_x_val",
                    "micL_y_val",
                    "micR_x_val",
                    "micR_y_val",
                    "mic_z_val",
                    "s1_x_val",
                    "s1_y_val",
                    "s1_z_val",
                    "s2_x_val",
                    "s2_y_val",
                    "s2_z_val"
                    ] 
                print("Meta dump added, extra keys:", extra_keys)
        if "encode_rirs" in hparams.keys():
            if hparams["encode_rirs"]:
                sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1_rir)
                extra_keys = extra_keys + ["s1_rir_sig"]
                if hparams["num_spks"] > 1:
                    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2_rir)
                    extra_keys = extra_keys + ["s2_rir_sig"]
    else:
        extra_keys = []

    if hparams["num_spks"] > 1 and "wham" in hparams["data_folder"]:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "basename", "s1_sig", "s2_sig", "noise_sig"]+extra_keys
        )
    elif "wham" in hparams["data_folder"]:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "basename", "s1_sig", "noise_sig"]+extra_keys
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "basename", "s1_sig", "s2_sig"]+extra_keys
        )
    if not extended:
        return train_data, valid_data, test_data
    else:
        return ext_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    run_opts["auto_mix_prec"] = hparams["auto_mix_prec"]

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        print(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )
        sys.exit(1)

    # Data preparation
    from prepare_data import prepare_wham_whamr_csv, prepare_wsjmix_csv

    if "wham" in hparams["data_folder"]:
        prep_kwargs={
                    "datapath": hparams["data_folder"],
                    "savepath": hparams["save_folder"],
                    "skip_prep": hparams["skip_prep"],
                    "fs": hparams["sample_rate"],
                    "mix_folder": hparams["mix_folder"],
                    "mini": hparams["mini"],
                    "num_spks": hparams["num_spks"],
                }

        for opt_key in [ # Add optional parameters
            "alternate_path",
            "version",
            "target_condition",
            "creation_path",
            "wham_noise_path",
            "eval_original",
            "meta_dump",
            "use_rirs"
            ]:
            if opt_key in hparams.keys():
                prep_kwargs[opt_key] = hparams[opt_key]
        
        run_on_main(
                prepare_wham_whamr_csv,
                kwargs=prep_kwargs
            )
    elif "wsj" in hparams["data_folder"]:
        prep_kwargs={
                    "datapath": hparams["data_folder"],
                    "savepath": hparams["save_folder"],
                    "skip_prep": hparams["skip_prep"],
                    "fs": hparams["sample_rate"],
                    "n_spks": hparams["num_spks"],
                    "librimix_addnoise": False
                }
        run_on_main(
                prepare_wsjmix_csv,
                kwargs=prep_kwargs
            )



    # if whamr, and we do speedaugment we need to prepare the csv file
    if "whamr" in hparams["data_folder"] and hparams["use_speedperturb"]:
        from prepare_data import create_whamr_rir_csv
        from meta.create_whamr_rirs import create_rirs

        # If the Room Impulse Responses do not exist, we create them
        if not os.path.exists(hparams["rir_path"]):
            print("ing Room Impulse Responses...")
            run_on_main(
                create_rirs,
                kwargs={
                    "output_dir": hparams["rir_path"],
                    "sr": hparams["sample_rate"],
                },
            )

        run_on_main(
            create_whamr_rir_csv,
            kwargs={
                "datapath": hparams["rir_path"],
                "savepath": hparams["save_folder"],
            },
        )

        hparams["reverb"] = sb.processing.speech_augmentation.AddReverb(
            os.path.join(hparams["save_folder"], "whamr_rirs.csv")
        )

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        # if the base_folder for dm is not processed, preprocess them
        from dynamic_mixing import dynamic_mix_data_prep
        if "processed" not in hparams["base_folder_dm"]:
            from meta.preprocess_dynamic_mixing import (
                resample_folder,
            )

            print("Resampling the base folder (WSJ0 for this dataset)")
            run_on_main(
                resample_folder,
                kwargs={
                    "input_folder": hparams["base_folder_dm"],
                    "output_folder": hparams["base_folder_dm"] + "_processed",
                    "fs": hparams["sample_rate"],
                    "regex": "**/*.wav",
                },
            )
            # adjust the base_folder_dm path
            hparams["base_folder_dm"] = hparams["base_folder_dm"] + "_processed"
        train_data = dynamic_mix_data_prep(hparams)
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        # if not "extended_test" in hparams.keys():
        train_data, valid_data, test_data = dataio_prep(hparams)
        

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    separator.force_to_self_devices()

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["test_dataloader_opts"] if "test_dataloader_opts" in hparams.keys() else hparams["dataloader_opts"],
        )

    separator.save_modules()

    # # # Eval
    if "skip_eval" in hparams.keys():
        if hparams["skip_eval"]:
            pass
        else:
            separator.evaluate(test_data, min_key="si-snr")
            separator.save_results(test_data)
    else:
        separator.evaluate(test_data, min_key="si-snr")
        separator.save_results(test_data)

    
    # implement ext_data
    if "extended_test" in hparams.keys():
        prep_kwargs={
            "datapath": hparams["ext_data_folder"], 
            "savepath": hparams["save_folder"], 
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
            "mini": hparams["mini"], 
            "mix_folder": hparams["ext_mix_folder"],
            "target_condition": hparams["ext_target_condition"],
            "set_types": ["tt"],
            "num_spks": hparams["num_spks"],
            "extended": True,
            "num_spks": hparams["num_spks"],
            "alternate_path": hparams["ext_alternate_path"],
        }

        for opt_key in [ # Add optional parameters
            
            "version",
            "creation_path",
            "wham_noise_path",
            "eval_original",
            "meta_dump",
            "use_rirs"
            ]:
            if opt_key in hparams.keys():
                prep_kwargs[opt_key] = hparams[opt_key]
        
        run_on_main(
            prepare_wham_whamr_csv,
            kwargs=prep_kwargs
        )
        ext_data = dataio_prep(hparams,extended=True)
        # separator.evaluate(ext_data, min_key="si-snr",non_standard=True)
        separator.save_results(ext_data,non_standard=True)
        



    
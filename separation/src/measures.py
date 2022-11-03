from multiprocessing import Pool
from functools import partial
import numpy as np
import torch
from speechbrain.nnet.losses import PitWrapper
from pysepm import stoi
from pesq import pesq
from srmrpy import srmr

def _pesq_call(ref_deg,fs):
    zipped = zip(ref_deg[0],ref_deg[1])
    results = [torch.Tensor([_pesq(r_d,fs)]) for r_d in zipped]
    return torch.Tensor(results)

def _pesq(ref_deg,fs): # reorder arguments
    """
    Input args = ((T), (T)), int
    Output args = int
    """
    ref, deg = ref_deg

    if fs == 8000:
        mode='nb'
    else:
        mode='wb'
    try :
        return pesq(fs,ref,deg,mode=mode)
    except:
        return 1.0

class PESQ():
    def __init__(self, fs):
        self.fs = fs
        self.pesq_with_pitwrapper = PitWrapper(self.cal_pesq_loss)

    def cal_pesq_loss(self, source, estimate):
        """
        Input size = T, B, C
        """
        
        assert  source.shape == estimate.shape

        source_np = source.detach().cpu().numpy()
        estimate_np = estimate.detach().cpu().numpy()
        source_np = np.moveaxis(source_np,0,-1)
        estimate_np = np.moveaxis(estimate_np,0,-1)

        zipped = zip(estimate_np, source_np)

        results = [_pesq_call(ref_deg, fs =self.fs) for ref_deg in zipped]

        results = -torch.stack(results)
        return results.unsqueeze(0)

    def get_pesq_loss_with_pit(self, source, estimate):
        """
        Input shapes = (B, T, C), (B, T, C)
        Output shape = (1), (B, C)
        """
        results, perms = self.pesq_with_pitwrapper(source,estimate)
        return results

    def pesq_measure_with_pit(self, source, estimate):
        """
        Input shapes = (B, T, C), (B, T, C)
        Output shape = (1), (B, C)
        """
        return -self.get_pesq_loss_with_pit(source,estimate)

def _stoi_call(ref_deg,fs,extended=False):
    zipped = zip(ref_deg[0],ref_deg[1])
    results = [torch.Tensor([_stoi(r_d,fs,extended)]) for r_d in zipped]
    return torch.Tensor(results)

def _stoi(ref_deg,fs,extended): # reorder arguments
    """
    Input args = ((T), (T)), int
    Output args = int
    """
    ref, deg = ref_deg
    try:
        return stoi(ref,deg,fs,extended=extended)
    except:
        return float("nan")



class STOI():
    def __init__(self, fs, extended=False):
        self.fs = fs
        self.extended = extended
        self.stoi_with_pitwrapper = PitWrapper(self.cal_stoi_loss)

    def cal_stoi_loss(self, source, estimate):
        """
        Input size = T, B, C
        """        
        assert  source.shape == estimate.shape

        source_np = source.detach().cpu().numpy()
        estimate_np = estimate.detach().cpu().numpy()
        source_np = np.moveaxis(source_np,0,-1)
        estimate_np = np.moveaxis(estimate_np,0,-1)

        zipped = zip(estimate_np, source_np) 
        results = [_stoi_call(ref_deg, fs=self.fs,extended=self.extended) for ref_deg in zipped]
        
        results = -torch.stack(results)
        return results.unsqueeze(0)

    def get_stoi_loss_with_pit(self, source, estimate):
        """
        Input shapes = (B, T, C), (B, T, C)
        Output shape = (1), (B, C)
        """
        results, perms = self.stoi_with_pitwrapper(source,estimate)
        median = torch.nanmedian(results)
        return torch.nan_to_num(results,nan=median) # account for nan values

    def stoi_measure_with_pit(self, source, estimate):
        """
        Input shapes = (B, T, C), (B, T, C)
        Output shape = (1), (B, C)
        """
        return -self.get_stoi_loss_with_pit(source,estimate)

def _srmr_call(deg,fs):
    results = [torch.Tensor([srmr(d,fs)[0]]) for d in deg]
    return torch.Tensor(results)

class SRMR():
    def __init__(self, fs=8000):
        self.fs =fs
        self.srmr_with_pitwrapper = PitWrapper(self.cal_srmr_loss)

    def cal_srmr_loss(self, _UNUSED, estimate):
        estimate_np = estimate.detach().cpu().numpy()
        estimate_np = np.moveaxis(estimate_np,0,-1)
        
        results = [_srmr_call(d, fs=self.fs) for d in estimate_np]

        results = -torch.stack(results)
        
        return results.unsqueeze(0)
    
    def get_srmr_loss_with_pit(self, estimate):
        """
        Input shapes = (B, T, C), (B, T, C)
        Output shape = (1), (B, C)
        """
        results, perms = self.srmr_with_pitwrapper(estimate,estimate)
        median = torch.nanmedian(results)
        return torch.nan_to_num(results,nan=median) # account for nan values

    def srmr_measure_with_pit(self, estimate):
        """
        Input shapes = (B, T, C), (B, T, C)
        Output shape = (1), (B, C)
        """
        return -self.get_srmr_loss_with_pit(estimate)

if __name__ == '__main__':
    import time


    class Timer():
        def __init__(self):
            self.start = time.time()
            self.end = self.start
        
        def tic(self):
            self.start = time.time()
        
        def toc(self):
            self.stop = time.time()
            self.time_taken = self.stop-self.start
            print("Ellapsed time: {}s\n".format(self.time_taken))
    
    timer = Timer()

    B = 2
    C = 2
    T = 32000
    fs = 8000

    x = torch.randn(B, T, C)
    xhat = (x + torch.randn(B, T, C))/2
    
    # PESQ
    print("PESQ")
    pesq_measure = PESQ(fs)
    timer.tic()
    print(pesq_measure.pesq_measure_with_pit(x,xhat))
    timer.toc()

    # STOI
    print("STOI")
    stoi_measure = STOI(fs)
    timer.tic()
    print(stoi_measure.stoi_measure_with_pit(x,xhat))
    timer.toc()

    # ESTOI
    print("ESTOI")
    estoi_measure = STOI(fs,extended=True)
    timer.tic()
    print(estoi_measure.stoi_measure_with_pit(x,xhat))
    timer.toc()

    # SRMR
    print("SRMR")
    srmr_measure = SRMR(fs)
    timer.tic()
    print(srmr_measure.srmr_measure_with_pit(xhat))
    timer.toc()
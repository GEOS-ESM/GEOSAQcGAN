
"""
    Datasets for GEOS CF, and CISESS data. Note that these Datasets read heavily processed data 
    stored in data_dir folders.
    The processing code for this data can be found in the scripts folder.
"""

import pickle
from typing import Literal
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from .dataset import GEOSCFLargeEnsembleDataset

# GEOS CF Dataset
class GEOSCFLargeDataset(GEOSCFLargeEnsembleDataset):
        

    @property
    def time_span(self) -> int:
        """
        Returns the number of unique input output time sequences
        """
        return self.n_timesteps - self.window_size + 1 

    def __getitem__(self, idx: int):
        """Get single data example

        Parameters:
            idx (int): data index

        Returns: tensors of dimensions [Channels, Time, Height, Width]
            X (torch.Tensor): input data (all features, includes features from future timepoints if future_feats=True)
            Y_prior (torch.Tensor): target data for input time sequence
            Y (torch.Tensor): target data for output time sequence
        """

        # get ensemble member, time, vertical level
        member_idx = self.member_idxs[idx // self.n_levels // self.time_span]
        time_idx = (idx // self.n_levels) %  self.time_span
        level = self.levels[idx % self.n_levels]
        member_memmap = np.lib.format.open_memmap(self.split_dir / f"{member_idx}.npy", mode="r", dtype=np.float32)
        time_memmap = np.lib.format.open_memmap(self.split_dir / f"{member_idx}_time.npy", mode="r", dtype=np.float32)

        n_ts = self.window_size
        n_level_feats = len(self.feat_idxs[level])

        # calculate number of channels for input
        if self.use_time:
            n_time_feats = len(time_memmap)
        else:
            n_time_feats = 0
        if self.use_latlon:
            n_ll_feats = 3  # lat/lon (lon is encoded as 2d vector on the unit circle)
        else:
            n_ll_feats = 0
        n_feats = n_level_feats + n_time_feats + n_ll_feats
        n_future_feats = len(self.feat_future_idxs[level])
        n_feats = n_level_feats + n_time_feats + n_ll_feats + n_future_feats

        # populate input data
        print('time_idx',time_idx)
        print('time_idx+self.window_size',time_idx+self.window_size)
        X = np.empty((n_feats, n_ts, member_memmap.shape[-2], member_memmap.shape[-1]), dtype=np.float32)
        X[:n_level_feats, :, :, :] = member_memmap[self.feat_idxs[level], time_idx:time_idx+self.window_size:self.step_size, :, :]    # level feats
        curr_feats = n_level_feats
        if n_time_feats > 0:
            X[curr_feats:curr_feats+n_time_feats, :, :, :] = time_memmap[:, time_idx:time_idx+self.window_size:self.step_size].reshape(n_time_feats,n_ts,1,1)
            curr_feats += n_time_feats
        if n_ll_feats > 0:
            X[curr_feats:curr_feats+1, :, :, :] = self.lat.reshape(1,1,len(self.lat),1) / 90 # lat (-1 to 1 scale)
            X[curr_feats+1:curr_feats+3, :, :, :] = np.stack((np.cos(self.lon * np.pi / 180), np.sin(self.lon * np.pi / 180)), axis=0).reshape(2,1,1,len(self.lon)) # lon
            curr_feats += 3
        if n_future_feats > 0:
            X[curr_feats:curr_feats+n_future_feats, :, :, :] = member_memmap[self.feat_future_idxs[level], time_idx+self.window_size:time_idx+self.window_size*2:self.step_size, :, :]    # future level feats
            curr_feats += n_future_feats

        # get target data from prior (input) timesteps
        n_targets = len(self.target_idxs[level])

        Y_prior = np.empty((n_targets, n_ts, member_memmap.shape[-2], member_memmap.shape[-1]), dtype=np.float32)
        Y_prior[:, :, :, :] = member_memmap[self.target_idxs[level], time_idx:time_idx+self.window_size:self.step_size, :, :]

        # get target data from output timesteps
        Y = np.empty((n_targets, n_ts, member_memmap.shape[-2], member_memmap.shape[-1]), dtype=np.float32)
#        Y[:, :, :, :] = member_memmap[self.target_idxs[level], time_idx+self.window_size:time_idx+self.window_size*2:self.step_size, :, :]

        return torch.tensor(X), torch.tensor(Y_prior), torch.tensor(Y)

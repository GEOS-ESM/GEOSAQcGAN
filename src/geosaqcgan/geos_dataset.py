# Copyright 2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

# Approved for public release; distribution is unlimited.
# This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Agreement No.
# HR00112290032.

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


# GEOS CF Dataset
class GEOSCFLargeDataset(Dataset):
        
    def __init__(self, 
              data_dir: Path, 
              split: Literal["train", "val", "test"], 
              member_idxs: list[int], 
              feat_names: dict[int, list[str]], 
              target_names: dict[int, list[str]], 
              n_frames: int = 8, 
              step_size: int = 1, 
              use_future_feats: bool = True, 
              use_latlon: bool = False, 
              use_time: bool = False, 
              ):
        """
        Initializes Geos CF Ensemble Dataset used to train an AQcGAN model.
        Operates on directories storing numpy files with data for each ensemble member across the same time span
        Each numpy file stores data of shape (C x N x H x W)
        Returns tuples of data of shape (C x T x H x W) where 
        C is the number of channels (different depending on if input or output data)
        N is the number of timesteps available for each ensemble member
        T the temporal length of the data (time sequence length)
        H, W are spatial dimensions for height and width (lat/lon)

        Parameters
            data_dir (Path): Directory storing the data. Contains meta.pkl file and subdirs for train, val, test
            split (str): Which dataset to use. Options are "train", "val", "test"
            member_idxs (list[int]): List of ensemble members the split consists of
            feat_names (dict[int, list[str]]): Dictionary containing mapping of the vertical level to the 
                list of features associated with that vertical level that should be input to the model.
            target_names (dict[int, list[str]]): Dictionary containing mapping of the vertical level to the 
                list of features associated with that vertical level that should be learned by the model.
            n_frames (int): Number of timesteps the model takes as input
            step_size (int): Temporal distance between time frames.
                If 1 then `n_frames` consecutive time frames are returned.
                If 2 then every other frame is returned
                If 3 then every third is returned, etc.
            use_future_feats (bool): If True then non-target input variables at target time sequence 
                (in the future) are used as additional input features to model.
            use_latlon (bool): If True then latitude and longitude are included as input features
            use_time (bool): If True then the time of day and time of year are used as input features
        """
        super(GEOSCFLargeEnsembleDataset, self).__init__()

        # Read in metadata
        with open(data_dir / "meta.pkl", "rb") as f:
            d = pickle.load(f)
            self.normalized_var_names = d["z_vars"]
            self.time_var_names = d["time_vars"]
            self.lat = d['lat']
            self.lon = d['lon']

        with open(data_dir / "norm_stats.pkl", "rb") as f:
            d = pickle.load(f)
            self.n_timesteps = d["n_timesteps"]

        self.data_dir = data_dir
        self.split_dir = data_dir / split
        self.member_idxs = member_idxs
        self.n_members = len(self.member_idxs)

        self.use_future_feats = use_future_feats
        self.setup_feats_targets(feat_names, target_names)

        self.n_frames = n_frames
        self.step_size = step_size

        self.use_latlon = use_latlon
        self.use_time = use_time

    def setup_feats_targets(self, feat_names, target_names):
        """
        Ensures the input and target features to be aligned with each other for each vertical level
        Sets the features retrieved by the dataset to be the oens specified

        Parameters:
            feat_names (dict[int, list[str]]): Dictionary containing mapping of the vertical level to the 
                list of features associated with that vertical level that should be input to the model.
            target_names (dict[int, list[str]]): Dictionary containing mapping of the vertical level to the 
                list of features associated with that vertical level that should be learned by the model.
        """
        self.feat_names = feat_names
        self.target_names = target_names

        # ensure both input feats and targets have same vertical levels
        assert sorted(feat_names.keys()) == sorted(target_names.keys())

        # ensure all listed features exist
        for level in feat_names.keys():
            for feat in feat_names[level] + target_names[level]:
                assert feat in self.normalized_var_names or feat in self.time_var_names, f"{feat} not a valid variable; must be one of {self.normalized_var_names} or {self.time_var_names}."

        # store vertical levels
        self.levels = list(feat_names.keys())

        # get indices in feature arrays for input and target features for each level
        self.feat_idxs = {level: [self.normalized_var_names.index(feat) for feat in feat_names[level]] for level in self.levels} 
        self.target_idxs = {level: [self.normalized_var_names.index(target) for target in target_names[level]] for level in self.levels}

        # get indices for input features for future timesteps (if used)
        self.feat_future_idxs = {level: [] for level in self.levels}
        if self.use_future_feats:
            self.feat_future_idxs = {level: [idx for idx in self.feat_idxs[level] if idx not in self.target_idxs[level]] for level in self.levels}

    @property
    def window_size(self) -> int:
        """
        Returns the number of temporal frames or timesteps encompassed by an input or output sequence
        """
        return self.n_frames*self.step_size

    @property
    def time_span(self) -> int:
        """
        Returns the number of unique input output time sequences
        """
        return self.n_timesteps - (self.window_size*2)

    @property
    def n_levels(self) -> int:
        """
        Returns the number of vertical levels in the dataset
        """
        return len(self.levels)

    def __len__(self):
        """
        Returns the total number of examples in the dataset
        """
        return self.n_members * self.time_span * self.n_levels

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
        Y[:, :, :, :] = member_memmap[self.target_idxs[level], time_idx+self.window_size:time_idx+self.window_size*2:self.step_size, :, :]

        return torch.tensor(X), torch.tensor(Y_prior), torch.tensor(Y)

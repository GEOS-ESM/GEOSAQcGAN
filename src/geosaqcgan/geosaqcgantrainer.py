# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC.
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

########################################
### Script for training AQcGAN Model ###
########################################

from pathlib import Path
from typing import Union, Optional
import yaml

import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

from AQcGAN.utils import GANArchitecture, TrainHyperParams, DiffusionParams, Metrics
from AQcGAN.models import Generator, Discriminator
from AQcGAN.datasets import DifferenceDataset
from .geos_dataset import GEOSCFLargeDataset
from .train_ens_ic import AQcGANTrainer

class GEOSAQcGANFCast(AQcGANTrainer):
    def __init__(self, 
            gan_architecture: GANArchitecture, 
            data_params: dict, 
            train_hyperparams: TrainHyperParams, 
            device: Union[int, str],
            chkpt_dir: Path, 
            diffusion_params: DiffusionParams, 
            chkpt_idx: Optional[int] = None):
        """
        Initializer for AQcGANTrainer Object

        Parameters:
            gan_architecture (GANArchitecture): Specifies architecture params of AQcGAN
            data_params (dict): Specifies parameters to initialize dataloader
            train_hyperparams (TrainHyperParams): Specifies hyperparameters for training AQcGAN
            device (int|str): GPU number to put AQcGAN on (or "cpu")
            chkpt_dir (Path): Directory to store model checkpoints
            diffusion_params (DiffusionParams): Parameters specifying Diffusion GAN settings
            chkpt_idx (int|None): If provided, checkpoint number (epoch) to load and resume training
        """
        self.gan_architecture = gan_architecture
        self.data_params = data_params
        self.train_hyperparams = train_hyperparams

        if isinstance(device, int):
            if not torch.cuda.is_available():
                raise ValueError("CUDA unavailable, but specified a device ID.")
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)

        # set up models
        self.gen = Generator(**gan_architecture.get_gen_kwargs()).to(device)
        self.disc = Discriminator(**gan_architecture.get_disc_kwargs()).to(device)

        # extract dataset class from config
        mapping = {"Large": GEOSCFLargeDataset}
        class_str = data_params.pop("class", "Large")
        dataset_class = mapping[class_str]

        # extract difference method
        diff_method = data_params.pop("difference", "none")

        # store ensemble member indices
        member_idxs = dict()
        for key in ["train_member_idxs", "val_member_idxs", "test_member_idxs"]:
            member_idxs[key] = data_params.pop(key, [])

        member_idxs["val_member_idxs"] = [1] # always 1

        # set up datasets
        self.val_data = DifferenceDataset(dataset_class(**data_params, split="val", member_idxs=member_idxs["val_member_idxs"]), diff_method)

        # set up dataloaders
        self.val_dl = DataLoader(self.val_data, batch_size=train_hyperparams.batch_size,
            shuffle=False, pin_memory=True, pin_memory_device=str(self.device))

        # set up optimizers
        self.gen_optim = Adam(self.gen.parameters(), **train_hyperparams.get_adam_kwargs())
        self.disc_optim = Adam(self.disc.parameters(), **train_hyperparams.get_adam_kwargs())

        # set up logging
        self.val_metrics = Metrics()
        self.train_metrics = Metrics()

        self.n_epochs = 0
        self.chkpt_dir = chkpt_dir

        # load checkpoint
        if chkpt_idx is not None:
            self.load_chkpt(chkpt_idx)

        # setup diffusion parameters
        self.diffusion = diffusion_params


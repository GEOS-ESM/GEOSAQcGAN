
"""
"""
from pathlib import Path
from typing import Literal
import numpy as np

import torch
import torch.nn.functional as F

from ..train_ens_ic import AQcGANTrainer
from AQcGAN.utils import GANArchitecture, TrainHyperParams, DiffusionParams

from .gen_utils import read_yaml_file

def parse_yaml_prediction(yaml_file: Path,
        chkpt_idx: Optional[int] = None,
        device: Optional[int] = None) -> AQcGANTrainer:
    """
    Use a config file to initialize an AQcGANTrainer object 
    and a corresponding dataset.

    Parameters
    ----------
    yaml_file : Path
       Full path to the YAML config file specifying the dataset, 
       AQcGAN architecture, training hyperparameters, and 
       diffusion GAN parameters.
    chkpt_idx : int 
       If given, the index of the checkpoint used to resume the AQcGAN training
    device : int 
       If given, the GPU device number to use for training the AQcGAN

    Returns
    -------
    trainer : AQcGANTrainer 
       AQcGANTrainer object
    """

    config = read_yaml_file(yaml_file)

    # gan architecture
    gan_architecture = GANArchitecture(**config["aqcgan"])

    # data params
    data_params = config["data"]
    config["data"]["data_dir"] = Path(data_params["data_dir"])
     
    # train_hyperparams
    train_hyperparams = TrainHyperParams(**config["train"])
     
    # device
    if not device:
        device = config["device"]
     
    if "diffusion" in config:
        diffusion_params = DiffusionParams(**config["diffusion"])
    else:
        diffusion_params = DiffusionParams(use=False, 
                                           noise_steps=10, 
                                           beta_start=0., beta_end=1.)
     
    trainer = AQcGANTrainer(gan_architecture, data_params, 
                            train_hyperparams, device, 
                            Path(config["chkpt_dir"]), 
                            diffusion_params=diffusion_params, 
                            chkpt_idx=chkpt_idx)  

    return trainer


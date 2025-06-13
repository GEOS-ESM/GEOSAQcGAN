
"""
Helper utility functions for running AQcGAN in with GEOS 
in an single member (non-ensemble) mode
"""
import numpy as np
import torch

from ..train_ens_ic import AQcGANTrainer
from ..inference.utils import collate

from pathlib import Path
from typing import Literal, Optional, Union
import numpy as np

import torch
import torch.nn.functional as F

from ..geosaqcgantrainer import GEOSAQcGANFCast
from AQcGAN.utils import GANArchitecture, TrainHyperParams, DiffusionParams

from .gen_utils import read_yaml_file

def get_predictions(trainer: AQcGANTrainer | GEOSAQcGANFCast, 
                    n_passes: int = 1) -> torch.Tensor:

    """
    Generates mean and spread predictions for an ensemble dataset.

    Parameters:
        trainer (AQcGANTrainer or GEOSAQcGANFcast): Trainer object
        n_passes (int): Number of time sequences the "prediction" is ahead of the input.
                        Used for autoregressive results farther out (e.g. 10 days)

    Returns:
        prediction (torch.Tensor): "Predicted" (or ground truth) for dataset
    """
    prediction = get_pred(trainer, n_passes=n_passes)

    return prediction

def get_pred(aqcgantrainer,
        n_passes: int = 1) -> torch.Tensor:
    """
    Get AQcGAN predictions

    Parameters:
        aqcgantrainer (AQcGANTrainer): AQcGANTrainer object
        n_passes: How many forward passes to predict for

    Returns:
        prediction (torch.Tensor)
        ens_sprad (torch.Tensor)
    """
    dataset = aqcgantrainer.val_data

    # update time_span depending on number of forward passes
    # (latest starting timestep we can get prediction for `n_passes` ahead and have ground
    #  truth to compare to)
    time_span = dataset.time_span - (dataset.n_frames*dataset.step_size)*(n_passes - 1)

    # update step size
    step_size = dataset.step_size*dataset.n_frames

    # only works if number of vertical levels in dataset is 1
    assert dataset.n_levels == 1
    vertical_level = dataset.levels[0]

    predictions = []

    # put generator into evaluation mode
    aqcgantrainer.gen.eval()

    # AQcGAN may take in (e.g.) 10 variables and only output 4
    # To do autoregression need to get ground truth for extra input variables not predicted
    # (these constitute meteorological variables assumed to be known ahead of time)
    # Figure out which channels of input are predicted by AQcGAN model
    n_members = 1
    collate_idxs = [dataset.feat_names[vertical_level].index(target) for target in dataset.target_names[vertical_level]]
    for ts_idx in range(time_span):
        x_pred = None
        for i in range(n_passes):
            # take advantage of indexing order for dataset to get data for all ensemble members
            # at same point in time
            x_batch, prior, _ = zip(*[dataset[m_idx*dataset.time_span+(ts_idx + i*step_size)] for m_idx in range(n_members)])
            x_batch = torch.stack(x_batch, dim=0)
            prior = torch.stack(prior, dim=0)
            # x_batch is input of all (e.g.) 10 variables into AQcGAN
            # prior is subset of x_batch corresponding to predicted variables (at input timestep)

            # collate batch w/ prediction
            # (use model's previous predictions for next autoregressive pass)
            if x_pred is not None:
                x_batch = collate(x_pred, x_batch, collate_idxs)
                prior = x_pred
            x_batch = x_batch.to(aqcgantrainer.device)

            # get model prediction and invert difference to get actual predicted values
            # (not residuals)
            with torch.inference_mode():
                x_pred = aqcgantrainer.gen(x_batch).cpu()
                x_pred = dataset.diff_operator.invert_difference(prior, x_pred)

        # get mean and spread (variance) of predictions across ensemble members at given timestep
        if x_pred is not None:
            predictions.append(x_pred.mean(dim=0))

    return torch.stack(predictions, dim=0)

def inv_pred(prediction: np.ndarray, 
             mu: np.ndarray, 
             sigma: np.ndarray) -> np.ndarray :
    """
    Takes predicted values in normalized
    space and converts them back to their native units

    Parameters:
        prediction (np.ndarray): Predicted ensemble means
        mu (np.ndarray): Mean used to normalize the data
        sigma (np.ndarray): Standard deviation used to normalize the data

    Returns:
        prediction (np.ndarray): Prediction in native units
    """
    n, c, t, h, w = prediction.shape
    mu = mu.reshape(1,c,1,1,1)
    sigma = sigma.reshape(1,c,1,1,1)

    prediction = prediction * sigma + mu
    return prediction

def parse_yaml_prediction(yaml_file: Path,
        chkpt_idx: Optional[int] = None,
        device: Optional[int] = None) -> GEOSAQcGANFCast:
    """
    Use a config file to initialize an GEOSAQcGANFCast trainer object 
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

    if config is None:
        print(f"Cannot read ${yaml_file}")
        exit(1)
    else:
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
        
        trainer = GEOSAQcGANFCast(gan_architecture, data_params, 
                                train_hyperparams, device, 
                                Path(config["chkpt_dir"]), 
                                diffusion_params=diffusion_params, 
                                chkpt_idx=chkpt_idx)  

        return trainer

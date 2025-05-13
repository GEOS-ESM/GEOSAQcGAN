
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
from dataset import GEOSCFLargeEnsembleDataset

# GEOS CF Dataset
class GEOSCFLargeDataset(GEOSCFLargeEnsembleDataset):
        

    @property
    def time_span(self) -> int:
        """
        Returns the number of unique input output time sequences
        """
        return self.n_timesteps 


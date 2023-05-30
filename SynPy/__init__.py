# Import to load the necessary functions to begin using python helper functions for NFTsim

# Data & Directory
import os, time, sys, glob

# NFTsim setup
# sys.path.insert(0, 'src/')
from .nftsim_setup import nftsim_setup

nftsim_setup() # Check if NFTsim exists in directory.  Optionally clone and compile if not. 
sys.path.append('nftsim/')
nftsim_path = 'nftsim/bin/nftsim'

for d in ('confs/', 'outputs/'):
    if not os.path.exists(d): # if the directory doesn't exist, create it
        os.makedirs(d)

# Local functions
from .nftsim_generator_FUNCTIONS import *
from .conf_handler import *
from .output_handler import *
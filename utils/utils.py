import os
import glob
import torch
import sys
import shutil
import random
import logging
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.spaces import MultiDiscrete, Discrete, Box
from moviepy.editor import VideoFileClip, concatenate_videoclips


def config_logging(log_file="main.log"):
    date_format = '%Y-%m-%d %H:%M:%S'
    log_format = '%(asctime)s: [%(levelname)s]: %(message)s'
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Set up the FileHandler for logging to a file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    
def set_seed_everywhere(env: gym.Env, seed=0):
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # pl.seed_everything(cfg.seed) #equal to fist 3 lines and it can pass seeds
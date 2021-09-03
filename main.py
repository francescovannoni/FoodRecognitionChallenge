# Downloading data (and exploring)

import os
import json
import cv2
import pandas as pd
import numpy as np
import platform

if not os.path.exists("data"):
    !mkdir data
    !mkdir data/train
    !mkdir data/val

if not os.path.exists("data/train/images"):
    !wget https://datasets.aicrowd.com/default/aicrowd-public-datasets/food-recognition-challenge/v0.4/train-v0.4.tar.gz
    !unzip -q train-v0.4.tar.gz -d ./data

if not os.path.exists("data/val/images"):
    !wget https://datasets.aicrowd.com/default/aicrowd-public-datasets/food-recognition-challenge/v0.4/val-v0.4.tar.gz
    !unzip -q val-v0.4.tar.gz -d ./data



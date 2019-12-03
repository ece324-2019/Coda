import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from model import ConvNN
from model import MultiLP
from model import MusicDataset

import argparse
from time import time
import ipdb

torch.manual_seed(0)

data = pd.read_pickle('data/part1.pkl')
data.columns = ["normalized", "instruments"]

# print(data["instruments"].value_counts())

labels = data["instruments"]
music_train = data["normalized"].values
music_train = music_train[:50]
music_train = np.stack(music_train[:50]).reshape(-1, 1025 * 130)    

# Encode instruments
oneh_encoder = OneHotEncoder(categories="auto")
label_oneh = oneh_encoder.fit_transform(labels.values.reshape(-1, 1)).toarray()
label_oneh = label_oneh[:50]

train_data = MusicDataset(music_train, label_oneh)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

ipdb.set_trace()

# return train_loader, 1025 * 130
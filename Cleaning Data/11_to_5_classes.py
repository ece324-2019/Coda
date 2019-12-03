import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

data = pd.read_pickle('mel_aug.pkl')
data.columns = ["normalized", "instruments"]
labels = data["instruments"].values
music_data = data["normalized"].values

# Remove bad data
music_data = np.append(music_data[:10092], music_data[10093:])
labels = np.append(labels[:10092], labels[10093:])

data = pd.DataFrame({'normalized': music_data, 'instruments': labels})

# Reclassify instruments
data.replace("cel", "string", inplace=True)
data.replace("gac", "string", inplace=True)
data.replace("gel", "string", inplace=True)
data.replace("vio", "string", inplace=True)

data.replace("flu", "wood", inplace=True)
data.replace("cla", "wood", inplace=True)

data.replace("tru", "brass", inplace=True)
data.replace("sax", "brass", inplace=True)

data.replace("pia", "key", inplace=True)
data.replace("org", "key", inplace=True)

# print(data['instruments'].value_counts())

# Slice so that they only have the lowest value count of a class
data_string = data[data["instruments"] == "string"].sample(frac=1).reset_index(drop=True)[:2333]
data_wood = data[data["instruments"] == "wood"].sample(frac=1).reset_index(drop=True)[:2333]
data_key = data[data["instruments"] == "key"].sample(frac=1).reset_index(drop=True)[:2333]
data_brass = data[data["instruments"] == "brass"].sample(frac=1).reset_index(drop=True)[:2333]
data_voi = data[data["instruments"] == "voi"].sample(frac=1).reset_index(drop=True)[:2333]

data = pd.concat([data_string, data_wood, data_key, data_brass, data_voi], ignore_index=True)

# Encode labels
label_encoder = LabelEncoder()
data['instruments'] = label_encoder.fit_transform(data['instruments'])

# Save
data.to_pickle("../data/5_class.pkl")
print("--Saved--")



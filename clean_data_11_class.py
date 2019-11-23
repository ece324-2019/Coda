import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

instruments_list = ["cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
# "cel", 
data = pd.read_pickle('mel_aug.pkl')
data.columns = ["normalized", "instruments"]
labels = data["instruments"].values
music_data = data["normalized"].values

# Remove bad data
music_data = np.append(music_data[:10092], music_data[10093:])
labels = np.append(labels[:10092], labels[10093:])

data = pd.DataFrame({'normalized': music_data, 'instruments': labels})

# Slice so that they only have the lowest value count of a class
data_temp = data[data["instruments"] == "cel"][:1163]
for i in instruments_list:
        data1 = data[data["instruments"] == i][:1163]
        data_temp = pd.concat([data_temp, data1], ignore_index=True)

data = data_temp
# print(data['instruments'].value_counts())

# Encode labels
label_encoder = LabelEncoder()
data['instruments'] = label_encoder.fit_transform(data['instruments'])

# Save
data.to_pickle("11_class.pkl")


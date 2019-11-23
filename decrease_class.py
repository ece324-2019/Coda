import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

data = pd.read_pickle('mel_aug.pkl')
data.columns = ["normalized", "instruments"]
labels = data["instruments"].values
music_data = data["normalized"].values

music_data = np.append(music_data[:3364], music_data[3365:])
labels = np.append(labels[:3364], labels[3365:])

data = pd.DataFrame({'normalized': music_data, 'instruments': labels})

print(data['instruments'].value_counts())


# print(data.head())

#128*130

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
data_string = data[data["instruments"] == "string"][:777]
data_wood = data[data["instruments"] == "wood"][:777]
data_key = data[data["instruments"] == "key"][:777]
data_brass = data[data["instruments"] == "brass"][:777]
data_voi = data[data["instruments"] == "voi"][:777]

# data_string = data[data["instruments"] == "voi"][:720]
# data_wood = data[data["instruments"] == "wood"][:720]
# data_key = data[data["instruments"] == "pia"][:720]
# data_brass = data[data["instruments"] == "brass"][:777]
# data_voi = data[data["instruments"] == "voi"][:777]

# print(data_string.shape)

data = pd.concat([data_string, data_wood, data_key, data_brass, data_voi], ignore_index=True)
# data = pd.concat([data_string, data_wood, data_key], ignore_index=True)
print(data['instruments'].value_counts())
print(data.head())

label_encoder = LabelEncoder()
data['instruments'] = label_encoder.fit_transform(data['instruments'])
labels = data["instruments"].values
music_data = data["normalized"].values

# print(data.head())

print(data['instruments'].value_counts())

data = data.sample(frac=1).reset_index(drop=True)

data.to_pickle("5_class.pkl")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import ipdb

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
# five_class = ["string", "wood", "key", "brass", "voi"]

# mel_aug requires some special cleaning
data = pd.read_pickle('mel_aug.pkl')
data.columns = ["normalized", "instruments"]
labels = data["instruments"].values
music_data = data["normalized"].values

# Make them a dataframe again
data = pd.DataFrame({'normalized': music_data, 'instruments': labels})

# for i in range(music_data4.shape[0]):
# 	if not ((music_data4[i] > -100).all() and (music_data4[i] < 100).all()):
# 		print('bad!, ', i)
# print("Done")

# Essentially one hot encodes it
for index, row in data.iterrows():
	b = np.zeros(len(instruments_list))
	b[instruments_list.index(row["instruments"])] = 1
	row['instruments'] = b

data = data.sample(frac=1).reset_index(drop=True)

# Save
data.to_pickle("../data/clean_mel_aug.pkl")
print("--Saved--")
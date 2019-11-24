import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import ipdb

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
# instruments_list.index("cel")

data = pd.read_pickle('test1.pkl')
data.columns = ["normalized", "instruments"]
labels = data["instruments"].values
music_data = data["normalized"].values

# Remove bad data

music_data = np.append(music_data[:357], music_data[358:])
labels = np.append(labels[:357], labels[358:])

# for i in range(music_data.shape[0]):
# 	if not ((music_data[i] > -100).all() and (music_data[i] < 100).all()):
# 		print('bad!, ', i)
# print("yo")

data = pd.DataFrame({'normalized': music_data, 'instruments': labels})

# five_class = ["string", "wood", "key", "brass", "voi"]

# Essentiall one hot encodes it
for index, row in data.iterrows():
	b = np.zeros(len(instruments_list))
	# ipdb.set_trace()
	for instrument in row['instruments']:
		b[instruments_list.index(instrument)] = 1
	row['instruments'] = b

# Save
data.to_pickle("11_multiclass.pkl")


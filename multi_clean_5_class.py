import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import ipdb

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
# five_class = ["string", "wood", "key", "brass", "voi"]

data1 = pd.read_pickle('test1.pkl')
data2 = pd.read_pickle('test2.pkl')
data3 = pd.read_pickle('test3.pkl')
data = pd.concat([data1, data2, data3], ignore_index=True)

data.columns = ["normalized", "instruments"]
labels = data["instruments"].values
music_data = data["normalized"].values


# mel_aug requires some special cleaning
data4 = pd.read_pickle('mel_aug.pkl')
data4.columns = ["normalized", "instruments"]
labels4 = data4["instruments"].values
music_data4 = data4["normalized"].values

# Remove bad data
music_data = np.append(music_data[:1187], music_data[1188:])
labels = np.append(labels[:1187], labels[1188:])
music_data = np.append(music_data[:357], music_data[358:])
labels = np.append(labels[:357], labels[358:])
music_data4 = np.append(music_data4[:10092], music_data4[10093:])
labels4 = np.append(labels4[:10092], labels4[10093:])

# Make them a dataframe again
data = pd.DataFrame({'normalized': music_data, 'instruments': labels})
data4 = pd.DataFrame({'normalized': music_data4, 'instruments': labels4})

# for i in range(music_data4.shape[0]):
# 	if not ((music_data4[i] > -100).all() and (music_data4[i] < 100).all()):
# 		print('bad!, ', i)
# print("yo")


# Essentially one hot encodes it
for index, row in data.iterrows():
	b = np.zeros(len(instruments_list))
	for instrument in row['instruments']:
		b[instruments_list.index(instrument)] = 1
	row['instruments'] = b

for index, row in data4.iterrows():
	b = np.zeros(len(instruments_list))
	# print("lisaa", row["instruments"])
	# for instrument in row['instruments']:
	b[instruments_list.index(row["instruments"])] = 1
	row['instruments'] = b

# print(data4.head())
data = pd.concat([data, data4], ignore_index=True)

# Save
data.to_pickle("11_multiclass.pkl")
print("--Saved--")
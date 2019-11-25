# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import ipdb

# instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
# concat_list = ["cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
 
# data = pd.read_pickle('mel_aug.pkl')
# data.columns = ["normalized", "instruments"]
# labels = data["instruments"].values
# music_data = data["normalized"].values

# # Remove bad data
# music_data = np.append(music_data[:10092], music_data[10093:])
# labels = np.append(labels[:10092], labels[10093:])

# data = pd.DataFrame({'normalized': music_data, 'instruments': labels})

# # Slice so that they only have the lowest value count of a class
# data_temp = data[data["instruments"] == "cel"][:1163]
# for i in concat_list:
# 	data1 = data[data["instruments"] == i][:1163]
# 	data_temp = pd.concat([data_temp, data1], ignore_index=True)

# data = data_temp


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
# print("yo")

# Essentially one hot encodes it
for index, row in data.iterrows():
	b = np.zeros(len(instruments_list))
	b[instruments_list.index(row["instruments"])] = 1
	row['instruments'] = b

data = data.sample(frac=1).reset_index(drop=True)

print(data)
# Save
data.to_pickle("clean_mel_aug.pkl")
print("--Saved--")







# labels = data["instruments"].values


# print(data['instruments'].value_counts())

# print(data.head())

# oneh_encoder = OneHotEncoder(categories="auto")
# label_oneh = oneh_encoder.fit_transform(labels_2.reshape(-1, 1)).toarray()
# ipdb.set_trace()

# for i in label_oneh:
# 	if len(i) != len(instruments_list):
# 		print(len(i))




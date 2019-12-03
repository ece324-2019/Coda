import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import ipdb
import random

np.set_printoptions(threshold=np.inf)

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
string = ["cel", "gac","gel", "vio"]
#           0       3    4     9   
stringIndex = []

data1 = pd.read_pickle('test1.pkl')
data2 = pd.read_pickle('test2.pkl')
data3 = pd.read_pickle('test3.pkl')
data = pd.concat([data1, data2, data3], ignore_index=True)

data.columns = ["normalized", "instruments"]
labels = data["instruments"].values
music_data = data["normalized"].values

# Remove bad data
music_data = np.append(music_data[:1187], music_data[1188:])
labels = np.append(labels[:1187], labels[1188:])
music_data = np.append(music_data[:357], music_data[358:])
labels = np.append(labels[:357], labels[358:])

# Make them a dataframe again
data = pd.DataFrame({'normalized': music_data, 'instruments': labels})

# for i in range(music_data4.shape[0]):
# 	if not ((music_data4[i] > -100).all() and (music_data4[i] < 100).all()):
# 		print('bad!, ', i)
# print("yo")


# Essentially one hot encodes it
for index, row in data.iterrows():
        b = np.zeros(len(instruments_list))
        for instrument in row['instruments']:
                b[instruments_list.index(instrument)] = 1
                if instrument in string and index not in stringIndex:
                        stringIndex.append(index)
        row['instruments'] = b

labels = list(data["instruments"].values)
music_data = list(data["normalized"].values)


random.shuffle(stringIndex)

stringIndex = stringIndex[:300]
stringIndex.sort(reverse = True) 

strData = []
strLabel = []
for index, i in enumerate(stringIndex):
        if index < 300: 
                strData.append(music_data.pop(i))
                strLabel.append(labels.pop(i))
        

data = pd.DataFrame({'normalized': music_data, 'instruments': labels})
stringData = pd.DataFrame({'normalized': strData, 'instruments': strLabel})
# ipdb.set_trace()
# Save
data.to_pickle("string_test.pkl")
stringData.to_pickle("string_test_pt2.pkl")
print("--Saved--")
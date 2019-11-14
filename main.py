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

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

def load_model(args, train_len):
        # model = ConvNN()
        # model = MultiLP(train_len)
        model = ConvNN()
        loss_func = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
 
        return model, loss_func, optimizer

def load_data():
        data = pd.read_pickle('data/part1.pkl')
        data.columns = ["normalized", "instruments"]
        # print(data.head())
        # print("shape: ", data.shape)
        # print(data["instruments"].value_counts())
        label_encoder = LabelEncoder()
        data['instruments'] = label_encoder.fit_transform(data['instruments'])
        labels = data["instruments"].values
        music_train = data["normalized"].values
        # music_train = music_train[-8:-1]
        # ipdb.set_trace()
        # Encode instruments
        oneh_encoder = OneHotEncoder(categories="auto", sparse=False)
        # labels = oneh_encoder.fit_transform(labels.values.reshape(-1, 1)).toarray()
        # labels = oneh_encoder.fit_transform(labels.reshape(-1, 1))
        train_data = MusicDataset(music_train, labels)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        return train_loader, (music_train[0].shape[0] * music_train[0].shape[1])

def main(args):

        train_loader, train_len = load_data()
        model, loss_func, optimizer = load_model(args, train_len)
        
        running_loss = []
        running_accuracy = []
        # running_valid_loss = []
        running_valid_accuracy = []
        nRec = []

        start = time()

        for epoch in range(args.epochs):
                train_acc = 0.0
                train_loss = 0.0
                total_count = 0.0

                for j, data in enumerate(train_loader):
                        # ipdb.set_trace()
                        feat, labels = data
                
                        optimizer.zero_grad()
                        predict = model(feat.unsqueeze(1)).float()
                        # print(labels)
                        loss = loss_func(predict, labels.long())
                        loss.backward()
                        optimizer.step()

                        _, predicted = torch.max(predict.data, 1)
                        correct = (predicted == labels).sum().item()
                        train_acc = (correct / labels.size(0))

                        # train_acc += int(((predict > 0.5).squeeze().float() == labels).sum())
                        total_count += args.batch_size
                        train_loss += loss.item()
            
                running_accuracy.append(train_acc/total_count)
                running_loss.append(train_loss/float(j+1))

                if epoch % args.eval_every == args.eval_every-1:
                        # print("Epoch: %d | Training accuracy: %f | Training loss: %f | Val accuracy: %f | Val loss: %f"
                        # % (epoch, running_accuracy[-1], running_loss[-1], running_valid_accuracy[-1], running_valid_loss[-1]))
                        print("Epoch: %d | Training accuracy: %f | Training loss: %f"
                        % (epoch+1, running_accuracy[-1], running_loss[-1]))

        end = time()
        print("====FINAL VALUES====")
        # print("Test accuracy: %f | Test loss: %f" % (test_accuracy, test_loss))  
        print("Training acc: %f | Valid acc: %f | Time: %f" % (max(running_accuracy), max(running_valid_accuracy), end - start))
        # print("Training loss: %f | Valid loss: %f " % (min(running_loss), min(running_valid_loss)))
        # print("overfit acc: %f | overfit loss: %f" % (max(overfit_accuracy), min(overfit_loss)))

        fig = plt.figure()
        ax = plt.subplot(1, 2, 1)
        ax.plot(nRec, running_loss, label='Training')
        # ax.plot(nRec,running_valid_loss, label='Validation')
        plt.title('Training Loss vs. epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax.legend()

        bx = plt.subplot(1, 2, 2)
        bx.plot(nRec, running_accuracy, label='Training')
        # bx.plot(nRec,running_valid_accuracy, label='Validation')
        plt.title('Training Accuracy vs. epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        bx.legend()
        plt.show()


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--eval_every', type=int, default=10)

        args = parser.parse_args()

        main(args)

# Remove instrument column and store in new variable
# instrument_oneh = data["instruments"]
# instrument_oneh = instrument_oneh.to_numpy()
# data = data.drop(columns="instruments")

# one_hot = []
# one_hot.append(oneh_encoder.fit_transform(data["instruments"].values.reshape(-1, 1)).toarray())
# one_hot = np.concatenate(one_hot, axis=1)
# print("one hot", one_hot)
# print("one hot shape", one_hot.shape)

# oneh_df = pd.DataFrame(one_hot) 
# data = data.drop(columns="instruments")
# data = pd.concat((data, oneh_df), axis=1)







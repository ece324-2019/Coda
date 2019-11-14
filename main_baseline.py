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
        model = MultiLP(train_len)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)
 
        return model, loss_func, optimizer

def load_data(batch_size):
        data = pd.read_pickle('data/part1.pkl')
        data.columns = ["normalized", "instruments"]

        # print(data["instruments"].value_counts())

        labels = data["instruments"]
        music_train = data["normalized"].values
        music_train = music_train
        music_train = np.stack(music_train).reshape(-1, 1025 * 130)    
        
        # Encode instruments
        oneh_encoder = OneHotEncoder(categories="auto")
        label_oneh = oneh_encoder.fit_transform(labels.values.reshape(-1, 1)).toarray()
        label_oneh = label_oneh
        
        train_data = MusicDataset(music_train, label_oneh)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        return train_loader, 1025 * 130

def main(args):

        train_loader, train_len = load_data(args.batch_size)
        model, loss_func, optimizer = load_model(args, train_len)
        
        running_loss = []
        running_accuracy = []
        # running_valid_loss = []
        # running_valid_accuracy = []
        nRec = []

        start = time()

        for epoch in range(args.epochs):
                train_acc = 0.0
                train_loss = 0.0
                total_count = 0.0

                if epoch == 0:
                        print("Starting...")

                for j, data in enumerate(train_loader):
                        
                        feat, labels = data
                
                        optimizer.zero_grad()
                        predict = model(feat).float()
                
                        loss = loss_func(predict.squeeze(), labels.float())
                        loss.backward()
                        optimizer.step()

                        train_acc += (predict.max(1)[1].float() == labels.max(1)[1].float()).sum().float().item()
                        # ipdb.set_trace()
                        total_count += args.batch_size
                        train_loss += loss.item()
            
                # ipdb.set_trace()
                running_accuracy.append(train_acc/total_count)
                running_loss.append(train_loss)  #/float(j+1)
                nRec.append(epoch)

                if epoch % args.eval_every == args.eval_every-1:
                        # print("Epoch: %d | Training accuracy: %f | Training loss: %f | Val accuracy: %f | Val loss: %f"
                        # % (epoch, running_accuracy[-1], running_loss[-1], running_valid_accuracy[-1], running_valid_loss[-1]))
                        print("Epoch: %d | Training accuracy: %f | Training loss: %f"
                        % (epoch+1, running_accuracy[-1], running_loss[-1]))


        end = time()
        print("====FINAL VALUES====")
        # print("Test accuracy: %f | Test loss: %f" % (test_accuracy, test_loss))  
        print("Training acc: %f" % (max(running_accuracy)))
        print("Training loss: %f" % (min(running_loss)))


        # plt.plot(running_accuracy, label="Training Error")
        # plt.title("Training Loss")
        # plt.xlabel("Epochs")
        # plt.xlim(left=0)
        # plt.ylabel("Loss")
        # plt.legend(loc="upper right")
        # plt.savefig("baseline.png")
        # plt.show()
        # plt.clf()

        
        fig = plt.figure()
        ax = plt.subplot(1, 2, 1)
        plt.plot(nRec, running_accuracy, label='Training')
        plt.title('Training Accuracy vs. Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        ax.legend()

        bx = plt.subplot(1, 2, 2)
        bx.plot(nRec, running_loss, label='Training')
        plt.title('Training Loss vs. Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        bx.legend()
        plt.show()
        plt.savefig("baseline.png")
        plt.clf()

        # # plt.figure(figsize=(9, 3))
        # # plt.subplot(121)
        # plt.plot(nRec, running_accuracy)
        # # plt.subplot(122)
        # # plt.plot(nRec, running_loss)
        # # plt.title('Training Accuracy and Loss vs. Epoch')
        # plt.show()



if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=6)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--eval_every', type=int, default=2)

        args = parser.parse_args()

        main(args)


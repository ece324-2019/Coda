import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from model import ConvNN
from model import MultiLP
from model import MusicDataset

import argparse
from time import time
import ipdb

torch.manual_seed(0)

gpu = False

if torch.cuda.is_available():
    gpu = True
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('Using CUDA')

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]


def load_model(args, train_len):
    model = MultiLP(train_len)

    if torch.cuda.is_available():
        model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return model, loss_func, optimizer


def load_data(batch_size):
    data = pd.read_pickle('clean_mel_aug.pkl')
    data.columns = ["normalized", "instruments"]
    data = data.sample(frac=1).reset_index(drop=True)

    # print(data["instruments"].value_counts())

    labels = data["instruments"]
    music_train = data["normalized"].values
    music_train = np.stack(music_train).reshape(-1, 65*128)

    # Encode instruments
    # label_encoder = LabelEncoder()
    # oneh_encoder = OneHotEncoder(categories="auto")
    # label_oneh = oneh_encoder.fit_transform(labels.values.reshape(-1, 1)).toarray()
    # labels = label_encoder.fit_transform(labels)

    train_data = MusicDataset(music_train, labels)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # print(train_loader.dataset)

    return train_loader, 65*128


def main(args):
    train_loader, train_len = load_data(args.batch_size)
    model, loss_func, optimizer = load_model(args, train_len)

    running_loss = []
    running_accuracy = []
    true = []
    pred = []
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
            if not gpu:
                # true.extend(np.where(labels == 1))
                true.extend(labels.numpy())

            if torch.cuda.is_available():
                feat, labels = feat.to(device), labels.to(device)
                true.extend(labels.cpu().numpy())

            optimizer.zero_grad()
            predict = model(feat).float()
            _, predicted = torch.max(predict.data, 1)
            if not gpu:
                pred.extend(predicted.numpy())
            else:
                pred.extend(predicted.cpu().numpy())

            loss = loss_func(predict.squeeze(), labels.nonzero()[:, 1].long())

            loss.backward()
            optimizer.step()
            ipdb.set_trace()
            train_acc += (predict.max(1)[1].float() == labels.max(1)[1].float()).sum().float().item()
            total_count += args.batch_size
            train_loss += loss.item()

            # confusion = confusion_matrix(labels, predict)
            # print("confusion", confusion)

            # print("train_loss", train_loss)

        running_accuracy.append(train_acc / total_count)
        running_loss.append(train_loss / float(j + 1))
        nRec.append(epoch)

        if epoch % args.eval_every == args.eval_every - 1:
            # print("Epoch: %d | Training accuracy: %f | Training loss: %f | Val accuracy: %f | Val loss: %f"
            # % (epoch, running_accuracy[-1], running_loss[-1], running_valid_accuracy[-1], running_valid_loss[-1]))
            print("Epoch: %d | Training accuracy: %f | Training loss: %f"
                  % (epoch + 1, running_accuracy[-1], running_loss[-1]))

    # confusion = None

    # # Confusion matrix
    # for k, data in enumerate(train_loader):
    #         feat, labels = data

    #         # rounded_labels = np.argmax(labels, axis=1)
    #         rounded_labels = labels.max(1)[1].float()

    #         predict = model(feat).float()
    #         loss = loss_func(predict.squeeze(), labels.nonzero()[:,1].long())
    #         predict = model(feat).float()
    #         # ipdb.set_trace()
    #         # train_acc += (predict.max(1)[1].float() == labels.max(1)[1].float()).sum().float().item()
    #         confusion = confusion_matrix(rounded_labels, predict.detach())
    #         print("confusion", confusion)

    end = time()
    print("====FINAL VALUES====")
    # print("Test accuracy: %f | Test loss: %f" % (test_accuracy, test_loss))
    print("Training acc: %f" % (max(running_accuracy)))
    print("Training loss: %f" % (min(running_loss)))

    print(confusion_matrix(true, pred))
    print(instruments_list)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=3)

    args = parser.parse_args()

    main(args)

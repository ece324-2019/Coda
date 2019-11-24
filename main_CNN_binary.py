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

torch.manual_seed(0)

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

gpu = False

if torch.cuda.is_available():
    gpu = True
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('Using CUDA')


def load_model(args, train_len):
    # model = ConvNN()
    # model = MultiLP(train_len)
    model = ConvNN()
    if torch.cuda.is_available():
        model.cuda()
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return model, loss_func, optimizer


def load_data():
    data = pd.read_pickle('mel_aug.pkl')
    data.columns = ["normalized", "instruments"]

    instrument = 'gac'
    for index, row in data.iterrows():
        if instrument in row['instruments']:
            row['instruments'] = 1
        else:
            row['instruments'] = 0

    num_samp = min(data["instruments"].value_counts())

    # Randomly balance the dataset by creating two dataframes containing only values of either class and sampling from the
    # larger one. Then, we concatenate to create a balanced dataframe
    yes = data[data['instruments'] == 1]
    no = data[data['instruments'] == 0]

    balanced_data = pd.concat([yes.sample(n=num_samp, random_state=1)
                                  , no.sample(n=num_samp, random_state=1)])

    music_data = balanced_data["normalized"].values
    labels = balanced_data["instruments"].values

    music_data = np.append(music_data[:10092], music_data[10093:])
    labels = np.append(labels[:10092], labels[10093:])

    train_data, valid_data, train_labels, valid_labels = train_test_split(music_data, labels, test_size=0.1, random_state=1)

    # np.savetxt('bad.csv', music_train[3364], delimiter=',')
    # for i in range(music_data.shape[0]):
    #     if not ((music_data[i] > -100).all() and (music_data[i] < 100).all()):
    #         print('bad!, ', i)

    # Encode instruments
    train_set = MusicDataset(train_data, train_labels)
    valid_set = MusicDataset(valid_data, valid_labels)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    return train_loader, valid_loader, (music_data[0].shape[0] * music_data[0].shape[1])


def main(args):
    def evaluate(model, iter):
        total_corr = 0
        total_num = 0
        running_loss = 0
        cnt = 0
        with torch.no_grad():
            for i, batch in enumerate(iter):
                # if i < int((0.25 * len(iter.dataset)) / args.batch_size):
                cnt += 1
                feat, labels = batch
                if torch.cuda.is_available():
                    feat, labels = feat.to(device), labels.to(device)

                predict = model(feat.unsqueeze(1)).squeeze()
                # Calculate loss
                loss = loss_func(input=predict, target=labels.float())
                running_loss += loss

                # Calculate correct labels and accuracy
                total_num += len(labels)
                corr = (predict > 0.5).squeeze().float() == labels.float()
                total_corr += int(corr.sum())
        return total_corr / total_num, running_loss / total_num

    train_loader, valid_loader, train_len = load_data()
    model, loss_func, optimizer = load_model(args, train_len)

    trainAccRec = []
    trainLossRec = []
    validAccRec = []
    validLossRec = []
    nRec = []
    true = []
    pred = []

    start = time()
    print('Starting training')
    for epoch in range(args.epochs):
        train_acc = 0.0
        train_loss = 0.0
        total_count = 0.0

        for j, data in enumerate(train_loader):
            feat, labels = data
            if not gpu:
                true.extend(labels.numpy())
            if torch.cuda.is_available():
                feat, labels = feat.to(device), labels.to(device)
                true.extend(labels.cpu().numpy())

            optimizer.zero_grad()
            predict = model(feat.unsqueeze(1)).squeeze()

            loss = loss_func(predict, labels.float())
            print(loss)
            loss.backward()
            optimizer.step()

        if epoch % args.eval_every == args.eval_every - 1:
            model = model.eval()
            train_acc, train_loss = evaluate(model, train_loader)
            trainAccRec.append(train_acc)
            trainLossRec.append(train_loss)
            val_acc, val_loss = evaluate(model, valid_loader)
            validAccRec.append(val_acc)
            validLossRec.append(val_loss)
            nRec.append(epoch)
            model.train()
            # print("Epoch: %d | Training accuracy: %f | Training loss: %f | Val accuracy: %f | Val loss: %f"
            # % (epoch, running_accuracy[-1], running_loss[-1], running_valid_accuracy[-1], running_valid_loss[-1]))
            print("Epoch: %d | Training accuracy: %f | Training loss: %f | Val accuracy: %f | Val loss: %f"
                  % (epoch + 1, train_acc, train_loss, val_acc, val_loss))

    end = time()
    # print("====FINAL VALUES====")
    # print("Test accuracy: %f | Test loss: %f" % (test_accuracy, test_loss))
    # print("Training acc: %f | Valid acc: %f | Time: %f" % (
    # max(train_acc), max(running_valid_accuracy), end - start))
    # print("Training loss: %f | Valid loss: %f " % (min(running_loss), min(running_valid_loss)))
    # print("overfit acc: %f | overfit loss: %f" % (max(overfit_accuracy), min(overfit_loss)))

    # print(confusion_matrix(true, pred))
    # print(instruments_list)

    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    fig.tight_layout()
    ax.plot(nRec, trainLossRec, label='Training')
    ax.plot(nRec, validLossRec, label='Validation')
    plt.title('Loss vs. epoch')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax.legend()

    bx = plt.subplot(1, 2, 2)
    fig.tight_layout()
    bx.plot(nRec, trainAccRec, label='Training')
    bx.plot(nRec, validAccRec, label='Validation')
    # bx.plot(nRec,running_valid_accuracy, label='Validation')
    plt.title('Accuracy vs. epoch')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim((0, 1))
    bx.legend()
    plt.show()
    plt.savefig("cnn.png")
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval_every', type=int, default=3)

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

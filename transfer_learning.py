from __future__ import print_function, division

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import copy

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

plt.ion()   # interactive mode

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

# data_dir = 'data/'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                              shuffle=True, num_workers=4)
#               for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# return train_loader, valid_loader, (music_data[0].shape[0] * music_data[0].shape[1])

def main(args):

        def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
                since = time()

                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = 0.0

                for epoch in range(num_epochs):
                        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                        print('-' * 10)

                        # Each epoch has a training and validation phase
                        # for phase in ['train', 'val']:
                        #         if phase == 'train':
                        #                 model.train()  # Set model to training mode
                        #         else:
                        #                 model.eval()   # Set model to evaluate mode

                        running_loss = 0.0
                        running_corrects = 0

                        # Iterate over data.
                        for j, data in enumerate(train_loader):
                                inputs, labels = data
                                if torch.cuda.is_available():
                                        inputs = inputs.to(device)
                                        labels = labels.to(device)

                                # zero the parameter gradients
                                optimizer.zero_grad()

                                inputs = np.repeat(inputs[..., np.newaxis], 3, -1).permute([0, 3, 1, 2])

                                # track history if only in train
                                # with torch.set_grad_enabled(phase == 'train'):

                                outputs = model(inputs)
                                _, preds = torch.max(outputs, 1)
                                loss = criterion(outputs, labels)

                                # backward + optimize only if in training phase
                                # if phase == 'train':
                                loss.backward()
                                optimizer.step()

                                # statistics
                                running_loss += loss.item() * inputs.size(0)
                                running_corrects += torch.sum(preds == labels.data)
                        
                        # if phase == 'train':
                        scheduler.step()

                        epoch_loss = running_loss / dataset_sizes
                        epoch_acc = running_corrects.double() / dataset_sizes

                        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                                phase, epoch_loss, epoch_acc))

                        # deep copy the model
                        # if phase == 'val' and epoch_acc > best_acc:
                        #         best_acc = epoch_acc
                        #         best_model_wts = copy.deepcopy(model.state_dict())

                        print()

                time_elapsed = time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                print('Best val Acc: {:4f}'.format(best_acc))

                # load best model weights
                model.load_state_dict(best_model_wts)
                return model

        data = pd.read_pickle('data/MFCC_harm.pkl')
        data.columns = ["normalized", "instruments"]
        label_encoder = LabelEncoder()
        data['instruments'] = label_encoder.fit_transform(data['instruments'])
        labels = data["instruments"].values
        music_data = data["normalized"].values

        music_data = np.append(music_data[:3364], music_data[3365:])
        labels = np.append(labels[:3364], labels[3365:])

        train_data, valid_data, train_labels, valid_labels = train_test_split(music_data, labels, test_size=0.2,
                                                                                random_state=1)
        oneh_encoder = OneHotEncoder(categories="auto", sparse=False)
        train_set = MusicDataset(train_data, train_labels)
        valid_set = MusicDataset(valid_data, valid_labels)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
        print(train_loader)

        # data_dir = 'data/'
        # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
        #                                           data_transforms[x])
        #                   for x in ['train', 'val']}
        # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
        #                                              shuffle=True, num_workers=4)
        #               for x in ['train', 'val']}
        # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        # class_names = image_datasets['train'].classes


        dataset_sizes = len(train_data) 

        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features


        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 11)

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--eval_every', type=int, default=3)

    args = parser.parse_args()

    main(args)
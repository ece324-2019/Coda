import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import models

a = 1000
b = 1000


class MultiLP(nn.Module):
    def __init__(self, input_size):
        super(MultiLP, self).__init__()
        self.fc1 = nn.Linear(input_size, a)
        self.fc2 = nn.Linear(a, b)
        self.fc3 = nn.Linear(b, 11)

    def forward(self, features):
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=1)
        return x


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 500, kernel_size=(128, 1))
        # self.pool = nn.MaxPool2d(kernel_size=(1, 129))
        # self.conv2 = nn.Conv2d(3, 8, 5)

        self.fc1 = nn.Linear(500 * 2 * 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 500 * 2 * 1)
        x = torch.sigmoid(self.fc1(x))
        return x


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 11)
        self.hd = hidden_dim

    def forward(self, x):
        packed_output, hidden = self.gru(x)
        hidden = torch.sigmoid(self.fc(hidden.squeeze(0)))
        hidden = torch.softmax(hidden, dim=1)
        return hidden


class VGGModule(nn.Module):
    def __init__(self):
        super(VGGModule, self).__init__()
        self.layer1 = nn.Linear(1000, 11)
        self.net = models.vgg16(pretrained=True)
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x):
        x1 = self.net(x)
        y = self.layer1(x1)
        return y


class MusicDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # return [self.X[index, :], self.y[index]]
        return self.X[index], self.y[index]

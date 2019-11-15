import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


neurons = 1000

class MultiLP(nn.Module):

    def __init__(self, input_size):
        super(MultiLP, self).__init__()
        self.fc1 = nn.Linear(input_size, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 11)

    def forward(self, features):
        x = F.relu(self.fc1(features.float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
        

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(1025,2))
        self.pool = nn.MaxPool2d(kernel_size=(1, 129))
        # self.conv2 = nn.Conv2d(3, 8, 5)

        self.fc1 = nn.Linear(50*1*1, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1,50*1*1)
        x = F.relu(self.fc1(x))
        return x


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim)

        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths):
        packed_output, hidden = self.gru(packed)
        hidden = torch.sigmoid(self.fc(hidden.squeeze(0))).squeeze(1)
        return hidden


class MusicDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # return [self.X[index, :], self.y[index]]
        return self.X[index], self.y[index]

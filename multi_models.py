import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import models

a = 1000
b = 1000
output_layers = 1

class MultiInstrumClass(nn.Module):
	def __init__(self, input_size, num_instruments, emb_dim, hidden_dim, model_name):
		super(MultiInstrumClass, self).__init__()
		print("Using %s..." % model_name)
		if model_name == "baseline":
			self.models = nn.ModuleList([MultiLP(input_size) for _ in range(num_instruments)])
		elif model_name == "cnn":
			self.models = nn.ModuleList([ConvNN() for _ in range(num_instruments)])
		elif model_name == "rnn":
			self.models = nn.ModuleList([RNN(emb_dim, hidden_dim) for _ in range(num_instruments)])

	def forward(self, x):
		out = []
		for i in range(len(self.models)):
			out.append(self.models[i](x))
		return torch.stack(out, 1).squeeze()

class MultiLP(nn.Module):
	def __init__(self, input_size):
		super(MultiLP, self).__init__()
		self.fc1 = nn.Linear(input_size, a)
		self.fc2 = nn.Linear(a, b)
		self.fc3 = nn.Linear(b, output_layers)

	def forward(self, features):
		x = torch.relu(self.fc1(features))
		x = torch.relu(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		return x

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 500, kernel_size=(128, 6))
        self.pool = nn.MaxPool2d(kernel_size=(1, 129))
        self.conv2 = nn.Conv2d(3, 8, 5)
        self.fc1 = nn.Linear(500 * 60 * 1, 1)

    def forward(self, x):
        x = x.view(-1, 1, 128, 65)
        x = F.relu(self.conv1(x))
        x = x.view(-1, 500 * 60 * 1)
        x = torch.sigmoid(self.fc1(x))
        return x

class RNN(nn.Module):
	def __init__(self, embedding_dim, hidden_dim):
		super(RNN, self).__init__()
		self.gru = nn.GRU(embedding_dim, hidden_dim)
		self.fc = nn.Linear(hidden_dim, 1)

	def forward(self, x):
		x = x.view(65, 64, 128)
		packed_output, hidden = self.gru(x)
		hidden = torch.sigmoid(self.fc(hidden.squeeze(0)))
		hidden = F.softmax(hidden)
		return hidden

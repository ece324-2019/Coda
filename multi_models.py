import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import models

a = 200
b = 84
output_layers = 2

class MLP_string(nn.Module):
	def __init__(self, input_size):
		super(MLP_string, self).__init__()
		self.fc1 = nn.Linear(input_size, a)
		self.fc2 = nn.Linear(a, b)
		self.fc3 = nn.Linear(b, output_layers)

	def forward(self, features):
		x = torch.relu(self.fc1(features))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class MLP_brass(nn.Module):
	def __init__(self, input_size):
		super(MLP_brass, self).__init__()
		self.fc1 = nn.Linear(input_size, a)
		self.fc2 = nn.Linear(a, b)
		self.fc3 = nn.Linear(b, output_layers)

	def forward(self, features):
		x = torch.relu(self.fc1(features))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class MLP_wood(nn.Module):
	def __init__(self, input_size):
		super(MLP_wood, self).__init__()
		self.fc1 = nn.Linear(input_size, a)
		self.fc2 = nn.Linear(a, b)
		self.fc3 = nn.Linear(b, output_layers)

	def forward(self, features):
		x = torch.relu(self.fc1(features))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class MLP_key(nn.Module):
	def __init__(self, input_size):
		super(MLP_key, self).__init__()
		self.fc1 = nn.Linear(input_size, a)
		self.fc2 = nn.Linear(a, b)
		self.fc3 = nn.Linear(b, output_layers)

	def forward(self, features):
		x = torch.relu(self.fc1(features))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class MLP_voice(nn.Module):
	def __init__(self, input_size):
		super(MLP_voice, self).__init__()
		self.fc1 = nn.Linear(input_size, a)
		self.fc2 = nn.Linear(a, b)
		self.fc3 = nn.Linear(b, output_layers)

	def forward(self, features):
		x = torch.relu(self.fc1(features))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import models

a = 200
b = 84
output_layers = 1

class MultiInstrumClass(nn.Module):
        def __init__(self, input_size, num_instruments):
                super(MultiInstrumClass, self).__init__()
                self.models = nn.ModuleList([MultiLP(input_size) for _ in range(num_instruments)])

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

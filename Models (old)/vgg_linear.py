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

torch.manual_seed(0)

gpu = False

if torch.cuda.is_available():
    gpu = True
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('Using CUDA')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

instruments_list = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
criterion = nn.CrossEntropyLoss()

def evaluate(model, dataloader, size):
	running_loss = 0.0
	running_corrects = 0
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			inputs, labels = data
			if torch.cuda.is_available():
				inputs = inputs.to(device)
				labels = labels.to(device)

			outputs = model(inputs.view(-1,128*3))
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)

			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels).item()
		epoch_loss = running_loss / size
		epoch_acc = running_corrects / size
	return epoch_loss, epoch_acc


def main(args):
	def train_model(model, optimizer, criterion, num_epochs):
		since = time()
		best_model_wts = copy.deepcopy(model.state_dict())

		for epoch in range(num_epochs):
			print('Epoch {}/{}'.format(epoch, num_epochs - 1))
			print('-' * 10)

			running_loss = 0.0
			running_corrects = 0

			for j, data in enumerate(train_loader):
				inputs, labels = data
				if torch.cuda.is_available():
					inputs = inputs.to(device)
					labels = labels.to(device)

				optimizer.zero_grad()
				outputs = model(inputs.view(-1, 128*3))
				_, preds = torch.max(outputs, 1)
				loss = criterion(outputs, labels)

				loss.backward()
				optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels).item()

			epoch_loss = running_loss / dataset_sizes
			epoch_acc = running_corrects/ dataset_sizes

			model.eval()
			val_loss, val_acc = evaluate(model, valid_loader, dataset_sizes_valid)
			model.train()

			plot_train_acc.append(epoch_acc)
			plot_valid_acc.append(val_acc)
			plot_train_loss.append(epoch_loss)
			plot_valid_loss.append(val_loss)
			nRec.append(epoch)

			print('Train Loss: {:.4f} Train Acc: {:.4f} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch_loss, epoch_acc, val_loss, val_acc))
			print()

		time_elapsed = time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		print('Best val Acc: {:4f}'.format(max(plot_valid_acc)))

		model.load_state_dict(best_model_wts)
		return model

	data = pd.read_pickle('vggish.pkl')
	# print(data['instruments'].value_counts())
	data.columns = ["normalized", "instruments"]
	label_encoder = LabelEncoder()
	data['instruments'] = label_encoder.fit_transform(data['instruments'])
	labels = data["instruments"].values
	music_data = data["normalized"].values

	train_data, valid_data, train_labels, valid_labels = train_test_split(music_data, labels, test_size=0.1, random_state=1)
	# train_data, valid_data, train_labels, valid_labels = train_data[0:100], valid_data[0:100], train_labels[0:100], valid_labels[0:100]

	train_set = MusicDataset(train_data, train_labels)
	valid_set = MusicDataset(valid_data, valid_labels)
	train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
	valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
	dataset_sizes = len(train_data)
	dataset_sizes_valid = len(valid_data)

	model_ft = MultiLP(128*3) #65*128, 1025 * 130
	if torch.cuda.is_available():
		model_ft.cuda()

	plot_train_acc, plot_valid_acc, plot_train_loss, plot_valid_loss, nRec = [], [], [], [], []

	optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=args.lr)
	loss_func = torch.nn.CrossEntropyLoss()
	# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
	train_model(model_ft, optimizer_ft, loss_func, args.epochs)

	fig = plt.figure()
	ax = plt.subplot(1, 2, 1)
	plt.plot(nRec, plot_train_acc, label='Training')
	plt.plot(nRec, plot_valid_acc, label='Validation')
	plt.title('Accuracy vs. Epoch')
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	ax.legend()

	bx = plt.subplot(1, 2, 2)
	bx.plot(nRec, plot_train_loss, label='Training')
	bx.plot(nRec, plot_valid_loss, label='Validation')
	plt.title('Loss vs. Epoch')
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	bx.legend()
	plt.show()
	plt.savefig("baseline.png")
	plt.clf()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--epochs', type=int, default=20)

	args = parser.parse_args()

	main(args)

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
from multi_models import MultiInstrumClass
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
criterion = nn.BCELoss()

def evaluate(model, dataloader):
	running_loss = 0.0
	running_corrects = []
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			inputs, labels = data
			if torch.cuda.is_available():
				inputs = inputs.to(device)
				labels = labels.to(device)

			outputs = model(inputs)
			preds = outputs > 0.5
			loss = criterion(outputs.view(-1).float(), labels.view(-1).float())

			running_loss += loss.item() 
			running_corrects.append(torch.sum((preds.float() == labels.float())*(labels.float() > 0)).item() / (1e-5 + (preds > 0).sum().item()))

		epoch_loss = running_loss / len(running_corrects)
		epoch_acc = sum(running_corrects)/ len(running_corrects)


	return epoch_loss, epoch_acc


def main(args):
	def train_model(model, criterion, optimizer, scheduler, num_epochs):
		since = time()
		best_model_wts = copy.deepcopy(model.state_dict())

		for epoch in range(num_epochs):
			print('Epoch {}/{}'.format(epoch, num_epochs - 1))
			print('-' * 10)

			running_loss = 0.0
			running_corrects = []

			for j, data in enumerate(train_loader):
				inputs, labels = data
				if torch.cuda.is_available():
					inputs = inputs.to(device)
					labels = labels.to(device)

				optimizer.zero_grad()
				outputs = model(inputs)

				# _, preds = torch.max(outputs, 1)
				preds = outputs > 0.5

				loss = criterion(outputs.view(-1).float(), labels.view(-1).float())

				loss.backward()
				optimizer.step()

				# ipdb.set_trace()
				running_loss += loss.item()
				running_corrects.append(torch.sum((preds.float() == labels.float())*(labels.float() > 0)).item() / (1e-5 + (preds > 0).sum().item()))

			scheduler.step()

			epoch_loss = running_loss / len(running_corrects)
			epoch_acc = sum(running_corrects)/ len(running_corrects)

			model.eval()
			val_loss, val_acc = evaluate(model, valid_loader)
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

	data = pd.read_pickle('11_multiclass.pkl')
	# print(data['instruments'].value_counts())
	labels = data["instruments"].values
	music_data = data["normalized"].values

	# oneh_encoder = OneHotEncoder(categories="auto")
	# labels = oneh_encoder.fit_transform(labels.reshape(-1, 1)).toarray()

	music_data = np.stack(music_data).reshape(-1, 128*65) #65*128, 1025 * 65

	train_data, valid_data, train_labels, valid_labels = train_test_split(music_data, labels, test_size=0.1, random_state=1)


	train_set = MusicDataset(train_data, train_labels)
	valid_set = MusicDataset(valid_data, valid_labels)
	train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
	valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
        
	model_ft = MultiInstrumClass(128*65, 11, args.emb_dim, args.hidden_dim, args.model)

	if torch.cuda.is_available():
		model.cuda()

	plot_train_acc, plot_valid_acc, plot_train_loss, plot_valid_loss, nRec = [], [], [], [], []

	optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=args.lr, weight_decay=.04)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
	model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, args.epochs)

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
	plt.savefig("%s.png" % args.model)
	plt.clf()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--model', type=str, default='baseline',
						help="Model type: baseline,rnn,cnn (Default: baseline)")
	parser.add_argument('--emb_dim', type=int, default=128)
	parser.add_argument('--hidden_dim', type=int, default=100)

	args = parser.parse_args()

	main(args)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# from model import ConvNN
# from model import MultiLP
# from model import MusicDataset

# import argparse
# from time import time
# from __future__ import print_function, division

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

# def load_model(args, train_len):
#     model = MultiLP(train_len)

#     if torch.cuda.is_available():
#         model.cuda()

# #     loss_func = torch.nn.CrossEntropyLoss()
#     # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     return model, loss_func, optimizer


def evaluate(model, dataloader, size):
	running_loss = 0.0
	running_corrects = 0
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			inputs, labels = data
			if torch.cuda.is_available():
				inputs = inputs.to(device)
				labels = labels.to(device)

			outputs = model(inputs.squeeze())
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs.squeeze(), labels)

			# ipdb.set_trace()

			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels).item()
		epoch_loss = running_loss / size
		epoch_acc = running_corrects / size
	return epoch_loss, epoch_acc


def load_data(batch_size):
	data = pd.read_pickle('MFCC_harm.pkl')
	data.columns = ["normalized", "instruments"]
	data = data.sample(frac=1).reset_index(drop=True)
	label_encoder = LabelEncoder()
	data['instruments'] = label_encoder.fit_transform(data['instruments'])
	labels = data["instruments"].values
	music_data = data["normalized"].values
	labels = labels[:50]
	music_data = music_data[:50]

	# music_data = np.append(music_data[:3364], music_data[3365:])
	# music_data = np.stack(music_data).reshape(-1, 1025 * 130)
	# labels = np.append(labels[:3364], labels[3365:])


	train_data, valid_data, train_labels, valid_labels = train_test_split(music_data, labels, test_size=0.2, random_state=1)

	train_data = MusicDataset(train_data, train_labels)
	valid_data = MusicDataset(valid_data, valid_labels)
	train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
	valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
	# print(train_loader.dataset)

	return train_loader, 1025 * 130


def main(args):
	def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
		since = time()

		best_model_wts = copy.deepcopy(model.state_dict())
		best_acc = 0.0

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

				outputs = model(inputs.squeeze())
				_, preds = torch.max(outputs, 1)

				# print("outputs",  preds)
				# ipdb.set_trace()
				# _, predicted = torch.max(predict.data, 1)

				# loss_func(predicted, labels.long())
				loss = criterion(outputs.squeeze(), labels)

				loss.backward()
				optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				
				# running_corrects += torch.sum(preds == labels.data)
				running_corrects += torch.sum(preds == labels).item()

			scheduler.step()

			epoch_loss = running_loss / dataset_sizes
			epoch_acc = running_corrects/ dataset_sizes

			model.eval()
			val_loss, val_acc = evaluate(model, valid_loader, dataset_sizes_valid)
			model.train()

			print('Train Loss: {:.4f} Train Acc: {:.4f} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch_loss, epoch_acc, val_loss, val_acc))

			# deep copy the model
			# if phase == 'val' and epoch_acc > best_acc:
			#         best_acc = epoch_acc
			#         best_model_wts = copy.deepcopy(model.state_dict())

			print()

		time_elapsed = time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		print('Best val Acc: {:4f}'.format(best_acc))

		# load best model weights
		model.load_state_dict(best_model_wts)
		return model
	


	data = pd.read_pickle('MFCC_harm.pkl')
	data.columns = ["normalized", "instruments"]
	label_encoder = LabelEncoder()
	data['instruments'] = label_encoder.fit_transform(data['instruments'])
	labels = data["instruments"].values

	print("labelsss", labels)
	music_data = data["normalized"].values

	music_data = np.append(music_data[:3364], music_data[3365:])
	labels = np.append(labels[:3364], labels[3365:])
	music_data = np.stack(music_data).reshape(-1, 65*128)

	train_data, valid_data, train_labels, valid_labels = train_test_split(music_data, labels, test_size=0.1, random_state=1)
	# print("hiiiii",train_data[0].shape)
	train_data, valid_data, train_labels, valid_labels = train_data[0:100], valid_data[0:100], train_labels[0:100], valid_labels[0:100]


	train_set = MusicDataset(train_data, train_labels)
	valid_set = MusicDataset(valid_data, valid_labels)
	train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
	valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)
	dataset_sizes = len(train_data)
	dataset_sizes_valid = len(valid_data)

	# model_ft = models.resnet18(pretrained=True)

	# print("train_loader.dataset",train_data.shape)
	model_ft = MultiLP(65*128)
	if torch.cuda.is_available():
		model.cuda()


	# num_ftrs = model_ft.fc.in_features
	# model_ft.fc = nn.Linear(num_ftrs, 11)
	# model_ft = model_ft.to(device)
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
	model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)


	# train_loader, train_len = load_data(args.batch_size)
	# model, loss_func, optimizer = load_model(args, train_len)

	# running_loss = []
	# running_accuracy = []
	# true = []
	# pred = []
	# running_valid_loss = []
	# running_valid_accuracy = []
	# nRec = []

	# start = time()
	# print("Starting...")

	# for epoch in range(args.epochs):
	# 	train_acc = 0.0
	# 	train_loss = 0.0
	# 	total_count = 0.0

	# 	for j, data in enumerate(train_loader):
	# 		feat, labels = data
	# 		if not gpu:
	# 			true.extend(np.where(labels == 1))

	# 		# if torch.cuda.is_available():
	# 		# feat, labels = feat.to(device), labels.to(device)
	# 		# true.extend(labels.cpu().numpy())

	# 		optimizer.zero_grad()
	# 		predict = model(feat).float()

	# 		# _, predicted = torch.max(predict.data, 1)
	# 		# if not gpu:
	# 		# 	pred.extend(predicted.numpy())
	# 		# else:
	# 		# 	pred.extend(predicted.cpu().numpy())

	# 		loss = loss_func(predict.squeeze(), labels.long())

	# 		loss.backward()
	# 		optimizer.step()

	# 		# ipdb.set_trace()
			
	# 		train_acc += (predict.max(1)[1].float() == labels.max().float()).sum().float().item()
	# 		total_count += args.batch_size
	# 		train_loss += loss.item()

	# 	running_accuracy.append(train_acc / total_count)
	# 	running_loss.append(train_loss / float(j + 1))
	# 	nRec.append(epoch)

	# 	if epoch % args.eval_every == args.eval_every - 1:
	# 		# print("Epoch: %d | Training accuracy: %f | Training loss: %f | Val accuracy: %f | Val loss: %f"
	# 		# % (epoch, running_accuracy[-1], running_loss[-1], running_valid_accuracy[-1], running_valid_loss[-1]))
	# 		print("Epoch: %d | Training accuracy: %f | Training loss: %f"
	# 		% (epoch + 1, running_accuracy[-1], running_loss[-1]))


	# end = time()
	# print("====FINAL VALUES====")
	# # print("Test accuracy: %f | Test loss: %f" % (test_accuracy, test_loss))
	# print("Training acc: %f" % (max(running_accuracy)))
	# print("Training loss: %f" % (min(running_loss)))

	# # print(confusion_matrix(true, pred))
	# # print(instruments_list)

	# fig = plt.figure()
	# ax = plt.subplot(1, 2, 1)
	# plt.plot(nRec, running_accuracy, label='Training')
	# plt.title('Training Accuracy vs. Epoch')
	# plt.xlabel("Epoch")
	# plt.ylabel("Accuracy")
	# ax.legend()

	# bx = plt.subplot(1, 2, 2)
	# bx.plot(nRec, running_loss, label='Training')
	# plt.title('Training Loss vs. Epoch')
	# plt.xlabel("Epoch")
	# plt.ylabel("Loss")
	# bx.legend()
	# plt.show()
	# plt.savefig("baseline.png")
	# plt.clf()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=20)
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--epochs', type=int, default=30)
	parser.add_argument('--eval_every', type=int, default=3)

	args = parser.parse_args()

	main(args)

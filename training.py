""" Model
This model assumes preprocessing to be done already.
"""
import torch 
import models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from random import randrange
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.autograd.variable import Variable

from utils import train_load, dev_load, test_load, EER

def fixed_length(array, max_length):
    x = []
    for i in range(array.shape[0]):
        if array[i].shape[0] >= max_length:
            # too long, we slice
            a = randrange(array[i].shape[0] - max_length + 1)
            b = a + max_length
            sliced = array[i][a:b, :]
            sliced = np.roll(sliced, randrange(max_length), axis=0)
            x.append(sliced)
        else:
            # too short, we pad
            pad_width = ((0, max_length - array[i].shape[0]), (0,0))
            padded = np.pad(array[i], pad_width, 'wrap')
            padded = np.roll(padded, randrange(max_length), axis=0)
            x.append(padded)
    return np.array(x)

class TrainDataset(Dataset):
    def __init__(self, path, parts, max_length):
        features, speakers, nspeakers = train_load(path, parts)
        # data and target
        self.data = fixed_length(features, max_length)
        self.labels = speakers
        # save as tensors
        self.data = torch.Tensor(self.data)
        self.labels = torch.Tensor(self.labels)
        # number of speakers, number of data points
        self._nspeak = nspeakers
        self._len = self.data.shape[0]
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
    
class ValDataset(Dataset):
    def __init__(self, path, max_length):
        trials, labels, enrol, test = dev_load(path)
        self.trials = trials
        self.X1 = fixed_length(enrol, max_length)
        self.X2 = fixed_length(test, max_length)
        self.labels = labels

    def __len__(self):
        return self.trials.shape[0]
    
    def __getitem__(self, i):
        a, b = self.trials[i]
        return (self.X1[a], self.X2[b]), self.labels[i]

def load_data(parts, max_length):
    print('Loading training dataset..')
    train_dataset = TrainDataset('dataset', parts, max_length)
    print('Loading validation dataset..')
    val_dataset = ValDataset('dataset/dev.preprocessed.npz', max_length)
    return train_dataset, val_dataset

class Trainer:
    def __init__(self, train_loader, val_loader, name, net, optimizer, criterion, scheduler):
        print('Loading Trainer class for {}. '.format(name))
        # save the loaders
        self.update_data(train_loader, val_loader)
        # update training model
        self.update_model(name, net, optimizer, criterion, scheduler)
        # check GPU availability
        self.gpu = torch.cuda.is_available()
        print('Using GPU' if self.gpu else 'Not using GPU')
    
    def save_model(self):
        torch.save(self.net.state_dict(), 'models/{}'.format(self.name))
    
    def update_data(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def update_model(self, name, net, optimizer, criterion, scheduler):
        self.net = net
        self.name = name 
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
    
    def train(self, epochs):
        print('Start training {}.'.format(self.name))

        # move network to GPU if possible
        self.net = self.net.cuda() if self.gpu else self.net 

        for epoch in range(epochs):

            if self.scheduler:
                self.scheduler.step()

            train_num = 0
            train_loss = 0
            train_correct = 0

            for batch_i, (batch_data, batch_labels) in enumerate(self.train_loader):
                
                # reset optimizer gradients to zero
                self.optimizer.zero_grad()

                # initialize the data as variables
                batch_data = Variable(batch_data)
                batch_labels = Variable(batch_labels.type(torch.LongTensor))

                # move data to GPU if possible
                batch_data = batch_data.cuda() if self.gpu else batch_data
                batch_labels = batch_labels.cuda() if self.gpu else batch_labels

                # forward pass of the data
                batch_output = self.net(batch_data)

                # evaluate the prediction and correctness
                batch_prediction = batch_output.data.max(1, keepdim = True)[1]
                batch_prediction = batch_prediction.eq(batch_labels.data.view_as(batch_prediction))
                train_correct += batch_prediction.sum()
                train_num += batch_data.data.shape[0]

                # compute the losss
                batch_loss = self.criterion(batch_output, batch_labels)
                
                # backward pass and optimizer step
                batch_loss.backward()
                self.optimizer.step()

                # sum up this batch's loss
                train_loss += batch_loss.data.item()

                # print training progress
                if batch_i % 10 == 0:
                    print('\rEpoch {:3} Progress {:7.2%} Accuracy {:7.2%}'.format(
                        epoch + 1, 
                        batch_i * self.train_loader.batch_size / len(self.train_loader.dataset),
                        train_correct.cpu().item() / ((batch_i + 1) * self.train_loader.batch_size)
                    ), end='')

            # compute epoch loss and accuracy
            train_loss = train_loss / train_num
            train_accuracy = train_correct.cpu().item() / train_num

            # print summary for this epoch
            print('\rEpoch {:3} finished.\t\t\t\nTraining Accuracy: {:5.2%}\nTraining Loss: {:10.7f}'.format(
                epoch + 1, 
                train_accuracy, 
                train_loss
            ))

            """ Ignore validation data for now

            ##############################
            # VALIDATION DATA
            ##############################

            val_num = 0
            val_loss = 0
            val_correct = 0

            for batch_i, (batch_datas, batch_labels) in enumerate(self.val_loader):

                batch_data1, batch_data2 = batch_datas

                # initialize the data as variables
                batch_data1 = Variable(batch_data1.view(-1, self.train_loader.dataset.el_length))
                # move data to GPU if possible
                batch_data1 = batch_data1.cuda() if self.gpu else batch_data1
                # forward pass of the data
                batch_output1 = self.net(batch_data1)

                # initialize the data as variables
                batch_data2 = Variable(batch_data2.view(-1, self.train_loader.dataset.el_length))
                # move data to GPU if possible
                batch_data2 = batch_data2.cuda() if self.gpu else batch_data2
                # forward pass of the data
                batch_output2 = self.net(batch_data2)

                # labels
                batch_labels = Variable(batch_labels)
                batch_labels = batch_labels.cuda() if self.gpu else batch_labels



                # evaluate the prediction and correctness
                batch_prediction = batch_output.data.max(1, keepdim = True)[1]
                batch_prediction = batch_prediction.eq(batch_labels.data.view_as(batch_prediction))
                val_correct += batch_prediction.sum()
                val_num += batch_data.data.shape[0]

                # compute the losss
                batch_loss = self.criterion(batch_output, batch_labels)

                # sum up this batch's loss
                val_loss += batch_loss.data.item()

            # compute validation loss and accuracy
            val_loss = val_loss / val_num
            val_accuracy = val_correct.cpu().item() / val_num

            # print validation stats
            print('Validation Accuracy: {:5.2%}\nValidation Loss: {:10.7f}'.format(
                val_accuracy, 
                val_loss
            ))
            """

            torch.save(self.net, 'models/{}_{}'.format(self.name, epoch))

        # move network back to CPU if needed
        self.net = self.net.cpu() if self.gpu else self.net 

def main():
    # parameters
    max_length = 14000
    batch_size = 16
    parts = [1]
    epochs = 50

    # datasets and loaders
    print('Loading datasets')
    train_dataset, val_dataset = load_data(parts, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=RandomSampler(val_dataset))

    # model
    net = models.Resnet(train_dataset._nspeak, alpha=16, frames=max_length)
    
    # initialization
    # net.apply(nn.init.kaiming_normal_)

    # training parameters
    optimizer = torch.optim.SGD(net.parameters(), nesterov=True, momentum=0.01, dampening=0, lr=0.01, weight_decay=0.01)
    scheduler = None
    criterion = nn.modules.loss.CrossEntropyLoss()

    # initialize trainer
    trainer = Trainer(train_loader, val_loader, 'resnet', net, optimizer, criterion, scheduler)

    # run the training
    trainer.train(epochs)

if __name__ == '__main__':
    main()
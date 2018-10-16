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

from utils import train_load, dev_load, test_load, EER, fixed_length

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
        self.X1 = torch.Tensor(self.X1)
        self.X2 = torch.Tensor(self.X2)
        self.llabels = self.labels.astype(int)
        self.llabels = torch.Tensor(self.llabels)

    def __len__(self):
        return self.trials.shape[0]
    
    def __getitem__(self, i):
        a, b = self.trials[i]
        return (self.X1[a], self.X2[b]), self.llabels[i]

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
        print('Start training {}. # batches: {}'.format(self.name, len(self.train_loader)))

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

                # forward pass of the data and compute loss
                batch_output = self.net(batch_data)
                batch_loss = self.criterion(batch_output, batch_labels)
                
                # evaluate the prediction
                batch_prediction = torch.max(batch_output, 1)[1]

                # backward pass
                batch_loss.backward()
                self.optimizer.step()

                # train stats
                train_correct += (batch_prediction == batch_labels).sum().cpu().item()
                train_num += batch_data.data.shape[0]
                train_loss += batch_loss.data.item()

                # print training progress
                if batch_i % 10 == 0:
                    print('\rEpoch {:3} Progress {:7.2%} Accuracy {:7.2%} Loss {:7.4f}'.format(
                        epoch + 1, 
                        (batch_i + 1) / len(self.train_loader),
                        train_correct / train_num,
                        train_loss / (batch_i + 1)
                    ), end='')

            # compute epoch loss and accuracy
            train_loss = train_loss / len(self.train_loader)
            train_accuracy = train_correct / train_num

            # print summary for this epoch
            print('\rEpoch {:3} Progress {:7.2%} Accuracy {:7.2%} Loss {:7.4f}'.format(
                epoch + 1, 
                1,
                train_accuracy,
                train_loss
            ))

            epoch_scores = []

            # validation
            for batch_i, (batch_data, _) in enumerate(self.val_loader):

                batch_data_a, batch_data_b = batch_data

                batch_data_a = Variable(batch_data_a)
                batch_data_b = Variable(batch_data_b)

                batch_data_a = batch_data_a.cuda() if self.gpu else batch_data_a
                batch_data_b = batch_data_b.cuda() if self.gpu else batch_data_b

                # computing the speaker embeddings
                batch_output_a = self.net(batch_data_a)
                batch_output_b = self.net(batch_data_b)

                # compute the similarity score
                similarity_scores = nn.functional.cosine_similarity(batch_output_a, batch_output_b, dim=1)
                epoch_scores.append(similarity_scores.cpu().detach().numpy())

                print('\rVal {:04}/{:04}'.format(
                    batch_i, 
                    len(self.val_loader)
                ), end='')

                if batch_i > 100:
                    break
            
            print('\rCompute EER', end='')
            # get all scores and labels
            epoch_scores = np.concatenate(epoch_scores, axis=0)

            # compute the EER
            eer, tresh = EER(self.val_loader.dataset.labels, similarity_scores)

            print('\rValid {:3} EER {:7.4f} Tresh {:7.4f}'.format(
                epoch + 1,
                eer, 
                tresh
            ))
            
            torch.save(self.net, 'models/{}_{}'.format(self.name, epoch))

        # move network back to CPU if needed
        self.net = self.net.cpu() if self.gpu else self.net 

def main():
    # parameters
    max_length = 14000
    batch_size = 16
    parts = [1]
    epochs = 50
    init_fn = nn.init.kaiming_normal_

    # datasets and loaders
    train_dataset = TrainDataset('dataset', parts, max_length)
    val_dataset = ValDataset('dataset/dev.preprocessed.npz', max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=batch_size//2, sampler=RandomSampler(val_dataset))

    # model
    net = models.Resnet(train_dataset._nspeak, alpha=16)
    
    # initialization
    for layer in net.children():
        if isinstance(layer, nn.Conv2d):
            init_fn(layer.weight)
        elif isinstance(layer, models.ResidualBlock):
            for llayer in layer.children():
                if isinstance(llayer, nn.Conv2d):
                    init_fn(llayer.weight)

    # training parameters
    optimizer = torch.optim.SGD(net.parameters(), nesterov=True, momentum=0.01, dampening=0, lr=0.01, weight_decay=0.01)
    scheduler = None
    criterion = nn.modules.loss.CrossEntropyLoss()

    # initialize trainer
    name = 'resnet'
    trainer = Trainer(train_loader, val_loader, name, net, optimizer, criterion, scheduler)

    # run the training
    trainer.train(epochs)

if __name__ == '__main__':
    main()
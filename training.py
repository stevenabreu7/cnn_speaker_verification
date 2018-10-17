""" Model
This model assumes preprocessing to be done already.
"""
import os
import torch 
import argparse
import numpy as np
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models import ResidualBlock, Resnet, Flatten
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.autograd.variable import Variable

from utils import train_load, dev_load, test_load, EER, fixed_length

class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : Name of the scalar
        value : value itself
        step :  training iteration
        """
        # Notice we're using the Summary "class" instead of the "tf.summary" public API.
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

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
    def __init__(self, train_loader, val_loader, name, net, optimizer, criterion, scheduler, logging=True):
        # save the loaders
        self.update_data(train_loader, val_loader)
        # update training model
        self.update_model(name, net, optimizer, criterion, scheduler)
        # check GPU availability
        self.gpu = torch.cuda.is_available()
        print('Using GPU' if self.gpu else 'Not using GPU')
        # validation
        self.validate = False
        # logging to tensorboard
        self.tLog = Logger('./logs/{}_tlog') if logging else None
    
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
        print('Start training {} in {} batches'.format(self.name, len(self.train_loader)))

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

            # save summary for this epoch to file
            with open('logs/{}_info.txt'.format(self.name), 'a') as f:
                f.write('Epoch {:3} Progress {:7.2%} Accuracy {:7.2%} Loss {:7.4f}\n'.format(
                epoch + 1, 
                1,
                train_accuracy,
                train_loss
            ))

            # tensorboard logging
            if self.tLog:
                # log scalar values to tensorboard
                self.tLog.log_scalar('loss', train_loss, epoch+1)
                self.tLog.log_scalar('acc', train_accuracy, epoch+1)
                # log parameter values and gradients
                for tag, value in self.net.named_parameters():
                    tag = tag.replace('.', '/')
                    self.tLog.log_histogram(tag, value.data.cpu().numpy(), epoch+1)
                    self.tLog.log_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

            # save model for this epoch
            torch.save(self.net, 'models/{}_{}'.format(self.name, epoch+1))

            if not self.val_loader or epoch == 0 or epoch % 10 != 0:
                continue

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

            print('\rCompute EER', end='')
            # get all scores and labels
            epoch_scores = np.concatenate(epoch_scores, axis=0)

            # compute the EER
            eer, tresh = EER(self.val_loader.dataset.labels, epoch_scores)

            print('\rValid {:3} EER {:7.4f} Tresh {:7.4f}'.format(
                epoch + 1,
                eer, 
                tresh
            ))

        # move network back to CPU if needed
        self.net = self.net.cpu() if self.gpu else self.net 

def parse_arguments():
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument(
        'name',
        help='model name.')
    parser.add_argument(
        '--existing', 
        nargs='?',
        help='path to existing model to load.')
    # data arguments
    parser.add_argument(
        '--maxlen', 
        type=int,
        default=14000,
        help='maximum number of frames in speech.')
    parser.add_argument(
        '--bsize', 
        type=int,
        default=16,
        help='Batch Size.')
    parser.add_argument(
        '--parts', 
        type=int,
        default=1,
        help='how many parts of the training data to use (1-6).')
    # learning arguments
    parser.add_argument(
        '--val',
        action='store_true',
        help='do validation')
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='number of epochs to run')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.01,
        help='momentum for SGD')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate')
    parser.add_argument(
        '--wdecay',
        type=float,
        default=0.01,
        help='weight decay')
    return parser.parse_args()

def save_model_info(args):
    # save the model parameters to a file
    info =  'Model name:    {}\n'.format(args.name)
    info += 'Max length:    {}\n'.format(args.maxlen)
    info += 'Batch size:    {}\n'.format(args.bsize)
    info += 'Parts used:    {}\n'.format(args.parts)
    info += 'Epochs:        {}\n'.format(args.epochs)
    info += 'Learning rate: {}\n'.format(args.lr)
    info += 'Weight decay:  {}\n'.format(args.wdecay)
    info += 'Momentum:      {}\n'.format(args.momentum)
    info += 'Existing:      {}\n'.format(args.existing)
    # make sure the model doesn't exist already
    if os.path.exists('logs/{}_info.txt'.format(args.name)):
        raise Exception('Model log file already exists. Choose a different name.')
    # write model information to file
    with open('logs/{}_info.txt'.format(args.name), 'w') as f:
        f.write(info)

def main():
    # parse arguments from command line
    args = parse_arguments()

    # save model info to file
    save_model_info(args)

    # general parameters
    name = args.name

    # data parameters
    max_length = args.maxlen
    batch_size = args.bsize
    parts = list(range(1, args.parts+1))
    
    # learning parameters
    val = args.val
    epochs = args.epochs
    lr = args.lr
    wdecay = args.wdecay
    momentum = args.momentum 

    # fixed parameter
    init_fn = nn.init.kaiming_normal_  

    # datasets and loaders
    train_dataset = TrainDataset('dataset', parts, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    if val:
        val_dataset = ValDataset('dataset/dev.preprocessed.npz', max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size//2, sampler=RandomSampler(val_dataset))
    else:
        val_loader = None

    # model
    if args.existing:
        net = torch.load(args.existing)
    else:
        net = Resnet(train_dataset._nspeak, alpha=16)
    
    # initialization
    for layer in net.children():
        if isinstance(layer, nn.Conv2d):
            init_fn(layer.weight)
        elif isinstance(layer, ResidualBlock):
            for llayer in layer.children():
                if isinstance(llayer, nn.Conv2d):
                    init_fn(llayer.weight)

    # training parameters
    optimizer = torch.optim.SGD(net.parameters(), nesterov=True, momentum=momentum, dampening=0, lr=lr, weight_decay=wdecay)
    scheduler = None
    criterion = nn.modules.loss.CrossEntropyLoss()

    # initialize trainer
    trainer = Trainer(train_loader, val_loader, name, net, optimizer, criterion, scheduler)

    # run the training
    trainer.train(epochs)

if __name__ == '__main__':
    main()
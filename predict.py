import torch 
import argparse
import numpy as np
import torch.nn as nn
from utils import test_load, fixed_length
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler

# parameters
max_length = 14000

class TestDataset(Dataset):
    def __init__(self, path, max_length):
        trials, enrol, test = test_load(path)
        self._len = len(trials)
        self.trials = trials
        self.enrol = fixed_length(enrol, max_length)
        self.test = fixed_length(test, max_length)
        self.enrol = torch.Tensor(self.enrol)
        self.test = torch.Tensor(self.test)
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, i):
        a, b = self.trials[i]
        return self.enrol[a], self.test[b]

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Path to the model')
args = parser.parse_args()

# load the model
print('\rLoading model', end='')
net = torch.load(args.filename)
net = net.cpu()

# load data
print('\rLoading test data', end='')
test_dataset = TestDataset('dataset/test.preprocessed.npz', max_length)
test_loader = DataLoader(test_dataset, batch_size=8)

# GPU
gpu = torch.cuda.is_available()
net = net.cuda() if gpu else net 

scores = []

for batch_i, (batch_enrol, batch_test) in enumerate(test_loader):
    
    batch_enrol = Variable(batch_enrol)
    batch_test = Variable(batch_test)

    batch_enrol = batch_enrol.cuda() if gpu else batch_enrol
    batch_test = batch_test.cuda() if gpu else batch_test

    batch_enrol_out = net(batch_enrol)
    batch_test_out = net(batch_test)

    similarity_scores = nn.functional.cosine_similarity(batch_enrol_out, batch_test_out, dim=1)
    scores.append(similarity_scores.cpu().detach().numpy())

    print('\rTesting Progress {:04}/{:04}'.format(
        batch_i, 
        len(test_loader)
    ), end='')

scores = np.concatenate(scores)
print('Score array with shape:', scores.shape)
np.save('pred_scores', scores)
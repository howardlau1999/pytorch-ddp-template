import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel

class FooModel(nn.Module):
    def __init__(self) -> None:
        super(FooModel, self).__init__()
        self.net1 = nn.Linear(10, 10)  # (batch_size, 10) -> (batch_size, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)  # (batch_size, 10) -> (batch_size, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
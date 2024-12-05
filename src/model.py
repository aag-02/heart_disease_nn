import torch.nn as nn
import torch.nn.functional as F

class TheNet(nn.Module):
    def __init__(self):
        super(TheNet, self).__init__()
        self.input = nn.Linear(13, 32)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 10)
        self.output = nn.Linear(10, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.output(x)


# Copilot: Generate a simple feedforward neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F


class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        output = self.fc2(hidden)
        return output


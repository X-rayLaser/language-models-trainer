import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, num_classes, hidden_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_classes, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden_states=None):
        x, states = self.lstm(x, hidden_states)
        return self.linear(x), states

    def sample(self, start_seq, encoder, is_done):
        hidden_states = torch.zeros(1, self.hidden_dim), torch.zeros(1, self.hidden_dim)
        for x in start_seq:
            outputs, hidden_states = self.lstm(x, hidden_states)

        while True:
            x = outputs
            _, hidden_states = self.lstm(x, hidden_states)
            is_done()

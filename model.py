from torch import nn
from torch.nn import functional as F


class Net(nn.Module):
    def __init__(self, num_classes, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(num_classes, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)

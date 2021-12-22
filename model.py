from torch import nn


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

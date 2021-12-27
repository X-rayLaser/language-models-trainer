import json
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, num_classes, hidden_dim=32):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_classes, hidden_dim, num_layers=2, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden_states=None):
        x, states = self.lstm(x, hidden_states)
        return self.linear(x), states

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_saved_model(cls, path, model_params):
        model = cls(**model_params)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model

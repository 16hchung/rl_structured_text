import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, state_dim=768, action_dim=1): # TODO incorporate device
        super(QNet, self).__init__()
        self.hidden_layers = [300, 150, 50]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = nn.Sequential(
                nn.Linear(self.state_dim + self.action_dim, self.hidden_layers[0], bias=True), 
                nn.ReLU(),
                nn.Linear(self.hidden_layers[0], self.hidden_layers[1], bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_layers[1], self.hidden_layers[2], bias=True),
                nn.ReLU(),
                nn.Linear(self.hidden_layers[2], 1, bias=True)
        )

    def forward(self, hidden, action):
        input = torch.cat((hidden, action), dim=-1)
        output = self.model(input)
        return output


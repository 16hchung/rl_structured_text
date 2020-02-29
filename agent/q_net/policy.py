import torch
import torch.nn
import torch.nn.functional as F
import random

class Policy(nn.Module):
    def __init__(self, state_dim=768, action_dim=5):
        super(Policy, self).__init__()
        self.hidden_layers=[300, 150, 50]
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.optimizer=torch.optim.Adam(lr=1e-3)
        self.criterion=torch.nn.CrossEntropyLoss()
        self.model=nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_layers[0], bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[0], self.hidden_layers[1], bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[1], self.hidden_layers[2], bias=True),
            nn.Linear(self.hidden_layers[2], self.action_dim, bias=True)
        )

        def forward(self, state):
            action_logits = self.model(state)
            return action_logits

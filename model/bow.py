import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class bow(BaseModel):

    def __init__(self, hidden_size, num_class):
        super(bow, self).__init__()
        self.fc1 = nn.Linear(hidden_size, int(hidden_size / 16))
        self.fc2 = nn.Linear(int(hidden_size / 16), int(hidden_size / 32))
        self.fc3 = nn.Linear(int(hidden_size / 32), num_class)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x
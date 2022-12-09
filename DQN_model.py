import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(10, 16, (3, 3), padding='same'), nn.ReLU(),  # Layer 1
            nn.Conv2d(16, 32, (3, 3), padding='same'), nn.ReLU(),  # Layer 2
            nn.Conv2d(32, 64, (5, 5), padding='same'), nn.ReLU(),  # Layer 3
            nn.Flatten(),
            nn.Linear(1280, 64), nn.ReLU(),  # Layer 4
            nn.Linear(64, 4), nn.ReLU()  # Layer 5 (Output)
        )

    def forward(self, x):
        return self.conv_net(x)




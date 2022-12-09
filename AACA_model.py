import torch.nn as nn


class AACA_model_logits(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(10, 16, (3, 3), padding='same'), nn.ReLU(),  # Layer 1
            nn.Conv2d(16, 32, (3, 3), padding='same'), nn.ReLU(),  # Layer 2
            nn.Flatten(),
            nn.Linear(640, 64), nn.ReLU(),  # Layer 3
            nn.Linear(64, 4)  # Layer 4 (Output)
        )

    def forward(self, x):
        return self.conv_net(x)


class AACA_model_values(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(10, 16, (3, 3), padding='same'), nn.ReLU(),  # Layer 1
            nn.Conv2d(16, 32, (3, 3), padding='same'), nn.ReLU(),  # Layer 2
            nn.Flatten(),
            nn.Linear(640, 64), nn.ReLU(),  # Layer 3
            nn.Linear(64, 1)  # Layer 4 (Output)
        )

    def forward(self, x):
        return self.conv_net(x)




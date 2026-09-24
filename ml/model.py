"""A small CNN (convolutional neural network) for classifying spectrograms."""

import torch.nn as nn


class AudioCNN(nn.Module):
    """
    Input:  (batch, 1, 64, 126)  -- one spectrogram per item, in this shape
    Output: (batch, num_classes) -- one score per class, per item

    Two "convolution -> shrink" blocks pull out patterns (edges, bursts,
    steady tones) at increasing scale, then two fully connected layers turn
    those patterns into a final decision.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (64, 126) -> (32, 63)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (32, 63) -> (16, 31)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 31, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # randomly ignore 30% of connections while training, to reduce overfitting
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
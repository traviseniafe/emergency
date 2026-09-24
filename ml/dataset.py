# Wraps the manifest so PyTorch can load spectrograms one at a time during training.

import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import audio_file_to_features

# Fixed order, so label 0 always means "background" everywhere in the project:
# in this file, in the model's output, and in any report we print later.

LABELS = ["background", "explosion", "gunshot", "siren"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


class AudioDataset(Dataset):

    # A PyTorch Dataset: something that knows how many items it has (__len__)
    # and how to fetch one of them by position (__getitem__). PyTorch calls
    # these two methods itself, in the background, while training.
    

    def __init__(self, manifest: pd.DataFrame):
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]

        spectrogram = audio_file_to_features(row["path"])
        # Add a "channel" dimension: a CNN expects (channels, height, width),
        # the same shape a colour image would have, but we only have 1 channel.
        features = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        label = LABEL_TO_INDEX[row["label"]]
        return features, label
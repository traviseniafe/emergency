"""Train the audio CNN and check how well it performs.

Run it directly:
    python3 -m ml.train
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import AudioDataset, LABELS
from .master import build_manifest
from .model import AudioCNN

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.1  # held out from the training clips, to watch for overfitting
RANDOM_SEED = 42

MODEL_PATH = "models/audio_cnn.pth"
LABELS_PATH = "models/labels.json"
CONFUSION_MATRIX_PATH = "reports/confusion_matrix.png"


def make_loaders(manifest):
    """Split the manifest into train / validation / test DataLoaders."""
    train_rows = manifest[manifest["split"] == "train"]
    test_rows = manifest[manifest["split"] == "test"]

    # stratify=... keeps the same proportion of each class in both halves,
    # rather than risking a split with too few examples of a rare class.
    train_rows, val_rows = train_test_split(
        train_rows,
        test_size=VALIDATION_FRACTION,
        stratify=train_rows["label"],
        random_state=RANDOM_SEED,
    )

    train_loader = DataLoader(AudioDataset(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AudioDataset(val_rows), batch_size=BATCH_SIZE)
    test_loader = DataLoader(AudioDataset(test_rows), batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


def run_one_epoch(model, loader, loss_function, optimizer, device):
    """Train for one pass over the data. optimizer=None means evaluate only."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, correct, total = 0.0, 0, 0
    # No gradient tracking needed during evaluation: it saves memory and time.
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)

            predictions = model(features)
            loss = loss_function(predictions, labels)

            if is_training:
                optimizer.zero_grad()  # clear gradients left over from the last batch
                loss.backward()        # work out how each weight contributed to the error
                optimizer.step()       # nudge each weight to reduce that error

            total_loss += loss.item() * len(labels)
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def evaluate_on_test_set(model, test_loader, device):
    """Run the finished model on the held-out test set and report per-class results."""
    model.eval()
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            predictions = model(features).argmax(dim=1).cpu()
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    print("\nTest set results:")
    print(classification_report(all_labels, all_predictions, target_names=LABELS, zero_division=0))

    matrix = confusion_matrix(all_labels, all_predictions, labels=range(len(LABELS)))
    display = ConfusionMatrixDisplay(matrix, display_labels=LABELS)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Test set confusion matrix")
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {CONFUSION_MATRIX_PATH}")


def train(manifest):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, val_loader, test_loader = make_loaders(manifest)

    model = AudioCNN(num_classes=len(LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_accuracy = run_one_epoch(model, val_loader, loss_function, None, device)
        print(
            f"Epoch {epoch:2d}/{EPOCHS}  "
            f"train loss {train_loss:.3f} acc {train_accuracy:.1%}  |  "
            f"val loss {val_loss:.3f} acc {val_accuracy:.1%}"
        )

    evaluate_on_test_set(model, test_loader, device)

    torch.save(model.state_dict(), MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(LABELS, f)
    print(f"Saved {MODEL_PATH} and {LABELS_PATH}")


if __name__ == "__main__":
    train(build_manifest())
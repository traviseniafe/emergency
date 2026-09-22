# Build one master list of every audio clip, with its label and dataset split.

# Run it directly to print a summary:
#     python3 -m ml.manifest
#

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"

# How each dataset's own class names map onto our four labels.

URBANSOUND_LABELS = {
    "gun_shot": "gunshot",
    "siren": "siren",
    # Every other UrbanSound8K class is everyday background noise.Not classified as SESA.
    "air_conditioner": "background",
    "car_horn": "background",
    "children_playing": "background",
    "dog_bark": "background",
    "drilling": "background",
    "engine_idling": "background",
    "jackhammer": "background",
    "street_music": "background",
}

SESA_LABELS = {
    "gunshot": "gunshot",
    "explosion": "explosion",
    "siren": "siren",
    "casual": "background",
}

# UrbanSound8K ships 10 predefined folds. We use fold 10 as our test set and
# folds 1-9 for training. This follows the dataset's own rule: no
# re-split of folds randomly and clip leaks between train and test.

URBANSOUND_TEST_FOLD = 10


def load_urbansound8k() -> pd.DataFrame:
    """Read UrbanSound8K's metadata file and turn it into our manifest format."""
    root = DATA_DIR / "UrbanSound8K"
    metadata = pd.read_csv(root / "metadata" / "UrbanSound8K.csv")

    paths = [
        root / "audio" / f"fold{fold}" / filename
        for fold, filename in zip(metadata["fold"], metadata["slice_file_name"])
    ]
    labels = metadata["class"].map(URBANSOUND_LABELS)
    splits = ["test" if fold == URBANSOUND_TEST_FOLD else "train" for fold in metadata["fold"]]

    return pd.DataFrame({"path": paths, "label": labels, "split": splits, "source": "urbansound8k"})


def load_sesa() -> pd.DataFrame:
    """SESA has no metadata file: each filename starts with its class name."""
    root = DATA_DIR / "SESA"
    rows = []
    for split, folder_name in [("train", "train"), ("test", "test")]:
        for path in sorted((root / folder_name).glob("*.wav")):
            class_name = path.stem.split("_")[0]  # "gunshot_041" -> "gunshot"
            rows.append({"path": path, "label": SESA_LABELS[class_name], "split": split, "source": "sesa"})
    return pd.DataFrame(rows)


def build_manifest() -> pd.DataFrame:
    # Combine both datasets into one manifest, and drop any file that's missing.
    manifest = pd.concat([load_urbansound8k(), load_sesa()], ignore_index=True)

    exists = manifest["path"].map(lambda p: p.exists())
    missing = manifest[~exists]
    if len(missing):
        print(f"Warning: {len(missing)} files listed but not found on disk. Example: {missing.iloc[0]['path']}")
    return manifest[exists].reset_index(drop=True)


def print_summary(manifest: pd.DataFrame):
    print(f"Total clips: {len(manifest)}\n")
    print("By label and split:")
    print(manifest.groupby(["label", "split"]).size().unstack(fill_value=0))
    print("\nBy source and split:")
    print(manifest.groupby(["source", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    print_summary(build_manifest())
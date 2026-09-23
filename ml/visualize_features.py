# Save a picture showing one example spectrogram per label, so you can see
# with your own eyes what the model will actually be learning from.
# Run it after ml/manifest.py works:
# python3 -m ml.visualize_features


import matplotlib
matplotlib.use("Agg")  # write straight to a file; don't try to open a window
import matplotlib.pyplot as plt

from ml.features import audio_file_to_features
from ml.master import build_manifest

OUTPUT_PATH = "reports/spectrograms.png"
LABELS_IN_ORDER = ["background", "gunshot", "explosion", "siren"]


def main():
    manifest = build_manifest()

    fig, axes = plt.subplots(1, len(LABELS_IN_ORDER), figsize=(14, 4))
    for axis, label in zip(axes, LABELS_IN_ORDER):
        matches = manifest[manifest["label"] == label]
        if matches.empty:
            axis.set_title(f"{label}\n(no examples found)")
            axis.axis("off")
            continue

        example_path = matches.iloc[0]["path"]
        spectrogram = audio_file_to_features(example_path)

        image = axis.imshow(spectrogram, origin="lower", aspect="auto", cmap="magma")
        axis.set_title(label)
        axis.set_xlabel("time")
    axes[0].set_ylabel("mel frequency band")
    fig.colorbar(image, ax=axes, label="dB", fraction=0.02)

    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
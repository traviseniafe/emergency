import os
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------- FIXED PATHS --------------------
ANNOTATION_FILE = "c:/Users/User/Documents/Project work/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "c:/Users/User/Documents/Project work/UrbanSound8K/audio"
MODEL_SAVE_PATH = "c:/Users/User/Documents/Project work/audio_cnn.pth"

# -------------------- CONFIG --------------------
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
TARGET_SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- DATASET --------------------
class UrbanSoundDataset(Dataset):
    def __init__(self, annotation_file, audio_dir, transformation=None, target_sample_rate=22050, num_samples=22050):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.transformation = transformation or torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.max_len = self._find_max_len()

    def _find_max_len(self):
        max_len = 0
        print("🔍 Detecting max spectrogram length...")
        for i in range(len(self.annotations)):
            path = self._get_audio_path(i)
            try:
                signal, sr = torchaudio.load(path)
                if sr != self.target_sample_rate:
                    signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)(signal)
                if signal.shape[0] > 1:
                    signal = torch.mean(signal, dim=0, keepdim=True)
                signal = self._pad_or_trim(signal)
                mel_spec = self.transformation(signal)
                if mel_spec.shape[-1] == 0:
                    raise ValueError("Spectrogram has zero time dimension")
                max_len = max(max_len, mel_spec.shape[-1])
            except Exception as e:
                print(f"⚠️ Skipping {path}: {e}")
        print(f"✅ Max spectrogram time dimension: {max_len}")
        return max_len if max_len > 0 else 44  # Safe fallback

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = self._get_audio_path(index)
        label = self.annotations.iloc[index]['classID']
        try:
            signal, sr = torchaudio.load(path)
            if sr != self.target_sample_rate:
                signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)(signal)
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            signal = self._pad_or_trim(signal)
            mel_spec = self.transformation(signal)

            if mel_spec.shape[-1] == 0:
                raise ValueError("Zero-length spectrogram")

            time_dim = mel_spec.shape[-1]
            if time_dim < self.max_len:
                mel_spec = F.pad(mel_spec, (0, self.max_len - time_dim))
            elif time_dim > self.max_len:
                mel_spec = mel_spec[:, :, :self.max_len]
            return mel_spec, label
        except Exception as e:
            print(f"❌ Skipping {path}: {e}")
            return self.__getitem__((index + 1) % len(self))

    def _pad_or_trim(self, signal):
        if signal.shape[1] > self.num_samples:
            return signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            return F.pad(signal, (0, self.num_samples - signal.shape[1]))
        return signal

    def _get_audio_path(self, index):
        fold = f"fold{self.annotations.iloc[index]['fold']}"
        file = self.annotations.iloc[index]['slice_file_name']
        return os.path.join(self.audio_dir, fold, file)

# -------------------- MODEL --------------------
class AudioClassifierCNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        if input_size < 3:
            print(f"⚠️ input_size={input_size} is too small. Using fallback size 44.")
            input_size = 44
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.dropout = nn.Dropout(0.3)

        dummy_input = torch.randn(1, 1, 64, input_size)
        out = self._forward_features(dummy_input)
        self.flatten_dim = out.shape[1]
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 10)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------- TRAIN --------------------
def train():
    print("🔥 Training function has started")
    print("📦 Loading dataset...")
    dataset = UrbanSoundDataset(ANNOTATION_FILE, AUDIO_DIR)

    print(f"📊 Dataset loaded. Total samples: {len(dataset)}")

    for i in range(min(5, len(dataset))):
        try:
            spec, label = dataset[i]
            print(f"✅ Sample {i}: spectrogram shape = {spec.shape}, label = {label}")
        except Exception as e:
            print(f"❌ Sample {i} failed: {e}")

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
   
    print("🧠 Building model...")
    model = AudioClassifierCNN(input_size=dataset.max_len).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("🚀 Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for mel_spec, labels in train_loader:
            mel_spec = mel_spec.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"📈 Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f}")
    print(f"💾 Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("✅ Training completed and model saved.")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("✅ Main execution started")
    train()

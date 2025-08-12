# ================== FINAL EMERGENCY DETECTION & TRAINING SYSTEM ==================
# Combines real-time multilingual emergency detection (audio, video, voice)
# with a CNN trained on UrbanSound8K (10-class audio emergency classifier)

import os
import cv2
import queue
import torch
import torchaudio
import pandas as pd
import numpy as np
import sounddevice as sd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import geocoder
import yagmail
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
from deep_translator import GoogleTranslator
from langdetect import detect
import speech_recognition as sr

# -------------------- CONFIG --------------------
BASE_DIR = r"c:/Users/User/Documents/Project work"
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "audio_cnn.pth")
ANNOTATION_FILE = os.path.join(BASE_DIR, "UrbanSound8K/metadata/UrbanSound8K.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "UrbanSound8K/audio")
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- AUDIO DATASET --------------------
class UrbanSoundDataset(data.Dataset):
    def __init__(self, annotation_file, audio_dir, transformation=None):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.transformation = transformation or torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
        self.max_len = self._find_max_len()

    def _find_max_len(self):
        max_len = 0
        for i in range(len(self.annotations)):
            path = self._get_audio_path(i)
            try:
                signal, sr = torchaudio.load(path)
                if sr != SAMPLE_RATE:
                    signal = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(signal)
                signal = signal.mean(dim=0, keepdim=True)
                signal = self._pad_or_trim(signal)
                mel_spec = self.transformation(signal)
                max_len = max(max_len, mel_spec.shape[-1])
            except: pass
        return max_len

    def __len__(self): return len(self.annotations)

    def __getitem__(self, index):
        path = self._get_audio_path(index)
        label = self.annotations.iloc[index]['classID']
        signal, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            signal = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(signal)
        signal = signal.mean(dim=0, keepdim=True)
        signal = self._pad_or_trim(signal)
        mel_spec = self.transformation(signal)
        mel_spec = F.pad(mel_spec, (0, self.max_len - mel_spec.shape[-1]))
        return mel_spec, label

    def _pad_or_trim(self, signal):
        if signal.shape[1] > NUM_SAMPLES:
            return signal[:, :NUM_SAMPLES]
        elif signal.shape[1] < NUM_SAMPLES:
            return F.pad(signal, (0, NUM_SAMPLES - signal.shape[1]))
        return signal

    def _get_audio_path(self, index):
        fold = f"fold{self.annotations.iloc[index]['fold']}"
        file = self.annotations.iloc[index]['slice_file_name']
        return os.path.join(self.audio_dir, fold, file)

# -------------------- AUDIO MODEL --------------------
class AudioClassifierCNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.dropout = nn.Dropout(0.3)

        dummy = torch.randn(1, 1, 64, input_size)
        dummy_out = self._forward_features(dummy)
        self.flatten_dim = dummy_out.shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 10)

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self._forward_features(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -------------------- LOAD MODEL --------------------
print("📥 Loading audio model...")
dataset = UrbanSoundDataset(ANNOTATION_FILE, AUDIO_DIR)
audio_model = AudioClassifierCNN(input_size=dataset.max_len).to(DEVICE)
audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=DEVICE))
audio_model.eval()

# -------------------- VOICE TRANSLATION --------------------
def recognize_and_translate():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("🎤 Listening...")
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            lang = detect(text)
            translation = GoogleTranslator(source='auto', target='en').translate(text)
            print(f"🗣️ {text} ({lang}) → {translation}")
            return translation.lower()
        except: return ""

# -------------------- REALTIME DETECTION --------------------
print("🛠️ Starting real-time emergency system...")
model_image = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
model_yolo = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status: print(status)
    audio_q.put(indata.copy())

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=NUM_SAMPLES)
stream.start()

emergency_keywords = ["gun", "fire", "siren", "help", "scream", "shout", "blood"]
location_data = geocoder.ip('me').json or {}

def send_alert():
    try:
        print("📨 Sending alert email...")
        yag = yagmail.SMTP("demoproj2025@gmail.com", "rsluuimxrjgdelsg")
        contents = f"""
        EMERGENCY DETECTED!
        City: {location_data.get('city')}
        Region: {location_data.get('region')}
        Country: {location_data.get('country')}
        Lat: {location_data.get('lat')}, Lng: {location_data.get('lng')}
        """
        yag.send("exa193@student.bham.ac.uk", "🚨 Emergency Alert", contents)
        print("✅ Alert sent.")
    except Exception as e:
        print(f"❌ Failed to send alert: {e}")

consec_count = 0
THRESH = 0.04
CONF_THRESH = 0.5

try:
    while True:
        # === AUDIO CNN CLASSIFICATION ===
        try:
            audio_data = audio_q.get_nowait().flatten()
        except queue.Empty:
            audio_data = np.zeros(NUM_SAMPLES)

        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        if audio_tensor.shape[1] > NUM_SAMPLES:
            audio_tensor = audio_tensor[:, :NUM_SAMPLES]
        elif audio_tensor.shape[1] < NUM_SAMPLES:
            audio_tensor = F.pad(audio_tensor, (0, NUM_SAMPLES - audio_tensor.shape[1]))

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
        )(audio_tensor.unsqueeze(0))

        mel_spec = F.pad(mel_spec, (0, dataset.max_len - mel_spec.shape[-1]))
        with torch.no_grad():
            out = audio_model(mel_spec.to(DEVICE))
        pred_class = torch.argmax(out, dim=1).item()
        is_emergency_audio = pred_class in [3, 4, 6, 7, 8]  # Adjust for relevant emergency classes

        # === IMAGE & YOLO DETECTION ===
        ret, frame = cap.read()
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(img_pil).unsqueeze(0)
        label_str = ResNet50_Weights.DEFAULT.meta['categories'][
            model_image(img_tensor).argmax().item()
        ].lower()

        yolo_labels = [box for r in model_yolo(frame) for box in r.boxes.cls.cpu().numpy()]
        yolo_names = model_yolo.names
        yolo_detected = [yolo_names[int(i)].lower() for i in yolo_labels]
        
        is_emergency_visual = any(k in label_str for k in emergency_keywords) or any(k in yolo_detected for k in emergency_keywords)

        # === VOICE TRANSLATION ===
        voice_text = recognize_and_translate()
        is_emergency_voice = any(k in voice_text for k in emergency_keywords)

        if is_emergency_audio or is_emergency_visual or is_emergency_voice:
            consec_count += 1
        else:
            consec_count = 0

        if consec_count >= 3:
            print("🚨 Emergency Confirmed!")
            send_alert()
            consec_count = 0
        else:
            print("✅ Normal State")

        cv2.imshow("Emergency Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    stream.stop()
    stream.close()

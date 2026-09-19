import torch
import torch.nn as nn
import numpy as np

MODEL_PATH = r"C:\LipReadingSSL\output\phase8_lipreading(รอบ1)\checkpoints\best_model.pt"
NPY_PATH = r"C:\LipReadingSSL\output\video001\clip_npy\clip_000001.npy"


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.net(x)
        return x.flatten(1)


class LipReadingModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2):
        super().__init__()

        self.visual_encoder = SpatialEncoder()

        self.temporal = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, x):
        # [B,T,1,H,W]
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)

        features = self.visual_encoder(x)

        features = features.reshape(B, T, -1)

        temporal, _ = self.temporal(features)

        return self.classifier(temporal)


# Load checkpoint
ckpt = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

itos = ckpt["itos"]

model = LipReadingModel(len(itos))

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()


# Load 29 frames
frames = np.load(NPY_PATH).astype(np.float32)

# EXACT Phase 8 preprocessing
frames = frames / 255.0
frames = (frames - 0.5) / 0.5

# [T,H,W] -> [B,T,1,H,W]
x = torch.from_numpy(frames).unsqueeze(0).unsqueeze(2)


with torch.no_grad():
    logits = model(x)


# CTC greedy decode
ids = torch.argmax(logits, dim=-1)[0].numpy()

result = []
previous = None

for idx in ids:
    idx = int(idx)

    if idx == 0:
        previous = idx
        continue

    if idx == previous:
        continue

    if idx < len(itos):
        token = itos[idx]

        if token != "<unk>":
            result.append(token)

    previous = idx


print("=" * 40)
print("PHASE 8 EXACT INFERENCE")
print("=" * 40)
print("Input :", tuple(x.shape))
print("Output:", tuple(logits.shape))
print("Text  :", repr("".join(result)))
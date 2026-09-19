import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    r"C:\LipReadingSSL\output\phase8_lipreading(รอบ1)\checkpoints\best_model.pt"
)

TEST_CSV = Path(
    r"C:\LipReadingSSL\output\dataset\export\test.csv"
)

DATASET_ROOT = Path(
    r"C:\LipReadingSSL\output"
)

NUM_CLIPS = 20


# ============================================================
# MODEL
# ============================================================

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
    def __init__(
        self,
        num_classes,
        hidden_size=256,
        num_layers=2
    ):
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

        x = x.reshape(
            B * T,
            C,
            H,
            W
        )

        features = self.visual_encoder(x)

        features = features.reshape(
            B,
            T,
            -1
        )

        temporal, _ = self.temporal(features)

        return self.classifier(temporal)


# ============================================================
# CTC DECODER
# ============================================================

def ctc_greedy_decode(logits, itos):
    """
    logits:
        [T, num_classes]

    return:
        decoded text
    """

    ids = torch.argmax(
        logits,
        dim=-1
    ).cpu().numpy()

    result = []

    previous = None

    for idx in ids:

        idx = int(idx)

        # CTC blank
        if idx == 0:
            previous = idx
            continue

        # Remove repeated token
        if idx == previous:
            continue

        if idx < len(itos):

            token = itos[idx]

            if token != "<unk>":
                result.append(token)

        previous = idx

    return "".join(result)


# ============================================================
# LOAD NPY
# ============================================================

def load_npy(npy_path):
    """
    Exact Phase 8 preprocessing:

    grayscale
    resize 96x96
    /255
    normalize to [-1,1]
    """

    frames = np.load(npy_path)

    frames = frames.astype(
        np.float32
    )

    # --------------------------------------------------------
    # NPY should already be 96x96.
    # Resize only if necessary.
    # --------------------------------------------------------

    if frames.ndim != 3:
        raise ValueError(
            f"Invalid NPY shape: {frames.shape}"
        )

    processed = []

    for frame in frames:

        if frame.shape != (96, 96):

            frame = cv2.resize(
                frame,
                (96, 96),
                interpolation=cv2.INTER_AREA
            )

        processed.append(frame)

    frames = np.stack(
        processed
    )

    # EXACT Phase 8 normalization
    frames = frames / 255.0

    frames = (
        frames - 0.5
    ) / 0.5

    # [T,H,W]
    # ->
    # [B,T,1,H,W]

    x = torch.from_numpy(
        frames
    ).unsqueeze(0).unsqueeze(2)

    return x


# ============================================================
# FIND NPY
# ============================================================

def find_npy(row):

    # --------------------------------------------------------
    # CSV contains:
    #
    # video001\clip_npy\clip_000001.npy
    #
    # --------------------------------------------------------

    clip_path = str(
        row["clip_path"]
    ).replace(
        "\\",
        "/"
    )

    candidate = (
        DATASET_ROOT /
        clip_path
    )

    if candidate.exists():
        return candidate

    # --------------------------------------------------------
    # Fallback using video_id + clip_id
    # --------------------------------------------------------

    video_id = row["video_id"]
    clip_id = row["clip_id"]

    candidate = (
        DATASET_ROOT /
        video_id /
        "clip_npy" /
        f"{clip_id}.npy"
    )

    if candidate.exists():
        return candidate

    return None


# ============================================================
# LOAD TEST CSV
# ============================================================

print("=" * 70)
print("MULTIPLE CLIPS TEST")
print("=" * 70)

print()
print("Model :", MODEL_PATH)
print("CSV   :", TEST_CSV)
print()


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"test.csv not found:\n{TEST_CSV}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("[1] Loading checkpoint...")

ckpt = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

itos = ckpt["itos"]

model = LipReadingModel(
    num_classes=len(itos)
)

model.load_state_dict(
    ckpt["model_state_dict"]
)

model.eval()

print(
    f"    Classes : {len(itos)}"
)

print(
    f"    Epoch   : {ckpt.get('epoch', '?')}"
)

print(
    f"    CER     : {ckpt.get('best_val_cer', '?')}"
)


# ============================================================
# READ TEST CSV
# ============================================================

print()
print("[2] Reading test.csv...")

rows = []

with open(
    TEST_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        rows.append(row)


print(
    f"    Test samples : {len(rows)}"
)


# ============================================================
# SELECT CLIPS
# ============================================================

rows = rows[:NUM_CLIPS]

print(
    f"    Testing      : {len(rows)} clips"
)


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)


success = 0
failed = 0


for i, row in enumerate(rows, start=1):

    clip_id = row.get(
        "clip_id",
        "unknown"
    )

    video_id = row.get(
        "video_id",
        "unknown"
    )

    # --------------------------------------------------------
    # Ground Truth
    #
    # Prefer normalized_text
    # then transcript
    # --------------------------------------------------------

    gt = (
        row.get("normalized_text")
        or row.get("transcript")
        or ""
    )

    gt = gt.strip()

    # --------------------------------------------------------
    # Find NPY
    # --------------------------------------------------------

    npy_path = find_npy(row)

    print()
    print(
        f"[{i:02d}/{len(rows):02d}] "
        f"{video_id} / {clip_id}"
    )

    print(
        f"NPY : {npy_path}"
    )

    if npy_path is None:

        print(
            "Pred: [NPY NOT FOUND]"
        )

        failed += 1
        continue

    try:

        # ----------------------------------------------------
        # Load input
        # ----------------------------------------------------

        x = load_npy(
            npy_path
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(x)

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        pred = ctc_greedy_decode(
            logits[0],
            itos
        )

        print(
            f"Shape: {tuple(x.shape)}"
        )

        print(
            f"GT   : {repr(gt)}"
        )

        print(
            f"Pred : {repr(pred)}"
        )

        success += 1

    except Exception as e:

        print(
            f"ERROR: {e}"
        )

        failed += 1


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Tested : {len(rows)}"
)

print(
    f"Success: {success}"
)

print(
    f"Failed : {failed}"
)

print()
print("Done.")
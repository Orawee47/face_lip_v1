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

PREDICTIONS_CSV = Path(
    r"C:\LipReadingSSL\output\phase8_lipreading(รอบ1)\results\test_predictions.csv"
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

            nn.Conv2d(
                1, 32, 3, padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64, 3, padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128, 3, padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
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

        temporal, _ = self.temporal(
            features
        )

        return self.classifier(
            temporal
        )


# ============================================================
# CTC DECODE
# ============================================================

def ctc_decode(logits, itos):

    ids = torch.argmax(
        logits,
        dim=-1
    ).cpu().numpy()

    result = []

    previous = None

    for idx in ids:

        idx = int(idx)

        # blank
        if idx == 0:

            previous = idx

            continue

        # repeated token
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

    frames = np.load(
        npy_path
    ).astype(
        np.float32
    )

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

    # ========================================================
    # EXACT PHASE 8 PREPROCESSING
    # ========================================================

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
# NORMALIZE TEXT
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    return text.strip()


# ============================================================
# FIND NPY
# ============================================================

def find_npy(row):

    video_id = row.get(
        "video_id",
        ""
    )

    clip_id = row.get(
        "clip_id",
        ""
    )

    # --------------------------------------------------------
    # Primary path
    # --------------------------------------------------------

    candidate = (
        DATASET_ROOT
        / video_id
        / "clip_npy"
        / f"{clip_id}.npy"
    )

    if candidate.exists():

        return candidate

    # --------------------------------------------------------
    # clip_path fallback
    # --------------------------------------------------------

    clip_path = row.get(
        "clip_path",
        ""
    )

    if clip_path:

        candidate = (
            DATASET_ROOT
            / Path(
                clip_path
            )
        )

        if candidate.exists():

            return candidate

    return None


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append(row)

    return rows


# ============================================================
# FIND PREDICTION ROW
# ============================================================

def make_prediction_lookup(rows):

    lookup = {}

    for row in rows:

        reference = clean_text(
            row.get(
                "reference",
                ""
            )
        )

        prediction = clean_text(
            row.get(
                "prediction",
                ""
            )
        )

        clip_id = clean_text(
            row.get(
                "clip_id",
                ""
            )
        )

        video_id = clean_text(
            row.get(
                "video_id",
                ""
            )
        )

        # ----------------------------------------------------
        # Possible identifiers
        # ----------------------------------------------------

        if clip_id:

            lookup[
                ("clip_id", clip_id)
            ] = (
                reference,
                prediction
            )

        if video_id and clip_id:

            lookup[
                ("video_clip", video_id, clip_id)
            ] = (
                reference,
                prediction
            )

    return lookup


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("VERIFY PHASE 8 TEST")
print("=" * 70)

print()
print("Model      :", MODEL_PATH)
print("test.csv   :", TEST_CSV)
print("predictions:", PREDICTIONS_CSV)
print()


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    MODEL_PATH,
    TEST_CSV,
    PREDICTIONS_CSV
]:

    if not path.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )


# ============================================================
# LOAD CHECKPOINT
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
# LOAD CSV FILES
# ============================================================

print()
print("[2] Loading CSV files...")

test_rows = load_csv(
    TEST_CSV
)

prediction_rows = load_csv(
    PREDICTIONS_CSV
)

print(
    f"    test.csv rows          : {len(test_rows)}"
)

print(
    f"    predictions.csv rows   : {len(prediction_rows)}"
)


# ============================================================
# SHOW COLUMNS
# ============================================================

print()
print("[3] Prediction CSV columns:")

if prediction_rows:

    print(
        "   ",
        list(prediction_rows[0].keys())
    )


# ============================================================
# BUILD LOOKUP
# ============================================================

prediction_lookup = (
    make_prediction_lookup(
        prediction_rows
    )
)


# ============================================================
# SELECT TEST CLIPS
# ============================================================

selected = test_rows[:NUM_CLIPS]


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)


matched_reference = 0
matched_prediction = 0
npy_success = 0
npy_failed = 0


for i, row in enumerate(
    selected,
    start=1
):

    video_id = clean_text(
        row.get(
            "video_id",
            ""
        )
    )

    clip_id = clean_text(
        row.get(
            "clip_id",
            ""
        )
    )

    # --------------------------------------------------------
    # Find prediction record
    # --------------------------------------------------------

    saved_reference = ""
    saved_prediction = ""

    key = (
        "video_clip",
        video_id,
        clip_id
    )

    if key in prediction_lookup:

        saved_reference, saved_prediction = (
            prediction_lookup[key]
        )

    elif (
        "clip_id",
        clip_id
    ) in prediction_lookup:

        saved_reference, saved_prediction = (
            prediction_lookup[
                ("clip_id", clip_id)
            ]
        )

    # --------------------------------------------------------
    # Find NPY
    # --------------------------------------------------------

    npy_path = find_npy(
        row
    )

    print()
    print(
        f"[{i:02d}/{len(selected):02d}] "
        f"{video_id} / {clip_id}"
    )

    print(
        f"NPY : {npy_path}"
    )

    print(
        f"Reference       : {repr(saved_reference)}"
    )

    print(
        f"Prediction(saved): {repr(saved_prediction)}"
    )

    # --------------------------------------------------------
    # If NPY missing
    # --------------------------------------------------------

    if npy_path is None:

        print(
            "Prediction(now)  : [NPY NOT FOUND]"
        )

        npy_failed += 1

        continue

    try:

        # ----------------------------------------------------
        # Load NPY
        # ----------------------------------------------------

        x = load_npy(
            npy_path
        )

        # ----------------------------------------------------
        # Current inference
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(x)

        current_prediction = ctc_decode(
            logits[0],
            itos
        )

        npy_success += 1

        print(
            f"Input shape     : {tuple(x.shape)}"
        )

        print(
            f"Prediction(now) : {repr(current_prediction)}"
        )

        # ----------------------------------------------------
        # Compare saved prediction vs current
        # ----------------------------------------------------

        if saved_prediction:

            if (
                saved_prediction
                == current_prediction
            ):

                print(
                    "CHECK            : SAME"
                )

                matched_prediction += 1

            else:

                print(
                    "CHECK            : DIFFERENT"
                )

        else:

            print(
                "CHECK            : No saved prediction found"
            )

        # ----------------------------------------------------
        # Reference exists?
        # ----------------------------------------------------

        if saved_reference:

            matched_reference += 1

    except Exception as e:

        npy_failed += 1

        print(
            f"ERROR           : {e}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Selected clips              : {len(selected)}"
)

print(
    f"NPY inference success       : {npy_success}"
)

print(
    f"NPY inference failed        : {npy_failed}"
)

print(
    f"Clips with Reference        : {matched_reference}"
)

print(
    f"Saved vs Current SAME       : {matched_prediction}"
)

print()
print("Done.")
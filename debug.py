# debug_transcript_flow_v4.py
#
# READ-ONLY
# Scan project for real transcript/label sources
# Then try to match them with video_id / clip_id / dataset_clip_key

from pathlib import Path
import csv
import json
import re
from collections import defaultdict

ROOT = Path(r"C:\LipReadingSSL")

# ============================================================
# CONFIG
# ============================================================

MASTER = ROOT / "output" / "dataset" / "dataset_master.csv"

# ไม่ต้องสแกน folder ใหญ่ที่ไม่มีประโยชน์
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    "venv310",
    "lipreading_env",
    "node_modules",
    ".idea",
    ".vscode",
}

TEXT_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".xml",
    ".jsonl",
    ".srt",
    ".vtt",
    ".yaml",
    ".yml",
    ".md",
}

# คำที่บ่งบอกว่า field น่าจะเกี่ยวกับ transcript
TRANSCRIPT_KEYS = {
    "transcript",
    "text",
    "sentence",
    "sentence_text",
    "utterance",
    "label",
    "caption",
    "subtitle",
    "phrase",
    "normalized_text",
    "words",
    "word",
}

IDENTITY_KEYS = {
    "video_id",
    "clip_id",
    "dataset_clip_key",
    "dataset_id",
    "speaker_id",
    "sentence_id",
    "video",
    "clip",
}

# ค่าที่ไม่ถือว่าเป็น transcript จริง
FALSE_VALUES = {
    "",
    "false",
    "none",
    "null",
    "nan",
    "unknown",
    "n/a",
    "na",
    "0",
}


# ============================================================
# HELPERS
# ============================================================

def normalize(value):
    if value is None:
        return ""

    value = str(value).strip()

    # normalize whitespace
    value = re.sub(r"\s+", " ", value)

    return value


def is_real_text(value):
    """
    ตรวจว่าค่านี้มีโอกาสเป็น transcript จริงหรือไม่
    """

    value = normalize(value)

    if not value:
        return False

    if value.lower() in FALSE_VALUES:
        return False

    # boolean
    if value.lower() in {"true", "false"}:
        return False

    # ตัวเลขล้วนไม่ถือเป็น transcript
    if re.fullmatch(r"[\d\W_]+", value):
        return False

    # log / path / package ไม่ถือเป็น transcript
    bad_patterns = [
        r"^collecting ",
        r"^using cached ",
        r"^installing ",
        r"^running command ",
        r"^resolution\s*:",
        r"^fps\s*:",
        r"^frames\s*:",
        r"^duration\s*:",
        r"^status\s*:",
        r"^video\s*:",
        r"^metadata\s*:",
        r"^preview saved",
        r"^metadata saved",
        r"^numpy",
        r"^python",
        r"^pip ",
        r"^https?://",
        r"^[A-Za-z]:\\",
    ]

    lower = value.lower()

    for pattern in bad_patterns:
        if re.search(pattern, lower):
            return False

    # transcript ต้องมีตัวอักษร
    if not re.search(r"[A-Za-zก-๙]", value):
        return False

    return True


def looks_like_transcript(value):
    """
    heuristic เพิ่มเติม
    """

    if not is_real_text(value):
        return False

    value = normalize(value)

    # ยาวเกินไปมากมักเป็น report/log
    if len(value) > 500:
        return False

    # filename/path
    if "\\" in value or "/" in value:
        return False

    # ข้อความที่มีคำอธิบาย metadata เยอะ ๆ
    metadata_words = [
        "resolution",
        "duration",
        "dataset_version",
        "created_at",
        "total_samples",
        "generated",
        "status",
        "speaker",
    ]

    low = value.lower()

    if sum(word in low for word in metadata_words) >= 2:
        return False

    return True


def safe_read_text(path):
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp874",
        "tis-620",
        "latin-1",
    ]

    for enc in encodings:
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            pass

    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_files():
    files = []

    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        # skip unwanted folders
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        if path.suffix.lower() in TEXT_EXTENSIONS:
            files.append(path)

    return files


# ============================================================
# MASTER DATASET
# ============================================================

def load_master():

    if not MASTER.exists():
        print("❌ MASTER DATASET NOT FOUND")
        return None

    print("=" * 80)
    print("STEP 2 — LOAD MASTER DATASET")
    print("=" * 80)

    with MASTER.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = []

        video_ids = set()
        clip_ids = set()
        dataset_keys = set()

        for row in reader:

            rows.append(row)

            video = normalize(row.get("video_id"))
            clip = normalize(row.get("clip_id"))
            key = normalize(row.get("dataset_clip_key"))

            if video:
                video_ids.add(video)

            if clip:
                clip_ids.add(clip)

            if key:
                dataset_keys.add(key)

    print(f"Master : {MASTER.relative_to(ROOT)}")
    print(f"Rows   : {len(rows):,}")
    print(f"Video IDs : {len(video_ids):,}")
    print(f"Clip IDs  : {len(clip_ids):,}")
    print(f"Dataset keys : {len(dataset_keys):,}")

    return {
        "rows": rows,
        "video_ids": video_ids,
        "clip_ids": clip_ids,
        "dataset_keys": dataset_keys,
    }


# ============================================================
# CSV / TSV
# ============================================================

def scan_csv(path):

    results = []

    try:

        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            if not reader.fieldnames:
                return results

            fields = [
                normalize(x).lower()
                for x in reader.fieldnames
                if x
            ]

            transcript_fields = [
                f for f in fields
                if (
                    f in TRANSCRIPT_KEYS
                    or "transcript" in f
                    or "sentence" in f
                    or "utterance" in f
                    or "caption" in f
                    or "subtitle" in f
                )
            ]

            identity_fields = [
                f for f in fields
                if (
                    f in IDENTITY_KEYS
                    or "video_id" in f
                    or "clip_id" in f
                    or "dataset_clip_key" in f
                )
            ]

            if not transcript_fields:
                return results

            for row in reader:

                real_values = {}

                for field in transcript_fields:

                    original_key = next(
                        (
                            k for k in row.keys()
                            if normalize(k).lower() == field
                        ),
                        None
                    )

                    if original_key is None:
                        continue

                    value = normalize(row.get(original_key))

                    if looks_like_transcript(value):
                        real_values[field] = value

                if real_values:

                    identities = {}

                    for field in identity_fields:

                        original_key = next(
                            (
                                k for k in row.keys()
                                if normalize(k).lower() == field
                            ),
                            None
                        )

                        if original_key:
                            identities[field] = normalize(
                                row.get(original_key)
                            )

                    results.append({
                        "identities": identities,
                        "transcripts": real_values,
                    })

    except Exception:
        pass

    return results


# ============================================================
# JSON
# ============================================================

def scan_json(path):

    results = []

    try:

        text = safe_read_text(path)

        if not text:
            return results

        data = json.loads(text)

        def walk(obj, identities=None):

            if identities is None:
                identities = {}

            if isinstance(obj, dict):

                current_identity = dict(identities)

                for key, value in obj.items():

                    k = normalize(key).lower()

                    if k in IDENTITY_KEYS:
                        current_identity[k] = normalize(value)

                for key, value in obj.items():

                    k = normalize(key).lower()

                    if (
                        k in TRANSCRIPT_KEYS
                        or "transcript" in k
                        or "sentence" in k
                        or "utterance" in k
                        or "caption" in k
                        or "subtitle" in k
                    ):

                        if isinstance(value, str):

                            if looks_like_transcript(value):

                                results.append({
                                    "identities": current_identity.copy(),
                                    "transcripts": {
                                        k: normalize(value)
                                    }
                                })

                    walk(value, current_identity)

            elif isinstance(obj, list):

                for item in obj:
                    walk(item, identities)

        walk(data)

    except Exception:
        pass

    return results


# ============================================================
# TEXT / SRT / VTT / XML / ETC.
# ============================================================

def scan_text(path):

    results = []

    text = safe_read_text(path)

    if not text:
        return results

    # XML
    if path.suffix.lower() == ".xml":

        matches = re.findall(
            r"<(?:text|transcript|sentence|utterance|caption|subtitle)[^>]*>"
            r"(.*?)"
            r"</(?:text|transcript|sentence|utterance|caption|subtitle)>",
            text,
            flags=re.I | re.S,
        )

        for match in matches:

            value = re.sub(
                r"<[^>]+>",
                " ",
                match
            )

            value = normalize(value)

            if looks_like_transcript(value):

                results.append({
                    "identities": {},
                    "transcripts": {
                        "xml_text": value
                    }
                })

        return results

    # TXT / SRT / VTT / MD / YAML etc.
    lines = text.splitlines()

    for line in lines:

        line = normalize(line)

        if not line:
            continue

        # skip SRT numbering
        if re.fullmatch(r"\d+", line):
            continue

        # skip timestamps
        if "-->" in line:
            continue

        # key:value
        match = re.match(
            r"^\s*(transcript|sentence|text|utterance|caption|subtitle)"
            r"\s*[:=]\s*(.+)$",
            line,
            flags=re.I,
        )

        if match:

            value = normalize(match.group(2))

            if looks_like_transcript(value):

                results.append({
                    "identities": {},
                    "transcripts": {
                        match.group(1).lower(): value
                    }
                })

            continue

        # generic line
        # แต่ไม่อ่าน report/log ทั้งหมดเป็น transcript
        if path.suffix.lower() in {
            ".srt",
            ".vtt",
            ".txt",
        }:

            if looks_like_transcript(line):

                results.append({
                    "identities": {},
                    "transcripts": {
                        "text": line
                    }
                })

    return results


# ============================================================
# SCAN ONE FILE
# ============================================================

def scan_file(path):

    ext = path.suffix.lower()

    if ext in {".csv", ".tsv"}:
        return scan_csv(path)

    if ext in {".json", ".jsonl"}:
        return scan_json(path)

    return scan_text(path)


# ============================================================
# MATCH
# ============================================================

def make_master_indexes(master):

    by_video = defaultdict(list)
    by_clip = defaultdict(list)
    by_key = defaultdict(list)

    for row in master["rows"]:

        video = normalize(row.get("video_id"))
        clip = normalize(row.get("clip_id"))
        key = normalize(row.get("dataset_clip_key"))

        if video:
            by_video[video].append(row)

        if clip:
            by_clip[clip].append(row)

        if key:
            by_key[key].append(row)

    return by_video, by_clip, by_key


def find_matches(source, by_video, by_clip, by_key):

    identity = source["identities"]

    matches = []

    # dataset_clip_key
    key = identity.get("dataset_clip_key")

    if key and key in by_key:
        matches.extend(
            ("dataset_clip_key", row)
            for row in by_key[key]
        )

    # clip_id
    clip = identity.get("clip_id")

    if clip and clip in by_clip:
        matches.extend(
            ("clip_id", row)
            for row in by_clip[clip]
        )

    # video_id
    video = identity.get("video_id")

    if video and video in by_video:
        matches.extend(
            ("video_id", row)
            for row in by_video[video]
        )

    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DEBUG — FIND REAL TRANSCRIPT SOURCE V4")
    print("=" * 80)

    print(f"Root : {ROOT}")
    print("Mode : READ-ONLY")

    print()

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print("=" * 80)
    print("STEP 1 — DISCOVER TEXT / METADATA FILES")
    print("=" * 80)

    files = discover_files()

    print(f"Candidate files : {len(files):,}")

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    master = load_master()

    if master is None:
        return

    by_video, by_clip, by_key = make_master_indexes(master)

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 3 — SEARCH FOR REAL TRANSCRIPT VALUES")
    print("=" * 80)

    sources = []

    scanned = 0

    for path in files:

        scanned += 1

        # progress ทุก 100 files
        if scanned % 100 == 0:

            print(
                f"Scanning : {scanned:,}/{len(files):,}",
                end="\r"
            )

        results = scan_file(path)

        if results:

            sources.append(
                (path, results)
            )

    print()
    print()

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print("=" * 80)
    print("STEP 4 — REAL TRANSCRIPT SOURCES")
    print("=" * 80)

    if not sources:

        print("❌ NO REAL TRANSCRIPT FOUND")

        print()
        print("Possible locations:")
        print("  - original dataset")
        print("  - annotation files")
        print("  - subtitle files")
        print("  - external label files")

        return

    print(
        f"✅ Files containing possible real transcript : "
        f"{len(sources)}"
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 5 — TRANSCRIPT SOURCES")
    print("=" * 80)

    total_values = 0

    for path, results in sources:

        print()
        print("-" * 80)
        print(f"FILE: {path.relative_to(ROOT)}")
        print(f"Transcript records : {len(results):,}")

        for item in results[:10]:

            total_values += 1

            print()

            if item["identities"]:
                print(
                    "Identity :",
                    item["identities"]
                )

            print(
                "Transcript :",
                item["transcripts"]
            )

        if len(results) > 10:

            print(
                f"... {len(results) - 10:,} more"
            )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 6 — MATCH TRANSCRIPT → MASTER")
    print("=" * 80)

    matched = []

    for path, results in sources:

        for source in results:

            matches = find_matches(
                source,
                by_video,
                by_clip,
                by_key
            )

            if matches:

                matched.append(
                    (
                        path,
                        source,
                        matches
                    )
                )

    print(
        f"Transcript records found : {total_values:,}"
    )

    print(
        f"Matched records           : {len(matched):,}"
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 7 — MATCHED TRANSCRIPT SAMPLES")
    print("=" * 80)

    if not matched:

        print("❌ NO IDENTITY MATCH")

        print()
        print("This means transcript source may not contain:")
        print("  video_id")
        print("  clip_id")
        print("  dataset_clip_key")

    else:

        for path, source, matches in matched[:20]:

            print()
            print("-" * 80)

            print(
                f"SOURCE : {path.relative_to(ROOT)}"
            )

            print(
                "Transcript :",
                source["transcripts"]
            )

            for method, row in matches[:3]:

                print(
                    f"Matched by {method}: "
                    f"video={row.get('video_id')} "
                    f"clip={row.get('clip_id')} "
                    f"key={row.get('dataset_clip_key')}"
                )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 8 — TRANSCRIPT GRANULARITY")
    print("=" * 80)

    granularities = defaultdict(int)

    for path, results in sources:

        for source in results:

            identities = source["identities"]

            if "dataset_clip_key" in identities:
                granularities["dataset_clip_key"] += 1

            elif "clip_id" in identities:
                granularities["clip_id"] += 1

            elif "video_id" in identities:
                granularities["video_id"] += 1

            else:
                granularities["no_identity"] += 1

    for key, count in sorted(
        granularities.items(),
        key=lambda x: -x[1]
    ):

        print(
            f"{key:20s}: {count:,}"
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL DIAGNOSIS")
    print("=" * 80)

    if matched:

        print(
            "✅ พบ transcript ที่สามารถจับคู่กับ dataset ได้"
        )

        print()
        print(
            f"Matched records : {len(matched):,}"
        )

        print()
        print(
            "ขั้นต่อไปสามารถสร้าง transcript mapping "
            "สำหรับ dataset_master.csv ได้"
        )

    elif sources:

        print(
            "⚠️ พบข้อความที่น่าจะเป็น transcript "
            "แต่ยังจับคู่กับ clip/video ไม่ได้"
        )

        print()
        print(
            "ต้องตรวจ identity format ของ source ต่อ"
        )

    else:

        print(
            "❌ ยังไม่พบ transcript จริง"
        )

        print()
        print(
            "ปัญหาน่าจะอยู่ที่ original dataset / annotation source"
        )

    print()
    print("=" * 80)
    print("DEBUG COMPLETE — READ ONLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
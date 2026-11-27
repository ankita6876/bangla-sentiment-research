
# =============================================================================
# STAGE 7D–7E — DGSL-NET
# DISAGREEMENT-GUIDED SPECIALIST LEARNING NETWORK
# STEPS 125–140
#
# GROUP-DISJOINT OOF LEARNING
# DIFFICULTY-AWARE SPECIALISTS
# COMPETENCE-SUPERVISED ROUTING
# MATCHED ARCHITECTURAL CONTROLS
# AUTOMATIC PERMANENT GITHUB BACKUP
# =============================================================================

import os
import re
import gc
import json
import time
import math
import shutil
import hashlib
import zipfile
import base64
import random
import subprocess
import unicodedata
import sys

from pathlib import Path
from getpass import getpass

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score
)

from scipy.special import softmax
from scipy.stats import binomtest

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

START = time.time()

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"

EXPECTED_COMMIT = "3798ab3fbc88f6889b952ecd044bcea2452ac7dc"

BASE_TAG = "stage6b-locked-baselines-v1"
PREVIOUS_TAG = "stage7c-specialist-discovery-v1"
STAGE_TAG = "stage7de-dgsl-screening-v1"

MODEL_NAME = "csebuetnlp/banglabert"

RAW_SHA = (
    "4580950af06bbfbe5bd79e20957b63a2ed"
    "122af2a6a3e685b10576d3f70241c5"
)

SEED = 42
FOLDS = 3

MAX_LENGTH = 64
TRAIN_BATCH = 32
EVAL_BATCH = 64

OOF_EPOCHS = 3
FINAL_EPOCHS = 4

ENCODER_LR = 2e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = .01
WARMUP = .10

DIM = 128
DROPOUT = .15

BOOTSTRAPS = 2000

MODES = [
    "standard",
    "matched_control",
    "dgsl_fixed",
    "dgsl_full"
]

CLASSES = ["Negative", "Neutral", "Positive"]
LABEL_MAP = dict(zip(CLASSES, range(3)))

OUT = WORK / "sentiment_research_stage7de"
REC = OUT / "recovered"
AUDIT = OUT / "audit"
TABLES = OUT / "tables"
PRED = OUT / "private_predictions"
CKPT = OUT / "checkpoints"
FIGURES = OUT / "figures"

PUB = WORK / "sentiment_publication_assets"

REPO = WORK / "sentiment_stage7de_github"
BACKUP = REPO / "research_stage7de"

for p in [
    OUT, REC, AUDIT, TABLES,
    PRED, CKPT, FIGURES, PUB
]:
    p.mkdir(parents=True, exist_ok=True)

for fmt in ["png", "pdf", "svg"]:
    (FIGURES / fmt).mkdir(parents=True, exist_ok=True)
    (PUB / "figures" / fmt).mkdir(parents=True, exist_ok=True)

for sub in [
    "tables/csv",
    "tables/latex",
    "captions",
    "scripts",
    "architecture"
]:
    (PUB / sub).mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), (
    "Enable GPU T4 in Kaggle settings."
)

DEVICE = torch.device("cuda")

def section(title):
    print("\n" + "=" * 94)
    print(title)
    print("=" * 94)

def seed_all(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(SEED)

def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024**2), b""
        ):
            h.update(chunk)

    return h.hexdigest()

def save_json(obj, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

def macro_f1(y, pred):

    return f1_score(
        y,
        pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

def metrics(y, probs):

    pred = probs.argmax(1)

    return {
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "macro_f1": float(
            macro_f1(y, pred)
        ),
        "nll": float(
            log_loss(y, probs, labels=[0, 1, 2])
        )
    }

def save_table(rows, name):

    df = pd.DataFrame(rows)

    df.to_csv(
        TABLES / f"{name}.csv",
        index=False
    )

    df.to_csv(
        PUB / "tables/csv" / f"{name}.csv",
        index=False
    )

    df.to_latex(
        PUB / "tables/latex" / f"{name}.tex",
        index=False,
        escape=True,
        float_format="%.4f"
    )

    return df

def clear_gpu():

    gc.collect()
    torch.cuda.empty_cache()

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

# =============================================================================
# STEP 125 — VERIFIED GITHUB CONNECTION
# =============================================================================

section("STEP 125 — VERIFIED GITHUB CONNECTION")

TOKEN = getpass(
    "Enter your working GitHub token: "
).strip()

API = "https://api.github.com"

SESSION = requests.Session()

SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

def api_get(path):

    response = SESSION.get(
        API + path,
        timeout=90
    )

    response.raise_for_status()

    return response.json()

assert api_get("/user")["login"].lower() == OWNER.lower()

public_info = api_get(
    f"/repos/{OWNER}/{PUBLIC_REPO}"
)

private_info = api_get(
    f"/repos/{OWNER}/{PRIVATE_REPO}"
)

assert public_info["permissions"]["push"]
assert private_info["permissions"]["push"]
assert private_info["private"]

GIT_ENV = os.environ.copy()

GIT_ENV.update({
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0":
        "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0":
        "Authorization: Basic " +
        base64.b64encode(
            f"x-access-token:{TOKEN}".encode()
        ).decode()
})

def git(*args):

    result = subprocess.run(
        ["git", *args],
        cwd=str(REPO) if REPO.exists() else str(WORK),
        env=GIT_ENV,
        capture_output=True,
        text=True
    )

    if result.returncode:

        raise RuntimeError(
            result.stderr.replace(
                TOKEN, "[REDACTED]"
            )
        )

    return result.stdout.strip()

if not (REPO / ".git").exists():

    git(
        "clone",
        "--branch", BRANCH,
        f"https://github.com/{OWNER}/{PUBLIC_REPO}.git",
        str(REPO)
    )

else:

    git("fetch", "origin")
    git("checkout", BRANCH)
    git("pull", "--ff-only", "origin", BRANCH)

git(
    "merge-base",
    "--is-ancestor",
    EXPECTED_COMMIT,
    "HEAD"
)

git("config", "user.name", OWNER)

git(
    "config",
    "user.email",
    f"{OWNER}@users.noreply.github.com"
)

BACKUP.mkdir(parents=True, exist_ok=True)

def push_public(message):

    git("add", "research_stage7de")

    if git("diff", "--cached", "--name-only"):

        git("commit", "-m", message)

    git(
        "push",
        "origin",
        f"HEAD:{BRANCH}"
    )

    local = git("rev-parse", "HEAD")

    remote = git(
        "ls-remote",
        "origin",
        f"refs/heads/{BRANCH}"
    ).split()[0]

    assert local == remote

    print("GitHub synchronized:", local)

    return local

print("Authentication: PASS")
print("Public repository access: PASS")
print("Private repository access: PASS")
print("Stage 7C commit: VERIFIED")

# Preserve complete source before training.

history = get_ipython().history_manager.input_hist_raw

sources = [
    s for s in history
    if "STAGE 7D–7E — DGSL-NET" in s
    and len(s) > 10000
]

assert sources

SOURCE = sources[-1]

assert TOKEN not in SOURCE

for path in [
    AUDIT / "stage7de_complete.py",
    BACKUP / "stage7de_complete.py",
    PUB / "scripts/stage7de_complete.py"
]:

    path.write_text(
        SOURCE,
        encoding="utf-8"
    )

push_public(
    "Stage 7DE: preserve DGSL source before training"
)

# =============================================================================
# STEP 126 — PRIVATE ARTIFACT STORAGE AND RECOVERY
# =============================================================================

section("STEP 126 — PRIVATE ARTIFACT RECOVERY")

def get_release(tag):

    return api_get(
        f"/repos/{OWNER}/{PRIVATE_REPO}/releases/tags/{tag}"
    )

def assets_of(release):

    result = []
    page = 1

    while True:

        batch = api_get(
            f"/repos/{OWNER}/{PRIVATE_REPO}/releases/"
            f"{release['id']}/assets?per_page=100&page={page}"
        )

        result.extend(batch)

        if len(batch) < 100:
            break

        page += 1

    return {
        a["name"]: a
        for a in result
    }

def download_asset(asset, target):

    target = Path(target)

    target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    digest = asset.get("digest") or ""

    valid = (
        target.exists()
        and target.stat().st_size == asset["size"]
        and digest.startswith("sha256:")
        and sha256(target) == digest.split(":", 1)[1]
    )

    if not valid:

        url = (
            f"{API}/repos/{OWNER}/{PRIVATE_REPO}/"
            f"releases/assets/{asset['id']}"
        )

        temp = target.with_suffix(
            target.suffix + ".partial"
        )

        with SESSION.get(
            url,
            headers={
                "Accept": "application/octet-stream"
            },
            stream=True,
            timeout=(30, 600)
        ) as response:

            response.raise_for_status()

            with open(temp, "wb") as f:

                for chunk in response.iter_content(
                    1024**2
                ):

                    if chunk:
                        f.write(chunk)

        temp.replace(target)

    assert target.stat().st_size == asset["size"]

    if digest.startswith("sha256:"):

        assert sha256(target) == digest.split(":", 1)[1]

    print("Recovered:", target.name)

    return target

def safe_extract(archive, destination):

    destination = Path(destination).resolve()

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    with zipfile.ZipFile(archive) as z:

        assert z.testzip() is None

        for name in z.namelist():

            assert (
                destination / name
            ).resolve().is_relative_to(destination)

        z.extractall(destination)

BASE_ASSETS = assets_of(
    get_release(BASE_TAG)
)

PREVIOUS_ASSETS = assets_of(
    get_release(PREVIOUS_TAG)
)

package = download_asset(
    BASE_ASSETS[
        "stage6b_reproducibility_package.zip"
    ],
    REC / "baseline_package.zip"
)

safe_extract(
    package,
    REC / "baseline_package"
)

previous_private = download_asset(
    PREVIOUS_ASSETS[
        "sentiment_stage7c_private_83204e6b757f.zip"
    ],
    REC / "previous_private.zip"
)

safe_extract(
    previous_private,
    REC / "previous_private"
)

previous_publication = download_asset(
    PREVIOUS_ASSETS[
        "sentiment_stage7c_publication_6b1cef4a2afa.zip"
    ],
    REC / "previous_publication.zip"
)

safe_extract(
    previous_publication,
    PUB
)

# Create or recover the new private release.

response = SESSION.get(
    f"{API}/repos/{OWNER}/{PRIVATE_REPO}/"
    f"releases/tags/{STAGE_TAG}",
    timeout=90
)

if response.status_code == 404:

    response = SESSION.post(
        f"{API}/repos/{OWNER}/{PRIVATE_REPO}/releases",
        json={
            "tag_name": STAGE_TAG,
            "name": "Stage 7DE — DGSL-Net screening",
            "body": (
                "Group-disjoint OOF predictions, "
                "DGSL checkpoints, private predictions "
                "and reproducibility artifacts."
            )
        },
        timeout=90
    )

response.raise_for_status()

RELEASE = response.json()

REMOTE = assets_of(RELEASE)

REMOTE_MANIFEST = []

def upload_verified(path, name=None):

    path = Path(path)

    name = name or path.name

    digest = sha256(path)

    if name in REMOTE:

        asset = REMOTE[name]

    else:

        with open(path, "rb") as f:

            response = SESSION.post(
                RELEASE["upload_url"].split("{")[0],
                params={"name": name},
                headers={
                    "Content-Type":
                        "application/octet-stream"
                },
                data=f,
                timeout=(30, 1200)
            )

        response.raise_for_status()

        asset = response.json()

        REMOTE[name] = asset

    assert asset["size"] == path.stat().st_size

    remote_digest = asset.get("digest") or ""

    if remote_digest.startswith("sha256:"):

        assert remote_digest.split(":", 1)[1] == digest

    else:

        check = download_asset(
            asset,
            REC / "remote_check" / name
        )

        assert sha256(check) == digest

        check.unlink()

    record = {
        "name": name,
        "asset_id": asset["id"],
        "size": asset["size"],
        "sha256": digest
    }

    REMOTE_MANIFEST.append(record)

    print("Remote verified:", name)

    return record

print("Private release:", RELEASE["html_url"])

for fmt in ["png", "pdf", "svg"]:

    print(
        "Existing", fmt.upper(), "figures:",
        len(list(
            (PUB / "figures" / fmt).rglob(f"*.{fmt}")
        ))
    )

# =============================================================================
# STEP 127 — EXACT DATASET RECOVERY
# =============================================================================

section("STEP 127 — LOCKED DATASET RECOVERY")

raw_path = next(
    (
        p for p in INPUT.rglob("BECS Dataset.csv")
        if sha256(p) == RAW_SHA
    ),
    None
)

assert raw_path is not None, (
    "Original BECS dataset not found."
)

raw = pd.read_csv(raw_path)

def normalize_text(text):

    text = unicodedata.normalize(
        "NFC", str(text)
    )

    text = re.sub(
        r"[\u200b\u200c\u200d\u2060\ufeff]",
        "",
        text
    )

    return re.sub(
        r"\s+", " ", text
    ).strip().casefold()

try:

    from normalizer import normalize as bn_normalize

except ImportError:

    subprocess.run(
        [
            sys.executable,
            "-m", "pip",
            "install", "-q",
            "git+https://github.com/csebuetnlp/normalizer"
        ],
        check=True
    )

    from normalizer import normalize as bn_normalize

DATA = {}

for split, expected in [
    ("train", 9204),
    ("val", 1840)
]:

    paths = list(
        (REC / "baseline_package").rglob(
            f"seed42_{split}.csv"
        )
    )

    assert len(paths) == 1

    membership = pd.read_csv(paths[0])

    membership = membership.sort_values(
        "partition_order"
    ).reset_index(drop=True)

    df = raw.iloc[
        membership["raw_row_id"].to_numpy(dtype=int)
    ].copy().reset_index(drop=True)

    df["Label"] = (
        df["Label"].astype(str).str.strip().str.title()
    )

    df["text_norm"] = df["Bangla"].map(
        normalize_text
    )

    df["sample_id"] = df["text_norm"].map(
        lambda s: hashlib.sha256(
            s.encode()
        ).hexdigest()
    )

    assert np.array_equal(
        df["sample_id"].to_numpy(dtype=str),
        membership["sample_id"].to_numpy(dtype=str)
    )

    assert np.array_equal(
        df["Label"].str.casefold().to_numpy(),
        membership["Label"].astype(str)
            .str.strip().str.casefold().to_numpy()
    )

    df["group_id"] = membership[
        "group_id"
    ].astype(str)

    df["label_id"] = df["Label"].map(
        LABEL_MAP
    ).astype(int)

    df["model_text"] = df["text_norm"].map(
        bn_normalize
    )

    df["word_count"] = df[
        "text_norm"
    ].str.split().str.len()

    assert len(df) == expected

    DATA[split] = df

assert not set(
    DATA["train"]["sample_id"]
).intersection(DATA["val"]["sample_id"])

assert not set(
    DATA["train"]["group_id"]
).intersection(DATA["val"]["group_id"])

assert not set(
    DATA["train"]["model_text"]
).intersection(DATA["val"]["model_text"])

TRAIN = DATA["train"]
VAL = DATA["val"]

Y_TRAIN = TRAIN["label_id"].to_numpy(dtype=int)
Y_VAL = VAL["label_id"].to_numpy(dtype=int)

TRAIN_IDS = TRAIN["sample_id"].to_numpy(dtype=str)
VAL_IDS = VAL["sample_id"].to_numpy(dtype=str)

GROUP_TRAIN = TRAIN["group_id"].to_numpy(dtype=str)
GROUP_VAL = VAL["group_id"].to_numpy(dtype=str)

N_TRAIN = len(TRAIN)
N_VAL = len(VAL)

NEG = ["না", "নয়", "নয়", "নেই", "নাই"]

CON = [
    "কিন্তু", "তবে", "যদিও",
    "অথচ", "তবুও", "তথাপি"
]

def cue_pattern(words):

    return re.compile(
        r"(?<![\w\u0980-\u09ff])(?:"
        + "|".join(map(re.escape, words))
        + r")(?![\w\u0980-\u09ff])"
    )

NEG_PATTERN = cue_pattern(NEG)
CON_PATTERN = cue_pattern(CON)

LONG_THRESHOLD = TRAIN["word_count"].quantile(.75)

def linguistic_features(df):

    neg = df["text_norm"].map(
        lambda s: bool(NEG_PATTERN.search(s))
    ).to_numpy()

    con = df["text_norm"].map(
        lambda s: bool(CON_PATTERN.search(s))
    ).to_numpy()

    long_text = (
        df["word_count"] > LONG_THRESHOLD
    ).to_numpy()

    length = np.minimum(
        np.log1p(df["word_count"].to_numpy()) / 5,
        1
    )

    features = np.column_stack([
        neg,
        con,
        long_text,
        length
    ]).astype(np.float32)

    return features

LING_TRAIN = linguistic_features(TRAIN)
LING_VAL = linguistic_features(VAL)

MASKS = {
    "all": np.ones(N_VAL, dtype=bool),
    "negation": LING_VAL[:, 0].astype(bool),
    "contrast": LING_VAL[:, 1].astype(bool),
    "any_operator": (
        LING_VAL[:, 0].astype(bool)
        | LING_VAL[:, 1].astype(bool)
    ),
    "both_operators": (
        LING_VAL[:, 0].astype(bool)
        & LING_VAL[:, 1].astype(bool)
    ),
    "long_text": LING_VAL[:, 2].astype(bool),
    "short_text": ~LING_VAL[:, 2].astype(bool)
}

MASKS["no_operator"] = ~MASKS["any_operator"]

# Recover the exact Stage 7C comparison predictions.

paths = list(
    (REC / "previous_private").rglob(
        "stage7c_aligned_15_models.npz"
    )
)

assert len(paths) == 1

with np.load(
    paths[0],
    allow_pickle=False
) as z:

    ids = z["sample_id"].astype(str)

    assert len(set(ids)) == len(ids)
    assert set(ids) == set(VAL_IDS)

    lookup = {
        sid: i
        for i, sid in enumerate(ids)
    }

    order = np.array([
        lookup[sid]
        for sid in VAL_IDS
    ])

    assert np.array_equal(
        z["labels"][order],
        Y_VAL
    )

    HISTORICAL = {
        "previous_nine":
            z["ensemble_previous_nine"][order],

        "prototype_family":
            z["ensemble_prototype_family"][order],

        "all_fifteen":
            z["ensemble_all_fifteen"][order]
    }

assert abs(
    metrics(
        Y_VAL,
        HISTORICAL["prototype_family"]
    )["macro_f1"] - .8287035909704495
) < 1e-6

print("Raw dataset SHA256: PASS")
print("Training samples:", N_TRAIN)
print("Validation samples:", N_VAL)
print("Group overlap: ZERO")
print("Normalized text overlap: ZERO")
print("Previous best result: REPRODUCED")
print("Test partition loaded: NO")

# =============================================================================
# STEP 128 — TOKENIZATION AND OOF FOLD CONSTRUCTION
# =============================================================================

section("STEP 128 — GROUP-DISJOINT OOF FOLDS")

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

ENCODED = {}

for split in ["train", "val"]:

    encoded = TOKENIZER(
        DATA[split]["model_text"].tolist(),
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_special_tokens_mask=True,
        return_tensors="pt"
    )

    ENCODED[split] = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "valid_mask": (
            encoded["attention_mask"].bool()
            & ~encoded["special_tokens_mask"].bool()
        )
    }

# Merge any training groups connected by the final
# tokenizer input representation, preventing identical
# model inputs from crossing OOF folds.

parent = list(range(N_TRAIN))

def find(i):

    while parent[i] != i:

        parent[i] = parent[parent[i]]
        i = parent[i]

    return i

def union(a, b):

    a, b = find(a), find(b)

    if a != b:

        parent[max(a, b)] = min(a, b)

group_seen = {}
token_seen = {}

for i in range(N_TRAIN):

    g = GROUP_TRAIN[i]

    key = tuple(
        ENCODED["train"]["input_ids"][i].tolist()
    )

    if g in group_seen:

        union(i, group_seen[g])

    else:

        group_seen[g] = i

    if key in token_seen:

        union(i, token_seen[key])

    else:

        token_seen[key] = i

OOF_GROUPS = np.array([
    find(i)
    for i in range(N_TRAIN)
])

splitter = StratifiedGroupKFold(
    n_splits=FOLDS,
    shuffle=True,
    random_state=SEED
)

FOLD_SPLITS = list(
    splitter.split(
        np.zeros(N_TRAIN),
        Y_TRAIN,
        OOF_GROUPS
    )
)

FOLD_ID = np.full(
    N_TRAIN,
    -1,
    dtype=int
)

FOLD_ROWS = []

for fold, (fit_idx, hold_idx) in enumerate(
    FOLD_SPLITS
):

    assert not set(
        OOF_GROUPS[fit_idx]
    ).intersection(OOF_GROUPS[hold_idx])

    assert (FOLD_ID[hold_idx] == -1).all()

    FOLD_ID[hold_idx] = fold

    FOLD_ROWS.append({
        "fold": fold,
        "fit_samples": len(fit_idx),
        "held_out_samples": len(hold_idx),
        "fit_groups": len(
            np.unique(OOF_GROUPS[fit_idx])
        ),
        "held_out_groups": len(
            np.unique(OOF_GROUPS[hold_idx])
        ),
        "group_overlap": 0
    })

assert (FOLD_ID >= 0).all()

FOLD_TABLE = save_table(
    FOLD_ROWS,
    "stage7de_01_oof_fold_integrity"
)

print(FOLD_TABLE.to_string(index=False))

# =============================================================================
# DATA LOADERS
# =============================================================================

class SentimentDataset(Dataset):

    def __init__(
        self,
        split,
        indices=None,
        difficulty=None,
        teacher=None
    ):

        self.split = split

        self.enc = ENCODED[split]

        self.y = (
            Y_TRAIN
            if split == "train"
            else Y_VAL
        )

        self.ling = (
            LING_TRAIN
            if split == "train"
            else LING_VAL
        )

        self.indices = (
            np.arange(len(self.y))
            if indices is None
            else np.asarray(indices)
        )

        self.difficulty = difficulty
        self.teacher = teacher

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, position):

        i = self.indices[position]

        return {
            "input_ids":
                self.enc["input_ids"][i],

            "attention_mask":
                self.enc["attention_mask"][i],

            "valid_mask":
                self.enc["valid_mask"][i],

            "labels":
                torch.tensor(
                    self.y[i],
                    dtype=torch.long
                ),

            "ling":
                torch.tensor(
                    self.ling[i],
                    dtype=torch.float32
                ),

            "difficulty":
                torch.tensor(
                    0.0
                    if self.difficulty is None
                    else self.difficulty[i],
                    dtype=torch.float32
                ),

            "teacher":
                torch.tensor(
                    np.ones(3) / 3
                    if self.teacher is None
                    else self.teacher[i],
                    dtype=torch.float32
                ),

            "index":
                torch.tensor(
                    i,
                    dtype=torch.long
                )
        }

def make_loader(
    split,
    indices=None,
    training=False,
    difficulty=None,
    teacher=None,
    seed=SEED
):

    generator = torch.Generator()

    generator.manual_seed(seed)

    return DataLoader(
        SentimentDataset(
            split,
            indices,
            difficulty,
            teacher
        ),
        batch_size=(
            TRAIN_BATCH
            if training
            else EVAL_BATCH
        ),
        shuffle=training,
        generator=generator,
        num_workers=0,
        pin_memory=True
    )

def to_device(batch):

    return {
        k: v.to(
            DEVICE,
            non_blocking=True
        )
        for k, v in batch.items()
    }

def make_optimizer(
    model,
    total_steps,
    custom=False
):

    if custom:

        encoder_params = list(
            model.encoder.parameters()
        ) + list(
            model.base_classifier.parameters()
        )

        encoder_ids = {
            id(p)
            for p in encoder_params
        }

        head_params = [
            p for p in model.parameters()
            if id(p) not in encoder_ids
        ]

        groups = [
            {
                "params": encoder_params,
                "lr": ENCODER_LR
            },
            {
                "params": head_params,
                "lr": HEAD_LR
            }
        ]

    else:

        groups = [
            {
                "params": model.parameters(),
                "lr": ENCODER_LR
            }
        ]

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_steps * WARMUP
        ),
        num_training_steps=total_steps
    )

    return optimizer, scheduler

def scaler_factory():

    return torch.amp.GradScaler(
        "cuda",
        enabled=True
    )

def cpu_state(model):

    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

# =============================================================================
# STEP 129 — OUT-OF-FOLD BASELINE TRAINING
# =============================================================================

section("STEP 129 — OUT-OF-FOLD BASELINE TRAINING")

OOF_PROBS = np.full(
    (N_TRAIN, 3),
    np.nan,
    dtype=np.float64
)

OOF_HISTORY = []
OOF_RESULTS = []

def run_fingerprint(config):

    return hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            default=str
        ).encode()
    ).hexdigest()

def remote_matches(name, expected_hash):

    assert name in REMOTE

    asset = REMOTE[name]

    digest = asset.get("digest") or ""

    if digest.startswith("sha256:"):

        assert digest.split(":", 1)[1] == expected_hash

    else:

        path = download_asset(
            asset,
            REC / "verification" / name
        )

        assert sha256(path) == expected_hash

        path.unlink()

    return True

for fold, (fit_idx, hold_idx) in enumerate(
    FOLD_SPLITS
):

    prefix = f"oof_fold{fold}_seed{SEED}"

    config = {
        "stage": "7de",
        "model": MODEL_NAME,
        "fold": fold,
        "seed": SEED + fold,
        "epochs": OOF_EPOCHS,
        "max_length": MAX_LENGTH,
        "batch": TRAIN_BATCH,
        "lr": ENCODER_LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup": WARMUP,
        "fit_ids_hash": hashlib.sha256(
            TRAIN_IDS[fit_idx].tobytes()
        ).hexdigest(),
        "hold_ids_hash": hashlib.sha256(
            TRAIN_IDS[hold_idx].tobytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(
            SOURCE.encode()
        ).hexdigest()
    }

    run_hash = run_fingerprint(config)

    meta_name = prefix + "_completed.json"
    pred_name = prefix + "_predictions.npz"
    model_name = prefix + "_model.pt"

    if all(
        n in REMOTE
        for n in [
            meta_name,
            pred_name,
            model_name
        ]
    ):

        meta_path = download_asset(
            REMOTE[meta_name],
            AUDIT / meta_name
        )

        meta = json.loads(
            meta_path.read_text()
        )

        assert meta["run_hash"] == run_hash

        pred_path = download_asset(
            REMOTE[pred_name],
            PRED / pred_name
        )

        assert sha256(pred_path) == meta["prediction_sha256"]

        remote_matches(
            model_name,
            meta["checkpoint_sha256"]
        )

        with np.load(
            pred_path,
            allow_pickle=False
        ) as z:

            assert np.array_equal(
                z["sample_id"],
                TRAIN_IDS[hold_idx]
            )

            assert np.array_equal(
                z["labels"],
                Y_TRAIN[hold_idx]
            )

            OOF_PROBS[hold_idx] = z["probabilities"]

        OOF_HISTORY.extend(meta["history"])
        OOF_RESULTS.append(meta["result"])

        print(prefix, "RECOVERED — training skipped")

        continue

    seed_all(SEED + fold)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    ).to(DEVICE)

    fit_loader = make_loader(
        "train",
        fit_idx,
        training=True,
        seed=SEED + fold
    )

    hold_loader = make_loader(
        "train",
        hold_idx
    )

    optimizer, scheduler = make_optimizer(
        model,
        len(fit_loader) * OOF_EPOCHS
    )

    scaler = scaler_factory()

    fold_history = []

    # Fixed epoch count.
    # Held-out fold labels are NOT used for checkpoint
    # selection or early stopping.

    for epoch in range(1, OOF_EPOCHS + 1):

        model.train()

        loss_sum = 0.0
        seen = 0

        for batch in fit_loader:

            batch = to_device(batch)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).logits

                loss = F.cross_entropy(
                    logits,
                    batch["labels"]
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            previous_scale = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            if scaler.get_scale() >= previous_scale:

                scheduler.step()

            n = len(batch["labels"])

            loss_sum += loss.item() * n
            seen += n

        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": loss_sum / seen
        }

        fold_history.append(row)

        print(
            f"OOF fold {fold + 1}/{FOLDS} "
            f"epoch {epoch}/{OOF_EPOCHS} "
            f"loss={row['train_loss']:.5f}"
        )

    model.eval()

    logits_all = []
    index_all = []

    with torch.no_grad():

        for batch in hold_loader:

            batch = to_device(batch)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).logits

            logits_all.append(
                logits.float().cpu().numpy()
            )

            index_all.append(
                batch["index"].cpu().numpy()
            )

    indices = np.concatenate(index_all)

    assert np.array_equal(
        indices,
        hold_idx
    )

    logits = np.concatenate(logits_all)

    probs = softmax(
        logits.astype(np.float64),
        axis=1
    )

    OOF_PROBS[hold_idx] = probs

    result = {
        "fold": fold,
        "n": len(hold_idx),
        **metrics(
            Y_TRAIN[hold_idx],
            probs
        )
    }

    pred_path = PRED / pred_name
    checkpoint = CKPT / model_name

    np.savez_compressed(
        pred_path,
        sample_id=TRAIN_IDS[hold_idx],
        labels=Y_TRAIN[hold_idx],
        probabilities=probs,
        logits=logits,
        fold=np.full(
            len(hold_idx),
            fold
        )
    )

    torch.save(
        cpu_state(model),
        checkpoint
    )

    meta = {
        "run_hash": run_hash,
        "config": config,
        "result": result,
        "history": fold_history,
        "checkpoint_sha256": sha256(checkpoint),
        "prediction_sha256": sha256(pred_path)
    }

    meta_path = AUDIT / meta_name

    save_json(meta, meta_path)

    # Metadata uploaded last marks a completed run.

    upload_verified(checkpoint, model_name)
    upload_verified(pred_path, pred_name)
    upload_verified(meta_path, meta_name)

    shutil.copy2(
        meta_path,
        BACKUP / meta_name
    )

    push_public(
        f"Stage 7DE: completed OOF fold {fold}"
    )

    OOF_HISTORY.extend(fold_history)
    OOF_RESULTS.append(result)

    checkpoint.unlink()

    del model, optimizer, scheduler, scaler

    clear_gpu()

assert np.isfinite(OOF_PROBS).all()

assert np.allclose(
    OOF_PROBS.sum(1),
    1
)

OOF_TABLE = save_table(
    OOF_RESULTS,
    "stage7de_02_oof_performance"
)

save_table(
    OOF_HISTORY,
    "stage7de_03_oof_training_history"
)

print("\nCombined OOF performance:")

print(
    metrics(
        Y_TRAIN,
        OOF_PROBS
    )
)

# =============================================================================
# STEP 130 — DIFFICULTY AND SPECIALIST SUPERVISION
# =============================================================================

section("STEP 130 — DIFFICULTY-AWARE SUPERVISION")

TRUE_PROB = OOF_PROBS[
    np.arange(N_TRAIN),
    Y_TRAIN
]

OOF_PRED = OOF_PROBS.argmax(1)

OOF_WRONG = (
    OOF_PRED != Y_TRAIN
).astype(np.float32)

OOF_ENTROPY = -(
    OOF_PROBS
    * np.log(
        np.clip(OOF_PROBS, 1e-12, 1)
    )
).sum(1) / np.log(3)

# Difficulty is derived only from predictions generated
# by models that did not train on the respective example.

DIFFICULTY = (
    .50 * (1 - TRUE_PROB)
    + .30 * OOF_WRONG
    + .20 * OOF_ENTROPY
).astype(np.float32)

assert (
    (DIFFICULTY >= 0)
    & (DIFFICULTY <= 1.00001)
).all()

OOF_PATH = PRED / "locked_oof_training_predictions.npz"

np.savez_compressed(
    OOF_PATH,
    sample_id=TRAIN_IDS,
    labels=Y_TRAIN,
    probabilities=OOF_PROBS,
    fold_id=FOLD_ID,
    group_id=OOF_GROUPS,
    difficulty=DIFFICULTY
)

upload_verified(
    OOF_PATH,
    f"locked_oof_{sha256(OOF_PATH)[:12]}.npz"
)

DIFFICULTY_ROWS = []

for label in range(3):

    mask = Y_TRAIN == label

    DIFFICULTY_ROWS.append({
        "class": CLASSES[label],
        "n": int(mask.sum()),
        "oof_error_rate": float(
            OOF_WRONG[mask].mean()
        ),
        "mean_difficulty": float(
            DIFFICULTY[mask].mean()
        ),
        "mean_true_class_probability": float(
            TRUE_PROB[mask].mean()
        )
    })

DIFFICULTY_TABLE = save_table(
    DIFFICULTY_ROWS,
    "stage7de_04_training_difficulty"
)

print(
    DIFFICULTY_TABLE.round(5).to_string(
        index=False
    )
)

# =============================================================================
# STEP 131 — DGSL-NET ARCHITECTURE
# =============================================================================

section("STEP 131 — DGSL-NET ARCHITECTURE")

def masked_mean(x, mask):

    w = mask.unsqueeze(-1).to(x.dtype)

    return (
        (x * w).sum(1)
        / w.sum(1).clamp_min(1)
    )

def masked_softmax(scores, mask):

    scores = scores.float()

    scores = scores.masked_fill(
        ~mask,
        -1e4
    )

    weights = torch.softmax(
        scores,
        dim=-1
    )

    weights = weights * mask.float()

    return (
        weights
        / weights.sum(
            -1,
            keepdim=True
        ).clamp_min(1e-8)
    )

def entropy(p):

    p = p.float().clamp_min(1e-8)

    return -(
        p * p.log()
    ).sum(-1) / math.log(3)

def js_divergence(p, q):

    p = p.float().clamp_min(1e-8)
    q = q.float().clamp_min(1e-8)

    m = (p + q) / 2

    return .5 * (
        (p * (p.log() - m.log())).sum(-1)
        + (q * (q.log() - m.log())).sum(-1)
    )

class PrototypeExpert(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

        # Three trainable prototypes per sentiment class.

        self.prototypes = nn.Parameter(
            torch.randn(3, 3, dim) * .02
        )

        self.scale = nn.Parameter(
            torch.tensor(2.0)
        )

        self.head = nn.Linear(
            dim,
            3
        )

    def forward(self, pooled):

        h = self.project(pooled)

        h_norm = F.normalize(
            h.float(),
            dim=-1
        )

        prototypes = F.normalize(
            self.prototypes.float(),
            dim=-1
        )

        similarity = torch.einsum(
            "bd,ckd->bck",
            h_norm,
            prototypes
        )

        scale = (
            F.softplus(self.scale) + 1
        )

        scores = torch.logsumexp(
            similarity * scale,
            dim=-1
        ) - math.log(3)

        logits = (
            self.head(h).float()
            + scores
        )

        return h, logits

class HardExampleExpert(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(
                dim,
                dim,
                kernel_size=k,
                padding=k // 2
            )
            for k in [1, 3, 5]
        ])

        self.scale_gate = nn.Linear(
            dim,
            1
        )

        self.token_gate = nn.Linear(
            dim,
            1
        )

        self.project = nn.Sequential(
            nn.Linear(dim * 2 + 4, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

        self.head = nn.Linear(
            dim,
            3
        )

    def forward(self, x, valid, ling):

        x = x * valid.unsqueeze(-1).to(x.dtype)

        channels = x.transpose(1, 2)

        local = torch.stack([
            F.gelu(conv(channels)).transpose(1, 2)
            for conv in self.convs
        ], dim=2)

        scale_weights = torch.softmax(
            self.scale_gate(local).squeeze(-1).float(),
            dim=2
        )

        local = (
            local.float()
            * scale_weights.unsqueeze(-1)
        ).sum(2)

        token_weights = masked_softmax(
            self.token_gate(local).squeeze(-1),
            valid
        )

        attended = (
            local
            * token_weights.unsqueeze(-1)
        ).sum(1)

        global_mean = masked_mean(
            x,
            valid
        )

        h = self.project(
            torch.cat([
                attended,
                global_mean,
                ling
            ], dim=-1)
        )

        return h, self.head(h).float()

class GenericExpert(nn.Module):

    def __init__(self, dim, width):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * dim + 4, width),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(width, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.head = nn.Linear(
            dim,
            3
        )

    def forward(self, pooled, ling):

        h = self.net(
            torch.cat([
                pooled,
                ling
            ], dim=-1)
        )

        return h, self.head(h).float()

class DGSLNet(nn.Module):

    def __init__(
        self,
        base,
        mode,
        generic_width=128
    ):

        super().__init__()

        self.mode = mode

        self.encoder = base.electra
        self.base_classifier = base.classifier

        hidden = base.config.hidden_size

        if mode == "standard":

            return

        self.projection = nn.Sequential(
            nn.Linear(hidden, DIM),
            nn.LayerNorm(DIM),
            nn.GELU()
        )

        self.semantic = nn.Sequential(
            nn.Linear(2 * DIM, DIM),
            nn.LayerNorm(DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

        self.semantic_delta = nn.Linear(
            DIM,
            3
        )

        nn.init.zeros_(
            self.semantic_delta.weight
        )

        nn.init.zeros_(
            self.semantic_delta.bias
        )

        if mode == "matched_control":

            self.expert_a = GenericExpert(
                DIM,
                generic_width
            )

            self.expert_b = GenericExpert(
                DIM,
                generic_width
            )

        else:

            self.expert_a = PrototypeExpert(
                DIM
            )

            self.expert_b = HardExampleExpert(
                DIM
            )

        # 3 expert representations + 9 probabilities
        # + 3 entropy values + 3 pairwise divergences
        # + 3 margins + 4 linguistic features.

        router_dim = 3 * DIM + 22

        self.router = nn.Sequential(
            nn.Linear(router_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(.10),
            nn.Linear(96, 3)
        )

        self.competence = nn.Sequential(
            nn.Linear(router_dim, 96),
            nn.GELU(),
            nn.Linear(96, 3)
        )

        nn.init.zeros_(
            self.router[-1].weight
        )

        with torch.no_grad():

            self.router[-1].bias.copy_(
                torch.log(
                    torch.tensor([.60, .20, .20])
                )
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        valid_mask,
        ling
    ):

        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        base_logits = self.base_classifier(
            hidden
        ).float()

        if self.mode == "standard":

            return {
                "logits": base_logits,
                "branches": base_logits.unsqueeze(1),
                "weights": torch.ones(
                    len(base_logits),
                    1,
                    device=base_logits.device
                ),
                "competence": None
            }

        x = self.projection(hidden)

        pooled = torch.cat([
            x[:, 0],
            masked_mean(x, valid_mask)
        ], dim=-1)

        semantic = self.semantic(pooled)

        z0 = (
            base_logits
            + self.semantic_delta(semantic).float()
        )

        if self.mode == "matched_control":

            h1, z1 = self.expert_a(
                pooled,
                ling
            )

            h2, z2 = self.expert_b(
                pooled,
                ling
            )

        else:

            h1, z1 = self.expert_a(
                pooled
            )

            h2, z2 = self.expert_b(
                x,
                valid_mask,
                ling
            )

        branches = torch.stack(
            [z0, z1, z2],
            dim=1
        )

        p = torch.softmax(
            branches.float(),
            dim=-1
        )

        ent = entropy(p)

        sorted_p = p.sort(
            dim=-1
        ).values

        margins = (
            sorted_p[..., -1]
            - sorted_p[..., -2]
        )

        divergence = torch.stack([
            js_divergence(p[:, 0], p[:, 1]),
            js_divergence(p[:, 0], p[:, 2]),
            js_divergence(p[:, 1], p[:, 2])
        ], dim=-1)

        # Stop gradients through routing features so
        # experts cannot manipulate confidence signals
        # solely to attract routing weight.

        router_input = torch.cat([
            semantic.detach(),
            h1.detach(),
            h2.detach(),
            p.detach().flatten(1),
            ent.detach(),
            divergence.detach(),
            margins.detach(),
            ling
        ], dim=-1)

        competence = self.competence(
            router_input
        ).float()

        router_logits = self.router(
            router_input
        ).float()

        if self.mode == "dgsl_fixed":

            weights = torch.tensor(
                [.60, .20, .20],
                device=branches.device
            ).expand(len(branches), -1)

        else:

            raw_weights = torch.softmax(
                router_logits
                + .50 * competence.detach(),
                dim=-1
            )

            # Semantic floor protects the primary expert.
            # All three weights still sum to one.

            weights = .70 * raw_weights

            floor = torch.tensor(
                [.30, 0., 0.],
                device=weights.device
            )

            weights = weights + floor

        final_logits = (
            branches
            * weights.unsqueeze(-1)
        ).sum(1)

        return {
            "logits": final_logits,
            "branches": branches,
            "weights": weights,
            "competence": competence
        }

def parameter_count(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def new_model(
    mode,
    seed=SEED,
    generic_width=128
):

    seed_all(seed)

    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    model = DGSLNet(
        base,
        mode,
        generic_width
    )

    return model

# =============================================================================
# STEP 132 — PARAMETER MATCHING AND TRAINING OBJECTIVES
# =============================================================================

section("STEP 132 — MATCHED CONTROLS AND OBJECTIVES")

base = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

standard_probe = DGSLNet(
    base,
    "standard"
)

BASE_PARAMS = parameter_count(
    standard_probe
)

full_probe = DGSLNet(
    base,
    "dgsl_full"
)

TARGET_PARAMS = parameter_count(
    full_probe
)

generic_probe = DGSLNet(
    base,
    "matched_control",
    generic_width=128
)

generic_params = parameter_count(
    generic_probe
)

# Two generic MLP experts:
# each additional hidden unit adds
# input_dim + 1 + output_dim parameters.

increment = 2 * (
    2 * DIM + 4 + 1 + DIM
)

MATCHED_WIDTH = max(
    8,
    int(round(
        128
        + (TARGET_PARAMS - generic_params)
        / increment
    ))
)

matched_probe = DGSLNet(
    base,
    "matched_control",
    generic_width=MATCHED_WIDTH
)

MATCHED_PARAMS = parameter_count(
    matched_probe
)

EXTRA_TARGET = TARGET_PARAMS - BASE_PARAMS

PARAMETER_MISMATCH = (
    abs(MATCHED_PARAMS - TARGET_PARAMS)
    / max(EXTRA_TARGET, 1)
)

assert PARAMETER_MISMATCH < .01

print("Standard parameters:", BASE_PARAMS)
print("DGSL parameters:", TARGET_PARAMS)
print("Matched parameters:", MATCHED_PARAMS)
print("Matched MLP width:", MATCHED_WIDTH)
print(
    "Extra-parameter mismatch:",
    100 * PARAMETER_MISMATCH,
    "%"
)

del (
    base,
    standard_probe,
    full_probe,
    generic_probe,
    matched_probe
)

clear_gpu()

def weighted_mean(values, weights):

    return (
        values * weights
    ).sum() / weights.sum().clamp_min(1e-8)

def objective(
    output,
    batch,
    mode
):

    y = batch["labels"]

    final = output["logits"]

    primary = F.cross_entropy(
        final,
        y
    )

    if mode == "standard":

        return primary, {
            "primary": primary.detach(),
            "expert": primary.detach() * 0,
            "competence": primary.detach() * 0,
            "routing": primary.detach() * 0,
            "protection": primary.detach() * 0
        }

    branches = output["branches"]

    difficulty = batch[
        "difficulty"
    ].detach()

    ling = batch["ling"]

    operator = (
        ling[:, 0] + ling[:, 1]
    ).clamp(0, 1)

    branch_ce = torch.stack([
        F.cross_entropy(
            branches[:, k],
            y,
            reduction="none"
        )
        for k in range(3)
    ], dim=1)

    # Expert 0: ordinary semantic classification.
    # Expert 1: difficult class boundaries.
    # Expert 2: OOF-hard and linguistic examples.
    #
    # Matched control receives exactly the same
    # specialization and routing objectives.

    semantic_weight = torch.ones_like(
        difficulty
    )

    boundary_weight = (
        1
        + 1.5 * difficulty
        + .50 * (y == 1).float()
    )

    hard_weight = (
        1
        + 2.0 * difficulty
        + .75 * operator
        + .25 * ling[:, 2]
    )

    expert_loss = (
        weighted_mean(
            branch_ce[:, 0],
            semantic_weight
        )
        + weighted_mean(
            branch_ce[:, 1],
            boundary_weight
        )
        + weighted_mean(
            branch_ce[:, 2],
            hard_weight
        )
    ) / 3

    # Explicit competence targets.
    # These are training-only, online expert targets;
    # OOF difficulty remains separately leakage-safe.

    correct = (
        branches.detach().argmax(-1)
        == y.unsqueeze(1)
    ).float()

    competence_loss = (
        F.binary_cross_entropy_with_logits(
            output["competence"],
            correct
        )
    )

    # Soft expert preference target.
    # Uses detached per-example expert loss.

    target = torch.softmax(
        -branch_ce.detach() / .50,
        dim=1
    )

    weights = output[
        "weights"
    ].clamp_min(1e-8)

    routing_loss = -(
        target * weights.log()
    ).sum(1).mean()

    # Penalize degradation on easy examples where
    # the primary semantic branch is already correct.

    final_ce = F.cross_entropy(
        final,
        y,
        reduction="none"
    )

    protection_loss = (
        F.relu(
            final_ce
            - branch_ce[:, 0].detach()
        )
        * correct[:, 0]
        * (1 - difficulty)
    ).mean()

    loss = (
        primary
        + .25 * expert_loss
        + .10 * competence_loss
        + .10 * routing_loss
        + .10 * protection_loss
    )

    return loss, {
        "primary": primary.detach(),
        "expert": expert_loss.detach(),
        "competence": competence_loss.detach(),
        "routing": routing_loss.detach(),
        "protection": protection_loss.detach()
    }

# =============================================================================
# STEP 133 — ARCHITECTURE SMOKE TEST
# =============================================================================

section("STEP 133 — ARCHITECTURE SMOKE TEST")

smoke_loader = make_loader(
    "train",
    indices=np.arange(8),
    difficulty=DIFFICULTY,
    teacher=OOF_PROBS
)

smoke = to_device(
    next(iter(smoke_loader))
)

ARCHITECTURE_ROWS = []

for mode in MODES:

    model = new_model(
        mode,
        generic_width=MATCHED_WIDTH
    ).to(DEVICE)

    output = model(
        smoke["input_ids"],
        smoke["attention_mask"],
        smoke["valid_mask"],
        smoke["ling"]
    )

    loss, components = objective(
        output,
        smoke,
        mode
    )

    assert output["logits"].shape == (8, 3)

    assert torch.isfinite(loss)

    assert torch.allclose(
        output["weights"].sum(1),
        torch.ones(8, device=DEVICE),
        atol=1e-5
    )

    loss.backward()

    grads = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]

    assert grads

    assert all(
        torch.isfinite(g).all().item()
        for g in grads
    )

    if mode in [
        "dgsl_full",
        "matched_control"
    ]:

        assert model.router[-1].weight.grad is not None

        assert (
            model.router[-1].weight.grad.abs().sum() > 0
        )

    ARCHITECTURE_ROWS.append({
        "mode": mode,
        "parameters": parameter_count(model),
        "extra_parameters": (
            parameter_count(model)
            - BASE_PARAMS
        ),
        "smoke_test": "PASS"
    })

    print(
        mode,
        "| parameters:",
        parameter_count(model),
        "| forward/backward: PASS"
    )

    del model, output, loss, grads

    clear_gpu()

save_table(
    ARCHITECTURE_ROWS,
    "stage7de_05_architecture_audit"
)

# =============================================================================
# STEP 134 — VALIDATION AND TRAINING UTILITIES
# =============================================================================

section("STEP 134 — TRAINING UTILITIES")

def evaluate(model, loader):

    model.eval()

    logits = []
    branches = []
    weights = []
    competence = []
    labels = []
    indices = []

    with torch.no_grad():

        for batch in loader:

            batch = to_device(batch)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                out = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["valid_mask"],
                    batch["ling"]
                )

            logits.append(
                out["logits"].float().cpu().numpy()
            )

            branches.append(
                out["branches"].float().cpu().numpy()
            )

            weights.append(
                out["weights"].float().cpu().numpy()
            )

            if out["competence"] is not None:

                competence.append(
                    torch.sigmoid(
                        out["competence"]
                    ).cpu().numpy()
                )

            labels.append(
                batch["labels"].cpu().numpy()
            )

            indices.append(
                batch["index"].cpu().numpy()
            )

    result = {
        "logits": np.concatenate(logits),
        "branches": np.concatenate(branches),
        "weights": np.concatenate(weights),
        "labels": np.concatenate(labels),
        "indices": np.concatenate(indices)
    }

    result["competence"] = (
        np.concatenate(competence)
        if competence
        else np.empty(
            (len(result["labels"]), 0)
        )
    )

    result["probabilities"] = softmax(
        result["logits"].astype(np.float64),
        axis=1
    )

    return result

CONFIG = {
    "stage": "7DE",
    "model": MODEL_NAME,
    "seed": SEED,
    "oof_folds": FOLDS,
    "oof_epochs": OOF_EPOCHS,
    "final_epochs": FINAL_EPOCHS,
    "max_length": MAX_LENGTH,
    "train_batch": TRAIN_BATCH,
    "encoder_lr": ENCODER_LR,
    "head_lr": HEAD_LR,
    "weight_decay": WEIGHT_DECAY,
    "warmup": WARMUP,
    "dim": DIM,
    "matched_width": MATCHED_WIDTH,
    "modes": MODES,
    "oof_sha256": sha256(OOF_PATH),
    "source_sha256": hashlib.sha256(
        SOURCE.encode()
    ).hexdigest(),
    "test_evaluated": False
}

save_json(
    CONFIG,
    AUDIT / "stage7de_config.json"
)

save_json(
    CONFIG,
    BACKUP / "stage7de_config.json"
)

push_public(
    "Stage 7DE: lock architecture and training configuration"
)

# =============================================================================
# STEP 135 — TRAIN FOUR COMPETITIVE ARCHITECTURES
# =============================================================================

section("STEP 135 — DGSL MODEL TRAINING")

RESULTS = []
HISTORY = []
OUTPUTS = {}

for mode in MODES:

    prefix = f"{mode}_seed{SEED}"

    model_name = prefix + "_model.pt"
    pred_name = prefix + "_val.npz"
    meta_name = prefix + "_completed.json"

    run_config = {
        **CONFIG,
        "mode": mode
    }

    run_hash = run_fingerprint(
        run_config
    )

    if all(
        n in REMOTE
        for n in [
            model_name,
            pred_name,
            meta_name
        ]
    ):

        meta_path = download_asset(
            REMOTE[meta_name],
            AUDIT / meta_name
        )

        meta = json.loads(
            meta_path.read_text()
        )

        assert meta["run_hash"] == run_hash

        pred_path = download_asset(
            REMOTE[pred_name],
            PRED / pred_name
        )

        assert sha256(pred_path) == meta["prediction_sha256"]

        remote_matches(
            model_name,
            meta["checkpoint_sha256"]
        )

        with np.load(
            pred_path,
            allow_pickle=False
        ) as z:

            assert np.array_equal(
                z["sample_id"],
                VAL_IDS
            )

            assert np.array_equal(
                z["labels"],
                Y_VAL
            )

            OUTPUTS[mode] = {
                key: z[key]
                for key in z.files
                if key != "sample_id"
            }

        RESULTS.append(meta["result"])
        HISTORY.extend(meta["history"])

        print(mode, "RECOVERED — training skipped")

        continue

    seed_all(SEED)

    model = new_model(
        mode,
        seed=SEED,
        generic_width=MATCHED_WIDTH
    ).to(DEVICE)

    train_loader = make_loader(
        "train",
        training=True,
        difficulty=DIFFICULTY,
        teacher=OOF_PROBS,
        seed=SEED
    )

    val_loader = make_loader(
        "val"
    )

    optimizer, scheduler = make_optimizer(
        model,
        len(train_loader) * FINAL_EPOCHS,
        custom=True
    )

    scaler = scaler_factory()

    best_f1 = -1
    best_epoch = None
    best_output = None

    checkpoint = CKPT / model_name

    run_history = []

    print("\nTRAINING:", mode)

    for epoch in range(
        1,
        FINAL_EPOCHS + 1
    ):

        model.train()

        totals = {
            "loss": 0.,
            "primary": 0.,
            "expert": 0.,
            "competence": 0.,
            "routing": 0.,
            "protection": 0.
        }

        seen = 0

        for batch in train_loader:

            batch = to_device(batch)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                out = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["valid_mask"],
                    batch["ling"]
                )

                loss, components = objective(
                    out,
                    batch,
                    mode
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            previous_scale = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            if scaler.get_scale() >= previous_scale:

                scheduler.step()

            n = len(batch["labels"])

            totals["loss"] += (
                loss.item() * n
            )

            for key, value in components.items():

                totals[key] += (
                    value.item() * n
                )

            seen += n

        validation = evaluate(
            model,
            val_loader
        )

        assert np.array_equal(
            validation["labels"],
            Y_VAL
        )

        assert np.array_equal(
            validation["indices"],
            np.arange(N_VAL)
        )

        score = metrics(
            Y_VAL,
            validation["probabilities"]
        )

        row = {
            "mode": mode,
            "seed": SEED,
            "epoch": epoch,
            **{
                f"train_{k}": v / seen
                for k, v in totals.items()
            },
            **{
                f"val_{k}": v
                for k, v in score.items()
            }
        }

        run_history.append(row)

        print(
            f"{mode} | epoch {epoch}/{FINAL_EPOCHS} "
            f"| loss={row['train_loss']:.4f} "
            f"| val_acc={score['accuracy']:.5f} "
            f"| val_f1={score['macro_f1']:.5f}"
        )

        if score["macro_f1"] > best_f1:

            best_f1 = score["macro_f1"]
            best_epoch = epoch
            best_output = validation

            torch.save(
                cpu_state(model),
                checkpoint
            )

    assert best_output is not None

    result = {
        "model": mode,
        "seed": SEED,
        "best_epoch": best_epoch,
        "parameters": parameter_count(model),
        **metrics(
            Y_VAL,
            best_output["probabilities"]
        )
    }

    pred_path = PRED / pred_name

    np.savez_compressed(
        pred_path,
        sample_id=VAL_IDS,
        **best_output
    )

    meta = {
        "run_hash": run_hash,
        "config": run_config,
        "result": result,
        "history": run_history,
        "checkpoint_sha256": sha256(checkpoint),
        "prediction_sha256": sha256(pred_path)
    }

    meta_path = AUDIT / meta_name

    save_json(
        meta,
        meta_path
    )

    # Permanent backup immediately after every model.

    upload_verified(
        checkpoint,
        model_name
    )

    upload_verified(
        pred_path,
        pred_name
    )

    upload_verified(
        meta_path,
        meta_name
    )

    shutil.copy2(
        meta_path,
        BACKUP / meta_name
    )

    RESULTS.append(result)
    HISTORY.extend(run_history)

    OUTPUTS[mode] = best_output

    save_json(
        RESULTS,
        BACKUP / "completed_model_results.json"
    )

    push_public(
        f"Stage 7DE: completed and verified {mode}"
    )

    checkpoint.unlink()

    del (
        model,
        optimizer,
        scheduler,
        scaler,
        validation
    )

    clear_gpu()

    print(
        "BEST RESULT:",
        result
    )

RESULT_TABLE = save_table(
    RESULTS,
    "stage7de_06_model_results"
)

HISTORY_TABLE = save_table(
    HISTORY,
    "stage7de_07_training_history"
)

# =============================================================================
# STEP 136 — ENSEMBLE AND ARCHITECTURE COMPARISON
# =============================================================================

section("STEP 136 — MODEL COMPARISON")

PROBS = {
    mode: OUTPUTS[mode]["probabilities"]
    for mode in MODES
}

PROBS.update(HISTORICAL)

PROBS["dgsl_pair_ensemble"] = (
    PROBS["dgsl_fixed"]
    + PROBS["dgsl_full"]
) / 2

PROBS["prototype_plus_dgsl"] = (
    PROBS["prototype_family"]
    + PROBS["dgsl_full"]
) / 2

ALL_RESULTS = save_table(
    [
        {
            "model": name,
            **metrics(Y_VAL, probs)
        }
        for name, probs in PROBS.items()
    ],
    "stage7de_08_all_comparisons"
).sort_values(
    "macro_f1",
    ascending=False
)

print(
    ALL_RESULTS.round(6).to_string(
        index=False
    )
)

# =============================================================================
# STEP 137 — PAIRED STATISTICAL ANALYSIS
# =============================================================================

section("STEP 137 — PAIRED STATISTICS")

rng = np.random.default_rng(SEED)

group_indices = [
    np.flatnonzero(
        GROUP_VAL == group
    )
    for group in np.unique(GROUP_VAL)
]

BOOT = [
    np.concatenate([
        group_indices[j]
        for j in rng.integers(
            0,
            len(group_indices),
            len(group_indices)
        )
    ])
    for _ in range(BOOTSTRAPS)
]

def fast_f1(y, pred):

    cm = np.bincount(
        3 * y + pred,
        minlength=9
    ).reshape(3, 3)

    tp = np.diag(cm)

    denominator = (
        cm.sum(0) + cm.sum(1)
    )

    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros(3),
        where=denominator > 0
    ).mean()

COMPARISONS = [
    ("dgsl_full", "standard"),
    ("dgsl_full", "matched_control"),
    ("dgsl_full", "dgsl_fixed"),
    ("dgsl_full", "prototype_family"),
    ("dgsl_fixed", "standard"),
    ("dgsl_fixed", "matched_control"),
    ("prototype_plus_dgsl", "prototype_family")
]

STATS_ROWS = []

for candidate, reference in COMPARISONS:

    a = PROBS[candidate].argmax(1)
    b = PROBS[reference].argmax(1)

    gains = int(
        ((a == Y_VAL) & (b != Y_VAL)).sum()
    )

    losses = int(
        ((a != Y_VAL) & (b == Y_VAL)).sum()
    )

    differences = np.array([
        fast_f1(
            Y_VAL[idx],
            a[idx]
        )
        - fast_f1(
            Y_VAL[idx],
            b[idx]
        )
        for idx in BOOT
    ])

    p = (
        binomtest(
            gains,
            gains + losses,
            .5
        ).pvalue
        if gains + losses > 0
        else 1.0
    )

    STATS_ROWS.append({
        "candidate": candidate,
        "reference": reference,
        "delta_macro_f1": (
            fast_f1(Y_VAL, a)
            - fast_f1(Y_VAL, b)
        ),
        "ci95_low": np.quantile(
            differences, .025
        ),
        "ci95_high": np.quantile(
            differences, .975
        ),
        "gained_correct": gains,
        "lost_correct": losses,
        "mcnemar_p": p
    })

# Holm adjustment across all listed comparisons.

pvalues = np.array([
    row["mcnemar_p"]
    for row in STATS_ROWS
])

order = np.argsort(pvalues)

adjusted = np.empty(
    len(pvalues)
)

running = 0.

for rank, idx in enumerate(order):

    running = max(
        running,
        (len(pvalues) - rank)
        * pvalues[idx]
    )

    adjusted[idx] = min(
        running,
        1.
    )

for i, row in enumerate(STATS_ROWS):

    row["holm_p"] = adjusted[i]

STATS = save_table(
    STATS_ROWS,
    "stage7de_09_paired_statistics"
)

print(
    STATS.round(6).to_string(
        index=False
    )
)

# =============================================================================
# STEP 138 — SPECIALIST AND ROUTER BEHAVIOR
# =============================================================================

section("STEP 138 — SPECIALIST BEHAVIOR")

SUBGROUP_ROWS = []
CLASS_ROWS = []
CONFUSION_ROWS = []

for name, probs in PROBS.items():

    pred = probs.argmax(1)

    for subgroup, mask in MASKS.items():

        if not mask.any():
            continue

        SUBGROUP_ROWS.append({
            "model": name,
            "subgroup": subgroup,
            "n": int(mask.sum()),
            "accuracy": accuracy_score(
                Y_VAL[mask],
                pred[mask]
            ),
            "macro_f1": macro_f1(
                Y_VAL[mask],
                pred[mask]
            )
        })

    report = classification_report(
        Y_VAL,
        pred,
        labels=[0, 1, 2],
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    for label in CLASSES:

        CLASS_ROWS.append({
            "model": name,
            "class": label,
            **report[label]
        })

    cm = confusion_matrix(
        Y_VAL,
        pred,
        labels=[0, 1, 2]
    )

    for i in range(3):

        for j in range(3):

            CONFUSION_ROWS.append({
                "model": name,
                "true_class": CLASSES[i],
                "predicted_class": CLASSES[j],
                "count": int(cm[i, j])
            })

SUBGROUPS = save_table(
    SUBGROUP_ROWS,
    "stage7de_10_linguistic_subgroups"
)

CLASSWISE = save_table(
    CLASS_ROWS,
    "stage7de_11_classwise_results"
)

save_table(
    CONFUSION_ROWS,
    "stage7de_12_confusion_matrices"
)

ROUTER_ROWS = []
EXPERT_ROWS = []
RESCUE_ROWS = []

for mode in [
    "matched_control",
    "dgsl_fixed",
    "dgsl_full"
]:

    out = OUTPUTS[mode]

    weights = out["weights"]

    branch_probs = softmax(
        out["branches"].astype(np.float64),
        axis=-1
    )

    branch_pred = branch_probs.argmax(-1)

    correct = (
        branch_pred
        == Y_VAL[:, None]
    )

    primary_correct = correct[:, 0]

    final_correct = (
        out["probabilities"].argmax(1)
        == Y_VAL
    )

    competence = out["competence"]

    oracle_accuracy = correct.any(1).mean()

    RESCUE_ROWS.append({
        "model": mode,
        "semantic_accuracy": primary_correct.mean(),
        "final_accuracy": final_correct.mean(),
        "expert_selection_oracle_accuracy":
            oracle_accuracy,
        "semantic_errors_rescued": int(
            (
                ~primary_correct
                & final_correct
            ).sum()
        ),
        "semantic_correct_predictions_lost": int(
            (
                primary_correct
                & ~final_correct
            ).sum()
        ),
        "all_experts_wrong": int(
            (~correct.any(1)).sum()
        ),
        "expert_prediction_disagreement": float(
            (
                (branch_pred[:, 0] != branch_pred[:, 1])
                | (branch_pred[:, 0] != branch_pred[:, 2])
            ).mean()
        )
    })

    for k, expert in enumerate([
        "semantic",
        "boundary",
        "hard_example"
    ]):

        competence_auc = np.nan

        if len(np.unique(correct[:, k])) == 2:

            competence_auc = roc_auc_score(
                correct[:, k],
                competence[:, k]
            )

        EXPERT_ROWS.append({
            "model": mode,
            "expert": expert,
            **metrics(
                Y_VAL,
                branch_probs[:, k]
            ),
            "competence_auroc":
                competence_auc,
            "unique_correct_examples": int(
                (
                    correct[:, k]
                    & (correct.sum(1) == 1)
                ).sum()
            )
        })

        for subgroup, mask in MASKS.items():

            if not mask.any():
                continue

            ROUTER_ROWS.append({
                "model": mode,
                "expert": expert,
                "subgroup": subgroup,
                "n": int(mask.sum()),
                "mean_weight": weights[mask, k].mean(),
                "weight_sd": weights[mask, k].std(),
                "expert_accuracy": correct[mask, k].mean()
            })

ROUTER = save_table(
    ROUTER_ROWS,
    "stage7de_13_router_weights"
)

EXPERTS = save_table(
    EXPERT_ROWS,
    "stage7de_14_expert_competence"
)

RESCUES = save_table(
    RESCUE_ROWS,
    "stage7de_15_specialist_rescues"
)

print("\nSPECIALIST PERFORMANCE")

print(
    EXPERTS.round(5).to_string(
        index=False
    )
)

print("\nROUTER BEHAVIOR")

print(
    ROUTER[
        ROUTER["subgroup"] == "all"
    ].round(5).to_string(
        index=False
    )
)

print("\nRESCUE ANALYSIS")

print(
    RESCUES.round(5).to_string(
        index=False
    )
)

# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def ece(y, p, bins=15):

    confidence = p.max(1)
    correct = p.argmax(1) == y

    result = 0.

    edges = np.linspace(
        0, 1, bins + 1
    )

    for i in range(bins):

        mask = (
            (confidence >= edges[i])
            & (
                confidence < edges[i + 1]
                if i < bins - 1
                else confidence <= edges[i + 1]
            )
        )

        if mask.any():

            result += (
                mask.mean()
                * abs(
                    correct[mask].mean()
                    - confidence[mask].mean()
                )
            )

    return float(result)

CALIBRATION = save_table(
    [
        {
            "model": name,
            "ece": ece(Y_VAL, probs),
            "nll": log_loss(
                Y_VAL,
                probs,
                labels=[0, 1, 2]
            ),
            "brier": np.mean(
                np.sum(
                    (
                        probs
                        - np.eye(3)[Y_VAL]
                    )**2,
                    axis=1
                )
            )
        }
        for name, probs in PROBS.items()
    ],
    "stage7de_16_calibration"
)

# =============================================================================
# STEP 139 — PUBLICATION-READY FIGURES
# =============================================================================

section("STEP 139 — PUBLICATION FIGURES")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

FIGURE_MANIFEST = []

def save_figure(fig, name, caption):

    for fmt in [
        "png",
        "pdf",
        "svg"
    ]:

        path = FIGURES / fmt / f"{name}.{fmt}"

        fig.savefig(
            path,
            dpi=600 if fmt == "png" else None
        )

        shutil.copy2(
            path,
            PUB / "figures" / fmt / path.name
        )

        destination = (
            BACKUP / "figures" / fmt / path.name
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            path,
            destination
        )

        FIGURE_MANIFEST.append({
            "figure": name,
            "format": fmt,
            "sha256": sha256(path),
            "caption": caption
        })

    caption_path = (
        PUB / "captions" / f"{name}.txt"
    )

    caption_path.write_text(
        caption,
        encoding="utf-8"
    )

    destination = (
        BACKUP / "captions" / caption_path.name
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        caption_path,
        destination
    )

    save_json(
        FIGURE_MANIFEST,
        BACKUP / "figure_manifest.json"
    )

    # Every generated figure is pushed immediately.

    push_public(
        "Stage 7DE figure: " + name
    )

    plt.show()
    plt.close(fig)

# Figure 1 — OOF fold performance.

fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

ax.bar(
    OOF_TABLE["fold"].astype(str),
    OOF_TABLE["macro_f1"]
)

ax.set_ylim(0, 1)
ax.set_xlabel("Held-out training fold")
ax.set_ylabel("Macro-F1")

save_figure(
    fig,
    "stage7de_01_oof_performance",
    "Group-disjoint out-of-fold training prediction performance."
)

# Figure 2 — Training difficulty distribution.

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for k, label in enumerate(CLASSES):

    ax.hist(
        DIFFICULTY[Y_TRAIN == k],
        bins=25,
        density=True,
        histtype="step",
        linewidth=2,
        label=label
    )

ax.set_xlabel("OOF-derived difficulty")
ax.set_ylabel("Density")
ax.legend(frameon=False)

save_figure(
    fig,
    "stage7de_02_difficulty_distribution",
    "Training-only difficulty distributions derived from held-out fold predictions."
)

# Figure 3 — Model comparison.

fig, ax = plt.subplots(
    figsize=(10, 6)
)

plot = ALL_RESULTS.sort_values(
    "macro_f1"
)

ax.barh(
    plot["model"],
    100 * plot["macro_f1"]
)

ax.set_xlim(0, 100)

ax.set_xlabel(
    "Validation macro-F1 (%)"
)

for i, value in enumerate(
    plot["macro_f1"]
):

    ax.text(
        100 * value + .3,
        i,
        f"{100 * value:.2f}",
        va="center",
        fontsize=9
    )

save_figure(
    fig,
    "stage7de_03_model_comparison",
    "DGSL-Net and control model validation performance."
)

# Figure 4 — Validation learning curves.

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for mode in MODES:

    part = HISTORY_TABLE[
        HISTORY_TABLE["mode"] == mode
    ]

    ax.plot(
        part["epoch"],
        part["val_macro_f1"],
        marker="o",
        label=mode
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation macro-F1")
ax.legend(frameon=False)

save_figure(
    fig,
    "stage7de_04_learning_curves",
    "Validation learning curves under the shared training budget."
)

# Figure 5 — Linguistic subgroup heatmap.

chosen = [
    "standard",
    "matched_control",
    "dgsl_fixed",
    "dgsl_full",
    "prototype_family"
]

pivot = SUBGROUPS[
    SUBGROUPS["model"].isin(chosen)
].pivot(
    index="subgroup",
    columns="model",
    values="macro_f1"
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

im = ax.imshow(
    pivot.to_numpy(),
    cmap="YlGnBu",
    vmin=0,
    vmax=1
)

ax.set_xticks(
    range(len(pivot.columns)),
    pivot.columns,
    rotation=35,
    ha="right"
)

ax.set_yticks(
    range(len(pivot.index)),
    pivot.index
)

for i in range(len(pivot.index)):

    for j in range(len(pivot.columns)):

        value = pivot.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=(
                "white"
                if value > .7
                else "black"
            ),
            fontsize=9
        )

fig.colorbar(
    im,
    ax=ax,
    label="Macro-F1"
)

save_figure(
    fig,
    "stage7de_05_linguistic_subgroups",
    "Validation performance across linguistic candidate subgroups."
)

# Figure 6 — Classwise performance.

fig, ax = plt.subplots(
    figsize=(10, 5)
)

CLASSWISE[
    CLASSWISE["model"].isin(chosen)
].pivot(
    index="class",
    columns="model",
    values="f1-score"
).plot(
    kind="bar",
    ax=ax,
    rot=0
)

ax.set_ylim(0, 1)
ax.set_ylabel("Class-specific F1")
ax.legend(
    frameon=False,
    fontsize=8
)

save_figure(
    fig,
    "stage7de_06_classwise_performance",
    "Class-specific sentiment classification performance."
)

# Figure 7 — Router weight distributions.

fig, ax = plt.subplots(
    figsize=(9, 5)
)

weights = OUTPUTS["dgsl_full"]["weights"]

ax.boxplot(
    [
        weights[:, 0],
        weights[:, 1],
        weights[:, 2]
    ],
    labels=[
        "Semantic",
        "Boundary",
        "Hard example"
    ],
    showfliers=False
)

ax.set_ylim(0, 1)
ax.set_ylabel("Routing weight")

save_figure(
    fig,
    "stage7de_07_router_distribution",
    "Distribution of sample-dependent expert routing weights."
)

# Figure 8 — Routing across linguistic subgroups.

pivot = ROUTER[
    ROUTER["model"] == "dgsl_full"
].pivot(
    index="subgroup",
    columns="expert",
    values="mean_weight"
)

fig, ax = plt.subplots(
    figsize=(11, 5)
)

pivot.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    rot=30
)

ax.set_ylim(0, 1)
ax.set_ylabel("Mean routing weight")
ax.legend(frameon=False)

save_figure(
    fig,
    "stage7de_08_subgroup_routing",
    "Mean DGSL expert contribution across linguistic subgroups."
)

# Figure 9 — Specialist rescue and loss.

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.barh(
    RESCUES["model"],
    RESCUES["semantic_errors_rescued"],
    label="Rescued"
)

ax.barh(
    RESCUES["model"],
    -RESCUES["semantic_correct_predictions_lost"],
    label="Lost"
)

ax.axvline(
    0,
    color="gray",
    linewidth=1
)

ax.set_xlabel(
    "Number of validation samples"
)

ax.legend(frameon=False)

save_figure(
    fig,
    "stage7de_09_specialist_rescues",
    "Corrected semantic-expert errors and damaged correct predictions."
)

# Figure 10 — Paired statistical comparisons.

fig, ax = plt.subplots(
    figsize=(11, 6)
)

labels = []

for i, row in STATS.iterrows():

    ax.plot(
        [
            100 * row["ci95_low"],
            100 * row["ci95_high"]
        ],
        [i, i],
        linewidth=2
    )

    ax.scatter(
        100 * row["delta_macro_f1"],
        i
    )

    labels.append(
        row["candidate"]
        + " vs "
        + row["reference"]
    )

ax.axvline(
    0,
    linestyle="--",
    color="gray"
)

ax.set_yticks(
    range(len(labels)),
    labels
)

ax.set_xlabel(
    "Macro-F1 difference (percentage points)"
)

save_figure(
    fig,
    "stage7de_10_paired_effects",
    "Exploratory paired group-bootstrap confidence intervals."
)

# Figure 11 — Expert competence discrimination.

fig, ax = plt.subplots(
    figsize=(9, 5)
)

EXPERTS.pivot(
    index="expert",
    columns="model",
    values="competence_auroc"
).plot(
    kind="bar",
    ax=ax,
    rot=0
)

ax.axhline(
    .5,
    linestyle="--",
    color="gray"
)

ax.set_ylim(0, 1)
ax.set_ylabel("Competence AUROC")
ax.legend(
    frameon=False,
    fontsize=8
)

save_figure(
    fig,
    "stage7de_11_expert_competence",
    "Ability of competence heads to predict expert correctness."
)

# Figure 12 — Confusion matrix.

fig, ax = plt.subplots(
    figsize=(6, 5)
)

cm = confusion_matrix(
    Y_VAL,
    PROBS["dgsl_full"].argmax(1),
    labels=[0, 1, 2]
)

im = ax.imshow(
    cm,
    cmap="Blues"
)

ax.set_xticks(
    range(3),
    CLASSES
)

ax.set_yticks(
    range(3),
    CLASSES
)

ax.set_xlabel("Predicted sentiment")
ax.set_ylabel("Recorded sentiment")

for i in range(3):

    for j in range(3):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > cm.max() / 2
                else "black"
            )
        )

fig.colorbar(im, ax=ax)

save_figure(
    fig,
    "stage7de_12_confusion_matrix",
    "DGSL-Net validation confusion matrix."
)

# Figure 13 — Architecture diagram.

from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch
)

fig, ax = plt.subplots(
    figsize=(13, 10)
)

ax.set_xlim(0, 13)
ax.set_ylim(0, 11)
ax.axis("off")

def architecture_box(
    x, y, w, h,
    label,
    color="#e2e8f0"
):

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=.04",
        facecolor=color,
        edgecolor="#334155",
        linewidth=1.4
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=10
    )

def arrow(
    a,
    b,
    dashed=False
):

    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color="#475569",
            linestyle=(
                "--"
                if dashed
                else "-"
            )
        )
    )

architecture_box(
    4.5, 9.8, 4, .7,
    "Bangla text"
)

architecture_box(
    4.0, 8.5, 5, .8,
    "BanglaBERT shared encoder",
    "#dbeafe"
)

architecture_box(
    4.0, 7.2, 5, .8,
    "Token projection and pooled representation"
)

architecture_box(
    .3, 5.4, 3.7, 1.1,
    "Semantic expert\nBase classifier + residual",
    "#dbeafe"
)

architecture_box(
    4.65, 5.4, 3.7, 1.1,
    "Boundary expert\nMulti-prototype representation",
    "#ccfbf1"
)

architecture_box(
    9.0, 5.4, 3.7, 1.1,
    "Hard-example expert\nMulti-scale phrase features",
    "#ede9fe"
)

architecture_box(
    3.7, 3.8, 5.6, .9,
    "Expert disagreement and uncertainty features",
    "#fef3c7"
)

architecture_box(
    3.7, 2.4, 5.6, .9,
    "Competence-aware routing\nSemantic weight floor",
    "#ccfbf1"
)

architecture_box(
    4.5, .9, 4, .8,
    "Weighted sentiment logits"
)

architecture_box(
    .2, 2.0, 2.9, 1.5,
    "Training-only\nGroup-disjoint OOF\ndifficulty supervision",
    "#f1f5f9"
)

architecture_box(
    10.0, 2.0, 2.8, 1.5,
    "Training-only\nExpert competence\nand routing targets",
    "#f1f5f9"
)

arrow((6.5, 9.8), (6.5, 9.3))
arrow((6.5, 8.5), (6.5, 8.0))

for x in [2.15, 6.5, 10.85]:

    arrow(
        (6.5, 7.2),
        (x, 6.5)
    )

    arrow(
        (x, 5.4),
        (6.5, 4.7)
    )

arrow((6.5, 3.8), (6.5, 3.3))
arrow((6.5, 2.4), (6.5, 1.7))

arrow(
    (3.1, 3.0),
    (4.65, 5.5),
    dashed=True
)

arrow(
    (3.1, 3.3),
    (9.0, 5.5),
    dashed=True
)

arrow(
    (10.0, 2.8),
    (9.3, 2.8),
    dashed=True
)

ax.set_title(
    "Disagreement-Guided Specialist Learning Network",
    fontsize=16,
    fontweight="bold"
)

save_figure(
    fig,
    "stage7de_13_dgsl_architecture",
    "Provisional DGSL-Net architecture. Dashed arrows denote training-only supervision."
)

for fmt in ["png", "pdf", "svg"]:

    shutil.copy2(
        FIGURES / fmt / f"stage7de_13_dgsl_architecture.{fmt}",
        PUB / "architecture" /
        f"stage7de_13_dgsl_architecture.{fmt}"
    )

# Figure 14 — Calibration comparison.

fig, ax = plt.subplots(
    figsize=(9, 5)
)

plot = CALIBRATION.sort_values(
    "ece",
    ascending=False
)

ax.barh(
    plot["model"],
    plot["ece"]
)

ax.set_xlabel(
    "Expected calibration error"
)

save_figure(
    fig,
    "stage7de_14_calibration",
    "Uncalibrated validation expected calibration error."
)

# =============================================================================
# STEP 140 — FINAL RESEARCH SYNTHESIS AND PERMANENT BACKUP
# =============================================================================

section("STEP 140 — VERIFIED PERMANENT BACKUP")

full_f1 = metrics(
    Y_VAL,
    PROBS["dgsl_full"]
)["macro_f1"]

standard_f1 = metrics(
    Y_VAL,
    PROBS["standard"]
)["macro_f1"]

matched_f1 = metrics(
    Y_VAL,
    PROBS["matched_control"]
)["macro_f1"]

fixed_f1 = metrics(
    Y_VAL,
    PROBS["dgsl_fixed"]
)["macro_f1"]

historical_f1 = metrics(
    Y_VAL,
    PROBS["prototype_family"]
)["macro_f1"]

full_vs_matched = STATS[
    (STATS["candidate"] == "dgsl_full")
    & (STATS["reference"] == "matched_control")
].iloc[0]

full_vs_standard = STATS[
    (STATS["candidate"] == "dgsl_full")
    & (STATS["reference"] == "standard")
].iloc[0]

full_vs_fixed = STATS[
    (STATS["candidate"] == "dgsl_full")
    & (STATS["reference"] == "dgsl_fixed")
].iloc[0]

if (
    full_f1 >= .85
    and full_f1 > historical_f1
    and full_vs_matched["ci95_low"] > 0
    and full_vs_standard["ci95_low"] > 0
    and full_vs_fixed["delta_macro_f1"] > 0
):

    DECISION = (
        "STRONG SCREENING RESULT — "
        "PROCEED TO MULTI-SEED CONFIRMATION"
    )

elif (
    full_f1 > matched_f1
    and full_f1 > standard_f1
    and full_f1 > fixed_f1
):

    DECISION = (
        "PROMISING — "
        "CONFIRM ACROSS SEEDS BEFORE FINALIZATION"
    )

else:

    DECISION = (
        "REFINEMENT REQUIRED — "
        "DO NOT CLAIM ARCHITECTURAL SUPERIORITY"
    )

SUMMARY = {
    "stage": "7DE",
    "architecture": "DGSL-Net",
    "training_samples": N_TRAIN,
    "validation_samples": N_VAL,
    "oof_folds": FOLDS,
    "oof_complete": True,
    "oof_metrics": metrics(
        Y_TRAIN,
        OOF_PROBS
    ),
    "standard_macro_f1": standard_f1,
    "matched_control_macro_f1": matched_f1,
    "dgsl_fixed_macro_f1": fixed_f1,
    "dgsl_full_macro_f1": full_f1,
    "previous_best_macro_f1": historical_f1,
    "delta_vs_standard": full_f1 - standard_f1,
    "delta_vs_matched": full_f1 - matched_f1,
    "delta_vs_fixed": full_f1 - fixed_f1,
    "delta_vs_previous_best": full_f1 - historical_f1,
    "best_overall_model":
        ALL_RESULTS.iloc[0]["model"],
    "best_overall_macro_f1":
        float(ALL_RESULTS.iloc[0]["macro_f1"]),
    "matched_parameter_difference":
        PARAMETER_MISMATCH,
    "new_unique_figures": len(
        set(
            x["figure"]
            for x in FIGURE_MANIFEST
        )
    ),
    "new_tables": len(
        list(TABLES.glob("*.csv"))
    ),
    "decision": DECISION,
    "training_seeds": [SEED],
    "test_evaluated": False,
    "external_evaluated": False,
    "novelty_verified": False,
    "limitations": [
        "Single-seed architecture screening.",
        "OOF difficulty comes from a single model family.",
        "Online competence targets are not cross-fitted expert predictions.",
        "Validation labels used for final checkpoint selection.",
        "Validation repeatedly reused during development.",
        "Historical ensemble has a different inference budget.",
        "Prototype and hard-example components require separate ablations.",
        "No independent external evaluation.",
        "No architectural novelty claim established."
    ]
}

save_json(
    SUMMARY,
    AUDIT / "stage7de_summary.json"
)

save_json(
    SUMMARY,
    BACKUP / "stage7de_summary.json"
)

pd.DataFrame(
    FIGURE_MANIFEST
).to_csv(
    PUB / "stage7de_figure_manifest.csv",
    index=False
)

for path in TABLES.glob("*.csv"):

    destination = (
        BACKUP / "tables" / path.name
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        path,
        destination
    )

    tex = (
        PUB / "tables/latex" /
        (path.stem + ".tex")
    )

    if tex.exists():

        destination = (
            BACKUP / "latex_tables" / tex.name
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            tex,
            destination
        )

# Save tokenizer and model configuration for recovery.

TOKENIZER.save_pretrained(
    AUDIT / "tokenizer"
)

from transformers import AutoConfig

AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=3
).save_pretrained(
    AUDIT / "backbone_config"
)

# Record private aligned predictions.

np.savez_compressed(
    PRED / "stage7de_all_validation_predictions.npz",
    sample_id=VAL_IDS,
    labels=Y_VAL,
    group_id=GROUP_VAL,
    **{
        name: probs
        for name, probs in PROBS.items()
    }
)

# Compact reproducibility package.
# Large weights are already stored individually.

compact = (
    WORK / "sentiment_stage7de_compact.zip"
)

with zipfile.ZipFile(
    compact,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for folder in [
        AUDIT,
        TABLES,
        PRED,
        FIGURES
    ]:

        for path in folder.rglob("*"):

            if path.is_file():

                z.write(
                    path,
                    path.relative_to(OUT)
                )

publication = (
    WORK / "sentiment_stage7de_publication.zip"
)

with zipfile.ZipFile(
    publication,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for path in PUB.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(PUB)
            )

for path in [
    compact,
    publication
]:

    with zipfile.ZipFile(path) as z:

        assert z.testzip() is None

    upload_verified(
        path,
        f"{path.stem}_{sha256(path)[:12]}.zip"
    )

# Include recovered and newly uploaded assets.

all_remote = assets_of(RELEASE)

manifest = {
    "release": STAGE_TAG,
    "assets": [
        {
            "name": name,
            "asset_id": asset["id"],
            "size": asset["size"],
            "digest": asset.get("digest")
        }
        for name, asset in all_remote.items()
    ],
    "verified_this_session": REMOTE_MANIFEST
}

manifest_path = (
    AUDIT / "stage7de_remote_manifest.json"
)

save_json(
    manifest,
    manifest_path
)

upload_verified(
    manifest_path,
    f"stage7de_manifest_{sha256(manifest_path)[:12]}.json"
)

shutil.copy2(
    manifest_path,
    BACKUP / manifest_path.name
)

FINAL_COMMIT = push_public(
    "Stage 7DE: complete DGSL screening and verified backup"
)

tracked = set(
    git(
        "ls-tree",
        "-r",
        "--name-only",
        "HEAD"
    ).splitlines()
)

for record in FIGURE_MANIFEST:

    relative = (
        "research_stage7de/figures/"
        + record["format"]
        + "/"
        + record["figure"]
        + "."
        + record["format"]
    )

    assert relative in tracked

    assert sha256(
        REPO / relative
    ) == record["sha256"]

# =============================================================================
# FINAL REPORT
# =============================================================================

section("FINAL STAGE 7D–7E RESEARCH REPORT")

print("\nOOF FOLD INTEGRITY")

print(
    FOLD_TABLE.to_string(
        index=False
    )
)

print("\nOOF PERFORMANCE")

print(
    OOF_TABLE.round(5).to_string(
        index=False
    )
)

print("\nTRAINING DIFFICULTY")

print(
    DIFFICULTY_TABLE.round(5).to_string(
        index=False
    )
)

print("\nMODEL RESULTS")

print(
    RESULT_TABLE.round(6).to_string(
        index=False
    )
)

print("\nALL COMPARISONS")

print(
    ALL_RESULTS.round(6).to_string(
        index=False
    )
)

print("\nPAIRED STATISTICS")

print(
    STATS.round(6).to_string(
        index=False
    )
)

print("\nLINGUISTIC SUBGROUPS")

print(
    SUBGROUPS[
        SUBGROUPS["model"].isin(chosen)
    ].pivot(
        index="subgroup",
        columns="model",
        values="macro_f1"
    ).round(5).to_string()
)

print("\nCLASSWISE PERFORMANCE")

print(
    CLASSWISE[
        CLASSWISE["model"].isin(chosen)
    ].pivot(
        index="class",
        columns="model",
        values="f1-score"
    ).round(5).to_string()
)

print("\nEXPERT COMPETENCE")

print(
    EXPERTS.round(5).to_string(
        index=False
    )
)

print("\nROUTER WEIGHTS")

print(
    ROUTER[
        ROUTER["subgroup"] == "all"
    ].round(5).to_string(
        index=False
    )
)

print("\nSPECIALIST RESCUES")

print(
    RESCUES.round(5).to_string(
        index=False
    )
)

print("\nCALIBRATION")

print(
    CALIBRATION.round(5).to_string(
        index=False
    )
)

print("\nRESEARCH SUMMARY")

print(
    json.dumps(
        SUMMARY,
        indent=2
    )
)

print("\nPUBLICATION ASSETS")

for fmt in [
    "png",
    "pdf",
    "svg"
]:

    print(
        fmt.upper(),
        "figures:",
        len(list(
            (PUB / "figures" / fmt).rglob(f"*.{fmt}")
        ))
    )

print("\nAll new figures committed and verified: YES")
print("All completed checkpoints backed up: YES")
print("Private prediction backup: PASS")
print("GitHub synchronized: True")
print("Final public commit:", FINAL_COMMIT)
print("Private release:", RELEASE["html_url"])

print("Manual file saving required: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("Test evaluated: NO")

print(
    "\nRuntime minutes:",
    round(
        (time.time() - START) / 60,
        2
    )
)

print("\nRESEARCH DECISION:", DECISION)

print("\nSTAGE 7D–7E COMPLETE")

clear_gpu()
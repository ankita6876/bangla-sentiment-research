
# =====================================================================
# BANGLA SENTIMENT RESEARCH
# STAGE 6C — AMSC-NET: END-TO-END ARCHITECTURAL SCREENING
# STEPS 70–79
#
# Includes:
# - GitHub-first recovery
# - Locked dataset verification
# - Proposed architecture implementation
# - Architecture ablations
# - Matched additional-training control
# - Statistical analysis
# - Publication figures
# - Automatic private checkpoint uploads
# - Automatic public source/result backups
#
# NO GOOGLE DRIVE
# NO MANUAL FILE DOWNLOADS
# NO PAID API
# NO TEST MODEL EVALUATION
# =====================================================================

import os
import sys
import gc
import re
import json
import math
import time
import random
import shutil
import hashlib
import getpass
import zipfile
import base64
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from scipy.special import softmax
from scipy.stats import binomtest

START = time.time()

# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================

WORK = Path("/kaggle/working")

OWNER = "ankita6876"

PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"

BASELINE_COMMIT = (
    "9aab6b515deee172f0afcdd70157d9f24dedb331"
)

BASELINE_TAG = "stage6b-locked-baselines-v1"
STAGE_TAG = "stage6c-amsc-screening-v1"

REPO = WORK / "sentiment_verified_research_repo"

S6A = WORK / "sentiment_reconstruction_v1"
S6B = WORK / "sentiment_research_stage6b"
S6C = WORK / "sentiment_research_stage6c"

PUB = WORK / "sentiment_publication_assets"

TABLES = S6C / "tables"
CHECKPOINTS = S6C / "checkpoints"
PRED = S6C / "private_predictions"
AUDIT = S6C / "audit"

RAW = Path(
    "/kaggle/input/datasets/ankitaroy98/"
    "becs-dataset/BECS Dataset.csv"
)

RAW_HASH = (
    "4580950af06bbfbe5bd79e20957b63a2ed"
    "122af2a6a3e685b10576d3f70241c5"
)

LABELS = ["Negative", "Neutral", "Positive"]

LABEL_MAP = {
    label: i for i, label in enumerate(LABELS)
}

SEED = 42

MAX_LENGTH = 64
TRAIN_BATCH = 32
EVAL_BATCH = 64

EPOCHS = 4
PATIENCE = 2

ENCODER_LR = 5e-6
HEAD_LR = 2e-4

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

DIM = 128
DROPOUT = 0.15

AUX_WEIGHT = 0.15
RESIDUAL_WEIGHT = 0.001

BOOTSTRAPS = 2000

MODES = [
    "global_control",
    "phrase_only",
    "directional_only",
    "no_disagreement",
    "fixed_fusion",
    "full"
]

for folder in [
    TABLES, CHECKPOINTS, PRED, AUDIT,
    PUB / "figures/png",
    PUB / "figures/pdf",
    PUB / "figures/svg",
    PUB / "tables",
    PUB / "captions",
    PUB / "architecture",
    PUB / "scripts"
]:
    folder.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), (
    "Enable GPU T4 before running this cell."
)

DEVICE = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

def section(title):

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(8 * 1024**2), b""
        ):
            h.update(chunk)

    return h.hexdigest()

def save_json(obj, path):

    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

def seed_all(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_table(df, name):

    df.to_csv(
        TABLES / f"{name}.csv",
        index=False
    )

    df.to_csv(
        PUB / "tables" / f"{name}.csv",
        index=False
    )

    try:

        df.to_latex(
            PUB / "tables" / f"{name}.tex",
            index=False,
            float_format="%.4f"
        )

    except Exception:
        pass

    return df

# =====================================================================
# STEP 70 — GITHUB AUTHENTICATION AND RECOVERY
# =====================================================================

section("STEP 70 — VERIFIED GITHUB CONNECTION")

TOKEN = getpass.getpass(
    "Enter your working GitHub token: "
).strip()

assert TOKEN

SESSION = requests.Session()

SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

API = "https://api.github.com"

PUBLIC_API = (
    f"{API}/repos/{OWNER}/{PUBLIC_REPO}"
)

PRIVATE_API = (
    f"{API}/repos/{OWNER}/{PRIVATE_REPO}"
)

def get_json(url, **kwargs):

    r = SESSION.get(
        url,
        timeout=60,
        **kwargs
    )

    r.raise_for_status()

    return r.json()

account = get_json(f"{API}/user")["login"]

assert account.lower() == OWNER.lower()

public_info = get_json(PUBLIC_API)
private_info = get_json(PRIVATE_API)

assert private_info["private"]

assert public_info["permissions"]["push"]
assert private_info["permissions"]["push"]

print("Authentication: PASS")
print("Public repository write access: PASS")
print("Private repository write access: PASS")

credential = base64.b64encode(
    f"x-access-token:{TOKEN}".encode()
).decode()

GIT_ENV = os.environ.copy()

GIT_ENV.update({
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0":
        "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0":
        f"AUTHORIZATION: basic {credential}"
})

def git(*args):

    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        env=GIT_ENV,
        timeout=300
    )

    if result.returncode:

        raise RuntimeError(
            f"Git command failed: {args[0]}"
        )

    return result.stdout.strip()

if not (REPO / ".git").exists():

    result = subprocess.run(
        [
            "git", "clone",
            f"https://github.com/{OWNER}/{PUBLIC_REPO}.git",
            str(REPO)
        ],
        capture_output=True,
        env=GIT_ENV,
        timeout=300
    )

    assert result.returncode == 0

assert not git("status", "--porcelain"), (
    "Repository contains uncommitted changes."
)

git("fetch", "origin")

git(
    "checkout", "-B", BRANCH,
    f"origin/{BRANCH}"
)

git(
    "merge-base", "--is-ancestor",
    BASELINE_COMMIT, "HEAD"
)

git("config", "user.name", account)

git(
    "config", "user.email",
    f"{account}@users.noreply.github.com"
)

BACKUP = REPO / "research_stage6c"

BACKUP.mkdir(
    parents=True, exist_ok=True
)

def push_public(message):

    git("add", "research_stage6c")

    if git("status", "--porcelain"):

        git("commit", "-m", message)

    git("push", "origin", BRANCH)

    local = git("rev-parse", "HEAD")

    remote = git(
        "ls-remote",
        "origin",
        f"refs/heads/{BRANCH}"
    ).split()[0]

    assert local == remote

    print("Public GitHub synchronized:", local)

    return local

# Recover previous private release.

baseline_release = get_json(
    f"{PRIVATE_API}/releases/tags/{BASELINE_TAG}"
)

baseline_assets = {
    item["name"]: item
    for item in baseline_release["assets"]
}

def download_asset(asset, destination, expected=None):

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    if destination.exists() and expected:

        if sha256(destination) == expected:

            print("Local artifact verified:", destination.name)

            return destination

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    h = hashlib.sha256()
    total = 0

    with SESSION.get(
        asset["url"],
        headers={
            "Accept": "application/octet-stream"
        },
        stream=True,
        timeout=(30, 180)
    ) as response:

        response.raise_for_status()

        with open(temporary, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=8 * 1024**2
            ):

                if chunk:

                    f.write(chunk)
                    h.update(chunk)
                    total += len(chunk)

    assert total == asset["size"]

    actual = h.hexdigest()

    if expected:

        assert actual == expected

    digest = asset.get("digest")

    if digest:

        assert digest == "sha256:" + actual

    temporary.replace(destination)

    print("Recovered:", destination.name)

    return destination

manifest_path = AUDIT / "baseline_backup_manifest.json"

download_asset(
    baseline_assets["backup_manifest.json"],
    manifest_path
)

baseline_manifest = json.loads(
    manifest_path.read_text()
)

baseline_records = {
    item["name"]: item
    for item in baseline_manifest["assets"]
}

package_name = (
    "stage6b_reproducibility_package.zip"
)

package_path = S6B / package_name

download_asset(
    baseline_assets[package_name],
    package_path,
    baseline_records[package_name]["sha256"]
)

with zipfile.ZipFile(package_path) as z:

    assert z.testzip() is None

    for member in z.infolist():

        target = (
            S6B / member.filename
        ).resolve()

        assert target.is_relative_to(
            S6B.resolve()
        )

    z.extractall(S6B)

BASE_DIR = S6B / "checkpoints/BanglaBERT"

BASE_CKPT = BASE_DIR / "model.pt"

base_asset_name = (
    "BanglaBERT_seed42_best_model.pt"
)

download_asset(
    baseline_assets[base_asset_name],
    BASE_CKPT,
    baseline_records[base_asset_name]["sha256"]
)

print("BanglaBERT checkpoint recovery: PASS")

# Restore publication assets from previous public stages.

for previous_stage in [
    "research_reconstruction_v1",
    "research_stage6b"
]:

    source_folder = (
        REPO / previous_stage
        / "publication_assets"
    )

    if source_folder.exists():

        for path in source_folder.rglob("*"):

            if not path.is_file():
                continue

            if path.name == "figure_manifest.csv":
                continue

            destination = (
                PUB / path.relative_to(source_folder)
            )

            destination.parent.mkdir(
                parents=True, exist_ok=True
            )

            if not destination.exists():

                shutil.copy2(
                    path, destination
                )

print("Previous publication assets recovered.")

# =====================================================================
# STEP 71 — LOCKED DATASET RECOVERY
# =====================================================================

section("STEP 71 — EXACT PARTITION RECOVERY")

if not RAW.exists():

    candidates = list(
        Path("/kaggle/input")
        .rglob("BECS Dataset.csv")
    )

    RAW = next(
        (
            p for p in candidates
            if sha256(p) == RAW_HASH
        ),
        None
    )

assert RAW is not None
assert RAW.exists()
assert sha256(RAW) == RAW_HASH

raw = pd.read_csv(RAW)

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

    subprocess.check_call([
        sys.executable,
        "-m", "pip", "install", "-q",
        "git+https://github.com/csebuetnlp/normalizer"
    ])

    from normalizer import normalize as bn_normalize

DATA = {}

for split, n_expected in [
    ("train", 9204),
    ("val", 1840)
]:

    membership = pd.read_csv(
        S6B / "partitions"
        / f"seed42_{split}.csv"
    ).sort_values("partition_order")

    assert len(membership) == n_expected

    frame = membership.copy()

    frame["Bangla"] = raw.iloc[
        frame["raw_row_id"].to_numpy()
    ]["Bangla"].to_numpy()

    raw_labels = raw.iloc[
        frame["raw_row_id"].to_numpy()
    ]["Label"].astype(str).str.strip().str.casefold()

    assert np.array_equal(
        raw_labels.to_numpy(),
        frame["Label"].str.casefold().to_numpy()
    )

    frame["text_norm"] = (
        frame["Bangla"].map(normalize_text)
    )

    actual_ids = frame["text_norm"].map(
        lambda t: hashlib.sha256(
            t.encode("utf-8")
        ).hexdigest()
    )

    assert np.array_equal(
        actual_ids.to_numpy(),
        frame["sample_id"].to_numpy()
    )

    frame["model_text"] = (
        frame["text_norm"].map(bn_normalize)
    )

    frame["label_id"] = (
        frame["Label"]
        .map(LABEL_MAP)
        .astype(int)
    )

    frame["word_count"] = (
        frame["text_norm"]
        .str.split()
        .str.len()
    )

    DATA[split] = frame.reset_index(drop=True)

    print(
        split,
        "| samples:", len(frame),
        "| exact membership: PASS"
    )

assert not (
    set(DATA["train"]["sample_id"])
    & set(DATA["val"]["sample_id"])
)

assert not (
    set(DATA["train"]["group_id"])
    & set(DATA["val"]["group_id"])
)

assert not (
    set(DATA["train"]["model_text"])
    & set(DATA["val"]["model_text"])
)

Y_VAL = DATA["val"]["label_id"].to_numpy()

BASE_PRED = np.load(
    S6B / "private_predictions/BanglaBERT_val.npz",
    allow_pickle=False
)

assert np.array_equal(
    BASE_PRED["sample_id"],
    DATA["val"]["sample_id"]
    .astype(str).to_numpy(dtype=str)
)

assert np.array_equal(
    BASE_PRED["labels"], Y_VAL
)

BASE_LOGITS = BASE_PRED["logits"]

BASE_F1 = f1_score(
    Y_VAL,
    BASE_LOGITS.argmax(axis=1),
    labels=[0, 1, 2],
    average="macro"
)

assert abs(
    BASE_F1 - 0.8159419852657895
) < 1e-8

print("Baseline validation F1:", BASE_F1)
print("Exact prediction alignment: PASS")
print("Train/validation leakage audit: PASS")

# =====================================================================
# STEP 72 — DATA LOADERS
# =====================================================================

section("STEP 72 — TOKENIZATION AND LINGUISTIC AUDIT")

TOKENIZER = AutoTokenizer.from_pretrained(
    BASE_DIR
)

class SentimentDataset(Dataset):

    def __init__(self, frame):

        encoded = TOKENIZER(
            frame["model_text"].tolist(),
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True,
            return_tensors="pt"
        )

        self.ids = encoded["input_ids"]

        self.attention = encoded[
            "attention_mask"
        ]

        self.valid = (
            encoded["attention_mask"].bool()
            & ~encoded["special_tokens_mask"].bool()
        )

        self.labels = torch.tensor(
            frame["label_id"].to_numpy(),
            dtype=torch.long
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, i):

        return {
            "input_ids": self.ids[i],
            "attention_mask": self.attention[i],
            "valid_mask": self.valid[i],
            "labels": self.labels[i]
        }

TRAIN_DS = SentimentDataset(DATA["train"])
VAL_DS = SentimentDataset(DATA["val"])

def loaders(seed):

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TRAIN_DS,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        VAL_DS,
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader

def cue_mask(series, words):

    pattern = (
        r"(?<![\w\u0980-\u09ff])(?:"
        + "|".join(map(re.escape, words))
        + r")(?![\w\u0980-\u09ff])"
    )

    return series.str.contains(
        pattern,
        regex=True
    ).to_numpy()

neg = cue_mask(
    DATA["val"]["text_norm"],
    ["না", "নয়", "নয়", "নেই", "নাই"]
)

con = cue_mask(
    DATA["val"]["text_norm"],
    ["কিন্তু", "তবে", "যদিও",
     "অথচ", "তবুও", "তথাপি"]
)

long_threshold = (
    DATA["train"]["word_count"]
    .quantile(0.75)
)

long_mask = (
    DATA["val"]["word_count"].to_numpy()
    > long_threshold
)

MASKS = {
    "all": np.ones(len(Y_VAL), dtype=bool),
    "negation_candidate": neg,
    "contrast_candidate": con,
    "both_operators": neg & con,
    "any_operator": neg | con,
    "no_detected_operator": ~(neg | con),
    "long_text": long_mask,
    "shorter_text": ~long_mask
}

for name, mask in MASKS.items():

    print(name, ":", int(mask.sum()))

# =====================================================================
# STEP 73 — PROPOSED AMSC-NET ARCHITECTURE
# =====================================================================

section("STEP 73 — AMSC-NET IMPLEMENTATION")

def masked_mean(x, mask):

    mask = mask.unsqueeze(-1).to(x.dtype)

    return (
        (x * mask).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1)
    )

def masked_softmax(scores, mask, dim=-1):

    scores = scores.float()

    scores = scores.masked_fill(
        ~mask, -1e4
    )

    weights = torch.softmax(
        scores, dim=dim
    )

    weights = weights * mask.float()

    weights = weights / weights.sum(
        dim=dim,
        keepdim=True
    ).clamp_min(1e-8)

    return weights

def js_divergence(logits_a, logits_b):

    a = torch.softmax(
        logits_a.float(), dim=-1
    ).clamp_min(1e-8)

    b = torch.softmax(
        logits_b.float(), dim=-1
    ).clamp_min(1e-8)

    m = 0.5 * (a + b)

    js = 0.5 * (
        (a * (a.log() - m.log())).sum(-1)
        +
        (b * (b.log() - m.log())).sum(-1)
    )

    return js.unsqueeze(-1)

class MultiScalePhraseBranch(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(
                dim, dim,
                kernel_size=k,
                padding=k // 2
            )
            for k in [1, 3, 5]
        ])

        self.scale_scores = nn.Linear(
            dim, 1
        )

        self.token_score = nn.Linear(
            dim, 1
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, valid):

        x = x * valid.unsqueeze(-1)

        features = []

        for conv in self.convs:

            h = conv(
                x.transpose(1, 2)
            ).transpose(1, 2)

            h = F.gelu(h)

            h = h * valid.unsqueeze(-1)

            features.append(h)

        scales = torch.stack(
            features, dim=2
        )

        scale_logits = self.scale_scores(
            scales
        ).squeeze(-1)

        scale_weights = torch.softmax(
            scale_logits.float(),
            dim=2
        ).to(scales.dtype)

        combined = (
            scales
            * scale_weights.unsqueeze(-1)
        ).sum(dim=2)

        combined = self.norm(combined)

        scores = self.token_score(
            combined
        ).squeeze(-1)

        weights = masked_softmax(
            scores, valid
        ).to(combined.dtype)

        pooled = (
            combined
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        return pooled

class DirectionalCompositionBranch(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.query = nn.Linear(
            dim, dim, bias=False
        )

        self.key = nn.Linear(
            dim, dim, bias=False
        )

        self.value = nn.Linear(
            dim, dim, bias=False
        )

        self.distance_penalty = nn.Parameter(
            torch.tensor(0.0)
        )

        self.compose = nn.Sequential(
            nn.Linear(6 * dim, dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

        self.token_score = nn.Linear(
            dim, 1
        )

    def forward(self, x, valid):

        b, length, d = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = (
            q.float()
            @ k.float().transpose(1, 2)
        ) / math.sqrt(d)

        positions = torch.arange(
            length, device=x.device
        )

        relative = (
            positions[None, :]
            - positions[:, None]
        )

        distance = relative.abs().float()

        scores = scores - (
            F.softplus(
                self.distance_penalty
            )
            * distance[None, :, :]
            / max(length, 1)
        )

        pair_valid = (
            valid[:, :, None]
            & valid[:, None, :]
        )

        left_mask = (
            pair_valid
            & (relative[None, :, :] < 0)
        )

        right_mask = (
            pair_valid
            & (relative[None, :, :] > 0)
        )

        left_weights = masked_softmax(
            scores, left_mask
        ).to(v.dtype)

        right_weights = masked_softmax(
            scores, right_mask
        ).to(v.dtype)

        left = left_weights @ v
        right = right_weights @ v

        composed = self.compose(
            torch.cat([
                x,
                left,
                right,
                x * left,
                x * right,
                right - left
            ], dim=-1)
        )

        token_logits = self.token_score(
            composed
        ).squeeze(-1)

        token_weights = masked_softmax(
            token_logits, valid
        ).to(composed.dtype)

        pooled = (
            composed
            * token_weights.unsqueeze(-1)
        ).sum(dim=1)

        return pooled

class AMSCNet(nn.Module):

    def __init__(
        self,
        base_model,
        mode="full",
        control_width=512
    ):

        super().__init__()

        self.mode = mode

        self.encoder = base_model.electra

        self.base_classifier = (
            base_model.classifier
        )

        hidden = (
            base_model.config.hidden_size
        )

        self.project = nn.Sequential(
            nn.Linear(hidden, DIM),
            nn.LayerNorm(DIM),
            nn.GELU()
        )

        self.global_repr = nn.Sequential(
            nn.Linear(2 * DIM, DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

        self.global_delta = nn.Linear(
            DIM, 3
        )

        self.phrase = MultiScalePhraseBranch(
            DIM
        )

        self.directional = (
            DirectionalCompositionBranch(DIM)
        )

        self.phrase_head = nn.Linear(
            DIM, 3
        )

        self.directional_head = nn.Linear(
            DIM, 3
        )

        self.gate = nn.Sequential(
            nn.Linear(3 * DIM + 5, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 2)
        )

        self.control = nn.Sequential(
            nn.Linear(2 * DIM, control_width),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(control_width, 3)
        )

        # Small initial residuals preserve
        # the fine-tuned baseline at initialization.

        nn.init.zeros_(
            self.global_delta.weight
        )

        nn.init.zeros_(
            self.global_delta.bias
        )

        nn.init.normal_(
            self.control[-1].weight,
            std=1e-3
        )

        nn.init.zeros_(
            self.control[-1].bias
        )

        nn.init.zeros_(
            self.gate[-1].weight
        )

        nn.init.constant_(
            self.gate[-1].bias,
            -2.0
        )

        self.configure_mode()

    def configure_mode(self):

        if self.mode == "global_control":

            inactive = [
                self.global_repr,
                self.global_delta,
                self.phrase,
                self.directional,
                self.phrase_head,
                self.directional_head,
                self.gate
            ]

        else:

            inactive = [self.control]

            if self.mode == "phrase_only":

                inactive += [
                    self.directional,
                    self.directional_head
                ]

            if self.mode == "directional_only":

                inactive += [
                    self.phrase,
                    self.phrase_head
                ]

            if self.mode == "fixed_fusion":

                inactive += [self.gate]

        for module in inactive:

            for parameter in module.parameters():

                parameter.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask,
        valid_mask
    ):

        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden = output.last_hidden_state

        base_logits = self.base_classifier(
            hidden
        )

        x = self.project(hidden)

        pooled = masked_mean(
            x, valid_mask
        )

        global_input = torch.cat(
            [x[:, 0], pooled],
            dim=-1
        )

        if self.mode == "global_control":

            delta = self.control(
                global_input
            )

            return {
                "logits": base_logits + delta,
                "global_logits": base_logits,
                "phrase_logits": None,
                "directional_logits": None,
                "delta": delta,
                "gates": torch.zeros(
                    len(x), 2,
                    device=x.device
                )
            }

        g = self.global_repr(
            global_input
        )

        zg = (
            base_logits
            + self.global_delta(g)
        )

        zeros = torch.zeros_like(g)

        use_phrase = (
            self.mode != "directional_only"
        )

        use_directional = (
            self.mode != "phrase_only"
        )

        p = (
            self.phrase(x, valid_mask)
            if use_phrase
            else zeros
        )

        c = (
            self.directional(x, valid_mask)
            if use_directional
            else zeros
        )

        zp = (
            self.phrase_head(p)
            if use_phrase
            else zg
        )

        zc = (
            self.directional_head(c)
            if use_directional
            else zg
        )

        # Disagreement statistics are detached
        # to prevent direct optimization of the
        # gate's diagnostic inputs.

        dp = js_divergence(
            zg.detach(), zp.detach()
        )

        dc = js_divergence(
            zg.detach(), zc.detach()
        )

        pg = torch.softmax(
            zg.detach().float(),
            dim=-1
        )

        entropy = (
            -(pg * pg.clamp_min(1e-8).log())
            .sum(dim=-1, keepdim=True)
            / math.log(3)
        )

        top2 = pg.topk(
            2, dim=-1
        ).values

        margin_uncertainty = (
            1 - top2[:, :1] + top2[:, 1:2]
        )

        max_uncertainty = (
            1 - pg.max(
                dim=-1,
                keepdim=True
            ).values
        )

        if self.mode == "no_disagreement":

            dp = torch.zeros_like(dp)
            dc = torch.zeros_like(dc)

        gate_input = torch.cat([
            g, p, c,
            dp.to(g.dtype),
            dc.to(g.dtype),
            entropy.to(g.dtype),
            margin_uncertainty.to(g.dtype),
            max_uncertainty.to(g.dtype)
        ], dim=-1)

        if self.mode == "fixed_fusion":

            alpha = torch.full(
                (len(x), 2),
                0.15,
                device=x.device,
                dtype=zg.dtype
            )

        else:

            # Bounded residual weights.
            # Their sum cannot exceed 0.8.

            alpha = (
                0.4 * torch.sigmoid(
                    self.gate(gate_input)
                )
            )

        active = torch.tensor(
            [use_phrase, use_directional],
            device=x.device,
            dtype=alpha.dtype
        )

        alpha = alpha * active

        delta = (
            alpha[:, :1] * (zp - zg)
            +
            alpha[:, 1:2] * (zc - zg)
        )

        final_logits = zg + delta

        return {
            "logits": final_logits,
            "global_logits": zg,
            "phrase_logits":
                zp if use_phrase else None,
            "directional_logits":
                zc if use_directional else None,
            "delta": delta,
            "gates": alpha
        }

# =====================================================================
# MODEL INITIALIZATION AND PARAMETER MATCHING
# =====================================================================

BASE_CONFIG = AutoConfig.from_pretrained(
    BASE_DIR
)

BASE_STATE = torch.load(
    BASE_CKPT,
    map_location="cpu",
    weights_only=True
)

def make_base():

    model = (
        AutoModelForSequenceClassification
        .from_config(BASE_CONFIG)
    )

    model.load_state_dict(
        BASE_STATE,
        strict=True
    )

    return model

def trainable_count(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def extra_count(model):

    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad
        and not name.startswith("encoder.")
        and not name.startswith("base_classifier.")
    )

seed_all(SEED)

temporary = AMSCNet(
    make_base(),
    mode="full"
)

target_extra = extra_count(temporary)

projection_extra = sum(
    p.numel()
    for p in temporary.project.parameters()
)

# MLP control width chosen to match the
# trainable extra parameter count.

CONTROL_WIDTH = max(
    1,
    round(
        (target_extra - projection_extra - 3)
        / (2 * DIM + 4)
    )
)

del temporary
gc.collect()

temporary = AMSCNet(
    make_base(),
    mode="global_control",
    control_width=CONTROL_WIDTH
)

control_extra = extra_count(temporary)

relative_difference = abs(
    control_extra - target_extra
) / target_extra

print("Full AMSC extra parameters:", target_extra)
print("MLP control width:", CONTROL_WIDTH)
print("Control extra parameters:", control_extra)
print("Relative mismatch:", relative_difference)

assert relative_difference < 0.01

del temporary
gc.collect()

# =====================================================================
# STEP 74 — SMOKE TEST AND LOCKED EXPERIMENT CONFIGURATION
# =====================================================================

section("STEP 74 — ARCHITECTURE VALIDATION")

seed_all(SEED)

smoke = AMSCNet(
    make_base(),
    mode="full",
    control_width=CONTROL_WIDTH
).to(DEVICE)

_, smoke_loader = loaders(SEED)

batch = next(iter(smoke_loader))

batch = {
    k: v.to(DEVICE)
    for k, v in batch.items()
}

smoke.eval()

with torch.no_grad():

    output = smoke(
        batch["input_ids"],
        batch["attention_mask"],
        batch["valid_mask"]
    )

assert output["logits"].shape == (
    len(batch["labels"]), 3
)

assert torch.isfinite(
    output["logits"]
).all()

assert (
    output["gates"].sum(dim=1)
    <= 0.8001
).all()

# Check that the model remains finite
# even if no ordinary tokens are available.

with torch.no_grad():

    empty_output = smoke(
        batch["input_ids"],
        batch["attention_mask"],
        torch.zeros_like(
            batch["valid_mask"]
        )
    )

assert torch.isfinite(
    empty_output["logits"]
).all()

# Gradient-flow check.

smoke.train()

output = smoke(
    batch["input_ids"],
    batch["attention_mask"],
    batch["valid_mask"]
)

loss = F.cross_entropy(
    output["logits"],
    batch["labels"]
)

loss.backward()

for name, module in [
    ("encoder", smoke.encoder),
    ("phrase", smoke.phrase),
    ("directional", smoke.directional),
    ("gate", smoke.gate)
]:

    gradients = [
        p.grad
        for p in module.parameters()
        if p.grad is not None
    ]

    assert gradients

    assert all(
        torch.isfinite(g).all()
        for g in gradients
    )

    print(name, "gradient flow: PASS")

del smoke, batch, output, empty_output
del smoke_loader, loss, gradients

gc.collect()
torch.cuda.empty_cache()

CONFIG = {
    "stage": "6c",
    "architecture": "AMSC-Net",
    "candidate_novelty": (
        "Disagreement-conditioned directional "
        "multi-scale residual composition"
    ),
    "seed": SEED,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "max_length": MAX_LENGTH,
    "train_batch": TRAIN_BATCH,
    "encoder_lr": ENCODER_LR,
    "head_lr": HEAD_LR,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "dim": DIM,
    "dropout": DROPOUT,
    "aux_weight": AUX_WEIGHT,
    "residual_weight": RESIDUAL_WEIGHT,
    "control_width": CONTROL_WIDTH,
    "modes": MODES,
    "base_checkpoint_sha256": sha256(BASE_CKPT),
    "base_validation_f1": BASE_F1,
    "raw_sha256": RAW_HASH,
    "source_commit": BASELINE_COMMIT,
    "selection_metric": "validation macro-F1",
    "test_evaluated": False,
    "novelty_verified": False,
    "torch_version": torch.__version__
}

save_json(
    CONFIG,
    AUDIT / "experiment_config.json"
)

shutil.copy2(
    AUDIT / "experiment_config.json",
    BACKUP / "experiment_config.json"
)

# Preserve full executable source before training.

history = (
    get_ipython()
    .history_manager
    .input_hist_raw
)

source_candidates = [
    cell for cell in history
    if "STAGE 6C" in cell
    and "END-TO-END ARCHITECTURAL SCREENING" in cell
    and len(cell) > 10000
]

assert source_candidates, (
    "Unable to preserve complete source."
)

source = source_candidates[-1]

assert not re.search(
    r"(?:ghp_|github_pat_|gho_|ghs_)"
    r"[A-Za-z0-9_]{20,}",
    source
)

SOURCE_PATH = AUDIT / "stage6c_complete.py"

SOURCE_PATH.write_text(
    source,
    encoding="utf-8"
)

shutil.copy2(
    SOURCE_PATH,
    BACKUP / "stage6c_complete.py"
)

shutil.copy2(
    SOURCE_PATH,
    PUB / "scripts/stage6c_complete.py"
)

push_public(
    "Lock AMSC-Net architecture and screening protocol"
)

# =====================================================================
# PRIVATE RELEASE AND CHECKPOINT BACKUP UTILITIES
# =====================================================================

section("STEP 74B — PRIVATE ARTIFACT STORAGE")

response = SESSION.get(
    f"{PRIVATE_API}/releases/tags/{STAGE_TAG}",
    timeout=60
)

if response.status_code == 404:

    response = SESSION.post(
        f"{PRIVATE_API}/releases",
        json={
            "tag_name": STAGE_TAG,
            "target_commitish":
                private_info["default_branch"],
            "name":
                "Stage 6C — AMSC-Net screening",
            "body":
                "Private architecture checkpoints "
                "and validation predictions.",
            "draft": False,
            "prerelease": False
        },
        timeout=60
    )

response.raise_for_status()

release = response.json()

RELEASE_ID = release["id"]

UPLOAD_URL = release[
    "upload_url"
].split("{")[0]

def remote_assets():

    items = []
    page = 1

    while True:

        batch = get_json(
            f"{PRIVATE_API}/releases/"
            f"{RELEASE_ID}/assets",
            params={
                "per_page": 100,
                "page": page
            }
        )

        items.extend(batch)

        if len(batch) < 100:
            break

        page += 1

    return {
        item["name"]: item
        for item in items
    }

def verify_remote_asset(asset, expected_hash):

    digest = asset.get("digest")

    if digest:

        assert digest == (
            "sha256:" + expected_hash
        )

        return

    h = hashlib.sha256()
    total = 0

    with SESSION.get(
        asset["url"],
        headers={
            "Accept": "application/octet-stream"
        },
        stream=True,
        timeout=(30, 180)
    ) as response:

        response.raise_for_status()

        for chunk in response.iter_content(
            chunk_size=8 * 1024**2
        ):

            if chunk:

                h.update(chunk)
                total += len(chunk)

    assert total == asset["size"]
    assert h.hexdigest() == expected_hash

def upload_private(path, name=None):

    path = Path(path)

    name = name or path.name

    expected_hash = sha256(path)

    assert path.stat().st_size < 2 * 1024**3

    existing = remote_assets().get(name)

    if existing:

        assert existing["state"] == "uploaded"

        assert existing["size"] == (
            path.stat().st_size
        )

        verify_remote_asset(
            existing, expected_hash
        )

        asset = existing

    else:

        with open(path, "rb") as f:

            response = SESSION.post(
                UPLOAD_URL,
                params={"name": name},
                headers={
                    "Content-Type":
                        "application/octet-stream"
                },
                data=f,
                timeout=(30, 1800)
            )

        response.raise_for_status()

        asset = response.json()

        assert asset["state"] == "uploaded"

        assert asset["size"] == (
            path.stat().st_size
        )

        verify_remote_asset(
            asset, expected_hash
        )

    print(
        "Private asset verified:",
        name
    )

    return {
        "name": name,
        "asset_id": asset["id"],
        "sha256": expected_hash,
        "size_bytes": path.stat().st_size
    }

print("Private release:", STAGE_TAG)

# =====================================================================
# STEP 75 — TRAINING AND METRICS UTILITIES
# =====================================================================

section("STEP 75 — TRAINING UTILITIES")

def metrics(y, logits):

    pred = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(y, pred),

        "macro_f1": f1_score(
            y, pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ),

        "macro_precision": precision_score(
            y, pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y, pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    }

def calibration(y, logits):

    probs = softmax(
        logits, axis=1
    )

    confidence = probs.max(axis=1)

    correct = (
        probs.argmax(axis=1) == y
    )

    ece = 0.0

    boundaries = np.linspace(
        0, 1, 16
    )

    for i in range(15):

        mask = (
            (confidence >= boundaries[i])
            & (
                confidence <= boundaries[i+1]
                if i == 14
                else confidence < boundaries[i+1]
            )
        )

        if mask.any():

            ece += (
                mask.mean()
                * abs(
                    correct[mask].mean()
                    - confidence[mask].mean()
                )
            )

    return {
        "ece": float(ece),

        "nll": float(
            -np.log(
                probs[np.arange(len(y)), y]
                .clip(1e-12, 1)
            ).mean()
        ),

        "brier": float(
            (
                (
                    probs - np.eye(3)[y]
                ) ** 2
            ).sum(axis=1).mean()
        )
    }

def objective(output, labels):

    main_loss = F.cross_entropy(
        output["logits"],
        labels
    )

    auxiliary = []

    for key in [
        "global_logits",
        "phrase_logits",
        "directional_logits"
    ]:

        if output[key] is not None:

            auxiliary.append(
                F.cross_entropy(
                    output[key], labels
                )
            )

    aux_loss = (
        torch.stack(auxiliary).mean()
        if auxiliary
        else main_loss * 0
    )

    residual = (
        output["delta"].float()
        .square().mean()
    )

    return (
        main_loss
        + AUX_WEIGHT * aux_loss
        + RESIDUAL_WEIGHT * residual
    )

@torch.inference_mode()
def evaluate(model, loader):

    model.eval()

    logits = []
    labels = []
    gates = []

    for batch in loader:

        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["valid_mask"]
            )

        logits.append(
            output["logits"]
            .float().cpu().numpy()
        )

        gates.append(
            output["gates"]
            .float().cpu().numpy()
        )

        labels.append(
            batch["labels"].cpu().numpy()
        )

    return (
        np.concatenate(logits),
        np.concatenate(labels),
        np.concatenate(gates)
    )

RESULTS = [{
    "model": "BanglaBERT",
    "seed": SEED,
    "best_epoch": 2,
    "extra_parameters": 0,
    "trainable_parameters": 110619651,
    **metrics(Y_VAL, BASE_LOGITS),
    **calibration(Y_VAL, BASE_LOGITS)
}]

HISTORY = []

ALL_LOGITS = {
    "BanglaBERT": BASE_LOGITS
}

ALL_GATES = {}

ASSET_RECORDS = []

def export_tables():

    for path in TABLES.glob("*.csv"):

        columns = pd.read_csv(
            path, nrows=0
        ).columns

        forbidden = {
            "bangla", "text", "text_norm",
            "model_text", "sample_id",
            "raw_row_id", "comment"
        }

        assert not (
            set(c.lower() for c in columns)
            & forbidden
        )

        destination = (
            BACKUP / "tables" / path.name
        )

        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        shutil.copy2(path, destination)

# =====================================================================
# STEP 76 — TRAIN ALL ARCHITECTURAL CONFIGURATIONS
# =====================================================================

section("STEP 76 — AMSC-NET TRAINING")

for mode in MODES:

    section(f"TRAINING CONFIGURATION — {mode}")

    seed_all(SEED)

    mode_dir = CHECKPOINTS / mode

    mode_dir.mkdir(
        parents=True, exist_ok=True
    )

    checkpoint = mode_dir / "model.pt"

    metadata_path = (
        mode_dir / "completed.json"
    )

    logits_path = (
        PRED / f"{mode}_val.npz"
    )

    asset_checkpoint = (
        f"{mode}_seed42_model.pt"
    )

    asset_predictions = (
        f"{mode}_seed42_val.npz"
    )

    asset_metadata = (
        f"{mode}_seed42_completed.json"
    )

    run_config = {
        **CONFIG,
        "mode": mode
    }

    run_hash = hashlib.sha256(
        json.dumps(
            run_config,
            sort_keys=True,
            default=str
        ).encode()
    ).hexdigest()

    assets = remote_assets()

    # Resume fully completed experiments
    # directly from the private release.

    if all(
        name in assets
        for name in [
            asset_checkpoint,
            asset_predictions,
            asset_metadata
        ]
    ):

        download_asset(
            assets[asset_metadata],
            metadata_path
        )

        completed = json.loads(
            metadata_path.read_text()
        )

        assert completed["run_hash"] == run_hash

        download_asset(
            assets[asset_predictions],
            logits_path,
            completed["prediction_sha256"]
        )

        verify_remote_asset(
            assets[asset_checkpoint],
            completed["checkpoint_sha256"]
        )

        saved = np.load(
            logits_path,
            allow_pickle=False
        )

        assert np.array_equal(
            saved["sample_id"],
            DATA["val"]["sample_id"]
            .astype(str).to_numpy(dtype=str)
        )

        assert np.array_equal(
            saved["labels"], Y_VAL
        )

        RESULTS.append(
            completed["result"]
        )

        HISTORY.extend(
            completed["history"]
        )

        ALL_LOGITS[mode] = saved["logits"]
        ALL_GATES[mode] = saved["gates"]

        print("Recovered completed experiment:", mode)

        continue

    # Recover a completed local run when an
    # earlier upload was interrupted.

    local_completed = False

    if (
        metadata_path.exists()
        and checkpoint.exists()
        and logits_path.exists()
    ):

        completed = json.loads(
            metadata_path.read_text()
        )

        local_completed = (
            completed["run_hash"] == run_hash
            and completed["checkpoint_sha256"]
                == sha256(checkpoint)
            and completed["prediction_sha256"]
                == sha256(logits_path)
        )

    if not local_completed:

        model = AMSCNet(
            make_base(),
            mode=mode,
            control_width=CONTROL_WIDTH
        ).to(DEVICE)

        train_loader, val_loader = loaders(
            SEED
        )

        total_params = trainable_count(
            model
        )

        extra_params = extra_count(
            model
        )

        print("Trainable parameters:", total_params)
        print("Extra parameters:", extra_params)

        encoder_parameters = [
            p for name, p
            in model.named_parameters()
            if p.requires_grad
            and name.startswith("encoder.")
        ]

        head_parameters = [
            p for name, p
            in model.named_parameters()
            if p.requires_grad
            and not name.startswith("encoder.")
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": encoder_parameters,
                    "lr": ENCODER_LR
                },
                {
                    "params": head_parameters,
                    "lr": HEAD_LR
                }
            ],
            weight_decay=WEIGHT_DECAY
        )

        total_steps = (
            len(train_loader) * EPOCHS
        )

        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    total_steps * WARMUP_RATIO
                ),
                num_training_steps=total_steps
            )
        )

        scaler = torch.amp.GradScaler(
            "cuda"
        )

        best_f1 = -1.0
        best_epoch = None
        stale = 0

        run_history = []

        t0 = time.time()

        for epoch in range(1, EPOCHS + 1):

            model.train()

            loss_sum = 0.0
            count = 0

            for batch in train_loader:

                batch = {
                    k: v.to(DEVICE)
                    for k, v in batch.items()
                }

                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    output = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["valid_mask"]
                    )

                    loss = objective(
                        output,
                        batch["labels"]
                    )

                assert torch.isfinite(loss)

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    [
                        p for p in model.parameters()
                        if p.requires_grad
                    ],
                    1.0
                )

                old_scale = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()

                if scaler.get_scale() >= old_scale:

                    scheduler.step()

                n = len(batch["labels"])

                loss_sum += loss.item() * n
                count += n

            logits, y, gates = evaluate(
                model, val_loader
            )

            assert np.array_equal(y, Y_VAL)

            scores = metrics(y, logits)

            row = {
                "model": mode,
                "seed": SEED,
                "epoch": epoch,
                "train_loss": loss_sum / count,
                **scores
            }

            run_history.append(row)

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Loss {row['train_loss']:.4f} | "
                f"Val Acc {scores['accuracy']:.4f} | "
                f"Val F1 {scores['macro_f1']:.4f}"
            )

            if scores["macro_f1"] > best_f1:

                best_f1 = scores["macro_f1"]
                best_epoch = epoch
                stale = 0

                # Preserve exact FP32 state.
                # Inactive modules are also saved
                # for strict architectural recovery.

                torch.save(
                    {
                        k: v.detach().cpu()
                        for k, v
                        in model.state_dict().items()
                    },
                    checkpoint
                )

                np.savez_compressed(
                    logits_path,
                    sample_id=(
                        DATA["val"]["sample_id"]
                        .astype(str)
                        .to_numpy(dtype=str)
                    ),
                    labels=Y_VAL,
                    logits=logits,
                    gates=gates
                )

                print("Best checkpoint saved.")

            else:

                stale += 1

            if stale >= PATIENCE:

                print("Early stopping.")
                break

        saved = np.load(
            logits_path,
            allow_pickle=False
        )

        result = {
            "model": mode,
            "seed": SEED,
            "best_epoch": best_epoch,
            "extra_parameters": extra_params,
            "trainable_parameters": total_params,
            **metrics(
                Y_VAL, saved["logits"]
            ),
            **calibration(
                Y_VAL, saved["logits"]
            ),
            "runtime_minutes":
                (time.time() - t0) / 60
        }

        completed = {
            "run_hash": run_hash,
            "config": run_config,
            "result": result,
            "history": run_history,
            "checkpoint_sha256":
                sha256(checkpoint),
            "prediction_sha256":
                sha256(logits_path)
        }

        save_json(
            completed,
            metadata_path
        )

        del model, optimizer, scheduler
        del scaler, train_loader, val_loader
        del batch, output, loss

        gc.collect()
        torch.cuda.empty_cache()

    else:

        print("Verified completed local run recovered.")

    saved = np.load(
        logits_path,
        allow_pickle=False
    )

    assert np.array_equal(
        saved["labels"], Y_VAL
    )

    assert np.array_equal(
        saved["sample_id"],
        DATA["val"]["sample_id"]
        .astype(str).to_numpy(dtype=str)
    )

    RESULTS.append(
        completed["result"]
    )

    HISTORY.extend(
        completed["history"]
    )

    ALL_LOGITS[mode] = saved["logits"]
    ALL_GATES[mode] = saved["gates"]

    # Upload checkpoint and predictions immediately.
    # Completion metadata is uploaded last.

    ASSET_RECORDS.append(
        upload_private(
            checkpoint,
            asset_checkpoint
        )
    )

    ASSET_RECORDS.append(
        upload_private(
            logits_path,
            asset_predictions
        )
    )

    ASSET_RECORDS.append(
        upload_private(
            metadata_path,
            asset_metadata
        )
    )

    save_table(
        pd.DataFrame(RESULTS),
        "stage6c_01_model_results"
    )

    save_table(
        pd.DataFrame(HISTORY),
        "stage6c_02_training_history"
    )

    export_tables()

    metadata_target = (
        BACKUP / "run_metadata"
        / f"{mode}.json"
    )

    metadata_target.parent.mkdir(
        parents=True, exist_ok=True
    )

    shutil.copy2(
        metadata_path,
        metadata_target
    )

    push_public(
        f"Preserve completed AMSC experiment: {mode}"
    )

    # Remove large local checkpoint only after
    # verified private upload.

    checkpoint.unlink()

    print(
        mode,
        "| F1:",
        round(
            completed["result"]["macro_f1"], 6
        ),
        "| Permanent backup: PASS"
    )

# =====================================================================
# STEP 77 — PERFORMANCE AND STATISTICAL ANALYSIS
# =====================================================================

section("STEP 77 — ARCHITECTURAL COMPARISON")

results_df = pd.DataFrame(
    RESULTS
).sort_values(
    "macro_f1",
    ascending=False
)

results_df["delta_vs_banglabert"] = (
    results_df["macro_f1"] - BASE_F1
)

save_table(
    results_df,
    "stage6c_01_model_results"
)

history_df = save_table(
    pd.DataFrame(HISTORY),
    "stage6c_02_training_history"
)

print(
    results_df[
        [
            "model",
            "accuracy",
            "macro_f1",
            "delta_vs_banglabert",
            "extra_parameters"
        ]
    ].round(5).to_string(index=False)
)

subgroup_rows = []
classwise_rows = []

for name, logits in ALL_LOGITS.items():

    pred = logits.argmax(axis=1)

    for subgroup, mask in MASKS.items():

        if not mask.any():
            continue

        subgroup_rows.append({
            "model": name,
            "subgroup": subgroup,
            "n": int(mask.sum()),
            **metrics(
                Y_VAL[mask],
                logits[mask]
            )
        })

    for i, label in enumerate(LABELS):

        classwise_rows.append({
            "model": name,
            "class": label,
            "support": int(
                (Y_VAL == i).sum()
            ),
            "f1": f1_score(
                Y_VAL == i,
                pred == i,
                zero_division=0
            ),
            "precision": precision_score(
                Y_VAL == i,
                pred == i,
                zero_division=0
            ),
            "recall": recall_score(
                Y_VAL == i,
                pred == i,
                zero_division=0
            )
        })

subgroup_df = save_table(
    pd.DataFrame(subgroup_rows),
    "stage6c_03_subgroup_results"
)

classwise_df = save_table(
    pd.DataFrame(classwise_rows),
    "stage6c_04_classwise_results"
)

# Paired group bootstrap.
# Validation intervals are exploratory because
# model selection uses the same validation set.

group_ids = DATA["val"]["group_id"].to_numpy()

unique_groups = np.unique(group_ids)

group_members = [
    np.flatnonzero(group_ids == g)
    for g in unique_groups
]

def fast_macro_f1(y, p):

    cm = np.bincount(
        3 * y + p,
        minlength=9
    ).reshape(3, 3)

    tp = np.diag(cm).astype(float)

    denominator = (
        cm.sum(axis=0)
        + cm.sum(axis=1)
    )

    f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros(3),
        where=denominator > 0
    )

    return float(f1.mean())

PREDICTIONS = {
    name: logits.argmax(axis=1)
    for name, logits in ALL_LOGITS.items()
}

pairs = [
    (mode, "BanglaBERT")
    for mode in MODES
] + [
    ("full", "global_control"),
    ("full", "no_disagreement"),
    ("full", "fixed_fusion"),
    ("full", "phrase_only"),
    ("full", "directional_only")
]

rng = np.random.default_rng(SEED)

bootstrap_values = {
    pair: []
    for pair in pairs
}

for _ in range(BOOTSTRAPS):

    sampled = rng.integers(
        0,
        len(unique_groups),
        size=len(unique_groups)
    )

    indices = np.concatenate([
        group_members[i]
        for i in sampled
    ])

    y = Y_VAL[indices]

    scores = {
        name: fast_macro_f1(
            y, pred[indices]
        )
        for name, pred
        in PREDICTIONS.items()
    }

    for a, b in pairs:

        bootstrap_values[(a, b)].append(
            scores[a] - scores[b]
        )

stat_rows = []

for a, b in pairs:

    pa = PREDICTIONS[a]
    pb = PREDICTIONS[b]

    ca = pa == Y_VAL
    cb = pb == Y_VAL

    gained = int(
        (ca & ~cb).sum()
    )

    lost = int(
        (~ca & cb).sum()
    )

    p_value = (
        binomtest(
            gained,
            gained + lost,
            p=0.5
        ).pvalue
        if gained + lost
        else 1.0
    )

    values = np.asarray(
        bootstrap_values[(a, b)]
    )

    stat_rows.append({
        "model": a,
        "reference": b,
        "delta_macro_f1":
            fast_macro_f1(Y_VAL, pa)
            - fast_macro_f1(Y_VAL, pb),
        "ci95_low":
            np.percentile(values, 2.5),
        "ci95_high":
            np.percentile(values, 97.5),
        "gained_correct": gained,
        "lost_correct": lost,
        "mcnemar_p": p_value
    })

stats_df = pd.DataFrame(stat_rows)

# Holm correction for the reported
# exploratory McNemar comparisons.

p = stats_df["mcnemar_p"].to_numpy()

order = np.argsort(p)

adjusted = np.empty(len(p))

running = 0.0

for rank, index in enumerate(order):

    running = max(
        running,
        (len(p) - rank) * p[index]
    )

    adjusted[index] = min(
        running, 1.0
    )

stats_df["mcnemar_p_holm"] = adjusted

save_table(
    stats_df,
    "stage6c_05_paired_statistics"
)

# Gate behavior.

gate_rows = []

for name, gates in ALL_GATES.items():

    for subgroup, mask in MASKS.items():

        if not mask.any():
            continue

        gate_rows.append({
            "model": name,
            "subgroup": subgroup,
            "n": int(mask.sum()),
            "mean_phrase_gate":
                gates[mask, 0].mean(),
            "mean_directional_gate":
                gates[mask, 1].mean(),
            "mean_total_gate":
                gates[mask].sum(axis=1).mean()
        })

gate_df = save_table(
    pd.DataFrame(gate_rows),
    "stage6c_06_gate_diagnostics"
)

print("\nPaired comparisons:")

print(
    stats_df.round(5)
    .to_string(index=False)
)

# =====================================================================
# STEP 78 — PUBLICATION-QUALITY FIGURES
# =====================================================================

section("STEP 78 — PUBLICATION ASSETS")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

FIGURES = []

def save_figure(fig, name, caption, source):

    fig.tight_layout()

    for ext in ["png", "pdf", "svg"]:

        fig.savefig(
            PUB / "figures" / ext
            / f"{name}.{ext}",
            dpi=600 if ext == "png" else 300,
            bbox_inches="tight"
        )

    (PUB / "captions" / f"{name}.txt").write_text(
        caption,
        encoding="utf-8"
    )

    FIGURES.append({
        "figure_id": name,
        "png": f"figures/png/{name}.png",
        "pdf": f"figures/pdf/{name}.pdf",
        "svg": f"figures/svg/{name}.svg",
        "caption": caption,
        "source_table": source,
        "stage": "6c"
    })

    plt.close(fig)

# Figure 1 — Architectural comparison.

plot = results_df.sort_values(
    "macro_f1"
)

fig, ax = plt.subplots(figsize=(8, 5.5))

bars = ax.barh(
    plot["model"],
    plot["macro_f1"],
    color=[
        "#237a68" if x == "full"
        else "#64748b" if x == "BanglaBERT"
        else "#8ba4ba"
        for x in plot["model"]
    ]
)

ax.bar_label(
    bars,
    fmt="%.4f",
    padding=4
)

ax.set_xlim(0, 1.05)

ax.axvline(
    BASE_F1,
    linestyle="--",
    color="#64748b",
    linewidth=1
)

ax.set_xlabel("Validation macro-F1")
ax.set_title("AMSC-Net architectural screening")

save_figure(
    fig,
    "stage6c_01_architecture_comparison",
    "Single-seed validation comparison of AMSC-Net "
    "and architectural controls.",
    "stage6c_01_model_results.csv"
)

# Figure 2 — Training convergence.

fig, ax = plt.subplots(figsize=(8, 5))

for name, group in history_df.groupby("model"):

    ax.plot(
        group["epoch"],
        group["macro_f1"],
        marker="o",
        label=name
    )

ax.axhline(
    BASE_F1,
    linestyle="--",
    color="#64748b",
    label="Original BanglaBERT"
)

ax.set_xlabel("Additional fine-tuning epoch")
ax.set_ylabel("Validation macro-F1")
ax.set_title("Architecture training convergence")
ax.legend(frameon=False, fontsize=8)

save_figure(
    fig,
    "stage6c_02_training_convergence",
    "Validation macro-F1 during additional "
    "end-to-end fine-tuning.",
    "stage6c_02_training_history.csv"
)

# Figure 3 — Paired effect sizes.

plot = stats_df[
    stats_df["reference"] == "BanglaBERT"
].copy().sort_values("delta_macro_f1")

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    plot["delta_macro_f1"],
    np.arange(len(plot)),
    xerr=[
        plot["delta_macro_f1"]
        - plot["ci95_low"],
        plot["ci95_high"]
        - plot["delta_macro_f1"]
    ],
    fmt="o",
    capsize=4,
    color="#237a68"
)

ax.set_yticks(
    np.arange(len(plot)),
    plot["model"]
)

ax.axvline(
    0, color="#64748b",
    linestyle="--"
)

ax.set_xlabel("Macro-F1 difference")
ax.set_title("Paired validation effect estimates")

save_figure(
    fig,
    "stage6c_03_paired_effects",
    "Exploratory paired group-bootstrap intervals "
    "relative to the original BanglaBERT baseline.",
    "stage6c_05_paired_statistics.csv"
)

# Figure 4 — Linguistic subgroup heatmap.

plot = subgroup_df.pivot(
    index="subgroup",
    columns="model",
    values="macro_f1"
)

fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(
    plot.to_numpy(),
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    aspect="auto"
)

for i in range(len(plot.index)):

    for j in range(len(plot.columns)):

        value = plot.iloc[i, j]

        ax.text(
            j, i,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color=(
                "white"
                if value > 0.65
                else "black"
            )
        )

ax.set_xticks(
    range(len(plot.columns)),
    plot.columns,
    rotation=35,
    ha="right"
)

ax.set_yticks(
    range(len(plot.index)),
    plot.index
)

ax.set_title("Linguistic subgroup robustness")

fig.colorbar(
    im, ax=ax,
    label="Macro-F1"
)

save_figure(
    fig,
    "stage6c_04_subgroup_heatmap",
    "Exploratory subgroup analysis using heuristic "
    "linguistic cue matching.",
    "stage6c_03_subgroup_results.csv"
)

# Figure 5 — Classwise F1.

plot = classwise_df.pivot(
    index="model",
    columns="class",
    values="f1"
).reindex(columns=LABELS)

fig, ax = plt.subplots(figsize=(10, 5))

plot.plot(
    kind="bar",
    ax=ax,
    color=[
        "#bd6262",
        "#94a3b8",
        "#438f79"
    ]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Class-specific F1")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=25)
ax.legend(frameon=False)
ax.set_title("Classwise sentiment performance")

save_figure(
    fig,
    "stage6c_05_classwise_f1",
    "Class-specific validation F1 across "
    "the architectural configurations.",
    "stage6c_04_classwise_results.csv"
)

# Figure 6 — Calibration.

plot = results_df.sort_values("ece")

fig, ax = plt.subplots(figsize=(8, 5))

ax.barh(
    plot["model"],
    plot["ece"],
    color="#4f83a3"
)

ax.set_xlabel("Expected calibration error")
ax.set_title("Uncalibrated validation reliability")

save_figure(
    fig,
    "stage6c_06_calibration",
    "Uncalibrated expected calibration error "
    "on the development cohort.",
    "stage6c_01_model_results.csv"
)

# Figure 7 — Adaptive gate behavior.

plot = gate_df[
    gate_df["model"] == "full"
].set_index("subgroup")

fig, ax = plt.subplots(figsize=(9, 5))

plot[
    [
        "mean_phrase_gate",
        "mean_directional_gate"
    ]
].plot(
    kind="bar",
    ax=ax,
    color=["#238875", "#9471bf"]
)

ax.set_ylabel("Mean learned gate weight")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=25)
ax.legend(frameon=False, fontsize=9)
ax.set_title("AMSC-Net adaptive fusion behavior")

save_figure(
    fig,
    "stage6c_07_gate_behavior",
    "Mean learned residual gate weights by "
    "heuristic validation subgroup.",
    "stage6c_06_gate_diagnostics.csv"
)

# Figure 8 — Parameter efficiency.

fig, ax = plt.subplots(figsize=(8, 5))

for _, row in results_df.iterrows():

    ax.scatter(
        row["extra_parameters"] / 1e6,
        row["macro_f1"],
        s=65
    )

    ax.annotate(
        row["model"],
        (
            row["extra_parameters"] / 1e6,
            row["macro_f1"]
        ),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlabel("Additional trainable parameters (millions)")
ax.set_ylabel("Validation macro-F1")
ax.set_title("Architectural parameter efficiency")

save_figure(
    fig,
    "stage6c_08_parameter_efficiency",
    "Validation performance relative to additional "
    "trainable classifier parameters.",
    "stage6c_01_model_results.csv"
)

# Figure 9 — Confusion matrix.

full_pred = PREDICTIONS["full"]

cm = confusion_matrix(
    Y_VAL,
    full_pred,
    labels=[0, 1, 2]
)

cm_table = pd.DataFrame(
    cm,
    columns=LABELS
)

cm_table.insert(
    0, "true_label", LABELS
)

save_table(
    cm_table,
    "stage6c_07_full_confusion_matrix"
)

fig, ax = plt.subplots(figsize=(5.8, 5))

im = ax.imshow(
    cm, cmap="Blues"
)

for i in range(3):

    for j in range(3):

        ax.text(
            j, i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > cm.max() / 2
                else "black"
            )
        )

ax.set_xticks(range(3), LABELS)
ax.set_yticks(range(3), LABELS)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Full AMSC-Net confusion matrix")

fig.colorbar(im, ax=ax)

save_figure(
    fig,
    "stage6c_09_confusion_matrix",
    "Full AMSC-Net validation confusion matrix.",
    "stage6c_07_full_confusion_matrix.csv"
)

# =====================================================================
# ARCHITECTURE DIAGRAM — ACTUAL IMPLEMENTED MODEL
# =====================================================================

from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

def diagram_box(
    x, y, w, h, text, color,
    fontsize=10
):

    patch = FancyBboxPatch(
        (x, y),
        w, h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor="#475569",
        linewidth=1.1
    )

    ax.add_patch(patch)

    ax.text(
        x + w/2,
        y + h/2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize
    )

def arrow(x1, y1, x2, y2):

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "->",
            "color": "#475569",
            "linewidth": 1.2
        }
    )

diagram_box(
    3.2, 10.9, 3.6, 0.65,
    "Bangla input / tokenization",
    "#f1f5f9"
)

diagram_box(
    2.7, 9.65, 4.6, 0.8,
    "Trainable BanglaBERT encoder\n"
    "Contextual token representations",
    "#dbeafe"
)

arrow(5, 10.9, 5, 10.45)

diagram_box(
    3.2, 8.4, 3.6, 0.65,
    "Projection: 768 → 128",
    "#f1f5f9"
)

arrow(5, 9.65, 5, 9.05)

diagram_box(
    0.3, 6.65, 2.8, 1.1,
    "Global branch\n"
    "CLS + masked mean",
    "#dbeafe"
)

diagram_box(
    3.6, 6.65, 2.8, 1.1,
    "Phrase branch\n"
    "Conv1D: 1, 3, 5\n"
    "Adaptive scale pooling",
    "#d1fae5"
)

diagram_box(
    6.9, 6.65, 2.8, 1.1,
    "Directional branch\n"
    "Left/right attention\n"
    "Context composition",
    "#ede9fe"
)

for x in [1.7, 5, 8.3]:

    arrow(5, 8.4, x, 7.75)

for x, label in [
    (0.7, "Global logits"),
    (4.0, "Phrase logits"),
    (7.3, "Directional logits")
]:

    diagram_box(
        x, 5.5, 2.0, 0.6,
        label, "#f8fafc", 9
    )

for x in [1.7, 5, 8.3]:

    arrow(x, 6.65, x, 6.1)

diagram_box(
    2.4, 4.05, 5.2, 0.85,
    "Jensen–Shannon disagreement\n"
    "+ entropy, margin and confidence",
    "#fef3c7"
)

for x in [1.7, 5, 8.3]:

    arrow(x, 5.5, 5, 4.9)

diagram_box(
    2.4, 2.8, 5.2, 0.75,
    "Adaptive gate MLP\n"
    "Bounded phrase and directional weights",
    "#d1fae5"
)

arrow(5, 4.05, 5, 3.55)

diagram_box(
    2.1, 1.5, 5.8, 0.8,
    "Residual logit fusion\n"
    "Global + weighted branch differences",
    "#dbeafe"
)

arrow(5, 2.8, 5, 2.3)

diagram_box(
    2.6, 0.25, 4.8, 0.7,
    "Negative / Neutral / Positive",
    "#f1f5f9"
)

arrow(5, 1.5, 5, 0.95)

ax.set_title(
    "Adaptive Multi-scale Sentiment Composition Network",
    fontsize=14,
    fontweight="bold"
)

save_figure(
    fig,
    "stage6c_10_amsc_architecture",
    "Implemented AMSC-Net architecture, including "
    "multi-scale phrase evidence, directional "
    "composition and disagreement-conditioned "
    "residual logit fusion. The original classifier "
    "contributes to the global logits.",
    "experiment_config.json"
)

for ext in ["png", "pdf", "svg"]:

    shutil.copy2(
        PUB / "figures" / ext
        / f"stage6c_10_amsc_architecture.{ext}",
        PUB / "architecture"
        / f"AMSCNet.{ext}"
    )

# =====================================================================
# STEP 79 — FINAL DECISION AND AUTOMATIC BACKUP
# =====================================================================

section("STEP 79 — RESEARCH DECISION")

full_row = results_df[
    results_df["model"] == "full"
].iloc[0]

control_row = results_df[
    results_df["model"] == "global_control"
].iloc[0]

best_row = results_df.iloc[0]

full_f1 = float(
    full_row["macro_f1"]
)

control_f1 = float(
    control_row["macro_f1"]
)

full_vs_base = stats_df[
    (stats_df["model"] == "full")
    & (stats_df["reference"] == "BanglaBERT")
].iloc[0]

full_vs_control = stats_df[
    (stats_df["model"] == "full")
    & (stats_df["reference"] == "global_control")
].iloc[0]

if (
    full_f1 > BASE_F1
    and full_f1 > control_f1
):

    DECISION = (
        "PROMISING: advance full AMSC-Net "
        "to multi-seed confirmation"
    )

elif full_f1 > BASE_F1:

    DECISION = (
        "PARTIAL: improved over original baseline "
        "but not over matched training control"
    )

else:

    DECISION = (
        "REFINEMENT REQUIRED: full architecture "
        "did not improve original baseline"
    )

summary = {
    "stage": "6c",
    "baseline_macro_f1": BASE_F1,
    "full_amsc_macro_f1": full_f1,
    "matched_control_macro_f1": control_f1,
    "best_configuration": best_row["model"],
    "best_validation_macro_f1":
        float(best_row["macro_f1"]),
    "full_delta_vs_baseline":
        full_f1 - BASE_F1,
    "full_delta_vs_control":
        full_f1 - control_f1,
    "full_vs_baseline_ci95": [
        float(full_vs_base["ci95_low"]),
        float(full_vs_base["ci95_high"])
    ],
    "full_vs_control_ci95": [
        float(full_vs_control["ci95_low"]),
        float(full_vs_control["ci95_high"])
    ],
    "decision": DECISION,
    "single_seed_screening": True,
    "validation_used_for_selection": True,
    "test_evaluated": False,
    "external_evaluation": False,
    "novelty_verified": False,
    "publication_figures_added": len(FIGURES)
}

save_json(
    summary,
    AUDIT / "stage6c_summary.json"
)

shutil.copy2(
    AUDIT / "stage6c_summary.json",
    BACKUP / "stage6c_summary.json"
)

# Restore and consolidate all figure manifests.

manifest_parts = []

for path in [
    PUB / "figure_manifest.csv",
    REPO / "research_reconstruction_v1"
         / "publication_assets/figure_manifest.csv",
    REPO / "research_stage6b"
         / "figure_manifest.csv"
]:

    if path.exists():

        frame = pd.read_csv(path)

        if "figure_id" in frame.columns:

            manifest_parts.append(frame)

manifest_parts.append(
    pd.DataFrame(FIGURES)
)

figure_manifest = pd.concat(
    manifest_parts,
    ignore_index=True
).drop_duplicates(
    "figure_id",
    keep="last"
)

figure_manifest.to_csv(
    PUB / "figure_manifest.csv",
    index=False
)

shutil.copy2(
    PUB / "figure_manifest.csv",
    BACKUP / "figure_manifest.csv"
)

for record in FIGURES:

    for ext in ["png", "pdf", "svg"]:

        relative = Path(record[ext])

        target = (
            BACKUP / "publication_assets"
            / relative
        )

        target.parent.mkdir(
            parents=True, exist_ok=True
        )

        shutil.copy2(
            PUB / relative,
            target
        )

    caption = (
        record["figure_id"] + ".txt"
    )

    target = (
        BACKUP / "publication_assets"
        / "captions" / caption
    )

    target.parent.mkdir(
        parents=True, exist_ok=True
    )

    shutil.copy2(
        PUB / "captions" / caption,
        target
    )

export_tables()

# Verify every model has a complete remote
# checkpoint/prediction/metadata set.

assets = remote_assets()

remote_records = []

for mode in MODES:

    metadata_name = (
        f"{mode}_seed42_completed.json"
    )

    metadata_path = (
        CHECKPOINTS / mode
        / "completed.json"
    )

    completed = json.loads(
        metadata_path.read_text()
    )

    for name, expected in [
        (
            f"{mode}_seed42_model.pt",
            completed["checkpoint_sha256"]
        ),
        (
            f"{mode}_seed42_val.npz",
            completed["prediction_sha256"]
        ),
        (
            metadata_name,
            sha256(metadata_path)
        )
    ]:

        assert name in assets

        verify_remote_asset(
            assets[name],
            expected
        )

        remote_records.append({
            "mode": mode,
            "name": name,
            "asset_id": assets[name]["id"],
            "sha256": expected,
            "size_bytes": assets[name]["size"]
        })

remote_manifest = {
    "repository": f"{OWNER}/{PRIVATE_REPO}",
    "release_tag": STAGE_TAG,
    "release_id": RELEASE_ID,
    "assets": remote_records,
    "experiment_config": CONFIG
}

manifest_file = (
    AUDIT / "stage6c_remote_manifest.json"
)

save_json(
    remote_manifest,
    manifest_file
)

upload_private(manifest_file)

shutil.copy2(
    manifest_file,
    BACKUP / "stage6c_remote_manifest.json"
)

# Compact reproducibility package.

compact_zip = (
    WORK / "sentiment_stage6c_compact.zip"
)

with zipfile.ZipFile(
    compact_zip,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for path in S6C.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix == ".pt":
            continue

        z.write(
            path,
            path.relative_to(S6C)
        )

with zipfile.ZipFile(compact_zip) as z:

    assert z.testzip() is None

# Content-addressed package name permits
# safe reruns without overwriting prior assets.

package_hash = sha256(compact_zip)

upload_private(
    compact_zip,
    f"stage6c_compact_{package_hash[:12]}.zip"
)

# Preserve publication package locally and
# upload it to the private artifact repository.

publication_zip = (
    WORK / "sentiment_publication_assets.zip"
)

with zipfile.ZipFile(
    publication_zip,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for path in PUB.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(PUB)
            )

with zipfile.ZipFile(publication_zip) as z:

    assert z.testzip() is None

publication_hash = sha256(publication_zip)

upload_private(
    publication_zip,
    f"publication_assets_{publication_hash[:12]}.zip"
)

FINAL_COMMIT = push_public(
    "Complete AMSC-Net screening, paired analysis "
    "and publication assets"
)

# =====================================================================
# FINAL RESEARCH REPORT
# =====================================================================

section("FINAL STAGE 6C RESEARCH REPORT")

print("Dataset recovery: PASS")
print("Baseline prediction alignment: PASS")
print("Architecture smoke tests: PASS")
print("Parameter matching: PASS")
print("Completed configurations:", len(MODES))
print("Public GitHub synchronized: True")
print("Private checkpoint backup: PASS")
print("Final public commit:", FINAL_COMMIT)

print("\nMODEL PERFORMANCE")

print(
    results_df[
        [
            "model",
            "accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "delta_vs_banglabert",
            "extra_parameters"
        ]
    ].round(5).to_string(index=False)
)

print("\nLINGUISTIC SUBGROUPS")

print(
    subgroup_df.pivot(
        index="subgroup",
        columns="model",
        values="macro_f1"
    ).round(4).to_string()
)

print("\nPAIRED STATISTICAL COMPARISONS")

print(
    stats_df.round(5)
    .to_string(index=False)
)

print("\nFULL MODEL GATE DIAGNOSTICS")

print(
    gate_df[
        gate_df["model"] == "full"
    ].round(4).to_string(index=False)
)

print("\nRESEARCH DECISION")

print("Original BanglaBERT F1:", BASE_F1)
print("Matched control F1:", control_f1)
print("Full AMSC-Net F1:", full_f1)

print(
    "Full improvement vs baseline:",
    round(
        100 * (full_f1 - BASE_F1), 4
    ),
    "percentage points"
)

print(
    "Full improvement vs matched control:",
    round(
        100 * (full_f1 - control_f1), 4
    ),
    "percentage points"
)

print("Best configuration:", best_row["model"])
print("Decision:", DECISION)

print("\nPUBLICATION ASSETS")

print("New unique figures:", len(FIGURES))

for ext in ["png", "pdf", "svg"]:

    print(
        ext.upper(),
        "figures:",
        len(list(
            (PUB / "figures" / ext)
            .glob(f"*.{ext}")
        ))
    )

print(
    "Architecture diagram formats:",
    "PNG, PDF, SVG"
)

print(
    "Verified model artifact files:",
    len(remote_records)
)

print(
    "Private release:",
    f"https://github.com/{OWNER}/"
    f"{PRIVATE_REPO}/releases/tag/{STAGE_TAG}"
)

print(
    "Public research branch:",
    f"https://github.com/{OWNER}/"
    f"{PUBLIC_REPO}/tree/{BRANCH}"
)

print("\nManual file saving required: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("Test evaluated: NO")
print("Novelty conclusively established: NO")
print("Publication-ready superiority established: NO")

print(
    "\nRuntime minutes:",
    round(
        (time.time() - START) / 60, 2
    )
)

print(
    "\nREADY FOR STAGE 6D — "
    "CONFIRMATORY MULTI-SEED EVALUATION "
    "OR TARGETED ARCHITECTURE REFINEMENT"
)

SESSION.headers.pop(
    "Authorization", None
)

SESSION.close()

GIT_ENV.clear()

TOKEN = None
credential = None

gc.collect()
torch.cuda.empty_cache()
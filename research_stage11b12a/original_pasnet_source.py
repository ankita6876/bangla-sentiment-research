
# =============================================================================
# STAGE 7B — STRONGER REPRESENTATION LEARNING AND PAS-NET
# STEPS 107–116
# GITHUB FIRST | GPU | NO TEST EVALUATION | NO MANUAL FILE SAVING
# =============================================================================

import os
import sys
import re
import gc
import json
import time
import math
import random
import shutil
import hashlib
import zipfile
import base64
import subprocess
import unicodedata
import warnings

from pathlib import Path
from getpass import getpass
from collections import defaultdict

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from scipy.special import softmax
from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    log_loss
)

warnings.filterwarnings("ignore")

START = time.time()

assert torch.cuda.is_available(), "Enable Kaggle GPU first."

DEVICE = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/csebuetnlp/normalizer"
    ],
    check=True
)

from normalizer import normalize as bn_normalize

from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"
EXPECTED_COMMIT = "beb674841f88696f31dddeb88f324f94383d2f32"

BASE_TAG = "stage6b-locked-baselines-v1"
PREVIOUS_TAG = "stage7a-bottleneck-discovery-v1"
NEW_TAG = "stage7b-pasnet-screening-v1"

RAW_SHA = (
    "4580950af06bbfbe5bd79e20957b63a2ed"
    "122af2a6a3e685b10576d3f70241c5"
)

OUT = WORK / "sentiment_research_stage7b"
REC = OUT / "recovered"
AUDIT = OUT / "audit"
TABLES = OUT / "tables"
MODELS = OUT / "checkpoints"
PRED = OUT / "private_predictions"
FIGURES = OUT / "figures"

PUB = WORK / "sentiment_publication_assets"
REPO = WORK / "sentiment_stage7b_github"
BACKUP = REPO / "research_stage7b"

SEED = 42
MAX_LENGTH = 96
EPOCHS = 4
PATIENCE = 2
EFFECTIVE_BATCH = 32

ENCODER_LR = 2e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = .01

DIM = 128
DROPOUT = .15
SUPCON_WEIGHT = .10
PROTOTYPE_WEIGHT = .10
HARD_GAMMA = 1.0
TEMPERATURE = .10

CLASSES = ["Negative", "Neutral", "Positive"]
LABEL_MAP = dict(zip(CLASSES, range(3)))

BASE_MODEL = "csebuetnlp/banglabert"
LARGE_MODEL = "csebuetnlp/banglabert_large"

# One-seed screening before expensive multi-seed confirmation.
EXPERIMENTS = [
    ("base_standard", BASE_MODEL, "standard"),
    ("base_supcon", BASE_MODEL, "supcon"),
    ("base_prototype", BASE_MODEL, "prototype"),
    ("base_pasnet", BASE_MODEL, "pasnet"),
    ("large_standard", LARGE_MODEL, "standard"),
    ("large_pasnet", LARGE_MODEL, "pasnet")
]

for p in [OUT, REC, AUDIT, TABLES, MODELS, PRED, FIGURES, PUB]:
    p.mkdir(parents=True, exist_ok=True)

for fmt in ["png", "pdf", "svg"]:
    (FIGURES / fmt).mkdir(parents=True, exist_ok=True)
    (PUB / "figures" / fmt).mkdir(parents=True, exist_ok=True)

for sub in [
    "tables/csv", "tables/latex",
    "captions", "scripts", "architecture"
]:
    (PUB / sub).mkdir(parents=True, exist_ok=True)

def section(title):
    print("\n" + "=" * 94)
    print(title)
    print("=" * 94)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(obj, path):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(
            y, pred, labels=[0,1,2],
            average="macro", zero_division=0
        ),
        "macro_precision": precision_score(
            y, pred, labels=[0,1,2],
            average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y, pred, labels=[0,1,2],
            average="macro", zero_division=0
        )
    }

def save_table(rows, name):
    df = pd.DataFrame(rows)

    df.to_csv(TABLES / f"{name}.csv", index=False)
    df.to_csv(PUB / "tables/csv" / f"{name}.csv", index=False)

    df.to_latex(
        PUB / "tables/latex" / f"{name}.tex",
        index=False, escape=True,
        float_format="%.4f"
    )

    return df

# =============================================================================
# STEP 107 — VERIFIED GITHUB CONNECTION
# =============================================================================

section("STEP 107 — VERIFIED GITHUB CONNECTION")

TOKEN = getpass("Enter your working GitHub token: ").strip()

API = "https://api.github.com"

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

def api_get(path):
    r = SESSION.get(API + path, timeout=90)
    r.raise_for_status()
    return r.json()

assert api_get("/user")["login"].lower() == OWNER.lower()

public_info = api_get(f"/repos/{OWNER}/{PUBLIC_REPO}")
private_info = api_get(f"/repos/{OWNER}/{PRIVATE_REPO}")

assert public_info["permissions"]["push"]
assert private_info["permissions"]["push"]
assert private_info["private"]

GIT_ENV = os.environ.copy()

GIT_ENV.update({
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0": "Authorization: Basic " +
        base64.b64encode(
            f"x-access-token:{TOKEN}".encode()
        ).decode()
})

def git(*args):
    r = subprocess.run(
        ["git", *args],
        cwd=str(REPO) if REPO.exists() else str(WORK),
        env=GIT_ENV,
        capture_output=True,
        text=True
    )

    if r.returncode:
        raise RuntimeError(
            r.stderr.replace(TOKEN, "[REDACTED]")
        )

    return r.stdout.strip()

if not (REPO / ".git").exists():
    git(
        "clone", "--branch", BRANCH,
        f"https://github.com/{OWNER}/{PUBLIC_REPO}.git",
        str(REPO)
    )
else:
    git("fetch", "origin")
    git("checkout", BRANCH)
    git("pull", "--ff-only", "origin", BRANCH)

git("merge-base", "--is-ancestor", EXPECTED_COMMIT, "HEAD")
git("config", "user.name", OWNER)
git("config", "user.email", f"{OWNER}@users.noreply.github.com")

BACKUP.mkdir(parents=True, exist_ok=True)

def push_public(message):
    git("add", "research_stage7b")

    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", message)

    git("push", "origin", f"HEAD:{BRANCH}")

    local = git("rev-parse", "HEAD")
    remote = git(
        "ls-remote", "origin", f"refs/heads/{BRANCH}"
    ).split()[0]

    assert local == remote
    print("Public GitHub synchronized:", local)
    return local

print("Authentication: PASS")
print("Public repository access: PASS")
print("Private repository access: PASS")

# =============================================================================
# STEP 108 — RECOVER PREVIOUS ARTIFACTS
# =============================================================================

section("STEP 108 — EXACT ARTIFACT RECOVERY")

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

    return {a["name"]: a for a in result}

def download_asset(asset, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

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

        temp = target.with_suffix(target.suffix + ".partial")

        with SESSION.get(
            url,
            headers={"Accept": "application/octet-stream"},
            stream=True,
            timeout=(30, 600)
        ) as response:

            response.raise_for_status()

            with open(temp, "wb") as f:
                for chunk in response.iter_content(1024**2):
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
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None

        for name in z.namelist():
            assert (destination / name).resolve().is_relative_to(
                destination
            )

        z.extractall(destination)

base_assets = assets_of(get_release(BASE_TAG))

package = download_asset(
    base_assets["stage6b_reproducibility_package.zip"],
    REC / "baseline_package.zip"
)

safe_extract(package, REC / "baseline_package")

previous_assets = assets_of(get_release(PREVIOUS_TAG))

publication_candidates = [
    a for n, a in previous_assets.items()
    if n.startswith("sentiment_stage7a_publication_")
    and n.endswith(".zip")
]

assert publication_candidates, "Stage 7A publication backup missing."

previous_publication = max(
    publication_candidates,
    key=lambda a: a["created_at"]
)

previous_zip = download_asset(
    previous_publication,
    REC / previous_publication["name"]
)

safe_extract(previous_zip, PUB)

private_candidates = [
    a for n, a in previous_assets.items()
    if n.startswith("sentiment_stage7a_private_")
    and n.endswith(".zip")
]

assert private_candidates

previous_private = max(
    private_candidates,
    key=lambda a: a["created_at"]
)

private_zip = download_asset(
    previous_private,
    REC / previous_private["name"]
)

safe_extract(private_zip, REC / "stage7a_private")

print("Previous publication assets recovered.")

for fmt in ["png", "pdf", "svg"]:
    print(
        fmt.upper(),
        len(list((PUB / "figures" / fmt).rglob(f"*.{fmt}")))
    )

# =============================================================================
# STEP 109 — EXACT TRAIN AND VALIDATION MEMBERSHIP
# =============================================================================

section("STEP 109 — LOCKED DATASET RECOVERY")

raw_path = next(
    (
        p for p in INPUT.rglob("BECS Dataset.csv")
        if sha256(p) == RAW_SHA
    ),
    None
)

assert raw_path is not None, "Verified BECS dataset not found."

raw = pd.read_csv(raw_path)

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()

raw["text_norm"] = raw["Bangla"].map(normalize_text)
raw["Label"] = raw["Label"].astype(str).str.strip().str.title()

raw["sample_id"] = raw["text_norm"].map(
    lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
)

raw["label_id"] = raw["Label"].map(LABEL_MAP)

conflict_ids = set(
    raw.groupby("sample_id")["label_id"].nunique()
       .loc[lambda s: s > 1].index
)

clean = raw.loc[
    ~raw["sample_id"].isin(conflict_ids)
].drop_duplicates("sample_id").set_index("sample_id")

assert len(clean) == 12893

DATA = {}

for split, expected_n in [("train", 9204), ("val", 1840)]:

    paths = list(
        (REC / "baseline_package").rglob(f"seed42_{split}.csv")
    )

    assert len(paths) == 1, paths

    membership = pd.read_csv(paths[0])

    if "partition_order" in membership:
        membership = membership.sort_values("partition_order")

    ids = membership["sample_id"].astype(str).tolist()

    df = clean.loc[ids].reset_index()
    df["group_id"] = membership["group_id"].astype(str).to_numpy()

    assert len(df) == expected_n
    assert df["sample_id"].tolist() == ids

    if "Label" in membership:
        assert (
            df["Label"].str.casefold().to_numpy()
            == membership["Label"].astype(str)
                         .str.strip().str.casefold().to_numpy()
        ).all()

    df["model_text"] = df["text_norm"].map(bn_normalize)
    df["word_count"] = df["text_norm"].str.split().str.len()

    DATA[split] = df

    print(split, "| samples:", len(df), "| exact membership: PASS")

assert not set(DATA["train"]["sample_id"]).intersection(
    DATA["val"]["sample_id"]
)

assert not set(DATA["train"]["group_id"]).intersection(
    DATA["val"]["group_id"]
)

assert not set(DATA["train"]["model_text"]).intersection(
    DATA["val"]["model_text"]
)

Y = DATA["val"]["label_id"].to_numpy(dtype=int)
IDS = DATA["val"]["sample_id"].to_numpy(dtype=str)
GROUPS = DATA["val"]["group_id"].to_numpy(dtype=str)

NEG = ["না", "নয়", "নয়", "নেই", "নাই"]
CON = ["কিন্তু", "তবে", "যদিও", "অথচ", "তবুও", "তথাপি"]

def cue_pattern(words):
    return re.compile(
        r"(?<![\w\u0980-\u09ff])(?:"
        + "|".join(map(re.escape, words))
        + r")(?![\w\u0980-\u09ff])"
    )

neg = DATA["val"]["text_norm"].map(
    lambda s: bool(cue_pattern(NEG).search(s))
).to_numpy()

con = DATA["val"]["text_norm"].map(
    lambda s: bool(cue_pattern(CON).search(s))
).to_numpy()

long_threshold = DATA["train"]["word_count"].quantile(.75)

MASKS = {
    "all": np.ones(len(Y), dtype=bool),
    "negation_candidate": neg,
    "contrast_candidate": con,
    "both_operators": neg & con,
    "any_operator": neg | con,
    "no_detected_operator": ~(neg | con),
    "long_text": (
        DATA["val"]["word_count"] > long_threshold
    ).to_numpy(),
    "shorter_text": (
        DATA["val"]["word_count"] <= long_threshold
    ).to_numpy()
}

print("Leakage audit: PASS")
print("Test partition loaded: NO")

# =============================================================================
# STEP 110 — EXPERIMENT PROTOCOL AND PRIVATE RELEASE
# =============================================================================

section("STEP 110 — EXPERIMENT PROTOCOL")

CONFIG = {
    "stage": "7B",
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "effective_batch": EFFECTIVE_BATCH,
    "encoder_lr": ENCODER_LR,
    "head_lr": HEAD_LR,
    "weight_decay": WEIGHT_DECAY,
    "dimension": DIM,
    "dropout": DROPOUT,
    "supcon_weight": SUPCON_WEIGHT,
    "prototype_weight": PROTOTYPE_WEIGHT,
    "hard_gamma": HARD_GAMMA,
    "temperature": TEMPERATURE,
    "experiments": EXPERIMENTS,
    "initialization": "original pretrained encoder",
    "selection": "development validation macro-F1",
    "test_evaluated": False,
    "raw_sha256": RAW_SHA,
    "train_membership_sha256": hashlib.sha256(
        "\n".join(DATA["train"]["sample_id"]).encode()
    ).hexdigest(),
    "validation_membership_sha256": hashlib.sha256(
        "\n".join(IDS).encode()
    ).hexdigest()
}

CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()

save_json(CONFIG, AUDIT / "experiment_config.json")
save_json(CONFIG, BACKUP / "experiment_config.json")

history = get_ipython().history_manager.input_hist_raw

source_candidates = [
    s for s in history
    if "STAGE 7B — STRONGER REPRESENTATION LEARNING" in s
    and len(s) > 10000
]

assert source_candidates, "Could not capture executable source."

source = source_candidates[-1]
assert TOKEN not in source

for path in [
    AUDIT / "stage7b_complete.py",
    BACKUP / "stage7b_complete.py",
    PUB / "scripts/stage7b_complete.py"
]:
    path.write_text(source, encoding="utf-8")

push_public("Stage 7B: preserve PAS-Net protocol and source")

response = SESSION.get(
    f"{API}/repos/{OWNER}/{PRIVATE_REPO}/releases/tags/{NEW_TAG}",
    timeout=90
)

if response.status_code == 404:
    response = SESSION.post(
        f"{API}/repos/{OWNER}/{PRIVATE_REPO}/releases",
        json={
            "tag_name": NEW_TAG,
            "name": "Stage 7B — PAS-Net screening",
            "body": "Private checkpoints and validation artifacts."
        },
        timeout=90
    )

response.raise_for_status()
RELEASE = response.json()

REMOTE_MANIFEST = []

def upload_verified(path, name=None):
    path = Path(path)
    name = name or path.name
    digest = sha256(path)

    assets = assets_of(RELEASE)

    if name in assets:
        asset = assets[name]

    else:
        with open(path, "rb") as f:
            response = SESSION.post(
                RELEASE["upload_url"].split("{")[0],
                params={"name": name},
                headers={"Content-Type": "application/octet-stream"},
                data=f,
                timeout=(30, 1200)
            )

        response.raise_for_status()
        asset = response.json()

    assert asset["size"] == path.stat().st_size

    remote_digest = asset.get("digest") or ""

    if remote_digest.startswith("sha256:"):
        assert remote_digest.split(":", 1)[1] == digest
    else:
        check = download_asset(
            asset, REC / "remote_check" / name
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

    print("Private remote asset verified:", name)
    return record

print("Private release:", RELEASE["tag_name"])

# =============================================================================
# STEP 111 — TOKENIZATION AND ARCHITECTURE
# =============================================================================

section("STEP 111 — PAS-NET IMPLEMENTATION")

TOKEN_CACHE = {}

class SentimentDataset(Dataset):

    def __init__(self, df, tokenizer):

        enc = tokenizer(
            df["model_text"].tolist(),
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt"
        )

        self.ids = enc["input_ids"]
        self.attention = enc["attention_mask"]
        self.valid = (
            enc["attention_mask"].bool()
            & ~enc["special_tokens_mask"].bool()
        )

        self.labels = torch.tensor(
            df["label_id"].to_numpy(),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.ids[i],
            "attention_mask": self.attention[i],
            "valid": self.valid[i],
            "labels": self.labels[i]
        }

def get_datasets(model_name):

    if model_name not in TOKEN_CACHE:

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        TOKEN_CACHE[model_name] = (
            tokenizer,
            SentimentDataset(DATA["train"], tokenizer),
            SentimentDataset(DATA["val"], tokenizer)
        )

    return TOKEN_CACHE[model_name]

class PASNet(nn.Module):

    def __init__(self, model_name, mode):

        super().__init__()

        self.mode = mode

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        # Shared semantic representation in every configuration.
        self.project = nn.Sequential(
            nn.Linear(hidden * 2, DIM),
            nn.LayerNorm(DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

        self.semantic_head = nn.Linear(DIM, 3)

        # These components are shared across all model modes.
        self.metric_head = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.GELU(),
            nn.Linear(DIM, DIM)
        )

        # Two trainable prototypes per sentiment class.
        self.prototypes = nn.Parameter(
            torch.randn(3, 2, DIM) * .02
        )

        self.gate = nn.Sequential(
            nn.Linear(DIM + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 0.0)

        if mode == "standard":
            for p in self.metric_head.parameters():
                p.requires_grad = False

        if mode not in ["prototype", "pasnet"]:
            self.prototypes.requires_grad = False

        if mode != "pasnet":
            for p in self.gate.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, valid):

        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        cls = hidden[:, 0]

        mask = valid.unsqueeze(-1).to(hidden.dtype)

        mean = (
            (hidden * mask).sum(1)
            / mask.sum(1).clamp_min(1)
        )

        z = self.project(torch.cat([cls, mean], dim=-1))

        semantic = self.semantic_head(z)

        if self.mode == "standard":
            return {
                "logits": semantic,
                "semantic": semantic,
                "embedding": None,
                "prototype": None,
                "gate": torch.zeros(
                    len(z), device=z.device
                )
            }

        embedding = F.normalize(
            self.metric_head(z).float(),
            dim=-1
        )

        if self.mode == "supcon":
            return {
                "logits": semantic,
                "semantic": semantic,
                "embedding": embedding,
                "prototype": None,
                "gate": torch.zeros(
                    len(z), device=z.device
                )
            }

        prototypes = F.normalize(
            self.prototypes.float(), dim=-1
        )

        similarities = torch.einsum(
            "bd,ckd->bck",
            embedding, prototypes
        )

        # Smooth aggregation of within-class prototypes.
        prototype_logits = (
            torch.logsumexp(10 * similarities, dim=-1)
            - math.log(2)
        )

        if self.mode == "prototype":

            alpha = torch.full(
                (len(z), 1), .25, device=z.device
            )

        else:

            probs = semantic.float().softmax(-1)

            sorted_p = probs.sort(dim=-1).values

            entropy = -(
                probs * probs.clamp_min(1e-8).log()
            ).sum(-1) / math.log(3)

            uncertainty = torch.stack([
                1 - probs.max(-1).values,
                entropy,
                1 - (
                    sorted_p[:, -1] - sorted_p[:, -2]
                )
            ], dim=-1)

            alpha = .5 * torch.sigmoid(
                self.gate(
                    torch.cat([
                        z,
                        uncertainty.detach().to(z.dtype)
                    ], dim=-1)
                )
            )

        logits = (
            (1 - alpha) * semantic.float()
            + alpha * prototype_logits
        )

        return {
            "logits": logits,
            "semantic": semantic,
            "embedding": embedding,
            "prototype": prototype_logits,
            "gate": alpha.squeeze(-1)
        }

def supervised_contrastive(embedding, labels):

    if embedding is None:
        return torch.zeros((), device=labels.device)

    similarity = (
        embedding @ embedding.T
    ) / TEMPERATURE

    n = len(labels)

    eye = torch.eye(
        n, device=labels.device, dtype=torch.bool
    )

    positive = (
        labels[:, None] == labels[None, :]
    ) & ~eye

    similarity = similarity.masked_fill(eye, -1e4)

    log_prob = (
        similarity
        - torch.logsumexp(similarity, dim=1, keepdim=True)
    )

    positive_count = positive.sum(1)
    active = positive_count > 0

    if not active.any():
        return embedding.sum() * 0

    per_sample = -(
        log_prob * positive.float()
    ).sum(1) / positive_count.clamp_min(1)

    return per_sample[active].mean()

def objective(output, labels, mode):

    ce = F.cross_entropy(
        output["logits"].float(),
        labels,
        reduction="none"
    )

    if mode == "pasnet":

        # Bounded hard-example emphasis.
        # Retains all examples rather than discarding easy samples.
        pt = torch.exp(-ce.detach())

        weights = 1 + (1 - pt).pow(HARD_GAMMA)
        weights = weights / weights.mean()

        classification = (weights * ce).mean()

    else:
        classification = ce.mean()

    supcon = torch.zeros((), device=labels.device)
    proto = torch.zeros((), device=labels.device)

    if mode in ["supcon", "prototype", "pasnet"]:
        supcon = supervised_contrastive(
            output["embedding"], labels
        )

    if mode in ["prototype", "pasnet"]:
        proto = F.cross_entropy(
            output["prototype"], labels
        )

    total = (
        classification
        + SUPCON_WEIGHT * supcon
        + PROTOTYPE_WEIGHT * proto
    )

    return total

# Architecture smoke test.
seed_all(SEED)

_, smoke_train, _ = get_datasets(BASE_MODEL)

smoke_model = PASNet(BASE_MODEL, "pasnet").to(DEVICE)

batch = next(iter(DataLoader(smoke_train, batch_size=8)))

batch = {k: v.to(DEVICE) for k, v in batch.items()}

labels = batch.pop("labels")

output = smoke_model(**batch)
loss = objective(output, labels, "pasnet")

assert output["logits"].shape == (8, 3)
assert torch.isfinite(loss)

loss.backward()

for name, module in [
    ("encoder", smoke_model.encoder),
    ("semantic", smoke_model.semantic_head),
    ("metric", smoke_model.metric_head),
    ("gate", smoke_model.gate)
]:
    grads = [
        p.grad for p in module.parameters()
        if p.requires_grad and p.grad is not None
    ]

    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    print(name, "gradient validation: PASS")

assert smoke_model.prototypes.grad is not None
assert torch.isfinite(smoke_model.prototypes.grad).all()

print("Prototype gradient validation: PASS")

del smoke_model, batch, output, loss
cleanup()

# =============================================================================
# STEP 112 — TRAINING AND EVALUATION UTILITIES
# =============================================================================

section("STEP 112 — TRAINING UTILITIES")

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    logits = []
    gates = []
    labels = []

    for batch in loader:

        batch = {
            k: v.to(DEVICE, non_blocking=True)
            for k, v in batch.items()
        }

        y = batch.pop("labels")

        with torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            output = model(**batch)

        logits.append(
            output["logits"].float().cpu().numpy()
        )

        gates.append(
            output["gate"].float().cpu().numpy()
        )

        labels.append(y.cpu().numpy())

    logits = np.concatenate(logits)
    gates = np.concatenate(gates)
    labels = np.concatenate(labels)

    assert np.array_equal(labels, Y)
    assert np.isfinite(logits).all()

    probs = softmax(logits, axis=1)

    return (
        metrics(labels, probs.argmax(1)),
        logits,
        gates
    )

RESULT_ROWS = []
HISTORY_ROWS = []
PREDICTIONS = {}
GATES = {}
FAILURES = []

def sync_tables():

    save_table(
        RESULT_ROWS, "stage7b_01_model_results"
    )

    save_table(
        HISTORY_ROWS, "stage7b_02_training_history"
    )

    for p in TABLES.glob("*.csv"):
        dest = BACKUP / "tables" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

    save_json(FAILURES, BACKUP / "failed_runs.json")

def train_experiment(name, model_name, mode):

    seed_all(SEED)

    is_large = model_name == LARGE_MODEL

    # Lower microbatch for large encoder.
    microbatch = 8 if is_large else 16
    accumulation = EFFECTIVE_BATCH // microbatch

    tokenizer, train_ds, val_ds = get_datasets(model_name)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=microbatch,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=16 if is_large else 32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = PASNet(model_name, mode).to(DEVICE)

    if is_large:
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()

    encoder_parameters = [
        p for p in model.encoder.parameters()
        if p.requires_grad
    ]

    head_parameters = [
        p for n, p in model.named_parameters()
        if not n.startswith("encoder.")
        and p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LR},
            {"params": head_parameters, "lr": HEAD_LR}
        ],
        weight_decay=WEIGHT_DECAY
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / accumulation
    )

    total_steps = EPOCHS * updates_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda")

    run_dir = MODELS / name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt = run_dir / "model.pt"

    best_f1 = -1
    best_metrics = None
    best_logits = None
    best_gates = None
    best_epoch = 0
    stale = 0

    run_history = []

    parameters = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("Trainable parameters:", parameters)
    print("Microbatch:", microbatch)
    print("Effective batch:", EFFECTIVE_BATCH)

    for epoch in range(1, EPOCHS + 1):

        model.train()
        optimizer.zero_grad(set_to_none=True)

        losses = []

        for step, batch in enumerate(train_loader, start=1):

            batch = {
                k: v.to(DEVICE, non_blocking=True)
                for k, v in batch.items()
            }

            labels = batch.pop("labels")

            window_start = (
                (step - 1) // accumulation
            ) * accumulation

            divisor = min(
                accumulation,
                len(train_loader) - window_start
            )

            with torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                output = model(**batch)
                loss = objective(output, labels, mode)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Nonfinite training loss in {name}"
                )

            scaler.scale(loss / divisor).backward()

            losses.append(float(loss.detach().cpu()))

            if (
                step % accumulation == 0
                or step == len(train_loader)
            ):

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                old_scale = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()

                if scaler.get_scale() >= old_scale:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

        result, logits, gates = evaluate(
            model, val_loader
        )

        record = {
            "model": name,
            "epoch": epoch,
            "train_loss": np.mean(losses),
            **result
        }

        run_history.append(record)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss {np.mean(losses):.4f} | "
            f"Acc {result['accuracy']:.4f} | "
            f"F1 {result['macro_f1']:.4f}",
            flush=True
        )

        if result["macro_f1"] > best_f1 + 1e-7:

            best_f1 = result["macro_f1"]
            best_metrics = result
            best_logits = logits.copy()
            best_gates = gates.copy()
            best_epoch = epoch
            stale = 0

            # FP16 storage only; training remains mixed precision.
            # Validation logits preserve original evaluation precision.
            state = {
                k: (
                    v.detach().cpu().half()
                    if v.is_floating_point()
                    else v.detach().cpu()
                )
                for k, v in model.state_dict().items()
            }

            temp = run_dir / "model.tmp"

            torch.save(state, temp)
            temp.replace(ckpt)

            del state

            print("Best checkpoint saved.")

        else:

            stale += 1

            if stale >= PATIENCE:
                print("Early stopping.")
                break

    assert best_metrics is not None
    assert ckpt.exists()

    # Check whether compressed checkpoint reproduces predictions.
    state = torch.load(
        ckpt, map_location="cpu", weights_only=True
    )

    model.load_state_dict(state, strict=True)
    del state

    restored_metrics, restored_logits, restored_gates = evaluate(
        model, val_loader
    )

    restored_agreement = float(
        (
            restored_logits.argmax(1)
            == best_logits.argmax(1)
        ).mean()
    )

    # Use restored checkpoint outputs for reported results.
    pred_path = PRED / f"{name}_seed42_val.npz"

    np.savez_compressed(
        pred_path,
        sample_id=IDS,
        labels=Y,
        logits=restored_logits,
        gates=restored_gates
    )

    result_row = {
        "model": name,
        "backbone": model_name,
        "mode": mode,
        "seed": SEED,
        "best_epoch": best_epoch,
        "trainable_parameters": parameters,
        "original_best_macro_f1": best_f1,
        "checkpoint_prediction_agreement": restored_agreement,
        **restored_metrics
    }

    metadata = {
        "config_hash": CONFIG_HASH,
        "result": result_row,
        "history": run_history,
        "checkpoint_sha256": sha256(ckpt),
        "prediction_sha256": sha256(pred_path),
        "checkpoint_dtype": "float16 storage",
        "model_name": model_name,
        "mode": mode
    }

    metadata_path = run_dir / "completed.json"
    save_json(metadata, metadata_path)

    del model, optimizer, scheduler, scaler
    cleanup()

    return metadata, ckpt, pred_path, metadata_path

# =============================================================================
# STEP 113 — SIX EXPERIMENTS WITH AUTOMATIC RESUME
# =============================================================================

section("STEP 113 — REPRESENTATION LEARNING EXPERIMENTS")

for name, model_name, mode in EXPERIMENTS:

    section("TRAINING CONFIGURATION — " + name)

    run_dir = MODELS / name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"{name}_seed42_model.pt"
    pred_name = f"{name}_seed42_val.npz"
    meta_name = f"{name}_seed42_completed.json"

    ckpt = run_dir / "model.pt"
    pred_path = PRED / pred_name
    meta_path = run_dir / "completed.json"

    remote = assets_of(RELEASE)

    try:

        if all(
            n in remote
            for n in [ckpt_name, pred_name, meta_name]
        ):

            download_asset(remote[meta_name], meta_path)
            download_asset(remote[pred_name], pred_path)

            metadata = json.loads(
                meta_path.read_text()
            )

            assert metadata["config_hash"] == CONFIG_HASH

            assert sha256(pred_path) == metadata[
                "prediction_sha256"
            ]

            digest = remote[ckpt_name].get("digest") or ""

            if digest.startswith("sha256:"):
                assert digest.split(":", 1)[1] == metadata[
                    "checkpoint_sha256"
                ]
            else:
                check = download_asset(
                    remote[ckpt_name], ckpt
                )
                assert sha256(check) == metadata[
                    "checkpoint_sha256"
                ]
                check.unlink()

            print("Previously completed remote run recovered.")

        else:

            local_complete = (
                ckpt.exists()
                and pred_path.exists()
                and meta_path.exists()
            )

            if local_complete:

                metadata = json.loads(meta_path.read_text())

                local_complete = (
                    metadata["config_hash"] == CONFIG_HASH
                    and sha256(ckpt) == metadata["checkpoint_sha256"]
                    and sha256(pred_path) == metadata["prediction_sha256"]
                )

            if not local_complete:

                # Do not overwrite a partially uploaded remote run.
                partial = [
                    n for n in [ckpt_name, pred_name, meta_name]
                    if n in remote
                ]

                assert not partial, (
                    "Partial remote run requires recovery: "
                    + str(partial)
                )

                metadata, ckpt, pred_path, meta_path = (
                    train_experiment(name, model_name, mode)
                )

            upload_verified(ckpt, ckpt_name)
            upload_verified(pred_path, pred_name)
            upload_verified(meta_path, meta_name)

            # Delete large local checkpoint only after remote verification.
            ckpt.unlink(missing_ok=True)

        with np.load(pred_path, allow_pickle=False) as z:

            assert np.array_equal(z["sample_id"], IDS)
            assert np.array_equal(z["labels"], Y)

            PREDICTIONS[name] = z["logits"].copy()
            GATES[name] = z["gates"].copy()

        RESULT_ROWS.append(metadata["result"])
        HISTORY_ROWS.extend(metadata["history"])

        dest = BACKUP / "run_metadata" / meta_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(meta_path, dest)

        sync_tables()

        push_public(
            f"Stage 7B: verified completion of {name}"
        )

        print(
            name,
            "| macro-F1:",
            round(metadata["result"]["macro_f1"], 5),
            "| permanent backup: PASS"
        )

    except Exception as e:

        message = str(e).replace(TOKEN, "[REDACTED]")

        FAILURES.append({
            "model": name,
            "error_type": type(e).__name__,
            "error": message[:1500]
        })

        print("RUN NOT COMPLETED:", name)
        print(message[:1500])

        cleanup()

        sync_tables()

        push_public(
            f"Stage 7B: record incomplete run {name}"
        )

        # Continue other experiments without fabricating results.

    cleanup()

assert RESULT_ROWS, "No experiment completed successfully."

RESULTS = save_table(
    RESULT_ROWS, "stage7b_01_model_results"
).sort_values("macro_f1", ascending=False)

HISTORY = save_table(
    HISTORY_ROWS, "stage7b_02_training_history"
)

print(RESULTS.round(5).to_string(index=False))

# =============================================================================
# STEP 114 — COMPARISONS, SUBGROUPS AND STATISTICS
# =============================================================================

section("STEP 114 — SCIENTIFIC COMPARISON")

ALL_PROBS = {
    name: softmax(logits, axis=1)
    for name, logits in PREDICTIONS.items()
}

# Recover previous nine-model ensemble for descriptive comparison.
previous_npz = list(
    (REC / "stage7a_private").rglob(
        "aligned_confirmation_predictions.npz"
    )
)

assert len(previous_npz) == 1

with np.load(previous_npz[0], allow_pickle=False) as z:

    old_ids = z["sample_id"].astype(str)

    lookup = {sid: i for i, sid in enumerate(old_ids)}

    assert set(lookup) == set(IDS)

    order = np.array([lookup[sid] for sid in IDS])

    assert np.array_equal(z["labels"][order], Y)

    previous_keys = [
        f"{family}_seed{seed}"
        for family in [
            "standard", "fixed_fusion", "matched_control"
        ]
        for seed in [42, 123, 2026]
    ]

    old_probs = np.mean(
        [z[k][order] for k in previous_keys],
        axis=0
    )

ALL_PROBS["previous_nine_ensemble"] = old_probs

assert abs(
    metrics(Y, old_probs.argmax(1))["macro_f1"]
    - .8215240777580629
) < 1e-6

COMPARISON = save_table([
    {
        "model": name,
        **metrics(Y, p.argmax(1)),
        "nll": log_loss(Y, p, labels=[0,1,2])
    }
    for name, p in ALL_PROBS.items()
], "stage7b_03_all_model_comparison")

SUBGROUPS = save_table([
    {
        "model": name,
        "subgroup": subgroup,
        "n": int(mask.sum()),
        **metrics(Y[mask], p.argmax(1)[mask])
    }
    for name, p in ALL_PROBS.items()
    for subgroup, mask in MASKS.items()
    if mask.any()
], "stage7b_04_subgroup_results")

class_rows = []
confusion_rows = []

for name, p in ALL_PROBS.items():

    pred = p.argmax(1)

    report = classification_report(
        Y, pred,
        labels=[0,1,2],
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    for label in CLASSES:
        class_rows.append({
            "model": name,
            "class": label,
            **report[label]
        })

    cm = confusion_matrix(Y, pred, labels=[0,1,2])

    for i in range(3):
        for j in range(3):
            confusion_rows.append({
                "model": name,
                "true_class": CLASSES[i],
                "predicted_class": CLASSES[j],
                "count": int(cm[i,j])
            })

CLASSWISE = save_table(
    class_rows, "stage7b_05_classwise_results"
)

CONFUSION = save_table(
    confusion_rows, "stage7b_06_confusion_matrices"
)

def fast_f1(y, pred):

    cm = np.bincount(
        3*y + pred, minlength=9
    ).reshape(3,3)

    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)

    return np.divide(
        2*tp, denom,
        out=np.zeros(3),
        where=denom > 0
    ).mean()

group_indices = [
    np.flatnonzero(GROUPS == g)
    for g in np.unique(GROUPS)
]

rng = np.random.default_rng(SEED)

bootstrap_indices = [
    np.concatenate([
        group_indices[j]
        for j in rng.integers(
            0, len(group_indices), len(group_indices)
        )
    ])
    for _ in range(2000)
]

PAIRS = [
    ("base_supcon", "base_standard"),
    ("base_prototype", "base_standard"),
    ("base_pasnet", "base_standard"),
    ("base_pasnet", "base_supcon"),
    ("base_pasnet", "base_prototype"),
    ("large_pasnet", "large_standard"),
    ("large_standard", "base_standard"),
    ("large_pasnet", "base_pasnet")
]

PAIRS += [
    (name, "previous_nine_ensemble")
    for name in PREDICTIONS
]

stat_rows = []

for name, ref in PAIRS:

    if name not in ALL_PROBS or ref not in ALL_PROBS:
        continue

    pred = ALL_PROBS[name].argmax(1)
    reference = ALL_PROBS[ref].argmax(1)

    diffs = np.array([
        fast_f1(Y[idx], pred[idx])
        - fast_f1(Y[idx], reference[idx])
        for idx in bootstrap_indices
    ])

    gain = int(((pred == Y) & (reference != Y)).sum())
    loss = int(((pred != Y) & (reference == Y)).sum())

    stat_rows.append({
        "model": name,
        "reference": ref,
        "delta_macro_f1": (
            fast_f1(Y, pred) - fast_f1(Y, reference)
        ),
        "ci95_low": np.quantile(diffs, .025),
        "ci95_high": np.quantile(diffs, .975),
        "gained_correct": gain,
        "lost_correct": loss,
        "mcnemar_p": (
            binomtest(gain, gain + loss, .5).pvalue
            if gain + loss else 1.0
        )
    })

STATS = pd.DataFrame(stat_rows)

if len(STATS):

    p = STATS["mcnemar_p"].to_numpy()
    order = np.argsort(p)

    adjusted = np.empty(len(p))

    adjusted[order] = np.minimum(
        1,
        np.maximum.accumulate(
            p[order] * (len(p) - np.arange(len(p)))
        )
    )

    STATS["mcnemar_p_holm"] = adjusted

save_table(STATS, "stage7b_07_paired_statistics")

GATE_RESULTS = save_table([
    {
        "model": name,
        "subgroup": subgroup,
        "n": int(mask.sum()),
        "mean_gate": float(gates[mask].mean()),
        "std_gate": float(gates[mask].std())
    }
    for name, gates in GATES.items()
    for subgroup, mask in MASKS.items()
    if mask.any()
], "stage7b_08_gate_diagnostics")

print("\nMODEL COMPARISON")
print(
    COMPARISON.sort_values(
        "macro_f1", ascending=False
    ).round(5).to_string(index=False)
)

print("\nLINGUISTIC SUBGROUPS")
print(
    SUBGROUPS.pivot(
        index="subgroup", columns="model", values="macro_f1"
    ).round(4).to_string()
)

print("\nCLASSWISE F1")
print(
    CLASSWISE.pivot(
        index="class", columns="model", values="f1-score"
    ).round(4).to_string()
)

print("\nPAIRED STATISTICS")
print(STATS.round(5).to_string(index=False))

# =============================================================================
# STEP 115 — PUBLICATION FIGURES AND IMMEDIATE GITHUB BACKUP
# =============================================================================

section("STEP 115 — PUBLICATION ASSETS")

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

    for fmt in ["png", "pdf", "svg"]:

        path = FIGURES / fmt / f"{name}.{fmt}"

        fig.savefig(
            path,
            dpi=600 if fmt == "png" else None
        )

        target = PUB / "figures" / fmt / path.name
        shutil.copy2(path, target)

        public_target = BACKUP / "figures" / fmt / path.name
        public_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, public_target)

        FIGURE_MANIFEST.append({
            "figure": name,
            "format": fmt,
            "sha256": sha256(path),
            "caption": caption
        })

    caption_path = PUB / "captions" / f"{name}.txt"
    caption_path.write_text(caption, encoding="utf-8")

    dest = BACKUP / "captions" / caption_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(caption_path, dest)

    # Every figure is backed up immediately after generation.
    push_public("Stage 7B figure: " + name)

    plt.show()
    plt.close(fig)

# Figure 1 — Model comparison.
fig, ax = plt.subplots(figsize=(9,5))

plot = COMPARISON.sort_values("macro_f1")

ax.barh(plot["model"], 100*plot["macro_f1"])
ax.set_xlim(0,100)
ax.set_xlabel("Validation macro-F1 (%)")
ax.set_title("Representation learning and sentiment architecture")

for i,v in enumerate(plot["macro_f1"]):
    ax.text(100*v+.4, i, f"{100*v:.2f}", va="center")

save_figure(
    fig, "stage7b_01_model_comparison",
    "Single-seed development screening and historical ensemble comparison."
)

# Figure 2 — Training convergence.
fig, ax = plt.subplots(figsize=(9,5))

for name, part in HISTORY.groupby("model"):

    ax.plot(
        part["epoch"], part["macro_f1"],
        marker="o", label=name
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation macro-F1")
ax.set_title("Training convergence")
ax.legend(frameon=False, fontsize=8)
ax.set_xticks(range(1,EPOCHS+1))

save_figure(
    fig, "stage7b_02_training_convergence",
    "Validation macro-F1 during model development."
)

# Figure 3 — Training loss.
fig, ax = plt.subplots(figsize=(9,5))

for name, part in HISTORY.groupby("model"):

    ax.plot(
        part["epoch"], part["train_loss"],
        marker="o", label=name
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Training objective")
ax.set_title("Training objective convergence")
ax.legend(frameon=False, fontsize=8)

save_figure(
    fig, "stage7b_03_training_loss",
    "Training objectives differ across modes and are not directly equivalent."
)

# Figure 4 — Linguistic subgroup heatmap.
pivot = SUBGROUPS.pivot(
    index="subgroup", columns="model", values="macro_f1"
)

fig, ax = plt.subplots(figsize=(11,6))

im = ax.imshow(
    pivot.to_numpy(),
    vmin=0, vmax=1, cmap="YlGnBu"
)

ax.set_xticks(
    range(len(pivot.columns)),
    pivot.columns, rotation=40, ha="right"
)

ax.set_yticks(range(len(pivot.index)), pivot.index)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.iloc[i,j]
        ax.text(
            j,i,f"{v:.3f}",
            ha="center",va="center",
            color="white" if v>.7 else "black",
            fontsize=8
        )

fig.colorbar(im, ax=ax, label="Macro-F1")

save_figure(
    fig, "stage7b_04_linguistic_subgroups",
    "Descriptive linguistic subgroup results based on lexical heuristics."
)

# Figure 5 — Class-specific F1.
fig, ax = plt.subplots(figsize=(11,5))

CLASSWISE.pivot(
    index="class", columns="model", values="f1-score"
).plot(kind="bar", ax=ax, rot=0)

ax.set_ylim(0,1)
ax.set_ylabel("F1")
ax.set_title("Sentiment class separation")
ax.legend(frameon=False, fontsize=8)

save_figure(
    fig, "stage7b_05_classwise_f1",
    "Class-specific validation performance."
)

# Figure 6 — Paired effects.
if len(STATS):

    fig, ax = plt.subplots(
        figsize=(10, max(5, .42*len(STATS)))
    )

    labels = []

    for i,row in STATS.reset_index(drop=True).iterrows():

        ax.plot(
            [100*row["ci95_low"],100*row["ci95_high"]],
            [i,i], linewidth=2
        )

        ax.scatter(100*row["delta_macro_f1"],i)

        labels.append(
            row["model"] + " vs " + row["reference"]
        )

    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(range(len(labels)),labels)
    ax.set_xlabel("Macro-F1 difference (percentage points)")

    save_figure(
        fig, "stage7b_06_paired_effects",
        "Exploratory paired group-bootstrap confidence intervals."
    )

# Figure 7 — Parameter efficiency.
fig, ax = plt.subplots(figsize=(9,5))

ax.scatter(
    RESULTS["trainable_parameters"]/1e6,
    100*RESULTS["macro_f1"]
)

for _,row in RESULTS.iterrows():

    ax.annotate(
        row["model"],
        (
            row["trainable_parameters"]/1e6,
            100*row["macro_f1"]
        ),
        xytext=(4,4),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlabel("Trainable parameters (millions)")
ax.set_ylabel("Macro-F1 (%)")

save_figure(
    fig, "stage7b_07_parameter_efficiency",
    "Parameter count versus development validation performance."
)

# Figure 8 — Best newly trained model confusion matrix.
best_name = RESULTS.iloc[0]["model"]

cm = confusion_matrix(
    Y, ALL_PROBS[best_name].argmax(1),
    labels=[0,1,2]
)

fig, ax = plt.subplots(figsize=(6,5))

im = ax.imshow(cm, cmap="Blues")

for i in range(3):
    for j in range(3):
        ax.text(
            j,i,str(cm[i,j]),
            ha="center",va="center",
            color="white" if cm[i,j]>cm.max()/2 else "black"
        )

ax.set_xticks(range(3),CLASSES)
ax.set_yticks(range(3),CLASSES)
ax.set_xlabel("Predicted sentiment")
ax.set_ylabel("Recorded sentiment")
ax.set_title(best_name)
fig.colorbar(im,ax=ax)

save_figure(
    fig, "stage7b_08_best_confusion_matrix",
    "Confusion matrix of the strongest newly trained screening configuration."
)

# Figure 9 — Architecture diagram.
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12,8))

ax.set_xlim(0,12)
ax.set_ylim(0,9)
ax.axis("off")

def block(x,y,w,h,label,color):

    patch = FancyBboxPatch(
        (x,y),w,h,
        boxstyle="round,pad=0.04",
        facecolor=color,
        edgecolor="#334155",
        linewidth=1.3
    )

    ax.add_patch(patch)

    ax.text(
        x+w/2,y+h/2,label,
        ha="center",va="center",
        fontsize=11
    )

def arrow(x1,y1,x2,y2,dashed=False):

    ax.annotate(
        "",
        xy=(x2,y2),
        xytext=(x1,y1),
        arrowprops={
            "arrowstyle":"->",
            "linewidth":1.5,
            "color":"#334155",
            "linestyle":"--" if dashed else "-"
        }
    )

block(4,8,4,.6,"Bangla input text","#e2e8f0")
block(3.5,6.8,5,.7,"Pretrained BanglaBERT encoder","#dbeafe")
block(3.5,5.6,5,.7,"CLS + masked mean projection","#dbeafe")

block(.5,4.1,4,.8,"Semantic classifier","#ccfbf1")
block(7.5,4.1,4,.8,"Normalized metric embedding","#ede9fe")

block(7.5,2.8,4,.8,"Two prototypes per class","#ede9fe")
block(4.1,1.5,3.8,.8,"Adaptive logit fusion","#fef3c7")
block(4.1,.2,3.8,.7,"Sentiment prediction","#e2e8f0")

block(.2,2.5,3.4,.8,"Hard-example weighted CE","#fee2e2")
block(8.7,5.7,3,.7,"Contrastive loss","#fee2e2")

arrow(6,8,6,7.5)
arrow(6,6.8,6,6.3)
arrow(4.5,5.6,2.5,4.9)
arrow(7.5,5.6,9.5,4.9)
arrow(9.5,4.1,9.5,3.6)
arrow(2.5,4.1,4.8,2.3)
arrow(9.5,2.8,7.2,2.3)
arrow(6,1.5,6,.9)
arrow(9.5,4.9,10.2,5.7,True)
arrow(4.1,1.9,3.6,2.8,True)

ax.text(
    6,8.9,
    "Prototype-Guided Adaptive Sentiment Network",
    ha="center",fontsize=15,fontweight="bold"
)

save_figure(
    fig, "stage7b_09_pasnet_architecture",
    "Provisional PAS-Net architecture; dashed arrows denote training supervision."
)

for fmt in ["png","pdf","svg"]:

    src = FIGURES / fmt / f"stage7b_09_pasnet_architecture.{fmt}"

    shutil.copy2(
        src, PUB / "architecture" / f"PASNet.{fmt}"
    )

# =============================================================================
# STEP 116 — FINAL BACKUP AND RESEARCH DECISION
# =============================================================================

section("STEP 116 — VERIFIED FINAL BACKUP")

best = RESULTS.iloc[0]

previous_f1 = metrics(
    Y, old_probs.argmax(1)
)["macro_f1"]

SUMMARY = {
    "stage": "7B",
    "completed_runs": len(RESULTS),
    "failed_runs": FAILURES,
    "best_new_model": best["model"],
    "best_new_macro_f1": float(best["macro_f1"]),
    "previous_nine_ensemble_macro_f1": previous_f1,
    "improvement_percentage_points": float(
        100*(best["macro_f1"]-previous_f1)
    ),
    "target_84_percent_reached": bool(
        best["macro_f1"] >= .84
    ),
    "new_unique_figures": len(
        set(x["figure"] for x in FIGURE_MANIFEST)
    ),
    "test_evaluated": False,
    "external_evaluated": False,
    "novelty_verified": False,
    "publication_superiority_established": False,
    "limitations": [
        "Single-seed development screening.",
        "Validation reused for model selection.",
        "Prototype and PAS-Net objectives differ.",
        "No separate capacity-matched control yet.",
        "Contrastive microbatch size differs by backbone.",
        "Architecture novelty requires literature verification."
    ]
}

save_json(SUMMARY, AUDIT / "stage7b_summary.json")
save_json(SUMMARY, BACKUP / "stage7b_summary.json")

save_json(
    FIGURE_MANIFEST,
    BACKUP / "figure_manifest.json"
)

manifest_path = PUB / "stage7b_figure_manifest.csv"

pd.DataFrame(FIGURE_MANIFEST).to_csv(
    manifest_path, index=False
)

for p in TABLES.glob("*.csv"):

    dest = BACKUP / "tables" / p.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p,dest)

    tex = PUB / "tables/latex" / (p.stem+".tex")

    if tex.exists():
        dest = BACKUP / "latex_tables" / tex.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tex,dest)

# Private compact research archive.
compact = WORK / "sentiment_stage7b_compact.zip"

with zipfile.ZipFile(
    compact,"w",zipfile.ZIP_DEFLATED
) as z:

    for folder in [AUDIT,TABLES,PRED,FIGURES,MODELS]:

        for p in folder.rglob("*"):

            if p.is_file() and p.suffix not in [".pt",".tmp"]:

                z.write(p,p.relative_to(OUT))

# Cumulative publication archive.
publication = WORK / "sentiment_stage7b_publication.zip"

with zipfile.ZipFile(
    publication,"w",zipfile.ZIP_DEFLATED
) as z:

    for p in PUB.rglob("*"):

        if p.is_file():

            z.write(p,p.relative_to(PUB))

for path in [compact,publication]:

    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None

    upload_verified(
        path,
        f"{path.stem}_{sha256(path)[:12]}.zip"
    )

remote = assets_of(RELEASE)

for row in RESULT_ROWS:

    name = row["model"]

    for suffix in [
        "model.pt","val.npz","completed.json"
    ]:

        asset_name = f"{name}_seed42_{suffix}"

        assert asset_name in remote

        asset = remote[asset_name]

        REMOTE_MANIFEST.append({
            "name":asset_name,
            "asset_id":asset["id"],
            "size":asset["size"],
            "digest":asset.get("digest")
        })

remote_manifest = AUDIT / "stage7b_remote_manifest.json"

save_json(
    {
        "release":NEW_TAG,
        "assets":REMOTE_MANIFEST
    },
    remote_manifest
)

upload_verified(
    remote_manifest,
    f"stage7b_manifest_{sha256(remote_manifest)[:12]}.json"
)

shutil.copy2(
    remote_manifest,
    BACKUP / remote_manifest.name
)

FINAL_COMMIT = push_public(
    "Stage 7B: complete representation screening and permanent backup"
)

section("FINAL STAGE 7B RESEARCH REPORT")

print("Completed configurations:",len(RESULTS))
print("Failed configurations:",len(FAILURES))
print("GitHub synchronized: True")
print("Private artifact backup: PASS")
print("Final public commit:",FINAL_COMMIT)

print("\nMODEL PERFORMANCE")
print(RESULTS.round(5).to_string(index=False))

print("\nALL MODEL COMPARISON")
print(
    COMPARISON.sort_values(
        "macro_f1",ascending=False
    ).round(5).to_string(index=False)
)

print("\nLINGUISTIC SUBGROUPS")
print(
    SUBGROUPS.pivot(
        index="subgroup",columns="model",values="macro_f1"
    ).round(5).to_string()
)

print("\nCLASSWISE F1")
print(
    CLASSWISE.pivot(
        index="class",columns="model",values="f1-score"
    ).round(5).to_string()
)

print("\nPAIRED STATISTICS")
print(STATS.round(5).to_string(index=False))

print("\nGATE DIAGNOSTICS")
print(GATE_RESULTS.round(5).to_string(index=False))

print("\nRESEARCH SUMMARY")
print(json.dumps(SUMMARY,indent=2))

print("\nPUBLICATION ASSETS")

for fmt in ["png","pdf","svg"]:

    print(
        fmt.upper(),"figures:",
        len(list((PUB/"figures"/fmt).rglob(f"*.{fmt}")))
    )

print("Every new figure uploaded to GitHub: YES")
print("Private release:",RELEASE["html_url"])
print("Manual file saving required: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("Test evaluated: NO")
print("External evaluation: NO")

print(
    "\nRuntime minutes:",
    round((time.time()-START)/60,2)
)

print("\nSTAGE 7B COMPLETE")
print("READY FOR STAGE 7C — TARGETED ARCHITECTURAL DEVELOPMENT")

cleanup()
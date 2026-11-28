# ============================================================
# STAGE 8B — CONTEXTUAL EVIDENCE RETRIEVAL NETWORK
# STEPS 156–166
# ============================================================

import os, sys, gc, re, json, time, math, random
import hashlib, shutil, zipfile, subprocess, getpass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from scipy.special import softmax
from scipy.stats import binomtest
from transformers import AutoTokenizer, AutoModelForSequenceClassification

START = time.time()

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"
EXPECTED_COMMIT = "ad9471d6b81f992ae8fdf4b84cb45f4fe7eaae7b"

BASE_TAG = "stage6b-locked-baselines-v1"
PREVIOUS_TAG = "stage8a-hds-screening-v1"
NEW_TAG = "stage8b-cernet-screening-v1"

OUT = WORK / "sentiment_research_stage8b"
REC = OUT / "recovered"
TABLES = OUT / "tables"
AUDIT = OUT / "audit"
MODELS = OUT / "models"
PRED = OUT / "private_predictions"
FIGURES = OUT / "figures"

REPO = WORK / "sentiment_stage8b_github"
PUB = WORK / "sentiment_publication_assets"

for p in [OUT, REC, TABLES, AUDIT, MODELS, PRED, FIGURES, PUB]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["Negative", "Neutral", "Positive"]

def section(s):
    print("\n" + "=" * 90)
    print(s)
    print("=" * 90)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def save_json(obj, path):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    )

seed_all(SEED)

# ============================================================
# STEP 156 — GITHUB AUTHENTICATION
# ============================================================

section("STEP 156 — VERIFIED GITHUB CONNECTION")

TOKEN = getpass.getpass("Enter your working GitHub token: ")

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

API = "https://api.github.com"

def api_get(url):
    r = SESSION.get(url, timeout=120)
    r.raise_for_status()
    return r.json()

assert api_get(f"{API}/user")["login"]

PUBLIC_API = f"{API}/repos/{OWNER}/{PUBLIC_REPO}"
PRIVATE_API = f"{API}/repos/{OWNER}/{PRIVATE_REPO}"

api_get(PUBLIC_API)
assert api_get(PRIVATE_API)["private"]

import base64

auth = base64.b64encode(
    f"x-access-token:{TOKEN}".encode()
).decode()

GIT_ENV = os.environ.copy()
GIT_ENV.update({
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
    "GIT_TERMINAL_PROMPT": "0"
})

def git(*args, cwd=REPO):
    r = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=GIT_ENV,
        capture_output=True,
        text=True
    )
    if r.returncode:
        raise RuntimeError("Git command failed: " + args[0])
    return r.stdout.strip()

if not (REPO / ".git").exists():
    git(
        "clone", "--single-branch", "--branch", BRANCH,
        f"https://github.com/{OWNER}/{PUBLIC_REPO}.git",
        str(REPO), cwd=WORK
    )

git("fetch", "origin", BRANCH)
git("checkout", BRANCH)

assert not git("status", "--porcelain"), "Repository has local changes."

git("pull", "--ff-only", "origin", BRANCH)
git("merge-base", "--is-ancestor", EXPECTED_COMMIT, "HEAD")

git("config", "user.name", "Research Backup")
git("config", "user.email", "research-backup@users.noreply.github.com")

BACKUP = REPO / "research_stage8b"
BACKUP.mkdir(exist_ok=True)

def push_public(message):
    git("add", "research_stage8b")
    if git("status", "--porcelain"):
        git("commit", "-m", message)
        git("push", "origin", BRANCH)
    local = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", f"refs/heads/{BRANCH}").split()[0]
    assert local == remote
    print("GitHub synchronized:", local)
    return local

print("Authentication: PASS")
print("Previous research commit: VERIFIED")
print("GPU:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU")

# ============================================================
# STEP 157 — RECOVER PRIVATE ARTIFACTS
# ============================================================

section("STEP 157 — PRIVATE ARTIFACT RECOVERY")

def get_release(tag):
    return api_get(f"{PRIVATE_API}/releases/tags/{tag}")

def release_assets(release):
    return api_get(
        f"{PRIVATE_API}/releases/{release['id']}/assets?per_page=100"
    )

def download_asset(asset, dest, expected_hash=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    digest = asset.get("digest") or ""
    expected = expected_hash or (
        digest.split(":", 1)[1]
        if digest.startswith("sha256:") else None
    )

    if dest.exists() and dest.stat().st_size == asset["size"]:
        if expected and sha256(dest) == expected:
            return dest

    with SESSION.get(
        asset["url"],
        headers={"Accept": "application/octet-stream"},
        stream=True, timeout=(30, 600)
    ) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    assert dest.stat().st_size == asset["size"]

    if expected:
        assert sha256(dest) == expected

    print("Recovered:", dest.name)
    return dest

def safe_extract(path, dest):
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            assert (dest / name).resolve().is_relative_to(dest)
        z.extractall(dest)

base_release = get_release(BASE_TAG)
base_assets = {a["name"]: a for a in release_assets(base_release)}

manifest_path = download_asset(
    base_assets["backup_manifest.json"],
    REC / "baseline_manifest.json"
)

manifest = json.loads(manifest_path.read_text())

def find_manifest_hash(obj, name):
    if isinstance(obj, dict):
        if obj.get("name") == name:
            return obj.get("sha256")
        for value in obj.values():
            result = find_manifest_hash(value, name)
            if result:
                return result
    elif isinstance(obj, list):
        for value in obj:
            result = find_manifest_hash(value, name)
            if result:
                return result
    return None

package_name = "stage6b_reproducibility_package.zip"
weight_name = "BanglaBERT_seed42_best_model.pt"

package = download_asset(
    base_assets[package_name],
    REC / package_name,
    find_manifest_hash(manifest, package_name)
)

weights = download_asset(
    base_assets[weight_name],
    REC / weight_name,
    find_manifest_hash(manifest, weight_name)
)

safe_extract(package, REC / "baseline")

previous_release = get_release(PREVIOUS_TAG)
previous_assets = release_assets(previous_release)

publication_asset = next(
    a for a in previous_assets
    if a["name"].startswith("sentiment_stage8a_publication_")
)

publication_zip = download_asset(
    publication_asset, REC / publication_asset["name"]
)

safe_extract(publication_zip, REC / "previous_publication")

for p in (REC / "previous_publication").rglob("*"):
    if not p.is_file():
        continue
    parts = p.relative_to(REC / "previous_publication").parts
    if "publication_assets" in parts:
        rel = Path(*parts[parts.index("publication_assets")+1:])
    else:
        rel = Path(*parts)
    dest = PUB / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)

release_response = SESSION.get(
    f"{PRIVATE_API}/releases/tags/{NEW_TAG}"
)

if release_response.status_code == 404:
    r = SESSION.post(
        f"{PRIVATE_API}/releases",
        json={
            "tag_name": NEW_TAG,
            "name": "Stage 8B CER-Net screening",
            "body": "Private research checkpoints and reproducibility artifacts.",
            "draft": False
        }
    )
    r.raise_for_status()
    RELEASE = r.json()
else:
    release_response.raise_for_status()
    RELEASE = release_response.json()

REMOTE_MANIFEST = []

def upload_private(path):
    path = Path(path)
    digest = sha256(path)

    assets = {a["name"]: a for a in release_assets(RELEASE)}
    name = path.name

    if name in assets:
        asset = assets[name]
        if asset.get("digest") != f"sha256:{digest}":
            name = f"{path.stem}_{digest[:12]}{path.suffix}"
            asset = assets.get(name)
        if asset is not None:
            assert asset["size"] == path.stat().st_size
            if asset.get("digest"):
                assert asset["digest"] == f"sha256:{digest}"
        else:
            asset = None
    else:
        asset = None

    if asset is None:
        url = RELEASE["upload_url"].split("{")[0]
        with open(path, "rb") as f:
            r = SESSION.post(
                url,
                params={"name": name},
                headers={"Content-Type": "application/octet-stream"},
                data=f,
                timeout=600
            )
        r.raise_for_status()
        asset = r.json()

    assert asset["size"] == path.stat().st_size

    if asset.get("digest"):
        assert asset["digest"] == f"sha256:{digest}"
    else:
        tmp = REC / ("verify_" + name)
        download_asset(asset, tmp, digest)
        assert sha256(tmp) == digest
        tmp.unlink()

    REMOTE_MANIFEST.append({
        "name": name,
        "sha256": digest,
        "size": asset["size"],
        "asset_id": asset["id"],
        "remote_verified": True
    })
    print("Remote verified:", name)

# ============================================================
# STEP 158 — LOCKED DATASET RECOVERY
# ============================================================

section("STEP 158 — LOCKED DATASET RECOVERY")

import unicodedata

RAW_HASH = "4580950af06bbfbe5bd79e20957b63a2ed122af2a6a3e685b10576d3f70241c5"

raw_candidates = [
    p for p in INPUT.rglob("*.csv")
    if "becs" in p.name.lower()
]

raw_path = next(
    (p for p in raw_candidates if sha256(p) == RAW_HASH),
    None
)

assert raw_path is not None, "Attach the original BECS Kaggle dataset."

raw = pd.read_csv(raw_path)

def norm_text(s):
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", s)
    return re.sub(r"\s+", " ", s).strip().casefold()

raw["text_norm"] = raw["Bangla"].map(norm_text)
raw["Label"] = raw["Label"].astype(str).str.strip().str.title()

raw["sample_id"] = raw["text_norm"].map(
    lambda x: hashlib.sha256(x.encode()).hexdigest()
)

try:
    from normalizer import normalize as bn_normalize
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/csebuetnlp/normalizer"
    ])
    from normalizer import normalize as bn_normalize

DATA = {}

for split in ["train", "val"]:
    files = list((REC / "baseline").rglob(f"seed42_{split}.csv"))
    assert len(files) == 1, (split, files)

    membership = pd.read_csv(files[0]).sort_values("partition_order")

    df = raw.iloc[membership["raw_row_id"].astype(int)].copy()
    df = df.reset_index(drop=True)

    assert np.array_equal(
        df["sample_id"].astype(str),
        membership["sample_id"].astype(str)
    )

    assert np.array_equal(
        df["Label"].astype(str),
        membership["Label"].astype(str)
    )

    df["group_id"] = membership["group_id"].to_numpy()
    df["label_id"] = df["Label"].map(dict(zip(LABELS, range(3))))
    df["model_text"] = df["text_norm"].map(bn_normalize)
    df["word_count"] = df["text_norm"].str.split().str.len()

    DATA[split] = df

assert len(DATA["train"]) == 9204
assert len(DATA["val"]) == 1840

for col in ["sample_id", "group_id", "model_text"]:
    assert not (
        set(DATA["train"][col]) & set(DATA["val"][col])
    ), col

YTR = DATA["train"]["label_id"].to_numpy(dtype=int)
YVA = DATA["val"]["label_id"].to_numpy(dtype=int)

GTR = DATA["train"]["group_id"].to_numpy()
GVA = DATA["val"]["group_id"].to_numpy()

IDS = DATA["val"]["sample_id"].to_numpy(dtype=str)

print("Training:", len(YTR))
print("Validation:", len(YVA))
print("Group and normalized-text overlap: ZERO")
print("Test partition loaded: NO")

# ============================================================
# STEP 159 — FROZEN SEMANTIC REPRESENTATIONS
# ============================================================

section("STEP 159 — SEMANTIC REPRESENTATION EXTRACTION")

MODEL_NAME = "csebuetnlp/banglabert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3
)

state = torch.load(weights, map_location="cpu", weights_only=True)

if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
elif isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

encoder.load_state_dict(state, strict=True)
del state

encoder.to(DEVICE).eval()

EMBED = {}
BASE_LOGITS = {}

for split in ["train", "val"]:
    vectors, logits = [], []
    texts = DATA[split]["model_text"].tolist()

    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = tokenizer(
                texts[start:start+64],
                padding="max_length",
                truncation=True,
                max_length=64,
                return_special_tokens_mask=True,
                return_tensors="pt"
            )

            special = batch.pop("special_tokens_mask").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.autocast(
                device_type=DEVICE,
                enabled=DEVICE == "cuda",
                dtype=torch.float16
            ):
                output = encoder(
                    **batch, output_hidden_states=True
                )

                h = output.hidden_states[-1].float()
                mask = (
                    batch["attention_mask"].bool() &
                    ~special.bool()
                ).float().unsqueeze(-1)

                mean = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
                rep = torch.cat([h[:, 0], mean], dim=1)

            vectors.append(rep.cpu().numpy())
            logits.append(output.logits.float().cpu().numpy())

    EMBED[split] = np.concatenate(vectors).astype(np.float32)
    BASE_LOGITS[split] = np.concatenate(logits).astype(np.float32)

    print(split, EMBED[split].shape)

del encoder
gc.collect()

if DEVICE == "cuda":
    torch.cuda.empty_cache()

base_f1 = f1_score(
    YVA, BASE_LOGITS["val"].argmax(1), average="macro"
)

assert abs(base_f1 - 0.8159419852657895) < 0.002

print("Recovered BanglaBERT validation F1:", base_f1)

# ============================================================
# STEP 160 — GROUP-DISJOINT RETRIEVAL
# ============================================================

section("STEP 160 — SEMANTIC AND CHARACTER RETRIEVAL")

# Important:
# The frozen encoder was trained on the complete training set.
# Group-disjoint neighbors prevent direct self/group retrieval,
# but training embeddings are still in-sample representations.
# This is an exploratory frozen-backbone experiment.

K = 16

def top_neighbors(sim, k):
    idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    values = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-values, axis=1)
    return (
        np.take_along_axis(idx, order, axis=1),
        np.take_along_axis(values, order, axis=1)
    )

def summarize_neighbors(idx, sim, labels):
    y = labels[idx]
    onehot = np.eye(3, dtype=np.float32)[y]
    w = softmax(sim / 0.10, axis=1)

    weighted = (onehot * w[:, :, None]).sum(1)
    counts = onehot.mean(1)

    class_max = np.stack([
        np.where(y == c, sim, -1.0).max(1)
        for c in range(3)
    ], axis=1)

    stats = np.stack([
        sim[:, 0], sim.mean(1), sim.std(1),
        sim[:, -1],
        -(weighted * np.log(weighted + 1e-8)).sum(1)
    ], axis=1)

    return np.concatenate(
        [weighted, counts, class_max, stats], axis=1
    ).astype(np.float32)

def retrieve(query_sem, ref_sem, query_char, ref_char,
             query_groups, ref_groups, ref_labels):

    result = []

    ref_sem = normalize(ref_sem)
    query_sem = normalize(query_sem)

    for start in range(0, len(query_sem), 128):
        end = min(start + 128, len(query_sem))

        semantic = query_sem[start:end] @ ref_sem.T
        lexical = (
            query_char[start:end] @ ref_char.T
        ).toarray()

        excluded = (
            query_groups[start:end, None] ==
            ref_groups[None, :]
        )

        semantic[excluded] = -np.inf
        lexical[excluded] = -np.inf

        assert np.all((~excluded).sum(1) >= K)

        combined = 0.65 * semantic + 0.35 * lexical

        features = []

        for matrix in [semantic, lexical, combined]:
            idx, sim = top_neighbors(matrix, K)

            assert not np.any(
                query_groups[start:end, None] == ref_groups[idx]
            )

            features.append(
                summarize_neighbors(idx, sim, ref_labels)
            )

        result.append(np.concatenate(features, axis=1))

    return np.concatenate(result)

RETR = {}

# Build training retrieval features from disjoint folds.
folds = StratifiedGroupKFold(
    n_splits=3, shuffle=True, random_state=42
)

for fold, (fit, hold) in enumerate(
    folds.split(EMBED["train"], YTR, GTR)
):
    assert not set(GTR[fit]) & set(GTR[hold])

    tfidf = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5),
        min_df=2, max_features=100000,
        sublinear_tf=True, dtype=np.float32
    )

    train_char = tfidf.fit_transform(
        DATA["train"].iloc[fit]["text_norm"]
    )
    hold_char = tfidf.transform(
        DATA["train"].iloc[hold]["text_norm"]
    )

    features = retrieve(
        EMBED["train"][hold],
        EMBED["train"][fit],
        hold_char, train_char,
        GTR[hold], GTR[fit], YTR[fit]
    )

    if "train" not in RETR:
        RETR["train"] = np.zeros(
            (len(YTR), features.shape[1]), dtype=np.float32
        )

    RETR["train"][hold] = features
    print("Retrieval fold", fold, "completed")

tfidf = TfidfVectorizer(
    analyzer="char", ngram_range=(3, 5),
    min_df=2, max_features=100000,
    sublinear_tf=True, dtype=np.float32
)

train_char = tfidf.fit_transform(DATA["train"]["text_norm"])
val_char = tfidf.transform(DATA["val"]["text_norm"])

RETR["val"] = retrieve(
    EMBED["val"], EMBED["train"],
    val_char, train_char,
    GVA, GTR, YTR
)

assert np.isfinite(RETR["train"]).all()
assert np.isfinite(RETR["val"]).all()

print("Retrieval feature dimension:", RETR["train"].shape[1])

# ============================================================
# STEP 161 — CER-NET ARCHITECTURE
# ============================================================

section("STEP 161 — CER-NET AND MATCHED CONTROLS")

D = EMBED["train"].shape[1]
R = RETR["train"].shape[1]
H = 128

class CERNet(nn.Module):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        self.semantic = nn.Sequential(
            nn.Linear(D, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.evidence = nn.Sequential(
            nn.Linear(R, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.interaction = nn.Sequential(
            nn.Linear(4 * H, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.semantic_head = nn.Linear(H, 3)
        self.evidence_head = nn.Linear(H, 3)
        self.correction = nn.Linear(H, 3)

        self.neutral_head = nn.Linear(H, 1)

        self.gate = nn.Sequential(
            nn.Linear(3 * H + 3, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, x, r, base):

        s = self.semantic(x)
        e = self.evidence(r)

        if self.mode == "matched_control":
            e = self.evidence(x[:, :R])

        zs = base + self.semantic_head(s)
        ze = self.evidence_head(e)

        p = zs.softmax(-1)

        uncertainty = torch.stack([
            1 - p.max(-1).values,
            -(p * p.clamp_min(1e-8).log()).sum(-1) / math.log(3),
            1 - (p.topk(2).values[:, 0] -
                 p.topk(2).values[:, 1])
        ], dim=1)

        joint = self.interaction(
            torch.cat([s, e, s * e, s - e], dim=1)
        )

        alpha = 0.5 * torch.sigmoid(
            self.gate(
                torch.cat([s, e, joint, uncertainty], dim=1)
            )
        )

        neutral = self.neutral_head(joint).squeeze(1)

        if self.mode == "semantic_only":
            z = zs
        elif self.mode == "retrieval_only":
            z = ze
        elif self.mode == "fixed_fusion":
            z = 0.75 * zs + 0.25 * ze
        else:
            z = zs + alpha * self.correction(joint)

        if self.mode in ["cer_full", "matched_control"]:
            adjustment = torch.stack([
                -0.5 * neutral,
                neutral,
                -0.5 * neutral
            ], dim=1)

            z = z + 0.15 * adjustment

        return z, ze, neutral, alpha

MODES = [
    "semantic_only",
    "retrieval_only",
    "fixed_fusion",
    "adaptive_fusion",
    "matched_control",
    "cer_full"
]

for mode in MODES:
    model = CERNet(mode)
    print(mode, sum(p.numel() for p in model.parameters()))

# ============================================================
# STEP 162 — FAST MULTI-SEED TRAINING
# ============================================================

section("STEP 162 — MULTI-SEED TRAINING")

from torch.utils.data import TensorDataset, DataLoader

def tensor_data(split):
    y = YTR if split == "train" else YVA
    return TensorDataset(
        torch.from_numpy(EMBED[split]),
        torch.from_numpy(RETR[split]),
        torch.from_numpy(BASE_LOGITS[split]),
        torch.from_numpy(y).long()
    )

TRAIN_DS = tensor_data("train")
VAL_DS = tensor_data("val")

def evaluate(model):
    model.eval()
    predictions, gates = [], []

    with torch.no_grad():
        for x, r, b, y in DataLoader(VAL_DS, batch_size=256):
            z, _, _, a = model(
                x.to(DEVICE), r.to(DEVICE), b.to(DEVICE)
            )
            predictions.append(z.cpu().numpy())
            gates.append(a.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(gates)

def metrics(y, p):
    pred = p.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "nll": float(log_loss(y, p, labels=[0, 1, 2]))
    }

RESULTS = []
HISTORY = []
PROBS = {}

CONFIG = {
    "stage": "8B",
    "backbone": MODEL_NAME,
    "backbone_sha256": sha256(weights),
    "seeds": [42, 123, 2026],
    "modes": MODES,
    "epochs": 15,
    "patience": 4,
    "batch_size": 128,
    "learning_rate": 0.0003,
    "retrieval_k": K,
    "test_evaluated": False
}

save_json(CONFIG, BACKUP / "config.json")

try:
    history_cells = get_ipython().user_ns.get("In", [])
    source = next(
        s for s in reversed(history_cells)
        if "STAGE 8B — CONTEXTUAL EVIDENCE RETRIEVAL NETWORK" in s
        and len(s) > 10000
    )
    (BACKUP / "stage8b_complete.py").write_text(source)
except Exception:
    print("Warning: notebook source capture unavailable.")

push_public("Preserve Stage 8B experimental protocol")

for mode in MODES:

    PROBS[mode] = []

    for seed in [42, 123, 2026]:

        seed_all(seed)

        model = CERNet(mode).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.01
        )

        loader = DataLoader(
            TRAIN_DS, batch_size=128, shuffle=True,
            generator=torch.Generator().manual_seed(seed)
        )

        best_f1 = -1
        best_state = None
        best_epoch = 0
        stale = 0

        for epoch in range(1, 16):

            model.train()
            losses = []

            for x, r, b, y in loader:

                x, r, b, y = [
                    t.to(DEVICE) for t in [x, r, b, y]
                ]

                optimizer.zero_grad(set_to_none=True)

                z, ze, neutral, alpha = model(x, r, b)

                loss = F.cross_entropy(z, y)

                if mode in [
                    "fixed_fusion", "adaptive_fusion",
                    "matched_control", "cer_full"
                ]:
                    loss = loss + 0.10 * F.cross_entropy(ze, y)

                if mode in ["cer_full", "matched_control"]:
                    loss = loss + 0.10 * (
                        F.binary_cross_entropy_with_logits(
                            neutral, (y == 1).float()
                        )
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

                losses.append(loss.item())

            logits, gates = evaluate(model)
            m = metrics(YVA, softmax(logits, axis=1))

            HISTORY.append({
                "mode": mode, "seed": seed,
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                **m
            })

            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1

            if stale >= 4:
                break

        model.load_state_dict(best_state)
        logits, gates = evaluate(model)
        probs = softmax(logits, axis=1)

        PROBS[mode].append(probs)

        result = {
            "mode": mode,
            "seed": seed,
            "best_epoch": best_epoch,
            "head_parameters": sum(
                p.numel() for p in model.parameters()
            ),
            **metrics(YVA, probs)
        }

        RESULTS.append(result)

        model_path = MODELS / f"{mode}_seed{seed}_model.pt"
        pred_path = PRED / f"{mode}_seed{seed}_val.npz"

        torch.save(best_state, model_path)

        np.savez_compressed(
            pred_path, sample_id=IDS, labels=YVA,
            logits=logits, probabilities=probs, gates=gates
        )

        upload_private(model_path)
        upload_private(pred_path)

        pd.DataFrame(RESULTS).to_csv(
            BACKUP / "individual_results.csv", index=False
        )

        pd.DataFrame(HISTORY).to_csv(
            BACKUP / "training_history.csv", index=False
        )

        push_public(f"Stage 8B {mode} seed {seed} completed")

        print(result)

        del model, optimizer, best_state
        gc.collect()

# ============================================================
# STEP 163 — PERFORMANCE AND SUBGROUP ANALYSIS
# ============================================================

section("STEP 163 — PERFORMANCE COMPARISONS")

def save_table(rows, name):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)

    for root in [TABLES, BACKUP / "tables", PUB / "tables"]:
        root.mkdir(parents=True, exist_ok=True)
        df.to_csv(root / f"{name}.csv", index=False)
        (root / f"{name}.tex").write_text(
            df.to_latex(index=False, float_format="%.5f")
        )

    return df

individual = save_table(
    RESULTS, "stage8b_01_individual_results"
)

history = save_table(
    HISTORY, "stage8b_02_training_history"
)

summary = save_table(
    individual.groupby("mode")["macro_f1"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
    .sort_values("mean", ascending=False),
    "stage8b_03_multiseed_summary"
)

ENSEMBLES = {
    mode: np.mean(PROBS[mode], axis=0)
    for mode in MODES
}

ENSEMBLES["frozen_banglabert"] = softmax(
    BASE_LOGITS["val"], axis=1
)

ENSEMBLES["semantic_knn"] = RETR["val"][:, :3]
ENSEMBLES["character_knn"] = RETR["val"][:, 14:17]
ENSEMBLES["hybrid_knn"] = RETR["val"][:, 28:31]

for name, p in ENSEMBLES.items():
    p = np.clip(p, 1e-8, None)
    ENSEMBLES[name] = p / p.sum(1, keepdims=True)

comparison = save_table(
    sorted(
        [{"model": name, **metrics(YVA, p)}
         for name, p in ENSEMBLES.items()],
        key=lambda r: r["macro_f1"], reverse=True
    ),
    "stage8b_04_ensemble_comparison"
)

print(comparison.to_string(index=False))

print("\nHistorical prototype family F1: 0.8287035909704495")

neg = ["না", "নয়", "নয়", "নেই", "নাই"]
contrast = ["কিন্তু", "তবে", "যদিও", "অথচ", "তবুও", "তথাপি"]

def cue_mask(words):
    pattern = (
        r"(?<![\w\u0980-\u09ff])(?:"
        + "|".join(map(re.escape, words))
        + r")(?![\w\u0980-\u09ff])"
    )
    return DATA["val"]["text_norm"].str.contains(
        pattern, regex=True
    ).to_numpy()

nmask = cue_mask(neg)
cmask = cue_mask(contrast)

longmask = (
    DATA["val"]["word_count"].to_numpy() >
    DATA["train"]["word_count"].quantile(0.75)
)

MASKS = {
    "all": np.ones(len(YVA), dtype=bool),
    "negation": nmask,
    "contrast": cmask,
    "both_operators": nmask & cmask,
    "any_operator": nmask | cmask,
    "no_operator": ~(nmask | cmask),
    "long_text": longmask,
    "short_text": ~longmask
}

subgroups, classwise = [], []

for name, p in ENSEMBLES.items():

    pred = p.argmax(1)

    for group, mask in MASKS.items():
        if mask.sum():
            subgroups.append({
                "model": name,
                "subgroup": group,
                "n": int(mask.sum()),
                "macro_f1": f1_score(
                    YVA[mask], pred[mask],
                    labels=[0, 1, 2],
                    average="macro", zero_division=0
                ),
                "accuracy": accuracy_score(
                    YVA[mask], pred[mask]
                )
            })

    report = classification_report(
        YVA, pred, labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True, zero_division=0
    )

    for label in LABELS:
        classwise.append({
            "model": name, "class": label,
            **report[label]
        })

subgroups = save_table(
    subgroups, "stage8b_05_subgroups"
)

classwise = save_table(
    classwise, "stage8b_06_classwise"
)

print("\nCLASSWISE F1")
print(classwise.pivot(
    index="model", columns="class", values="f1-score"
).round(4))

# ============================================================
# STEP 164 — PAIRED STATISTICAL ANALYSIS
# ============================================================

section("STEP 164 — GROUP BOOTSTRAP AND MCNEMAR")

rng = np.random.default_rng(42)

unique_groups = np.unique(GVA)
group_rows = [
    np.flatnonzero(GVA == g) for g in unique_groups
]

bootstrap_indices = [
    np.concatenate([
        group_rows[i] for i in rng.integers(
            0, len(group_rows), len(group_rows)
        )
    ])
    for _ in range(2000)
]

pairs = [
    ("cer_full", "frozen_banglabert"),
    ("cer_full", "semantic_only"),
    ("cer_full", "matched_control"),
    ("cer_full", "adaptive_fusion"),
    ("cer_full", "fixed_fusion"),
    ("adaptive_fusion", "semantic_only"),
    ("fixed_fusion", "semantic_only")
]

stats = []

for candidate, reference in pairs:

    a = ENSEMBLES[candidate].argmax(1)
    b = ENSEMBLES[reference].argmax(1)

    delta = (
        f1_score(YVA, a, average="macro") -
        f1_score(YVA, b, average="macro")
    )

    differences = [
        f1_score(YVA[idx], a[idx], average="macro") -
        f1_score(YVA[idx], b[idx], average="macro")
        for idx in bootstrap_indices
    ]

    gained = int(((a == YVA) & (b != YVA)).sum())
    lost = int(((a != YVA) & (b == YVA)).sum())

    p = (
        binomtest(gained, gained + lost, 0.5).pvalue
        if gained + lost else 1.0
    )

    stats.append({
        "candidate": candidate,
        "reference": reference,
        "delta_macro_f1": delta,
        "ci95_low": np.quantile(differences, 0.025),
        "ci95_high": np.quantile(differences, 0.975),
        "gained_correct": gained,
        "lost_correct": lost,
        "mcnemar_p": p
    })

stats = pd.DataFrame(stats)

order = np.argsort(stats["mcnemar_p"].to_numpy())
adjusted = np.empty(len(stats))
running = 0

for rank, i in enumerate(order):
    running = max(
        running,
        min(1.0, (len(stats) - rank) * stats.loc[i, "mcnemar_p"])
    )
    adjusted[i] = running

stats["holm_p"] = adjusted

stats = save_table(stats, "stage8b_07_paired_statistics")

print(stats.to_string(index=False))

# ============================================================
# STEP 165 — PUBLICATION FIGURES
# ============================================================

section("STEP 165 — PUBLICATION FIGURES AND BACKUP")

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120
})

CAPTIONS = []

def save_figure(fig, name, caption):

    for root in [FIGURES, PUB / "figures", BACKUP / "figures"]:
        for ext in ["png", "pdf", "svg"]:
            dest = root / ext
            dest.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                dest / f"{name}.{ext}",
                dpi=600 if ext == "png" else None,
                bbox_inches="tight"
            )

    CAPTIONS.append({"figure": name, "caption": caption})

    pd.DataFrame(CAPTIONS).to_csv(
        BACKUP / "figure_captions.csv", index=False
    )

    push_public(f"Preserve publication figure {name}")
    plt.show()
    plt.close(fig)

# Figure 1 — Overall performance
fig, ax = plt.subplots(figsize=(9, 6))

df = comparison.sort_values("macro_f1")
ax.barh(df["model"], df["macro_f1"], color="#52789B")

ax.axvline(
    0.8287035909704495,
    linestyle="--", color="#A33F3F",
    label="Previous prototype family"
)

ax.set_xlabel("Validation macro-F1")
ax.legend()
fig.tight_layout()

save_figure(
    fig, "stage8b_01_model_comparison",
    "Validation macro-F1 for CER-Net variants and retrieval controls."
)

# Figure 2 — Multi-seed performance
fig, ax = plt.subplots(figsize=(9, 5))

df = summary.sort_values("mean")

ax.barh(
    df["mode"], df["mean"],
    xerr=df["std"], color="#52789B", capsize=3
)

ax.set_xlabel("Mean macro-F1 ± seed standard deviation")
fig.tight_layout()

save_figure(
    fig, "stage8b_02_multiseed",
    "Mean and standard deviation across three fusion-head seeds."
)

# Figure 3 — Classwise performance
fig, ax = plt.subplots(figsize=(11, 5))

classwise.pivot(
    index="model", columns="class", values="f1-score"
).plot.bar(ax=ax)

ax.set_ylabel("Class F1")
ax.legend(title="Sentiment")
fig.tight_layout()

save_figure(
    fig, "stage8b_03_classwise",
    "Class-specific performance, including neutral sentiment."
)

# Figure 4 — Linguistic subgroup heatmap
fig, ax = plt.subplots(figsize=(11, 7))

matrix = subgroups.pivot(
    index="model", columns="subgroup", values="macro_f1"
)

im = ax.imshow(
    matrix.to_numpy(), cmap="viridis",
    aspect="auto", vmin=0, vmax=1
)

ax.set_xticks(range(len(matrix.columns)))
ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
ax.set_yticks(range(len(matrix.index)))
ax.set_yticklabels(matrix.index)

for i in range(len(matrix.index)):
    for j in range(len(matrix.columns)):
        v = matrix.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=8, color="white" if v < 0.6 else "black")

fig.colorbar(im, ax=ax, label="Macro-F1")
fig.tight_layout()

save_figure(
    fig, "stage8b_04_subgroup_heatmap",
    "Performance across linguistic difficulty subgroups."
)

# Figure 5 — Paired effects
fig, ax = plt.subplots(figsize=(9, 6))

for i, row in stats.iterrows():
    ax.plot(
        [row["ci95_low"], row["ci95_high"]],
        [i, i], color="#52789B"
    )
    ax.scatter(row["delta_macro_f1"], i, color="#24476A")

ax.axvline(0, linestyle="--", color="gray")
ax.set_yticks(range(len(stats)))
ax.set_yticklabels(
    stats["candidate"] + " vs " + stats["reference"]
)
ax.set_xlabel("Macro-F1 difference with 95% bootstrap interval")
fig.tight_layout()

save_figure(
    fig, "stage8b_05_paired_effects",
    "Group-bootstrap confidence intervals for paired model differences."
)

# Figure 6 — Training curves
fig, ax = plt.subplots(figsize=(9, 5))

for mode in MODES:
    h = history[
        (history["mode"] == mode) & (history["seed"] == 42)
    ]
    ax.plot(h["epoch"], h["macro_f1"], marker="o", label=mode)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation macro-F1")
ax.legend(fontsize=8)
fig.tight_layout()

save_figure(
    fig, "stage8b_06_training_curves",
    "Validation learning curves for fusion-head screening."
)

# Figure 7 — Confusion matrix
best_name = comparison.iloc[0]["model"]

cm = confusion_matrix(
    YVA, ENSEMBLES[best_name].argmax(1),
    labels=[0, 1, 2]
)

fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3), LABELS)
ax.set_yticks(range(3), LABELS)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")

for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

fig.colorbar(im, ax=ax)
fig.tight_layout()

save_figure(
    fig, "stage8b_07_confusion_matrix",
    f"Confusion matrix of the highest-scoring Stage 8B model: {best_name}."
)

# Figure 8 — Architecture diagram
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

def block(x, y, w, h, text, color):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="#334155",
        linewidth=1.1
    )
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center")

def arrow(x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3)
    )

block(4.5, 7, 3, .6, "Bangla input", "#E2E8F0")
block(.5, 5.6, 4.5, .8, "Frozen BanglaBERT\nSemantic representation", "#DBEAFE")
block(7, 5.6, 4.5, .8, "Training-only memory\nSemantic + character retrieval", "#CCFBF1")
block(.5, 4, 4.5, .8, "Semantic projection\nBase sentiment logits", "#DBEAFE")
block(7, 4, 4.5, .8, "Evidence projection\nNeighbor sentiment statistics", "#CCFBF1")
block(3.5, 2.5, 5, .9, "Semantic–evidence interaction\nAdaptive uncertainty gate", "#EDE9FE")
block(3.5, 1.1, 5, .7, "Residual correction + neutrality head", "#FCE7F3")
block(4, .1, 4, .5, "Negative / Neutral / Positive", "#E2E8F0")

arrow(6, 7, 2.75, 6.4)
arrow(6, 7, 9.25, 6.4)
arrow(2.75, 5.6, 2.75, 4.8)
arrow(9.25, 5.6, 9.25, 4.8)
arrow(2.75, 4, 5, 3.4)
arrow(9.25, 4, 7, 3.4)
arrow(6, 2.5, 6, 1.8)
arrow(6, 1.1, 6, .6)

save_figure(
    fig, "stage8b_08_cernet_architecture",
    "Provisional CER-Net frozen-backbone retrieval-fusion architecture."
)

# ============================================================
# STEP 166 — PERMANENT ARTIFACT BACKUP
# ============================================================

section("STEP 166 — FINAL RESEARCH SYNTHESIS")

for name, p in ENSEMBLES.items():

    path = PRED / f"{name}_ensemble_val.npz"

    np.savez_compressed(
        path, sample_id=IDS, labels=YVA,
        probabilities=p
    )

    upload_private(path)

cer_score = metrics(YVA, ENSEMBLES["cer_full"])["macro_f1"]
control_score = metrics(YVA, ENSEMBLES["matched_control"])["macro_f1"]

previous_best = 0.8287035909704495

REPORT = {
    "stage": "8B",
    "architecture": "CER-Net",
    "training_samples": len(YTR),
    "validation_samples": len(YVA),
    "completed_runs": len(RESULTS),
    "head_seeds": [42, 123, 2026],
    "frozen_backbone_seeds": [42],
    "best_stage8b_model": best_name,
    "best_stage8b_macro_f1": float(comparison.iloc[0]["macro_f1"]),
    "cer_full_macro_f1": cer_score,
    "matched_control_macro_f1": control_score,
    "historical_prototype_family_macro_f1": previous_best,
    "cer_delta_vs_previous_best": cer_score - previous_best,
    "cer_delta_vs_matched_control": cer_score - control_score,
    "new_figures": len(CAPTIONS),
    "test_evaluated": False,
    "external_evaluated": False,
    "novelty_verified": False,
    "limitations": [
        "Frozen encoder trained on complete training cohort.",
        "Training representations are in-sample despite group-disjoint retrieval.",
        "Training retrieval pools are smaller than validation retrieval pool.",
        "Three seeds vary fusion heads, not backbone training.",
        "Repeated exploratory use of the same validation set.",
        "Matched control uses a semantic feature slice instead of retrieval evidence.",
        "Historical prototype family comparison has a different inference budget.",
        "Independent evaluation and novelty verification remain necessary."
    ],
    "decision": (
        "PROMISING — REQUIRE INDEPENDENT CONFIRMATION"
        if cer_score >= previous_best + 0.01
        and cer_score > control_score
        else "DO NOT FINALIZE WITHOUT COMPARATIVE REVIEW"
    )
}

save_json(REPORT, AUDIT / "stage8b_summary.json")
save_json(REPORT, BACKUP / "stage8b_summary.json")

pd.DataFrame(CAPTIONS).to_csv(
    PUB / "stage8b_figure_captions.csv", index=False
)

figure_manifest = []

for p in sorted(PUB.rglob("*")):
    if p.is_file() and p.suffix.lower() in [".png", ".pdf", ".svg"]:
        figure_manifest.append({
            "path": str(p.relative_to(PUB)),
            "sha256": sha256(p),
            "bytes": p.stat().st_size
        })

pd.DataFrame(figure_manifest).to_csv(
    PUB / "figure_manifest.csv", index=False
)

shutil.copy2(
    PUB / "figure_manifest.csv",
    BACKUP / "figure_manifest.csv"
)

def archive_directory(root, destination):
    with zipfile.ZipFile(
        destination, "w", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
    return destination

compact = OUT / "stage8b_compact.zip"

with zipfile.ZipFile(
    compact, "w", zipfile.ZIP_DEFLATED
) as z:
    for root in [TABLES, AUDIT, MODELS, PRED, BACKUP]:
        for p in root.rglob("*"):
            if p.is_file():
                z.write(p, f"{root.name}/{p.relative_to(root)}")

with zipfile.ZipFile(compact) as z:
    assert z.testzip() is None

publication = archive_directory(
    PUB, OUT / "stage8b_publication.zip"
)

upload_private(compact)
upload_private(publication)

save_json(
    REMOTE_MANIFEST, AUDIT / "stage8b_remote_manifest.json"
)

shutil.copy2(
    AUDIT / "stage8b_remote_manifest.json",
    BACKUP / "stage8b_remote_manifest.json"
)

upload_private(AUDIT / "stage8b_remote_manifest.json")

FINAL_COMMIT = push_public(
    "Complete Stage 8B CER-Net screening and verified backups"
)

section("FINAL STAGE 8B RESEARCH REPORT")

print("\nINDIVIDUAL RESULTS")
print(individual.to_string(index=False))

print("\nMULTI-SEED SUMMARY")
print(summary.to_string(index=False))

print("\nENSEMBLE COMPARISON")
print(comparison.to_string(index=False))

print("\nCLASSWISE PERFORMANCE")
print(classwise.pivot(
    index="model", columns="class", values="f1-score"
).round(5))

print("\nLINGUISTIC SUBGROUPS")
print(subgroups.pivot(
    index="model", columns="subgroup", values="macro_f1"
).round(5))

print("\nPAIRED STATISTICS")
print(stats.to_string(index=False))

print("\nRESEARCH SUMMARY")
print(json.dumps(REPORT, indent=2))

print("\nPERMANENT PUBLICATION ASSETS")

for ext in ["png", "pdf", "svg"]:
    print(ext.upper(), "figures:", len(list(PUB.rglob(f"*.{ext}"))))

print("Remote verified assets:", len(REMOTE_MANIFEST))
print("Final public commit:", FINAL_COMMIT)
print("Private release:", RELEASE["html_url"])
print("GitHub synchronized: True")
print("Manual saving required: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("Test evaluated: NO")
print("Runtime minutes:", round((time.time()-START)/60, 2))
print("\nRESEARCH DECISION:", REPORT["decision"])
# ============================================================
# STAGE 8C — RETRIEVAL-AUGMENTED PROTOTYPE NETWORK
# RAP-NET — STEPS 167–177
# ============================================================

import os
import sys
import gc
import re
import json
import math
import time
import random
import hashlib
import shutil
import zipfile
import subprocess
import getpass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import TensorDataset, DataLoader

START8C = time.time()

WORK = Path("/kaggle/working")
OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"

PREVIOUS_COMMIT = (
    "c633b522153bd1a8884028f4d86335307fcd894b"
)

PREVIOUS_TAG = "stage8b-cernet-screening-v1"
STAGE_TAG = "stage8c-rapnet-final-screening-v1"

ROOT8C = WORK / "sentiment_research_stage8c"
ROOT8C.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [42, 123, 2026]
LABELS = ["Negative", "Neutral", "Positive"]

def section8c(text):
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(obj, path):
    Path(path).write_text(
        json.dumps(
            obj, indent=2, ensure_ascii=False, default=str
        )
    )

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# STEP 167 — RECOVER LOCKED FEATURES AND DATA
# ============================================================

section8c("STEP 167 — LOCKED FEATURE RECOVERY")

required = [
    "EMBED", "RETR", "BASE_LOGITS",
    "YTR", "YVA", "GTR", "GVA",
    "IDS", "DATA"
]

available = all(k in globals() for k in required)

if available:
    available = (
        len(YTR) == 9204
        and len(YVA) == 1840
        and EMBED["train"].shape == (9204, 1536)
        and RETR["train"].shape == (9204, 42)
    )

if available:

    print("Reusing Stage 8B features from current session.")

else:

    print("Recovering Stage 8B feature pipeline from GitHub.")

    source_url = (
        f"https://raw.githubusercontent.com/"
        f"{OWNER}/{PUBLIC_REPO}/{PREVIOUS_COMMIT}/"
        "research_stage8b/stage8b_complete.py"
    )

    response = requests.get(source_url, timeout=120)
    response.raise_for_status()

    original_source = response.text

    marker = "# STEP 161 — CER-NET ARCHITECTURE"

    assert marker in original_source, (
        "Stage 8B source structure changed."
    )

    prefix = original_source.split(marker, 1)[0]

    recovery_source = ROOT8C / "stage8b_feature_recovery.py"
    recovery_source.write_text(prefix)

    exec(
        compile(prefix, str(recovery_source), "exec"),
        globals()
    )

assert len(YTR) == 9204
assert len(YVA) == 1840

assert not set(GTR) & set(GVA)

assert EMBED["train"].shape == (9204, 1536)
assert EMBED["val"].shape == (1840, 1536)

assert RETR["train"].shape == (9204, 42)
assert RETR["val"].shape == (1840, 42)

assert np.isfinite(EMBED["train"]).all()
assert np.isfinite(RETR["train"]).all()

baseline_f1 = f1_score(
    YVA,
    BASE_LOGITS["val"].argmax(1),
    average="macro"
)

assert abs(
    baseline_f1 - 0.8159419852657895
) < 0.002

print("Locked training samples:", len(YTR))
print("Locked validation samples:", len(YVA))
print("Baseline macro-F1:", baseline_f1)
print("Test partition used: NO")

# ============================================================
# STEP 168 — GITHUB CONNECTION AND PRIVATE RELEASE
# ============================================================

section8c("STEP 168 — VERIFIED PERMANENT BACKUP CONNECTION")

import base64

if "TOKEN" not in globals() or not TOKEN:
    TOKEN = getpass.getpass("Enter working GitHub token: ")

SESSION8C = requests.Session()

SESSION8C.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

API8C = "https://api.github.com"

PUBLIC_API8C = (
    f"{API8C}/repos/{OWNER}/{PUBLIC_REPO}"
)

PRIVATE_API8C = (
    f"{API8C}/repos/{OWNER}/{PRIVATE_REPO}"
)

def get_json(url):
    r = SESSION8C.get(url, timeout=120)
    r.raise_for_status()
    return r.json()

assert get_json(f"{API8C}/user")["login"]
assert get_json(PRIVATE_API8C)["private"]

REPO8C = WORK / "sentiment_stage8c_github"

auth8c = base64.b64encode(
    f"x-access-token:{TOKEN}".encode()
).decode()

ENV8C = os.environ.copy()

ENV8C.update({
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth8c}",
    "GIT_TERMINAL_PROMPT": "0"
})

def git8c(*args, cwd=REPO8C):

    r = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=ENV8C,
        capture_output=True,
        text=True
    )

    if r.returncode:
        raise RuntimeError("Git failed: " + args[0])

    return r.stdout.strip()

if not (REPO8C / ".git").exists():

    git8c(
        "clone", "--single-branch", "--branch", BRANCH,
        f"https://github.com/{OWNER}/{PUBLIC_REPO}.git",
        str(REPO8C), cwd=WORK
    )

git8c("fetch", "origin", BRANCH)
git8c("checkout", BRANCH)

assert not git8c("status", "--porcelain")

git8c("pull", "--ff-only", "origin", BRANCH)

git8c(
    "merge-base", "--is-ancestor",
    PREVIOUS_COMMIT, "HEAD"
)

git8c("config", "user.name", "Research Backup")

git8c(
    "config", "user.email",
    "research-backup@users.noreply.github.com"
)

BACKUP8C = REPO8C / "research_stage8c"

TABLES8C = ROOT8C / "tables"
MODELS8C = ROOT8C / "models"
PRED8C = ROOT8C / "private_predictions"
AUDIT8C = ROOT8C / "audit"
REC8C = ROOT8C / "recovered"
FIG8C = ROOT8C / "figures"

PUB8C = WORK / "sentiment_publication_assets"

for p in [
    BACKUP8C, TABLES8C, MODELS8C,
    PRED8C, AUDIT8C, REC8C, FIG8C, PUB8C
]:
    p.mkdir(parents=True, exist_ok=True)

def push8c(message):

    git8c("add", "research_stage8c")

    if git8c("status", "--porcelain"):
        git8c("commit", "-m", message)
        git8c("push", "origin", BRANCH)

    local = git8c("rev-parse", "HEAD")

    remote = git8c(
        "ls-remote", "origin",
        f"refs/heads/{BRANCH}"
    ).split()[0]

    assert local == remote

    print("GitHub synchronized:", local)

    return local

def get_release8c(tag):
    return get_json(
        f"{PRIVATE_API8C}/releases/tags/{tag}"
    )

def assets8c(release):

    result = []
    page = 1

    while True:

        rows = get_json(
            f"{PRIVATE_API8C}/releases/"
            f"{release['id']}/assets"
            f"?per_page=100&page={page}"
        )

        result.extend(rows)

        if len(rows) < 100:
            break

        page += 1

    return {a["name"]: a for a in result}

def download8c(asset, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    expected = asset.get("digest") or ""

    if (
        path.exists()
        and path.stat().st_size == asset["size"]
        and expected == "sha256:" + hash_file(path)
    ):
        return path

    with SESSION8C.get(
        asset["url"],
        headers={"Accept": "application/octet-stream"},
        stream=True, timeout=(30, 600)
    ) as r:

        r.raise_for_status()

        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    assert path.stat().st_size == asset["size"]

    if expected:
        assert expected == "sha256:" + hash_file(path)

    return path

def extract8c(path, destination):

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as z:

        for name in z.namelist():
            assert (
                destination / name
            ).resolve().is_relative_to(destination)

        z.extractall(destination)

r = SESSION8C.get(
    f"{PRIVATE_API8C}/releases/tags/{STAGE_TAG}",
    timeout=120
)

if r.status_code == 404:

    r = SESSION8C.post(
        f"{PRIVATE_API8C}/releases",
        json={
            "tag_name": STAGE_TAG,
            "name": "Stage 8C RAP-Net final screening",
            "body": "Verified private research artifacts.",
            "draft": False
        },
        timeout=120
    )

r.raise_for_status()

RELEASE8C = r.json()

REMOTE8C = assets8c(RELEASE8C)
MANIFEST8C = {}

def upload8c(path):

    path = Path(path)

    digest = hash_file(path)
    name = path.name

    asset = REMOTE8C.get(name)

    if asset is not None:

        if asset.get("digest") != "sha256:" + digest:

            name = (
                f"{path.stem}_{digest[:12]}"
                f"{path.suffix}"
            )

            asset = REMOTE8C.get(name)

    if asset is None:

        with open(path, "rb") as f:

            r = SESSION8C.post(
                RELEASE8C["upload_url"].split("{")[0],
                params={"name": name},
                headers={
                    "Content-Type": "application/octet-stream"
                },
                data=f,
                timeout=600
            )

        r.raise_for_status()

        asset = r.json()
        REMOTE8C[name] = asset

    assert asset["size"] == path.stat().st_size

    if asset.get("digest"):

        assert asset["digest"] == "sha256:" + digest

    else:

        verify = REC8C / ("verify_" + name)

        download8c(asset, verify)

        assert hash_file(verify) == digest

        verify.unlink()

    MANIFEST8C[name] = {
        "name": name,
        "sha256": digest,
        "size": asset["size"],
        "asset_id": asset["id"],
        "remote_verified": True
    }

    print("Remote verified:", name)

    return name

print("GitHub authentication: PASS")

# ============================================================
# STEP 169 — RECOVER PREVIOUS BEST PREDICTIONS
# ============================================================

section8c("STEP 169 — HISTORICAL MODEL COMPLEMENTARITY")

previous8b = assets8c(
    get_release8c(PREVIOUS_TAG)
)

def load_aligned_npz(path):

    with np.load(path, allow_pickle=False) as z:

        data = {k: z[k] for k in z.files}

    sample_ids = data["sample_id"].astype(str)

    assert len(set(sample_ids)) == len(sample_ids)
    assert set(sample_ids) == set(IDS.astype(str))

    lookup = {
        sid: i for i, sid in enumerate(sample_ids)
    }

    order = np.array([
        lookup[sid] for sid in IDS.astype(str)
    ])

    assert np.array_equal(
        data["labels"][order], YVA
    )

    if "probabilities" in data:
        p = data["probabilities"][order]
    else:
        p = softmax(data["logits"][order], axis=1)

    assert p.shape == (len(YVA), 3)
    assert np.isfinite(p).all()

    p = np.clip(p, 1e-8, None)
    return p / p.sum(1, keepdims=True)

HISTORICAL8C = {
    "frozen_banglabert": softmax(
        BASE_LOGITS["val"], axis=1
    )
}

for name in [
    "retrieval_only",
    "hybrid_knn",
    "cer_full",
    "matched_control"
]:

    filename = f"{name}_ensemble_val.npz"

    path = download8c(
        previous8b[filename],
        REC8C / filename
    )

    HISTORICAL8C["stage8b_" + name] = (
        load_aligned_npz(path)
    )

# Recover the exact previous prototype-family ensemble.
# Discover archive names rather than guessing hashes.

prototype_recovered = False

try:

    assets7c = assets8c(
        get_release8c(
            "stage7c-specialist-discovery-v1"
        )
    )

    candidates = [
        a for name, a in assets7c.items()
        if "private" in name and name.endswith(".zip")
    ]

    assert candidates

    archive = download8c(
        candidates[0],
        REC8C / candidates[0]["name"]
    )

    extract8c(archive, REC8C / "stage7c")

    for path in (REC8C / "stage7c").rglob("*.npz"):

        with np.load(path, allow_pickle=False) as z:

            keys = z.files

            if (
                "sample_id" not in keys
                or "labels" not in keys
            ):
                continue

            candidate_keys = [
                k for k in keys
                if "prototype" in k.lower()
                and "family" in k.lower()
            ]

            for key in candidate_keys:

                p = z[key]

                if p.shape != (len(YVA), 3):
                    continue

                sample_ids = z["sample_id"].astype(str)

                if set(sample_ids) != set(IDS.astype(str)):
                    continue

                lookup = {
                    sid: i for i, sid in enumerate(sample_ids)
                }

                order = np.array([
                    lookup[sid] for sid in IDS.astype(str)
                ])

                assert np.array_equal(
                    z["labels"][order], YVA
                )

                p = p[order]

                if (
                    p.min() < 0
                    or not np.allclose(
                        p.sum(1), 1, atol=1e-4
                    )
                ):
                    p = softmax(p, axis=1)

                score = f1_score(
                    YVA, p.argmax(1), average="macro"
                )

                if abs(score - 0.8287035909704495) < 0.0002:

                    HISTORICAL8C[
                        "historical_prototype_family"
                    ] = p

                    prototype_recovered = True

                    break

        if prototype_recovered:
            break

except Exception as e:

    print(
        "Prototype-family recovery warning:",
        type(e).__name__, str(e)[:200]
    )

print("Exact prototype family recovered:", prototype_recovered)

def metrics8c(y, p):

    p = np.clip(p, 1e-8, None)
    p = p / p.sum(1, keepdims=True)

    return {
        "accuracy": float(
            accuracy_score(y, p.argmax(1))
        ),
        "macro_f1": float(
            f1_score(y, p.argmax(1), average="macro")
        ),
        "nll": float(
            log_loss(y, p, labels=[0, 1, 2])
        )
    }

complementarity = []

names = list(HISTORICAL8C)

for i, a in enumerate(names):

    for b in names[i+1:]:

        pa = HISTORICAL8C[a]
        pb = HISTORICAL8C[b]

        ca = pa.argmax(1) == YVA
        cb = pb.argmax(1) == YVA

        avg = (pa + pb) / 2

        complementarity.append({
            "model_a": a,
            "model_b": b,
            "both_correct": int((ca & cb).sum()),
            "only_a_correct": int((ca & ~cb).sum()),
            "only_b_correct": int((~ca & cb).sum()),
            "both_wrong": int((~ca & ~cb).sum()),
            "oracle_accuracy": float((ca | cb).mean()),
            "equal_average_macro_f1": metrics8c(
                YVA, avg
            )["macro_f1"]
        })

COMPLEMENT8C = pd.DataFrame(complementarity)

print(
    COMPLEMENT8C.sort_values(
        "equal_average_macro_f1",
        ascending=False
    ).to_string(index=False)
)

# ============================================================
# STEP 170 — RAP-NET ARCHITECTURE
# ============================================================

section8c("STEP 170 — RETRIEVAL-AUGMENTED PROTOTYPE NETWORK")

D8C = 1536
R8C = 42
H8C = 128
P8C = 3

class RAPNet(nn.Module):

    def __init__(self, mode):

        super().__init__()

        self.mode = mode

        self.semantic_projection = nn.Sequential(
            nn.Linear(D8C, H8C),
            nn.LayerNorm(H8C),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.semantic_head = nn.Linear(H8C, 3)

        self.metric_projection = nn.Sequential(
            nn.Linear(H8C, H8C),
            nn.GELU(),
            nn.Linear(H8C, H8C)
        )

        self.prototypes = nn.Parameter(
            torch.randn(3, P8C, H8C) * 0.02
        )

        self.prototype_scale = nn.Parameter(
            torch.tensor(2.0)
        )

        self.generic_metric = nn.Linear(H8C, 3)

        self.retrieval_projection = nn.Sequential(
            nn.Linear(R8C, H8C),
            nn.LayerNorm(H8C),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.retrieval_head = nn.Linear(H8C, 3)

        self.interaction = nn.Sequential(
            nn.Linear(4 * H8C, H8C),
            nn.LayerNorm(H8C),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        # 9 branch probabilities
        # 3 entropy values
        # 3 confidence margins
        # 3 pairwise disagreements
        # 9 retrieval class-vote features

        self.router = nn.Sequential(
            nn.Linear(H8C + 27, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 9)
        )

        self.neutrality = nn.Linear(H8C, 1)

        nn.init.zeros_(self.semantic_head.weight)
        nn.init.zeros_(self.semantic_head.bias)

        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        nn.init.zeros_(self.neutrality.weight)
        nn.init.zeros_(self.neutrality.bias)

    def forward(self, x, r, base):

        s = self.semantic_projection(x)

        zs = base + self.semantic_head(s)

        embedding = F.normalize(
            self.metric_projection(s), dim=-1
        )

        prototypes = F.normalize(
            self.prototypes, dim=-1
        )

        similarity = torch.einsum(
            "bd,cpd->bcp",
            embedding, prototypes
        )

        scale = F.softplus(
            self.prototype_scale
        ) + 1.0

        zp = (
            torch.logsumexp(
                scale * similarity, dim=-1
            ) - math.log(P8C)
        )

        if self.mode == "generic_control":
            zp = self.generic_metric(embedding)

        e = self.retrieval_projection(r)
        zr = self.retrieval_head(e)

        joint = self.interaction(
            torch.cat(
                [s, e, s * e, s - e],
                dim=1
            )
        )

        branches = torch.stack(
            [zs, zp, zr], dim=1
        )

        probs = branches.softmax(-1)

        entropy = -(
            probs * probs.clamp_min(1e-8).log()
        ).sum(-1) / math.log(3)

        top = probs.topk(2, dim=-1).values
        margins = top[:, :, 0] - top[:, :, 1]

        disagreements = torch.stack([
            (probs[:, 0] - probs[:, 1]).abs().mean(-1),
            (probs[:, 0] - probs[:, 2]).abs().mean(-1),
            (probs[:, 1] - probs[:, 2]).abs().mean(-1)
        ], dim=1)

        votes = torch.cat([
            r[:, 0:3],
            r[:, 14:17],
            r[:, 28:31]
        ], dim=1)

        router_input = torch.cat([
            joint,
            probs.detach().flatten(1),
            entropy.detach(),
            margins.detach(),
            disagreements.detach(),
            votes
        ], dim=1)

        weights = self.router(
            router_input
        ).reshape(-1, 3, 3)

        # Softmax across experts separately for each class.
        weights = 0.85 * weights.softmax(dim=1) + 0.05

        if self.mode == "fixed_fusion":

            weights = torch.tensor(
                [0.4, 0.3, 0.3],
                device=x.device
            ).view(1, 3, 1).expand(
                len(x), 3, 3
            )

        if self.mode == "global_router":

            shared = weights.mean(
                dim=2, keepdim=True
            )

            weights = shared.expand(-1, -1, 3)

        # Center branch logits before class-specific mixing.
        # This removes arbitrary branch-level logit offsets.

        centered = branches - branches.mean(
            dim=-1, keepdim=True
        )

        z = (weights * centered).sum(1)

        neutral = self.neutrality(joint).squeeze(-1)

        if self.mode == "prototype_only":
            z = zp

        elif self.mode == "retrieval_only":
            z = zr

        elif self.mode == "semantic_only":
            z = zs

        elif self.mode in [
            "rap_full", "generic_control"
        ]:

            correction = torch.stack([
                -0.5 * neutral,
                neutral,
                -0.5 * neutral
            ], dim=1)

            z = z + 0.15 * correction

        return {
            "logits": z,
            "branches": branches,
            "weights": weights,
            "neutrality": neutral,
            "embedding": embedding
        }

MODES8C = [
    "semantic_only",
    "prototype_only",
    "retrieval_only",
    "fixed_fusion",
    "global_router",
    "class_router",
    "generic_control",
    "rap_full"
]

def objective8c(output, y, mode):

    loss = F.cross_entropy(
        output["logits"], y
    )

    if mode in [
        "semantic_only",
        "prototype_only",
        "retrieval_only"
    ]:
        return loss

    branches = output["branches"]

    auxiliary = sum(
        F.cross_entropy(branches[:, j], y)
        for j in range(3)
    ) / 3

    loss = loss + 0.15 * auxiliary

    if mode in ["rap_full", "generic_control"]:

        loss = loss + 0.10 * (
            F.binary_cross_entropy_with_logits(
                output["neutrality"],
                (y == 1).float()
            )
        )

    return loss

# ============================================================
# STEP 171 — PARAMETER AND GRADIENT AUDIT
# ============================================================

section8c("STEP 171 — MODEL INTEGRITY CHECKS")

PARAMETER_AUDIT8C = []

for mode in MODES8C:

    seed_everything(42)

    model = RAPNet(mode).to(DEVICE)

    x = torch.from_numpy(
        EMBED["train"][:32]
    ).to(DEVICE)

    r = torch.from_numpy(
        RETR["train"][:32]
    ).to(DEVICE)

    b = torch.from_numpy(
        BASE_LOGITS["train"][:32]
    ).to(DEVICE)

    y = torch.from_numpy(
        YTR[:32]
    ).long().to(DEVICE)

    output = model(x, r, b)

    loss = objective8c(output, y, mode)
    loss.backward()

    assert torch.isfinite(loss)

    active = sum(
        p.numel() for p in model.parameters()
        if p.grad is not None
    )

    total = sum(
        p.numel() for p in model.parameters()
    )

    assert all(
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    )

    PARAMETER_AUDIT8C.append({
        "mode": mode,
        "allocated_parameters": total,
        "active_parameters": active,
        "finite_gradients": True
    })

    print(
        mode,
        "allocated:", total,
        "active:", active,
        "PASS"
    )

    del model, output, loss

gc.collect()

# ============================================================
# STEP 172 — SOURCE AND CONFIGURATION LOCK
# ============================================================

section8c("STEP 172 — REPRODUCIBILITY LOCK")

CONFIG8C = {
    "architecture": "RAP-Net",
    "previous_commit": PREVIOUS_COMMIT,
    "modes": MODES8C,
    "seeds": SEEDS,
    "backbone": "csebuetnlp/banglabert",
    "backbone_frozen": True,
    "hidden_dimension": H8C,
    "prototypes_per_class": P8C,
    "retrieval_dimension": R8C,
    "epochs": 15,
    "patience": 4,
    "batch_size": 128,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "train_samples": len(YTR),
    "validation_samples": len(YVA),
    "test_evaluated": False
}

try:

    cells = get_ipython().user_ns.get("In", [])

    SOURCE8C = next(
        s for s in reversed(cells)
        if "STAGE 8C — RETRIEVAL-AUGMENTED PROTOTYPE NETWORK" in s
        and len(s) > 10000
    )

except Exception:

    raise RuntimeError(
        "Cannot capture current source. "
        "Run the complete cell directly in Kaggle."
    )

SOURCE_HASH8C = hashlib.sha256(
    SOURCE8C.encode()
).hexdigest()

CONFIG8C["source_sha256"] = SOURCE_HASH8C

feature_hash = hashlib.sha256()

for split in ["train", "val"]:

    for array in [
        EMBED[split],
        RETR[split],
        BASE_LOGITS[split]
    ]:

        feature_hash.update(
            np.ascontiguousarray(array).tobytes()
        )

CONFIG8C["feature_sha256"] = feature_hash.hexdigest()

(BACKUP8C / "stage8c_complete.py").write_text(
    SOURCE8C
)

write_json(
    CONFIG8C,
    BACKUP8C / "config.json"
)

pd.DataFrame(PARAMETER_AUDIT8C).to_csv(
    BACKUP8C / "parameter_audit.csv",
    index=False
)

push8c("Lock Stage 8C RAP-Net protocol and source")

# ============================================================
# STEP 173 — MULTI-SEED TRAINING WITH AUTO-RESUME
# ============================================================

section8c("STEP 173 — FAST RAP-NET TRAINING")

def dataset8c(split):

    y = YTR if split == "train" else YVA

    return TensorDataset(
        torch.from_numpy(
            EMBED[split].astype(np.float32)
        ),
        torch.from_numpy(
            RETR[split].astype(np.float32)
        ),
        torch.from_numpy(
            BASE_LOGITS[split].astype(np.float32)
        ),
        torch.from_numpy(y).long()
    )

TRAIN8C = dataset8c("train")
VAL8C = dataset8c("val")

def evaluate8c(model):

    model.eval()

    collected = {
        "logits": [],
        "branches": [],
        "weights": []
    }

    with torch.inference_mode():

        for x, r, b, y in DataLoader(
            VAL8C, batch_size=256
        ):

            out = model(
                x.to(DEVICE),
                r.to(DEVICE),
                b.to(DEVICE)
            )

            for key in collected:

                collected[key].append(
                    out[key].cpu().numpy()
                )

    return {
        k: np.concatenate(v)
        for k, v in collected.items()
    }

RESULTS8C = []
HISTORY8C = []
PROBS8C = {}
OUTPUTS8C = {}

for mode in MODES8C:

    PROBS8C[mode] = []
    OUTPUTS8C[mode] = []

    for seed in SEEDS:

        prefix = f"{mode}_seed{seed}"

        run_config = {
            **CONFIG8C,
            "mode": mode,
            "seed": seed
        }

        run_hash = hashlib.sha256(
            json.dumps(
                run_config, sort_keys=True
            ).encode()
        ).hexdigest()

        meta_name = prefix + "_completed.json"

        resumed = False

        # Recover only a completed run with matching
        # source, configuration and feature hashes.

        if meta_name in REMOTE8C:

            meta_path = download8c(
                REMOTE8C[meta_name],
                REC8C / meta_name
            )

            metadata = json.loads(
                meta_path.read_text()
            )

            if metadata.get("run_hash") == run_hash:

                pred_asset = metadata["prediction_asset"]
                model_asset = metadata["model_asset"]

                assert model_asset in REMOTE8C

                pred_path = download8c(
                    REMOTE8C[pred_asset],
                    PRED8C / pred_asset
                )

                assert hash_file(pred_path) == (
                    metadata["prediction_sha256"]
                )

                model_digest = REMOTE8C[
                    model_asset
                ].get("digest")

                if model_digest:

                    assert model_digest == (
                        "sha256:" +
                        metadata["checkpoint_sha256"]
                    )

                else:

                    check = download8c(
                        REMOTE8C[model_asset],
                        REC8C / model_asset
                    )

                    assert hash_file(check) == (
                        metadata["checkpoint_sha256"]
                    )

                    check.unlink()

                with np.load(
                    pred_path, allow_pickle=False
                ) as z:

                    assert np.array_equal(
                        z["sample_id"].astype(str),
                        IDS.astype(str)
                    )

                    assert np.array_equal(
                        z["labels"], YVA
                    )

                    output = {
                        k: z[k]
                        for k in [
                            "logits",
                            "branches",
                            "weights"
                        ]
                    }

                p = softmax(
                    output["logits"], axis=1
                )

                result = metadata["result"]

                assert abs(
                    metrics8c(YVA, p)["macro_f1"]
                    - result["macro_f1"]
                ) < 1e-8

                RESULTS8C.append(result)
                HISTORY8C.extend(metadata["history"])

                PROBS8C[mode].append(p)
                OUTPUTS8C[mode].append(output)

                for asset_name, digest in [
                    (
                        model_asset,
                        metadata["checkpoint_sha256"]
                    ),
                    (
                        pred_asset,
                        metadata["prediction_sha256"]
                    ),
                    (
                        meta_name,
                        hash_file(meta_path)
                    )
                ]:

                    asset = REMOTE8C[asset_name]

                    MANIFEST8C[asset_name] = {
                        "name": asset_name,
                        "sha256": digest,
                        "size": asset["size"],
                        "asset_id": asset["id"],
                        "remote_verified": True
                    }

                resumed = True

                print("Recovered completed run:", prefix)

        if resumed:
            continue

        seed_everything(seed)

        model = RAPNet(mode).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.01
        )

        loader = DataLoader(
            TRAIN8C,
            batch_size=128,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed)
        )

        best = -1
        best_epoch = 0
        best_state = None
        stale = 0

        run_history = []

        for epoch in range(1, 16):

            model.train()
            losses = []

            for x, r, b, y in loader:

                x, r, b, y = [
                    t.to(DEVICE)
                    for t in [x, r, b, y]
                ]

                optimizer.zero_grad(set_to_none=True)

                output = model(x, r, b)

                loss = objective8c(
                    output, y, mode
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                optimizer.step()

                losses.append(loss.item())

            output = evaluate8c(model)

            p = softmax(
                output["logits"], axis=1
            )

            m = metrics8c(YVA, p)

            row = {
                "mode": mode,
                "seed": seed,
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                **m
            }

            run_history.append(row)

            if m["macro_f1"] > best:

                best = m["macro_f1"]
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

        output = evaluate8c(model)

        p = softmax(
            output["logits"], axis=1
        )

        result = {
            "mode": mode,
            "seed": seed,
            "best_epoch": best_epoch,
            **metrics8c(YVA, p)
        }

        RESULTS8C.append(result)
        HISTORY8C.extend(run_history)

        PROBS8C[mode].append(p)
        OUTPUTS8C[mode].append(output)

        model_path = (
            MODELS8C / (prefix + "_model.pt")
        )

        pred_path = (
            PRED8C / (prefix + "_val.npz")
        )

        torch.save(best_state, model_path)

        np.savez_compressed(
            pred_path,
            sample_id=IDS.astype(str),
            labels=YVA,
            **output
        )

        model_asset = upload8c(model_path)
        pred_asset = upload8c(pred_path)

        metadata = {
            "run_hash": run_hash,
            "config": run_config,
            "result": result,
            "history": run_history,
            "model_asset": model_asset,
            "prediction_asset": pred_asset,
            "checkpoint_sha256": hash_file(model_path),
            "prediction_sha256": hash_file(pred_path)
        }

        meta_path = AUDIT8C / meta_name

        write_json(metadata, meta_path)

        upload8c(meta_path)

        pd.DataFrame(RESULTS8C).to_csv(
            BACKUP8C / "individual_results.csv",
            index=False
        )

        pd.DataFrame(HISTORY8C).to_csv(
            BACKUP8C / "training_history.csv",
            index=False
        )

        push8c(
            f"Stage 8C {mode} seed {seed} verified"
        )

        print(result)

        del model, optimizer, best_state

        gc.collect()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# ============================================================
# STEP 174 — FINAL VALIDATION COMPARISONS
# ============================================================

section8c("STEP 174 — PERFORMANCE SYNTHESIS")

def table8c(data, name):

    df = (
        data.copy()
        if isinstance(data, pd.DataFrame)
        else pd.DataFrame(data)
    )

    for root in [
        TABLES8C,
        BACKUP8C / "tables",
        PUB8C / "tables"
    ]:

        root.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            root / f"{name}.csv",
            index=False
        )

        (root / f"{name}.tex").write_text(
            df.to_latex(
                index=False,
                float_format="%.5f"
            )
        )

    return df

individual8c = table8c(
    RESULTS8C, "stage8c_01_individual_results"
)

history8c = table8c(
    HISTORY8C, "stage8c_02_training_history"
)

summary8c = table8c(
    individual8c.groupby("mode")["macro_f1"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
    .sort_values("mean", ascending=False),
    "stage8c_03_multiseed_summary"
)

table8c(
    COMPLEMENT8C,
    "stage8c_04_historical_complementarity"
)

table8c(
    PARAMETER_AUDIT8C,
    "stage8c_05_parameter_audit"
)

ENSEMBLES8C = {
    mode: np.mean(PROBS8C[mode], axis=0)
    for mode in MODES8C
}

ENSEMBLES8C.update(HISTORICAL8C)

# Prespecified simple averaging controls.

ENSEMBLES8C["prototype_retrieval_average"] = (
    ENSEMBLES8C["prototype_only"]
    + ENSEMBLES8C["retrieval_only"]
) / 2

ENSEMBLES8C["three_branch_average"] = (
    ENSEMBLES8C["prototype_only"]
    + ENSEMBLES8C["retrieval_only"]
    + ENSEMBLES8C["semantic_only"]
) / 3

if prototype_recovered:

    ENSEMBLES8C["historical_prototype_plus_retrieval"] = (
        ENSEMBLES8C["historical_prototype_family"]
        + ENSEMBLES8C["stage8b_retrieval_only"]
    ) / 2

    ENSEMBLES8C["historical_prototype_plus_rap"] = (
        ENSEMBLES8C["historical_prototype_family"]
        + ENSEMBLES8C["rap_full"]
    ) / 2

comparison8c = table8c(
    sorted(
        [
            {"model": name, **metrics8c(YVA, p)}
            for name, p in ENSEMBLES8C.items()
        ],
        key=lambda x: x["macro_f1"],
        reverse=True
    ),
    "stage8c_06_ensemble_comparison"
)

print(comparison8c.to_string(index=False))

# ============================================================
# STEP 175 — STATISTICS AND DIFFICULT SUBGROUPS
# ============================================================

section8c("STEP 175 — STATISTICAL CONFIRMATION")

groups = np.unique(GVA)

group_rows = [
    np.flatnonzero(GVA == g)
    for g in groups
]

rng = np.random.default_rng(42)

bootstrap = [
    np.concatenate([
        group_rows[i]
        for i in rng.integers(
            0, len(groups), len(groups)
        )
    ])
    for _ in range(2000)
]

references = [
    "frozen_banglabert",
    "semantic_only",
    "prototype_only",
    "retrieval_only",
    "fixed_fusion",
    "global_router",
    "class_router",
    "generic_control",
    "prototype_retrieval_average",
    "three_branch_average",
    "stage8b_retrieval_only"
]

if prototype_recovered:
    references.append("historical_prototype_family")

stat_rows = []

for reference in references:

    a = ENSEMBLES8C["rap_full"].argmax(1)
    b = ENSEMBLES8C[reference].argmax(1)

    delta = (
        f1_score(YVA, a, average="macro")
        - f1_score(YVA, b, average="macro")
    )

    effects = [
        f1_score(
            YVA[idx], a[idx], average="macro"
        ) - f1_score(
            YVA[idx], b[idx], average="macro"
        )
        for idx in bootstrap
    ]

    gained = int(
        ((a == YVA) & (b != YVA)).sum()
    )

    lost = int(
        ((a != YVA) & (b == YVA)).sum()
    )

    pvalue = (
        binomtest(
            gained, gained + lost, 0.5
        ).pvalue
        if gained + lost else 1.0
    )

    stat_rows.append({
        "candidate": "rap_full",
        "reference": reference,
        "delta_macro_f1": delta,
        "ci95_low": np.quantile(effects, 0.025),
        "ci95_high": np.quantile(effects, 0.975),
        "gained": gained,
        "lost": lost,
        "mcnemar_p": pvalue
    })

stats8c = pd.DataFrame(stat_rows)

order = np.argsort(
    stats8c["mcnemar_p"].to_numpy()
)

adjusted = np.empty(len(order))
running = 0

for rank, idx in enumerate(order):

    running = max(
        running,
        min(
            1.0,
            (len(order) - rank)
            * stats8c.loc[idx, "mcnemar_p"]
        )
    )

    adjusted[idx] = running

stats8c["holm_p"] = adjusted

table8c(
    stats8c, "stage8c_07_paired_statistics"
)

neg_words = ["না", "নয়", "নয়", "নেই", "নাই"]

contrast_words = [
    "কিন্তু", "তবে", "যদিও",
    "অথচ", "তবুও", "তথাপি"
]

def mask8c(words):

    pattern = (
        r"(?<![\w\u0980-\u09ff])(?:"
        + "|".join(map(re.escape, words))
        + r")(?![\w\u0980-\u09ff])"
    )

    return DATA["val"]["text_norm"].str.contains(
        pattern, regex=True
    ).to_numpy()

neg = mask8c(neg_words)
contrast = mask8c(contrast_words)

long_text = (
    DATA["val"]["word_count"].to_numpy()
    > DATA["train"]["word_count"].quantile(0.75)
)

MASKS8C = {
    "all": np.ones(len(YVA), dtype=bool),
    "negation": neg,
    "contrast": contrast,
    "both_operators": neg & contrast,
    "any_operator": neg | contrast,
    "no_operator": ~(neg | contrast),
    "long_text": long_text,
    "short_text": ~long_text
}

class_rows = []
subgroup_rows = []
confusion_rows = []

for name, p in ENSEMBLES8C.items():

    pred = p.argmax(1)

    report = classification_report(
        YVA, pred,
        labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True,
        zero_division=0
    )

    for label in LABELS:

        class_rows.append({
            "model": name,
            "class": label,
            **report[label]
        })

    for subgroup, mask in MASKS8C.items():

        if mask.sum():

            subgroup_rows.append({
                "model": name,
                "subgroup": subgroup,
                "n": int(mask.sum()),
                "macro_f1": f1_score(
                    YVA[mask],
                    pred[mask],
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0
                ),
                "accuracy": accuracy_score(
                    YVA[mask], pred[mask]
                )
            })

    cm = confusion_matrix(
        YVA, pred, labels=[0, 1, 2]
    )

    for i in range(3):

        for j in range(3):

            confusion_rows.append({
                "model": name,
                "true_class": LABELS[i],
                "predicted_class": LABELS[j],
                "count": int(cm[i, j])
            })

classwise8c = table8c(
    class_rows, "stage8c_08_classwise"
)

subgroups8c = table8c(
    subgroup_rows, "stage8c_09_subgroups"
)

table8c(
    confusion_rows, "stage8c_10_confusion"
)

# Analyze expert weights separately by sentiment class.

router_rows = []

for mode in [
    "fixed_fusion",
    "global_router",
    "class_router",
    "generic_control",
    "rap_full"
]:

    weights8c = np.mean([
        out["weights"]
        for out in OUTPUTS8C[mode]
    ], axis=0)

    for c, label in enumerate(LABELS):

        for e, expert in enumerate([
            "semantic", "prototype", "retrieval"
        ]):

            router_rows.append({
                "mode": mode,
                "class": label,
                "expert": expert,
                "mean_weight": float(
                    weights8c[:, e, c].mean()
                ),
                "std_weight": float(
                    weights8c[:, e, c].std()
                )
            })

router8c = table8c(
    router_rows, "stage8c_11_router_weights"
)

print("\nCLASSWISE F1")

print(
    classwise8c.pivot(
        index="model",
        columns="class",
        values="f1-score"
    ).round(5)
)

print("\nPAIRED STATISTICS")
print(stats8c.to_string(index=False))

# ============================================================
# STEP 176 — PUBLICATION FIGURES
# ============================================================

section8c("STEP 176 — PUBLICATION-READY FIGURES")

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120
})

CAPTIONS8C = []

def figure8c(fig, name, caption):

    for root in [
        FIG8C,
        PUB8C / "figures",
        BACKUP8C / "figures"
    ]:

        for ext in ["png", "pdf", "svg"]:

            dest = root / ext
            dest.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                dest / f"{name}.{ext}",
                dpi=600 if ext == "png" else None,
                bbox_inches="tight"
            )

    CAPTIONS8C.append({
        "figure": name,
        "caption": caption
    })

    pd.DataFrame(CAPTIONS8C).to_csv(
        BACKUP8C / "figure_captions.csv",
        index=False
    )

    push8c("Preserve figure " + name)

    plt.show()
    plt.close(fig)

# Figure 1: Overall comparison

fig, ax = plt.subplots(figsize=(10, 8))

df = comparison8c.sort_values("macro_f1")

colors = [
    "#D97706" if x == "rap_full"
    else "#52789B"
    for x in df["model"]
]

ax.barh(
    df["model"], df["macro_f1"],
    color=colors
)

ax.axvline(
    0.8287035909704495,
    linestyle="--",
    color="#A33F3F",
    label="Historical best"
)

ax.set_xlabel("Validation macro-F1")
ax.legend()

fig.tight_layout()

figure8c(
    fig,
    "stage8c_01_overall_comparison",
    "Validation performance of RAP-Net, ablations and historical controls."
)

# Figure 2: Multi-seed stability

fig, ax = plt.subplots(figsize=(9, 5))

df = summary8c.sort_values("mean")

ax.barh(
    df["mode"], df["mean"],
    xerr=df["std"],
    color="#52789B",
    capsize=4
)

ax.set_xlabel("Macro-F1: mean ± seed SD")

fig.tight_layout()

figure8c(
    fig,
    "stage8c_02_multiseed_stability",
    "Performance stability across three fusion-head seeds."
)

# Figure 3: Classwise results

fig, ax = plt.subplots(figsize=(12, 6))

matrix = classwise8c.pivot(
    index="model",
    columns="class",
    values="f1-score"
)

matrix.plot.bar(ax=ax)

ax.set_ylabel("Class F1")
ax.legend(title="Sentiment")

fig.tight_layout()

figure8c(
    fig,
    "stage8c_03_classwise",
    "Class-specific performance with emphasis on neutral sentiment."
)

# Figure 4: Subgroup heatmap

fig, ax = plt.subplots(figsize=(12, 9))

matrix = subgroups8c.pivot(
    index="model",
    columns="subgroup",
    values="macro_f1"
)

im = ax.imshow(
    matrix.to_numpy(),
    cmap="viridis",
    aspect="auto",
    vmin=0,
    vmax=1
)

ax.set_xticks(range(len(matrix.columns)))

ax.set_xticklabels(
    matrix.columns,
    rotation=45,
    ha="right"
)

ax.set_yticks(range(len(matrix.index)))
ax.set_yticklabels(matrix.index)

for i in range(len(matrix.index)):

    for j in range(len(matrix.columns)):

        value = matrix.iloc[i, j]

        ax.text(
            j, i, f"{value:.2f}",
            ha="center", va="center",
            fontsize=8,
            color="white" if value < 0.6 else "black"
        )

fig.colorbar(im, ax=ax, label="Macro-F1")

fig.tight_layout()

figure8c(
    fig,
    "stage8c_04_subgroups",
    "Linguistic subgroup performance across architecture variants."
)

# Figure 5: Statistical comparisons

fig, ax = plt.subplots(figsize=(10, 7))

for i, row in stats8c.iterrows():

    ax.plot(
        [row["ci95_low"], row["ci95_high"]],
        [i, i],
        color="#52789B"
    )

    ax.scatter(
        row["delta_macro_f1"], i,
        color="#24476A"
    )

ax.axvline(0, linestyle="--", color="gray")

ax.set_yticks(range(len(stats8c)))
ax.set_yticklabels(stats8c["reference"])

ax.set_xlabel(
    "RAP-Net macro-F1 difference with 95% bootstrap CI"
)

fig.tight_layout()

figure8c(
    fig,
    "stage8c_05_paired_effects",
    "Paired group-bootstrap intervals for RAP-Net comparisons."
)

# Figure 6: Class-conditional expert weights

fig, ax = plt.subplots(figsize=(7, 5))

matrix = router8c[
    router8c["mode"] == "rap_full"
].pivot(
    index="class",
    columns="expert",
    values="mean_weight"
)

matrix.plot.bar(
    stacked=True, ax=ax
)

ax.set_ylabel("Mean expert weight")
ax.set_ylim(0, 1)

fig.tight_layout()

figure8c(
    fig,
    "stage8c_06_expert_weights",
    "Class-conditional semantic, prototype and retrieval weights."
)

# Figure 7: Historical complementarity

fig, ax = plt.subplots(figsize=(10, 7))

df = COMPLEMENT8C.sort_values(
    "equal_average_macro_f1",
    ascending=False
).head(12).copy()

df["pair"] = (
    df["model_a"] + "\n+ " + df["model_b"]
)

df.set_index("pair")[
    ["only_a_correct", "only_b_correct", "both_wrong"]
].plot.barh(
    stacked=True,
    ax=ax
)

ax.set_xlabel("Validation samples")

fig.tight_layout()

figure8c(
    fig,
    "stage8c_07_complementarity",
    "Complementary errors among historical prototype and retrieval models."
)

# Figure 8: Learning curves

fig, ax = plt.subplots(figsize=(10, 5))

for mode in MODES8C:

    h = history8c[
        (history8c["mode"] == mode)
        & (history8c["seed"] == 42)
    ]

    ax.plot(
        h["epoch"],
        h["macro_f1"],
        marker="o",
        label=mode
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation macro-F1")
ax.legend(fontsize=8)

fig.tight_layout()

figure8c(
    fig,
    "stage8c_08_learning_curves",
    "Validation learning curves for RAP-Net screening."
)

# Figure 9: Confusion matrix

best_name8c = comparison8c.iloc[0]["model"]

cm = confusion_matrix(
    YVA,
    ENSEMBLES8C[best_name8c].argmax(1),
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

        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center"
        )

fig.colorbar(im, ax=ax)

fig.tight_layout()

figure8c(
    fig,
    "stage8c_09_confusion_matrix",
    f"Confusion matrix for {best_name8c}."
)

# Figure 10: RAP-Net architecture

from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 8))

ax.set_xlim(0, 13)
ax.set_ylim(0, 9)
ax.axis("off")

def block8c(x, y, w, h, text, color):

    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.07",
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.2
        )
    )

    ax.text(
        x+w/2, y+h/2, text,
        ha="center", va="center"
    )

def arrow8c(x1, y1, x2, y2):

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "->",
            "color": "#334155",
            "lw": 1.4
        }
    )

block8c(
    4.5, 8.1, 4, .55,
    "Bangla input text",
    "#E2E8F0"
)

block8c(
    4.2, 6.9, 4.6, .75,
    "Frozen BanglaBERT\nCLS + masked mean representation",
    "#DBEAFE"
)

block8c(
    .2, 5.1, 3.8, 1.1,
    "Semantic expert\nResidual sentiment classifier",
    "#DBEAFE"
)

block8c(
    4.6, 5.1, 3.8, 1.1,
    "Prototype expert\nThree prototypes per class",
    "#EDE9FE"
)

block8c(
    9, 5.1, 3.8, 1.1,
    "Retrieval expert\nTraining-only neighbor evidence",
    "#CCFBF1"
)

block8c(
    3.4, 3.3, 6.2, 1.05,
    "Class-conditional evidence router\n"
    "Entropy, margins, disagreement, neighbor votes",
    "#FEF3C7"
)

block8c(
    4, 1.8, 5, .8,
    "Weighted centered logits\nNeutrality correction",
    "#FCE7F3"
)

block8c(
    4.1, .4, 4.8, .65,
    "Negative | Neutral | Positive",
    "#E2E8F0"
)

arrow8c(6.5, 8.1, 6.5, 7.65)

for x in [2.1, 6.5, 10.9]:

    arrow8c(6.5, 6.9, x, 6.2)
    arrow8c(x, 5.1, 6.5, 4.35)

arrow8c(6.5, 3.3, 6.5, 2.6)
arrow8c(6.5, 1.8, 6.5, 1.05)

fig.tight_layout()

figure8c(
    fig,
    "stage8c_10_rapnet_architecture",
    "Provisional retrieval-augmented prototype architecture."
)

# ============================================================
# STEP 177 — FINAL BACKUP AND RESEARCH DECISION
# ============================================================

section8c("STEP 177 — VERIFIED FINAL BACKUP")

for name, p in ENSEMBLES8C.items():

    path = PRED8C / (
        name + "_ensemble_val.npz"
    )

    np.savez_compressed(
        path,
        sample_id=IDS.astype(str),
        labels=YVA,
        probabilities=p
    )

    upload8c(path)

# Preserve the frozen features privately for future
# experiments without repeated transformer inference.

feature_path = PRED8C / "stage8c_locked_features.npz"

np.savez_compressed(
    feature_path,
    train_embeddings=EMBED["train"],
    val_embeddings=EMBED["val"],
    train_retrieval=RETR["train"],
    val_retrieval=RETR["val"],
    train_logits=BASE_LOGITS["train"],
    val_logits=BASE_LOGITS["val"],
    train_labels=YTR,
    val_labels=YVA,
    train_sample_id=DATA["train"][
        "sample_id"
    ].to_numpy(dtype=str),
    val_sample_id=IDS.astype(str),
    train_groups=GTR,
    val_groups=GVA
)

upload8c(feature_path)

# Restore previous publication files if needed.

previous_publication = [
    a for name, a in previous8b.items()
    if name.startswith("stage8b_publication")
    and name.endswith(".zip")
]

if previous_publication:

    p = download8c(
        previous_publication[0],
        REC8C / previous_publication[0]["name"]
    )

    extract8c(
        p, REC8C / "previous_publication"
    )

    for source in (
        REC8C / "previous_publication"
    ).rglob("*"):

        if not source.is_file():
            continue

        rel = source.relative_to(
            REC8C / "previous_publication"
        )

        dest = PUB8C / rel

        if not dest.exists():

            dest.parent.mkdir(
                parents=True, exist_ok=True
            )

            shutil.copy2(source, dest)

# Count unique figure content rather than duplicated paths.

figure_inventory = []

for path in sorted(PUB8C.rglob("*")):

    if (
        path.is_file()
        and path.suffix.lower()
        in [".png", ".pdf", ".svg"]
    ):

        figure_inventory.append({
            "path": str(path.relative_to(PUB8C)),
            "format": path.suffix.lower()[1:],
            "sha256": hash_file(path),
            "bytes": path.stat().st_size
        })

inventory8c = pd.DataFrame(figure_inventory)

inventory8c.to_csv(
    PUB8C / "figure_manifest.csv",
    index=False
)

table8c(
    inventory8c,
    "stage8c_12_figure_inventory"
)

unique_counts = (
    inventory8c.groupby("format")["sha256"]
    .nunique().to_dict()
)

rap_score = metrics8c(
    YVA, ENSEMBLES8C["rap_full"]
)["macro_f1"]

control_score = metrics8c(
    YVA, ENSEMBLES8C["generic_control"]
)["macro_f1"]

historical_best = 0.8287035909704495

rap_mean = float(
    summary8c.loc[
        summary8c["mode"] == "rap_full",
        "mean"
    ].iloc[0]
)

control_mean = float(
    summary8c.loc[
        summary8c["mode"] == "generic_control",
        "mean"
    ].iloc[0]
)

control_stats = stats8c[
    stats8c["reference"] == "generic_control"
].iloc[0]

PROMISING8C = bool(
    rap_score > historical_best
    and rap_score > control_score
    and rap_mean > control_mean
    and control_stats["ci95_low"] > 0
)

REPORT8C = {
    "stage": "8C",
    "architecture": "RAP-Net",
    "completed_runs": len(RESULTS8C),
    "training_samples": len(YTR),
    "validation_samples": len(YVA),
    "historical_prototype_recovered": prototype_recovered,
    "historical_best_macro_f1": historical_best,
    "best_current_model": best_name8c,
    "best_current_macro_f1": float(
        comparison8c.iloc[0]["macro_f1"]
    ),
    "rap_full_macro_f1": rap_score,
    "rap_full_mean_seed_macro_f1": rap_mean,
    "generic_control_macro_f1": control_score,
    "generic_control_mean_seed_macro_f1": control_mean,
    "rap_delta_vs_historical_best": (
        rap_score - historical_best
    ),
    "rap_delta_vs_generic_control": (
        rap_score - control_score
    ),
    "rap_control_ci95": [
        float(control_stats["ci95_low"]),
        float(control_stats["ci95_high"])
    ],
    "new_figures": len(CAPTIONS8C),
    "unique_figure_content_by_format": unique_counts,
    "test_evaluated": False,
    "external_evaluated": False,
    "novelty_verified": False,
    "backbone_frozen": True,
    "limitations": [
        "Repeated use of the same validation partition.",
        "Frozen encoder trained on the full training cohort.",
        "Training retrieval is group-disjoint but embeddings are not fully cross-fitted.",
        "Training and validation retrieval memory sizes differ.",
        "Three seeds vary fusion heads rather than the encoder.",
        "Allocated parameter equality does not guarantee active parameter equality.",
        "Historical ensembles have different inference budgets.",
        "Independent confirmation and novelty verification are required."
    ],
    "decision": (
        "PROMISING: LOCK AND CONFIRM INDEPENDENTLY"
        if PROMISING8C else
        "END ARCHITECTURE SEARCH AND REVIEW PUBLICATION CONTRIBUTION"
    )
}

write_json(
    REPORT8C,
    AUDIT8C / "stage8c_summary.json"
)

write_json(
    REPORT8C,
    BACKUP8C / "stage8c_summary.json"
)

pd.DataFrame(CAPTIONS8C).to_csv(
    PUB8C / "stage8c_figure_captions.csv",
    index=False
)

def archive8c(root, destination):

    with zipfile.ZipFile(
        destination, "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for p in sorted(root.rglob("*")):

            if p.is_file():

                z.write(
                    p, p.relative_to(root)
                )

    with zipfile.ZipFile(destination) as z:

        assert z.testzip() is None

    return destination

compact8c = ROOT8C / "stage8c_compact.zip"

with zipfile.ZipFile(
    compact8c, "w",
    zipfile.ZIP_DEFLATED
) as z:

    for root in [
        TABLES8C,
        AUDIT8C,
        MODELS8C,
        PRED8C,
        BACKUP8C
    ]:

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            # Features are already independently backed up.

            if path.name == feature_path.name:
                continue

            z.write(
                path,
                f"{root.name}/{path.relative_to(root)}"
            )

with zipfile.ZipFile(compact8c) as z:

    assert z.testzip() is None

publication8c = archive8c(
    PUB8C,
    ROOT8C / "stage8c_publication.zip"
)

upload8c(compact8c)
upload8c(publication8c)

manifest8c = AUDIT8C / "stage8c_remote_manifest.json"

write_json(
    list(MANIFEST8C.values()),
    manifest8c
)

shutil.copy2(
    manifest8c,
    BACKUP8C / manifest8c.name
)

upload8c(manifest8c)

FINAL_COMMIT8C = push8c(
    "Complete Stage 8C RAP-Net screening and permanent backups"
)

# ============================================================
# FINAL REPORT
# ============================================================

section8c("FINAL STAGE 8C RESEARCH REPORT")

print("\nINDIVIDUAL RESULTS")
print(individual8c.to_string(index=False))

print("\nMULTI-SEED SUMMARY")
print(summary8c.to_string(index=False))

print("\nENSEMBLE COMPARISON")
print(comparison8c.to_string(index=False))

print("\nCLASSWISE PERFORMANCE")

print(
    classwise8c.pivot(
        index="model",
        columns="class",
        values="f1-score"
    ).round(5)
)

print("\nLINGUISTIC SUBGROUPS")

print(
    subgroups8c.pivot(
        index="model",
        columns="subgroup",
        values="macro_f1"
    ).round(5)
)

print("\nPAIRED STATISTICS")
print(stats8c.to_string(index=False))

print("\nCLASS-CONDITIONAL ROUTING")

print(
    router8c[
        router8c["mode"] == "rap_full"
    ].to_string(index=False)
)

print("\nRESEARCH SUMMARY")
print(json.dumps(REPORT8C, indent=2))

print("\nPERMANENT BACKUP")

print("Verified assets:", len(MANIFEST8C))
print("Final commit:", FINAL_COMMIT8C)
print("Private release:", RELEASE8C["html_url"])
print("Unique figure content:", unique_counts)
print("GitHub synchronized: True")
print("Manual saving required: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("Test evaluated: NO")

print(
    "Runtime minutes:",
    round((time.time() - START8C) / 60, 2)
)

print("\nRESEARCH DECISION:", REPORT8C["decision"])
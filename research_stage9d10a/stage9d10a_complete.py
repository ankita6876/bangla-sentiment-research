# ============================================================
# STAGE 9D–10A — CONSOLIDATED FINAL EVALUATION PIPELINE
# STEPS 193–209
#
# Run the entire block once.
# GPU T4 recommended.
# GitHub backup is automatic.
# ============================================================

import os
import re
import io
import ast
import gc
import json
import math
import time
import hashlib
import zipfile
import getpass
import unicodedata
import subprocess
import sys
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from scipy.special import softmax
from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    log_loss
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression

START_ALL = time.time()

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"

LOCKED_COMMIT = "a84cbf71c40f169fd33722c3de67e235431354a7"

NEW_TAG = "stage9d10a-final-evaluation-v1"

ROOT = WORK / "sentiment_final_evaluation"
TABLES_NEW = ROOT / "tables"
PRED_NEW = ROOT / "private_predictions"
AUDIT_NEW = ROOT / "audit"
FIG_NEW = ROOT / "figures"
REC_NEW = ROOT / "recovered"

for p in [
    ROOT, TABLES_NEW, PRED_NEW,
    AUDIT_NEW, FIG_NEW, REC_NEW
]:
    p.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

assert DEVICE.type == "cuda", (
    "Enable GPU accelerator before running."
)

torch.manual_seed(42)
np.random.seed(42)

LABELS = ["Negative", "Neutral", "Positive"]

LABEL_MAP = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

BOOTSTRAPS = 5000
BATCH_SIZE = 32

def banner(text):
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)

def hash_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()

def dump_json(data, path):

    Path(path).write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str
        )
    )

# ============================================================
# STEP 193 — AUTOMATIC GITHUB RECOVERY
# ============================================================

banner("STEP 193 — AUTOMATIC GITHUB RECOVERY")

TOKEN_NEW = getpass.getpass(
    "Enter your working GitHub token: "
).strip()

# Recover the previously verified GitHub infrastructure.
# Do not execute its evaluation or training sections.

source_url = (
    f"https://raw.githubusercontent.com/"
    f"{OWNER}/{PUBLIC_REPO}/{LOCKED_COMMIT}/"
    "research_stage9c/stage9c_complete.py"
)

response = requests.get(
    source_url,
    headers={
        "Authorization": f"Bearer {TOKEN_NEW}"
    },
    timeout=120
)

response.raise_for_status()

original_9c = response.text

marker = (
    'section("STEP 188 — EXACT VALIDATION RECOVERY")'
)

assert marker in original_9c

prefix = original_9c.split(marker, 1)[0]

# Replace token prompt with the token already entered.

prefix = re.sub(
    r'TOKEN\s*=\s*getpass\.getpass\([\s\S]*?\)\.strip\(\)',
    "TOKEN = TOKEN_NEW",
    prefix,
    count=1
)

# Exclude previous-stage source capture and commit.

capture_marker = "# Preserve complete current source."

assert capture_marker in prefix

prefix = prefix.split(capture_marker, 1)[0]

exec(compile(
    prefix,
    "<verified_stage9c_infrastructure>",
    "exec"
), globals())

# Use new output directory in the synchronized repository.

BACKUP_NEW = REPO / "research_stage9d10a"
BACKUP_NEW.mkdir(exist_ok=True)

def push_new(message):

    git("add", "research_stage9d10a")

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

    print("GitHub synchronized:", local)

    return local

# Preserve this complete cell before processing.

source_candidates = [
    s for s in get_ipython().user_ns.get("In", [])
    if "STAGE 9D–10A — CONSOLIDATED FINAL EVALUATION PIPELINE" in s
    and len(s) > 10000
]

assert source_candidates

SOURCE_NEW = source_candidates[-1]

(
    BACKUP_NEW / "stage9d10a_complete.py"
).write_text(SOURCE_NEW)

protocol_path = (
    REPO / "research_stage9c"
    / "locked_candidate_protocol.json"
)

PROTOCOL_NEW = json.loads(
    protocol_path.read_text()
)

assert PROTOCOL_NEW["candidate_members"] == [
    "base_prototype",
    "base_pasnet",
    "large_pasnet"
]

assert PROTOCOL_NEW[
    "three_member_standard_control"
] == [
    "locked_banglabert",
    "base_standard",
    "large_standard"
]

dump_json(
    PROTOCOL_NEW,
    BACKUP_NEW / "original_locked_protocol.json"
)

push_new(
    "Preserve consolidated final evaluation source"
)

# ============================================================
# PRIVATE RELEASE AND VERIFIED UPLOAD INFRASTRUCTURE
# ============================================================

r = SESSION.get(
    PRIVATE_API + f"/releases/tags/{NEW_TAG}",
    timeout=120
)

if r.status_code == 404:

    r = SESSION.post(
        PRIVATE_API + "/releases",
        json={
            "tag_name": NEW_TAG,
            "name": "Consolidated final evaluation",
            "body": (
                "Verified inference, locked evaluation "
                "predictions and publication assets."
            ),
            "draft": False
        },
        timeout=120
    )

r.raise_for_status()

RELEASE_NEW = r.json()

REMOTE_NEW = {
    a["name"]: a
    for a in paginate(
        PRIVATE_API
        + f"/releases/{RELEASE_NEW['id']}/assets"
    )
}

MANIFEST_NEW = []

def upload_new(path):

    path = Path(path)

    digest = hash_file(path)

    name = (
        f"{path.stem}_{digest[:12]}"
        f"{path.suffix}"
    )

    asset = REMOTE_NEW.get(name)

    if asset is None:

        with open(path, "rb") as f:

            r = SESSION.post(
                RELEASE_NEW["upload_url"].split("{")[0],
                params={"name": name},
                headers={
                    "Content-Type":
                        "application/octet-stream"
                },
                data=f,
                timeout=900
            )

        r.raise_for_status()

        asset = r.json()

        REMOTE_NEW[name] = asset

    assert asset["size"] == path.stat().st_size

    if asset.get("digest"):

        assert asset["digest"] == "sha256:" + digest

    else:

        verify_path = download(
            asset,
            REC_NEW / ("verify_" + name),
            expected=digest
        )

        assert hash_file(verify_path) == digest

        verify_path.unlink()

    record = {
        "name": name,
        "asset_id": asset["id"],
        "sha256": digest,
        "bytes": asset["size"],
        "remote_verified": True
    }

    MANIFEST_NEW.append(record)

    dump_json(
        MANIFEST_NEW,
        AUDIT_NEW / "running_remote_manifest.json"
    )

    print("Remote verified:", name)

    return record

def table_new(data, name):

    df = pd.DataFrame(data)

    for folder in [
        TABLES_NEW,
        BACKUP_NEW / "tables",
        PUB / "tables"
    ]:

        folder.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            folder / f"{name}.csv",
            index=False
        )

        df.to_latex(
            folder / f"{name}.tex",
            index=False,
            escape=True,
            float_format="%.5f"
        )

    return df

# ============================================================
# STEP 194 — RECONSTRUCT EXACT DATASET PARTITIONS
# ============================================================

banner("STEP 194 — LOCKED DATASET RECOVERY")

try:

    from normalizer import normalize as bn_normalize

except ImportError:

    subprocess.check_call([
        sys.executable,
        "-m", "pip", "install", "-q",
        "git+https://github.com/csebuetnlp/normalizer"
    ])

    from normalizer import normalize as bn_normalize

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

RAW_HASH = (
    "4580950af06bbfbe5bd79e20957b63a2ed"
    "122af2a6a3e685b10576d3f70241c5"
)

raw_candidates = list(
    INPUT.rglob("BECS Dataset.csv")
)

assert raw_candidates, "BECS dataset not attached."

RAW_PATH = next(
    (
        p for p in raw_candidates
        if hash_file(p) == RAW_HASH
    ),
    None
)

assert RAW_PATH is not None, (
    "Original dataset checksum mismatch."
)

RAW = pd.read_csv(RAW_PATH)

assert len(RAW) == 15584
assert {"Bangla", "Label"}.issubset(RAW.columns)

def normalize_text(text):

    text = unicodedata.normalize("NFC", str(text))

    text = re.sub(
        r"[\u200b\u200c\u200d\u2060\ufeff]",
        "",
        text
    )

    return re.sub(
        r"\s+", " ", text
    ).strip().casefold()

def sample_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

base_assets_new = {
    a["name"]: a
    for a in assets(
        "stage6b-locked-baselines-v1"
    )
}

package_path = download(
    base_assets_new[
        "stage6b_reproducibility_package.zip"
    ],
    REC_NEW / "baseline_package.zip"
)

def read_zip_member(path, suffix):

    with zipfile.ZipFile(path) as z:

        names = [
            n for n in z.namelist()
            if n.endswith(suffix)
        ]

        assert len(names) == 1, (
            suffix, names
        )

        return z.read(names[0])

def recover_partition(split):

    membership = pd.read_csv(
        io.BytesIO(
            read_zip_member(
                package_path,
                f"partitions/seed42_{split}.csv"
            )
        )
    ).sort_values("partition_order")

    frame = RAW.iloc[
        membership["raw_row_id"].astype(int).values
    ].copy().reset_index(drop=True)

    frame["text_norm"] = frame[
        "Bangla"
    ].map(normalize_text)

    frame["sample_id"] = frame[
        "text_norm"
    ].map(sample_hash)

    frame["Label"] = (
        frame["Label"].astype(str)
        .str.strip().str.title()
    )

    assert np.array_equal(
        frame["sample_id"].values,
        membership["sample_id"].astype(str).values
    )

    assert np.array_equal(
        frame["Label"].values,
        membership["Label"].astype(str).values
    )

    frame["group_id"] = (
        membership["group_id"].astype(str).values
    )

    frame["label_id"] = frame[
        "Label"
    ].map(LABEL_MAP)

    assert frame["label_id"].notna().all()

    frame["model_text"] = frame[
        "text_norm"
    ].map(bn_normalize)

    frame["word_count"] = frame[
        "text_norm"
    ].str.split().str.len()

    return frame

# Do not load the test partition until all validation
# inference checks have passed.

TRAIN_NEW = recover_partition("train")
VAL_NEW = recover_partition("val")

assert len(TRAIN_NEW) == 9204
assert len(VAL_NEW) == 1840

for col in [
    "sample_id",
    "group_id",
    "model_text"
]:

    assert not (
        set(TRAIN_NEW[col])
        & set(VAL_NEW[col])
    ), f"Train-validation overlap: {col}"

YVAL = VAL_NEW["label_id"].to_numpy()

print("Original dataset SHA256: PASS")
print("Training samples:", len(TRAIN_NEW))
print("Validation samples:", len(VAL_NEW))
print("Test loaded: NO")

# ============================================================
# STEP 195 — RESTORE EXACT PAS-NET ARCHITECTURE
# ============================================================

banner("STEP 195 — EXACT PAS-NET IMPLEMENTATION")

pas_source_path = (
    REPO / "research_stage7b"
    / "stage7b_complete.py"
)

assert hash_file(pas_source_path) == (
    PROTOCOL_NEW["architecture_source_sha256"]
)

pas_source = pas_source_path.read_text()

pas_tree = ast.parse(pas_source)

DIM = 128
DROPOUT = .15
MAX_LENGTH = 96

# Execute only the exact class definitions.
# No original training loop is executed.

for class_name in [
    "SentimentDataset",
    "PASNet"
]:

    nodes = [
        n for n in pas_tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == class_name
    ]

    assert len(nodes) == 1

    module = ast.Module(
        body=nodes,
        type_ignores=[]
    )

    exec(
        compile(
            module,
            "<original_pasnet_architecture>",
            "exec"
        ),
        globals()
    )

(
    BACKUP_NEW / "original_pasnet_source.py"
).write_text(pas_source)

print("Original architecture SHA256: PASS")
print("PAS-Net implementation: RESTORED")

# ============================================================
# STEP 196 — CHECKPOINT RECOVERY AND INFERENCE ENGINE
# ============================================================

banner("STEP 196 — STRICT MODEL RELOAD ENGINE")

inventory = pd.DataFrame(
    PROTOCOL_NEW["checkpoint_inventory"]
)

inventory = inventory.set_index("model")

MODEL_NAMES = inventory.index.tolist()

PAS_NAMES = [
    "base_standard",
    "base_supcon",
    "base_prototype",
    "base_pasnet",
    "large_standard",
    "large_pasnet"
]

pas_assets_new = {
    a["name"]: a
    for a in assets(
        "stage7b-pasnet-screening-v1"
    )
}

CHECKPOINT_DIR = (
    WORK / "sentiment_research_stage9c"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True, exist_ok=True
)

# Recover original baseline tokenizer/config files.

BASE_CONFIG_ROOT = REC_NEW / "baseline_configs"

BASE_CONFIG_ROOT.mkdir(exist_ok=True)

with zipfile.ZipFile(package_path) as z:

    for member in z.namelist():

        parts = Path(member).parts

        if "checkpoints" not in parts:
            continue

        if member.endswith("/"):
            continue

        if Path(member).suffix.lower() == ".pt":
            continue

        i = parts.index("checkpoints")

        relative = Path(*parts[i+1:])

        assert ".." not in relative.parts

        destination = BASE_CONFIG_ROOT / relative

        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        destination.write_bytes(z.read(member))

BASE_FOLDER = {
    "locked_banglabert": "BanglaBERT",
    "xlm_roberta": "XLM-RoBERTa",
    "mbert": "mBERT"
}

def checkpoint_path(name):

    row = inventory.loc[name]

    filename = row["filename"]

    available = (
        pas_assets_new
        if name in PAS_NAMES
        else base_assets_new
    )

    return download(
        available[filename],
        CHECKPOINT_DIR / filename,
        expected=row["sha256"]
    )

def build_model(name):

    row = inventory.loc[name]

    path = checkpoint_path(name)

    if name in PAS_NAMES:

        # Use the exact original constructor.
        # All pretrained parameters are then replaced
        # by the verified saved checkpoint.

        model = PASNet(
            row["backbone"],
            row["mode"]
        )

        tokenizer = AutoTokenizer.from_pretrained(
            row["backbone"]
        )

    else:

        folder = (
            BASE_CONFIG_ROOT / BASE_FOLDER[name]
        )

        assert (
            folder / "config.json"
        ).exists(), f"Missing saved config: {name}"

        config = AutoConfig.from_pretrained(
            str(folder)
        )

        model = (
            AutoModelForSequenceClassification
            .from_config(config)
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(folder)
        )

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if "state_dict" in state:

        state = state["state_dict"]

    elif "model_state_dict" in state:

        state = state["model_state_dict"]

    model.load_state_dict(
        state,
        strict=True
    )

    del state

    model.to(DEVICE)
    model.eval()

    return model, tokenizer

def cleanup_new():

    gc.collect()

    torch.cuda.empty_cache()

def inference(name, model, tokenizer, frame):

    model.eval()

    if name in PAS_NAMES:

        dataset = SentimentDataset(
            frame, tokenizer
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

    else:

        texts = (
            frame["model_text"].tolist()
            if name == "locked_banglabert"
            else frame["text_norm"].tolist()
        )

        encoded = tokenizer(
            texts,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        class BaselineDataset(Dataset):

            def __len__(self):

                return len(texts)

            def __getitem__(self, i):

                return {
                    k: v[i]
                    for k, v in encoded.items()
                }

        loader = DataLoader(
            BaselineDataset(),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

    logits = []
    gates = []

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        for batch in loader:

            batch = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            batch.pop("labels", None)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                if name in PAS_NAMES:

                    result = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        valid=batch["valid"]
                    )

                    z = result["logits"]

                    gates.append(
                        result["gate"]
                        .float().cpu().numpy()
                    )

                else:

                    z = model(**batch).logits

            logits.append(
                z.float().cpu().numpy()
            )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    logits = np.concatenate(logits)

    assert logits.shape == (len(frame), 3)
    assert np.isfinite(logits).all()

    probabilities = softmax(
        logits.astype(np.float64),
        axis=1
    )

    gate_values = (
        np.concatenate(gates)
        if gates
        else np.zeros(len(frame))
    )

    return {
        "logits": logits,
        "probabilities": probabilities,
        "gates": gate_values,
        "seconds": elapsed,
        "samples_per_second":
            len(frame) / max(elapsed, 1e-9)
    }

# ============================================================
# STEP 197 — RECOVER SAVED VALIDATION REFERENCES
# ============================================================

banner("STEP 197 — VALIDATION REFERENCE RECOVERY")

REFERENCES_NEW = {}

for name in PAS_NAMES:

    asset = pas_assets_new[
        f"{name}_seed42_val.npz"
    ]

    path = download(
        asset,
        REC_NEW / asset["name"]
    )

    with np.load(
        path, allow_pickle=False
    ) as d:

        assert np.array_equal(
            d["sample_id"].astype(str),
            VAL_NEW["sample_id"].astype(str).values
        )

        assert np.array_equal(
            d["labels"].astype(int),
            YVAL
        )

        REFERENCES_NEW[name] = (
            d["logits"].astype(np.float64)
        )

for name, folder in BASE_FOLDER.items():

    raw = read_zip_member(
        package_path,
        f"private_predictions/{folder}_val.npz"
    )

    with np.load(
        io.BytesIO(raw),
        allow_pickle=False
    ) as d:

        assert np.array_equal(
            d["sample_id"].astype(str),
            VAL_NEW["sample_id"].astype(str).values
        )

        assert np.array_equal(
            d["labels"].astype(int),
            YVAL
        )

        REFERENCES_NEW[name] = (
            d["logits"].astype(np.float64)
        )

# ============================================================
# STEP 198 — REPRODUCE ALL NINE VALIDATION MODELS
# ============================================================

banner("STEP 198 — STRICT VALIDATION REPRODUCTION")

# Tolerances fixed before inference:
# small numerical variation is permitted,
# but materially different predictions stop test access.

MAX_F1_DIFFERENCE = .002
MIN_PREDICTION_AGREEMENT = .995
MAX_MEAN_PROBABILITY_ERROR = .002

def macro(y, p):

    return float(
        f1_score(
            y,
            p.argmax(axis=1),
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    )

VALIDATION_PROBS = {}
VALIDATION_RECORDS = []
COMPLEXITY = []

for name in MODEL_NAMES:

    print("\nReproducing:", name)

    model, tokenizer = build_model(name)

    result = inference(
        name,
        model,
        tokenizer,
        VAL_NEW
    )

    p = result["probabilities"]

    reference = softmax(
        REFERENCES_NEW[name],
        axis=1
    )

    agreement = float(
        (
            p.argmax(1)
            == reference.argmax(1)
        ).mean()
    )

    f1_actual = macro(YVAL, p)
    f1_reference = macro(YVAL, reference)

    probability_error = float(
        np.abs(p - reference).mean()
    )

    passed = bool(
        abs(f1_actual - f1_reference)
        <= MAX_F1_DIFFERENCE
        and agreement
        >= MIN_PREDICTION_AGREEMENT
        and probability_error
        <= MAX_MEAN_PROBABILITY_ERROR
    )

    record = {
        "model": name,
        "strict_reload": True,
        "saved_macro_f1": f1_reference,
        "reproduced_macro_f1": f1_actual,
        "absolute_f1_difference":
            abs(f1_actual - f1_reference),
        "prediction_agreement": agreement,
        "mean_probability_error":
            probability_error,
        "max_probability_error":
            float(np.abs(p-reference).max()),
        "passed": passed
    }

    VALIDATION_RECORDS.append(record)

    VALIDATION_PROBS[name] = p

    COMPLEXITY.append({
        "model": name,
        "parameters": sum(
            x.numel()
            for x in model.parameters()
        ),
        "validation_seconds":
            result["seconds"],
        "samples_per_second":
            result["samples_per_second"],
        "max_length":
            96 if name in PAS_NAMES else 64,
        "device":
            torch.cuda.get_device_name(0),
        "batch_size": BATCH_SIZE
    })

    path = (
        PRED_NEW
        / f"{name}_reproduced_val.npz"
    )

    np.savez_compressed(
        path,
        sample_id=VAL_NEW["sample_id"].astype(str).values,
        labels=YVAL,
        logits=result["logits"],
        probabilities=p,
        gates=result["gates"]
    )

    upload_new(path)

    table_new(
        VALIDATION_RECORDS,
        "final_01_validation_reproduction"
    )

    table_new(
        COMPLEXITY,
        "final_02_model_complexity"
    )

    push_new(
        "Preserve validation reproduction " + name
    )

    print(record)

    del model, tokenizer, result

    cleanup_new()

VALIDATION_AUDIT = table_new(
    VALIDATION_RECORDS,
    "final_01_validation_reproduction"
)

assert VALIDATION_AUDIT["passed"].all(), (
    "Validation reproduction failed. "
    "Evidence backed up. Test remains unopened."
)

PROPOSED_NEW = PROTOCOL_NEW[
    "candidate_members"
]

CONTROL_NEW = PROTOCOL_NEW[
    "three_member_standard_control"
]

candidate_val = np.mean(
    [
        VALIDATION_PROBS[n]
        for n in PROPOSED_NEW
    ],
    axis=0
)

control_val = np.mean(
    [
        VALIDATION_PROBS[n]
        for n in CONTROL_NEW
    ],
    axis=0
)

assert abs(
    macro(YVAL, candidate_val)
    - PROTOCOL_NEW[
        "validation_candidate_macro_f1"
    ]
) <= MAX_F1_DIFFERENCE

print("\nALL VALIDATION CHECKS: PASS")
print(
    "Reproduced candidate F1:",
    macro(YVAL, candidate_val)
)
print(
    "Reproduced control F1:",
    macro(YVAL, control_val)
)

# ============================================================
# LOCK ANALYSIS POLICY BEFORE TEST ACCESS
# ============================================================

EVAL_POLICY = {
    "candidate": PROPOSED_NEW,
    "control": CONTROL_NEW,
    "aggregation": "equal probability mean",
    "primary_metric": "macro_f1",
    "primary_comparison":
        "prototype_family versus three_member_standard_control",
    "bootstrap_replicates": BOOTSTRAPS,
    "validation_reproduction_passed": True,
    "validation_tolerances": {
        "max_f1_difference": MAX_F1_DIFFERENCE,
        "min_prediction_agreement":
            MIN_PREDICTION_AGREEMENT,
        "max_mean_probability_error":
            MAX_MEAN_PROBABILITY_ERROR
    },
    "external_tuning": False,
    "test_tuning": False,
    "test_accessed_at_policy_creation": False,
    "source_sha256": hashlib.sha256(
        SOURCE_NEW.encode()
    ).hexdigest()
}

dump_json(
    EVAL_POLICY,
    BACKUP_NEW / "final_evaluation_policy.json"
)

push_new(
    "Lock final evaluation policy before test access"
)

# ============================================================
# STEP 199 — LOCKED TEST PARTITION AND INFERENCE
# ============================================================

banner("STEP 199 — LOCKED INTERNAL TEST EVALUATION")

TEST_NEW = recover_partition("test")

assert len(TEST_NEW) == 1849

for col in [
    "sample_id",
    "group_id",
    "model_text"
]:

    assert not (
        set(TEST_NEW[col])
        & set(TRAIN_NEW[col])
    ), f"Train-test overlap: {col}"

    assert not (
        set(TEST_NEW[col])
        & set(VAL_NEW[col])
    ), f"Validation-test overlap: {col}"

YTEST = TEST_NEW[
    "label_id"
].to_numpy()

TEST_GROUPS = TEST_NEW[
    "group_id"
].astype(str).values

print("Locked test samples:", len(TEST_NEW))
print("Partition overlap checks: PASS")

# ============================================================
# STEP 207 PREPARATION — EXTERNAL DATASET DISCOVERY
#
# Discover before model inference to avoid loading each
# model again. Only attached datasets with explicit
# sentiment labels are accepted.
# ============================================================

banner("EXTERNAL DATASET DISCOVERY")

EXTERNAL_NEW = {}
EXTERNAL_AUDIT = []

SOURCE_TEXTS = set(
    pd.concat([
        TRAIN_NEW["text_norm"],
        VAL_NEW["text_norm"],
        TEST_NEW["text_norm"]
    ])
)

SOURCE_NORMALIZED = set(
    pd.concat([
        TRAIN_NEW["model_text"],
        VAL_NEW["model_text"],
        TEST_NEW["model_text"]
    ])
)

TEXT_COLUMNS = [
    "text", "sentence", "review",
    "comment", "bangla", "content"
]

LABEL_COLUMNS = [
    "label", "sentiment", "polarity",
    "class", "sentiment_label"
]

EXTERNAL_FILES = []

for p in INPUT.rglob("*"):

    if not p.is_file():
        continue

    if p.suffix.lower() not in [
        ".csv", ".tsv", ".jsonl", ".parquet"
    ]:
        continue

    name = str(p).lower()

    if (
        "sentnob" in name
        or "sent-nob" in name
        or "sentigold" in name
        or "senti-gold" in name
    ):

        EXTERNAL_FILES.append(p)

for path in EXTERNAL_FILES:

    try:

        if path.suffix == ".parquet":

            frame = pd.read_parquet(path)

        elif path.suffix == ".jsonl":

            frame = pd.read_json(
                path, lines=True
            )

        else:

            frame = pd.read_csv(
                path,
                sep=(
                    "\t"
                    if path.suffix == ".tsv"
                    else ","
                )
            )

        columns = {
            str(c).strip().lower(): c
            for c in frame.columns
        }

        text_col = next(
            (
                columns[c]
                for c in TEXT_COLUMNS
                if c in columns
            ),
            None
        )

        label_col = next(
            (
                columns[c]
                for c in LABEL_COLUMNS
                if c in columns
            ),
            None
        )

        assert text_col is not None
        assert label_col is not None

        frame = frame[
            [text_col, label_col]
        ].dropna().copy()

        frame.columns = ["Bangla", "Label"]

        frame["Label"] = (
            frame["Label"].astype(str)
            .str.strip().str.title()
        )

        # Never guess numeric label mappings.

        assert set(
            frame["Label"]
        ).issubset(LABEL_MAP), (
            "Label mapping requires source verification."
        )

        frame["text_norm"] = frame[
            "Bangla"
        ].map(normalize_text)

        frame = frame[
            frame["text_norm"].str.len() > 0
        ].copy()

        original_n = len(frame)

        frame["model_text"] = frame[
            "text_norm"
        ].map(bn_normalize)

        overlap = (
            frame["text_norm"].isin(SOURCE_TEXTS)
            | frame["model_text"].isin(
                SOURCE_NORMALIZED
            )
        )

        frame = frame[~overlap].copy()

        conflict = (
            frame.groupby("text_norm")["Label"]
            .transform("nunique") > 1
        )

        frame = frame[~conflict].copy()

        frame = frame.drop_duplicates(
            "text_norm"
        ).reset_index(drop=True)

        frame["label_id"] = frame[
            "Label"
        ].map(LABEL_MAP)

        frame["sample_id"] = frame[
            "text_norm"
        ].map(sample_hash)

        frame["group_id"] = frame["sample_id"]

        frame["word_count"] = frame[
            "text_norm"
        ].str.split().str.len()

        assert len(frame) >= 100

        assert frame["label_id"].nunique() == 3

        name = (
            re.sub(
                r"[^a-zA-Z0-9]+",
                "_",
                path.stem
            )
            + "_"
            + hash_file(path)[:8]
        )

        EXTERNAL_NEW[name] = frame

        EXTERNAL_AUDIT.append({
            "dataset": name,
            "filename": path.name,
            "source_sha256": hash_file(path),
            "original_samples": original_n,
            "exact_source_overlap_removed":
                int(overlap.sum()),
            "conflicting_rows_removed":
                int(conflict.sum()),
            "evaluation_samples": len(frame),
            "status": "PROVISIONAL_EXTERNAL",
            "limitation":
                "Dataset provenance and near-duplicate "
                "overlap still require verification."
        })

    except Exception as e:

        EXTERNAL_AUDIT.append({
            "dataset": path.name,
            "status": "SKIPPED",
            "reason": str(e)
        })

if not EXTERNAL_FILES:

    EXTERNAL_AUDIT.append({
        "dataset": "SentNoB / SentiGOLD",
        "status": "NOT_ATTACHED",
        "reason":
            "No supported external dataset files "
            "were found in Kaggle input."
    })

table_new(
    EXTERNAL_AUDIT,
    "final_03_external_dataset_audit"
)

print("External files discovered:", len(EXTERNAL_FILES))
print("Provisional external cohorts:", len(EXTERNAL_NEW))

# ============================================================
# ALL MODEL INFERENCE — INTERNAL AND AVAILABLE EXTERNAL
# ============================================================

COHORTS = {
    "internal_test": TEST_NEW,
    **EXTERNAL_NEW
}

ALL_PROBS = {
    cohort: {}
    for cohort in COHORTS
}

ALL_GATES = {
    cohort: {}
    for cohort in COHORTS
}

INFERENCE_TIMES = []

for name in MODEL_NAMES:

    print("\nEvaluating model:", name)

    model, tokenizer = build_model(name)

    for cohort, frame in COHORTS.items():

        result = inference(
            name,
            model,
            tokenizer,
            frame
        )

        ALL_PROBS[cohort][name] = (
            result["probabilities"]
        )

        ALL_GATES[cohort][name] = (
            result["gates"]
        )

        INFERENCE_TIMES.append({
            "cohort": cohort,
            "model": name,
            "samples": len(frame),
            "seconds": result["seconds"],
            "samples_per_second":
                result["samples_per_second"]
        })

        path = (
            PRED_NEW
            / f"{cohort}_{name}.npz"
        )

        np.savez_compressed(
            path,
            sample_id=frame[
                "sample_id"
            ].astype(str).values,
            labels=frame[
                "label_id"
            ].to_numpy(),
            group_id=frame[
                "group_id"
            ].astype(str).values,
            logits=result["logits"],
            probabilities=result["probabilities"],
            gates=result["gates"]
        )

        upload_new(path)

        print(
            cohort,
            "| macro-F1:",
            round(
                macro(
                    frame["label_id"].to_numpy(),
                    result["probabilities"]
                ),
                6
            )
        )

    table_new(
        INFERENCE_TIMES,
        "final_04_inference_times"
    )

    push_new(
        "Preserve inference audit " + name
    )

    del model, tokenizer

    cleanup_new()

# ============================================================
# STEP 200 — CLASSICAL ML BASELINE
# ============================================================

banner("STEP 200 — LOCKED TF-IDF LOGISTIC BASELINE")

features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            tokenizer=str.split,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    )
])

X_train = features.fit_transform(
    TRAIN_NEW["text_norm"]
)

classifier = LogisticRegression(
    C=4,
    max_iter=2000,
    random_state=42
)

classifier.fit(
    X_train,
    TRAIN_NEW["label_id"]
)

assert np.array_equal(
    classifier.classes_,
    np.array([0, 1, 2])
)

for cohort, frame in COHORTS.items():

    p = classifier.predict_proba(
        features.transform(
            frame["text_norm"]
        )
    )

    ALL_PROBS[cohort][
        "tfidf_logistic_regression"
    ] = p

    path = (
        PRED_NEW
        / f"{cohort}_tfidf_logistic_regression.npz"
    )

    np.savez_compressed(
        path,
        sample_id=frame["sample_id"].astype(str).values,
        labels=frame["label_id"].to_numpy(),
        probabilities=p
    )

    upload_new(path)

del X_train

gc.collect()

# ============================================================
# STEP 201 — LOCKED ENSEMBLES
# ============================================================

banner("STEP 201 — FINAL ENSEMBLE EVALUATION")

ENSEMBLE_DEFINITIONS = {
    "prototype_family": PROPOSED_NEW,
    "three_member_standard_control": CONTROL_NEW,
    "base_large_standard_ensemble": [
        "base_standard",
        "large_standard"
    ],
    "base_large_pasnet_ensemble": [
        "base_pasnet",
        "large_pasnet"
    ]
}

for cohort, systems in ALL_PROBS.items():

    for name, members in (
        ENSEMBLE_DEFINITIONS.items()
    ):

        systems[name] = np.mean(
            [systems[m] for m in members],
            axis=0
        )

    frame = COHORTS[cohort]

    path = (
        PRED_NEW
        / f"{cohort}_all_systems.npz"
    )

    np.savez_compressed(
        path,
        sample_id=frame["sample_id"].astype(str).values,
        labels=frame["label_id"].to_numpy(),
        group_id=frame["group_id"].astype(str).values,
        model_names=np.array(
            list(systems.keys()), dtype=str
        ),
        probabilities=np.stack(
            list(systems.values())
        )
    )

    upload_new(path)

# ============================================================
# STEP 202 — COMPLETE PERFORMANCE METRICS
# ============================================================

banner("STEP 202 — PERFORMANCE AND CLASSWISE METRICS")

def calibration_error(y, p, bins=15):

    confidence = p.max(1)
    correct = p.argmax(1) == y

    bin_ids = np.minimum(
        (confidence * bins).astype(int),
        bins - 1
    )

    ece = 0.

    for i in range(bins):

        mask = bin_ids == i

        if mask.any():

            ece += mask.mean() * abs(
                correct[mask].mean()
                - confidence[mask].mean()
            )

    return float(ece)

def metrics_new(y, p):

    pred = p.argmax(1)

    pr, rc, f1, _ = (
        precision_recall_fscore_support(
            y,
            pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": pr,
        "macro_recall": rc,
        "macro_f1": f1,
        "weighted_f1": f1_score(
            y,
            pred,
            average="weighted",
            zero_division=0
        ),
        "nll": log_loss(
            y, p, labels=[0, 1, 2]
        ),
        "brier": np.mean(
            np.sum(
                (p - np.eye(3)[y])**2,
                axis=1
            )
        ),
        "ece": calibration_error(y, p)
    }

RESULT_ROWS = []
CLASS_ROWS = []
CONFUSION_ROWS = []

for cohort, systems in ALL_PROBS.items():

    y = COHORTS[cohort][
        "label_id"
    ].to_numpy()

    for name, p in systems.items():

        RESULT_ROWS.append({
            "cohort": cohort,
            "model": name,
            "samples": len(y),
            **metrics_new(y, p)
        })

        pred = p.argmax(1)

        pr, rc, f1, support = (
            precision_recall_fscore_support(
                y,
                pred,
                labels=[0, 1, 2],
                zero_division=0
            )
        )

        cm = confusion_matrix(
            y, pred,
            labels=[0, 1, 2]
        )

        for i, label in enumerate(LABELS):

            CLASS_ROWS.append({
                "cohort": cohort,
                "model": name,
                "class": label,
                "precision": pr[i],
                "recall": rc[i],
                "f1": f1[i],
                "support": support[i]
            })

            for j, predicted in enumerate(LABELS):

                CONFUSION_ROWS.append({
                    "cohort": cohort,
                    "model": name,
                    "true_class": label,
                    "predicted_class": predicted,
                    "count": int(cm[i, j])
                })

RESULTS_NEW = table_new(
    RESULT_ROWS,
    "final_05_all_performance"
)

CLASSWISE_NEW = table_new(
    CLASS_ROWS,
    "final_06_classwise_performance"
)

CONFUSIONS_NEW = table_new(
    CONFUSION_ROWS,
    "final_07_confusion_matrices"
)

print(
    RESULTS_NEW[
        RESULTS_NEW["cohort"] == "internal_test"
    ].sort_values(
        "macro_f1", ascending=False
    ).to_string(index=False)
)

# ============================================================
# STEP 203 — PAIRED STATISTICAL ANALYSIS
# ============================================================

banner("STEP 203 — PAIRED GROUP BOOTSTRAP")

def fast_macro(y, pred):

    cm = np.bincount(
        3 * y + pred,
        minlength=9
    ).reshape(3, 3)

    denominator = (
        cm.sum(0) + cm.sum(1)
    )

    return float(
        np.divide(
            2 * np.diag(cm),
            denominator,
            out=np.zeros(3),
            where=denominator > 0
        ).mean()
    )

def holm_adjust(pvalues):

    pvalues = np.asarray(pvalues)

    order = np.argsort(pvalues)

    adjusted = np.empty(len(order))

    running = 0.

    for rank, i in enumerate(order):

        running = max(
            running,
            (len(order)-rank)*pvalues[i]
        )

        adjusted[i] = min(1., running)

    return adjusted

PAIRED_ROWS = []
BOOTSTRAP_ROWS = []

for cohort, systems in ALL_PROBS.items():

    frame = COHORTS[cohort]

    y = frame["label_id"].to_numpy()

    groups = frame["group_id"].astype(str).values

    _, inverse = np.unique(
        groups, return_inverse=True
    )

    group_rows = [
        np.flatnonzero(inverse == i)
        for i in range(inverse.max()+1)
    ]

    candidate = systems[
        "prototype_family"
    ].argmax(1)

    references = [
        name for name in systems
        if name != "prototype_family"
    ]

    reference_predictions = {
        name: systems[name].argmax(1)
        for name in references
    }

    rng = np.random.default_rng(42)

    deltas = np.empty(
        (BOOTSTRAPS, len(references))
    )

    candidate_scores = np.empty(BOOTSTRAPS)

    for b in range(BOOTSTRAPS):

        selected = rng.integers(
            0,
            len(group_rows),
            len(group_rows)
        )

        idx = np.concatenate([
            group_rows[j]
            for j in selected
        ])

        s = fast_macro(
            y[idx],
            candidate[idx]
        )

        candidate_scores[b] = s

        for j, name in enumerate(references):

            deltas[b, j] = (
                s - fast_macro(
                    y[idx],
                    reference_predictions[name][idx]
                )
            )

    rows = []

    for j, name in enumerate(references):

        ref = reference_predictions[name]

        gained = int(
            (
                (candidate == y)
                & (ref != y)
            ).sum()
        )

        lost = int(
            (
                (candidate != y)
                & (ref == y)
            ).sum()
        )

        pvalue = (
            binomtest(
                min(gained, lost),
                gained+lost,
                .5
            ).pvalue
            if gained+lost
            else 1.
        )

        low, high = np.quantile(
            deltas[:, j],
            [.025, .975]
        )

        rows.append({
            "cohort": cohort,
            "candidate": "prototype_family",
            "reference": name,
            "primary_comparison": (
                name == "three_member_standard_control"
            ),
            "delta_macro_f1": (
                fast_macro(y, candidate)
                - fast_macro(y, ref)
            ),
            "ci95_low": low,
            "ci95_high": high,
            "gained_correct": gained,
            "lost_correct": lost,
            "mcnemar_p": pvalue,
            "bootstrap_replicates": BOOTSTRAPS
        })

    adjusted = holm_adjust([
        r["mcnemar_p"] for r in rows
    ])

    for row, pvalue in zip(rows, adjusted):

        row["holm_p"] = pvalue

    PAIRED_ROWS.extend(rows)

    low, high = np.quantile(
        candidate_scores,
        [.025, .975]
    )

    BOOTSTRAP_ROWS.append({
        "cohort": cohort,
        "model": "prototype_family",
        "macro_f1":
            fast_macro(y, candidate),
        "ci95_low": low,
        "ci95_high": high
    })

    path = (
        PRED_NEW
        / f"{cohort}_bootstrap_distributions.npz"
    )

    np.savez_compressed(
        path,
        references=np.array(
            references, dtype=str
        ),
        deltas=deltas,
        candidate_scores=candidate_scores
    )

    upload_new(path)

PAIRED_NEW = table_new(
    PAIRED_ROWS,
    "final_08_paired_statistics"
)

table_new(
    BOOTSTRAP_ROWS,
    "final_09_candidate_confidence_intervals"
)

print(
    PAIRED_NEW[
        PAIRED_NEW["primary_comparison"]
    ].to_string(index=False)
)

# ============================================================
# STEP 204 — LINGUISTIC SUBGROUPS AND CALIBRATION
# ============================================================

banner("STEP 204 — SUBGROUP AND CALIBRATION ANALYSIS")

NEGATION = [
    "না", "নয়", "নয়", "নেই", "নাই",
    "কখনোই", "নাহ", "নাকো"
]

CONTRAST = [
    "কিন্তু", "তবে", "যদিও",
    "অথচ", "তবুও", "বরং"
]

def cue_pattern(words):

    return (
        r"(?<![\w\u0980-\u09ff])(?:"
        + "|".join(
            re.escape(w) for w in words
        )
        + r")(?![\w\u0980-\u09ff])"
    )

LONG_THRESHOLD = float(
    TRAIN_NEW["word_count"].quantile(.75)
)

SUBGROUP_ROWS = []
RELIABILITY_ROWS = []
GATE_ROWS = []

SELECTED = [
    "locked_banglabert",
    "base_standard",
    "large_standard",
    "base_pasnet",
    "large_pasnet",
    "three_member_standard_control",
    "prototype_family"
]

for cohort, frame in COHORTS.items():

    y = frame["label_id"].to_numpy()

    neg = frame["text_norm"].str.contains(
        cue_pattern(NEGATION),
        regex=True
    ).to_numpy()

    contrast = frame["text_norm"].str.contains(
        cue_pattern(CONTRAST),
        regex=True
    ).to_numpy()

    long = (
        frame["word_count"].to_numpy()
        > LONG_THRESHOLD
    )

    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "negation": neg,
        "contrast": contrast,
        "both": neg & contrast,
        "any_operator": neg | contrast,
        "no_operator": ~(neg | contrast),
        "long": long,
        "shorter": ~long
    }

    for name in SELECTED:

        p = ALL_PROBS[cohort][name]

        for subgroup, mask in masks.items():

            if not mask.any():
                continue

            SUBGROUP_ROWS.append({
                "cohort": cohort,
                "model": name,
                "subgroup": subgroup,
                "samples": int(mask.sum()),
                "class_count": len(
                    np.unique(y[mask])
                ),
                "accuracy": accuracy_score(
                    y[mask],
                    p[mask].argmax(1)
                ),
                "macro_f1": macro(
                    y[mask],
                    p[mask]
                ),
                "small_subgroup":
                    bool(mask.sum() < 50)
            })

        conf = p.max(1)
        correct = p.argmax(1) == y

        bins = np.minimum(
            (conf * 15).astype(int), 14
        )

        for i in range(15):

            mask = bins == i

            if mask.any():

                RELIABILITY_ROWS.append({
                    "cohort": cohort,
                    "model": name,
                    "bin": i,
                    "samples": int(mask.sum()),
                    "mean_confidence":
                        conf[mask].mean(),
                    "observed_accuracy":
                        correct[mask].mean()
                })

    for name in [
        "base_prototype",
        "base_pasnet",
        "large_pasnet"
    ]:

        gates = ALL_GATES[cohort][name]

        for subgroup, mask in masks.items():

            if mask.any():

                GATE_ROWS.append({
                    "cohort": cohort,
                    "model": name,
                    "subgroup": subgroup,
                    "samples": int(mask.sum()),
                    "mean_gate": gates[mask].mean(),
                    "std_gate": gates[mask].std(),
                    "median_gate":
                        np.median(gates[mask])
                })

SUBGROUPS_NEW = table_new(
    SUBGROUP_ROWS,
    "final_10_linguistic_subgroups"
)

RELIABILITY_NEW = table_new(
    RELIABILITY_ROWS,
    "final_11_reliability_bins"
)

GATES_NEW = table_new(
    GATE_ROWS,
    "final_12_adaptive_gate_analysis"
)

# ============================================================
# STEP 205 — MODEL COMPLEXITY AND ENSEMBLE COST
# ============================================================

banner("STEP 205 — INFERENCE COST ANALYSIS")

COMPLEXITY_NEW = pd.DataFrame(COMPLEXITY)

COST_ROWS = []

for name, members in (
    ENSEMBLE_DEFINITIONS.items()
):

    selected = COMPLEXITY_NEW[
        COMPLEXITY_NEW["model"].isin(members)
    ]

    COST_ROWS.append({
        "model": name,
        "member_count": len(members),
        "total_parameters":
            int(selected["parameters"].sum()),
        "summed_validation_inference_seconds":
            selected["validation_seconds"].sum(),
        "estimated_sequential_samples_per_second":
            len(VAL_NEW)
            / selected["validation_seconds"].sum(),
        "device": torch.cuda.get_device_name(0),
        "batch_size": BATCH_SIZE,
        "timing_scope":
            "Forward evaluation loop; "
            "excludes checkpoint loading and tokenization"
    })

COSTS_NEW = table_new(
    COST_ROWS,
    "final_13_ensemble_complexity"
)

print(COSTS_NEW.to_string(index=False))

# ============================================================
# STEP 206 — PUBLICATION-QUALITY FIGURES
# ============================================================

banner("STEP 206 — PUBLICATION FIGURE GENERATION")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white"
})

CAPTIONS_NEW = []
FIGURE_MANIFEST = []

def save_fig_new(fig, name, caption):

    for root in [
        FIG_NEW,
        BACKUP_NEW / "figures",
        PUB / "figures"
    ]:

        for ext in [
            "png", "pdf", "svg"
        ]:

            folder = root / ext

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

            path = folder / f"{name}.{ext}"

            fig.savefig(
                path,
                dpi=600 if ext == "png" else None,
                bbox_inches="tight"
            )

            if root == BACKUP_NEW / "figures":

                FIGURE_MANIFEST.append({
                    "figure": name,
                    "format": ext,
                    "path": str(
                        path.relative_to(REPO)
                    ),
                    "sha256": hash_file(path),
                    "bytes": path.stat().st_size
                })

    CAPTIONS_NEW.append({
        "figure": name,
        "caption": caption
    })

    table_new(
        CAPTIONS_NEW,
        "final_figure_captions"
    )

    table_new(
        FIGURE_MANIFEST,
        "final_figure_manifest"
    )

    push_new("Preserve figure " + name)

    plt.show()
    plt.close(fig)

INTERNAL = RESULTS_NEW[
    RESULTS_NEW["cohort"] == "internal_test"
].copy()

# ------------------------------------------------------------
# FIGURE 1 — COMPLETE TEST PERFORMANCE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

df = INTERNAL.sort_values("macro_f1")

colors = [
    "#00796B" if n == "prototype_family"
    else "#526D82" if n == "three_member_standard_control"
    else "#A8B5C1"
    for n in df["model"]
]

ax.barh(
    df["model"],
    100*df["macro_f1"],
    color=colors
)

for i, value in enumerate(df["macro_f1"]):

    ax.text(
        value*100+.2,
        i,
        f"{value*100:.2f}",
        va="center",
        fontsize=9
    )

ax.set_xlim(0, 105)
ax.set_xlabel("Macro-F1 (%)")
ax.set_title("Locked internal test performance")

fig.tight_layout()

save_fig_new(
    fig,
    "final_01_internal_test_performance",
    "Comparison of all prespecified models "
    "on the reconstructed internal test partition."
)

# ------------------------------------------------------------
# FIGURE 2 — ACCURACY AND MACRO-F1
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

df = INTERNAL[
    INTERNAL["model"].isin(SELECTED)
].set_index("model")

(
    df[["accuracy", "macro_f1"]]*100
).plot.bar(ax=ax)

ax.set_ylabel("Performance (%)")
ax.set_ylim(0, 100)
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=35)

fig.tight_layout()

save_fig_new(
    fig,
    "final_02_accuracy_macro_f1",
    "Accuracy and macro-F1 of selected "
    "standard and prototype-based systems."
)

# ------------------------------------------------------------
# FIGURE 3 — CLASSWISE PERFORMANCE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

df = CLASSWISE_NEW[
    (
        CLASSWISE_NEW["cohort"]
        == "internal_test"
    )
    & CLASSWISE_NEW["model"].isin([
        "locked_banglabert",
        "three_member_standard_control",
        "prototype_family"
    ])
].pivot(
    index="class",
    columns="model",
    values="f1"
).reindex(LABELS)

(df*100).plot.bar(ax=ax)

ax.set_ylabel("Classwise F1 (%)")
ax.set_xlabel("")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=0)

fig.tight_layout()

save_fig_new(
    fig,
    "final_03_classwise_f1",
    "Classwise test performance for the proposed "
    "ensemble and standard controls."
)

# ------------------------------------------------------------
# FIGURES 4–6 — CONFUSION MATRICES
# ------------------------------------------------------------

for i, name in enumerate([
    "locked_banglabert",
    "three_member_standard_control",
    "prototype_family"
], start=4):

    p = ALL_PROBS["internal_test"][name]

    cm = confusion_matrix(
        YTEST,
        p.argmax(1),
        labels=[0, 1, 2]
    )

    normalized = cm / cm.sum(
        axis=1, keepdims=True
    ).clip(min=1)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    im = ax.imshow(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    for r in range(3):

        for c in range(3):

            ax.text(
                c, r,
                f"{cm[r,c]}\n{normalized[r,c]:.1%}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if normalized[r,c] > .5
                    else "black"
                )
            )

    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)

    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("True sentiment")
    ax.set_title(name.replace("_", " "))

    fig.colorbar(im, ax=ax)

    fig.tight_layout()

    save_fig_new(
        fig,
        f"final_{i:02d}_confusion_{name}",
        "Test confusion matrix with counts "
        "and row-normalized percentages."
    )

# ------------------------------------------------------------
# FIGURE 7 — PAIRED EFFECTS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

df = PAIRED_NEW[
    PAIRED_NEW["cohort"] == "internal_test"
].sort_values(
    "delta_macro_f1"
).reset_index(drop=True)

ypos = np.arange(len(df))

ax.hlines(
    ypos,
    df["ci95_low"]*100,
    df["ci95_high"]*100,
    color="#526D82"
)

ax.scatter(
    df["delta_macro_f1"]*100,
    ypos,
    color="#00796B"
)

ax.axvline(
    0,
    linestyle="--",
    color="black"
)

ax.set_yticks(
    ypos,
    df["reference"]
)

ax.set_xlabel(
    "Proposed minus reference macro-F1 "
    "(percentage points)"
)

fig.tight_layout()

save_fig_new(
    fig,
    "final_07_paired_effects",
    "Paired group-bootstrap confidence intervals "
    "from 5,000 test resamples."
)

# ------------------------------------------------------------
# FIGURE 8 — RELIABILITY DIAGRAM
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6))

for name in [
    "locked_banglabert",
    "three_member_standard_control",
    "prototype_family"
]:

    df = RELIABILITY_NEW[
        (
            RELIABILITY_NEW["cohort"]
            == "internal_test"
        )
        & (
            RELIABILITY_NEW["model"]
            == name
        )
    ]

    ax.plot(
        df["mean_confidence"],
        df["observed_accuracy"],
        marker="o",
        label=name
    )

ax.plot(
    [0, 1],
    [0, 1],
    "--",
    color="gray"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean confidence")
ax.set_ylabel("Observed accuracy")
ax.legend(fontsize=8)

fig.tight_layout()

save_fig_new(
    fig,
    "final_08_reliability",
    "Uncalibrated test reliability diagrams. "
    "No test-set temperature fitting was performed."
)

# ------------------------------------------------------------
# FIGURE 9 — LINGUISTIC SUBGROUP PERFORMANCE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 5))

df = SUBGROUPS_NEW[
    (
        SUBGROUPS_NEW["cohort"]
        == "internal_test"
    )
    & SUBGROUPS_NEW["model"].isin([
        "locked_banglabert",
        "three_member_standard_control",
        "prototype_family"
    ])
    & (
        SUBGROUPS_NEW["samples"] >= 50
    )
].pivot(
    index="subgroup",
    columns="model",
    values="macro_f1"
)

(df*100).plot.bar(ax=ax)

ax.set_ylabel("Macro-F1 (%)")
ax.set_ylim(0, 100)
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=25)

fig.tight_layout()

save_fig_new(
    fig,
    "final_09_linguistic_subgroups",
    "Descriptive linguistic subgroup results. "
    "Subgroups smaller than 50 samples are omitted "
    "from this figure but retained in the tables."
)

# ------------------------------------------------------------
# FIGURE 10 — ADAPTIVE GATE DISTRIBUTIONS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

for name in [
    "base_pasnet",
    "large_pasnet"
]:

    gates = ALL_GATES["internal_test"][name]

    ax.hist(
        gates,
        bins=30,
        alpha=.55,
        density=True,
        label=name
    )

ax.set_xlim(0, .5)
ax.set_xlabel("Prototype fusion coefficient")
ax.set_ylabel("Density")
ax.legend()

fig.tight_layout()

save_fig_new(
    fig,
    "final_10_gate_distribution",
    "Distribution of learned prototype fusion "
    "coefficients on the internal test partition."
)

# ------------------------------------------------------------
# FIGURE 11 — PERFORMANCE VERSUS PARAMETERS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

param_map = dict(zip(
    COMPLEXITY_NEW["model"],
    COMPLEXITY_NEW["parameters"]
))

param_map.update(dict(zip(
    COSTS_NEW["model"],
    COSTS_NEW["total_parameters"]
)))

for name in [
    "locked_banglabert",
    "base_standard",
    "large_standard",
    "base_pasnet",
    "large_pasnet",
    "three_member_standard_control",
    "prototype_family"
]:

    row = INTERNAL[
        INTERNAL["model"] == name
    ].iloc[0]

    x = param_map[name]/1e6
    y = row["macro_f1"]*100

    ax.scatter(x, y)

    ax.annotate(
        name,
        (x, y),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlabel("Total parameters (millions)")
ax.set_ylabel("Test macro-F1 (%)")

fig.tight_layout()

save_fig_new(
    fig,
    "final_11_performance_complexity",
    "Performance versus total model parameters. "
    "Ensemble parameter counts are summed."
)

# ------------------------------------------------------------
# FIGURE 12 — VALIDATION REPRODUCIBILITY
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    VALIDATION_AUDIT["saved_macro_f1"]*100,
    VALIDATION_AUDIT["reproduced_macro_f1"]*100
)

ax.plot(
    [70, 90],
    [70, 90],
    "--",
    color="gray"
)

ax.set_xlim(70, 90)
ax.set_ylim(70, 90)

ax.set_xlabel("Saved validation macro-F1 (%)")
ax.set_ylabel("Reproduced validation macro-F1 (%)")

fig.tight_layout()

save_fig_new(
    fig,
    "final_12_validation_reproduction",
    "Reproducibility of all nine saved model "
    "checkpoints before internal test access."
)

# ============================================================
# FIGURE 13 — DETAILED PAS-NET ARCHITECTURE
# ============================================================

from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch
)

fig, ax = plt.subplots(figsize=(13, 9))

ax.set_xlim(0, 13)
ax.set_ylim(0, 10)
ax.axis("off")

def architecture_block(
    x, y, w, h, text,
    color="#E8EEF5"
):

    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor="#334155",
        linewidth=1.2
    )

    ax.add_patch(patch)

    ax.text(
        x+w/2,
        y+h/2,
        text,
        ha="center",
        va="center",
        fontsize=10
    )

def architecture_arrow(x1, y1, x2, y2):

    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            color="#475569",
            linewidth=1.2
        )
    )

architecture_block(
    4.5, 9, 4, .6,
    "Normalized Bangla text"
)

architecture_block(
    4.5, 7.8, 4, .7,
    "BanglaBERT Base / Large encoder",
    "#DBEAFE"
)

architecture_block(
    4.5, 6.5, 4, .8,
    "CLS + masked mean pooling\n"
    "Projection to 128 dimensions"
)

architecture_block(
    .5, 4.8, 3.5, 1,
    "Semantic classifier\nLinear: 128 to 3",
    "#DCEFE9"
)

architecture_block(
    4.75, 4.8, 3.5, 1,
    "Uncertainty features\n"
    "Confidence, entropy, margin",
    "#FDECC8"
)

architecture_block(
    9, 4.8, 3.5, 1,
    "Metric projection\n"
    "L2-normalized embedding",
    "#EDE3FA"
)

architecture_block(
    9, 3.1, 3.5, 1,
    "Two prototypes per class\n"
    "Cosine similarity + logsumexp",
    "#EDE3FA"
)

architecture_block(
    4.75, 3.1, 3.5, 1,
    "Adaptive fusion gate\n"
    "Alpha = 0.5 × sigmoid",
    "#FDECC8"
)

architecture_block(
    3.5, 1.5, 6, .8,
    "Fused logits = (1 − alpha) × semantic\n"
    "+ alpha × prototype logits",
    "#DCEFE9"
)

architecture_block(
    4, .15, 5, .7,
    "Negative / Neutral / Positive"
)

architecture_arrow(6.5, 9, 6.5, 8.5)
architecture_arrow(6.5, 7.8, 6.5, 7.3)

architecture_arrow(5.2, 6.5, 2.25, 5.8)
architecture_arrow(6.5, 6.5, 6.5, 5.8)
architecture_arrow(7.8, 6.5, 10.75, 5.8)

architecture_arrow(4, 5.3, 4.75, 5.3)

architecture_arrow(6.5, 4.8, 6.5, 4.1)
architecture_arrow(10.75, 4.8, 10.75, 4.1)

architecture_arrow(2.25, 4.8, 4.5, 2.3)
architecture_arrow(6.5, 3.1, 6.5, 2.3)
architecture_arrow(10.75, 3.1, 8.5, 2.3)

architecture_arrow(6.5, 1.5, 6.5, .85)

ax.set_title(
    "Prototype-Guided Adaptive Sentiment Network",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()

save_fig_new(
    fig,
    "final_13_pasnet_architecture",
    "PAS-Net architecture with semantic and prototype "
    "classification branches and uncertainty-conditioned "
    "fusion. The fixed prototype variant uses alpha=0.25."
)

# ============================================================
# FIGURE 14 — FINAL ENSEMBLE ARCHITECTURE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

architecture_block(
    4, 6, 4, .6,
    "Bangla input text"
)

for x, text in [
    (
        .3,
        "BanglaBERT Base\n"
        "Fixed prototype fusion"
    ),
    (
        4.3,
        "BanglaBERT Base\n"
        "Adaptive PAS-Net"
    ),
    (
        8.3,
        "BanglaBERT Large\n"
        "Adaptive PAS-Net"
    )
]:

    architecture_block(
        x, 3.9, 3.4, 1.2,
        text,
        "#DBEAFE"
    )

    architecture_arrow(
        6, 6,
        x+1.7, 5.1
    )

    architecture_block(
        x, 2.6, 3.4, .6,
        "Class probabilities",
        "#EDE3FA"
    )

    architecture_arrow(
        x+1.7, 3.9,
        x+1.7, 3.2
    )

    architecture_arrow(
        x+1.7, 2.6,
        6, 1.8
    )

architecture_block(
    3.5, 1.1, 5, .7,
    "Equal probability averaging",
    "#DCEFE9"
)

architecture_block(
    3.5, .05, 5, .6,
    "Final sentiment prediction"
)

architecture_arrow(6, 1.1, 6, .65)

ax.set_title(
    "Locked Prototype-Family Ensemble",
    fontsize=15,
    fontweight="bold"
)

fig.tight_layout()

save_fig_new(
    fig,
    "final_14_ensemble_architecture",
    "The locked three-member prototype-family "
    "ensemble with equal probability averaging."
)

# ============================================================
# STEP 207 — EXTERNAL EVALUATION FIGURES
# ============================================================

banner("STEP 207 — EXTERNAL EVALUATION SUMMARY")

if EXTERNAL_NEW:

    fig, ax = plt.subplots(figsize=(11, 5))

    df = RESULTS_NEW[
        RESULTS_NEW["model"].isin([
            "locked_banglabert",
            "three_member_standard_control",
            "prototype_family"
        ])
    ].pivot(
        index="cohort",
        columns="model",
        values="macro_f1"
    )

    (df*100).plot.bar(ax=ax)

    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()

    save_fig_new(
        fig,
        "final_15_cross_dataset_performance",
        "Internal and provisional external "
        "performance without target-domain tuning. "
        "External provenance and near-duplicate "
        "audits remain required."
    )

else:

    print(
        "External evaluation deferred: "
        "no verified compatible attached datasets."
    )

# ============================================================
# STEP 208 — AUTOMATIC PERMANENT GITHUB BACKUP
# ============================================================

banner("STEP 208 — PERMANENT GITHUB BACKUP")

primary = PAIRED_NEW[
    (
        PAIRED_NEW["cohort"]
        == "internal_test"
    )
    & PAIRED_NEW["primary_comparison"]
].iloc[0]

candidate_result = INTERNAL[
    INTERNAL["model"] == "prototype_family"
].iloc[0]

control_result = INTERNAL[
    INTERNAL["model"]
    == "three_member_standard_control"
].iloc[0]

baseline_result = INTERNAL[
    INTERNAL["model"]
    == "locked_banglabert"
].iloc[0]

if primary["ci95_low"] > 0:

    DECISION = (
        "POSITIVE INTERNAL EVIDENCE; "
        "EXTERNAL CONFIRMATION REQUIRED"
    )

elif primary["ci95_high"] < 0:

    DECISION = (
        "STANDARD ENSEMBLE SUPERIOR "
        "ON INTERNAL TEST"
    )

else:

    DECISION = (
        "ARCHITECTURAL SUPERIORITY "
        "NOT ESTABLISHED"
    )

REPORT_NEW = {
    "stage": "9D-10A",
    "candidate_members": PROPOSED_NEW,
    "control_members": CONTROL_NEW,
    "validation_reproduction_passed": True,
    "strict_checkpoint_reloads": len(MODEL_NAMES),
    "train_samples": len(TRAIN_NEW),
    "validation_samples": len(VAL_NEW),
    "test_samples": len(TEST_NEW),
    "internal_test_evaluated": True,
    "candidate_test_macro_f1":
        float(candidate_result["macro_f1"]),
    "candidate_test_accuracy":
        float(candidate_result["accuracy"]),
    "control_test_macro_f1":
        float(control_result["macro_f1"]),
    "banglabert_test_macro_f1":
        float(baseline_result["macro_f1"]),
    "primary_delta_macro_f1":
        float(primary["delta_macro_f1"]),
    "primary_ci95_low":
        float(primary["ci95_low"]),
    "primary_ci95_high":
        float(primary["ci95_high"]),
    "primary_mcnemar_p":
        float(primary["mcnemar_p"]),
    "primary_holm_p":
        float(primary["holm_p"]),
    "external_cohorts_evaluated":
        list(EXTERNAL_NEW.keys()),
    "external_status": (
        "PROVISIONAL_PENDING_PROVENANCE_AND_NEAR_DUPLICATE_AUDIT"
        if EXTERNAL_NEW
        else "DATASETS_NOT_AVAILABLE"
    ),
    "independent_external_confirmation": False,
    "new_figures": len(CAPTIONS_NEW),
    "new_figure_files": len(FIGURE_MANIFEST),
    "bootstrap_replicates": BOOTSTRAPS,
    "test_tuning": False,
    "external_tuning": False,
    "new_neural_training": False,
    "classical_baseline_fitted_on_train_only": True,
    "manual_saving_required": False,
    "google_drive_used": False,
    "paid_api_used": False,
    "novelty_verified": False,
    "state_of_the_art_claim_supported": False,
    "limitations": [
        "Reconstructed test cohort overlaps historical "
        "research data exposure.",
        "Repeated validation-driven architecture exploration.",
        "Standard ensemble is backbone-comparable but "
        "not fully training-matched.",
        "External cohorts require provenance, label mapping "
        "and near-duplicate overlap verification.",
        "No new multi-seed PAS-Net confirmation.",
        "Inference timing is descriptive rather than "
        "a controlled hardware benchmark.",
        "Architectural novelty remains unverified."
    ],
    "decision": DECISION
}

dump_json(
    REPORT_NEW,
    BACKUP_NEW / "final_evaluation_summary.json"
)

dump_json(
    REPORT_NEW,
    AUDIT_NEW / "final_evaluation_summary.json"
)

# Verify every newly generated figure before packaging.

for record in FIGURE_MANIFEST:

    path = REPO / record["path"]

    assert path.exists()

    assert hash_file(path) == record["sha256"]

# Compact publication package.

publication_zip = (
    ROOT / "final_publication_assets.zip"
)

with zipfile.ZipFile(
    publication_zip,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for path in BACKUP_NEW.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(BACKUP_NEW)
            )

with zipfile.ZipFile(publication_zip) as z:

    assert z.testzip() is None

upload_new(publication_zip)

# Private reproducibility package.
# No raw review text or token is included.

private_zip = (
    ROOT / "final_private_evaluation.zip"
)

with zipfile.ZipFile(
    private_zip,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for folder in [
        PRED_NEW,
        TABLES_NEW,
        AUDIT_NEW
    ]:

        for path in folder.rglob("*"):

            if path.is_file():

                z.write(
                    path,
                    path.relative_to(ROOT)
                )

with zipfile.ZipFile(private_zip) as z:

    assert z.testzip() is None

upload_new(private_zip)

manifest_path = (
    AUDIT_NEW / "final_remote_manifest.json"
)

dump_json(
    MANIFEST_NEW,
    manifest_path
)

dump_json(
    MANIFEST_NEW,
    BACKUP_NEW / "final_remote_manifest.json"
)

upload_new(manifest_path)

FINAL_COMMIT_NEW = push_new(
    "Complete consolidated final evaluation "
    "and publication asset backup"
)

# ============================================================
# STEP 209 — FINAL CONSOLIDATED RESEARCH REPORT
# ============================================================

banner("STEP 209 — FINAL CONSOLIDATED RESEARCH REPORT")

print("\nVALIDATION REPRODUCTION")

print(
    VALIDATION_AUDIT.to_string(index=False)
)

print("\nINTERNAL TEST RESULTS")

print(
    INTERNAL.sort_values(
        "macro_f1",
        ascending=False
    ).to_string(index=False)
)

print("\nPRIMARY STATISTICAL COMPARISON")

print(
    PAIRED_NEW[
        PAIRED_NEW["primary_comparison"]
    ].to_string(index=False)
)

print("\nCLASSWISE PERFORMANCE")

print(
    CLASSWISE_NEW[
        (
            CLASSWISE_NEW["cohort"]
            == "internal_test"
        )
        & CLASSWISE_NEW["model"].isin([
            "locked_banglabert",
            "three_member_standard_control",
            "prototype_family"
        ])
    ].to_string(index=False)
)

print("\nLINGUISTIC SUBGROUPS")

print(
    SUBGROUPS_NEW[
        (
            SUBGROUPS_NEW["cohort"]
            == "internal_test"
        )
        & (
            SUBGROUPS_NEW["model"]
            == "prototype_family"
        )
    ].to_string(index=False)
)

print("\nEXTERNAL DATASET AUDIT")

print(
    pd.DataFrame(
        EXTERNAL_AUDIT
    ).to_string(index=False)
)

print("\nMODEL COMPLEXITY")

print(COSTS_NEW.to_string(index=False))

print("\nRESEARCH SUMMARY")

print(json.dumps(
    REPORT_NEW,
    indent=2,
    ensure_ascii=False
))

tracked = [
    REPO / p
    for p in git("ls-files").splitlines()
]

counts = {}

for ext in ["png", "pdf", "svg"]:

    counts[ext] = len({
        p.stem
        for p in tracked
        if p.suffix.lower() == "."+ext
        and p.is_file()
    })

print("\nPERMANENT PUBLICATION ASSETS")

for ext, count in counts.items():

    print(ext.upper(), "figures:", count)

print(
    "Total figure files:",
    sum(counts.values())
)

print(
    "New publication figures:",
    len(CAPTIONS_NEW)
)

print(
    "Remote verified assets:",
    len(MANIFEST_NEW)
)

print("Final public commit:", FINAL_COMMIT_NEW)

print(
    "Private release:",
    RELEASE_NEW["html_url"]
)

print("GitHub synchronized: True")
print("Manual saving required: NO")
print("Google Drive used: NO")
print("Paid API used: NO")

print(
    "Total runtime minutes:",
    round(
        (time.time()-START_ALL)/60,
        2
    )
)

print("\nFINAL RESEARCH DECISION:", DECISION)

banner("CONSOLIDATED EVALUATION COMPLETE")
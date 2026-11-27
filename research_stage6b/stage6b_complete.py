
# =====================================================================
# BANGLA SENTIMENT RESEARCH
# STAGE 6B — COMPETITIVE TRANSFORMER BASELINES
# STEPS 59–65
# =====================================================================

import os
import sys
import gc
import re
import json
import time
import math
import random
import shutil
import hashlib
import getpass
import base64
import zipfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from scipy.special import softmax
from scipy.optimize import minimize_scalar

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)

START = time.time()

WORK = Path("/kaggle/working")

OWNER = "ankita6876"
REPOSITORY = "bangla-sentiment-research"

BRANCH = "research/reconstruction-20261127-080232"

EXPECTED_PARENT = (
    "e8bf19a3fc8f2a793f7eaa99eb6fc916a6e792fc"
)

URL = f"https://github.com/{OWNER}/{REPOSITORY}.git"

REPO = WORK / "sentiment_verified_research_repo"

S6A = WORK / "sentiment_reconstruction_v1"
S6B = WORK / "sentiment_research_stage6b"

PUB = WORK / "sentiment_publication_assets"

TABLES = S6B / "tables"
MODELS = S6B / "checkpoints"
PRED = S6B / "private_predictions"
AUDIT = S6B / "audit"

SEED = 42
MAX_LENGTH = 64
EPOCHS = 4
TRAIN_BATCH = 16
EVAL_BATCH = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

LABELS = ["Negative", "Neutral", "Positive"]

LABEL_MAP = {
    label: i for i, label in enumerate(LABELS)
}

MODEL_SPECS = {
    "BanglaBERT": {
        "hf_id": "csebuetnlp/banglabert",
        "normalization": "banglabert"
    },
    "XLM-RoBERTa": {
        "hf_id": "FacebookAI/xlm-roberta-base",
        "normalization": "standard"
    },
    "mBERT": {
        "hf_id": "google-bert/bert-base-multilingual-cased",
        "normalization": "standard"
    }
}

for folder in [
    TABLES, MODELS, PRED, AUDIT,
    PUB / "figures/png",
    PUB / "figures/pdf",
    PUB / "figures/svg",
    PUB / "tables",
    PUB / "captions",
    PUB / "scripts",
    PUB / "architecture"
]:
    folder.mkdir(parents=True, exist_ok=True)

def section(title):

    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)

def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
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

def seed_everything(seed):

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

assert torch.cuda.is_available(), (
    "Enable GPU T4 in Kaggle settings."
)

DEVICE = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))

# =====================================================================
# STEP 59 — VERIFIED GITHUB CONNECTION
# =====================================================================

section("STEP 59 — GITHUB CONNECTION AND RECOVERY")

TOKEN = getpass.getpass(
    "Enter your working GitHub token: "
).strip()

assert TOKEN, "GitHub token is required."

SESSION = requests.Session()

SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json"
})

response = SESSION.get(
    "https://api.github.com/user",
    timeout=30
)

print("Authentication HTTP:", response.status_code)

assert response.status_code == 200

ACCOUNT = response.json()["login"]

response = SESSION.get(
    f"https://api.github.com/repos/{OWNER}/{REPOSITORY}",
    timeout=30
)

assert response.status_code == 200

assert response.json().get(
    "permissions", {}
).get("push", False)

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

    if result.returncode != 0:

        raise RuntimeError(
            f"Git command failed: {args[0]}"
        )

    return result.stdout.strip()

if not (REPO / ".git").exists():

    result = subprocess.run(
        ["git", "clone", URL, str(REPO)],
        capture_output=True,
        env=GIT_ENV,
        timeout=300
    )

    assert result.returncode == 0

git("fetch", "origin")

assert not git("status", "--porcelain"), (
    "Existing repository has uncommitted changes."
)

git(
    "checkout", "-B", BRANCH,
    f"origin/{BRANCH}"
)

git(
    "merge-base", "--is-ancestor",
    EXPECTED_PARENT, "HEAD"
)

git("config", "user.name", ACCOUNT)

git(
    "config", "user.email",
    f"{ACCOUNT}@users.noreply.github.com"
)

print("Research branch:", BRANCH)
print("Current commit:", git("rev-parse", "HEAD"))

BACKUP = REPO / "research_stage6b"

BACKUP.mkdir(parents=True, exist_ok=True)

PREVIOUS = REPO / "research_reconstruction_v1"

assert PREVIOUS.exists()

def backup_to_github(message):

    git("add", "research_stage6b")

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

    print("GitHub backup verified:", local)

    return local

# =====================================================================
# STEP 60 — RECOVER AND LOCK THE DATASET
# =====================================================================

section("STEP 60 — DATASET INTEGRITY")

DATA_DIR = S6A / "datasets/seed_42"

expected_table = pd.read_csv(
    PREVIOUS / "tables/"
    "reconstruction_02_split_summary.csv"
)

expected_table = expected_table[
    expected_table["seed"] == 42
]

EXPECTED_HASHES = dict(
    zip(
        expected_table["split"],
        expected_table["sha256"]
    )
)

# If the session restarted, reconstruct the dataset
# using the verified source stored on GitHub.
# Execute only the dataset preparation sections.

required_paths = [
    DATA_DIR / "train.csv",
    DATA_DIR / "val.csv"
]

if not all(p.exists() for p in required_paths):

    print("Local splits missing. Recovering source...")

    source_path = (
        PREVIOUS / "src/reconstruction_complete.py"
    )

    assert source_path.exists()

    source = source_path.read_text(
        encoding="utf-8"
    )

    start_marker = (
        '# STEP 55 — VERIFIED RAW DATASET '
        'AND NORMALIZATION'
    )

    end_marker = (
        '# STEP 57 — CLASSICAL BASELINES '
        'ON DEVELOPMENT DATA ONLY'
    )

    start_index = source.find(start_marker)
    end_index = source.find(end_marker)

    assert start_index >= 0
    assert end_index > start_index

    import unicodedata

    from sklearn.feature_extraction.text import (
        TfidfVectorizer
    )

    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import StratifiedGroupKFold

    recovery_namespace = {
        "Path": Path,
        "pd": pd,
        "np": np,
        "re": re,
        "gc": gc,
        "hashlib": hashlib,
        "unicodedata": unicodedata,
        "TfidfVectorizer": TfidfVectorizer,
        "NearestNeighbors": NearestNeighbors,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "INPUT": Path("/kaggle/input"),
        "RAW_EXPECTED": (
            "4580950af06bbfbe5bd79e20957b63a2ed"
            "122af2a6a3e685b10576d3f70241c5"
        ),
        "PRIVATE": S6A / "private",
        "DATASETS": S6A / "datasets",
        "SEEDS": [42, 123, 2026],
        "EXPECTED_SPLITS": EXPECTED_HASHES,
        "section": section,
        "sha256": sha256,
        "save_table": lambda df, name: df
    }

    (S6A / "private").mkdir(
        parents=True, exist_ok=True
    )

    exec(
        source[start_index:end_index],
        recovery_namespace
    )

    del recovery_namespace
    gc.collect()

DATA = {}

for split, expected_n in [
    ("train", 9204),
    ("val", 1840)
]:

    path = DATA_DIR / f"{split}.csv"

    assert path.exists()

    actual_hash = sha256(path)

    assert actual_hash == EXPECTED_HASHES[split], (
        f"{split} checksum mismatch. "
        "Do not train on a different cohort."
    )

    DATA[split] = pd.read_csv(path)

    assert len(DATA[split]) == expected_n

    DATA[split]["label_id"] = (
        DATA[split]["Label"]
        .map(LABEL_MAP)
        .astype(int)
    )

    print(
        split,
        "| rows:", len(DATA[split]),
        "| SHA256: PASS"
    )

assert not (
    set(DATA["train"]["group_id"])
    & set(DATA["val"]["group_id"])
)

assert not (
    set(DATA["train"]["sample_id"])
    & set(DATA["val"]["sample_id"])
)

print("Group integrity: PASS")
print("Test data used for training or selection: NO")

# =====================================================================
# STEP 61 — OFFICIAL NORMALIZATION AND MODEL CONFIGURATION
# =====================================================================

section("STEP 61 — MODEL PREPARATION")

try:

    from normalizer import normalize as bn_normalize

except ImportError:

    subprocess.check_call([
        sys.executable,
        "-m", "pip", "install", "-q",
        "git+https://github.com/csebuetnlp/normalizer"
    ])

    from normalizer import normalize as bn_normalize

for split in DATA:

    DATA[split]["banglabert_text"] = (
        DATA[split]["text_norm"]
        .map(bn_normalize)
    )

    DATA[split]["standard_text"] = (
        DATA[split]["text_norm"]
    )

normalized_overlap = len(
    set(DATA["train"]["banglabert_text"])
    & set(DATA["val"]["banglabert_text"])
)

print(
    "BanglaBERT-normalized train/val overlap:",
    normalized_overlap
)

assert normalized_overlap == 0, (
    "Official normalization created cross-split "
    "overlap. Audit required before training."
)

CONFIG = {
    "dataset_branch": BRANCH,
    "split_hashes": EXPECTED_HASHES,
    "seed": SEED,
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "train_batch": TRAIN_BATCH,
    "eval_batch": EVAL_BATCH,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "models": MODEL_SPECS,
    "selection_metric": "validation macro-F1",
    "test_evaluated": False,
    "training_seeds_per_model": 1,
    "experiment_type": "baseline screening"
}

save_json(
    CONFIG,
    AUDIT / "training_config.json"
)

shutil.copy2(
    AUDIT / "training_config.json",
    BACKUP / "training_config.json"
)

# Preserve complete source before expensive training.

source_saved = False

try:

    history = (
        get_ipython()
        .history_manager
        .input_hist_raw
    )

    candidates = [
        cell for cell in history
        if "STAGE 6B" in cell
        and "COMPETITIVE TRANSFORMER BASELINES" in cell
        and len(cell) > 5000
    ]

    if candidates:

        source = candidates[-1]

        assert not re.search(
            r"(?:ghp_|github_pat_|gho_|ghs_)"
            r"[A-Za-z0-9_]{20,}",
            source
        )

        (AUDIT / "stage6b_complete.py").write_text(
            source,
            encoding="utf-8"
        )

        shutil.copy2(
            AUDIT / "stage6b_complete.py",
            BACKUP / "stage6b_complete.py"
        )

        shutil.copy2(
            AUDIT / "stage6b_complete.py",
            PUB / "scripts/stage6b_complete.py"
        )

        source_saved = True

except Exception as e:

    print("Source recovery warning:", type(e).__name__)

assert source_saved, (
    "Complete source could not be preserved. "
    "Training stopped to prevent another loss."
)

backup_to_github(
    "Lock transformer baseline protocol and source"
)

# =====================================================================
# STEP 62 — TRAINING AND EVALUATION UTILITIES
# =====================================================================

section("STEP 62 — TRAINING UTILITIES")

class SentimentDataset(Dataset):

    def __init__(self, frame, tokenizer, text_col):

        self.labels = (
            frame["label_id"]
            .to_numpy(dtype=np.int64)
        )

        self.encodings = tokenizer(
            frame[text_col].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        item["labels"] = int(self.labels[index])

        return item

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

def calibration_metrics(y, probs, bins=15):

    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = prediction == y

    ece = 0.0

    boundaries = np.linspace(
        0, 1, bins + 1
    )

    for i in range(bins):

        mask = (
            (confidence >= boundaries[i])
            & (
                confidence <= boundaries[i+1]
                if i == bins - 1
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

    onehot = np.eye(3)[y]

    return {
        "ece": float(ece),

        "nll": float(
            -np.log(
                probs[np.arange(len(y)), y]
                .clip(1e-12, 1)
            ).mean()
        ),

        "brier": float(
            np.mean(
                np.sum(
                    (probs - onehot)**2,
                    axis=1
                )
            )
        ),

        "mean_confidence": float(
            confidence.mean()
        )
    }

def fit_temperature(logits, y):

    def objective(log_t):

        t = np.exp(log_t)

        p = softmax(
            logits / t,
            axis=1
        )

        return -np.log(
            p[np.arange(len(y)), y]
            .clip(1e-12, 1)
        ).mean()

    result = minimize_scalar(
        objective,
        bounds=(-3, 3),
        method="bounded"
    )

    return float(np.exp(result.x))

@torch.inference_mode()
def evaluate(model, loader):

    model.eval()

    logits = []
    labels = []

    for batch in loader:

        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            output = model(**batch)

        logits.append(
            output.logits.float().cpu().numpy()
        )

        labels.append(
            batch["labels"].cpu().numpy()
        )

    return (
        np.concatenate(logits),
        np.concatenate(labels)
    )

def export_aggregate_tables():

    target = BACKUP / "tables"

    target.mkdir(
        parents=True, exist_ok=True
    )

    for path in TABLES.glob("*.csv"):

        columns = pd.read_csv(
            path, nrows=0
        ).columns

        forbidden = {
            "bangla", "text", "text_norm",
            "text_exact", "sample_id",
            "raw_row_id", "comment"
        }

        assert not (
            set(c.lower() for c in columns)
            & forbidden
        )

        shutil.copy2(
            path,
            target / path.name
        )

# =====================================================================
# STEP 63 — TRAIN THREE COMPETITIVE TRANSFORMERS
# =====================================================================

section("STEP 63 — TRANSFORMER TRAINING")

RESULTS = []
HISTORY = []
CLASSWISE = []
CALIBRATION = []
SUBGROUPS = []
TOKEN_AUDIT = []

Y_VAL = DATA["val"]["label_id"].to_numpy()

NEG_WORDS = [
    "না", "নয়", "নয়", "নেই", "নাই"
]

CON_WORDS = [
    "কিন্তু", "তবে", "যদিও",
    "অথচ", "তবুও", "তথাপি"
]

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

val_text = DATA["val"]["text_norm"]

neg_mask = cue_mask(val_text, NEG_WORDS)
con_mask = cue_mask(val_text, CON_WORDS)

long_threshold = float(
    DATA["train"]["word_count"].quantile(0.75)
)

long_mask = (
    DATA["val"]["word_count"].to_numpy()
    > long_threshold
)

MASKS = {
    "all": np.ones(len(Y_VAL), dtype=bool),
    "negation_candidate": neg_mask,
    "contrast_candidate": con_mask,
    "both_operators": neg_mask & con_mask,
    "any_operator": neg_mask | con_mask,
    "no_detected_operator": ~(neg_mask | con_mask),
    "long_text": long_mask,
    "shorter_text": ~long_mask
}

def flush_tables():

    save_table(
        pd.DataFrame(RESULTS),
        "stage6b_01_model_results"
    )

    save_table(
        pd.DataFrame(HISTORY),
        "stage6b_02_training_history"
    )

    save_table(
        pd.DataFrame(CLASSWISE),
        "stage6b_03_classwise_results"
    )

    save_table(
        pd.DataFrame(CALIBRATION),
        "stage6b_04_calibration"
    )

    save_table(
        pd.DataFrame(SUBGROUPS),
        "stage6b_05_linguistic_subgroups"
    )

    save_table(
        pd.DataFrame(TOKEN_AUDIT),
        "stage6b_06_tokenization_audit"
    )

for model_name, spec in MODEL_SPECS.items():

    section(f"TRAINING — {model_name}")

    seed_everything(SEED)

    model_dir = MODELS / model_name

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = model_dir / "model.pt"

    completion_path = model_dir / "completed.json"

    logits_path = PRED / f"{model_name}_val.npz"

    run_config = {
        **CONFIG,
        "model_name": model_name,
        "hf_id": spec["hf_id"]
    }

    resumed = False

    if (
        completion_path.exists()
        and checkpoint.exists()
        and logits_path.exists()
    ):

        completed = json.loads(
            completion_path.read_text()
        )

        if (
            completed.get("config") == run_config
            and completed.get("checkpoint_sha256")
                == sha256(checkpoint)
            and completed.get("prediction_sha256")
                == sha256(logits_path)
        ):

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

            val_logits = saved["logits"]

            result = completed["result"]

            HISTORY.extend(
                completed["history"]
            )

            TOKEN_AUDIT.extend(
                completed["token_audit"]
            )

            resumed = True

            print(
                "Verified completed run recovered."
            )

    if not resumed:

        tokenizer = AutoTokenizer.from_pretrained(
            spec["hf_id"]
        )

        text_col = (
            "banglabert_text"
            if spec["normalization"] == "banglabert"
            else "standard_text"
        )

        model_token_audit = []

        for split in ["train", "val"]:

            lengths = []

            texts = DATA[split][text_col].tolist()

            for start in range(0, len(texts), 256):

                encoded = tokenizer(
                    texts[start:start+256],
                    truncation=False,
                    padding=False,
                    verbose=False
                )

                lengths.extend(
                    len(ids)
                    for ids in encoded["input_ids"]
                )

            lengths = np.array(lengths)

            model_token_audit.append({
                "model": model_name,
                "split": split,
                "mean_tokens": lengths.mean(),
                "p95_tokens": np.percentile(lengths, 95),
                "p99_tokens": np.percentile(lengths, 99),
                "truncated_count": int(
                    (lengths > MAX_LENGTH).sum()
                ),
                "truncated_fraction": float(
                    (lengths > MAX_LENGTH).mean()
                )
            })

        TOKEN_AUDIT.extend(model_token_audit)

        train_ds = SentimentDataset(
            DATA["train"],
            tokenizer,
            text_col
        )

        val_ds = SentimentDataset(
            DATA["val"],
            tokenizer,
            text_col
        )

        collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )

        generator = torch.Generator()
        generator.manual_seed(SEED)

        train_loader = DataLoader(
            train_ds,
            batch_size=TRAIN_BATCH,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=EVAL_BATCH,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                spec["hf_id"],
                num_labels=3,
                id2label={
                    i: label
                    for i, label in enumerate(LABELS)
                },
                label2id=LABEL_MAP
            )
            .to(DEVICE)
        )

        parameter_count = sum(
            p.numel()
            for p in model.parameters()
        )

        print("Parameters:", parameter_count)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        total_steps = (
            len(train_loader) * EPOCHS
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                total_steps * WARMUP_RATIO
            ),
            num_training_steps=total_steps
        )

        scaler = torch.amp.GradScaler("cuda")

        best_f1 = -1.0
        best_epoch = None
        model_history = []

        t0 = time.time()

        for epoch in range(1, EPOCHS + 1):

            model.train()

            total_loss = 0.0
            total_examples = 0

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

                    output = model(**batch)

                    loss = output.loss

                assert torch.isfinite(loss)

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                previous_scale = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()

                if scaler.get_scale() >= previous_scale:

                    scheduler.step()

                n = batch["labels"].shape[0]

                total_loss += loss.item() * n
                total_examples += n

            val_logits, val_y = evaluate(
                model, val_loader
            )

            assert np.array_equal(
                val_y, Y_VAL
            )

            scores = metrics(
                val_y, val_logits
            )

            epoch_result = {
                "model": model_name,
                "seed": SEED,
                "epoch": epoch,
                "train_loss":
                    total_loss / total_examples,
                **scores
            }

            model_history.append(epoch_result)
            HISTORY.append(epoch_result)

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Loss {epoch_result['train_loss']:.4f} | "
                f"Val Acc {scores['accuracy']:.4f} | "
                f"Val F1 {scores['macro_f1']:.4f}"
            )

            if scores["macro_f1"] > best_f1:

                best_f1 = scores["macro_f1"]
                best_epoch = epoch

                torch.save(
                    {
                        k: v.detach().cpu()
                        for k, v in
                        model.state_dict().items()
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
                    logits=val_logits
                )

                model.config.save_pretrained(
                    model_dir
                )

                tokenizer.save_pretrained(
                    model_dir
                )

                print("Best checkpoint saved.")

            save_table(
                pd.DataFrame(HISTORY),
                "stage6b_02_training_history"
            )

        saved = np.load(
            logits_path,
            allow_pickle=False
        )

        val_logits = saved["logits"]

        result = {
            "model": model_name,
            "hf_id": spec["hf_id"],
            "seed": SEED,
            "best_epoch": best_epoch,
            "parameters": parameter_count,
            "n_validation": len(Y_VAL),
            **metrics(Y_VAL, val_logits),
            "runtime_minutes": (
                time.time() - t0
            ) / 60
        }

        save_json({
            "config": run_config,
            "result": result,
            "history": model_history,
            "token_audit": model_token_audit,
            "checkpoint_sha256": sha256(checkpoint),
            "prediction_sha256": sha256(logits_path)
        }, completion_path)

        del model, optimizer, scheduler
        del scaler, train_loader, val_loader
        del train_ds, val_ds, tokenizer

        gc.collect()
        torch.cuda.empty_cache()

    RESULTS.append(result)

    pred = val_logits.argmax(axis=1)

    # Per-class metrics.

    for i, label in enumerate(LABELS):

        CLASSWISE.append({
            "model": model_name,
            "class": label,
            "support": int((Y_VAL == i).sum()),
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

    # Validation calibration diagnostics.
    # Temperature is fitted and assessed on validation.
    # These are not independent calibration estimates.

    temperature = fit_temperature(
        val_logits, Y_VAL
    )

    for variant, probs in [
        (
            "uncalibrated",
            softmax(val_logits, axis=1)
        ),
        (
            "validation_temperature_scaled",
            softmax(
                val_logits / temperature,
                axis=1
            )
        )
    ]:

        CALIBRATION.append({
            "model": model_name,
            "variant": variant,
            "temperature": (
                temperature
                if variant != "uncalibrated"
                else 1.0
            ),
            **calibration_metrics(Y_VAL, probs)
        })

    # Linguistic subgroup diagnostics.

    for subgroup, mask in MASKS.items():

        if not mask.any():
            continue

        SUBGROUPS.append({
            "model": model_name,
            "subgroup": subgroup,
            "n": int(mask.sum()),
            **metrics(
                Y_VAL[mask],
                val_logits[mask]
            )
        })

    flush_tables()

    export_aggregate_tables()

    # Back up checkpoint metadata, not large weights.

    metadata_dir = (
        BACKUP / "checkpoint_metadata"
        / model_name
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        completion_path,
        metadata_dir / "completed.json"
    )

    shutil.copy2(
        model_dir / "config.json",
        metadata_dir / "config.json"
    )

    backup_to_github(
        f"Preserve verified {model_name} "
        "baseline results and checkpoint provenance"
    )

    print(
        model_name,
        "| Best validation F1:",
        round(result["macro_f1"], 6)
    )

# =====================================================================
# STEP 64 — COMPARATIVE AND LINGUISTIC ANALYSIS
# =====================================================================

section("STEP 64 — BASELINE COMPARISON")

results_df = pd.DataFrame(
    RESULTS
).sort_values(
    "macro_f1",
    ascending=False
)

history_df = pd.DataFrame(HISTORY)

class_df = pd.DataFrame(CLASSWISE)

cal_df = pd.DataFrame(CALIBRATION)

subgroup_df = pd.DataFrame(SUBGROUPS)

classical = pd.read_csv(
    PREVIOUS / "tables/"
    "reconstruction_05_baseline_results.csv"
)

classical = classical[
    classical["seed"] == SEED
].copy()

comparison = pd.concat([
    results_df[
        ["model", "accuracy", "macro_f1"]
    ].assign(family="Transformer"),

    classical[
        ["model", "accuracy", "macro_f1"]
    ].assign(family="Classical")

], ignore_index=True).sort_values(
    "macro_f1",
    ascending=False
)

save_table(
    comparison,
    "stage6b_07_all_baselines"
)

BEST_MODEL = results_df.iloc[0]["model"]

BEST_F1 = float(
    results_df.iloc[0]["macro_f1"]
)

print("\nAll seed-42 validation baselines:")

print(
    comparison.round(4)
    .to_string(index=False)
)

print("\nStrongest transformer:", BEST_MODEL)

print("Strongest validation macro-F1:", BEST_F1)

print("\nLinguistic subgroup results:")

print(
    subgroup_df.pivot(
        index="subgroup",
        columns="model",
        values="macro_f1"
    ).round(4).to_string()
)

# =====================================================================
# STEP 65 — PUBLICATION FIGURES AND FINAL BACKUP
# =====================================================================

section("STEP 65 — PUBLICATION ASSETS")

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

        path = (
            PUB / "figures" / ext
            / f"{name}.{ext}"
        )

        fig.savefig(
            path,
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
        "stage": "6b"
    })

    plt.close(fig)

# Figure 1 — Overall baseline comparison.

plot_df = comparison.sort_values(
    "macro_f1"
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.barh(
    plot_df["model"],
    plot_df["macro_f1"],
    color=[
        "#237a68"
        if x == "Transformer"
        else "#8ba4ba"
        for x in plot_df["family"]
    ]
)

ax.bar_label(
    bars,
    fmt="%.3f",
    padding=4
)

ax.set_xlim(0, 1.05)
ax.set_xlabel("Validation macro-F1")
ax.set_title("Competitive sentiment classification baselines")

save_figure(
    fig,
    "stage6b_01_baseline_comparison",
    "Classical and transformer baseline comparison "
    "on the locked seed-42 validation cohort.",
    "stage6b_07_all_baselines.csv"
)

# Figure 2 — Training convergence.

fig, ax = plt.subplots(figsize=(7, 4.8))

for name, group in history_df.groupby("model"):

    ax.plot(
        group["epoch"],
        group["macro_f1"],
        marker="o",
        label=name
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation macro-F1")
ax.set_xticks(range(1, EPOCHS + 1))
ax.set_title("Transformer training convergence")
ax.legend(frameon=False)

save_figure(
    fig,
    "stage6b_02_training_convergence",
    "Validation macro-F1 throughout transformer fine-tuning.",
    "stage6b_02_training_history.csv"
)

# Figure 3 — Classwise performance.

plot_df = class_df.pivot(
    index="model",
    columns="class",
    values="f1"
).reindex(columns=LABELS)

fig, ax = plt.subplots(figsize=(8, 5))

plot_df.plot(
    kind="bar",
    ax=ax,
    color=["#bd6262", "#94a3b8", "#438f79"]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Class-specific F1")
ax.set_xlabel("")
ax.set_title("Transformer performance by sentiment class")
ax.tick_params(axis="x", rotation=0)
ax.legend(frameon=False)

save_figure(
    fig,
    "stage6b_03_classwise_performance",
    "Class-specific validation F1 for the three transformers.",
    "stage6b_03_classwise_results.csv"
)

# Figure 4 — Linguistic robustness.

plot_df = subgroup_df.pivot(
    index="subgroup",
    columns="model",
    values="macro_f1"
)

fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(
    plot_df.to_numpy(),
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    aspect="auto"
)

for i in range(len(plot_df.index)):

    for j in range(len(plot_df.columns)):

        value = plot_df.iloc[i, j]

        ax.text(
            j, i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=(
                "white"
                if value > 0.65
                else "black"
            )
        )

ax.set_xticks(
    range(len(plot_df.columns)),
    plot_df.columns,
    rotation=15
)

ax.set_yticks(
    range(len(plot_df.index)),
    plot_df.index
)

ax.set_title("Linguistic subgroup performance")

fig.colorbar(
    image, ax=ax,
    label="Macro-F1"
)

save_figure(
    fig,
    "stage6b_04_linguistic_robustness",
    "Exploratory performance on heuristic linguistic "
    "subgroups. Cue matches are not gold annotations.",
    "stage6b_05_linguistic_subgroups.csv"
)

# Figure 5 — Calibration.

plot_df = cal_df.pivot(
    index="model",
    columns="variant",
    values="ece"
)

fig, ax = plt.subplots(figsize=(8, 4.8))

plot_df.plot(
    kind="bar",
    ax=ax,
    color=["#237a68", "#94a3b8"]
)

ax.set_ylabel("Expected calibration error")
ax.set_xlabel("")
ax.set_title("Development-set calibration diagnostics")
ax.tick_params(axis="x", rotation=0)
ax.legend(frameon=False, fontsize=9)

save_figure(
    fig,
    "stage6b_05_calibration",
    "Validation ECE before and after temperature scaling. "
    "Temperature fitting and assessment use the same "
    "validation cohort; results are descriptive.",
    "stage6b_04_calibration.csv"
)

# Figure 6 — Strongest transformer confusion matrix.

saved = np.load(
    PRED / f"{BEST_MODEL}_val.npz",
    allow_pickle=False
)

best_pred = saved["logits"].argmax(axis=1)

cm = confusion_matrix(
    Y_VAL,
    best_pred,
    labels=[0, 1, 2]
)

cm_df = pd.DataFrame(
    cm,
    columns=LABELS
)

cm_df.insert(
    0, "true_label", LABELS
)

save_table(
    cm_df,
    "stage6b_08_best_confusion_matrix"
)

fig, ax = plt.subplots(figsize=(5.8, 5))

image = ax.imshow(
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
ax.set_title(f"{BEST_MODEL}: confusion matrix")

fig.colorbar(image, ax=ax)

save_figure(
    fig,
    "stage6b_06_confusion_matrix",
    "Validation confusion matrix of the strongest "
    "screened transformer baseline.",
    "stage6b_08_best_confusion_matrix.csv"
)

# Figure manifest.

manifest_path = PUB / "figure_manifest.csv"

new_manifest = pd.DataFrame(FIGURES)

if manifest_path.exists():

    old_manifest = pd.read_csv(manifest_path)

else:

    previous_manifest = (
        PREVIOUS / "publication_assets"
        / "figure_manifest.csv"
    )

    old_manifest = (
        pd.read_csv(previous_manifest)
        if previous_manifest.exists()
        else pd.DataFrame()
    )

# Restore previous publication figures from GitHub
# when the current session does not contain them.

previous_pub = PREVIOUS / "publication_assets"

if previous_pub.exists():

    for path in previous_pub.rglob("*"):

        if not path.is_file():
            continue

        if path.name == "figure_manifest.csv":
            continue

        target = PUB / path.relative_to(previous_pub)

        if not target.exists():

            target.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(path, target)

manifest = pd.concat(
    [old_manifest, new_manifest],
    ignore_index=True
).drop_duplicates(
    "figure_id",
    keep="last"
)

manifest.to_csv(
    manifest_path,
    index=False
)

# Copy new publication assets to GitHub.

for record in FIGURES:

    for ext in ["png", "pdf", "svg"]:

        relative = Path(record[ext])

        target = (
            BACKUP / "publication_assets"
            / relative
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            PUB / relative,
            target
        )

    caption = record["figure_id"] + ".txt"

    target = (
        BACKUP / "publication_assets"
        / "captions" / caption
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        PUB / "captions" / caption,
        target
    )

shutil.copy2(
    manifest_path,
    BACKUP / "figure_manifest.csv"
)

export_aggregate_tables()

SUMMARY = {
    "stage": "6b",
    "strongest_transformer": BEST_MODEL,
    "strongest_validation_macro_f1": BEST_F1,
    "models_completed": len(RESULTS),
    "training_seeds_per_model": 1,
    "locked_split_hashes": EXPECTED_HASHES,
    "results": RESULTS,
    "test_evaluated": False,
    "external_evaluation": False,
    "proposed_architecture_trained": False,
    "novelty_verified": False,
    "publication_figures_added": len(FIGURES),
    "checkpoint_weights_uploaded_to_github": False
}

save_json(
    SUMMARY,
    AUDIT / "stage6b_summary.json"
)

shutil.copy2(
    AUDIT / "stage6b_summary.json",
    BACKUP / "stage6b_summary.json"
)

# Compact archive excluding large checkpoints.

COMPACT_ZIP = (
    WORK / "sentiment_stage6b_compact.zip"
)

with zipfile.ZipFile(
    COMPACT_ZIP,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for path in S6B.rglob("*"):

        if not path.is_file():
            continue

        if path.name == "model.pt":
            continue

        z.write(
            path,
            path.relative_to(S6B)
        )

with zipfile.ZipFile(COMPACT_ZIP) as z:

    assert z.testzip() is None

PUBLICATION_ZIP = (
    WORK / "sentiment_publication_assets.zip"
)

with zipfile.ZipFile(
    PUBLICATION_ZIP,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for path in PUB.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(PUB)
            )

with zipfile.ZipFile(PUBLICATION_ZIP) as z:

    assert z.testzip() is None

FINAL_COMMIT = backup_to_github(
    "Complete competitive transformer screening "
    "and publication-ready baseline analysis"
)

# =====================================================================
# FINAL REPORT
# =====================================================================

section("FINAL STAGE 6B RESEARCH REPORT")

print("GitHub authentication: PASS")
print("GitHub synchronized: True")
print("Final verified commit:", FINAL_COMMIT)

print("\nDATASET")

print("Training samples:", len(DATA["train"]))
print("Validation samples:", len(DATA["val"]))
print("Dataset checksum verification: PASS")
print("Constructed group overlap: ZERO")

print("\nMODEL COMPARISON")

print(
    comparison.round(4)
    .to_string(index=False)
)

print("\nStrongest transformer:", BEST_MODEL)
print("Best validation macro-F1:", BEST_F1)

print("\nLINGUISTIC SUBGROUPS")

print(
    subgroup_df.pivot(
        index="subgroup",
        columns="model",
        values="macro_f1"
    ).round(4).to_string()
)

print("\nSUBGROUP SAMPLE COUNTS")

for name, mask in MASKS.items():

    print(name, ":", int(mask.sum()))

print("\nCALIBRATION")

print(
    cal_df.round(4)
    .to_string(index=False)
)

print("\nTOKENIZATION AUDIT")

print(
    pd.DataFrame(TOKEN_AUDIT)
    .round(4)
    .to_string(index=False)
)

print("\nCHECKPOINTS")

for path in MODELS.rglob("model.pt"):

    print(
        path,
        "| MB:",
        round(path.stat().st_size / 1024**2, 2)
    )

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

print("Compact archive:", COMPACT_ZIP)
print("Publication archive:", PUBLICATION_ZIP)
print("Archive integrity: PASS")

print("\nTest evaluated: NO")
print("External evaluation: NO")
print("Proposed model trained: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")

print(
    "\nRuntime minutes:",
    round((time.time()-START)/60, 2)
)

print(
    "\nREADY FOR STAGE 6C — "
    "PROPOSED NOVEL ARCHITECTURE"
)

SESSION.headers.pop("Authorization", None)
SESSION.close()

GIT_ENV.clear()

TOKEN = None
credential = None

gc.collect()
torch.cuda.empty_cache()
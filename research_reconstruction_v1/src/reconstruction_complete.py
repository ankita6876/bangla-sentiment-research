
# =====================================================================
# BANGLA SENTIMENT RESEARCH
# STAGE 6A — TARGETED RECONSTRUCTION AND VERIFIED GITHUB BACKUP
# STEPS 54–58
# =====================================================================

import os
import re
import gc
import json
import time
import random
import hashlib
import shutil
import zipfile
import getpass
import base64
import subprocess
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)

START = time.time()

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

OWNER = "ankita6876"
REPOSITORY = "bangla-sentiment-research"

URL = f"https://github.com/{OWNER}/{REPOSITORY}.git"
API = f"https://api.github.com/repos/{OWNER}/{REPOSITORY}"

REPO = WORK / "sentiment_verified_research_repo"

OUT = WORK / "sentiment_reconstruction_v1"

PUB = WORK / "sentiment_publication_assets"

DATASETS = OUT / "datasets"
TABLES = OUT / "tables"
AUDIT = OUT / "audit"
PRIVATE = OUT / "private"

SEEDS = [42, 123, 2026]

LABELS = ["Negative", "Neutral", "Positive"]

LABEL_MAP = {
    label: i for i, label in enumerate(LABELS)
}

RAW_EXPECTED = (
    "4580950af06bbfbe5bd79e20957b63a2ed"
    "122af2a6a3e685b10576d3f70241c5"
)

EXPECTED_SPLITS = {
    "train": (
        "1df52c4ac4396422129237bb1ecdd28d"
        "be63929aec0327ce9537eb22e6e0e9b7"
    ),
    "val": (
        "c70cc4f571a8a92ffb4dd4cf3db53c9ce"
        "6a8c754ebd64d1047b9fb07b24badf0"
    ),
    "test": (
        "b9b1ac03e873e38304eef8aebac4f2972"
        "f33abe3f3488d3029258f4c74e8cf5f"
    )
}

for folder in [
    DATASETS, TABLES, AUDIT, PRIVATE,
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

def save_table(df, name):
    df.to_csv(TABLES / f"{name}.csv", index=False)

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

FIGURE_RECORDS = []

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

def save_figure(fig, name, caption, source):

    fig.tight_layout()

    for ext in ["png", "pdf", "svg"]:

        destination = (
            PUB / "figures" / ext /
            f"{name}.{ext}"
        )

        fig.savefig(
            destination,
            dpi=600 if ext == "png" else 300,
            bbox_inches="tight"
        )

    (PUB / "captions" / f"{name}.txt").write_text(
        caption,
        encoding="utf-8"
    )

    FIGURE_RECORDS.append({
        "figure_id": name,
        "png": f"figures/png/{name}.png",
        "pdf": f"figures/pdf/{name}.pdf",
        "svg": f"figures/svg/{name}.svg",
        "caption": caption,
        "source_table": source,
        "stage": "reconstruction_v1"
    })

    plt.close(fig)

# =====================================================================
# STEP 54 — AUTHENTICATION AND ACTUAL GITHUB WRITE VERIFICATION
# =====================================================================

section("STEP 54 — GITHUB AUTHENTICATION")

TOKEN = getpass.getpass(
    "Enter your regenerated GitHub token: "
).strip()

assert TOKEN, "GitHub token is required."

SESSION = requests.Session()

SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

response = SESSION.get(
    "https://api.github.com/user",
    timeout=30
)

print("Authentication HTTP:", response.status_code)

assert response.status_code == 200, (
    "Authentication failed. No experiments started."
)

ACCOUNT = response.json()["login"]

response = SESSION.get(API, timeout=30)

assert response.status_code == 200

REPO_INFO = response.json()

assert REPO_INFO.get(
    "permissions", {}
).get("push", False), (
    "Repository write permission is missing."
)

print("Authenticated account:", ACCOUNT)
print("Repository:", REPO_INFO["full_name"])
print("Write permission: PASS")

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
        text=True,
        env=GIT_ENV,
        timeout=300
    )

    assert result.returncode == 0, (
        "GitHub clone failed."
    )

assert not git("status", "--porcelain"), (
    "Uncommitted repository changes detected."
)

git("fetch", "origin")

BRANCH = (
    "research/reconstruction-"
    + time.strftime("%Y%m%d-%H%M%S")
)

DEFAULT_BRANCH = REPO_INFO["default_branch"]

git(
    "checkout", "-b", BRANCH,
    f"origin/{DEFAULT_BRANCH}"
)

git("config", "user.name", ACCOUNT)

git(
    "config", "user.email",
    f"{ACCOUNT}@users.noreply.github.com"
)

BACKUP = REPO / "research_reconstruction_v1"

BACKUP.mkdir(parents=True, exist_ok=True)

save_json({
    "project": "Bangla sentiment research",
    "authentication_verified": True,
    "branch": BRANCH,
    "raw_dataset_expected_sha256": RAW_EXPECTED,
    "purpose": "Reconstruct essential research artifacts",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
}, BACKUP / "recovery_start.json")

git(
    "add",
    "research_reconstruction_v1/recovery_start.json"
)

git(
    "commit", "-m",
    "Verify GitHub access before research reconstruction"
)

git("push", "-u", "origin", BRANCH)

def verify_remote():

    local = git("rev-parse", "HEAD")

    remote = git(
        "ls-remote",
        "origin",
        f"refs/heads/{BRANCH}"
    )

    remote = remote.split()[0] if remote else None

    assert local == remote, (
        "Remote synchronization failed."
    )

    return local

INITIAL_COMMIT = verify_remote()

print("Actual GitHub push: PASS")
print("Remote synchronization: PASS")
print("Research branch:", BRANCH)
print("Verified commit:", INITIAL_COMMIT)

# =====================================================================
# STEP 55 — VERIFIED RAW DATASET AND NORMALIZATION
# =====================================================================

section("STEP 55 — DATASET RECONSTRUCTION")

RAW = INPUT / (
    "datasets/ankitaroy98/"
    "becs-dataset/BECS Dataset.csv"
)

if not RAW.exists():

    candidates = list(
        INPUT.rglob("BECS Dataset.csv")
    )

    RAW = next(
        (
            p for p in candidates
            if sha256(p) == RAW_EXPECTED
        ),
        None
    )

assert RAW is not None and RAW.exists()

assert sha256(RAW) == RAW_EXPECTED, (
    "Raw dataset checksum mismatch."
)

raw = pd.read_csv(RAW)

print("Raw dataset SHA256: PASS")
print("Raw dataset shape:", raw.shape)
print("Columns:", raw.columns.tolist())

def normalize_column_name(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower()
    )

column_lookup = {
    normalize_column_name(c): c
    for c in raw.columns
}

text_col = next(
    (
        column_lookup[c]
        for c in [
            "bangla", "comment", "text",
            "review", "sentence"
        ]
        if c in column_lookup
    ),
    None
)

label_col = next(
    (
        column_lookup[c]
        for c in [
            "label", "labelsentiment",
            "sentiment", "sentimentlabel"
        ]
        if c in column_lookup
    ),
    None
)

assert text_col is not None
assert label_col is not None

print("Text column:", text_col)
print("Label column:", label_col)

def normalize_text(text):

    text = unicodedata.normalize(
        "NFC", str(text)
    )

    text = re.sub(
        r"[\u200b\u200c\u200d\u2060\ufeff]",
        "",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip().casefold()

def normalize_label(value):

    value = str(value).strip().casefold()

    mapping = {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
        "neg": "Negative",
        "neu": "Neutral",
        "pos": "Positive"
    }

    return mapping.get(value)

df = pd.DataFrame({
    "Bangla": raw[text_col],
    "Label": raw[label_col].map(normalize_label),
    "raw_row_id": np.arange(len(raw))
})

df = df.dropna(
    subset=["Bangla", "Label"]
).copy()

df["Bangla"] = df["Bangla"].astype(str)

df["text_exact"] = df["Bangla"].str.strip()

df["text_norm"] = df["Bangla"].map(
    normalize_text
)

df = df[df["text_norm"].str.len() > 0].copy()

VALID_ROWS = len(df)

conflict_counts = (
    df.groupby("text_norm")["Label"]
    .nunique()
)

conflicting = set(
    conflict_counts[
        conflict_counts > 1
    ].index
)

conflict_mask = df["text_norm"].isin(
    conflicting
)

CONFLICT_GROUPS = len(conflicting)
CONFLICT_ROWS = int(conflict_mask.sum())

df.loc[conflict_mask].to_csv(
    PRIVATE / "conflicting_label_records.csv",
    index=False
)

df = df.loc[~conflict_mask].copy()

AFTER_CONFLICT_REMOVAL = len(df)

df = df.drop_duplicates(
    "text_norm", keep="first"
).reset_index(drop=True)

df["sample_id"] = df["text_norm"].map(
    lambda t: hashlib.sha256(
        t.encode("utf-8")
    ).hexdigest()
)

df["word_count"] = (
    df["text_norm"].str.split().str.len()
)

NEGATION = [
    "না", "নয়", "নয়", "নেই", "নাই",
    "কখনোই", "কখনওই"
]

CONTRAST = [
    "কিন্তু", "তবে", "যদিও",
    "অথচ", "তবুও", "তথাপি"
]

def cue_pattern(words):

    return re.compile(
        r"(?<![\w\u0980-\u09ff])(?:"
        + "|".join(map(re.escape, words))
        + r")(?![\w\u0980-\u09ff])"
    )

NEG_PATTERN = cue_pattern(NEGATION)
CON_PATTERN = cue_pattern(CONTRAST)

df["contains_negation"] = (
    df["text_norm"].map(
        lambda x: bool(NEG_PATTERN.search(x))
    )
)

df["contains_contrast"] = (
    df["text_norm"].map(
        lambda x: bool(CON_PATTERN.search(x))
    )
)

df["contains_latin"] = (
    df["text_norm"].str.contains(
        r"[A-Za-z]", regex=True
    )
)

df["contains_emoji"] = (
    df["text_norm"].str.contains(
        r"[\U0001F300-\U0001FAFF]",
        regex=True
    )
)

print("Valid rows:", VALID_ROWS)
print("Conflicting normalized groups:", CONFLICT_GROUPS)
print("Conflicting rows excluded:", CONFLICT_ROWS)
print("After conflict removal:", AFTER_CONFLICT_REMOVAL)
print("Final normalized unique rows:", len(df))

dataset_audit = pd.DataFrame([
    {"stage": "Raw dataset", "rows": len(raw)},
    {"stage": "Valid records", "rows": VALID_ROWS},
    {
        "stage": "After conflict exclusion",
        "rows": AFTER_CONFLICT_REMOVAL
    },
    {
        "stage": "After normalized deduplication",
        "rows": len(df)
    }
])

save_table(
    dataset_audit,
    "reconstruction_01_dataset_audit"
)

# =====================================================================
# STEP 56 — NEAR-DUPLICATE GROUPS AND LEAKAGE-CONTROLLED SPLITS
# =====================================================================

section("STEP 56 — GROUPED DATASET SPLITS")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000,
    dtype=np.float32
)

X = vectorizer.fit_transform(
    df["text_norm"]
)

nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    radius=0.1000001,
    n_jobs=-1
)

nn.fit(X)

parent = np.arange(len(df))

def find_root(x):

    while parent[x] != x:

        parent[x] = parent[parent[x]]
        x = parent[x]

    return x

def union(a, b):

    ra = find_root(a)
    rb = find_root(b)

    if ra != rb:

        parent[max(ra, rb)] = min(ra, rb)

edges = []

for start in range(0, len(df), 256):

    stop = min(start + 256, len(df))

    distances, indices = nn.radius_neighbors(
        X[start:stop],
        return_distance=True
    )

    for local_i, (ds, js) in enumerate(
        zip(distances, indices)
    ):

        i = start + local_i

        for distance, j in zip(ds, js):

            j = int(j)

            if j <= i:
                continue

            similarity = 1 - float(distance)

            if similarity >= 0.90 - 1e-7:

                union(i, j)

                edges.append({
                    "sample_a": df.iloc[i]["sample_id"],
                    "sample_b": df.iloc[j]["sample_id"],
                    "similarity": similarity
                })

df["group_id"] = [
    find_root(i) for i in range(len(df))
]

group_sizes = df.groupby(
    "group_id"
).size()

print("Near-duplicate edges:", len(edges))
print("Unique groups:", len(group_sizes))
print("Multi-sample groups:", int((group_sizes > 1).sum()))
print("Largest group:", int(group_sizes.max()))

pd.DataFrame(edges).to_csv(
    PRIVATE / "near_duplicate_edges.csv",
    index=False
)

del X, nn, vectorizer
gc.collect()

split_rows = []
integrity_rows = []
SPLITS = {}

for seed in SEEDS:

    outer = StratifiedGroupKFold(
        n_splits=7,
        shuffle=True,
        random_state=seed
    )

    development_idx, test_idx = next(
        outer.split(
            df,
            df["Label"],
            df["group_id"]
        )
    )

    development = df.iloc[
        development_idx
    ].reset_index(drop=True)

    inner = StratifiedGroupKFold(
        n_splits=6,
        shuffle=True,
        random_state=seed
    )

    train_idx, val_idx = next(
        inner.split(
            development,
            development["Label"],
            development["group_id"]
        )
    )

    splits = {
        "train": development.iloc[
            train_idx
        ].reset_index(drop=True),

        "val": development.iloc[
            val_idx
        ].reset_index(drop=True),

        "test": df.iloc[
            test_idx
        ].reset_index(drop=True)
    }

    SPLITS[seed] = splits

    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test")
    ]:

        group_overlap = len(
            set(splits[a]["group_id"])
            & set(splits[b]["group_id"])
        )

        sample_overlap = len(
            set(splits[a]["sample_id"])
            & set(splits[b]["sample_id"])
        )

        assert group_overlap == 0
        assert sample_overlap == 0

        integrity_rows.append({
            "seed": seed,
            "split_a": a,
            "split_b": b,
            "group_overlap": group_overlap,
            "sample_overlap": sample_overlap
        })

    folder = DATASETS / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)

    for name, split_df in splits.items():

        path = folder / f"{name}.csv"

        split_df.to_csv(path, index=False)

        counts = split_df["Label"].value_counts()

        split_rows.append({
            "seed": seed,
            "split": name,
            "rows": len(split_df),
            "groups": split_df["group_id"].nunique(),
            "negative": int(counts.get("Negative", 0)),
            "neutral": int(counts.get("Neutral", 0)),
            "positive": int(counts.get("Positive", 0)),
            "sha256": sha256(path)
        })

    print(
        f"Seed {seed}:",
        {
            name: len(part)
            for name, part in splits.items()
        }
    )

split_table = save_table(
    pd.DataFrame(split_rows),
    "reconstruction_02_split_summary"
)

save_table(
    pd.DataFrame(integrity_rows),
    "reconstruction_03_split_integrity"
)

checksum_rows = []

for name, expected in EXPECTED_SPLITS.items():

    actual = sha256(
        DATASETS / "seed_42" / f"{name}.csv"
    )

    checksum_rows.append({
        "split": name,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "exact_match": actual == expected
    })

checksum_table = save_table(
    pd.DataFrame(checksum_rows),
    "reconstruction_04_historical_checksum_comparison"
)

EXACT_REPRODUCTION = bool(
    checksum_table["exact_match"].all()
)

print("\nHistorical checksum comparison:")

print(
    checksum_table[
        ["split", "exact_match"]
    ].to_string(index=False)
)

if EXACT_REPRODUCTION:

    print("Exact historical splits reproduced: YES")

else:

    print("Exact historical splits reproduced: NO")
    print(
        "These splits are a new reconstruction cohort."
    )
    print(
        "Historical metrics will not be mixed with "
        "new experimental metrics."
    )

print("Constructed group leakage: NONE")

# =====================================================================
# STEP 57 — CLASSICAL BASELINES ON DEVELOPMENT DATA ONLY
# =====================================================================

section("STEP 57 — RECONSTRUCT CLASSICAL BASELINES")

def text_features():

    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                tokenizer=str.split,
                token_pattern=None,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=100000
            )
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=100000
            )
        )
    ])

def metrics(y, pred):

    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(
            y, pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
        "macro_precision": precision_score(
            y, pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y, pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        )
    }

baseline_rows = []
classwise_rows = []
confusion_rows = []

for seed in SEEDS:

    train = SPLITS[seed]["train"]
    val = SPLITS[seed]["val"]

    models = {
        "Majority": DummyClassifier(
            strategy="most_frequent"
        ),

        "ComplementNB": ComplementNB(
            alpha=1.0
        ),

        "LogisticRegression": LogisticRegression(
            C=4.0,
            max_iter=2000,
            random_state=seed
        ),

        "LinearSVC": LinearSVC(
            C=1.0,
            random_state=seed
        )
    }

    # Fit feature extraction on training data only.

    features = text_features()

    X_train = features.fit_transform(
        train["text_norm"]
    )

    X_val = features.transform(
        val["text_norm"]
    )

    for model_name, model in models.items():

        t0 = time.time()

        model.fit(
            X_train,
            train["Label"]
        )

        pred = model.predict(X_val)

        result = metrics(
            val["Label"], pred
        )

        baseline_rows.append({
            "seed": seed,
            "model": model_name,
            "split": "validation",
            "n": len(val),
            **result,
            "runtime_seconds": time.time() - t0
        })

        pred_df = pd.DataFrame({
            "sample_id": val["sample_id"],
            "group_id": val["group_id"],
            "true_label": val["Label"],
            "predicted_label": pred
        })

        pred_df.to_csv(
            PRIVATE /
            f"{model_name}_seed{seed}_val.csv",
            index=False
        )

        for label in LABELS:

            binary_true = (
                val["Label"].to_numpy() == label
            )

            binary_pred = pred == label

            classwise_rows.append({
                "seed": seed,
                "model": model_name,
                "class": label,
                "support": int(binary_true.sum()),
                "f1": f1_score(
                    binary_true,
                    binary_pred,
                    zero_division=0
                ),
                "precision": precision_score(
                    binary_true,
                    binary_pred,
                    zero_division=0
                ),
                "recall": recall_score(
                    binary_true,
                    binary_pred,
                    zero_division=0
                )
            })

        cm = confusion_matrix(
            val["Label"], pred,
            labels=LABELS
        )

        for i, true_label in enumerate(LABELS):

            for j, predicted_label in enumerate(LABELS):

                confusion_rows.append({
                    "seed": seed,
                    "model": model_name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": int(cm[i, j])
                })

        print(
            f"Seed {seed} | "
            f"{model_name:20s} | "
            f"Acc {result['accuracy']:.4f} | "
            f"F1 {result['macro_f1']:.4f}"
        )

    del X_train, X_val, features
    gc.collect()

baseline_table = save_table(
    pd.DataFrame(baseline_rows),
    "reconstruction_05_baseline_results"
)

summary_table = (
    baseline_table.groupby("model")
    .agg(
        mean_accuracy=("accuracy", "mean"),
        sd_accuracy=("accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        sd_macro_f1=("macro_f1", "std")
    )
    .reset_index()
    .sort_values(
        "mean_macro_f1",
        ascending=False
    )
)

save_table(
    summary_table,
    "reconstruction_06_baseline_summary"
)

classwise_table = save_table(
    pd.DataFrame(classwise_rows),
    "reconstruction_07_classwise_results"
)

confusion_table = save_table(
    pd.DataFrame(confusion_rows),
    "reconstruction_08_confusion_matrices"
)

BEST_CLASSICAL = summary_table.iloc[0]["model"]

print("\nThree-split validation summary:")

print(
    summary_table.round(4)
    .to_string(index=False)
)

print("Best classical baseline:", BEST_CLASSICAL)

# =====================================================================
# STEP 58 — PUBLICATION FIGURES AND AUTOMATIC GITHUB BACKUP
# =====================================================================

section("STEP 58 — PUBLICATION ASSETS")

# Figure 1: Dataset reconstruction

fig, ax = plt.subplots(figsize=(8, 4.8))

ax.barh(
    dataset_audit["stage"],
    dataset_audit["rows"],
    color=["#64748b", "#64748b", "#4f83a3", "#237a68"]
)

ax.invert_yaxis()

for i, value in enumerate(dataset_audit["rows"]):

    ax.text(
        value + 100,
        i,
        f"{value:,}",
        va="center"
    )

ax.set_xlim(0, len(raw) * 1.15)
ax.set_xlabel("Number of records")
ax.set_title("Dataset integrity and reconstruction")

save_figure(
    fig,
    "reconstruction_01_dataset_flow",
    "Dataset reconstruction after invalid-record, "
    "conflicting-label and normalized-duplicate exclusion.",
    "reconstruction_01_dataset_audit.csv"
)

# Figure 2: Class distribution

counts = df["Label"].value_counts().reindex(LABELS)

fig, ax = plt.subplots(figsize=(6.5, 4.5))

bars = ax.bar(
    LABELS,
    counts.values,
    color=["#bd6262", "#94a3b8", "#438f79"]
)

ax.bar_label(
    bars,
    labels=[f"{x:,}" for x in counts.values],
    padding=4
)

ax.set_ylim(0, counts.max() * 1.18)
ax.set_ylabel("Number of samples")
ax.set_title("Reconstructed dataset class distribution")

save_figure(
    fig,
    "reconstruction_02_class_distribution",
    "Sentiment label distribution after dataset cleaning.",
    "reconstruction_02_split_summary.csv"
)

# Figure 3: Split composition

plot_df = split_table[
    split_table["seed"] == 42
].set_index("split").loc[
    ["train", "val", "test"]
]

fig, ax = plt.subplots(figsize=(7, 4.8))

bottom = np.zeros(len(plot_df))

for column, color in [
    ("negative", "#bd6262"),
    ("neutral", "#94a3b8"),
    ("positive", "#438f79")
]:

    values = plot_df[column].to_numpy()

    ax.bar(
        plot_df.index,
        values,
        bottom=bottom,
        label=column.capitalize(),
        color=color
    )

    bottom += values

ax.set_ylabel("Number of samples")
ax.set_title("Group-disjoint dataset partitions")
ax.legend(frameon=False)

save_figure(
    fig,
    "reconstruction_03_split_composition",
    "Class composition of the reconstructed seed-42 "
    "group-disjoint dataset partitions.",
    "reconstruction_02_split_summary.csv"
)

# Figure 4: Baseline comparison

plot_df = summary_table.sort_values(
    "mean_macro_f1"
)

fig, ax = plt.subplots(figsize=(8, 4.8))

ax.barh(
    plot_df["model"],
    plot_df["mean_macro_f1"],
    xerr=plot_df["sd_macro_f1"],
    capsize=4,
    color="#4f83a3"
)

ax.set_xlim(0, 1)
ax.set_xlabel("Validation macro-F1")
ax.set_title("Classical sentiment classification baselines")

save_figure(
    fig,
    "reconstruction_04_classical_baselines",
    "Mean validation macro-F1 across three grouped "
    "dataset partitions. Error bars represent split-level "
    "standard deviations, not confidence intervals.",
    "reconstruction_06_baseline_summary.csv"
)

# Figure 5: Per-class performance

class_plot = (
    classwise_table.groupby(
        ["model", "class"]
    )["f1"]
    .mean()
    .unstack()
    .reindex(columns=LABELS)
)

fig, ax = plt.subplots(figsize=(8, 5))

class_plot.plot(
    kind="bar",
    ax=ax,
    color=["#bd6262", "#94a3b8", "#438f79"]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Mean class F1")
ax.set_xlabel("")
ax.set_title("Classwise validation performance")
ax.tick_params(axis="x", rotation=20)
ax.legend(frameon=False)

save_figure(
    fig,
    "reconstruction_05_classwise_performance",
    "Mean class-specific F1 across reconstructed "
    "validation partitions.",
    "reconstruction_07_classwise_results.csv"
)

# Figure 6: Best baseline confusion matrix

cm_df = confusion_table[
    (confusion_table["seed"] == 42)
    & (confusion_table["model"] == BEST_CLASSICAL)
]

cm = (
    cm_df.pivot(
        index="true_label",
        columns="predicted_label",
        values="count"
    )
    .reindex(index=LABELS, columns=LABELS)
    .to_numpy()
)

fig, ax = plt.subplots(figsize=(5.8, 5))

image = ax.imshow(cm, cmap="Blues")

for i in range(3):

    for j in range(3):

        ax.text(
            j, i, f"{cm[i, j]:,}",
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
ax.set_title(f"{BEST_CLASSICAL}: validation confusion matrix")

fig.colorbar(image, ax=ax, fraction=0.046)

save_figure(
    fig,
    "reconstruction_06_confusion_matrix",
    "Seed-42 validation confusion matrix of the "
    "strongest reconstructed classical baseline.",
    "reconstruction_08_confusion_matrices.csv"
)

# Publication figure manifest

manifest_path = PUB / "figure_manifest.csv"

new_manifest = pd.DataFrame(FIGURE_RECORDS)

if manifest_path.exists():

    old_manifest = pd.read_csv(manifest_path)

    if "figure_id" in old_manifest.columns:

        new_manifest = pd.concat(
            [old_manifest, new_manifest],
            ignore_index=True
        ).drop_duplicates(
            "figure_id",
            keep="last"
        )

new_manifest.to_csv(
    manifest_path,
    index=False
)

# Save experiment configuration

config = {
    "raw_dataset_sha256": RAW_EXPECTED,
    "seeds": SEEDS,
    "labels": LABELS,
    "normalization": "NFC, zero-width removal, whitespace, casefold",
    "conflicting_labels": "exclude all conflicting normalized groups",
    "deduplication": "normalized text, keep first",
    "near_duplicate_similarity": 0.90,
    "near_duplicate_method": "character TF-IDF cosine",
    "outer_group_folds": 7,
    "inner_group_folds": 6,
    "exact_historical_split_reproduction": EXACT_REPRODUCTION,
    "test_predictions_generated": False,
    "test_metrics_computed": False,
    "best_classical_baseline": BEST_CLASSICAL,
    "proposed_model_finalized": False,
    "novelty_verified": False
}

save_json(
    config,
    AUDIT / "reconstruction_config.json"
)

# Recover the complete current cell source.
# The token was entered through getpass, not source code.

source_saved = False

try:

    ip = get_ipython()

    history = ip.history_manager.input_hist_raw

    source_candidates = [
        cell for cell in history
        if "STAGE 6A" in cell
        and "TARGETED RECONSTRUCTION" in cell
        and len(cell) > 5000
    ]

    if source_candidates:

        source = source_candidates[-1]

        secret_pattern = (
            r"(?:ghp_|github_pat_|gho_|ghs_)"
            r"[A-Za-z0-9_]{20,}"
        )

        assert not re.search(
            secret_pattern, source
        ), "Possible embedded credential detected."

        (AUDIT / "reconstruction_complete.py").write_text(
            source,
            encoding="utf-8"
        )

        (PUB / "scripts" / "reconstruction_complete.py").write_text(
            source,
            encoding="utf-8"
        )

        source_saved = True

except Exception as e:

    print(
        "Source preservation warning:",
        type(e).__name__
    )

# Save private dataset reconstruction archive.
# This archive is not uploaded to the public repository.

PRIVATE_ZIP = WORK / "sentiment_reconstruction_private.zip"

with zipfile.ZipFile(
    PRIVATE_ZIP,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for path in OUT.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(OUT)
            )

with zipfile.ZipFile(PRIVATE_ZIP) as z:
    assert z.testzip() is None

# Save publication assets archive

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

# =====================================================================
# VERIFIED GITHUB BACKUP
# =====================================================================

section("AUTOMATIC GITHUB BACKUP")

for folder in [
    BACKUP / "tables",
    BACKUP / "configs",
    BACKUP / "src",
    BACKUP / "publication_assets"
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

# Explicit allowlist: only aggregate tables.

for path in TABLES.glob("*.csv"):

    columns = pd.read_csv(
        path, nrows=0
    ).columns

    forbidden = {
        "bangla", "text", "text_norm",
        "text_exact", "comment", "review",
        "sample_id", "raw_row_id"
    }

    assert not (
        set(c.lower() for c in columns)
        & forbidden
    ), f"Private column detected: {path.name}"

    shutil.copy2(
        path,
        BACKUP / "tables" / path.name
    )

shutil.copy2(
    AUDIT / "reconstruction_config.json",
    BACKUP / "configs" / "reconstruction_config.json"
)

if source_saved:

    shutil.copy2(
        AUDIT / "reconstruction_complete.py",
        BACKUP / "src" / "reconstruction_complete.py"
    )

# Copy only figures and captions generated by this cell.

for record in FIGURE_RECORDS:

    for ext in ["png", "pdf", "svg"]:

        relative = Path(record[ext])

        destination = (
            BACKUP / "publication_assets" / relative
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            PUB / relative,
            destination
        )

    caption_name = (
        record["figure_id"] + ".txt"
    )

    destination = (
        BACKUP / "publication_assets" /
        "captions" / caption_name
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        PUB / "captions" / caption_name,
        destination
    )

pd.DataFrame(FIGURE_RECORDS).to_csv(
    BACKUP / "publication_assets" /
    "figure_manifest.csv",
    index=False
)

# Preserve historical findings as historical reports,
# not as newly reproduced experimental results.

historical = {
    "status": "historical reported results; not reproduced",
    "stage3_banglabert_ensemble_test_macro_f1": 0.8108,
    "stage4_global_validation_macro_f1": 0.829533,
    "stage4_ulcnet_validation_macro_f1": 0.824094,
    "stage5_ulcnet_v2_validation_macro_f1": 0.824765,
    "interpretation": (
        "Earlier linguistic heads did not surpass "
        "the strongest global residual control."
    )
}

save_json(
    historical,
    BACKUP / "historical_results.json"
)

# Artifact integrity manifest

artifact_records = []

for path in BACKUP.rglob("*"):

    if path.is_file():

        artifact_records.append({
            "path": str(path.relative_to(BACKUP)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path)
        })

pd.DataFrame(artifact_records).to_csv(
    BACKUP / "artifact_manifest.csv",
    index=False
)

git(
    "add",
    "research_reconstruction_v1"
)

git(
    "commit", "-m",
    "Reconstruct grouped sentiment dataset and "
    "preserve classical baselines and publication assets"
)

git("push", "origin", BRANCH)

FINAL_COMMIT = verify_remote()

# =====================================================================
# FINAL REPORT
# =====================================================================

section("FINAL STAGE 6A RECONSTRUCTION REPORT")

print("GitHub authentication: PASS")
print("GitHub initial write verification: PASS")
print("GitHub final backup: PASS")
print("GitHub synchronized: True")

print("\nRepository:", REPO_INFO["full_name"])
print("Research branch:", BRANCH)
print("Final verified commit:", FINAL_COMMIT)

print("\nDATASET RECONSTRUCTION")

print("Original raw checksum: PASS")
print("Raw rows:", len(raw))
print("Conflicting groups:", CONFLICT_GROUPS)
print("Conflicting rows:", CONFLICT_ROWS)
print("Clean unique samples:", len(df))
print("Near-duplicate edges:", len(edges))
print("Constructed groups:", len(group_sizes))
print("Cross-split group overlap: ZERO")

print(
    "Exact historical split reproduction:",
    EXACT_REPRODUCTION
)

print("\nCLASSICAL BASELINES")

print(
    summary_table.round(4)
    .to_string(index=False)
)

print("\nPUBLICATION ASSETS")

print("New unique figures:", len(FIGURE_RECORDS))

for ext in ["png", "pdf", "svg"]:

    print(
        ext.upper(),
        "figures:",
        len(list(
            (PUB / "figures" / ext)
            .glob(f"*.{ext}")
        ))
    )

print("Aggregate result tables:", len(list(TABLES.glob("*.csv"))))
print("Complete source preserved:", source_saved)
print("Private archive integrity: PASS")
print("Publication archive integrity: PASS")

print("\nPrivate archive:", PRIVATE_ZIP)
print("Publication archive:", PUBLICATION_ZIP)

print(
    "\nGitHub backup:",
    f"https://github.com/{OWNER}/{REPOSITORY}/tree/{BRANCH}"
)

print("\nRaw dataset uploaded publicly: NO")
print("Test model predictions generated: NO")
print("Test metrics computed: NO")
print("Transformer training performed: NO")
print("Proposed architecture finalized: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")

print(
    "\nRuntime minutes:",
    round((time.time() - START) / 60, 2)
)

print(
    "\nREADY FOR STAGE 6B — "
    "TRANSFORMER BASELINES AND PROPOSED ARCHITECTURE"
)

SESSION.headers.pop("Authorization", None)
SESSION.close()

GIT_ENV.clear()

TOKEN = None
credential = None

gc.collect()
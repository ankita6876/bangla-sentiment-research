# ================================================================
# STAGE 10B — EXTERNAL DATA ACQUISITION & INTEGRITY AUDIT
# Single-cell Kaggle execution
# No Google Drive | No paid API | Automatic GitHub backup
# ================================================================

import os
import re
import sys
import gc
import json
import time
import shutil
import hashlib
import zipfile
import getpass
import subprocess
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from huggingface_hub import HfApi, hf_hub_download

warnings.filterwarnings("ignore")
START = time.time()

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

ROOT = WORK / "sentiment_research_stage10b"
DATA = ROOT / "private_data"
TABLES = ROOT / "tables"
AUDIT = ROOT / "audit"
FIG = ROOT / "figures"
REPO = WORK / "sentiment_stage10b_github"

for p in [ROOT, DATA, TABLES, AUDIT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"
BRANCH = "research/reconstruction-20261127-080232"

PREVIOUS_COMMIT = (
    "58a17a541372c17733c872ce1b6d414de678f900"
)

TAG = "stage10b-external-data-audit-v1"

LABELS = ["Negative", "Neutral", "Positive"]
SEED = 42
NEAR_THRESHOLD = 0.90
rng = np.random.default_rng(SEED)

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def text_hash(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()

def write_json(obj, path):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

def normalize(s):
    s = unicodedata.normalize("NFC", str(s))
    s = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().casefold()

# ================================================================
# STEP 210 — AUTHENTICATED GITHUB RECOVERY
# ================================================================

section("STEP 210 — VERIFIED GITHUB CONNECTION")

TOKEN = getpass.getpass("Enter your working GitHub token: ")

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

PUBLIC_API = (
    f"https://api.github.com/repos/{OWNER}/{PUBLIC_REPO}"
)

PRIVATE_API = (
    f"https://api.github.com/repos/{OWNER}/{PRIVATE_REPO}"
)

for endpoint in [PUBLIC_API, PRIVATE_API]:
    r = SESSION.get(endpoint, timeout=60)
    r.raise_for_status()

assert SESSION.get(
    PRIVATE_API, timeout=60
).json()["private"], "Artifact repository must remain private."

print("Authentication: PASS")
print("Private artifact repository: VERIFIED")

if REPO.exists():
    shutil.rmtree(REPO)

env = os.environ.copy()

import base64

auth = base64.b64encode(
    f"x-access-token:{TOKEN}".encode()
).decode()

env.update({
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "http.extraHeader",
    "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
    "GIT_TERMINAL_PROMPT": "0"
})

def git(*args, cwd=REPO, check=True):
    p = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            p.stderr.replace(TOKEN, "[REDACTED]")
            .replace(auth, "[REDACTED]")
        )
    return p.stdout.strip()

git(
    "clone", "--single-branch", "--branch", BRANCH,
    f"https://github.com/{OWNER}/{PUBLIC_REPO}.git",
    str(REPO),
    cwd=WORK
)

git("config", "user.name", "Research Automation")
git("config", "user.email", "research@users.noreply.github.com")

git("merge-base", "--is-ancestor", PREVIOUS_COMMIT, "HEAD")

BACKUP = REPO / "research_stage10b"
BACKUP.mkdir(exist_ok=True)

for p in ["tables", "audit", "figures/png", "figures/pdf",
          "figures/svg", "captions"]:
    (BACKUP / p).mkdir(parents=True, exist_ok=True)

def push(message):
    git("add", "research_stage10b")

    status = git("status", "--porcelain")

    if status:
        git("commit", "-m", message)
        git("push", "origin", BRANCH)

    head = git("rev-parse", "HEAD")
    remote = git(
        "ls-remote", "origin", f"refs/heads/{BRANCH}"
    ).split()[0]

    assert head == remote
    print("GitHub synchronized:", head)
    return head

def save_table(df, name):
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False)

    shutil.copy2(path, BACKUP / "tables" / path.name)

    tex = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.4f}"
    )

    (BACKUP / "tables" / f"{name}.tex").write_text(
        tex, encoding="utf-8"
    )

    return path

print("Previous research commit: VERIFIED")

# ================================================================
# STEP 211 — ORIGINAL DATASET RECOVERY
# ================================================================

section("STEP 211 — ORIGINAL BECS DATASET")

EXPECTED_RAW_SHA = (
    "4580950af06bbfbe5bd79e20957b63a2ed"
    "122af2a6a3e685b10576d3f70241c5"
)

raw_candidates = list(INPUT.rglob("BECS Dataset.csv"))

RAW_PATH = next(
    (p for p in raw_candidates
     if sha256(p) == EXPECTED_RAW_SHA),
    None
)

assert RAW_PATH is not None, (
    "Exact original BECS dataset not attached."
)

raw = pd.read_csv(RAW_PATH)

assert {"Bangla", "Label"}.issubset(raw.columns)

raw = raw.dropna(subset=["Bangla", "Label"]).copy()
raw["text_norm"] = raw["Bangla"].map(normalize)
raw = raw[raw["text_norm"].str.len() > 0].copy()

SOURCE_TEXTS = sorted(raw["text_norm"].unique())
SOURCE_SET = set(SOURCE_TEXTS)

print("Original dataset SHA256: PASS")
print("Original rows:", len(raw))
print("Unique normalized texts:", len(SOURCE_TEXTS))

# Using the entire original corpus for overlap screening.
# No internal test performance is used for model optimization.

# ================================================================
# STEP 212 — EXTERNAL DATASET DOWNLOAD
# ================================================================

section("STEP 212 — EXTERNAL DATASET DOWNLOAD")

api = HfApi()

PROVENANCE = []
ERRORS = []
COHORTS = {}

def record_error(dataset, error):
    ERRORS.append({
        "dataset": dataset,
        "error": str(error)
    })
    print(dataset, "| BLOCKED:", str(error))

def acquire_hf_files(repo_id, files):
    info = api.dataset_info(repo_id)
    revision = info.sha

    paths = []

    for filename in files:
        p = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            cache_dir=str(ROOT / "hf_cache")
        )

        p = Path(p)
        paths.append(p)

        PROVENANCE.append({
            "repository": repo_id,
            "revision": revision,
            "file": filename,
            "sha256": sha256(p),
            "bytes": p.stat().st_size
        })

    return paths

# SentNoB: official author-hosted copy.
# Official mapping: 0 neutral, 1 positive, 2 negative.

try:
    sent_paths = acquire_hf_files(
        "khondoker/SentNoB",
        ["Train.csv", "Val.csv", "Test.csv"]
    )

    sent_frames = []

    for split, path in zip(
        ["train", "validation", "test"], sent_paths
    ):
        df = pd.read_csv(path)
        df.columns = [
            str(c).strip().lstrip("\ufeff")
            for c in df.columns
        ]

        lookup = {c.casefold(): c for c in df.columns}

        assert "data" in lookup and "label" in lookup, (
            f"Unexpected SentNoB schema: {list(df.columns)}"
        )

        df = df.rename(columns={
            lookup["data"]: "text",
            lookup["label"]: "original_label"
        })

        df["original_split"] = split

        sent_frames.append(
            df[["text", "original_label", "original_split"]]
        )

    sent = pd.concat(sent_frames, ignore_index=True)

    sent["label"] = (
        pd.to_numeric(
            sent["original_label"], errors="coerce"
        ).map({
            0: "Neutral",
            1: "Positive",
            2: "Negative"
        })
    )

    sent["domain"] = "social_media"

    COHORTS["SentNoB"] = sent

    print("SentNoB downloaded:", len(sent))

except Exception as e:
    record_error("SentNoB", e)

# SentiGOLD: publicly hosted mirror.
# Retain original five-class labels and domain metadata.

try:
    info = api.dataset_info("SayedShaun/sentigold")

    files = api.list_repo_files(
        "SayedShaun/sentigold",
        repo_type="dataset",
        revision=info.sha
    )

    parquet_files = sorted(
        p for p in files if p.endswith(".parquet")
    )

    assert parquet_files, "No Parquet data files found."

    gold_paths = acquire_hf_files(
        "SayedShaun/sentigold", parquet_files
    )

    gold = pd.concat(
        [pd.read_parquet(p) for p in gold_paths],
        ignore_index=True
    )

    assert {"Text", "Polarity"}.issubset(gold.columns)

    gold = gold.rename(columns={
        "Text": "text",
        "Polarity": "original_label",
        "Domain": "domain"
    })

    gold["label"] = (
        gold["original_label"]
        .astype(str).str.strip().str.upper()
        .map({
            "SN": "Negative",
            "WN": "Negative",
            "NU": "Neutral",
            "WP": "Positive",
            "SP": "Positive"
        })
    )

    if "domain" not in gold:
        gold["domain"] = "unspecified"

    gold["original_split"] = "published_corpus"

    COHORTS["SentiGOLD"] = gold[
        ["text", "original_label", "label",
         "domain", "original_split"]
    ].copy()

    print("SentiGOLD downloaded:", len(gold))

except Exception as e:
    record_error("SentiGOLD", e)

save_table(
    pd.DataFrame(PROVENANCE),
    "external_01_provenance"
)

write_json(ERRORS, AUDIT / "acquisition_errors.json")
shutil.copy2(
    AUDIT / "acquisition_errors.json",
    BACKUP / "audit"
)

push("Preserve external dataset acquisition provenance")

assert COHORTS, "No external datasets could be acquired."

# ================================================================
# STEP 213 — SCHEMA, LABEL AND DUPLICATE AUDIT
# ================================================================

section("STEP 213 — EXTERNAL COHORT CLEANING")

CLEAN = {}
AUDIT_ROWS = []
CLASS_ROWS = []

for name, original in COHORTS.items():

    df = original.copy()
    initial = len(df)

    unknown = df["label"].isna().sum()

    if unknown:
        record_error(name, f"{unknown} unmapped labels")
        continue

    df = df.dropna(subset=["text"]).copy()
    df["text_norm"] = df["text"].map(normalize)

    df = df[df["text_norm"].str.len() > 0].copy()

    valid_rows = len(df)

    conflict_keys = set(
        df.groupby("text_norm")["label"]
        .nunique()
        .loc[lambda x: x > 1].index
    )

    conflicting = df["text_norm"].isin(conflict_keys)

    conflict_rows = int(conflicting.sum())

    df = df[~conflicting].copy()

    before_dedup = len(df)

    df = df.drop_duplicates("text_norm").copy()

    duplicate_rows = before_dedup - len(df)

    exact_mask = df["text_norm"].isin(SOURCE_SET)

    exact_overlap = int(exact_mask.sum())

    df = df[~exact_mask].reset_index(drop=True)

    df["sample_id"] = df["text_norm"].map(text_hash)

    CLEAN[name] = df

    AUDIT_ROWS.append({
        "dataset": name,
        "raw_rows": initial,
        "valid_rows": valid_rows,
        "conflicting_rows_removed": conflict_rows,
        "duplicate_rows_removed": duplicate_rows,
        "exact_becs_overlap_removed": exact_overlap,
        "remaining_before_near_audit": len(df)
    })

    print(
        name,
        "| Raw:", initial,
        "| Exact overlap:", exact_overlap,
        "| Remaining:", len(df)
    )

assert CLEAN, "No cohort passed schema and label checks."

# ================================================================
# STEP 214 — NEAR-DUPLICATE OVERLAP AUDIT
# ================================================================

section("STEP 214 — CROSS-DATASET NEAR-DUPLICATE AUDIT")

# Character n-gram cosine similarity.
# The vectorizer is fitted only on the original BECS corpus.
# Near-duplicates are excluded before external inference.

VEC = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=1,
    max_features=None,
    dtype=np.float32,
    norm="l2"
)

SOURCE_X = VEC.fit_transform(SOURCE_TEXTS)

NN = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
).fit(SOURCE_X)

FINAL = {}
OVERLAP_ROWS = []

for name, df in CLEAN.items():

    df = df.copy()

    sims = []
    coverage = []

    BATCH = 256

    for start in range(0, len(df), BATCH):

        batch = df["text_norm"].iloc[start:start+BATCH]

        X = VEC.transform(batch)

        nonzero = np.asarray(
            X.getnnz(axis=1) > 0
        ).ravel()

        distance, _ = NN.kneighbors(X)

        similarity = np.clip(
            1 - distance[:, 0], 0, 1
        )

        similarity[~nonzero] = 0

        sims.extend(similarity.tolist())
        coverage.extend(nonzero.tolist())

        if start % (BATCH * 30) == 0:
            print(
                f"{name}: {start}/{len(df)} audited"
            )

    df["max_becs_similarity"] = sims
    df["tfidf_nonzero"] = coverage

    near = df["max_becs_similarity"] >= NEAR_THRESHOLD

    weak_coverage = ~df["tfidf_nonzero"]

    OVERLAP_ROWS.append({
        "dataset": name,
        "threshold": NEAR_THRESHOLD,
        "audited_rows": len(df),
        "near_duplicates_removed": int(near.sum()),
        "zero_feature_rows": int(weak_coverage.sum()),
        "median_max_similarity": float(
            df["max_becs_similarity"].median()
        ),
        "p95_max_similarity": float(
            df["max_becs_similarity"].quantile(.95)
        )
    })

    # Exclude rows whose overlap status could not be assessed.
    df = df[~near & ~weak_coverage].copy()

    df = df.reset_index(drop=True)

    assert not df["text_norm"].isin(SOURCE_SET).any()
    assert df["sample_id"].is_unique

    FINAL[name] = df

    print(name, "| Audited cohort:", len(df))

    gc.collect()

save_table(
    pd.DataFrame(AUDIT_ROWS),
    "external_02_cleaning_audit"
)

save_table(
    pd.DataFrame(OVERLAP_ROWS),
    "external_03_near_duplicate_audit"
)

# ================================================================
# STEP 215 — CROSS-EXTERNAL OVERLAP
# ================================================================

section("STEP 215 — CROSS-EXTERNAL DATASET OVERLAP")

CROSS = []

if "SentNoB" in FINAL and "SentiGOLD" in FINAL:

    a = FINAL["SentNoB"]
    b = FINAL["SentiGOLD"]

    intersection = set(a["sample_id"]) & set(b["sample_id"])

    CROSS.append({
        "dataset_a": "SentNoB",
        "dataset_b": "SentiGOLD",
        "exact_shared_samples": len(intersection),
        "cross_external_near_duplicates": "not assessed",
        "action": "Exclude exact shared rows from SentiGOLD"
    })

    FINAL["SentiGOLD"] = b[
        ~b["sample_id"].isin(intersection)
    ].reset_index(drop=True)

else:
    CROSS.append({
        "status": "Second external cohort unavailable"
    })

save_table(
    pd.DataFrame(CROSS),
    "external_04_cross_external_overlap"
)

# ================================================================
# STEP 216 — EXTERNAL COHORT LOCK
# ================================================================

section("STEP 216 — EXTERNAL COHORT LOCK")

LOCK_ROWS = []
DOMAIN_ROWS = []
SPLIT_ROWS = []

for name, df in FINAL.items():

    df = df.copy()

    df["word_count"] = (
        df["text_norm"].str.split().str.len()
    )

    df["label_id"] = df["label"].map({
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    })

    df["has_latin"] = df["text_norm"].str.contains(
        r"[A-Za-z]", regex=True
    )

    df["has_negation"] = df["text_norm"].str.contains(
        r"(?<!\w)(না|নেই|নয়|নাই|হয়নি|করিনি|কখনোই)(?!\w)",
        regex=True
    )

    df["has_contrast"] = df["text_norm"].str.contains(
        r"(?<!\w)(কিন্তু|তবে|যদিও|অথচ|বরং)(?!\w)",
        regex=True
    )

    FINAL[name] = df

    # Raw external text remains local.
    local_path = DATA / f"{name}_audited.csv"
    df.to_csv(local_path, index=False)

    # Preserve membership and mappings, without raw text.
    membership = df[
        ["sample_id", "label_id", "original_label",
         "original_split", "domain", "word_count",
         "max_becs_similarity"]
    ].copy()

    membership_path = DATA / f"{name}_membership.csv"
    membership.to_csv(membership_path, index=False)

    LOCK_ROWS.append({
        "dataset": name,
        "samples": len(df),
        "class_count": df["label"].nunique(),
        "membership_sha256": sha256(membership_path),
        "median_words": float(df["word_count"].median()),
        "mean_words": float(df["word_count"].mean()),
        "negation_samples": int(df["has_negation"].sum()),
        "contrast_samples": int(df["has_contrast"].sum()),
        "latin_samples": int(df["has_latin"].sum()),
        "role": "external diagnostic cohort"
    })

    for label in LABELS:
        count = int((df["label"] == label).sum())

        CLASS_ROWS.append({
            "dataset": name,
            "class": label,
            "samples": count,
            "proportion": count / max(len(df), 1)
        })

    for domain, part in df.groupby("domain", dropna=False):
        DOMAIN_ROWS.append({
            "dataset": name,
            "domain": domain,
            "samples": len(part)
        })

    for split, part in df.groupby("original_split"):
        SPLIT_ROWS.append({
            "dataset": name,
            "original_split": split,
            "samples": len(part)
        })

LOCK_DF = pd.DataFrame(LOCK_ROWS)
CLASS_DF = pd.DataFrame(CLASS_ROWS)

save_table(LOCK_DF, "external_05_cohort_lock")
save_table(CLASS_DF, "external_06_class_distribution")
save_table(pd.DataFrame(DOMAIN_ROWS),
           "external_07_domain_distribution")
save_table(pd.DataFrame(SPLIT_ROWS),
           "external_08_original_split_distribution")

print(LOCK_DF.to_string(index=False))

push("Lock external diagnostic cohorts and membership hashes")

# ================================================================
# STEP 217 — TRAINING DATA QUALITY AUDIT
# ================================================================

section("STEP 217 — BECS DATA QUALITY AUDIT")

groups = raw.groupby("text_norm")["Label"]

stats = groups.agg(["size", "nunique"]).reset_index()

conflicts = stats[stats["nunique"] > 1]

quality = pd.DataFrame([{
    "original_rows": len(raw),
    "normalized_unique": len(stats),
    "duplicate_rows": len(raw) - len(stats),
    "conflicting_groups": len(conflicts),
    "conflicting_rows": int(conflicts["size"].sum()),
    "single_occurrence_groups": int(
        (stats["size"] == 1).sum()
    ),
    "repeated_groups": int(
        (stats["size"] > 1).sum()
    )
}])

save_table(
    quality,
    "external_09_becs_annotation_quality"
)

print(quality.to_string(index=False))

# No internal test labels are used to select new training
# samples, train another model, or tune hyperparameters.

# ================================================================
# STEP 218 — PUBLICATION FIGURES
# ================================================================

section("STEP 218 — PUBLICATION FIGURE GENERATION")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight"
})

FIGURE_RECORDS = []

def save_figure(fig, name, caption):

    for ext in ["png", "pdf", "svg"]:

        path = BACKUP / "figures" / ext / f"{name}.{ext}"

        fig.savefig(
            path,
            dpi=600 if ext == "png" else 150,
            bbox_inches="tight"
        )

        FIGURE_RECORDS.append({
            "figure": name,
            "format": ext,
            "path": str(path.relative_to(REPO)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size
        })

    (BACKUP / "captions" / f"{name}.txt").write_text(
        caption, encoding="utf-8"
    )

    plt.show()
    plt.close(fig)

    push(f"Preserve publication figure {name}")

# Figure 1: external dataset class distribution

fig, ax = plt.subplots(figsize=(8, 5))

p = CLASS_DF.pivot(
    index="dataset",
    columns="class",
    values="proportion"
).reindex(columns=LABELS)

p.plot.bar(ax=ax, rot=0)

ax.set_ylabel("Class proportion")
ax.set_xlabel("")
ax.set_title("External cohort sentiment distribution")
ax.legend(title="Sentiment", frameon=False)

save_figure(
    fig,
    "stage10b_01_external_class_distribution",
    "Sentiment proportions after external cohort cleaning."
)

# Figure 2: raw versus audited cohort size

fig, ax = plt.subplots(figsize=(8, 5))

a = pd.DataFrame(AUDIT_ROWS).set_index("dataset")
b = LOCK_DF.set_index("dataset")

counts = pd.DataFrame({
    "Original": a["raw_rows"],
    "Audited": b["samples"]
})

counts.plot.bar(ax=ax, rot=0)

ax.set_ylabel("Number of samples")
ax.set_xlabel("")
ax.set_title("External cohort construction")

save_figure(
    fig,
    "stage10b_02_external_cohort_construction",
    "Sample counts before and after integrity screening."
)

# Figure 3: exact and near-duplicate overlaps

fig, ax = plt.subplots(figsize=(8, 5))

o = pd.DataFrame(OVERLAP_ROWS).set_index("dataset")

overlap = pd.DataFrame({
    "Exact overlap": a["exact_becs_overlap_removed"],
    "Near overlap": o["near_duplicates_removed"]
})

overlap.plot.bar(ax=ax, rot=0)

ax.set_ylabel("Excluded samples")
ax.set_xlabel("")
ax.set_title("Overlap with the original BECS corpus")

save_figure(
    fig,
    "stage10b_03_cross_dataset_overlap",
    "Exact and character TF-IDF cosine near-duplicate exclusions."
)

# Figure 4: text-length distribution

fig, ax = plt.subplots(figsize=(8, 5))

for name, df in FINAL.items():

    lengths = np.sort(
        df["word_count"].to_numpy()
    )

    if len(lengths):
        ax.plot(
            lengths,
            np.arange(1, len(lengths)+1) / len(lengths),
            label=name
        )

ax.set_xlim(0, 100)
ax.set_xlabel("Words per text")
ax.set_ylabel("Cumulative proportion")
ax.set_title("External text-length distributions")
ax.legend(frameon=False)

save_figure(
    fig,
    "stage10b_04_external_text_lengths",
    "Empirical cumulative distribution of text lengths; "
    "display restricted to 100 words."
)

# Figure 5: external domain composition

domains = pd.DataFrame(DOMAIN_ROWS)

fig, ax = plt.subplots(figsize=(9, 5))

if "SentiGOLD" in FINAL:
    plot_domains = domains[
        domains["dataset"] == "SentiGOLD"
    ].nlargest(15, "samples")
else:
    plot_domains = domains.nlargest(15, "samples")

ax.barh(
    plot_domains["domain"].astype(str),
    plot_domains["samples"]
)

ax.invert_yaxis()
ax.set_xlabel("Number of samples")
ax.set_title("Audited external domain composition")

save_figure(
    fig,
    "stage10b_05_external_domains",
    "Fifteen largest domain categories, or all categories "
    "when fewer than fifteen are available."
)

save_table(
    pd.DataFrame(FIGURE_RECORDS),
    "external_10_figure_manifest"
)

# ================================================================
# STEP 219 — PERMANENT PRIVATE ARCHIVE
# ================================================================

section("STEP 219 — PRIVATE GITHUB ARTIFACT BACKUP")

PROTOCOL = {
    "stage": "10B",
    "source_dataset_sha256": EXPECTED_RAW_SHA,
    "class_order": LABELS,
    "sentnob_mapping": {
        "0": "Neutral",
        "1": "Positive",
        "2": "Negative"
    },
    "sentigold_mapping": {
        "SN": "Negative",
        "WN": "Negative",
        "NU": "Neutral",
        "WP": "Positive",
        "SP": "Positive"
    },
    "near_duplicate_threshold": NEAR_THRESHOLD,
    "external_role": "diagnostic evaluation",
    "external_model_inference_completed": False,
    "new_training_completed": False,
    "internal_test_tuning": False,
    "raw_external_text_uploaded": False,
    "limitations": [
        "SentiGOLD copy is a public mirror.",
        "Original SentiGOLD distribution and licensing "
        "require confirmation before publication.",
        "SentNoB has appeared in BanglaBERT benchmarking.",
        "Pretraining corpus overlap cannot be ruled out.",
        "Character similarity cannot detect all paraphrases.",
        "Cross-external near-duplicate overlap not yet assessed.",
        "External diagnostic cohorts cannot subsequently "
        "serve as untouched confirmation sets if used for tuning."
    ]
}

write_json(
    PROTOCOL,
    BACKUP / "audit" / "external_protocol.json"
)

write_json(
    ERRORS,
    BACKUP / "audit" / "acquisition_errors.json"
)

# Preserve the executable source without token values.

try:
    inputs = get_ipython().user_ns.get("In", [])

    source = next(
        s for s in reversed(inputs)
        if "STAGE 10B — EXTERNAL DATA ACQUISITION" in s
        and len(s) > 5000
    )

    assert TOKEN not in source

    (BACKUP / "stage10b_complete.py").write_text(
        source, encoding="utf-8"
    )

    source_saved = True

except Exception:
    source_saved = False

push("Preserve external evaluation protocol and source")

ARCHIVE = ROOT / "stage10b_external_membership.zip"

with zipfile.ZipFile(
    ARCHIVE, "w", zipfile.ZIP_DEFLATED
) as z:

    for p in DATA.glob("*_membership.csv"):
        z.write(p, f"membership/{p.name}")

    for p in BACKUP.rglob("*"):
        if p.is_file():
            z.write(
                p,
                f"research/{p.relative_to(BACKUP)}"
            )

with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None

release = SESSION.get(
    f"{PRIVATE_API}/releases/tags/{TAG}",
    timeout=60
)

if release.status_code == 404:

    release = SESSION.post(
        f"{PRIVATE_API}/releases",
        json={
            "tag_name": TAG,
            "name": "Stage 10B — External cohort audit",
            "body": (
                "External cohort memberships, audit tables, "
                "source and publication figures. No raw text."
            ),
            "draft": False,
            "prerelease": False
        },
        timeout=60
    )

release.raise_for_status()

release_info = release.json()
release_id = release_info["id"]

upload_url = release_info["upload_url"].split("{")[0]

REMOTE_ASSETS = []

def upload_verified(path):

    digest = sha256(path)

    name = f"{path.stem}_{digest[:12]}{path.suffix}"

    r = SESSION.get(
        f"{PRIVATE_API}/releases/{release_id}/assets",
        params={"per_page": 100},
        timeout=60
    )

    r.raise_for_status()

    existing = next(
        (x for x in r.json() if x["name"] == name),
        None
    )

    if existing is None:

        with open(path, "rb") as f:

            r = SESSION.post(
                upload_url,
                params={"name": name},
                data=f,
                headers={
                    "Content-Type": "application/octet-stream"
                },
                timeout=600
            )

        r.raise_for_status()
        existing = r.json()

    assert existing["size"] == path.stat().st_size

    expected = f"sha256:{digest}"

    if existing.get("digest"):

        assert existing["digest"] == expected

    else:

        response = SESSION.get(
            existing["url"],
            headers={
                "Accept": "application/octet-stream"
            },
            stream=True,
            timeout=600
        )

        response.raise_for_status()

        verified_hash = hashlib.sha256()

        for chunk in response.iter_content(1024 * 1024):
            verified_hash.update(chunk)

        assert verified_hash.hexdigest() == digest

    REMOTE_ASSETS.append({
        "name": name,
        "asset_id": existing["id"],
        "sha256": digest,
        "bytes": path.stat().st_size,
        "remote_verified": True
    })

    print("Remote verified:", name)

upload_verified(ARCHIVE)

manifest_path = ROOT / "stage10b_remote_manifest.json"

write_json(REMOTE_ASSETS, manifest_path)

upload_verified(manifest_path)

write_json(
    REMOTE_ASSETS,
    BACKUP / "audit" / "remote_assets.json"
)

# ================================================================
# STEP 220 — CONSOLIDATED RESEARCH REPORT
# ================================================================

section("STEP 220 — FINAL EXTERNAL COHORT AUDIT REPORT")

summary = {
    "stage": "10B",
    "datasets_acquired": list(COHORTS),
    "datasets_audited": list(FINAL),
    "audited_cohort_sizes": {
        name: len(df) for name, df in FINAL.items()
    },
    "source_dataset_verified": True,
    "source_code_saved": source_saved,
    "external_inference_completed": False,
    "external_cohorts_locked_for_diagnostics": True,
    "new_model_training": False,
    "internal_test_tuning": False,
    "new_publication_figures": len(FIGURE_RECORDS) // 3,
    "new_figure_files": len(FIGURE_RECORDS),
    "remote_verified_assets": len(REMOTE_ASSETS),
    "manual_saving_required": False,
    "google_drive_used": False,
    "paid_api_used": False,
    "acquisition_errors": ERRORS,
    "runtime_minutes": round(
        (time.time() - START) / 60, 2
    ),
    "decision": "REVIEW COHORT AUDIT BEFORE EXTERNAL INFERENCE"
}

write_json(
    summary,
    BACKUP / "stage10b_summary.json"
)

FINAL_COMMIT = push(
    "Complete Stage 10B external dataset integrity audit"
)

print("\nDATASET CLEANING")
print(pd.DataFrame(AUDIT_ROWS).to_string(index=False))

print("\nNEAR-DUPLICATE AUDIT")
print(pd.DataFrame(OVERLAP_ROWS).to_string(index=False))

print("\nEXTERNAL COHORTS")
print(LOCK_DF.to_string(index=False))

print("\nCLASS DISTRIBUTION")
print(CLASS_DF.to_string(index=False))

print("\nRESEARCH SUMMARY")
print(json.dumps(summary, indent=2, default=str))

figure_counts = {
    ext: len(list(REPO.rglob(f"*.{ext}")))
    for ext in ["png", "pdf", "svg"]
}

print("\nGITHUB IMAGE FILE INVENTORY")
print(figure_counts)

print("\nFinal public commit:", FINAL_COMMIT)
print("Private release:", release_info["html_url"])
print("GitHub synchronized: True")
print("Manual saving required: NO")
print("Google Drive used: NO")
print("External model inference: PENDING")
print("New model training: NO")

section("STAGE 10B COMPLETE")
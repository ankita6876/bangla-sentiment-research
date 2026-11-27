
# =============================================================================
# STAGE 7A — ADVANCED ERROR ANALYSIS AND PERFORMANCE BOTTLENECK DISCOVERY
# STEPS 98–106 | CPU COMPATIBLE | GITHUB-FIRST | NO TEST EVALUATION
# =============================================================================

import os
import re
import gc
import io
import json
import time
import hashlib
import zipfile
import warnings
import itertools
import subprocess
import unicodedata
from pathlib import Path
from getpass import getpass

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score
)
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

START = time.time()

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"
EXPECTED_COMMIT = "3fae7e25062c71c8d8ad9b664d4809e0bb864fc9"

BASE_RELEASE = "stage6b-locked-baselines-v1"
CONFIRM_RELEASE = "stage6e-amsc-confirmation-v1"
NEW_RELEASE = "stage7a-bottleneck-discovery-v1"

RAW_SHA = (
    "4580950af06bbfbe5bd79e20957b63a2ed"
    "122af2a6a3e685b10576d3f70241c5"
)

OUT = WORK / "sentiment_research_stage7a"
REC = OUT / "recovered"
TABLES = OUT / "tables"
AUDIT = OUT / "audit"
PRIVATE = OUT / "private"
FIGURES = OUT / "figures"
PUB = WORK / "sentiment_publication_assets"

REPO = WORK / "sentiment_stage7a_github"
BACKUP = REPO / "research_stage7a"

SEEDS = [42, 123, 2026]
FAMILIES = ["standard", "fixed_fusion", "matched_control"]
CLASSES = ["Negative", "Neutral", "Positive"]
LABEL_MAP = dict(zip(CLASSES, range(3)))

for p in [OUT, REC, TABLES, AUDIT, PRIVATE, FIGURES, PUB]:
    p.mkdir(parents=True, exist_ok=True)

for fmt in ["png", "pdf", "svg"]:
    (FIGURES / fmt).mkdir(parents=True, exist_ok=True)
    (PUB / "figures" / fmt).mkdir(parents=True, exist_ok=True)

for sub in ["tables/csv", "tables/latex", "captions", "scripts"]:
    (PUB / sub).mkdir(parents=True, exist_ok=True)

def section(title):
    print("\n" + "=" * 94)
    print(title)
    print("=" * 94)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(obj, path):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(
            y, pred, labels=[0, 1, 2],
            average="macro", zero_division=0
        ),
        "macro_precision": precision_score(
            y, pred, labels=[0, 1, 2],
            average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y, pred, labels=[0, 1, 2],
            average="macro", zero_division=0
        )
    }

SAVED_TABLES = []
SAVED_FIGURES = []

def save_table(df, name):
    df = pd.DataFrame(df)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    df.to_csv(PUB / "tables/csv" / f"{name}.csv", index=False)

    df.to_latex(
        PUB / "tables/latex" / f"{name}.tex",
        index=False,
        float_format="%.4f",
        escape=True
    )

    SAVED_TABLES.append(name)
    return df

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

def save_figure(fig, name, caption):
    for fmt in ["png", "pdf", "svg"]:
        path = FIGURES / fmt / f"{name}.{fmt}"
        fig.savefig(path, dpi=600 if fmt == "png" else None)

        target = PUB / "figures" / fmt / path.name
        target.write_bytes(path.read_bytes())

    (PUB / "captions" / f"{name}.txt").write_text(
        caption, encoding="utf-8"
    )

    SAVED_FIGURES.append({
        "name": name,
        "caption": caption,
        "stage": "7A"
    })

    plt.show()
    plt.close(fig)


# =============================================================================
# STEP 98 — AUTHENTICATION AND VERIFIED SOURCE RECOVERY
# =============================================================================

section("STEP 98 — VERIFIED GITHUB CONNECTION")

TOKEN = getpass("Enter your working GitHub token: ").strip()

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

API = "https://api.github.com"

def api_get(path):
    r = SESSION.get(API + path, timeout=90)
    r.raise_for_status()
    return r.json()

def api_post(path, payload):
    r = SESSION.post(API + path, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()

user = api_get("/user")
assert user["login"].lower() == OWNER.lower()

public_info = api_get(f"/repos/{OWNER}/{PUBLIC_REPO}")
private_info = api_get(f"/repos/{OWNER}/{PRIVATE_REPO}")

assert public_info["permissions"]["push"]
assert private_info["permissions"]["push"]
assert private_info["private"]

api_get(
    f"/repos/{OWNER}/{PUBLIC_REPO}/commits/{EXPECTED_COMMIT}"
)

print("Authentication: PASS")
print("Public repository write access: PASS")
print("Private repository visibility: PRIVATE")
print("Stage 6E source commit: VERIFIED")

import base64

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

def git(*args, cwd=None):
    r = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        env=GIT_ENV,
        capture_output=True,
        text=True
    )

    if r.returncode:
        raise RuntimeError(
            "Git operation failed: " +
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
    git("fetch", "origin", cwd=REPO)
    git("checkout", BRANCH, cwd=REPO)
    git("pull", "--ff-only", "origin", BRANCH, cwd=REPO)

git("merge-base", "--is-ancestor", EXPECTED_COMMIT, "HEAD", cwd=REPO)
git("config", "user.name", OWNER, cwd=REPO)
git("config", "user.email", f"{OWNER}@users.noreply.github.com", cwd=REPO)

BACKUP.mkdir(parents=True, exist_ok=True)

print("Current source commit:", git("rev-parse", "HEAD", cwd=REPO))

def push_backup(message):
    git("add", "research_stage7a", cwd=REPO)

    if git("status", "--porcelain", cwd=REPO):
        git("commit", "-m", message, cwd=REPO)

    git("push", "origin", f"HEAD:{BRANCH}", cwd=REPO)

    local = git("rev-parse", "HEAD", cwd=REPO)
    remote = git(
        "ls-remote", "origin", f"refs/heads/{BRANCH}", cwd=REPO
    ).split()[0]

    assert local == remote
    print("GitHub synchronized:", local)
    return local

# Preserve executable source before analysis.
source_saved = False

try:
    history = get_ipython().history_manager.input_hist_raw

    candidates = [
        s for s in history
        if "STAGE 7A — ADVANCED ERROR ANALYSIS" in s
        and len(s) > 5000
    ]

    if candidates:
        source = candidates[-1]
        assert TOKEN not in source

        for p in [
            AUDIT / "stage7a_complete.py",
            BACKUP / "stage7a_complete.py",
            PUB / "scripts/stage7a_complete.py"
        ]:
            p.write_text(source, encoding="utf-8")

        source_saved = True
except Exception as e:
    print("Source capture warning:", type(e).__name__)

assert source_saved, "Executable source could not be preserved."

PROTOCOL = {
    "stage": "7A",
    "seeds": SEEDS,
    "families": FAMILIES,
    "test_evaluation": False,
    "external_evaluation": False,
    "new_training": False,
    "fusion_evaluation": "five-fold grouped cross-validation within development validation",
    "oracle_is_deployable": False,
    "raw_text_publication": False
}

save_json(PROTOCOL, BACKUP / "protocol.json")
push_backup("Stage 7A: preserve analysis protocol and source")


# =============================================================================
# STEP 99 — RECOVER PRIVATE PREDICTIONS AND DATASET MEMBERSHIP
# =============================================================================

section("STEP 99 — RECOVER PRIVATE RESEARCH ARTIFACTS")

def get_release(tag):
    return api_get(
        f"/repos/{OWNER}/{PRIVATE_REPO}/releases/tags/{tag}"
    )

def release_assets(release):
    assets = []
    page = 1

    while True:
        batch = api_get(
            f"/repos/{OWNER}/{PRIVATE_REPO}/releases/"
            f"{release['id']}/assets?per_page=100&page={page}"
        )

        assets.extend(batch)

        if len(batch) < 100:
            break

        page += 1

    return {a["name"]: a for a in assets}

def download_asset(asset, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    digest = asset.get("digest") or ""

    def valid():
        if not target.exists():
            return False

        if target.stat().st_size != asset["size"]:
            return False

        if digest.startswith("sha256:"):
            return sha256(target) == digest.split(":", 1)[1]

        return False

    if not valid():
        url = (
            f"{API}/repos/{OWNER}/{PRIVATE_REPO}/"
            f"releases/assets/{asset['id']}"
        )

        with SESSION.get(
            url,
            headers={"Accept": "application/octet-stream"},
            stream=True,
            timeout=(30, 300)
        ) as r:
            r.raise_for_status()

            temp = target.with_suffix(target.suffix + ".partial")

            with open(temp, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

            temp.replace(target)

    assert target.stat().st_size == asset["size"]

    if digest.startswith("sha256:"):
        assert sha256(target) == digest.split(":", 1)[1]
        integrity = "SHA256 PASS"
    else:
        integrity = "size PASS; remote digest unavailable"

    print("Recovered:", target.name, "|", integrity)
    return target

def safe_extract(path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None

        for name in z.namelist():
            target = (destination / name).resolve()
            assert target.is_relative_to(destination)

        z.extractall(destination)

base_assets = release_assets(get_release(BASE_RELEASE))
confirm_assets = release_assets(get_release(CONFIRM_RELEASE))

base_package = download_asset(
    base_assets["stage6b_reproducibility_package.zip"],
    REC / "stage6b_reproducibility_package.zip"
)

safe_extract(base_package, REC / "baseline_package")

for family in FAMILIES:
    for seed in SEEDS:
        name = f"{family}_seed{seed}_val.npz"

        assert name in confirm_assets, f"Missing remote asset: {name}"

        download_asset(confirm_assets[name], REC / name)

# Restore the most recent publication archive, without model weights.
publication_names = [
    n for n in confirm_assets
    if n.startswith("publication_assets_") and n.endswith(".zip")
]

if publication_names:
    name = sorted(publication_names)[-1]
    archive = download_asset(confirm_assets[name], REC / name)
    safe_extract(archive, REC / "previous_publication")

    import shutil

    for p in (REC / "previous_publication").rglob("*"):
        if not p.is_file():
            continue

        parts = p.relative_to(
            REC / "previous_publication"
        ).parts

        for marker in ["figures", "tables", "captions", "scripts", "architecture"]:
            if marker in parts:
                index = parts.index(marker)
                dest = PUB.joinpath(*parts[index:])
                dest.parent.mkdir(parents=True, exist_ok=True)

                if not dest.exists():
                    shutil.copy2(p, dest)

                break

print("Nine confirmation prediction files recovered.")
print("Model checkpoints downloaded: NO")


# =============================================================================
# STEP 100 — PREDICTION ALIGNMENT AND RAW DATA RECOVERY
# =============================================================================

section("STEP 100 — EXACT SAMPLE ALIGNMENT")

def read_predictions(path):
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)

        id_key = next(
            (k for k in ["sample_id", "sample_ids", "ids"] if k in keys),
            None
        )

        label_key = next(
            (k for k in ["labels", "y_true", "label", "targets"] if k in keys),
            None
        )

        logit_key = next(
            (k for k in ["logits", "raw_logits"] if k in keys),
            None
        )

        prob_key = next(
            (k for k in ["probs", "probabilities", "proba"] if k in keys),
            None
        )

        assert id_key and label_key, (str(path), keys)
        assert logit_key or prob_key, (str(path), keys)

        ids = z[id_key].astype(str).reshape(-1)
        labels = z[label_key].astype(int).reshape(-1)

        if logit_key:
            logits = z[logit_key].astype(np.float64)
            probs = softmax(logits, axis=1)
        else:
            probs = z[prob_key].astype(np.float64)
            assert np.all(probs >= 0)
            probs /= probs.sum(axis=1, keepdims=True)
            logits = np.log(np.clip(probs, 1e-12, 1))

    assert len(set(ids)) == len(ids)
    assert len(ids) == len(labels) == len(probs)
    assert probs.shape[1] == 3
    assert np.isfinite(probs).all()
    assert set(np.unique(labels)).issubset({0, 1, 2})

    return ids, labels, logits, probs

IDS, Y, _, _ = read_predictions(
    REC / "standard_seed42_val.npz"
)

assert len(IDS) == 1840

RUNS = {}
LOGITS = {}

for family in FAMILIES:
    for seed in SEEDS:
        name = f"{family}_seed{seed}"

        ids, labels, logits, probs = read_predictions(
            REC / f"{name}_val.npz"
        )

        lookup = {sid: i for i, sid in enumerate(ids)}
        assert set(lookup) == set(IDS)

        order = np.array([lookup[sid] for sid in IDS])
        assert np.array_equal(labels[order], Y)

        RUNS[name] = probs[order]
        LOGITS[name] = logits[order]

print("Exact nine-model prediction alignment: PASS")

# Recover raw reviews locally; never upload review text.

raw_candidates = list(INPUT.rglob("BECS Dataset.csv"))
assert raw_candidates, "BECS Dataset.csv is not attached to Kaggle."

raw_path = next(
    (p for p in raw_candidates if sha256(p) == RAW_SHA),
    None
)

assert raw_path is not None, "Raw dataset checksum mismatch."

raw = pd.read_csv(raw_path)

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()

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
].drop_duplicates("sample_id").copy()

assert len(clean) == 12893

VAL = clean.set_index("sample_id").loc[IDS].reset_index()
assert np.array_equal(VAL["label_id"].to_numpy(), Y)

# Locate exact archived partition identifiers.
partition_candidates = []

for p in (REC / "baseline_package").rglob("*.csv"):
    try:
        df = pd.read_csv(p, dtype={"sample_id": str})

        if "sample_id" not in df.columns:
            continue

        sid_set = set(df["sample_id"].astype(str))

        if len(df) == 1840 and sid_set == set(IDS):
            partition_candidates.append((p, df))

    except Exception:
        pass

assert partition_candidates, (
    "Exact archived validation membership could not be located."
)

group_candidates = [
    (p, df) for p, df in partition_candidates
    if "group_id" in df.columns
]

assert group_candidates, (
    "Archived validation group identifiers are required for grouped CV."
)

partition_path, partition = group_candidates[0]

GROUPS = (
    partition.assign(sample_id=partition["sample_id"].astype(str))
    .set_index("sample_id")
    .loc[IDS, "group_id"]
    .astype(str)
    .to_numpy()
)

assert len(GROUPS) == len(Y)
assert pd.notna(GROUPS).all()

train_candidates = []

for p in (REC / "baseline_package").rglob("*.csv"):
    try:
        df = pd.read_csv(p, dtype={"sample_id": str})

        if (
            "sample_id" in df.columns
            and len(df) == 9204
            and len(set(df["sample_id"])) == 9204
            and not set(df["sample_id"]).intersection(IDS)
            and set(df["sample_id"]).issubset(set(clean["sample_id"]))
        ):
            train_candidates.append((p, df))

    except Exception:
        pass

assert train_candidates, "Exact archived training membership missing."

train_path, train_partition = train_candidates[0]

TRAIN = clean.set_index("sample_id").loc[
    train_partition["sample_id"].astype(str)
].reset_index()

if "group_id" in train_partition.columns:
    assert not set(
        train_partition["group_id"].astype(str)
    ).intersection(GROUPS)

print("Raw dataset checksum: PASS")
print("Training membership:", len(TRAIN))
print("Validation membership:", len(VAL))
print("Validation groups:", len(set(GROUPS)))
print("Validation group source:", partition_path.name)
print("Training/validation identity overlap: ZERO")
print("Test predictions generated: NO")


# =============================================================================
# STEP 101 — LINGUISTIC AND DATASET-LEVEL QUALITY AUDIT
# =============================================================================

section("STEP 101 — LINGUISTIC AND DATASET QUALITY AUDIT")

NEG = ["না", "নয়", "নয়", "নেই", "নাই"]
CON = ["কিন্তু", "তবে", "যদিও", "অথচ", "তবুও", "তথাপি"]

def cue_pattern(words):
    return re.compile(
        r"(?<![\w\u0980-\u09ff])(?:"
        + "|".join(map(re.escape, words))
        + r")(?![\w\u0980-\u09ff])"
    )

neg_pattern = cue_pattern(NEG)
con_pattern = cue_pattern(CON)

for df in [TRAIN, VAL]:
    df["word_count"] = df["text_norm"].str.split().str.len()
    df["char_count"] = df["text_norm"].str.len()

    df["negation"] = df["text_norm"].map(
        lambda x: bool(neg_pattern.search(x))
    )

    df["contrast"] = df["text_norm"].map(
        lambda x: bool(con_pattern.search(x))
    )

    df["latin"] = df["text_norm"].str.contains(
        r"[a-zA-Z]", regex=True
    )

    df["emoji"] = df["text_norm"].str.contains(
        r"[\U0001F300-\U0001FAFF]", regex=True
    )

    df["repeated_character"] = df["text_norm"].str.contains(
        r"(.)\1{2,}", regex=True
    )

    df["question"] = df["text_norm"].str.contains(
        r"[?？]", regex=True
    )

    df["punctuation_only"] = df["text_norm"].map(
        lambda x: bool(x) and all(
            unicodedata.category(c).startswith(("P", "S", "Z"))
            for c in x
        )
    )

LONG_THRESHOLD = float(TRAIN["word_count"].quantile(.75))

MASKS = {
    "all": np.ones(len(Y), dtype=bool),
    "negation_candidate": VAL["negation"].to_numpy(),
    "contrast_candidate": VAL["contrast"].to_numpy(),
    "both_operators": (VAL["negation"] & VAL["contrast"]).to_numpy(),
    "any_operator": (VAL["negation"] | VAL["contrast"]).to_numpy(),
    "no_detected_operator": (~(VAL["negation"] | VAL["contrast"])).to_numpy(),
    "long_text": (VAL["word_count"] > LONG_THRESHOLD).to_numpy(),
    "shorter_text": (VAL["word_count"] <= LONG_THRESHOLD).to_numpy(),
    "single_word": (VAL["word_count"] == 1).to_numpy(),
    "latin_code_mixed": VAL["latin"].to_numpy(),
    "emoji": VAL["emoji"].to_numpy(),
    "repeated_character": VAL["repeated_character"].to_numpy(),
    "question": VAL["question"].to_numpy(),
    "punctuation_only": VAL["punctuation_only"].to_numpy()
}

audit_rows = []

for split, df in [("train", TRAIN), ("val", VAL)]:
    for label in CLASSES:
        part = df[df["Label"] == label]

        audit_rows.append({
            "split": split,
            "class": label,
            "n": len(part),
            "mean_words": part["word_count"].mean(),
            "median_words": part["word_count"].median(),
            "single_word_pct": 100 * (part["word_count"] == 1).mean(),
            "negation_pct": 100 * part["negation"].mean(),
            "contrast_pct": 100 * part["contrast"].mean(),
            "latin_pct": 100 * part["latin"].mean(),
            "emoji_pct": 100 * part["emoji"].mean(),
            "punctuation_only_pct": 100 * part["punctuation_only"].mean()
        })

QUALITY = save_table(
    audit_rows, "stage7a_01_dataset_quality"
)

save_table([
    {
        "raw_rows": len(raw),
        "conflicting_normalized_groups": len(conflict_ids),
        "conflicting_raw_rows": raw["sample_id"].isin(conflict_ids).sum(),
        "clean_unique_samples": len(clean),
        "train_samples": len(TRAIN),
        "validation_samples": len(VAL),
        "long_text_threshold": LONG_THRESHOLD
    }
], "stage7a_02_label_conflict_audit")

print(QUALITY.round(4).to_string(index=False))

print("\nValidation subgroup sizes:")
for name, mask in MASKS.items():
    print(f"{name:26s}: {mask.sum()}")

# These are review candidates, not verified annotation errors.
train_review = TRAIN.loc[
    TRAIN["punctuation_only"]
    | (TRAIN["word_count"] == 1)
    | (TRAIN["negation"] & TRAIN["contrast"])
].copy()

train_review["human_label"] = ""
train_review["review_notes"] = ""

train_review[
    ["sample_id", "Bangla", "Label", "word_count",
     "negation", "contrast", "punctuation_only",
     "human_label", "review_notes"]
].to_csv(
    PRIVATE / "training_annotation_candidates_LOCAL_ONLY.csv",
    index=False
)

print("Training annotation candidates:", len(train_review))
print("Human-verified label corrections: ZERO")


# =============================================================================
# STEP 102 — MODEL DISAGREEMENT AND ORACLE CEILING
# =============================================================================

section("STEP 102 — MODEL COMPLEMENTARITY AND ORACLE ANALYSIS")

FAMILY_PROBS = {
    family: np.mean(
        [RUNS[f"{family}_seed{s}"] for s in SEEDS],
        axis=0
    )
    for family in FAMILIES
}

ALL_PROBS = {**RUNS, **FAMILY_PROBS}

ALL_PROBS["all_nine_uniform"] = np.mean(
    list(RUNS.values()), axis=0
)

RESULTS = save_table([
    {
        "model": name,
        **metrics(Y, p.argmax(1)),
        "nll": log_loss(Y, p, labels=[0, 1, 2])
    }
    for name, p in ALL_PROBS.items()
], "stage7a_03_model_results")

print(
    RESULTS.sort_values(
        "macro_f1", ascending=False
    ).round(5).to_string(index=False)
)

NAMES = list(RUNS)
PRED_MATRIX = np.stack(
    [RUNS[n].argmax(1) for n in NAMES],
    axis=1
)

CORRECT = PRED_MATRIX == Y[:, None]
N_CORRECT = CORRECT.sum(axis=1)

oracle_rows = []

for name, members in [
    ("standard_three", [n for n in NAMES if n.startswith("standard_")]),
    ("fixed_fusion_three", [n for n in NAMES if n.startswith("fixed_fusion_")]),
    ("matched_control_three", [n for n in NAMES if n.startswith("matched_control_")]),
    ("all_nine", NAMES)
]:
    p = np.stack([RUNS[n].argmax(1) for n in members], axis=1)
    available = (p == Y[:, None]).any(1)

    oracle_pred = np.where(
        available, Y,
        np.mean([RUNS[n] for n in members], axis=0).argmax(1)
    )

    oracle_rows.append({
        "pool": name,
        "models": len(members),
        "oracle_accuracy": available.mean(),
        "oracle_correct_when_available_macro_f1": f1_score(
            Y, oracle_pred, average="macro"
        ),
        "all_models_wrong": int((~available).sum())
    })

ORACLE = save_table(
    oracle_rows, "stage7a_04_oracle_ceiling"
)

print("\nOracle analysis — requires knowledge of true labels:")
print(ORACLE.round(5).to_string(index=False))

agreement_rows = []

for a, b in itertools.combinations(ALL_PROBS, 2):
    pa = ALL_PROBS[a].argmax(1)
    pb = ALL_PROBS[b].argmax(1)

    agreement_rows.append({
        "model_a": a,
        "model_b": b,
        "agreement": (pa == pb).mean(),
        "a_only_correct": int(((pa == Y) & (pb != Y)).sum()),
        "b_only_correct": int(((pb == Y) & (pa != Y)).sum()),
        "both_wrong": int(((pa != Y) & (pb != Y)).sum())
    })

AGREEMENT = save_table(
    agreement_rows, "stage7a_05_pairwise_complementarity"
)

difficulty_rows = [
    {
        "models_correct": k,
        "n": int((N_CORRECT == k).sum()),
        "fraction": float((N_CORRECT == k).mean())
    }
    for k in range(10)
]

DIFFICULTY = save_table(
    difficulty_rows, "stage7a_06_sample_difficulty"
)

subgroup_rows = []

for subgroup, mask in MASKS.items():
    if mask.sum() == 0:
        continue

    for name, p in {
        **FAMILY_PROBS,
        "all_nine_uniform": ALL_PROBS["all_nine_uniform"]
    }.items():

        subgroup_rows.append({
            "subgroup": subgroup,
            "model": name,
            "n": int(mask.sum()),
            **metrics(Y[mask], p.argmax(1)[mask]),
            "oracle_accuracy": float(
                (N_CORRECT[mask] > 0).mean()
            ),
            "unanimously_wrong": int(
                (N_CORRECT[mask] == 0).sum()
            )
        })

SUBGROUPS = save_table(
    subgroup_rows, "stage7a_07_subgroup_bottlenecks"
)

print("\nSubgroup macro-F1:")
print(
    SUBGROUPS.pivot(
        index="subgroup", columns="model", values="macro_f1"
    ).round(4).to_string()
)

print("\nUnanimously incorrect samples:", int((N_CORRECT == 0).sum()))
print("Unanimously correct samples:", int((N_CORRECT == 9).sum()))

# Local review queue; no human label corrections are assumed.
mean_probs = ALL_PROBS["all_nine_uniform"]
mean_pred = mean_probs.argmax(1)

review = VAL[["sample_id", "Bangla", "Label"]].copy()
review["models_correct"] = N_CORRECT
review["ensemble_prediction"] = [CLASSES[i] for i in mean_pred]
review["confidence"] = mean_probs.max(1)
review["human_label"] = ""
review["review_notes"] = ""

review = review.loc[N_CORRECT < 9].sort_values(
    ["models_correct", "confidence"],
    ascending=[True, False]
)

review.to_csv(
    PRIVATE / "validation_error_review_LOCAL_ONLY.csv",
    index=False
)


# =============================================================================
# STEP 103 — GROUPED CROSS-VALIDATED FUSION SCREENING
# =============================================================================

section("STEP 103 — CROSS-VALIDATED PROBABILITY FUSION")

# This is exploratory development-set stacking.
# The underlying models already used validation for checkpoint selection.
# Therefore, these are not independent generalization estimates.

def temperature_scale(p, temperature):
    return softmax(
        np.log(np.clip(p, 1e-12, 1)) / temperature,
        axis=1
    )

def fit_temperature(p, y):
    result = minimize_scalar(
        lambda log_t: log_loss(
            y,
            temperature_scale(p, np.exp(log_t)),
            labels=[0, 1, 2]
        ),
        bounds=(-2, 2),
        method="bounded"
    )

    return float(np.exp(result.x))

FAMILY_NAMES = list(FAMILY_PROBS)

# Fixed candidate grid declared before evaluating folds.
WEIGHT_GRID = []

for a in range(5):
    for b in range(5 - a):
        c = 4 - a - b
        WEIGHT_GRID.append(np.array([a, b, c]) / 4)

# Also include equal weighting.
WEIGHT_GRID.append(np.ones(3) / 3)

cv = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

OOF = {
    "uniform_raw": np.zeros((len(Y), 3)),
    "uniform_calibrated": np.zeros((len(Y), 3)),
    "weighted_calibrated": np.zeros((len(Y), 3))
}

fold_rows = []
fold_assignment = np.full(len(Y), -1)

for fold, (fit_idx, eval_idx) in enumerate(
    cv.split(np.zeros(len(Y)), Y, GROUPS), start=1
):
    assert not set(GROUPS[fit_idx]).intersection(GROUPS[eval_idx])

    fold_assignment[eval_idx] = fold

    fit_p = []
    eval_p = []
    temps = []

    for name in FAMILY_NAMES:
        p = FAMILY_PROBS[name]
        t = fit_temperature(p[fit_idx], Y[fit_idx])

        temps.append(t)
        fit_p.append(temperature_scale(p[fit_idx], t))
        eval_p.append(temperature_scale(p[eval_idx], t))

    fit_p = np.stack(fit_p)
    eval_p = np.stack(eval_p)

    # Select weights only on the fitting portion of this fold.
    scores = []

    for weights in WEIGHT_GRID:
        fused = np.einsum("m,mnc->nc", weights, fit_p)

        scores.append(
            f1_score(
                Y[fit_idx], fused.argmax(1),
                average="macro"
            )
        )

    best_weights = WEIGHT_GRID[int(np.argmax(scores))]

    OOF["uniform_raw"][eval_idx] = np.mean(
        [FAMILY_PROBS[n][eval_idx] for n in FAMILY_NAMES],
        axis=0
    )

    OOF["uniform_calibrated"][eval_idx] = eval_p.mean(0)

    OOF["weighted_calibrated"][eval_idx] = np.einsum(
        "m,mnc->nc", best_weights, eval_p
    )

    for method, p in OOF.items():
        fold_rows.append({
            "fold": fold,
            "method": method,
            "fit_samples": len(fit_idx),
            "evaluation_samples": len(eval_idx),
            "weight_standard": best_weights[0],
            "weight_fixed_fusion": best_weights[1],
            "weight_matched_control": best_weights[2],
            "temperature_standard": temps[0],
            "temperature_fixed_fusion": temps[1],
            "temperature_matched_control": temps[2],
            **metrics(Y[eval_idx], p[eval_idx].argmax(1))
        })

    print(
        f"Fold {fold} | "
        f"weights {best_weights.round(3)} | "
        f"F1 {metrics(Y[eval_idx], OOF['weighted_calibrated'][eval_idx].argmax(1))['macro_f1']:.5f}"
    )

assert (fold_assignment > 0).all()

for p in OOF.values():
    assert np.allclose(p.sum(1), 1)

FOLD_RESULTS = save_table(
    fold_rows, "stage7a_08_fusion_cv_folds"
)

FUSION_RESULTS = save_table([
    {
        "method": name,
        **metrics(Y, p.argmax(1)),
        "nll": log_loss(Y, p, labels=[0, 1, 2])
    }
    for name, p in OOF.items()
], "stage7a_09_fusion_cv_results")

print("\nCross-validated fusion results:")
print(FUSION_RESULTS.round(5).to_string(index=False))

np.savez_compressed(
    PRIVATE / "fusion_oof_predictions.npz",
    sample_id=IDS,
    labels=Y,
    fold=fold_assignment,
    **OOF
)


# =============================================================================
# STEP 104 — STATISTICAL AND CLASS-CONFUSION ANALYSIS
# =============================================================================

section("STEP 104 — PAIRED STATISTICS AND CLASS CONFUSION")

COMPARISON_PROBS = {
    **FAMILY_PROBS,
    **OOF
}

unique_groups = np.unique(GROUPS)

group_indices = [
    np.flatnonzero(GROUPS == group)
    for group in unique_groups
]

rng = np.random.default_rng(42)

# Shared bootstrap samples for paired group-level comparisons.
BOOTSTRAPS = [
    np.concatenate([
        group_indices[j]
        for j in rng.integers(
            0, len(group_indices), len(group_indices)
        )
    ])
    for _ in range(1500)
]

def fast_f1(y, pred):
    cm = np.bincount(
        3 * y + pred, minlength=9
    ).reshape(3, 3)

    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)

    return np.divide(
        2 * tp, denom,
        out=np.zeros(3, dtype=float),
        where=denom > 0
    ).mean()

stat_rows = []

for name, probs in COMPARISON_PROBS.items():
    if name == "matched_control":
        continue

    pred = probs.argmax(1)
    ref = FAMILY_PROBS["matched_control"].argmax(1)

    diffs = np.array([
        fast_f1(Y[idx], pred[idx])
        - fast_f1(Y[idx], ref[idx])
        for idx in BOOTSTRAPS
    ])

    gain = int(((pred == Y) & (ref != Y)).sum())
    loss = int(((pred != Y) & (ref == Y)).sum())

    pvalue = (
        binomtest(gain, gain + loss, .5).pvalue
        if gain + loss else 1.0
    )

    stat_rows.append({
        "model": name,
        "reference": "matched_control",
        "delta_macro_f1": fast_f1(Y, pred) - fast_f1(Y, ref),
        "ci95_low": np.quantile(diffs, .025),
        "ci95_high": np.quantile(diffs, .975),
        "gained_correct": gain,
        "lost_correct": loss,
        "mcnemar_p_unadjusted": pvalue
    })

STATS = pd.DataFrame(stat_rows)

order = np.argsort(STATS["mcnemar_p_unadjusted"].to_numpy())
p = STATS["mcnemar_p_unadjusted"].to_numpy()

adjusted = np.empty(len(p))
adjusted[order] = np.minimum(
    1, np.maximum.accumulate(
        p[order] * (len(p) - np.arange(len(p)))
    )
)

STATS["mcnemar_p_holm"] = adjusted

save_table(STATS, "stage7a_10_paired_statistics")

print(STATS.round(5).to_string(index=False))

class_rows = []
confusion_rows = []

for name, p in COMPARISON_PROBS.items():
    pred = p.argmax(1)

    report = classification_report(
        Y, pred, labels=[0, 1, 2],
        target_names=CLASSES,
        output_dict=True, zero_division=0
    )

    for label in CLASSES:
        class_rows.append({
            "model": name,
            "class": label,
            **report[label]
        })

    cm = confusion_matrix(Y, pred, labels=[0, 1, 2])

    for i, j in itertools.product(range(3), repeat=2):
        confusion_rows.append({
            "model": name,
            "true_class": CLASSES[i],
            "predicted_class": CLASSES[j],
            "count": int(cm[i, j])
        })

CLASSWISE = save_table(
    class_rows, "stage7a_11_classwise_performance"
)

CONFUSION = save_table(
    confusion_rows, "stage7a_12_confusion_matrices"
)

# Identify whether uncertainty predicts actual errors.
uncertainty_rows = []

for name, p in COMPARISON_PROBS.items():
    error = (p.argmax(1) != Y).astype(int)

    sorted_p = np.sort(p, axis=1)
    entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]

    for score_name, score in [
        ("entropy", entropy),
        ("one_minus_confidence", 1 - p.max(1)),
        ("one_minus_margin", 1 - margin)
    ]:
        uncertainty_rows.append({
            "model": name,
            "uncertainty_measure": score_name,
            "error_detection_auroc": roc_auc_score(error, score)
        })

UNCERTAINTY = save_table(
    uncertainty_rows, "stage7a_13_uncertainty_diagnostics"
)

# Length-dependent error analysis.
length_rows = []

length_bins = [
    (1, 1), (2, 3), (4, 6),
    (7, 10), (11, 20), (21, np.inf)
]

for low, high in length_bins:
    mask = (
        (VAL["word_count"] >= low)
        & (VAL["word_count"] <= high)
    ).to_numpy()

    if not mask.any():
        continue

    for name, p in FAMILY_PROBS.items():
        length_rows.append({
            "length_bin": f"{low}-{high}" if np.isfinite(high) else f"{low}+",
            "model": name,
            "n": int(mask.sum()),
            **metrics(Y[mask], p.argmax(1)[mask])
        })

LENGTH_RESULTS = save_table(
    length_rows, "stage7a_14_length_performance"
)

print("\nClass-wise F1:")
print(
    CLASSWISE.pivot(
        index="class", columns="model", values="f1-score"
    ).round(4).to_string()
)

print("\nUncertainty diagnostics:")
print(UNCERTAINTY.round(4).to_string(index=False))


# =============================================================================
# STEP 105 — PUBLICATION-QUALITY FIGURES
# =============================================================================

section("STEP 105 — PUBLICATION ASSETS")

# Figure 1: Model performance.
fig, ax = plt.subplots(figsize=(9, 6))

plot_df = RESULTS.sort_values("macro_f1")

ax.barh(plot_df["model"], 100 * plot_df["macro_f1"])
ax.set_xlabel("Validation macro-F1 (%)")
ax.set_title("Performance across individual models and ensembles")
ax.set_xlim(0, 100)

for i, v in enumerate(plot_df["macro_f1"]):
    ax.text(100*v + .5, i, f"{100*v:.2f}", va="center", fontsize=9)

save_figure(
    fig, "stage7a_01_model_performance",
    "Development validation macro-F1 across nine individual runs and ensembles."
)

# Figure 2: Oracle ceiling.
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.bar(
    ORACLE["pool"],
    100 * ORACLE["oracle_accuracy"]
)

ax.set_ylabel("Oracle accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Potential benefit from complementary predictions")

for i, v in enumerate(ORACLE["oracle_accuracy"]):
    ax.text(i, 100*v + 1, f"{100*v:.2f}%", ha="center")

save_figure(
    fig, "stage7a_02_oracle_ceiling",
    "Oracle accuracy assumes access to true labels and is not deployable performance."
)

# Figure 3: Difficulty distribution.
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.bar(
    DIFFICULTY["models_correct"],
    DIFFICULTY["n"]
)

ax.set_xticks(range(10))
ax.set_xlabel("Number of models predicting correctly")
ax.set_ylabel("Validation samples")
ax.set_title("Distribution of sample-level model difficulty")

save_figure(
    fig, "stage7a_03_sample_difficulty",
    "Number of correct predictions per validation example across nine model runs."
)

# Figure 4: Model agreement matrix.
agreement_matrix = np.zeros((len(NAMES), len(NAMES)))

for i, j in itertools.product(range(len(NAMES)), repeat=2):
    agreement_matrix[i, j] = (
        PRED_MATRIX[:, i] == PRED_MATRIX[:, j]
    ).mean()

fig, ax = plt.subplots(figsize=(9, 8))

im = ax.imshow(
    agreement_matrix, vmin=0, vmax=1, cmap="Blues"
)

ax.set_xticks(range(len(NAMES)), NAMES, rotation=60, ha="right")
ax.set_yticks(range(len(NAMES)), NAMES)

for i, j in itertools.product(range(len(NAMES)), repeat=2):
    value = agreement_matrix[i, j]

    ax.text(
        j, i, f"{value:.2f}",
        ha="center", va="center",
        color="white" if value > .65 else "black",
        fontsize=8
    )

fig.colorbar(im, ax=ax, label="Prediction agreement")
ax.set_title("Pairwise model prediction agreement")

save_figure(
    fig, "stage7a_04_model_agreement",
    "Pairwise prediction agreement between individual model runs."
)

# Figure 5: Subgroup performance.
pivot = SUBGROUPS.loc[
    SUBGROUPS["n"] >= 10
].pivot(
    index="subgroup", columns="model", values="macro_f1"
)

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(
    pivot.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1
)

ax.set_xticks(
    range(len(pivot.columns)), pivot.columns,
    rotation=35, ha="right"
)

ax.set_yticks(range(len(pivot.index)), pivot.index)

for i, j in itertools.product(
    range(len(pivot.index)), range(len(pivot.columns))
):
    value = pivot.iloc[i, j]

    ax.text(
        j, i, f"{value:.3f}",
        ha="center", va="center",
        color="white" if value > .7 else "black",
        fontsize=9
    )

fig.colorbar(im, ax=ax, label="Macro-F1")
ax.set_title("Linguistic subgroup performance")

save_figure(
    fig, "stage7a_05_subgroup_performance",
    "Descriptive subgroup performance; lexical cues do not establish gold linguistic annotations."
)

# Figure 6: Class-wise performance.
fig, ax = plt.subplots(figsize=(9, 5))

plot = CLASSWISE[
    CLASSWISE["model"].isin(FAMILIES)
].pivot(index="class", columns="model", values="f1-score")

plot.plot(kind="bar", ax=ax, rot=0)
ax.set_ylim(0, 1)
ax.set_ylabel("Class F1")
ax.set_title("Sentiment-specific classification performance")
ax.legend(frameon=False)

save_figure(
    fig, "stage7a_06_classwise_performance",
    "Class-specific F1 for the three confirmed model families."
)

# Figure 7: Cross-validated fusion.
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.bar(
    FUSION_RESULTS["method"],
    100 * FUSION_RESULTS["macro_f1"]
)

ax.set_ylim(0, 100)
ax.set_ylabel("Development macro-F1 (%)")
ax.tick_params(axis="x", rotation=15)
ax.set_title("Grouped cross-validated fusion screening")

for i, v in enumerate(FUSION_RESULTS["macro_f1"]):
    ax.text(i, 100*v + 1, f"{100*v:.2f}%", ha="center")

save_figure(
    fig, "stage7a_07_fusion_screening",
    "Grouped cross-validation of fusion weights within the previously used development validation set."
)

# Figure 8: Paired confidence intervals.
fig, ax = plt.subplots(figsize=(9, 5))

for i, row in STATS.reset_index(drop=True).iterrows():
    ax.plot(
        [100*row["ci95_low"], 100*row["ci95_high"]],
        [i, i], linewidth=2
    )

    ax.scatter(100*row["delta_macro_f1"], i, s=40)

ax.axvline(0, linestyle="--", color="gray")
ax.set_yticks(range(len(STATS)), STATS["model"])
ax.set_xlabel("Macro-F1 difference (percentage points)")
ax.set_title("Paired group-bootstrap intervals vs matched control")

save_figure(
    fig, "stage7a_08_paired_statistics",
    "Exploratory paired group-bootstrap intervals conditional on the trained models."
)

# Figure 9: Review-length performance.
fig, ax = plt.subplots(figsize=(8, 5))

for family in FAMILIES:
    part = LENGTH_RESULTS[LENGTH_RESULTS["model"] == family]

    ax.plot(
        part["length_bin"],
        part["macro_f1"],
        marker="o", label=family
    )

ax.set_ylim(0, 1)
ax.set_xlabel("Review length (words)")
ax.set_ylabel("Macro-F1")
ax.set_title("Sentiment performance by review length")
ax.legend(frameon=False)

save_figure(
    fig, "stage7a_09_length_performance",
    "Descriptive macro-F1 across review-length bins."
)

# Figure 10: Confusion matrix of current strongest mean control.
cm = confusion_matrix(
    Y, FAMILY_PROBS["matched_control"].argmax(1),
    labels=[0, 1, 2]
)

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, cmap="Blues")

for i, j in itertools.product(range(3), repeat=2):
    ax.text(
        j, i, str(cm[i, j]),
        ha="center", va="center",
        color="white" if cm[i, j] > cm.max()/2 else "black"
    )

ax.set_xticks(range(3), CLASSES)
ax.set_yticks(range(3), CLASSES)
ax.set_xlabel("Predicted sentiment")
ax.set_ylabel("Recorded sentiment")
ax.set_title("Parameter-matched control confusion matrix")

fig.colorbar(im, ax=ax)

save_figure(
    fig, "stage7a_10_confusion_matrix",
    "Validation confusion matrix of the parameter-matched control ensemble."
)

print("New publication figures:", len(SAVED_FIGURES))
print("New aggregate tables:", len(SAVED_TABLES))


# =============================================================================
# STEP 106 — RESEARCH DECISION AND PERMANENT GITHUB BACKUP
# =============================================================================

section("STEP 106 — RESEARCH DECISION AND PERMANENT BACKUP")

import shutil

best_fusion = FUSION_RESULTS.sort_values(
    "macro_f1", ascending=False
).iloc[0]

best_family = RESULTS[
    RESULTS["model"].isin(FAMILIES)
].sort_values("macro_f1", ascending=False).iloc[0]

oracle_all = ORACLE.loc[
    ORACLE["pool"] == "all_nine"
].iloc[0]

SUMMARY = {
    "stage": "7A",
    "validation_samples": len(Y),
    "training_samples": len(TRAIN),
    "confirmed_runs_recovered": len(RUNS),
    "best_family": best_family["model"],
    "best_family_macro_f1": float(best_family["macro_f1"]),
    "best_exploratory_fusion": best_fusion["method"],
    "best_exploratory_fusion_macro_f1": float(best_fusion["macro_f1"]),
    "all_nine_oracle_accuracy": float(oracle_all["oracle_accuracy"]),
    "oracle_correct_when_available_macro_f1": float(
        oracle_all["oracle_correct_when_available_macro_f1"]
    ),
    "unanimously_wrong": int((N_CORRECT == 0).sum()),
    "unanimously_correct": int((N_CORRECT == 9).sum()),
    "training_annotation_candidates": len(train_review),
    "human_verified_label_errors": 0,
    "new_figures": len(SAVED_FIGURES),
    "new_tables": len(SAVED_TABLES),
    "test_evaluated": False,
    "external_evaluated": False,
    "new_model_trained": False,
    "novelty_verified": False,
    "limitations": [
        "Validation previously used for checkpoint selection.",
        "Fusion cross-validation is exploratory, not independent testing.",
        "Oracle requires true labels.",
        "Linguistic cues are heuristic.",
        "Annotation candidates are not verified label errors.",
        "Bootstrap intervals are conditional on existing trained models."
    ]
}

save_json(SUMMARY, AUDIT / "stage7a_summary.json")
save_json(SUMMARY, BACKUP / "stage7a_summary.json")

save_json(SAVED_FIGURES, AUDIT / "figure_manifest.json")

# Private diagnostic predictions, excluding raw review text.
np.savez_compressed(
    PRIVATE / "aligned_confirmation_predictions.npz",
    sample_id=IDS,
    labels=Y,
    group_id=GROUPS,
    models_correct=N_CORRECT,
    **RUNS
)

# Public repository: aggregate tables, figures, source and protocol only.
for p in TABLES.glob("*.csv"):
    dest = BACKUP / "tables" / p.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)

    tex = PUB / "tables/latex" / (p.stem + ".tex")

    if tex.exists():
        dest = BACKUP / "latex_tables" / tex.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tex, dest)

for p in FIGURES.rglob("*"):
    if p.is_file():
        dest = BACKUP / "figures" / p.relative_to(FIGURES)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

for item in SAVED_FIGURES:
    p = PUB / "captions" / f"{item['name']}.txt"
    dest = BACKUP / "captions" / p.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)

save_json(SAVED_FIGURES, BACKUP / "figure_manifest.json")

# Merge publication manifest without replacing historical entries.
manifest_path = PUB / "figure_manifest.csv"

new_manifest = pd.DataFrame(SAVED_FIGURES)

if manifest_path.exists():
    try:
        old_manifest = pd.read_csv(manifest_path)
        new_manifest = pd.concat(
            [old_manifest, new_manifest],
            ignore_index=True
        )
    except Exception:
        pass

new_manifest.to_csv(manifest_path, index=False)

# Private archive: identifiers and predictions, not raw review text.
private_archive = WORK / "sentiment_stage7a_private.zip"

with zipfile.ZipFile(
    private_archive, "w", zipfile.ZIP_DEFLATED
) as z:
    for folder in [TABLES, AUDIT, FIGURES]:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(OUT))

    for p in PRIVATE.glob("*.npz"):
        z.write(p, p.relative_to(OUT))

    # Preserve annotation queue identifiers only.
    for name in [
        "training_annotation_candidates_LOCAL_ONLY.csv",
        "validation_error_review_LOCAL_ONLY.csv"
    ]:
        p = PRIVATE / name

        if p.exists():
            df = pd.read_csv(p)
            df = df.drop(columns=["Bangla"], errors="ignore")

            z.writestr(
                "private/" + name.replace("_LOCAL_ONLY", "_identifiers"),
                df.to_csv(index=False)
            )

publication_archive = WORK / "sentiment_stage7a_publication.zip"

with zipfile.ZipFile(
    publication_archive, "w", zipfile.ZIP_DEFLATED
) as z:
    for p in PUB.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(PUB))

for archive in [private_archive, publication_archive]:
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None

# Create or reuse private release.
release_response = SESSION.get(
    f"{API}/repos/{OWNER}/{PRIVATE_REPO}/releases/tags/{NEW_RELEASE}",
    timeout=60
)

if release_response.status_code == 404:
    release = api_post(
        f"/repos/{OWNER}/{PRIVATE_REPO}/releases",
        {
            "tag_name": NEW_RELEASE,
            "name": "Stage 7A — Bottleneck Discovery",
            "body": "Private development diagnostics. No raw reviews or test predictions.",
            "draft": False,
            "prerelease": False
        }
    )
else:
    release_response.raise_for_status()
    release = release_response.json()

REMOTE_MANIFEST = []

def upload_verified(path):
    path = Path(path)
    digest = sha256(path)

    name = f"{path.stem}_{digest[:12]}{path.suffix}"

    assets = release_assets(release)

    if name in assets:
        asset = assets[name]
    else:
        upload_url = release["upload_url"].split("{")[0]

        with open(path, "rb") as f:
            response = SESSION.post(
                upload_url,
                params={"name": name},
                headers={"Content-Type": "application/octet-stream"},
                data=f,
                timeout=(30, 600)
            )

        response.raise_for_status()
        asset = response.json()

    assert asset["size"] == path.stat().st_size

    remote_digest = asset.get("digest") or ""

    if remote_digest.startswith("sha256:"):
        assert remote_digest.split(":", 1)[1] == digest
    else:
        verify_path = REC / "remote_verification" / name
        download_asset(asset, verify_path)
        assert sha256(verify_path) == digest
        verify_path.unlink()

    REMOTE_MANIFEST.append({
        "name": name,
        "asset_id": asset["id"],
        "size": asset["size"],
        "sha256": digest
    })

    print("Private remote asset verified:", name)

upload_verified(private_archive)
upload_verified(publication_archive)

manifest = AUDIT / "stage7a_remote_manifest.json"

save_json({
    "release": NEW_RELEASE,
    "assets": REMOTE_MANIFEST
}, manifest)

upload_verified(manifest)

shutil.copy2(
    manifest, BACKUP / "stage7a_remote_manifest.json"
)

FINAL_COMMIT = push_backup(
    "Stage 7A: complete bottleneck discovery and verified research backup"
)

# Final reporting.
section("FINAL STAGE 7A RESEARCH REPORT")

print("Authentication: PASS")
print("Exact dataset recovery: PASS")
print("Nine-model prediction alignment: PASS")
print("Grouped fusion cross-validation: PASS")
print("GitHub synchronized: True")
print("Private artifact backup: PASS")
print("Final public commit:", FINAL_COMMIT)

print("\nMODEL PERFORMANCE")
print(
    RESULTS.sort_values(
        "macro_f1", ascending=False
    ).round(5).to_string(index=False)
)

print("\nORACLE CEILING")
print(ORACLE.round(5).to_string(index=False))

print("\nSAMPLE DIFFICULTY")
print(DIFFICULTY.to_string(index=False))

print("\nLINGUISTIC SUBGROUP PERFORMANCE")
print(
    SUBGROUPS.pivot(
        index="subgroup",
        columns="model",
        values="macro_f1"
    ).round(5).to_string()
)

print("\nCROSS-VALIDATED FUSION")
print(FUSION_RESULTS.round(5).to_string(index=False))

print("\nPAIRED STATISTICS")
print(STATS.round(5).to_string(index=False))

print("\nCLASS-WISE PERFORMANCE")
print(
    CLASSWISE.pivot(
        index="class",
        columns="model",
        values="f1-score"
    ).round(5).to_string()
)

print("\nRESEARCH SUMMARY")
print(json.dumps(SUMMARY, indent=2))

print("\nPUBLICATION ASSETS")
print("New figures:", len(SAVED_FIGURES))
print("New tables:", len(SAVED_TABLES))

for fmt in ["png", "pdf", "svg"]:
    print(
        fmt.upper(), "figures:",
        len(list((PUB / "figures" / fmt).rglob(f"*.{fmt}")))
    )

print("Private release:", release["html_url"])
print("Verified remote assets:", len(REMOTE_MANIFEST))

print("\nManual file saving required: NO")
print("Raw review text uploaded: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("New model training: NO")
print("Test evaluated: NO")
print("External evaluation: NO")

print("\nRuntime minutes:", round((time.time() - START) / 60, 2))

print("\nSTAGE 7A COMPLETE")
print("READY FOR STAGE 7B — STRONGER REPRESENTATION LEARNING")

gc.collect()
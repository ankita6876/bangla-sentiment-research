
# =============================================================================
# STAGE 7C — COMPLEMENTARY ERROR DISCOVERY
# STEPS 117–124
# GITHUB FIRST | NO RETRAINING | NO TEST ACCESS | AUTOMATIC BACKUP
# =============================================================================

import os
import re
import gc
import json
import time
import shutil
import hashlib
import zipfile
import base64
import subprocess
import unicodedata
import itertools

from pathlib import Path
from getpass import getpass

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss
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

EXPECTED_COMMIT = "61863df91794b21ec9d0da6fadad1602671d984b"

BASE_TAG = "stage6b-locked-baselines-v1"
TAG_7A = "stage7a-bottleneck-discovery-v1"
TAG_7B = "stage7b-pasnet-screening-v1"
TAG_7C = "stage7c-specialist-discovery-v1"

RAW_SHA = (
    "4580950af06bbfbe5bd79e20957b63a2ed"
    "122af2a6a3e685b10576d3f70241c5"
)

OUT = WORK / "sentiment_research_stage7c"
REC = OUT / "recovered"
AUDIT = OUT / "audit"
TABLES = OUT / "tables"
PRED = OUT / "private_predictions"
FIGURES = OUT / "figures"

PUB = WORK / "sentiment_publication_assets"

REPO = WORK / "sentiment_stage7c_github"
BACKUP = REPO / "research_stage7c"

CLASSES = ["Negative", "Neutral", "Positive"]
LABEL_MAP = dict(zip(CLASSES, range(3)))

SEEDS = [42, 123, 2026]
RNG = np.random.default_rng(42)

for p in [OUT, REC, AUDIT, TABLES, PRED, FIGURES, PUB]:
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

def macro_f1(y, pred):
    return f1_score(
        y, pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

def metrics(y, probs):
    pred = probs.argmax(1)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": macro_f1(y, pred),
        "nll": log_loss(y, probs, labels=[0, 1, 2])
    }

def save_table(rows, name):
    df = pd.DataFrame(rows)

    df.to_csv(TABLES / f"{name}.csv", index=False)

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

# =============================================================================
# STEP 117 — VERIFIED GITHUB CONNECTION
# =============================================================================

section("STEP 117 — VERIFIED GITHUB CONNECTION")

TOKEN = getpass("Enter your working GitHub token: ").strip()

API = "https://api.github.com"

SESSION = requests.Session()

SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

def api_get(path):
    response = SESSION.get(API + path, timeout=90)
    response.raise_for_status()
    return response.json()

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
    response = subprocess.run(
        ["git", *args],
        cwd=str(REPO) if REPO.exists() else str(WORK),
        env=GIT_ENV,
        capture_output=True,
        text=True
    )

    if response.returncode:
        raise RuntimeError(
            response.stderr.replace(TOKEN, "[REDACTED]")
        )

    return response.stdout.strip()

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

    git("add", "research_stage7c")

    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", message)

    git("push", "origin", f"HEAD:{BRANCH}")

    local = git("rev-parse", "HEAD")

    remote = git(
        "ls-remote", "origin", f"refs/heads/{BRANCH}"
    ).split()[0]

    assert local == remote

    print("GitHub synchronized:", local)

    return local

print("Authentication: PASS")
print("Public write access: PASS")
print("Private artifact access: PASS")

# Preserve source before experiments.

history = get_ipython().history_manager.input_hist_raw

source_candidates = [
    s for s in history
    if "STAGE 7C — COMPLEMENTARY ERROR DISCOVERY" in s
    and len(s) > 10000
]

assert source_candidates

source = source_candidates[-1]

assert TOKEN not in source

for path in [
    AUDIT / "stage7c_complete.py",
    BACKUP / "stage7c_complete.py",
    PUB / "scripts/stage7c_complete.py"
]:
    path.write_text(source, encoding="utf-8")

push_public("Stage 7C: preserve specialist discovery source")

# =============================================================================
# STEP 118 — RECOVER ALL REQUIRED PRIVATE ARTIFACTS
# =============================================================================

section("STEP 118 — PRIVATE ARTIFACT RECOVERY")

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

def select_asset(assets, prefix, suffix=".zip"):

    candidates = [
        a for name, a in assets.items()
        if name.startswith(prefix)
        and name.endswith(suffix)
    ]

    assert candidates, (
        f"Missing asset prefix: {prefix}\n"
        f"Available assets: {list(assets)}"
    )

    return max(candidates, key=lambda a: a["created_at"])

BASE_ASSETS = assets_of(get_release(BASE_TAG))
ASSETS_7A = assets_of(get_release(TAG_7A))
ASSETS_7B = assets_of(get_release(TAG_7B))

package = download_asset(
    BASE_ASSETS["stage6b_reproducibility_package.zip"],
    REC / "baseline_package.zip"
)

safe_extract(package, REC / "baseline_package")

asset = select_asset(
    ASSETS_7A,
    "sentiment_stage7a_private_"
)

path = download_asset(asset, REC / asset["name"])

safe_extract(path, REC / "stage7a_private")

asset = select_asset(
    ASSETS_7B,
    "sentiment_stage7b_compact_"
)

path = download_asset(asset, REC / asset["name"])

safe_extract(path, REC / "stage7b_private")

asset = select_asset(
    ASSETS_7B,
    "sentiment_stage7b_publication_"
)

path = download_asset(asset, REC / asset["name"])

safe_extract(path, PUB)

print("\nPrevious publication assets:")

for fmt in ["png", "pdf", "svg"]:
    print(
        fmt.upper(),
        len(list((PUB / "figures" / fmt).rglob(f"*.{fmt}")))
    )

print("Large model checkpoints downloaded: NO")

# =============================================================================
# STEP 119 — LOCKED DATASET AND PREDICTION ALIGNMENT
# =============================================================================

section("STEP 119 — EXACT PREDICTION ALIGNMENT")

raw_path = next(
    (
        p for p in INPUT.rglob("BECS Dataset.csv")
        if sha256(p) == RAW_SHA
    ),
    None
)

assert raw_path is not None, "Attach the original BECS dataset."

raw = pd.read_csv(raw_path)

def normalize_text(text):

    text = unicodedata.normalize("NFC", str(text))

    text = re.sub(
        r"[\u200b\u200c\u200d\u2060\ufeff]",
        "",
        text
    )

    return re.sub(r"\s+", " ", text).strip().casefold()

raw["text_norm"] = raw["Bangla"].map(normalize_text)

raw["Label"] = (
    raw["Label"].astype(str).str.strip().str.title()
)

raw["sample_id"] = raw["text_norm"].map(
    lambda s: hashlib.sha256(s.encode()).hexdigest()
)

raw["label_id"] = raw["Label"].map(LABEL_MAP)

conflicts = set(
    raw.groupby("sample_id")["label_id"]
       .nunique()
       .loc[lambda x: x > 1]
       .index
)

clean = (
    raw.loc[~raw["sample_id"].isin(conflicts)]
       .drop_duplicates("sample_id")
       .set_index("sample_id")
)

assert len(clean) == 12893

DATA = {}

for split, expected in [("train", 9204), ("val", 1840)]:

    candidates = list(
        (REC / "baseline_package").rglob(
            f"seed42_{split}.csv"
        )
    )

    assert len(candidates) == 1

    membership = pd.read_csv(candidates[0])

    if "partition_order" in membership:

        membership = membership.sort_values(
            "partition_order"
        )

    ids = membership["sample_id"].astype(str).tolist()

    df = clean.loc[ids].reset_index()

    df["group_id"] = (
        membership["group_id"].astype(str).to_numpy()
    )

    assert len(df) == expected

    assert df["sample_id"].tolist() == ids

    assert (
        df["Label"].str.casefold().to_numpy()
        == membership["Label"].astype(str)
                     .str.strip().str.casefold().to_numpy()
    ).all()

    df["word_count"] = df["text_norm"].str.split().str.len()

    DATA[split] = df

assert not set(DATA["train"]["sample_id"]).intersection(
    DATA["val"]["sample_id"]
)

assert not set(DATA["train"]["group_id"]).intersection(
    DATA["val"]["group_id"]
)

VAL = DATA["val"]

IDS = VAL["sample_id"].to_numpy(dtype=str)
Y = VAL["label_id"].to_numpy(dtype=int)
GROUPS = VAL["group_id"].to_numpy(dtype=str)

N = len(Y)

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

NEG_MASK = VAL["text_norm"].map(
    lambda s: bool(neg_pattern.search(s))
).to_numpy()

CON_MASK = VAL["text_norm"].map(
    lambda s: bool(con_pattern.search(s))
).to_numpy()

LONG_THRESHOLD = DATA["train"]["word_count"].quantile(.75)

LONG_MASK = (
    VAL["word_count"] > LONG_THRESHOLD
).to_numpy()

MASKS = {
    "all": np.ones(N, dtype=bool),
    "negation": NEG_MASK,
    "contrast": CON_MASK,
    "both_operators": NEG_MASK & CON_MASK,
    "any_operator": NEG_MASK | CON_MASK,
    "no_operator": ~(NEG_MASK | CON_MASK),
    "long_text": LONG_MASK,
    "short_text": ~LONG_MASK,
    "negative_class": Y == 0,
    "neutral_class": Y == 1,
    "positive_class": Y == 2
}

PROBS = {}
PROVENANCE = []

def alignment_order(source_ids, source_labels):

    source_ids = np.asarray(source_ids).astype(str)

    assert len(source_ids) == len(set(source_ids))
    assert set(source_ids) == set(IDS)

    lookup = {
        sid: i for i, sid in enumerate(source_ids)
    }

    order = np.array([lookup[sid] for sid in IDS])

    assert np.array_equal(
        np.asarray(source_labels)[order], Y
    )

    return order

def as_probabilities(array):

    array = np.asarray(array, dtype=np.float64)

    assert array.shape == (N, 3)
    assert np.isfinite(array).all()

    if (
        (array >= 0).all()
        and (array <= 1).all()
        and np.allclose(array.sum(1), 1, atol=1e-5)
    ):

        return array / array.sum(1, keepdims=True)

    return softmax(array, axis=1)

# Recover the previous nine-model confirmation cohort.

previous = list(
    (REC / "stage7a_private").rglob(
        "aligned_confirmation_predictions.npz"
    )
)

assert len(previous) == 1

OLD_NAMES = []

with np.load(previous[0], allow_pickle=False) as z:

    order = alignment_order(
        z["sample_id"], z["labels"]
    )

    for family in [
        "standard",
        "fixed_fusion",
        "matched_control"
    ]:

        for seed in SEEDS:

            name = f"{family}_seed{seed}"

            assert name in z.files, (
                name, z.files
            )

            PROBS[name] = as_probabilities(
                z[name][order]
            )

            OLD_NAMES.append(name)

            PROVENANCE.append({
                "model": name,
                "source": "stage7a_confirmation",
                "source_sha256": sha256(previous[0]),
                "alignment": "PASS"
            })

OLD_ENSEMBLE = np.mean(
    [PROBS[n] for n in OLD_NAMES],
    axis=0
)

old_f1 = metrics(Y, OLD_ENSEMBLE)["macro_f1"]

assert abs(old_f1 - .8215240777580629) < 1e-6

# Recover six Stage 7B models.

NEW_NAMES = [
    "base_standard",
    "base_supcon",
    "base_prototype",
    "base_pasnet",
    "large_standard",
    "large_pasnet"
]

EXPECTED_7B = {
    "base_standard": .80913,
    "base_supcon": .81020,
    "base_prototype": .81442,
    "base_pasnet": .81639,
    "large_standard": .81289,
    "large_pasnet": .81735
}

for name in NEW_NAMES:

    filename = f"{name}_seed42_val.npz"

    paths = list(
        (REC / "stage7b_private").rglob(filename)
    )

    if len(paths) == 1:

        path = paths[0]

        # Verify against remote asset digest when available.
        remote_digest = (
            ASSETS_7B[filename].get("digest") or ""
        )

        if remote_digest.startswith("sha256:"):

            assert sha256(path) == remote_digest.split(":", 1)[1]

    else:

        path = download_asset(
            ASSETS_7B[filename],
            REC / filename
        )

    with np.load(path, allow_pickle=False) as z:

        order = alignment_order(
            z["sample_id"], z["labels"]
        )

        PROBS[name] = softmax(
            z["logits"][order].astype(np.float64),
            axis=1
        )

    f1 = metrics(Y, PROBS[name])["macro_f1"]

    assert abs(f1 - EXPECTED_7B[name]) < 1e-5

    PROVENANCE.append({
        "model": name,
        "source": "stage7b_screening",
        "source_sha256": sha256(path),
        "alignment": "PASS"
    })

NAMES = list(PROBS)

assert len(NAMES) == 15

P = np.stack([PROBS[n] for n in NAMES])

HARD = P.argmax(2)

CORRECT = HARD == Y[None, :]

CONFIDENCE = P.max(2)

ERROR_COUNT = (~CORRECT).sum(0)

save_table(
    PROVENANCE,
    "stage7c_01_prediction_provenance"
)

print("Aligned models:", len(NAMES))
print("Validation samples:", N)
print("Previous ensemble reproduction: PASS")
print("All six Stage 7B metric checks: PASS")
print("Test predictions accessed: NO")

# =============================================================================
# STEP 120 — ENSEMBLE AND ORACLE ANALYSIS
# =============================================================================

section("STEP 120 — ENSEMBLE AND ORACLE ANALYSIS")

ENSEMBLES = {
    "previous_nine": OLD_ENSEMBLE,

    "new_six": np.mean(
        [PROBS[n] for n in NEW_NAMES],
        axis=0
    ),

    "all_fifteen": P.mean(0),

    "base_pasnet_large_pasnet": np.mean(
        [
            PROBS["base_pasnet"],
            PROBS["large_pasnet"]
        ],
        axis=0
    ),

    "prototype_family": np.mean(
        [
            PROBS["base_prototype"],
            PROBS["base_pasnet"],
            PROBS["large_pasnet"]
        ],
        axis=0
    ),

    "old_plus_pasnet_family": np.mean(
        [
            *[PROBS[n] for n in OLD_NAMES],
            PROBS["base_prototype"],
            PROBS["base_pasnet"],
            PROBS["large_pasnet"]
        ],
        axis=0
    )
}

# Equal family weighting avoids giving greater influence
# to architectures merely because more seeds were trained.

FAMILIES = {
    "standard": [
        f"standard_seed{s}" for s in SEEDS
    ],

    "fixed_fusion": [
        f"fixed_fusion_seed{s}" for s in SEEDS
    ],

    "matched_control": [
        f"matched_control_seed{s}" for s in SEEDS
    ],

    **{name: [name] for name in NEW_NAMES}
}

FAMILY_PROBS = {
    name: np.mean([PROBS[n] for n in members], axis=0)
    for name, members in FAMILIES.items()
}

ENSEMBLES["equal_family_weight"] = np.mean(
    list(FAMILY_PROBS.values()),
    axis=0
)

# Hard voting control.

votes = np.eye(3)[HARD].sum(0)

ENSEMBLES["majority_vote_15"] = (
    votes + 1e-8 * P.mean(0)
) / (len(NAMES) + 1e-8)

ALL = {**PROBS, **ENSEMBLES}

RESULTS = save_table(
    [
        {
            "model": name,
            "type": (
                "ensemble" if name in ENSEMBLES else "individual"
            ),
            **metrics(Y, probs)
        }
        for name, probs in ALL.items()
    ],
    "stage7c_02_all_model_results"
).sort_values("macro_f1", ascending=False)

print(RESULTS.round(5).to_string(index=False))

old_pred = OLD_ENSEMBLE.argmax(1)

old_correct = old_pred == Y

any_correct = CORRECT.any(0)

all_correct = CORRECT.all(0)

# Oracle uses ground truth to select a correct model.
# It is NOT a deployable classifier.
# Its accuracy is an upper bound for hard model selection.
# Macro-F1 below is a particular oracle construction.

oracle_pred = np.where(
    any_correct,
    Y,
    P.mean(0).argmax(1)
)

# For a strict model-selection oracle, choose an actual
# model prediction when all models are wrong.

all_wrong = ~any_correct

oracle_pred[all_wrong] = HARD[
    CONFIDENCE[:, all_wrong].argmax(0),
    np.flatnonzero(all_wrong)
]

assert (
    oracle_pred[all_wrong] != Y[all_wrong]
).all()

oracle_accuracy = accuracy_score(Y, oracle_pred)
oracle_f1 = macro_f1(Y, oracle_pred)

recoverable_old_errors = (~old_correct) & any_correct

unrecoverable_old_errors = (~old_correct) & (~any_correct)

ORACLE = save_table(
    [{
        "validation_samples": N,
        "all_models_correct": int(all_correct.sum()),
        "all_models_wrong": int(all_wrong.sum()),
        "mixed_correctness": int(
            (~all_correct & ~all_wrong).sum()
        ),
        "previous_ensemble_errors": int(
            (~old_correct).sum()
        ),
        "previous_errors_recoverable_by_model_selection": int(
            recoverable_old_errors.sum()
        ),
        "previous_errors_unrecoverable_by_model_selection": int(
            unrecoverable_old_errors.sum()
        ),
        "oracle_accuracy": oracle_accuracy,
        "oracle_macro_f1_diagnostic": oracle_f1
    }],
    "stage7c_03_oracle_analysis"
)

print("\nORACLE ANALYSIS")
print(ORACLE.to_string(index=False))

# =============================================================================
# STEP 121 — MODEL COMPLEMENTARITY AND SPECIALIST DISCOVERY
# =============================================================================

section("STEP 121 — SPECIALIST DISCOVERY")

PAIR_ROWS = []

for i, j in itertools.combinations(range(len(NAMES)), 2):

    a = CORRECT[i]
    b = CORRECT[j]

    pair_probs = (P[i] + P[j]) / 2

    pair_oracle = a | b

    PAIR_ROWS.append({
        "model_a": NAMES[i],
        "model_b": NAMES[j],

        "prediction_disagreement": float(
            (HARD[i] != HARD[j]).mean()
        ),

        "both_correct": int((a & b).sum()),
        "only_a_correct": int((a & ~b).sum()),
        "only_b_correct": int((~a & b).sum()),
        "both_wrong": int((~a & ~b).sum()),

        "pair_oracle_accuracy": float(
            pair_oracle.mean()
        ),

        "pair_ensemble_macro_f1": macro_f1(
            Y, pair_probs.argmax(1)
        )
    })

PAIRS = save_table(
    PAIR_ROWS,
    "stage7c_04_pairwise_complementarity"
)

print("\nTOP COMPLEMENTARY MODEL PAIRS")

print(
    PAIRS.sort_values(
        "pair_ensemble_macro_f1",
        ascending=False
    ).head(15).round(5).to_string(index=False)
)

RESCUE_ROWS = []

for i, name in enumerate(NAMES):

    pred = HARD[i]

    gained = (~old_correct) & CORRECT[i]

    lost = old_correct & (~CORRECT[i])

    unique = CORRECT[i] & (CORRECT.sum(0) == 1)

    RESCUE_ROWS.append({
        "model": name,
        "rescued_previous_errors": int(gained.sum()),
        "lost_previous_correct": int(lost.sum()),
        "unique_correct_examples": int(unique.sum()),
        "net_correct_change": int(
            gained.sum() - lost.sum()
        ),
        "rescue_rate": float(
            gained.sum() / max((~old_correct).sum(), 1)
        )
    })

RESCUES = save_table(
    RESCUE_ROWS,
    "stage7c_05_specialist_rescues"
)

print("\nSPECIALIST RESCUE CAPABILITY")

print(
    RESCUES.sort_values(
        "rescued_previous_errors",
        ascending=False
    ).to_string(index=False)
)

SUBGROUP_ROWS = []

for name, probs in ALL.items():

    pred = probs.argmax(1)

    for subgroup, mask in MASKS.items():

        if not mask.any():
            continue

        row = {
            "model": name,
            "subgroup": subgroup,
            "n": int(mask.sum()),
            "accuracy": accuracy_score(
                Y[mask], pred[mask]
            ),
            "rescued_previous_errors": int(
                (
                    mask
                    & (~old_correct)
                    & (pred == Y)
                ).sum()
            ),
            "lost_previous_correct": int(
                (
                    mask
                    & old_correct
                    & (pred != Y)
                ).sum()
            )
        }

        # Single true-class subsets use recall, not macro-F1.

        if subgroup.endswith("_class"):

            row["macro_f1"] = np.nan

            row["class_recall"] = row["accuracy"]

        else:

            row["macro_f1"] = macro_f1(
                Y[mask], pred[mask]
            )

            row["class_recall"] = np.nan

        SUBGROUP_ROWS.append(row)

SUBGROUPS = save_table(
    SUBGROUP_ROWS,
    "stage7c_06_subgroup_specialization"
)

CLASS_ROWS = []

for name, probs in ALL.items():

    report = classification_report(
        Y,
        probs.argmax(1),
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

CLASSWISE = save_table(
    CLASS_ROWS,
    "stage7c_07_classwise_results"
)

BEST_SUBGROUP_ROWS = []

for subgroup in [
    "negation", "contrast",
    "any_operator", "long_text",
    "short_text", "no_operator"
]:

    part = SUBGROUPS[
        (SUBGROUPS["subgroup"] == subgroup)
        & SUBGROUPS["model"].isin(NAMES)
    ].sort_values("macro_f1", ascending=False)

    best = part.iloc[0]

    baseline = SUBGROUPS[
        (SUBGROUPS["subgroup"] == subgroup)
        & (SUBGROUPS["model"] == "previous_nine")
    ].iloc[0]

    BEST_SUBGROUP_ROWS.append({
        "subgroup": subgroup,
        "n": int(best["n"]),
        "best_individual_model": best["model"],
        "best_macro_f1": best["macro_f1"],
        "previous_ensemble_macro_f1": baseline["macro_f1"],
        "delta_macro_f1": (
            best["macro_f1"] - baseline["macro_f1"]
        )
    })

BEST_SUBGROUPS = save_table(
    BEST_SUBGROUP_ROWS,
    "stage7c_08_best_subgroup_specialists"
)

print("\nBEST SUBGROUP SPECIALISTS")
print(BEST_SUBGROUPS.round(5).to_string(index=False))

# =============================================================================
# STEP 122 — DIFFICULTY, DISAGREEMENT AND DATA QUALITY
# =============================================================================

section("STEP 122 — ERROR STRUCTURE AND DATA QUALITY")

mean_probs = P.mean(0)

predictive_entropy = -(
    mean_probs * np.log(
        np.clip(mean_probs, 1e-12, 1)
    )
).sum(1) / np.log(3)

individual_entropy = -(
    P * np.log(np.clip(P, 1e-12, 1))
).sum(2).mean(0) / np.log(3)

js_disagreement = np.maximum(
    predictive_entropy - individual_entropy,
    0
)

vote_distribution = votes / len(NAMES)

vote_entropy = -(
    vote_distribution
    * np.log(np.clip(vote_distribution, 1e-12, 1))
).sum(1) / np.log(3)

sorted_probs = np.sort(mean_probs, axis=1)

margin = sorted_probs[:, -1] - sorted_probs[:, -2]

ERROR_TYPES = np.select(
    [
        all_correct,
        all_wrong,
        recoverable_old_errors
    ],
    [
        "all_models_correct",
        "all_models_wrong",
        "recoverable_previous_error"
    ],
    default="other_mixed_correctness"
)

PRIVATE_ERRORS = pd.DataFrame({
    "sample_id": IDS,
    "group_id": GROUPS,
    "label": Y,
    "previous_prediction": old_pred,
    "all15_prediction": mean_probs.argmax(1),
    "correct_model_count": CORRECT.sum(0),
    "incorrect_model_count": ERROR_COUNT,
    "predictive_entropy": predictive_entropy,
    "js_disagreement": js_disagreement,
    "vote_entropy": vote_entropy,
    "confidence_margin": margin,
    "word_count": VAL["word_count"],
    "negation": NEG_MASK,
    "contrast": CON_MASK,
    "long_text": LONG_MASK,
    "error_type": ERROR_TYPES
})

for i, name in enumerate(NAMES):

    PRIVATE_ERRORS[f"correct_{name}"] = CORRECT[i]

PRIVATE_ERRORS.to_csv(
    PRED / "validation_error_diagnostics.csv",
    index=False
)

DIFFICULTY_ROWS = []

for count in range(16):

    mask = ERROR_COUNT == count

    if not mask.any():
        continue

    DIFFICULTY_ROWS.append({
        "incorrect_model_count": count,
        "n": int(mask.sum()),
        "mean_entropy": predictive_entropy[mask].mean(),
        "mean_js_disagreement": js_disagreement[mask].mean(),
        "mean_vote_entropy": vote_entropy[mask].mean(),
        "mean_margin": margin[mask].mean(),
        "mean_word_count": VAL.loc[mask, "word_count"].mean(),
        "negation_fraction": NEG_MASK[mask].mean(),
        "contrast_fraction": CON_MASK[mask].mean(),
        "neutral_fraction": (Y[mask] == 1).mean()
    })

DIFFICULTY = save_table(
    DIFFICULTY_ROWS,
    "stage7c_09_difficulty_distribution"
)

ERROR_GROUP_ROWS = []

for name in np.unique(ERROR_TYPES):

    mask = ERROR_TYPES == name

    ERROR_GROUP_ROWS.append({
        "error_type": name,
        "n": int(mask.sum()),
        "mean_entropy": predictive_entropy[mask].mean(),
        "mean_js_disagreement": js_disagreement[mask].mean(),
        "mean_margin": margin[mask].mean(),
        "mean_word_count": VAL.loc[mask, "word_count"].mean(),
        "neutral_fraction": (Y[mask] == 1).mean(),
        "negation_fraction": NEG_MASK[mask].mean(),
        "contrast_fraction": CON_MASK[mask].mean(),
        "long_fraction": LONG_MASK[mask].mean()
    })

ERROR_GROUPS = save_table(
    ERROR_GROUP_ROWS,
    "stage7c_10_error_group_profiles"
)

# Quantify how well label-free uncertainty signals identify errors.
# These are diagnostic AUROCs, not trained routing results.

from sklearn.metrics import roc_auc_score

DETECTION_ROWS = []

for target_name, target in {
    "previous_ensemble_error": ~old_correct,
    "all15_ensemble_error": mean_probs.argmax(1) != Y,
    "all_models_wrong": all_wrong,
    "recoverable_previous_error": recoverable_old_errors
}.items():

    for signal_name, score in {
        "predictive_entropy": predictive_entropy,
        "js_disagreement": js_disagreement,
        "vote_entropy": vote_entropy,
        "one_minus_margin": 1 - margin,
        "one_minus_confidence": 1 - mean_probs.max(1)
    }.items():

        if len(np.unique(target)) < 2:
            continue

        DETECTION_ROWS.append({
            "target": target_name,
            "signal": signal_name,
            "positive_samples": int(target.sum()),
            "auroc": roc_auc_score(target, score)
        })

DETECTION = save_table(
    DETECTION_ROWS,
    "stage7c_11_uncertainty_detection"
)

# Create a private, blinded review queue.
# Recorded labels and model predictions are intentionally
# excluded from the annotation sheet.

priority = np.flatnonzero(
    all_wrong | recoverable_old_errors
)

priority = sorted(
    priority,
    key=lambda i: (
        bool(all_wrong[i]),
        float(mean_probs[i].max()),
        float(js_disagreement[i])
    ),
    reverse=True
)

selected = np.array(priority[:300], dtype=int)

QUEUE = pd.DataFrame({
    "sample_id": IDS[selected],
    "Bangla": VAL.iloc[selected]["Bangla"].to_numpy(),
    "human_sentiment": "",
    "ambiguity": "",
    "annotation_notes": ""
})

QUEUE = QUEUE.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

QUEUE.to_csv(
    PRED / "blinded_annotation_candidates.csv",
    index=False
)

# No human annotation is assumed to have happened.

# Audit conflicting normalized labels in the original dataset.
# No raw review text is placed in public tables.

conflict_rows = raw[
    raw["sample_id"].isin(conflicts)
]

DATA_QUALITY = save_table(
    [{
        "raw_rows": len(raw),
        "normalized_unique_groups": raw["sample_id"].nunique(),
        "conflicting_label_groups": len(conflicts),
        "rows_in_conflicting_groups": len(conflict_rows),
        "clean_unique_samples": len(clean),
        "validation_all_models_wrong": int(all_wrong.sum()),
        "blinded_review_candidates": len(QUEUE),
        "human_annotations_completed": 0
    }],
    "stage7c_12_data_quality_audit"
)

print("\nERROR GROUP PROFILES")
print(ERROR_GROUPS.round(5).to_string(index=False))

print("\nUNCERTAINTY DETECTION")
print(
    DETECTION.sort_values(
        ["target", "auroc"],
        ascending=[True, False]
    ).round(5).to_string(index=False)
)

print("\nDATA QUALITY AUDIT")
print(DATA_QUALITY.to_string(index=False))

# =============================================================================
# EXPLORATORY PAIRED GROUP-BOOTSTRAP COMPARISONS
# =============================================================================

group_indices = [
    np.flatnonzero(GROUPS == group)
    for group in np.unique(GROUPS)
]

def fast_f1(y, pred):

    cm = np.bincount(
        3 * y + pred,
        minlength=9
    ).reshape(3, 3)

    tp = np.diag(cm)

    denominator = cm.sum(0) + cm.sum(1)

    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros(3),
        where=denominator > 0
    ).mean()

BOOTSTRAPS = 2000

BOOT = [
    np.concatenate([
        group_indices[j]
        for j in RNG.integers(
            0,
            len(group_indices),
            len(group_indices)
        )
    ])
    for _ in range(BOOTSTRAPS)
]

STATS_ROWS = []

for name, probs in ENSEMBLES.items():

    if name == "previous_nine":
        continue

    pred = probs.argmax(1)

    differences = np.array([
        fast_f1(Y[idx], pred[idx])
        - fast_f1(Y[idx], old_pred[idx])
        for idx in BOOT
    ])

    STATS_ROWS.append({
        "model": name,
        "reference": "previous_nine",
        "delta_macro_f1": (
            fast_f1(Y, pred) - fast_f1(Y, old_pred)
        ),
        "ci95_low": np.quantile(differences, .025),
        "ci95_high": np.quantile(differences, .975),
        "gained_correct": int(
            ((pred == Y) & (~old_correct)).sum()
        ),
        "lost_correct": int(
            ((pred != Y) & old_correct).sum()
        ),
        "analysis": "exploratory_validation"
    })

STATS = save_table(
    STATS_ROWS,
    "stage7c_13_paired_ensemble_statistics"
)

print("\nPAIRED ENSEMBLE STATISTICS")
print(STATS.round(5).to_string(index=False))

# =============================================================================
# STEP 123 — PUBLICATION FIGURES
# =============================================================================

section("STEP 123 — PUBLICATION FIGURES AND GITHUB BACKUP")

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

        shutil.copy2(
            path,
            PUB / "figures" / fmt / path.name
        )

        dest = BACKUP / "figures" / fmt / path.name

        dest.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(path, dest)

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

    save_json(
        FIGURE_MANIFEST,
        BACKUP / "figure_manifest.json"
    )

    # Immediate permanent backup after every figure.

    push_public("Stage 7C figure: " + name)

    plt.show()
    plt.close(fig)

# Figure 1 — All models and ensembles.

fig, ax = plt.subplots(figsize=(10, 9))

plot = RESULTS.sort_values("macro_f1")

colors = [
    "#0f766e" if t == "ensemble" else "#64748b"
    for t in plot["type"]
]

ax.barh(
    plot["model"],
    100 * plot["macro_f1"],
    color=colors
)

ax.set_xlabel("Validation macro-F1 (%)")
ax.set_xlim(0, 100)

for i, v in enumerate(plot["macro_f1"]):

    ax.text(
        100*v + .3,
        i,
        f"{100*v:.2f}",
        va="center",
        fontsize=8
    )

save_figure(
    fig,
    "stage7c_01_model_comparison",
    "Exploratory comparison of 15 trained models and probability ensembles."
)

# Figure 2 — Pairwise disagreement matrix.

disagreement_matrix = np.mean(
    HARD[:, None, :] != HARD[None, :, :],
    axis=2
)

fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(
    disagreement_matrix,
    cmap="YlOrRd",
    vmin=0
)

ax.set_xticks(
    range(len(NAMES)),
    NAMES,
    rotation=65,
    ha="right",
    fontsize=8
)

ax.set_yticks(
    range(len(NAMES)),
    NAMES,
    fontsize=8
)

for i in range(len(NAMES)):

    for j in range(len(NAMES)):

        ax.text(
            j, i,
            f"{disagreement_matrix[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=6
        )

fig.colorbar(im, ax=ax, label="Disagreement fraction")

save_figure(
    fig,
    "stage7c_02_pairwise_disagreement",
    "Pairwise disagreement between hard sentiment predictions."
)

# Figure 3 — Pairwise error overlap.

double_fault = (
    (~CORRECT[:, None, :])
    & (~CORRECT[None, :, :])
).mean(2)

fig, ax = plt.subplots(figsize=(12, 10))

im = ax.imshow(double_fault, cmap="Blues", vmin=0)

ax.set_xticks(
    range(len(NAMES)),
    NAMES,
    rotation=65,
    ha="right",
    fontsize=8
)

ax.set_yticks(
    range(len(NAMES)),
    NAMES,
    fontsize=8
)

fig.colorbar(im, ax=ax, label="Shared error fraction")

save_figure(
    fig,
    "stage7c_03_shared_error_matrix",
    "Fraction of validation samples misclassified by both models."
)

# Figure 4 — Rescue and loss analysis.

plot = RESCUES.sort_values(
    "rescued_previous_errors"
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    plot["model"],
    plot["rescued_previous_errors"],
    label="Rescued errors",
    color="#0f766e"
)

ax.barh(
    plot["model"],
    -plot["lost_previous_correct"],
    label="Lost correct predictions",
    color="#b45309"
)

ax.axvline(0, color="gray", linewidth=1)
ax.set_xlabel("Number of validation examples")
ax.legend(frameon=False)

save_figure(
    fig,
    "stage7c_04_specialist_rescues",
    "Individual model gains and losses relative to the previous nine-model ensemble."
)

# Figure 5 — Difficulty distribution.

fig, ax = plt.subplots(figsize=(9, 5))

counts = np.bincount(
    ERROR_COUNT,
    minlength=16
)

ax.bar(range(16), counts)

ax.set_xticks(range(16))
ax.set_xlabel("Number of models making an error")
ax.set_ylabel("Validation samples")

save_figure(
    fig,
    "stage7c_05_example_difficulty",
    "Distribution of model error counts across validation samples."
)

# Figure 6 — Subgroup specialization.

chosen = [
    "previous_nine",
    "all_fifteen",
    "equal_family_weight",
    "base_prototype",
    "base_pasnet",
    "large_standard",
    "large_pasnet"
]

pivot = SUBGROUPS[
    SUBGROUPS["model"].isin(chosen)
    & SUBGROUPS["macro_f1"].notna()
].pivot(
    index="subgroup",
    columns="model",
    values="macro_f1"
)

fig, ax = plt.subplots(figsize=(12, 6))

im = ax.imshow(
    pivot.to_numpy(),
    cmap="YlGnBu",
    vmin=0,
    vmax=1
)

ax.set_xticks(
    range(len(pivot.columns)),
    pivot.columns,
    rotation=45,
    ha="right"
)

ax.set_yticks(
    range(len(pivot.index)),
    pivot.index
)

for i in range(len(pivot.index)):

    for j in range(len(pivot.columns)):

        v = pivot.iloc[i, j]

        ax.text(
            j, i, f"{v:.3f}",
            ha="center",
            va="center",
            color="white" if v > .7 else "black",
            fontsize=8
        )

fig.colorbar(im, ax=ax, label="Macro-F1")

save_figure(
    fig,
    "stage7c_06_subgroup_specialization",
    "Linguistic subgroup performance of selected models and ensembles."
)

# Figure 7 — Class-specific performance.

fig, ax = plt.subplots(figsize=(12, 5))

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
ax.legend(frameon=False, fontsize=8)

save_figure(
    fig,
    "stage7c_07_classwise_specialization",
    "Class-specific sentiment discrimination."
)

# Figure 8 — Oracle diagnostic.

fig, ax = plt.subplots(figsize=(8, 5))

labels = [
    "Previous ensemble",
    "15-model ensemble",
    "Model-selection oracle"
]

values = [
    old_correct.mean(),
    (mean_probs.argmax(1) == Y).mean(),
    oracle_accuracy
]

ax.bar(
    labels,
    np.array(values)*100,
    color=["#64748b", "#0f766e", "#b45309"]
)

ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")

for i, v in enumerate(values):

    ax.text(
        i, 100*v + 1,
        f"{100*v:.2f}%",
        ha="center"
    )

save_figure(
    fig,
    "stage7c_08_oracle_gap",
    "Oracle accuracy is a non-deployable ground-truth-assisted selection diagnostic."
)

# Figure 9 — Error detectability.

pivot = DETECTION.pivot(
    index="signal",
    columns="target",
    values="auroc"
)

fig, ax = plt.subplots(figsize=(11, 5))

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

        ax.text(
            j, i,
            f"{pivot.iloc[i,j]:.3f}",
            ha="center",
            va="center",
            fontsize=9
        )

fig.colorbar(im, ax=ax, label="AUROC")

save_figure(
    fig,
    "stage7c_09_uncertainty_detection",
    "Diagnostic ability of label-free uncertainty signals to identify errors."
)

# Figure 10 — Paired ensemble effects.

fig, ax = plt.subplots(figsize=(10, 5))

labels = []

for i, row in STATS.reset_index(drop=True).iterrows():

    ax.plot(
        [
            100*row["ci95_low"],
            100*row["ci95_high"]
        ],
        [i, i],
        linewidth=2
    )

    ax.scatter(
        100*row["delta_macro_f1"],
        i
    )

    labels.append(row["model"])

ax.axvline(0, color="gray", linestyle="--")

ax.set_yticks(range(len(labels)), labels)

ax.set_xlabel(
    "Macro-F1 difference vs previous ensemble (percentage points)"
)

save_figure(
    fig,
    "stage7c_10_paired_ensemble_effects",
    "Exploratory paired group-bootstrap confidence intervals on development data."
)

# Figure 11 — Error structure.

fig, ax = plt.subplots(figsize=(10, 5))

ERROR_GROUPS.set_index("error_type")[
    [
        "neutral_fraction",
        "negation_fraction",
        "contrast_fraction",
        "long_fraction"
    ]
].plot(
    kind="bar",
    ax=ax,
    rot=20
)

ax.set_ylabel("Fraction of examples")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8)

save_figure(
    fig,
    "stage7c_11_error_group_structure",
    "Linguistic and class composition of model error categories."
)

# Figure 12 — Difficulty versus uncertainty.

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    DIFFICULTY["incorrect_model_count"],
    DIFFICULTY["mean_entropy"],
    marker="o",
    label="Predictive entropy"
)

ax.plot(
    DIFFICULTY["incorrect_model_count"],
    DIFFICULTY["mean_js_disagreement"],
    marker="s",
    label="JS disagreement"
)

ax.plot(
    DIFFICULTY["incorrect_model_count"],
    DIFFICULTY["mean_vote_entropy"],
    marker="^",
    label="Vote entropy"
)

ax.set_xlabel("Number of incorrect models")
ax.set_ylabel("Mean uncertainty signal")
ax.legend(frameon=False)

save_figure(
    fig,
    "stage7c_12_difficulty_uncertainty",
    "Relationship between model error counts and uncertainty signals."
)

# =============================================================================
# STEP 124 — PERMANENT PRIVATE AND PUBLIC BACKUP
# =============================================================================

section("STEP 124 — VERIFIED PERMANENT BACKUP")

response = SESSION.get(
    f"{API}/repos/{OWNER}/{PRIVATE_REPO}/releases/tags/{TAG_7C}",
    timeout=90
)

if response.status_code == 404:

    response = SESSION.post(
        f"{API}/repos/{OWNER}/{PRIVATE_REPO}/releases",
        json={
            "tag_name": TAG_7C,
            "name": "Stage 7C — Specialist discovery",
            "body": (
                "Private validation diagnostics and "
                "publication artifacts. No test evaluation."
            )
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

    remote = assets_of(RELEASE)

    if name in remote:

        asset = remote[name]

    else:

        with open(path, "rb") as f:

            response = SESSION.post(
                RELEASE["upload_url"].split("{")[0],
                params={"name": name},
                headers={
                    "Content-Type": "application/octet-stream"
                },
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

# Preserve aligned probabilities for future experiments.

np.savez_compressed(
    PRED / "stage7c_aligned_15_models.npz",
    sample_id=IDS,
    labels=Y,
    model_names=np.array(NAMES, dtype=str),
    probabilities=P,
    group_id=GROUPS,
    **{
        f"ensemble_{name}": probs
        for name, probs in ENSEMBLES.items()
    }
)

best = RESULTS.iloc[0]

best_ensemble = RESULTS[
    RESULTS["type"] == "ensemble"
].iloc[0]

best_individual = RESULTS[
    RESULTS["type"] == "individual"
].iloc[0]

SUMMARY = {
    "stage": "7C",
    "models_recovered": len(NAMES),
    "validation_samples": N,
    "best_overall_model": best["model"],
    "best_overall_macro_f1": float(best["macro_f1"]),
    "best_individual_model": best_individual["model"],
    "best_individual_macro_f1": float(
        best_individual["macro_f1"]
    ),
    "best_ensemble": best_ensemble["model"],
    "best_ensemble_macro_f1": float(
        best_ensemble["macro_f1"]
    ),
    "previous_ensemble_macro_f1": old_f1,
    "all15_ensemble_macro_f1": metrics(
        Y, ENSEMBLES["all_fifteen"]
    )["macro_f1"],
    "oracle_accuracy": float(oracle_accuracy),
    "oracle_macro_f1_diagnostic": float(oracle_f1),
    "previous_ensemble_errors": int(
        (~old_correct).sum()
    ),
    "recoverable_previous_errors": int(
        recoverable_old_errors.sum()
    ),
    "all_models_wrong": int(all_wrong.sum()),
    "new_unique_figures": len(
        set(x["figure"] for x in FIGURE_MANIFEST)
    ),
    "new_tables": len(list(TABLES.glob("*.csv"))),
    "human_annotations_completed": 0,
    "new_model_training": False,
    "router_training": False,
    "test_evaluated": False,
    "external_evaluated": False,
    "novelty_verified": False,
    "limitations": [
        "Validation set reused for exploratory analysis.",
        "Oracle uses recorded ground-truth labels.",
        "Oracle accuracy is not deployable performance.",
        "Subgroup selection is descriptive and post hoc.",
        "Shared model errors do not prove label noise.",
        "Out-of-fold training predictions remain necessary.",
        "No learned specialist router has been evaluated."
    ]
}

save_json(
    SUMMARY,
    AUDIT / "stage7c_summary.json"
)

save_json(
    SUMMARY,
    BACKUP / "stage7c_summary.json"
)

save_json(
    FIGURE_MANIFEST,
    BACKUP / "figure_manifest.json"
)

pd.DataFrame(FIGURE_MANIFEST).to_csv(
    PUB / "stage7c_figure_manifest.csv",
    index=False
)

for path in TABLES.glob("*.csv"):

    dest = BACKUP / "tables" / path.name

    dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(path, dest)

    tex = PUB / "tables/latex" / (path.stem + ".tex")

    if tex.exists():

        dest = BACKUP / "latex_tables" / tex.name

        dest.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(tex, dest)

# Private archive includes sample-level diagnostics.
# The blinded review queue remains private.

compact = WORK / "sentiment_stage7c_private.zip"

with zipfile.ZipFile(
    compact, "w", zipfile.ZIP_DEFLATED
) as z:

    for folder in [AUDIT, TABLES, PRED, FIGURES]:

        for path in folder.rglob("*"):

            if path.is_file():

                z.write(path, path.relative_to(OUT))

publication = WORK / "sentiment_stage7c_publication.zip"

with zipfile.ZipFile(
    publication, "w", zipfile.ZIP_DEFLATED
) as z:

    for path in PUB.rglob("*"):

        if path.is_file():

            z.write(path, path.relative_to(PUB))

for path in [compact, publication]:

    with zipfile.ZipFile(path) as z:

        assert z.testzip() is None

    upload_verified(
        path,
        f"{path.stem}_{sha256(path)[:12]}.zip"
    )

manifest_path = AUDIT / "stage7c_remote_manifest.json"

save_json(
    {
        "release": TAG_7C,
        "assets": REMOTE_MANIFEST
    },
    manifest_path
)

upload_verified(
    manifest_path,
    f"stage7c_manifest_{sha256(manifest_path)[:12]}.json"
)

shutil.copy2(
    manifest_path,
    BACKUP / manifest_path.name
)

FINAL_COMMIT = push_public(
    "Stage 7C: complete specialist discovery and verified backup"
)

# Verify every generated figure exists in the committed tree.

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
        "research_stage7c/figures/"
        + record["format"]
        + "/"
        + record["figure"]
        + "."
        + record["format"]
    )

    assert relative in tracked

    assert sha256(REPO / relative) == record["sha256"]

# =============================================================================
# FINAL REPORT
# =============================================================================

section("FINAL STAGE 7C RESEARCH REPORT")

print("Models recovered:", len(NAMES))
print("Prediction alignment: PASS")
print("Historical metric reproduction: PASS")

print("\nALL MODEL AND ENSEMBLE RESULTS")
print(RESULTS.round(5).to_string(index=False))

print("\nORACLE ANALYSIS")
print(ORACLE.to_string(index=False))

print("\nSPECIALIST RESCUE CAPABILITY")
print(
    RESCUES.sort_values(
        "rescued_previous_errors",
        ascending=False
    ).to_string(index=False)
)

print("\nBEST SUBGROUP SPECIALISTS")
print(BEST_SUBGROUPS.round(5).to_string(index=False))

print("\nTOP COMPLEMENTARY PAIRS")
print(
    PAIRS.sort_values(
        "pair_ensemble_macro_f1",
        ascending=False
    ).head(15).round(5).to_string(index=False)
)

print("\nERROR GROUP PROFILES")
print(ERROR_GROUPS.round(5).to_string(index=False))

print("\nUNCERTAINTY DETECTION")
print(
    DETECTION.sort_values(
        ["target", "auroc"],
        ascending=[True, False]
    ).round(5).to_string(index=False)
)

print("\nPAIRED ENSEMBLE STATISTICS")
print(STATS.round(5).to_string(index=False))

print("\nRESEARCH SUMMARY")
print(json.dumps(SUMMARY, indent=2))

print("\nPUBLICATION ASSETS")

for fmt in ["png", "pdf", "svg"]:

    print(
        fmt.upper(),
        "figures:",
        len(list(
            (PUB / "figures" / fmt).rglob(f"*.{fmt}")
        ))
    )

print("All new figures committed and verified: YES")
print("Private backup: PASS")
print("GitHub synchronized: True")
print("Final public commit:", FINAL_COMMIT)
print("Private release:", RELEASE["html_url"])

print("Manual file saving required: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("New training performed: NO")
print("Test evaluated: NO")

print(
    "\nRuntime minutes:",
    round((time.time() - START) / 60, 2)
)

print("\nSTAGE 7C COMPLETE")
print("READY FOR STAGE 7D — OUT-OF-FOLD SPECIALIST LEARNING")

gc.collect()
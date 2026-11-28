# ============================================================
# STAGE 9B — FINAL MODEL RECOVERY AND VERIFICATION
# STEPS 183–186
# ============================================================

import os
import io
import re
import json
import time
import hashlib
import getpass
import zipfile
import itertools
import subprocess
import base64

from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support
)

from scipy.stats import binomtest

START = time.time()

WORK = Path("/kaggle/working")
OUT = WORK / "sentiment_research_stage9b"

OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"

PREVIOUS_COMMIT = (
    "a3c74f6b6110b5b1f08a933b3ad50aafba72201f"
)

TAG = "stage9b-final-model-recovery-v1"

REPO = WORK / "sentiment_stage9b_github"

TABLES = OUT / "tables"
AUDIT = OUT / "audit"
REC = OUT / "recovered"
FIGURES = OUT / "figures"

PUB = WORK / "sentiment_publication_assets"

for p in [OUT, TABLES, AUDIT, REC, FIGURES, PUB]:
    p.mkdir(parents=True, exist_ok=True)

LABELS = ["Negative", "Neutral", "Positive"]

def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()

def write_json(data, path):
    Path(path).write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str
        )
    )

# ============================================================
# STEP 183 — VERIFIED GITHUB RECOVERY
# ============================================================

section("STEP 183 — VERIFIED GITHUB RECOVERY")

TOKEN = getpass.getpass(
    "Enter your working GitHub token: "
).strip()

SESSION = requests.Session()

SESSION.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

API = "https://api.github.com"

PUBLIC_API = f"{API}/repos/{OWNER}/{PUBLIC_REPO}"
PRIVATE_API = f"{API}/repos/{OWNER}/{PRIVATE_REPO}"

def api_get(url, **kwargs):
    r = SESSION.get(url, timeout=120, **kwargs)
    r.raise_for_status()
    return r.json()

assert api_get(API + "/user")["login"]
assert api_get(PRIVATE_API)["private"]

AUTH = base64.b64encode(
    f"x-access-token:{TOKEN}".encode()
).decode()

ENV = os.environ.copy()

ENV.update({
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0":
        "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0":
        f"AUTHORIZATION: basic {AUTH}",
    "GIT_TERMINAL_PROMPT": "0"
})

def git(*args, cwd=REPO):

    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=ENV,
        capture_output=True,
        text=True
    )

    if result.returncode:
        raise RuntimeError(
            "Git operation failed: " + args[0]
        )

    return result.stdout.strip()

if not (REPO / ".git").exists():

    git(
        "clone",
        "--single-branch",
        "--branch", BRANCH,
        f"https://github.com/{OWNER}/{PUBLIC_REPO}.git",
        str(REPO),
        cwd=WORK
    )

git("fetch", "origin", BRANCH)
git("checkout", BRANCH)

assert not git("status", "--porcelain")

git("pull", "--ff-only", "origin", BRANCH)

git(
    "merge-base",
    "--is-ancestor",
    PREVIOUS_COMMIT,
    "HEAD"
)

git("config", "user.name", "Research Backup")

git(
    "config",
    "user.email",
    "research-backup@users.noreply.github.com"
)

BACKUP = REPO / "research_stage9b"
BACKUP.mkdir(exist_ok=True)

def push(message):

    git("add", "research_stage9b")

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

def paginate(url):

    records = []
    page = 1

    while True:

        rows = api_get(
            url,
            params={
                "per_page": 100,
                "page": page
            }
        )

        records.extend(rows)

        if len(rows) < 100:
            break

        page += 1

    return records

def release(tag):

    return api_get(
        PRIVATE_API + f"/releases/tags/{tag}"
    )

def assets(tag):

    r = release(tag)

    return paginate(
        PRIVATE_API
        + f"/releases/{r['id']}/assets"
    )

def download(asset, destination):

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    expected = asset.get("digest")

    if (
        destination.exists()
        and destination.stat().st_size == asset["size"]
        and expected
        and expected == "sha256:" + sha256(destination)
    ):
        return destination

    with SESSION.get(
        asset["url"],
        headers={
            "Accept": "application/octet-stream"
        },
        stream=True,
        timeout=(30, 600)
    ) as r:

        r.raise_for_status()

        with open(destination, "wb") as f:

            for chunk in r.iter_content(1024 * 1024):

                if chunk:
                    f.write(chunk)

    assert destination.stat().st_size == asset["size"]

    if expected:
        assert expected == "sha256:" + sha256(destination)

    print("Recovered:", asset["name"])

    return destination

def select_asset(tag, name):

    found = [
        a for a in assets(tag)
        if a["name"] == name
    ]

    assert len(found) == 1, (tag, name)

    return found[0]

def save_table(data, name):

    df = pd.DataFrame(data)

    for root in [
        TABLES,
        BACKUP / "tables",
        PUB / "tables"
    ]:

        root.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            root / f"{name}.csv",
            index=False
        )

        df.to_latex(
            root / f"{name}.tex",
            index=False,
            escape=True,
            float_format="%.5f"
        )

    return df

print("Authentication: PASS")
print("Previous research commit: VERIFIED")

# Preserve the source before further processing.

source_cells = get_ipython().user_ns.get("In", [])

SOURCE = next(
    s for s in reversed(source_cells)
    if "STAGE 9B — FINAL MODEL RECOVERY AND VERIFICATION" in s
    and len(s) > 10000
)

(BACKUP / "stage9b_complete.py").write_text(SOURCE)

push("Preserve Stage 9B reproducible source")

# ============================================================
# RECOVER LOCKED VALIDATION PREDICTIONS
# ============================================================

STAGE7C = "stage7c-specialist-discovery-v1"
STAGE7B = "stage7b-pasnet-screening-v1"
STAGE6B = "stage6b-locked-baselines-v1"

archive_candidates = [
    a for a in assets(STAGE7C)
    if "private" in a["name"]
    and a["name"].endswith(".zip")
]

assert len(archive_candidates) == 1

archive = download(
    archive_candidates[0],
    REC / archive_candidates[0]["name"]
)

with zipfile.ZipFile(archive) as z:

    assert z.testzip() is None

    candidates = [
        n for n in z.namelist()
        if n.endswith("stage7c_aligned_15_models.npz")
    ]

    assert len(candidates) == 1

    with np.load(
        io.BytesIO(z.read(candidates[0])),
        allow_pickle=False
    ) as data:

        IDS = data["sample_id"].astype(str)
        Y = data["labels"].astype(int)
        GROUPS = data["group_id"].astype(str)

        NAMES = data["model_names"].astype(str)
        PROBS = data["probabilities"].astype(np.float64)

        TARGET = data[
            "ensemble_prototype_family"
        ].astype(np.float64)

        PREVIOUS_NINE = data[
            "ensemble_previous_nine"
        ].astype(np.float64)

        ALL_FIFTEEN = data[
            "ensemble_all_fifteen"
        ].astype(np.float64)

assert PROBS.shape == (15, 1840, 3)
assert TARGET.shape == (1840, 3)

assert len(np.unique(IDS)) == len(IDS)

assert np.isfinite(PROBS).all()
assert np.isfinite(TARGET).all()

assert np.allclose(
    PROBS.sum(axis=-1), 1, atol=1e-5
)

assert np.allclose(
    TARGET.sum(axis=-1), 1, atol=1e-5
)

def macro_f1(p, labels=Y):

    return float(
        f1_score(
            labels,
            p.argmax(axis=1),
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    )

EXPECTED_F1 = 0.8287035909704495

assert abs(
    macro_f1(TARGET) - EXPECTED_F1
) < 1e-10

print("Aligned models:", len(NAMES))
print("Validation samples:", len(Y))
print("Recovered prototype F1:", macro_f1(TARGET))

# ============================================================
# VERIFY EXACT VALIDATION MEMBERSHIP
# ============================================================

baseline_asset = select_asset(
    STAGE6B,
    "stage6b_reproducibility_package.zip"
)

baseline_zip = download(
    baseline_asset,
    REC / baseline_asset["name"]
)

with zipfile.ZipFile(baseline_zip) as z:

    matches = [
        n for n in z.namelist()
        if n.endswith("partitions/seed42_val.csv")
    ]

    assert len(matches) == 1

    membership = pd.read_csv(
        io.BytesIO(z.read(matches[0]))
    ).sort_values("partition_order")

assert np.array_equal(
    membership["sample_id"].astype(str).values,
    IDS
)

raw_labels = membership["Label"]

if pd.api.types.is_numeric_dtype(raw_labels):

    membership_labels = raw_labels.astype(int).values

else:

    membership_labels = (
        raw_labels.astype(str)
        .str.strip()
        .str.title()
        .map({
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        })
        .values
    )

assert np.array_equal(membership_labels, Y)

assert np.array_equal(
    membership["group_id"].astype(str).values,
    GROUPS
)

print("Locked validation membership: EXACT MATCH")
print("Test partition accessed: NO")

# ============================================================
# STEP 184 — EXACT ARCHITECTURE COMPOSITION
# ============================================================

section("STEP 184 — EXACT ENSEMBLE COMPOSITION")

source7c = (
    REPO / "research_stage7c"
    / "stage7c_complete.py"
)

assert source7c.exists()

text7c = source7c.read_text()
lines7c = text7c.splitlines()

source_evidence = []

for i, line in enumerate(lines7c):

    if "prototype_family" in line.lower():

        lo = max(0, i - 8)
        hi = min(len(lines7c), i + 9)

        excerpt = "\n".join(
            f"{j+1}: {lines7c[j]}"
            for j in range(lo, hi)
        )

        source_evidence.append(excerpt)

evidence_text = "\n\n".join(source_evidence)

(AUDIT / "prototype_source_evidence.txt").write_text(
    evidence_text
)

(BACKUP / "prototype_source_evidence.txt").write_text(
    evidence_text
)

print(evidence_text[:18000])

# Recover the exact equal-weight ensemble membership
# by numerical identity, NOT by maximizing F1.

# A small sample screens combinations efficiently.
# Every candidate match is then verified on all rows.

screen_idx = np.linspace(
    0, len(Y) - 1, 64, dtype=int
)

screen_probs = PROBS[:, screen_idx]
screen_target = TARGET[screen_idx]

exact_matches = []

for k in range(1, len(NAMES) + 1):

    for combo in itertools.combinations(
        range(len(NAMES)), k
    ):

        screen_mean = screen_probs[
            list(combo)
        ].mean(axis=0)

        if not np.allclose(
            screen_mean,
            screen_target,
            atol=1e-8,
            rtol=1e-7
        ):
            continue

        full_mean = PROBS[
            list(combo)
        ].mean(axis=0)

        error = float(
            np.max(np.abs(full_mean - TARGET))
        )

        if error <= 1e-7:

            exact_matches.append({
                "indices": list(combo),
                "members": NAMES[list(combo)].tolist(),
                "member_count": k,
                "maximum_probability_error": error
            })

write_json(
    exact_matches,
    BACKUP / "exact_ensemble_matches.json"
)

print("\nNumerically exact ensemble matches:")

print(json.dumps(exact_matches, indent=2))

if len(exact_matches) == 1:

    MEMBERS = exact_matches[0]["members"]
    MEMBER_INDICES = exact_matches[0]["indices"]

    MEMBERSHIP_STATUS = "UNIQUE_NUMERICAL_MATCH"

else:

    MEMBERS = []
    MEMBER_INDICES = []

    MEMBERSHIP_STATUS = (
        "SOURCE_REVIEW_REQUIRED"
    )

print("Membership status:", MEMBERSHIP_STATUS)

# ============================================================
# CHECKPOINT AND ARCHITECTURE INVENTORY
# ============================================================

source7b = (
    REPO / "research_stage7b"
    / "stage7b_complete.py"
)

assert source7b.exists()

source7b_text = source7b.read_text()

stage7b_assets = assets(STAGE7B)

checkpoint_records = []

for name in NAMES:

    related = [
        a for a in stage7b_assets
        if a["name"].startswith(
            str(name) + "_seed42_"
        )
    ]

    checkpoint = next(
        (
            a for a in related
            if a["name"].endswith("_model.pt")
        ),
        None
    )

    metadata = next(
        (
            a for a in related
            if a["name"].endswith("_completed.json")
        ),
        None
    )

    prediction = next(
        (
            a for a in related
            if a["name"].endswith("_val.npz")
        ),
        None
    )

    if checkpoint is None:
        continue

    record = {
        "model": str(name),
        "selected_member": str(name) in MEMBERS,
        "checkpoint": checkpoint["name"],
        "checkpoint_bytes": checkpoint["size"],
        "checkpoint_remote_digest": checkpoint.get("digest"),
        "metadata_available": metadata is not None,
        "prediction_available": prediction is not None,
        "metadata_checkpoint_hash_match": False
    }

    if metadata is not None:

        path = download(
            metadata,
            REC / metadata["name"]
        )

        info = json.loads(path.read_text())

        expected = info.get("checkpoint_sha256")

        record["metadata_checkpoint_hash_match"] = bool(
            expected
            and checkpoint.get("digest")
            == "sha256:" + expected
        )

        record["model_name"] = info.get("model_name")
        record["mode"] = info.get("mode")

    checkpoint_records.append(record)

CHECKPOINTS = save_table(
    checkpoint_records,
    "stage9b_01_checkpoint_verification"
)

print("\nCHECKPOINT INVENTORY")

print(CHECKPOINTS.to_string(index=False))

# Record architecture source identity.

SOURCE_INVENTORY = save_table(
    [
        {
            "source": str(p.relative_to(REPO)),
            "sha256": sha256(p),
            "bytes": p.stat().st_size
        }
        for p in [source7b, source7c]
    ],
    "stage9b_02_architecture_source_inventory"
)

# ============================================================
# STEP 185 — COMPARATIVE EVALUATION
# ============================================================

section("STEP 185 — COMPARATIVE VALIDATION EVIDENCE")

SYSTEMS = {
    str(name): PROBS[i]
    for i, name in enumerate(NAMES)
}

SYSTEMS.update({
    "prototype_family": TARGET,
    "previous_nine": PREVIOUS_NINE,
    "all_fifteen": ALL_FIFTEEN
})

# Recover the locked BanglaBERT baseline predictions.

baseline_pred_asset = select_asset(
    STAGE6B,
    "stage6b_reproducibility_package.zip"
)

with zipfile.ZipFile(baseline_zip) as z:

    matches = [
        n for n in z.namelist()
        if n.endswith(
            "private_predictions/BanglaBERT_val.npz"
        )
    ]

    assert len(matches) == 1

    with np.load(
        io.BytesIO(z.read(matches[0])),
        allow_pickle=False
    ) as d:

        assert np.array_equal(
            d["sample_id"].astype(str),
            IDS
        )

        assert np.array_equal(
            d["labels"].astype(int),
            Y
        )

        logits = d["logits"].astype(np.float64)

        logits -= logits.max(axis=1, keepdims=True)

        p = np.exp(logits)

        p /= p.sum(axis=1, keepdims=True)

        SYSTEMS["locked_banglabert"] = p

assert abs(
    macro_f1(SYSTEMS["locked_banglabert"])
    - 0.8159419852657895
) < 1e-8

# Same-backbone-family standard controls.
# These are not automatically parameter-matched.

if (
    "base_standard" in SYSTEMS
    and "large_standard" in SYSTEMS
):

    SYSTEMS["base_large_standard_ensemble"] = (
        SYSTEMS["base_standard"]
        + SYSTEMS["large_standard"]
    ) / 2

if (
    "base_pasnet" in SYSTEMS
    and "large_pasnet" in SYSTEMS
):

    SYSTEMS["base_large_pasnet_ensemble"] = (
        SYSTEMS["base_pasnet"]
        + SYSTEMS["large_pasnet"]
    ) / 2

def ece(p, bins=15):

    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == Y

    value = 0.0

    for i in range(bins):

        low = i / bins
        high = (i + 1) / bins

        mask = (
            (confidence >= low)
            & (
                (confidence <= high)
                if i == bins - 1
                else (confidence < high)
            )
        )

        if mask.any():

            value += mask.mean() * abs(
                correct[mask].mean()
                - confidence[mask].mean()
            )

    return float(value)

results = []
classwise = []

for name, p in SYSTEMS.items():

    pred = p.argmax(axis=1)

    results.append({
        "model": name,
        "accuracy": accuracy_score(Y, pred),
        "macro_f1": macro_f1(p),
        "weighted_f1": f1_score(
            Y, pred,
            average="weighted",
            zero_division=0
        ),
        "nll": log_loss(
            Y, p, labels=[0, 1, 2]
        ),
        "brier": np.mean(
            np.sum(
                (p - np.eye(3)[Y]) ** 2,
                axis=1
            )
        ),
        "ece": ece(p),
        "evaluation_scope":
            "Exploratory validation"
    })

    pr, rc, f1, support = (
        precision_recall_fscore_support(
            Y,
            pred,
            labels=[0, 1, 2],
            zero_division=0
        )
    )

    for j, label in enumerate(LABELS):

        classwise.append({
            "model": name,
            "class": label,
            "precision": pr[j],
            "recall": rc[j],
            "f1": f1[j],
            "support": support[j]
        })

RESULTS = save_table(
    pd.DataFrame(results).sort_values(
        "macro_f1", ascending=False
    ),
    "stage9b_03_comparative_results"
)

CLASSWISE = save_table(
    classwise,
    "stage9b_04_classwise_results"
)

print(RESULTS.to_string(index=False))

# ============================================================
# MEMBER CONTRIBUTION ANALYSIS
# ============================================================

contributions = []

if MEMBER_INDICES:

    for idx in MEMBER_INDICES:

        remaining = [
            j for j in MEMBER_INDICES
            if j != idx
        ]

        row = {
            "member": NAMES[idx],
            "individual_macro_f1":
                macro_f1(PROBS[idx]),
            "full_ensemble_macro_f1":
                macro_f1(TARGET)
        }

        if remaining:

            reduced = PROBS[remaining].mean(axis=0)

            row["without_member_macro_f1"] = (
                macro_f1(reduced)
            )

            row["member_contribution"] = (
                macro_f1(TARGET)
                - macro_f1(reduced)
            )

        contributions.append(row)

CONTRIBUTIONS = save_table(
    pd.DataFrame(
        contributions,
        columns=[
            "member",
            "individual_macro_f1",
            "full_ensemble_macro_f1",
            "without_member_macro_f1",
            "member_contribution"
        ]
    ),
    "stage9b_05_member_contributions"
)

print("\nMEMBER CONTRIBUTIONS")

print(CONTRIBUTIONS.to_string(index=False))

# ============================================================
# GROUP-BOOTSTRAP STATISTICS
# ============================================================

rng = np.random.default_rng(42)

unique_groups, group_inverse = np.unique(
    GROUPS, return_inverse=True
)

group_indices = [
    np.flatnonzero(group_inverse == j)
    for j in range(len(unique_groups))
]

references = [
    name for name in [
        "locked_banglabert",
        "base_standard",
        "large_standard",
        "base_large_standard_ensemble",
        "base_large_pasnet_ensemble",
        "previous_nine",
        "all_fifteen"
    ]
    if name in SYSTEMS
]

target_pred = TARGET.argmax(axis=1)

ref_predictions = {
    name: SYSTEMS[name].argmax(axis=1)
    for name in references
}

def fast_f1(y, pred):

    cm = np.bincount(
        3 * y + pred,
        minlength=9
    ).reshape(3, 3)

    denom = cm.sum(axis=0) + cm.sum(axis=1)

    values = np.divide(
        2 * np.diag(cm),
        denom,
        out=np.zeros(3, dtype=float),
        where=denom > 0
    )

    return float(values.mean())

B = 5000

bootstrap = np.zeros(
    (B, len(references)),
    dtype=np.float64
)

for b in range(B):

    sampled = rng.integers(
        0,
        len(group_indices),
        size=len(group_indices)
    )

    idx = np.concatenate([
        group_indices[j]
        for j in sampled
    ])

    score = fast_f1(
        Y[idx], target_pred[idx]
    )

    for j, name in enumerate(references):

        bootstrap[b, j] = (
            score
            - fast_f1(
                Y[idx],
                ref_predictions[name][idx]
            )
        )

paired = []

for j, name in enumerate(references):

    ref = ref_predictions[name]

    gained = int(
        ((target_pred == Y) & (ref != Y)).sum()
    )

    lost = int(
        ((target_pred != Y) & (ref == Y)).sum()
    )

    pvalue = (
        binomtest(
            min(gained, lost),
            gained + lost,
            0.5
        ).pvalue
        if gained + lost
        else 1.0
    )

    low, high = np.quantile(
        bootstrap[:, j],
        [0.025, 0.975]
    )

    paired.append({
        "candidate": "prototype_family",
        "reference": name,
        "delta_macro_f1":
            macro_f1(TARGET)
            - macro_f1(SYSTEMS[name]),
        "ci95_low": low,
        "ci95_high": high,
        "gained_correct": gained,
        "lost_correct": lost,
        "mcnemar_p": pvalue
    })

PAIRED = pd.DataFrame(paired)

pvalues = PAIRED["mcnemar_p"].values

order = np.argsort(pvalues)

adjusted = np.empty(len(order))

running = 0.0

for rank, j in enumerate(order):

    running = max(
        running,
        (len(order) - rank) * pvalues[j]
    )

    adjusted[j] = min(1.0, running)

PAIRED["holm_p"] = adjusted

PAIRED = save_table(
    PAIRED,
    "stage9b_06_paired_statistics"
)

print("\nPAIRED STATISTICS")

print(PAIRED.to_string(index=False))

# ============================================================
# STEP 186 — EVALUATION PROTOCOL PREPARATION
# ============================================================

section("STEP 186 — EVALUATION PROTOCOL PREPARATION")

selected_checkpoints = CHECKPOINTS[
    CHECKPOINTS["selected_member"] == True
]

checkpoint_metadata_verified = bool(
    MEMBERS
    and len(selected_checkpoints) == len(MEMBERS)
    and selected_checkpoints[
        "metadata_checkpoint_hash_match"
    ].all()
)

PROTOCOL = {
    "status": "DRAFT_PENDING_FINAL_REVIEW",
    "candidate": "prototype_family",
    "candidate_members": MEMBERS,
    "membership_status": MEMBERSHIP_STATUS,
    "validation_macro_f1": macro_f1(TARGET),
    "checkpoint_metadata_verified":
        checkpoint_metadata_verified,
    "full_checkpoint_bytes_reloaded": False,
    "aggregation": (
        "Equal probability averaging"
        if len(exact_matches) == 1
        else "Requires source verification"
    ),
    "primary_metric": "macro_f1",
    "partition_seed": 42,
    "test_partition_source": STAGE6B,
    "test_partition_expected_samples": 1849,
    "external_candidates": [
        "SentNoB",
        "SentiGOLD"
    ],
    "external_overlap_audit_required": True,
    "external_training_allowed": False,
    "external_threshold_tuning_allowed": False,
    "baseline_requirements": [
        "Locked BanglaBERT",
        "XLM-RoBERTa",
        "mBERT",
        "TF-IDF Logistic Regression",
        "Equivalent-backbone ensemble controls",
        "Comparable inference-budget controls"
    ],
    "bootstrap_replicates": 5000,
    "novelty_verified": False,
    "architecture_superiority_established": False,
    "test_loaded": False,
    "test_evaluated": False,
    "external_evaluated": False,
    "limitations": [
        "Repeated exploratory validation use",
        "Numerical membership recovery must agree with source",
        "Different inference budgets across models",
        "Remote metadata verification is not model reload",
        "Reconstructed test overlaps historical research cohorts",
        "Independent external evaluation remains necessary"
    ]
}

write_json(
    PROTOCOL,
    BACKUP / "evaluation_protocol_review.json"
)

# ============================================================
# PUBLICATION FIGURES
# ============================================================

section("PUBLICATION FIGURES AND GITHUB BACKUP")

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})

CAPTIONS = []

def save_figure(fig, name, caption):

    for root in [
        FIGURES,
        BACKUP / "figures",
        PUB / "figures"
    ]:

        for ext in ["png", "pdf", "svg"]:

            folder = root / ext
            folder.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                folder / f"{name}.{ext}",
                dpi=600 if ext == "png" else None,
                bbox_inches="tight"
            )

    CAPTIONS.append({
        "figure": name,
        "caption": caption
    })

    save_table(
        CAPTIONS,
        "stage9b_figure_captions"
    )

    push("Preserve " + name)

    plt.show()
    plt.close(fig)

# Figure 1 — Complete comparative evidence.

fig, ax = plt.subplots(figsize=(10, 8))

df = RESULTS.sort_values("macro_f1")

ax.barh(
    df["model"],
    df["macro_f1"] * 100
)

for i, value in enumerate(df["macro_f1"]):

    ax.text(
        value * 100 + 0.15,
        i,
        f"{value * 100:.2f}",
        va="center",
        fontsize=9
    )

ax.set_xlim(0, 90)
ax.set_xlabel("Validation macro-F1 (%)")
ax.set_title("Comparative validation performance")

fig.tight_layout()

save_figure(
    fig,
    "stage9b_01_complete_comparison",
    "Exploratory comparison of recovered systems "
    "on the identical validation cohort. "
    "Inference budgets differ."
)

# Figure 2 — Classwise performance.

chosen = [
    name for name in [
        "locked_banglabert",
        "base_large_standard_ensemble",
        "base_large_pasnet_ensemble",
        "prototype_family"
    ]
    if name in SYSTEMS
]

fig, ax = plt.subplots(figsize=(10, 5))

df = CLASSWISE[
    CLASSWISE["model"].isin(chosen)
].pivot(
    index="class",
    columns="model",
    values="f1"
).reindex(LABELS)

(df * 100).plot.bar(ax=ax)

ax.set_ylabel("Classwise F1 (%)")
ax.set_xlabel("")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=0)
ax.legend(fontsize=8)

fig.tight_layout()

save_figure(
    fig,
    "stage9b_02_classwise_comparison",
    "Classwise validation F1 for selected "
    "prototype and standard systems."
)

# Figure 3 — Paired effects.

fig, ax = plt.subplots(figsize=(9, 5))

df = PAIRED.reset_index(drop=True)

ypos = np.arange(len(df))

ax.errorbar(
    df["delta_macro_f1"] * 100,
    ypos,
    xerr=np.vstack([
        (
            df["delta_macro_f1"]
            - df["ci95_low"]
        ).clip(lower=0) * 100,
        (
            df["ci95_high"]
            - df["delta_macro_f1"]
        ).clip(lower=0) * 100
    ]),
    fmt="o",
    capsize=4
)

ax.axvline(0, linestyle="--")

ax.set_yticks(ypos)
ax.set_yticklabels(df["reference"])

ax.set_xlabel(
    "Prototype-family macro-F1 difference "
    "(percentage points)"
)

fig.tight_layout()

save_figure(
    fig,
    "stage9b_03_paired_effects",
    "Paired group-bootstrap confidence intervals "
    "using 5,000 validation resamples. "
    "These are exploratory comparisons."
)

# Figure 4 — Ensemble member contribution.

if (
    len(CONTRIBUTIONS)
    and CONTRIBUTIONS[
        "member_contribution"
    ].notna().any()
):

    fig, ax = plt.subplots(figsize=(9, 4))

    df = CONTRIBUTIONS.dropna(
        subset=["member_contribution"]
    )

    ax.barh(
        df["member"],
        df["member_contribution"] * 100
    )

    ax.axvline(0, linestyle="--")

    ax.set_xlabel(
        "Full minus leave-one-member-out "
        "macro-F1 (percentage points)"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "stage9b_04_member_contribution",
        "Exploratory leave-one-member-out "
        "analysis of the recovered ensemble."
    )

# Figure 5 — Confidence calibration.

fig, ax = plt.subplots(figsize=(6, 6))

for name in chosen:

    p = SYSTEMS[name]

    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == Y

    xs, ys = [], []

    for i in range(10):

        mask = (
            (conf >= i / 10)
            & (
                (conf <= (i + 1) / 10)
                if i == 9
                else (conf < (i + 1) / 10)
            )
        )

        if mask.any():

            xs.append(conf[mask].mean())
            ys.append(correct[mask].mean())

    ax.plot(
        xs, ys,
        marker="o",
        label=name
    )

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

ax.set_xlabel("Mean confidence")
ax.set_ylabel("Observed accuracy")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.legend(fontsize=7)

fig.tight_layout()

save_figure(
    fig,
    "stage9b_05_reliability_comparison",
    "Validation reliability diagrams for "
    "selected standard and prototype systems."
)

# ============================================================
# PRIVATE ARTIFACT BACKUP
# ============================================================

section("PERMANENT PRIVATE ARTIFACT BACKUP")

r = SESSION.get(
    PRIVATE_API + f"/releases/tags/{TAG}",
    timeout=120
)

if r.status_code == 404:

    r = SESSION.post(
        PRIVATE_API + "/releases",
        json={
            "tag_name": TAG,
            "name": "Stage 9B final model recovery",
            "body":
                "Exact ensemble recovery and "
                "comparative validation evidence.",
            "draft": False
        },
        timeout=120
    )

r.raise_for_status()

RELEASE = r.json()

REMOTE = {
    a["name"]: a
    for a in paginate(
        PRIVATE_API
        + f"/releases/{RELEASE['id']}/assets"
    )
}

MANIFEST = []

def upload(path):

    path = Path(path)

    digest = sha256(path)

    name = (
        f"{path.stem}_{digest[:12]}"
        f"{path.suffix}"
    )

    asset = REMOTE.get(name)

    if asset is None:

        with open(path, "rb") as f:

            response = SESSION.post(
                RELEASE["upload_url"].split("{")[0],
                params={"name": name},
                headers={
                    "Content-Type":
                        "application/octet-stream"
                },
                data=f,
                timeout=600
            )

        response.raise_for_status()

        asset = response.json()

        REMOTE[name] = asset

    assert asset["size"] == path.stat().st_size

    if asset.get("digest"):

        assert asset["digest"] == "sha256:" + digest

    else:

        verification = download(
            asset,
            REC / ("verify_" + name)
        )

        assert sha256(verification) == digest

        verification.unlink()

    MANIFEST.append({
        "name": name,
        "asset_id": asset["id"],
        "sha256": digest,
        "size": asset["size"],
        "remote_verified": True
    })

    print("Remote verified:", name)

# Preserve predictions privately.

prediction_file = OUT / "stage9b_aligned_predictions.npz"

np.savez_compressed(
    prediction_file,
    sample_id=IDS,
    labels=Y,
    group_id=GROUPS,
    model_names=np.array(
        list(SYSTEMS.keys()),
        dtype=str
    ),
    probabilities=np.stack(
        list(SYSTEMS.values())
    )
)

upload(prediction_file)

REPORT = {
    "stage": "9B",
    "validation_membership_verified": True,
    "validation_samples": len(Y),
    "exact_ensemble_matches": exact_matches,
    "prototype_members": MEMBERS,
    "membership_status": MEMBERSHIP_STATUS,
    "prototype_macro_f1": macro_f1(TARGET),
    "checkpoint_metadata_verified":
        checkpoint_metadata_verified,
    "checkpoint_inventory_rows": len(CHECKPOINTS),
    "comparative_systems": len(SYSTEMS),
    "new_figures": len(CAPTIONS),
    "bootstrap_replicates": B,
    "final_architecture_locked": False,
    "novelty_verified": False,
    "test_loaded": False,
    "test_evaluated": False,
    "external_evaluated": False,
    "manual_saving_required": False,
    "google_drive_used": False,
    "decision": (
        "REVIEW EXACT MEMBERSHIP AND "
        "LOCK INFERENCE PROTOCOL"
    )
}

write_json(
    REPORT,
    BACKUP / "stage9b_summary.json"
)

package = OUT / "stage9b_recovery_package.zip"

with zipfile.ZipFile(
    package, "w", zipfile.ZIP_DEFLATED
) as z:

    for path in BACKUP.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(BACKUP)
            )

with zipfile.ZipFile(package) as z:

    assert z.testzip() is None

upload(package)

manifest_file = AUDIT / "stage9b_remote_manifest.json"

write_json(MANIFEST, manifest_file)

write_json(
    MANIFEST,
    BACKUP / "stage9b_remote_manifest.json"
)

upload(manifest_file)

FINAL_COMMIT = push(
    "Complete Stage 9B exact model recovery "
    "and comparative evaluation preparation"
)

# Count unique figure names in the synchronized repository.

tracked = [
    REPO / p
    for p in git("ls-files").splitlines()
]

figure_counts = {}

for ext in ["png", "pdf", "svg"]:

    figure_counts[ext] = len({
        p.stem
        for p in tracked
        if p.suffix.lower() == "." + ext
        and p.is_file()
    })

# ============================================================
# FINAL REPORT
# ============================================================

section("FINAL STAGE 9B RESEARCH REPORT")

print("\nEXACT PROTOTYPE MEMBERSHIP")

print(json.dumps(exact_matches, indent=2))

print("\nCHECKPOINT VERIFICATION")

print(CHECKPOINTS.to_string(index=False))

print("\nCOMPARATIVE PERFORMANCE")

print(RESULTS.to_string(index=False))

print("\nMEMBER CONTRIBUTIONS")

print(CONTRIBUTIONS.to_string(index=False))

print("\nPAIRED STATISTICS")

print(PAIRED.to_string(index=False))

print("\nRESEARCH SUMMARY")

print(json.dumps(REPORT, indent=2))

print("\nPERMANENT PUBLICATION ASSETS")

print("PNG figures:", figure_counts["png"])
print("PDF figures:", figure_counts["pdf"])
print("SVG figures:", figure_counts["svg"])

print("Remote verified assets:", len(MANIFEST))
print("Final public commit:", FINAL_COMMIT)
print("Private release:", RELEASE["html_url"])

print("GitHub synchronized: True")
print("Manual saving required: NO")
print("Google Drive used: NO")
print("New model training: NO")
print("Test evaluated: NO")

print(
    "Runtime minutes:",
    round((time.time() - START) / 60, 2)
)

print("\nRESEARCH DECISION:", REPORT["decision"])
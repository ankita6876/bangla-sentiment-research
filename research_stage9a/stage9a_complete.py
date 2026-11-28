# ============================================================
# STAGE 9A — COMPREHENSIVE RESEARCH FINALIZATION AUDIT
# STEPS 178–182
# ============================================================

import os
import re
import io
import csv
import json
import time
import base64
import hashlib
import getpass
import zipfile
import shutil
import subprocess

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

START9 = time.time()

WORK = Path("/kaggle/working")
ROOT = WORK / "sentiment_research_stage9a"

OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"

PREVIOUS_COMMIT = (
    "f4539b80b11e63fdcc9142360a723acdef7bd094"
)

TAG = "stage9a-final-research-audit-v1"

REPO = WORK / "sentiment_stage9a_github"

TABLES = ROOT / "tables"
AUDIT = ROOT / "audit"
FIGURES = ROOT / "figures"
RECOVERED = ROOT / "recovered"

PUB = WORK / "sentiment_publication_assets"

for path in [
    ROOT, TABLES, AUDIT, FIGURES,
    RECOVERED, PUB
]:
    path.mkdir(parents=True, exist_ok=True)

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
# STEP 178 — VERIFY GITHUB AND RECOVER COMPLETE RESEARCH
# ============================================================

section("STEP 178 — COMPLETE GITHUB RESEARCH AUDIT")

TOKEN9 = getpass.getpass(
    "Enter your working GitHub token: "
)

SESSION9 = requests.Session()

SESSION9.headers.update({
    "Authorization": f"Bearer {TOKEN9}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
})

API = "https://api.github.com"

PUBLIC_API = f"{API}/repos/{OWNER}/{PUBLIC_REPO}"
PRIVATE_API = f"{API}/repos/{OWNER}/{PRIVATE_REPO}"

def api_get(url, **kwargs):

    r = SESSION9.get(
        url, timeout=120, **kwargs
    )

    r.raise_for_status()

    return r.json()

assert api_get(f"{API}/user")["login"]

assert api_get(PRIVATE_API)["private"]

print("GitHub authentication: PASS")

AUTH9 = base64.b64encode(
    f"x-access-token:{TOKEN9}".encode()
).decode()

ENV9 = os.environ.copy()

ENV9.update({
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0":
        "http.https://github.com/.extraheader",
    "GIT_CONFIG_VALUE_0":
        f"AUTHORIZATION: basic {AUTH9}",
    "GIT_TERMINAL_PROMPT": "0"
})

def git(*args, cwd=REPO):

    p = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=ENV9,
        capture_output=True,
        text=True
    )

    if p.returncode:
        raise RuntimeError(
            "Git failed: " + args[0]
        )

    return p.stdout.strip()

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
    "merge-base", "--is-ancestor",
    PREVIOUS_COMMIT, "HEAD"
)

git("config", "user.name", "Research Backup")

git(
    "config", "user.email",
    "research-backup@users.noreply.github.com"
)

BACKUP = REPO / "research_stage9a"
BACKUP.mkdir(exist_ok=True)

def push(message):

    git("add", "research_stage9a")

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

TRACKED = [
    REPO / p
    for p in git("ls-files").splitlines()
]

print("Tracked research files:", len(TRACKED))

def paginate(url):

    result = []
    page = 1

    while True:

        rows = api_get(
            url,
            params={
                "per_page": 100,
                "page": page
            }
        )

        result.extend(rows)

        if len(rows) < 100:
            break

        page += 1

    return result

RELEASES = paginate(
    PRIVATE_API + "/releases"
)

ASSETS = []

for release in RELEASES:

    assets = paginate(
        PRIVATE_API
        + f"/releases/{release['id']}/assets"
    )

    for asset in assets:

        ASSETS.append({
            "release": release["tag_name"],
            "name": asset["name"],
            "asset_id": asset["id"],
            "api_url": asset["url"],
            "size": asset["size"],
            "digest": asset.get("digest"),
            "state": asset["state"]
        })

ASSET_DF = pd.DataFrame(ASSETS)

assert len(ASSET_DF) > 0

print("Private releases:", len(RELEASES))
print("Private assets:", len(ASSETS))

print(
    "Total private storage (GiB):",
    round(
        ASSET_DF["size"].sum() / 1024**3, 3
    )
)

print(
    "Assets with remote SHA256:",
    ASSET_DF["digest"].fillna("")
    .str.startswith("sha256:").sum()
)

def download(asset, destination):

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    expected = asset.get("digest")

    if (
        destination.exists()
        and destination.stat().st_size
            == asset["size"]
        and expected
        and expected
            == "sha256:" + sha256(destination)
    ):
        return destination

    with SESSION9.get(
        asset["api_url"],
        headers={
            "Accept": "application/octet-stream"
        },
        stream=True,
        timeout=(30, 600)
    ) as r:

        r.raise_for_status()

        with open(destination, "wb") as f:

            for chunk in r.iter_content(
                1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

    assert destination.stat().st_size == asset["size"]

    if expected:

        assert (
            "sha256:" + sha256(destination)
            == expected
        )

    return destination

def select_asset(tag, name):

    matches = [
        a for a in ASSETS
        if a["release"] == tag
        and a["name"] == name
    ]

    assert len(matches) == 1, (
        tag, name, len(matches)
    )

    return matches[0]

# ============================================================
# STEP 179 — RECOVER ALL EXPERIMENTAL TABLES
# ============================================================

section("STEP 179 — HISTORICAL EXPERIMENTAL EVIDENCE")

def save_table(df, name):

    df = pd.DataFrame(df)

    for root in [
        TABLES,
        BACKUP / "tables",
        PUB / "tables"
    ]:

        root.mkdir(
            parents=True, exist_ok=True
        )

        df.to_csv(
            root / f"{name}.csv",
            index=False
        )

    return df

inventory = []

for path in TRACKED:

    if not path.is_file():
        continue

    if "research_stage9a" in path.parts:
        continue

    inventory.append({
        "path": str(path.relative_to(REPO)),
        "format": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path)
    })

PUBLIC_INVENTORY = save_table(
    inventory,
    "stage9a_01_public_inventory"
)

PRIVATE_INVENTORY = save_table(
    ASSET_DF.drop(columns=["api_url"]),
    "stage9a_02_private_asset_inventory"
)

table_inventory = []
experiment_rows = []
summary_records = []

def stage_of(path):

    parts = path.relative_to(REPO).parts

    for p in parts:

        if p.startswith("research_stage"):

            return p.replace(
                "research_stage", ""
            ).upper()

        if p.startswith(
            "research_reconstruction"
        ):
            return "6A"

    return "historical"

for path in TRACKED:

    if (
        not path.is_file()
        or path.suffix != ".csv"
        or "research_stage9a" in path.parts
    ):
        continue

    try:

        df = pd.read_csv(path)

    except Exception:
        continue

    stage = stage_of(path)

    table_inventory.append({
        "stage": stage,
        "path": str(path.relative_to(REPO)),
        "rows": len(df),
        "columns": len(df.columns),
        "schema": "|".join(
            map(str, df.columns)
        )
    })

    cols = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    f1_col = next(
        (
            cols[c] for c in [
                "macro_f1",
                "macro_f1_score",
                "val_macro_f1",
                "validation_macro_f1"
            ]
            if c in cols
        ),
        None
    )

    model_col = next(
        (
            cols[c] for c in [
                "model",
                "mode",
                "architecture",
                "candidate",
                "name"
            ]
            if c in cols
        ),
        None
    )

    if f1_col is None or model_col is None:
        continue

    # Do not mix training curves, class-specific
    # results or subgroup scores into the ranking.

    if any(
        c in cols
        for c in [
            "epoch", "subgroup",
            "class", "reference"
        ]
    ):
        continue

    for _, row in df.iterrows():

        value = pd.to_numeric(
            row[f1_col], errors="coerce"
        )

        if not np.isfinite(value):
            continue

        if not 0 <= value <= 1:
            continue

        split_col = next(
            (
                cols[c] for c in [
                    "split", "partition"
                ]
                if c in cols
            ),
            None
        )

        split = (
            str(row[split_col])
            if split_col is not None
            else "not_explicitly_recorded"
        )

        experiment_rows.append({
            "stage": stage,
            "model": str(row[model_col]),
            "macro_f1": float(value),
            "seed": (
                str(row[cols["seed"]])
                if "seed" in cols
                else ""
            ),
            "split": split,
            "source": str(
                path.relative_to(REPO)
            )
        })

for path in TRACKED:

    if (
        not path.is_file()
        or path.suffix != ".json"
        or "summary" not in path.name.lower()
        or "research_stage9a" in path.parts
    ):
        continue

    try:

        data = json.loads(path.read_text())

        if isinstance(data, dict):

            summary_records.append({
                "stage": stage_of(path),
                "path": str(
                    path.relative_to(REPO)
                ),
                "summary": data
            })

    except Exception:
        continue

TABLE_INVENTORY = save_table(
    table_inventory,
    "stage9a_03_table_inventory"
)

EXPERIMENTS = pd.DataFrame(
    experiment_rows
).drop_duplicates()

save_table(
    EXPERIMENTS,
    "stage9a_04_discovered_experiment_records"
)

write_json(
    summary_records,
    BACKUP / "historical_stage_summaries.json"
)

print("Readable tables:", len(table_inventory))

print(
    "Discovered experimental records:",
    len(EXPERIMENTS)
)

print("Stage summaries:", len(summary_records))

if len(EXPERIMENTS):

    print(
        EXPERIMENTS.groupby("stage")
        .size()
        .to_string()
    )

# ============================================================
# STEP 180 — VERIFIED MODEL COMPARISON
# ============================================================

section("STEP 180 — RECONSTRUCTED-COHORT MODEL COMPARISON")

# Explicitly select known comparable results.
# Historical scores from different cohorts are
# NOT used in this primary comparison.

V8C_PATH = (
    REPO / "research_stage8c"
    / "tables"
    / "stage8c_06_ensemble_comparison.csv"
)

assert V8C_PATH.exists()

v8c = pd.read_csv(V8C_PATH)

expected = {
    "historical_prototype_family":
        0.8287035909704495,
    "rap_full":
        0.819573449382636,
    "generic_control":
        0.8210806495923202,
    "frozen_banglabert":
        0.8159419852657895
}

for name, score in expected.items():

    rows = v8c[v8c["model"] == name]

    assert len(rows) == 1

    assert abs(
        float(rows.iloc[0]["macro_f1"]) - score
    ) < 1e-6

# Previously verified stage reports.
# Additional architecture records are reconciled
# from repository tables below.

confirmed = [
    {
        "stage": "6B",
        "model": "BanglaBERT",
        "macro_f1": 0.8159419852657895,
        "type": "single_model_baseline"
    },
    {
        "stage": "6B",
        "model": "XLM-RoBERTa",
        "macro_f1": 0.783859,
        "type": "single_model_baseline"
    },
    {
        "stage": "6B",
        "model": "mBERT",
        "macro_f1": 0.775318,
        "type": "single_model_baseline"
    },
    {
        "stage": "6E",
        "model": "AMSC-Net fixed fusion",
        "macro_f1": 0.8187778484929229,
        "type": "three_seed_ensemble"
    },
    {
        "stage": "7DE",
        "model": "DGSL-Net",
        "macro_f1": 0.8112616737232622,
        "type": "single_model_proposed"
    },
    {
        "stage": "8B",
        "model": "CER-Net",
        "macro_f1": 0.8200126717683632,
        "type": "three_head_seed_ensemble"
    },
    {
        "stage": "8C",
        "model": "RAP-Net",
        "macro_f1": expected["rap_full"],
        "type": "three_head_seed_ensemble"
    },
    {
        "stage": "8B",
        "model": "Retrieval-only",
        "macro_f1": 0.8248455705705845,
        "type": "three_head_seed_ensemble"
    },
    {
        "stage": "7C",
        "model": "Prototype family",
        "macro_f1":
            expected["historical_prototype_family"],
        "type": "heterogeneous_ensemble"
    }
]

CONFIRMED = pd.DataFrame(confirmed)

CONFIRMED["evidence_scope"] = (
    "Exploratory reconstructed validation cohort"
)

CONFIRMED["source_status"] = (
    "Previously reported; repository reconciliation required"
)

# Identify matching numerical records in the
# historical repository tables.

for i, row in CONFIRMED.iterrows():

    if not len(EXPERIMENTS):
        continue

    matching = EXPERIMENTS[
        (
            EXPERIMENTS["macro_f1"]
            - row["macro_f1"]
        ).abs() < 1e-6
    ]

    matching = matching[
        matching["stage"] == row["stage"]
    ]

    if len(matching):

        CONFIRMED.loc[
            i, "source_status"
        ] = "Numerical repository match"

        CONFIRMED.loc[
            i, "matching_sources"
        ] = " | ".join(
            matching["source"].unique()
        )

CONFIRMED = CONFIRMED.sort_values(
    "macro_f1", ascending=False
)

save_table(
    CONFIRMED,
    "stage9a_05_confirmed_model_comparison"
)

print(CONFIRMED.to_string(index=False))

# Discover HDS-Net/PAS-Net results without
# inventing missing scores.

architecture_candidates = EXPERIMENTS[
    EXPERIMENTS["stage"].isin(
        ["7B", "7C", "8A"]
    )
].copy()

save_table(
    architecture_candidates,
    "stage9a_06_pasnet_hdsnet_candidates"
)

print("\nPAS-NET / HDS-NET EVIDENCE")

if len(architecture_candidates):

    print(
        architecture_candidates.sort_values(
            "macro_f1", ascending=False
        ).head(30).to_string(index=False)
    )

# ============================================================
# EXACT PROTOTYPE FAMILY RECOVERY
# ============================================================

section("STEP 180B — EXACT PROTOTYPE FAMILY RECOVERY")

prototype_assets = [
    a for a in ASSETS

    if a["release"] ==
        "stage7c-specialist-discovery-v1"

    and "private" in a["name"]
    and a["name"].endswith(".zip")
]

assert len(prototype_assets) > 0

prototype_archive = download(
    prototype_assets[0],
    RECOVERED / prototype_assets[0]["name"]
)

PROTOTYPE_CONTENTS = []

with zipfile.ZipFile(prototype_archive) as z:

    assert z.testzip() is None

    PROTOTYPE_CONTENTS = z.namelist()

    for name in PROTOTYPE_CONTENTS:

        if not name.endswith(".npz"):
            continue

        try:

            with np.load(
                io.BytesIO(z.read(name)),
                allow_pickle=False
            ) as data:

                print("\nArchive:", name)

                for key in data.files:

                    value = data[key]

                    print(
                        key,
                        value.shape,
                        value.dtype
                    )

        except Exception as e:

            print(
                "Schema inspection warning:",
                type(e).__name__
            )

save_table(
    [{"archive_path": n}
     for n in PROTOTYPE_CONTENTS],
    "stage9a_07_prototype_archive_inventory"
)

# Inventory exact prototype-related checkpoints
# and metadata needed for future inference.

CHECKPOINT_CANDIDATES = ASSET_DF[
    ASSET_DF["name"].str.contains(
        "prototype|pasnet|large_standard|base_standard",
        case=False, regex=True
    )
].copy()

save_table(
    CHECKPOINT_CANDIDATES.drop(
        columns=["api_url"]
    ),
    "stage9a_08_candidate_checkpoint_inventory"
)

print("\nRECOVERABLE CANDIDATE CHECKPOINTS")

print(
    CHECKPOINT_CANDIDATES[
        ["release", "name", "size"]
    ].to_string(index=False)
)

# ============================================================
# STEP 181 — PRIOR ART AND NOVELTY EVIDENCE
# ============================================================

section("STEP 181 — LITERATURE AND NOVELTY AUDIT")

# These are starting references, not proof
# of architectural novelty.

references = [
    {
        "title":
            "BanglaBERT: Language Model Pretraining "
            "and Benchmarks for Low-Resource "
            "Language Understanding Evaluation in Bangla",
        "year": 2022,
        "topic": "Bangla transformer baseline",
        "url":
            "https://aclanthology.org/2022.findings-naacl.98/",
        "status":
            "Bibliographic verification required"
    },
    {
        "title":
            "SentiGOLD: A Large Bangla Gold Standard "
            "Multi-Domain Sentiment Analysis Dataset "
            "and its Evaluation",
        "year": 2023,
        "topic": "External sentiment evaluation",
        "url": "https://arxiv.org/abs/2306.06147",
        "status": "Identified from prior literature search"
    },
    {
        "title": "Supervised Contrastive Learning",
        "year": 2020,
        "topic": "Contrastive representation learning",
        "url": "https://arxiv.org/abs/2004.11362",
        "status": "Established prior method"
    },
    {
        "title":
            "Prototypical Networks for Few-shot Learning",
        "year": 2017,
        "topic": "Prototype-based classification",
        "url": "https://arxiv.org/abs/1703.05175",
        "status": "Established prior method"
    },
    {
        "title":
            "Sentiment analysis in Bangla: "
            "A three-class dataset and comparative "
            "evaluation of machine learning, deep "
            "learning, and transformer approaches",
        "year": 2026,
        "topic": "Recent Bangla sentiment baseline",
        "url": "https://doi.org/10.1371/journal.pone.0360952",
        "status": "Identified from prior literature search"
    }
]

save_table(
    references,
    "stage9a_09_literature_starting_points"
)

# ============================================================
# STEP 182 — PRELIMINARY EVALUATION PROTOCOL
# ============================================================

section("STEP 182 — FINAL EVALUATION PROTOCOL DRAFT")

PROTOCOL = {

    "status":
        "DRAFT — FINAL ARCHITECTURE APPROVAL PENDING",

    "source_dataset": "BECS",

    "locked_partition_source":
        "stage6b-locked-baselines-v1",

    "partition_seed": 42,

    "training_samples": 9204,

    "validation_samples": 1840,

    "expected_test_samples": 1849,

    "primary_metric": "macro_f1",

    "secondary_metrics": [
        "accuracy",
        "classwise_precision",
        "classwise_recall",
        "classwise_f1",
        "weighted_f1",
        "negative_log_likelihood",
        "brier_score",
        "expected_calibration_error"
    ],

    "baseline_candidates": [
        "BanglaBERT",
        "XLM-RoBERTa",
        "mBERT",
        "TF-IDF Logistic Regression"
    ],

    "architecture_candidates": [
        "PAS-Net",
        "AMSC-Net",
        "HDS-Net",
        "DGSL-Net",
        "CER-Net",
        "RAP-Net"
    ],

    "best_validation_system":
        "Historical prototype family",

    "best_validation_macro_f1":
        0.8287035909704495,

    "final_proposed_architecture": None,

    "ensemble_membership":
        "Must be recovered from exact Stage 7C source",

    "external_primary": {
        "dataset": "SentNoB",
        "training_on_external": False,
        "threshold_tuning_on_external": False,
        "label_mapping":
            "Negative / Neutral / Positive",
        "overlap_audit_required": True
    },

    "external_secondary": {
        "dataset": "SentiGOLD",
        "training_on_external": False,
        "threshold_tuning_on_external": False,
        "label_mapping": {
            "SN": "Negative",
            "WN": "Negative",
            "NU": "Neutral",
            "WP": "Positive",
            "SP": "Positive"
        },
        "overlap_audit_required": True
    },

    "statistics": {
        "paired_group_bootstrap_replicates": 5000,
        "confidence_level": 0.95,
        "mcnemar_test": "secondary",
        "multiplicity_correction": "Holm"
    },

    "linguistic_subgroups": [
        "negation",
        "contrast",
        "both_operators",
        "any_operator",
        "long_text",
        "short_text"
    ],

    "selection_policy": [
        "Use development evidence only.",
        "Lock final architecture before evaluation.",
        "Do not select candidates using test results.",
        "Do not tune external dataset thresholds.",
        "Report all locked comparisons.",
        "Distinguish single-model and ensemble budgets."
    ],

    "limitations": [
        "Repeated exploratory validation use.",
        "Historical partitions overlap reconstructed data.",
        "Reconstructed test is not entirely independent "
        "of all historical experiments.",
        "External dataset overlap must be audited.",
        "Published results from different datasets "
        "are not directly comparable.",
        "No verified novel architecture superiority yet."
    ],

    "test_loaded": False,

    "test_evaluated": False,

    "external_evaluated": False,

    "novelty_verified": False
}

write_json(
    PROTOCOL,
    BACKUP / "evaluation_protocol_draft.json"
)

write_json(
    PROTOCOL,
    AUDIT / "evaluation_protocol_draft.json"
)

# ============================================================
# FIGURE AUDIT AND PUBLICATION FIGURES
# ============================================================

section("STEP 182B — COMPLETE PUBLICATION FIGURE AUDIT")

figure_records = []

for path in TRACKED:

    if not path.is_file():
        continue

    if path.suffix.lower() not in [
        ".png", ".pdf", ".svg"
    ]:
        continue

    if "research_stage9a" in path.parts:
        continue

    figure_records.append({
        "stage": stage_of(path),
        "name": path.stem,
        "format": path.suffix.lower()[1:],
        "path": str(path.relative_to(REPO)),
        "sha256": sha256(path),
        "size": path.stat().st_size
    })

FIGURE_INVENTORY = pd.DataFrame(
    figure_records
)

save_table(
    FIGURE_INVENTORY,
    "stage9a_10_historical_figure_inventory"
)

UNIQUE_FIGURES = (
    FIGURE_INVENTORY.groupby("format")["name"]
    .nunique().to_dict()
)

UNIQUE_HASHES = (
    FIGURE_INVENTORY.groupby("format")["sha256"]
    .nunique().to_dict()
)

print(
    "Unique figure names:",
    UNIQUE_FIGURES
)

print(
    "Unique figure file hashes:",
    UNIQUE_HASHES
)

# Identify same-named figures with different content.

FIGURE_CONFLICTS = (
    FIGURE_INVENTORY
    .groupby(["name", "format"])["sha256"]
    .nunique()
    .reset_index(name="unique_hashes")
)

FIGURE_CONFLICTS = FIGURE_CONFLICTS[
    FIGURE_CONFLICTS["unique_hashes"] > 1
]

save_table(
    FIGURE_CONFLICTS,
    "stage9a_11_figure_version_conflicts"
)

CAPTIONS = []

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})

def save_figure(fig, name, caption):

    for root in [
        FIGURES,
        BACKUP / "figures",
        PUB / "figures"
    ]:

        for ext in ["png", "pdf", "svg"]:

            folder = root / ext

            folder.mkdir(
                parents=True, exist_ok=True
            )

            fig.savefig(
                folder / f"{name}.{ext}",
                dpi=600 if ext == "png" else None,
                bbox_inches="tight"
            )

    CAPTIONS.append({
        "figure": name,
        "caption": caption
    })

    pd.DataFrame(CAPTIONS).to_csv(
        BACKUP / "figure_captions.csv",
        index=False
    )

    push("Preserve " + name)

    plt.show()
    plt.close(fig)

# FIGURE 1 — Overall research comparison.

fig, ax = plt.subplots(figsize=(10, 6))

df = CONFIRMED.sort_values("macro_f1")

ax.barh(
    df["model"],
    df["macro_f1"] * 100
)

ax.set_xlabel("Validation macro-F1 (%)")

ax.set_title(
    "Exploratory comparison of selected research systems"
)

for i, score in enumerate(df["macro_f1"]):

    ax.text(
        score * 100 + 0.2,
        i,
        f"{score * 100:.2f}",
        va="center"
    )

ax.set_xlim(0, 90)

fig.tight_layout()

save_figure(
    fig,
    "stage9a_01_selected_model_comparison",
    "Selected validation results. "
    "Inference budgets differ across systems."
)

# FIGURE 2 — Architectural comparison.

fig, ax = plt.subplots(figsize=(9, 5))

architectures = CONFIRMED[
    CONFIRMED["model"].isin([
        "AMSC-Net fixed fusion",
        "DGSL-Net",
        "CER-Net",
        "RAP-Net"
    ])
].sort_values("macro_f1")

ax.barh(
    architectures["model"],
    architectures["macro_f1"] * 100
)

ax.axvline(
    expected["frozen_banglabert"] * 100,
    linestyle="--",
    label="BanglaBERT"
)

ax.axvline(
    expected["historical_prototype_family"] * 100,
    linestyle=":",
    label="Prototype ensemble"
)

ax.set_xlim(0, 90)

ax.set_xlabel("Validation macro-F1 (%)")

ax.legend()

fig.tight_layout()

save_figure(
    fig,
    "stage9a_02_proposed_architecture_comparison",
    "Exploratory comparison of selected proposed "
    "architectures against baseline systems."
)

# FIGURE 3 — Publication asset inventory.

fig, ax = plt.subplots(figsize=(7, 4))

counts = pd.Series(UNIQUE_FIGURES)

counts.plot.bar(ax=ax)

ax.set_ylabel("Distinct figure names")

ax.set_xlabel("File format")

ax.set_title("Historical publication assets")

fig.tight_layout()

save_figure(
    fig,
    "stage9a_03_publication_asset_inventory",
    "Distinct figure names preserved in GitHub "
    "before Stage 9A figure generation."
)

# FIGURE 4 — Private artifact storage by release.

fig, ax = plt.subplots(figsize=(10, 6))

sizes = (
    ASSET_DF.groupby("release")["size"]
    .sum() / 1024**3
).sort_values()

sizes.plot.barh(ax=ax)

ax.set_xlabel("Storage (GiB)")

ax.set_title(
    "Persistent private research artifacts"
)

fig.tight_layout()

save_figure(
    fig,
    "stage9a_04_private_artifact_inventory",
    "Private checkpoint and reproducibility "
    "artifact storage by research release."
)

# ============================================================
# PERMANENT PRIVATE BACKUP
# ============================================================

section("STEP 182C — VERIFIED PERMANENT BACKUP")

r = SESSION9.get(
    PRIVATE_API + f"/releases/tags/{TAG}",
    timeout=120
)

if r.status_code == 404:

    r = SESSION9.post(
        PRIVATE_API + "/releases",
        json={
            "tag_name": TAG,
            "name": "Stage 9A final research audit",
            "body":
                "Research evidence and evaluation protocol.",
            "draft": False
        },
        timeout=120
    )

r.raise_for_status()

RELEASE = r.json()

remote_assets = {
    a["name"]: a

    for a in paginate(
        PRIVATE_API
        + f"/releases/{RELEASE['id']}/assets"
    )
}

REMOTE_MANIFEST = []

def upload(path):

    path = Path(path)

    digest = sha256(path)

    name = (
        f"{path.stem}_{digest[:12]}"
        f"{path.suffix}"
    )

    asset = remote_assets.get(name)

    if asset is None:

        with open(path, "rb") as f:

            response = SESSION9.post(
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

        remote_assets[name] = asset

    assert asset["size"] == path.stat().st_size

    if asset.get("digest"):

        assert asset["digest"] == "sha256:" + digest

    else:

        record = {
            "api_url": asset["url"],
            "size": asset["size"],
            "digest": asset.get("digest")
        }

        verification = download(
            record,
            RECOVERED / ("verify_" + name)
        )

        assert sha256(verification) == digest

        verification.unlink()

    REMOTE_MANIFEST.append({
        "name": name,
        "asset_id": asset["id"],
        "sha256": digest,
        "size": asset["size"],
        "remote_verified": True
    })

    print("Remote verified:", name)

# Preserve complete current cell source.

cells = get_ipython().user_ns.get("In", [])

SOURCE9 = next(
    s for s in reversed(cells)

    if "STAGE 9A — COMPREHENSIVE RESEARCH FINALIZATION AUDIT" in s
    and len(s) > 10000
)

(BACKUP / "stage9a_complete.py").write_text(
    SOURCE9
)

REPORT = {

    "stage": "9A",

    "repository_audit": "PASS",

    "historical_tracked_files": len(TRACKED),

    "historical_private_releases": len(RELEASES),

    "historical_private_assets": len(ASSETS),

    "readable_experimental_tables":
        len(TABLE_INVENTORY),

    "discovered_experimental_records":
        len(EXPERIMENTS),

    "prototype_archive_recovered": True,

    "candidate_checkpoint_assets":
        len(CHECKPOINT_CANDIDATES),

    "historical_figure_names":
        UNIQUE_FIGURES,

    "historical_figure_hashes":
        UNIQUE_HASHES,

    "figure_version_conflicts":
        len(FIGURE_CONFLICTS),

    "new_publication_figures":
        len(CAPTIONS),

    "best_validation_system":
        "Historical prototype family",

    "best_validation_macro_f1":
        0.8287035909704495,

    "final_architecture_locked": False,

    "novelty_verified": False,

    "test_loaded": False,

    "test_evaluated": False,

    "external_evaluated": False,

    "manual_saving_required": False,

    "google_drive_used": False,

    "decision":
        "RECOVER EXACT PROTOTYPE SYSTEM AND "
        "FINALIZE COMPARATIVE EVALUATION PROTOCOL"
}

write_json(
    REPORT,
    BACKUP / "stage9a_summary.json"
)

write_json(
    REPORT,
    AUDIT / "stage9a_summary.json"
)

archive = ROOT / "stage9a_research_audit.zip"

with zipfile.ZipFile(
    archive, "w", zipfile.ZIP_DEFLATED
) as z:

    for path in BACKUP.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(BACKUP)
            )

with zipfile.ZipFile(archive) as z:

    assert z.testzip() is None

upload(archive)

manifest = AUDIT / "stage9a_remote_manifest.json"

write_json(
    REMOTE_MANIFEST,
    manifest
)

shutil.copy2(
    manifest,
    BACKUP / manifest.name
)

upload(manifest)

FINAL_COMMIT = push(
    "Complete comprehensive Stage 9A research audit"
)

# ============================================================
# FINAL RESEARCH REPORT
# ============================================================

section("FINAL STAGE 9A RESEARCH REPORT")

print("\nCONFIRMED SELECTED MODELS")

print(
    CONFIRMED[
        [
            "stage",
            "model",
            "macro_f1",
            "type",
            "source_status"
        ]
    ].to_string(index=False)
)

print("\nPAS-NET AND HDS-NET CANDIDATES")

if len(architecture_candidates):

    print(
        architecture_candidates.sort_values(
            "macro_f1", ascending=False
        ).head(30).to_string(index=False)
    )

print("\nRECOVERABLE CHECKPOINTS")

print(
    CHECKPOINT_CANDIDATES[
        ["release", "name", "size"]
    ].to_string(index=False)
)

print("\nRESEARCH SUMMARY")

print(json.dumps(REPORT, indent=2))

print("\nPERMANENT BACKUP")

print("Final commit:", FINAL_COMMIT)
print("Private release:", RELEASE["html_url"])
print("Remote verified assets:", len(REMOTE_MANIFEST))
print("GitHub synchronized: True")
print("Manual saving required: NO")
print("Google Drive used: NO")
print("New model training: NO")
print("Test evaluated: NO")

print(
    "Runtime minutes:",
    round((time.time() - START9) / 60, 2)
)

print("\nRESEARCH DECISION:", REPORT["decision"])
# ============================================================
# STAGE 9C — FINAL CHECKPOINT RECOVERY AND PROTOCOL LOCK
# STEPS 187–192
# ============================================================

import os
import io
import ast
import json
import time
import base64
import hashlib
import getpass
import zipfile
import subprocess
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score

START = time.time()

WORK = Path("/kaggle/working")
OUT = WORK / "sentiment_research_stage9c"
REC = OUT / "recovered"
MODELS = OUT / "checkpoints"
AUDIT = OUT / "audit"

REPO = WORK / "sentiment_stage9c_github"
PUB = WORK / "sentiment_publication_assets"

for p in [OUT, REC, MODELS, AUDIT, PUB]:
    p.mkdir(parents=True, exist_ok=True)

OWNER = "ankita6876"
PUBLIC_REPO = "bangla-sentiment-research"
PRIVATE_REPO = "bangla-sentiment-research-artifacts"

BRANCH = "research/reconstruction-20261127-080232"
EXPECTED_COMMIT = "812c74faf0c62b396b068c508a5b160b36793257"

TAG = "stage9c-locked-inference-protocol-v1"
PAS_TAG = "stage7b-pasnet-screening-v1"
BASE_TAG = "stage6b-locked-baselines-v1"
PREV_TAG = "stage9b-final-model-recovery-v1"

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
# STEP 187 — GITHUB RECOVERY
# ============================================================

section("STEP 187 — VERIFIED GITHUB RECOVERY")

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

PRIVATE_API = (
    f"{API}/repos/{OWNER}/{PRIVATE_REPO}"
)

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

    p = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=ENV,
        capture_output=True,
        text=True
    )

    if p.returncode:
        raise RuntimeError(
            "Git operation failed: " + args[0]
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
    EXPECTED_COMMIT, "HEAD"
)

git("config", "user.name", "Research Backup")

git(
    "config", "user.email",
    "research-backup@users.noreply.github.com"
)

BACKUP = REPO / "research_stage9c"
BACKUP.mkdir(exist_ok=True)

def push(message):

    git("add", "research_stage9c")

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

def get_release(tag):
    return api_get(
        PRIVATE_API + f"/releases/tags/{tag}"
    )

def assets(tag):

    r = get_release(tag)

    return paginate(
        PRIVATE_API
        + f"/releases/{r['id']}/assets"
    )

def download(asset, destination, expected=None):

    destination = Path(destination)

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    digest = asset.get("digest")

    if digest:
        remote_hash = digest.removeprefix("sha256:")

        if expected:
            assert expected == remote_hash

        expected = remote_hash

    assert expected, (
        "No trusted checksum:",
        asset["name"]
    )

    if (
        destination.exists()
        and destination.stat().st_size == asset["size"]
        and sha256(destination) == expected
    ):
        print("Cached and verified:", asset["name"])
        return destination

    temporary = destination.with_suffix(
        destination.suffix + ".partial"
    )

    with SESSION.get(
        asset["url"],
        headers={
            "Accept": "application/octet-stream"
        },
        stream=True,
        timeout=(30, 600)
    ) as r:

        r.raise_for_status()

        with open(temporary, "wb") as f:

            for chunk in r.iter_content(1024 * 1024):

                if chunk:
                    f.write(chunk)

    assert temporary.stat().st_size == asset["size"]
    assert sha256(temporary) == expected

    temporary.replace(destination)

    print("Downloaded and SHA256 verified:", asset["name"])

    return destination

def save_table(data, name):

    df = pd.DataFrame(data)

    for folder in [
        OUT / "tables",
        BACKUP / "tables",
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

print("Authentication: PASS")
print("Previous commit: VERIFIED")
print("CUDA available:", torch.cuda.is_available())

# Preserve complete current source.

cells = get_ipython().user_ns.get("In", [])

SOURCE = next(
    s for s in reversed(cells)
    if "STAGE 9C — FINAL CHECKPOINT RECOVERY AND PROTOCOL LOCK" in s
    and len(s) > 10000
)

(BACKUP / "stage9c_complete.py").write_text(SOURCE)

push("Preserve Stage 9C source")

# ============================================================
# STEP 188 — RECOVER VERIFIED VALIDATION EVIDENCE
# ============================================================

section("STEP 188 — EXACT VALIDATION RECOVERY")

previous_assets = assets(PREV_TAG)

pred_asset = next(
    a for a in previous_assets
    if a["name"].startswith(
        "stage9b_aligned_predictions_"
    )
    and a["name"].endswith(".npz")
)

pred_path = download(
    pred_asset,
    REC / pred_asset["name"]
)

with np.load(
    pred_path,
    allow_pickle=False
) as data:

    IDS = data["sample_id"].astype(str)
    Y = data["labels"].astype(int)
    GROUPS = data["group_id"].astype(str)

    NAMES = data["model_names"].astype(str)
    P = data["probabilities"].astype(np.float64)

SYSTEMS = {
    name: P[i]
    for i, name in enumerate(NAMES)
}

assert len(Y) == 1840
assert len(np.unique(IDS)) == 1840
assert np.isfinite(P).all()
assert np.allclose(P.sum(axis=-1), 1, atol=1e-5)

PROPOSED = [
    "base_prototype",
    "base_pasnet",
    "large_pasnet"
]

# Existing three-member standard ensemble:
# two base backbones and one large backbone.
# The locked BanglaBERT has a different
# classifier/training implementation.

CONTROL = [
    "locked_banglabert",
    "base_standard",
    "large_standard"
]

candidate = np.mean(
    [SYSTEMS[n] for n in PROPOSED],
    axis=0
)

control = np.mean(
    [SYSTEMS[n] for n in CONTROL],
    axis=0
)

assert np.allclose(
    candidate,
    SYSTEMS["prototype_family"],
    atol=1e-10
)

def score(p):

    return f1_score(
        Y,
        p.argmax(axis=1),
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

assert abs(
    score(candidate) - 0.8287035909704495
) < 1e-10

SYSTEMS["three_member_standard_control"] = control

comparison_names = [
    "locked_banglabert",
    "base_standard",
    "large_standard",
    "base_prototype",
    "base_pasnet",
    "large_pasnet",
    "base_large_standard_ensemble",
    "base_large_pasnet_ensemble",
    "three_member_standard_control",
    "prototype_family"
]

RESULTS = save_table(
    [
        {
            "model": name,
            "accuracy": accuracy_score(
                Y,
                SYSTEMS[name].argmax(axis=1)
            ),
            "macro_f1": score(SYSTEMS[name]),
            "scope": "Exploratory validation"
        }

        for name in comparison_names
    ],
    "stage9c_01_final_candidate_comparison"
).sort_values(
    "macro_f1",
    ascending=False
)

print(RESULTS.to_string(index=False))

print(
    "\nPrototype versus three-member control:",
    round(
        100 * (
            score(candidate) - score(control)
        ),
        4
    ),
    "percentage points"
)

# ============================================================
# STEP 189 — RECOVER ACTUAL MODEL CHECKPOINTS
# ============================================================

section("STEP 189 — ACTUAL CHECKPOINT BYTE VERIFICATION")

pas_assets = {
    a["name"]: a
    for a in assets(PAS_TAG)
}

base_assets = {
    a["name"]: a
    for a in assets(BASE_TAG)
}

REQUIRED_PAS = [
    "base_standard",
    "base_supcon",
    "base_prototype",
    "base_pasnet",
    "large_standard",
    "large_pasnet"
]

CHECKPOINT_RECORDS = []
CHECKPOINT_SCHEMAS = []
METADATA = {}

def inspect_checkpoint(
    name, path, expected_hash,
    backbone, mode, release_tag
):

    assert sha256(path) == expected_hash

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    if (
        isinstance(state, dict)
        and "state_dict" in state
    ):
        state = state["state_dict"]

    elif (
        isinstance(state, dict)
        and "model_state_dict" in state
    ):
        state = state["model_state_dict"]

    assert isinstance(state, dict)

    tensors = {
        k: v for k, v in state.items()
        if isinstance(v, torch.Tensor)
    }

    assert len(tensors) > 0

    for key, value in tensors.items():

        CHECKPOINT_SCHEMAS.append({
            "model": name,
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "elements": value.numel()
        })

    record = {
        "model": name,
        "backbone": backbone,
        "mode": mode,
        "release": release_tag,
        "filename": path.name,
        "sha256": expected_hash,
        "size_bytes": path.stat().st_size,
        "tensor_count": len(tensors),
        "state_tensor_elements": sum(
            v.numel() for v in tensors.values()
        ),
        "checkpoint_bytes_verified": True,
        "deserialization_passed": True,
        "strict_model_reload_verified": False
    }

    CHECKPOINT_RECORDS.append(record)

    print(
        name,
        "| tensors:", len(tensors),
        "| size MiB:",
        round(path.stat().st_size / 1024**2, 2)
    )

    del state, tensors

    import gc
    gc.collect()

for name in REQUIRED_PAS:

    prefix = f"{name}_seed42"

    metadata_asset = pas_assets[
        prefix + "_completed.json"
    ]

    metadata_path = download(
        metadata_asset,
        REC / metadata_asset["name"]
    )

    info = json.loads(
        metadata_path.read_text()
    )

    METADATA[name] = info

    checkpoint_asset = pas_assets[
        prefix + "_model.pt"
    ]

    checkpoint = download(
        checkpoint_asset,
        MODELS / checkpoint_asset["name"],
        expected=info["checkpoint_sha256"]
    )

    inspect_checkpoint(
        name,
        checkpoint,
        info["checkpoint_sha256"],
        info["model_name"],
        info["mode"],
        PAS_TAG
    )

# Recover locked baseline checkpoints.

BASELINE_FILES = {
    "locked_banglabert":
        "BanglaBERT_seed42_best_model.pt",

    "xlm_roberta":
        "XLM-RoBERTa_seed42_best_model.pt",

    "mbert":
        "mBERT_seed42_best_model.pt"
}

BASELINE_BACKBONES = {
    "locked_banglabert":
        "csebuetnlp/banglabert",

    "xlm_roberta":
        "FacebookAI/xlm-roberta-base",

    "mbert":
        "google-bert/bert-base-multilingual-cased"
}

for name, filename in BASELINE_FILES.items():

    asset = base_assets[filename]

    checkpoint = download(
        asset,
        MODELS / filename
    )

    inspect_checkpoint(
        name,
        checkpoint,
        sha256(checkpoint),
        BASELINE_BACKBONES[name],
        "standard",
        BASE_TAG
    )

CHECKPOINTS = save_table(
    CHECKPOINT_RECORDS,
    "stage9c_02_verified_checkpoints"
)

write_json(
    CHECKPOINT_SCHEMAS,
    AUDIT / "checkpoint_tensor_schemas.json"
)

write_json(
    CHECKPOINT_SCHEMAS,
    BACKUP / "checkpoint_tensor_schemas.json"
)

print(
    "\nTotal actual checkpoints verified:",
    len(CHECKPOINTS)
)

# ============================================================
# STEP 190 — EXACT ARCHITECTURE IMPLEMENTATION RECOVERY
# ============================================================

section("STEP 190 — EXACT MODEL SOURCE RECOVERY")

source_path = (
    REPO / "research_stage7b"
    / "stage7b_complete.py"
)

assert source_path.exists()

source_text = source_path.read_text()

tree = ast.parse(source_text)

source_lines = source_text.splitlines()

SOURCE_RECORDS = []
EXCERPTS = []

for node in ast.walk(tree):

    if not isinstance(
        node,
        (
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef
        )
    ):
        continue

    SOURCE_RECORDS.append({
        "name": node.name,
        "kind": type(node).__name__,
        "start_line": node.lineno,
        "end_line": node.end_lineno
    })

    # Preserve complete class definitions and
    # relevant model/tokenization functions.

    relevant = (
        isinstance(node, ast.ClassDef)
        or any(
            term in node.name.lower()
            for term in [
                "model",
                "encode",
                "token",
                "dataset",
                "loader",
                "evaluate",
                "predict",
                "prototype",
                "supcon",
                "checkpoint"
            ]
        )
    )

    if relevant:

        excerpt = "\n".join(
            source_lines[
                node.lineno - 1:node.end_lineno
            ]
        )

        EXCERPTS.append(
            f"\n# {node.name}\n" + excerpt
        )

SOURCE_INVENTORY = save_table(
    SOURCE_RECORDS,
    "stage9c_03_source_definition_inventory"
)

(BACKUP / "original_stage7b_source.py").write_text(
    source_text
)

(AUDIT / "architecture_source_excerpts.txt").write_text(
    "\n\n".join(EXCERPTS)
)

(BACKUP / "architecture_source_excerpts.txt").write_text(
    "\n\n".join(EXCERPTS)
)

print("\nEXACT ARCHITECTURE DEFINITIONS")

for node in tree.body:

    if isinstance(node, ast.ClassDef):

        print(
            "\n" + "-" * 70
        )

        print(
            "\n".join(
                source_lines[
                    node.lineno - 1:node.end_lineno
                ]
            )
        )

print("\nSOURCE DEFINITION INVENTORY")

print(SOURCE_INVENTORY.to_string(index=False))

# Recover important training configuration
# directly from original source.

CONFIG_LINES = []

for i, line in enumerate(source_lines):

    if any(
        key in line
        for key in [
            "MAX_LEN",
            "MAX_LENGTH",
            "DIM =",
            "DROPOUT",
            "AutoTokenizer",
            "AutoModel",
            "normalize(",
            "normalize as",
            "load_state_dict",
            "model.eval()",
            "softmax("
        ]
    ):

        CONFIG_LINES.append(
            f"{i+1}: {line}"
        )

config_text = "\n".join(CONFIG_LINES)

(BACKUP / "inference_configuration_evidence.txt").write_text(
    config_text
)

print("\nINFERENCE CONFIGURATION EVIDENCE")
print(config_text)

# ============================================================
# STEP 191 — LOCK CANDIDATE AND EVALUATION POLICY
# ============================================================

section("STEP 191 — FINAL EVALUATION POLICY")

PROTOCOL = {
    "status":
        "CANDIDATE_MEMBERSHIP_LOCKED_INFERENCE_PENDING",

    "proposed_architecture":
        "Prototype-Guided Adaptive Sentiment Network",

    "proposed_architecture_abbreviation":
        "PAS-Net",

    "primary_candidate_system":
        "prototype_family",

    "candidate_members": PROPOSED,

    "candidate_aggregation":
        "Equal arithmetic mean of class probabilities",

    "three_member_standard_control": CONTROL,

    "control_limitation":
        "Comparable backbone composition, but not "
        "identical classifier and training protocol.",

    "source_dataset": "BECS",

    "partition_seed": 42,

    "partition_release": BASE_TAG,

    "expected_train_samples": 9204,

    "expected_validation_samples": 1840,

    "expected_test_samples": 1849,

    "primary_metric": "macro_f1",

    "primary_comparison":
        "prototype_family versus three_member_standard_control",

    "secondary_comparisons": [
        "locked_banglabert",
        "xlm_roberta",
        "mbert",
        "tfidf_logistic_regression",
        "base_standard",
        "large_standard",
        "base_supcon",
        "base_prototype",
        "base_pasnet",
        "large_pasnet",
        "base_large_standard_ensemble",
        "base_large_pasnet_ensemble"
    ],

    "external_candidates": [
        "SentNoB",
        "SentiGOLD"
    ],

    "external_overlap_audit_required": True,

    "external_label_mapping_verified": False,

    "external_threshold_tuning": False,

    "external_fine_tuning": False,

    "bootstrap_replicates": 5000,

    "confidence_level": 0.95,

    "secondary_significance_test":
        "McNemar with Holm correction",

    "checkpoint_inventory":
        CHECKPOINT_RECORDS,

    "architecture_source_sha256":
        sha256(source_path),

    "validation_candidate_macro_f1":
        float(score(candidate)),

    "validation_control_macro_f1":
        float(score(control)),

    "checkpoint_bytes_verified": True,

    "strict_model_reload_verified": False,

    "validation_inference_reproduced": False,

    "test_loaded": False,

    "test_evaluated": False,

    "external_evaluated": False,

    "novelty_verified": False,

    "state_of_the_art_claim_supported": False,

    "selection_rules": [
        "Do not change ensemble membership after test access.",
        "Do not optimize ensemble weights using test labels.",
        "Reproduce saved validation inference before test.",
        "Report all prespecified comparisons.",
        "Report parameter and inference cost differences.",
        "Treat validation statistics as exploratory."
    ],

    "limitations": [
        "Repeated validation-set use.",
        "Historical exposure to overlapping source samples.",
        "No fully training-matched three-member control yet.",
        "Architectural novelty not established.",
        "External evaluation not yet performed."
    ]
}

write_json(
    PROTOCOL,
    BACKUP / "locked_candidate_protocol.json"
)

write_json(
    PROTOCOL,
    AUDIT / "locked_candidate_protocol.json"
)

print(json.dumps(
    {
        k: v for k, v in PROTOCOL.items()
        if k != "checkpoint_inventory"
    },
    indent=2
))

# ============================================================
# STEP 192 — PUBLICATION FIGURES
# ============================================================

section("STEP 192 — PUBLICATION FIGURES AND BACKUP")

CAPTIONS = []

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})

def save_figure(fig, name, caption):

    for root in [
        OUT / "figures",
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

    save_table(
        CAPTIONS,
        "stage9c_figure_captions"
    )

    push("Preserve " + name)

    plt.show()
    plt.close(fig)

# Figure 1 — Comparable backbone composition.

fig, ax = plt.subplots(figsize=(9, 5))

names = [
    "locked_banglabert",
    "base_large_standard_ensemble",
    "three_member_standard_control",
    "prototype_family"
]

display = [
    "Locked BanglaBERT",
    "Two-member standard",
    "Three-member standard",
    "Prototype-family ensemble"
]

values = [
    100 * score(SYSTEMS[n])
    for n in names
]

ax.barh(display, values)

for i, v in enumerate(values):

    ax.text(
        v + 0.2,
        i,
        f"{v:.2f}",
        va="center"
    )

ax.set_xlim(0, 90)

ax.set_xlabel("Validation macro-F1 (%)")

ax.set_title(
    "Candidate and comparable-backbone controls"
)

fig.tight_layout()

save_figure(
    fig,
    "stage9c_01_backbone_control_comparison",
    "Exploratory validation comparison. "
    "The three-member control matches the base/large "
    "backbone composition but not every training detail."
)

# Figure 2 — Checkpoint storage.

fig, ax = plt.subplots(figsize=(9, 5))

df = CHECKPOINTS.sort_values(
    "size_bytes"
)

ax.barh(
    df["model"],
    df["size_bytes"] / 1024**2
)

ax.set_xlabel("Stored checkpoint size (MiB)")

ax.set_title("Verified checkpoint inventory")

fig.tight_layout()

save_figure(
    fig,
    "stage9c_02_checkpoint_inventory",
    "Actual verified checkpoint storage sizes. "
    "Storage size is not equivalent to inference cost "
    "because checkpoint precision differs."
)

# ============================================================
# PERMANENT PRIVATE BACKUP
# ============================================================

section("VERIFIED PERMANENT PRIVATE BACKUP")

r = SESSION.get(
    PRIVATE_API + f"/releases/tags/{TAG}",
    timeout=120
)

if r.status_code == 404:

    r = SESSION.post(
        PRIVATE_API + "/releases",
        json={
            "tag_name": TAG,
            "name": "Stage 9C locked inference protocol",
            "body":
                "Verified checkpoint recovery and "
                "candidate evaluation policy.",
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
            REC / ("verify_" + name),
            expected=digest
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

# Store aligned control predictions privately.

prediction_path = OUT / "stage9c_locked_predictions.npz"

np.savez_compressed(
    prediction_path,
    sample_id=IDS,
    labels=Y,
    group_id=GROUPS,
    candidate_probabilities=candidate,
    control_probabilities=control
)

upload(prediction_path)

REPORT = {
    "stage": "9C",
    "candidate_membership_locked": True,
    "candidate_members": PROPOSED,
    "control_members": CONTROL,
    "candidate_validation_macro_f1":
        float(score(candidate)),
    "control_validation_macro_f1":
        float(score(control)),
    "validation_delta_macro_f1":
        float(score(candidate) - score(control)),
    "actual_checkpoints_verified":
        len(CHECKPOINTS),
    "strict_model_reload_verified": False,
    "validation_inference_reproduced": False,
    "architecture_source_recovered": True,
    "new_figures": len(CAPTIONS),
    "new_training": False,
    "test_loaded": False,
    "test_evaluated": False,
    "external_evaluated": False,
    "novelty_verified": False,
    "manual_saving_required": False,
    "google_drive_used": False,
    "decision":
        "REPRODUCE EXACT VALIDATION INFERENCE "
        "THEN EVALUATE LOCKED TEST AND EXTERNAL DATA"
}

write_json(
    REPORT,
    BACKUP / "stage9c_summary.json"
)

package = OUT / "stage9c_recovery_package.zip"

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

manifest = AUDIT / "stage9c_remote_manifest.json"

write_json(MANIFEST, manifest)

write_json(
    MANIFEST,
    BACKUP / "stage9c_remote_manifest.json"
)

upload(manifest)

FINAL_COMMIT = push(
    "Complete Stage 9C checkpoint verification "
    "and candidate protocol lock"
)

# Verify cumulative figure inventory.

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

section("FINAL STAGE 9C RESEARCH REPORT")

print("\nFINAL CANDIDATE COMPARISON")
print(RESULTS.to_string(index=False))

print("\nVERIFIED CHECKPOINTS")
print(CHECKPOINTS.to_string(index=False))

print("\nRESEARCH SUMMARY")
print(json.dumps(REPORT, indent=2))

print("\nPERMANENT PUBLICATION ASSETS")

for ext, count in figure_counts.items():
    print(ext.upper(), "figures:", count)

print("Total figure files:", sum(figure_counts.values()))

print("Remote verified assets:", len(MANIFEST))
print("Final public commit:", FINAL_COMMIT)
print("Private release:", RELEASE["html_url"])

print("GitHub synchronized: True")
print("Manual saving required: NO")
print("Google Drive used: NO")
print("New training: NO")
print("Test evaluated: NO")

print(
    "Runtime minutes:",
    round((time.time() - START) / 60, 2)
)

print("\nRESEARCH DECISION:", REPORT["decision"])
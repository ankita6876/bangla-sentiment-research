MODEL_ALIAS12 = {'BanglaBERT': 'locked_banglabert', 'XLM-RoBERTa': 'xlm_roberta', 'mBERT': 'mbert', 'base_standard': 'base_standard', 'base_supcon': 'base_supcon', 'base_prototype': 'base_prototype', 'base_pasnet': 'base_pasnet', 'large_standard': 'large_standard', 'large_pasnet': 'large_pasnet'}

# ==================================================================
# STAGE 11B–12A — CONSOLIDATED EXTERNAL MODEL EVALUATION
# Nine models | Two datasets | Statistical analysis | GitHub backup
# No retraining | No Google Drive | No manual file saving
# ==================================================================

import os
import gc
import re
import ast
import json
import time
import shutil
import hashlib
import zipfile
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    log_loss
)

warnings.filterwarnings("ignore")

START12 = time.time()

WORK12 = Path("/kaggle/working")
ROOT12 = WORK12 / "sentiment_research_stage11b12a"
PRED12 = ROOT12 / "private_predictions"
TABLE12 = ROOT12 / "tables"
AUDIT12 = ROOT12 / "audit"
REC12 = ROOT12 / "recovered"

for p in [ROOT12, PRED12, TABLE12, AUDIT12, REC12]:
    p.mkdir(parents=True, exist_ok=True)

assert torch.cuda.is_available(), "Enable GPU in Kaggle."

DEVICE12 = torch.device("cuda")
LABELS12 = ["Negative", "Neutral", "Positive"]
TAG12 = "stage11b12a-external-evaluation-v1"
BOOT12 = 5000

def section12(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

# ==================================================================
# STEP 227 — VERIFY CURRENT RESEARCH ENVIRONMENT
# ==================================================================

section12("STEP 227 — RESEARCH ENVIRONMENT VERIFICATION")

required = [
    "REPO11", "S11", "PRIVATE_API", "PUBLIC_API",
    "ASSETS11", "MEMBERSHIP11", "git11",
    "download11", "sha256", "write_json",
    "BRANCH", "TOKEN11"
]

missing = [x for x in required if x not in globals()]

assert not missing, (
    "Previous Stage 11A environment is unavailable: "
    + str(missing)
    + ". Rerun Stage 11A first."
)

git11("pull", "--ff-only", "origin", BRANCH)

BACKUP12 = REPO11 / "research_stage11b12a"

for p in [
    "tables", "audit", "figures/png",
    "figures/pdf", "figures/svg", "captions"
]:
    (BACKUP12 / p).mkdir(parents=True, exist_ok=True)

def push12(message):

    git11("add", "research_stage11b12a")

    if git11("status", "--porcelain"):
        git11("commit", "-m", message)
        git11("push", "origin", BRANCH)

    head = git11("rev-parse", "HEAD")

    remote = git11(
        "ls-remote", "origin",
        f"refs/heads/{BRANCH}"
    ).split()[0]

    assert head == remote
    print("GitHub synchronized:", head)

    return head

def table12(df, name):

    path = TABLE12 / f"{name}.csv"
    df.to_csv(path, index=False)

    shutil.copy2(
        path, BACKUP12 / "tables" / path.name
    )

    (BACKUP12 / "tables" / f"{name}.tex").write_text(
        df.to_latex(
            index=False,
            escape=True,
            float_format=lambda x: f"{x:.4f}"
        ),
        encoding="utf-8"
    )

    return path

print("GPU:", torch.cuda.get_device_name(0))
print("Research branch:", BRANCH)

# ==================================================================
# STEP 228 — LOCKED EXTERNAL DATA RECOVERY
# ==================================================================

section12("STEP 228 — EXTERNAL DATA INTEGRITY")

DATA12 = {}

for dataset in ["SentNoB", "SentiGOLD"]:

    path = (
        WORK12
        / "sentiment_research_stage10b"
        / "private_data"
        / f"{dataset}_audited.csv"
    )

    assert path.exists(), f"Missing dataset: {path}"

    frame = pd.read_csv(
        path, dtype={"sample_id": str}
    )

    membership = MEMBERSHIP11[dataset]

    assert (
        frame["sample_id"].tolist()
        == membership["sample_id"].tolist()
    )

    assert (
        frame["label_id"].tolist()
        == membership["label_id"].tolist()
    )

    # Verify text identity, not only membership order.

    calculated_ids = frame["text_norm"].map(
        lambda x: hashlib.sha256(
            str(x).encode("utf-8")
        ).hexdigest()
    )

    assert (
        calculated_ids.tolist()
        == frame["sample_id"].tolist()
    )

    assert frame["sample_id"].is_unique

    DATA12[dataset] = frame

    print(dataset, "| Verified:", len(frame))

# ==================================================================
# STEP 229 — RECOVER ORIGINAL INFERENCE IMPLEMENTATION
# ==================================================================

section12("STEP 229 — ORIGINAL MODEL IMPLEMENTATION")

SOURCE12 = (
    REPO11
    / "research_stage9d10a"
    / "stage9d10a_complete.py"
)

PAS_SOURCE12 = (
    REPO11
    / "research_stage9d10a"
    / "original_pasnet_source.py"
)

assert SOURCE12.exists()
assert PAS_SOURCE12.exists()

assert sha256(SOURCE12) == (
    "57fc6777cccaa4914a2c9ad8e374e8d210"
    "380ad5890731f5f210bac280be4aaf"
)

assert sha256(PAS_SOURCE12) == (
    "99bce9be8b2404d758e4c065e3e39a4fa"
    "ec1cef786ab0aabbd4e50e441f930ae"
)

SOURCE_TEXT12 = SOURCE12.read_text(encoding="utf-8")
TREE12 = ast.parse(SOURCE_TEXT12)

shutil.copy2(
    SOURCE12,
    BACKUP12 / "original_inference_source.py"
)

shutil.copy2(
    PAS_SOURCE12,
    BACKUP12 / "original_pasnet_source.py"
)

FUNCTIONS12 = {
    node.name: node
    for node in TREE12.body
    if isinstance(node, ast.FunctionDef)
}

assert "build_model" in FUNCTIONS12
assert "inference" in FUNCTIONS12

# Prefer the original live functions from Stage 9D–10A.
# Verify that the functions match the archived implementation.

BUILD12 = globals().get("build_model")
INFER12 = globals().get("inference")

def verify_function12(fn, expected_node):

    if not callable(fn):
        return False

    try:
        import textwrap

        recovered = ast.parse(
            textwrap.dedent(inspect.getsource(fn))
        ).body[0]

        return (
            ast.dump(recovered, include_attributes=False)
            == ast.dump(
                expected_node,
                include_attributes=False
            )
        )

    except Exception:
        return False

LIVE_OK12 = (
    verify_function12(
        BUILD12, FUNCTIONS12["build_model"]
    )
    and verify_function12(
        INFER12, FUNCTIONS12["inference"]
    )
)

# If earlier functions do not exist in memory, extract
# the archived function definitions without executing
# the historical test evaluation or training pipeline.

if not LIVE_OK12:

    scope12 = dict(globals())

    for name in ["build_model", "inference"]:

        node = FUNCTIONS12[name]

        module = ast.Module(
            body=[node], type_ignores=[]
        )

        exec(
            compile(
                ast.fix_missing_locations(module),
                str(SOURCE12),
                "exec"
            ),
            scope12
        )

    BUILD12 = scope12["build_model"]
    INFER12 = scope12["inference"]

    print("Archived inference functions extracted.")

else:

    print("Live inference functions match archived source.")

print("\nORIGINAL MODEL LOADER")
print(ast.get_source_segment(
    SOURCE_TEXT12,
    FUNCTIONS12["build_model"]
))

print("\nORIGINAL INFERENCE FUNCTION")
print(ast.get_source_segment(
    SOURCE_TEXT12,
    FUNCTIONS12["inference"]
))

# Identify missing runtime dependencies before inference.

dependency_rows12 = []

for fn in [BUILD12, INFER12]:

    dependencies = inspect.getclosurevars(fn)

    dependency_rows12.append({
        "function": fn.__name__,
        "missing_or_unresolved_names": sorted(
            dependencies.unbound
        )
    })

write_json(
    dependency_rows12,
    AUDIT12 / "inference_dependencies.json"
)

shutil.copy2(
    AUDIT12 / "inference_dependencies.json",
    BACKUP12 / "audit"
)

# ==================================================================
# STEP 230 — PRIVATE RELEASE AND RESUMABLE BACKUP
# ==================================================================

section12("STEP 230 — PRIVATE ARTIFACT VAULT")

release12 = S11.get(
    f"{PRIVATE_API}/releases/tags/{TAG12}",
    timeout=60
)

if release12.status_code == 404:

    release12 = S11.post(
        f"{PRIVATE_API}/releases",
        json={
            "tag_name": TAG12,
            "name": (
                "Stage 11B–12A — External evaluation"
            ),
            "body": (
                "Verified external predictions, "
                "statistical comparisons and "
                "publication assets. No raw text."
            ),
            "draft": False,
            "prerelease": False
        },
        timeout=60
    )

release12.raise_for_status()

REL12 = release12.json()
RID12 = REL12["id"]
UPLOAD12 = REL12["upload_url"].split("{")[0]

REMOTE12 = []

def existing_assets12():

    response = S11.get(
        f"{PRIVATE_API}/releases/{RID12}/assets",
        params={"per_page": 100},
        timeout=60
    )

    response.raise_for_status()

    return response.json()

def upload12(path):

    path = Path(path)
    digest = sha256(path)

    name = (
        f"{path.stem}_{digest[:12]}{path.suffix}"
    )

    existing = next(
        (
            a for a in existing_assets12()
            if a["name"] == name
        ),
        None
    )

    if existing is None:

        with open(path, "rb") as f:

            response = S11.post(
                UPLOAD12,
                params={"name": name},
                data=f,
                headers={
                    "Content-Type":
                    "application/octet-stream"
                },
                timeout=600
            )

        response.raise_for_status()
        existing = response.json()

    assert existing["size"] == path.stat().st_size

    assert existing.get("digest") == (
        "sha256:" + digest
    ), "Remote digest verification failed."

    record = {
        "local_name": path.name,
        "remote_name": name,
        "asset_id": existing["id"],
        "sha256": digest,
        "bytes": path.stat().st_size,
        "remote_verified": True
    }

    REMOTE12.append(record)

    print("Remote verified:", name)

    return record

print("Private release:", REL12["html_url"])

# ==================================================================
# STEP 231 — EXTERNAL INFERENCE PROTOCOL
# ==================================================================

section12("STEP 231 — FREEZE EXTERNAL EVALUATION PROTOCOL")

MODELS12 = [
    "BanglaBERT",
    "XLM-RoBERTa",
    "mBERT",
    "base_standard",
    "base_supcon",
    "base_prototype",
    "base_pasnet",
    "large_standard",
    "large_pasnet"
]

ENSEMBLES12 = {
    "proposed_prototype_family": [
        "base_prototype",
        "base_pasnet",
        "large_pasnet"
    ],
    "base_large_standard": [
        "base_standard",
        "large_standard"
    ],
    "three_member_standard": [
        "BanglaBERT",
        "base_standard",
        "large_standard"
    ],
    "base_large_pasnet": [
        "base_pasnet",
        "large_pasnet"
    ],
    "all_nine_models": MODELS12
}

PROTOCOL12 = {
    "models": MODELS12,
    "ensembles": ENSEMBLES12,
    "ensemble_method": "uniform probability averaging",
    "class_order": LABELS12,
    "datasets": {
        k: len(v) for k, v in DATA12.items()
    },
    "primary_metric": "macro_f1",
    "secondary_metrics": [
        "accuracy", "weighted_f1", "class_f1",
        "nll", "brier", "ece"
    ],
    "bootstrap_replicates": BOOT12,
    "bootstrap_unit": "individual external document",
    "new_training": False,
    "external_calibration_fitted": False,
    "external_threshold_tuning": False,
    "internal_test_tuning": False,
    "interpretation": "external diagnostic evaluation"
}

write_json(
    PROTOCOL12,
    BACKUP12 / "audit" / "external_protocol.json"
)

push12("Freeze external model comparison protocol")

# ==================================================================
# STEP 232 — PREDICTION VALIDATION AND RECOVERY HELPERS
# ==================================================================

section12("STEP 232 — RESUMABLE PREDICTION PIPELINE")

from transformers import AutoTokenizer
from normalizer import normalize as bn_normalize

def probability12(output, n):

    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()

    if isinstance(output, dict):

        if "probabilities" in output:
            output = output["probabilities"]

        elif "probs" in output:
            output = output["probs"]

        elif "logits" in output:
            output = softmax(
                np.asarray(output["logits"]), axis=1
            )

        else:
            raise ValueError(
                "Unknown inference dictionary keys: "
                + str(list(output))
            )

    elif isinstance(output, (tuple, list)):

        # Require an unambiguous N x 3 prediction array.
        candidates = []

        for value in output:

            try:
                arr = np.asarray(value)

                if arr.shape == (n, 3):
                    candidates.append(arr)

            except Exception:
                pass

        assert len(candidates) == 1, (
            "Ambiguous original inference output. "
            "Inspect the recovered function."
        )

        output = candidates[0]

    arr = np.asarray(output, dtype=np.float64)

    assert arr.shape == (n, 3)
    assert np.isfinite(arr).all()

    if (
        (arr >= 0).all()
        and (arr <= 1).all()
        and np.allclose(
            arr.sum(axis=1), 1, atol=1e-5
        )
    ):
        return arr

    return softmax(arr, axis=1)

def verify_npz12(path, frame):

    with np.load(path, allow_pickle=False) as z:

        ids = z["sample_id"].astype(str)
        labels = z["labels"].astype(int)
        probs = z["probabilities"]

        assert np.array_equal(
            ids, frame["sample_id"].to_numpy(str)
        )

        assert np.array_equal(
            labels, frame["label_id"].to_numpy(int)
        )

        assert probs.shape == (len(frame), 3)
        assert np.isfinite(probs).all()
        assert (probs >= 0).all()

        assert np.allclose(
            probs.sum(axis=1), 1, atol=1e-5
        )

        return probs.copy()

def recover_prediction12(path, frame):

    if path.exists():

        probs = verify_npz12(path, frame)
        upload12(path)

        print("Reusing local prediction:", path.name)

        return probs

    candidates = [
        a for a in existing_assets12()
        if a["name"].startswith(path.stem + "_")
        and a["name"].endswith(".npz")
    ]

    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple remote predictions: " + path.name
        )

    if len(candidates) == 1:

        a = candidates[0]

        asset = {
            "url": a["url"],
            "size": a["size"],
            "digest": a.get("digest")
        }

        download11(asset, path)

        return verify_npz12(path, frame)

    return None

# ==================================================================
# STEP 233 — EXACT MODEL REPLAY ON EXTERNAL DATASETS
# ==================================================================

section12("STEP 233 — NINE-MODEL EXTERNAL INFERENCE")

PROBS12 = {
    dataset: {} for dataset in DATA12
}

TIMINGS12 = []

for name in MODELS12:

    print("\nMODEL:", name)

    pending = []

    for dataset, frame in DATA12.items():

        path = PRED12 / f"{dataset}_{name}.npz"

        existing = recover_prediction12(
            path, frame
        )

        if existing is None:
            pending.append(dataset)
        else:
            PROBS12[dataset][name] = existing

    if not pending:
        continue

    model_start = time.time()

    try:

        built = BUILD12(MODEL_ALIAS12[name])

        assert isinstance(built, (tuple, list)), (
            "Unexpected original build_model return type."
        )

        assert len(built) == 2, (
            "Expected original model and tokenizer pair."
        )

        model, tokenizer = built

        model = model.to(DEVICE12)
        model.eval()

        for dataset in pending:

            frame = DATA12[dataset].copy()

            # Preserve the original normalization conventions.
            frame["Bangla"] = frame["text_norm"]
            frame["Label"] = frame["label_id"].map({0: "Negative", 1: "Neutral", 2: "Positive"})
            frame["label"] = frame["label_id"]
            frame["labels"] = frame["label_id"]

            uses_banglabert = name not in [
                "XLM-RoBERTa", "mBERT"
            ]

            if uses_banglabert:

                frame["model_text"] = (
                    frame["text_norm"].map(bn_normalize)
                )

            else:

                frame["model_text"] = frame["text_norm"]

            frame["text"] = frame["model_text"]

            batches = []

            # Limit individual calls to reduce GPU memory
            # pressure and permit progress reporting.

            CHUNK12 = 2048

            for start in range(
                0, len(frame), CHUNK12
            ):

                part = frame.iloc[
                    start:start + CHUNK12
                ].reset_index(drop=True)

                with torch.inference_mode():

                    output = INFER12(
                        MODEL_ALIAS12[name], model, tokenizer, part
                    )

                probabilities = probability12(
                    output, len(part)
                )

                batches.append(probabilities)

                if start % (CHUNK12 * 5) == 0:

                    print(
                        dataset,
                        min(start + len(part), len(frame)),
                        "/", len(frame)
                    )

            probabilities = np.concatenate(
                batches, axis=0
            )

            original = DATA12[dataset]

            path = PRED12 / f"{dataset}_{name}.npz"

            np.savez_compressed(
                path,
                sample_id=original[
                    "sample_id"
                ].to_numpy(str),
                labels=original[
                    "label_id"
                ].to_numpy(int),
                probabilities=probabilities.astype(
                    np.float32
                )
            )

            PROBS12[dataset][name] = verify_npz12(
                path, original
            )

            upload12(path)

            print(
                dataset, name,
                "| Inference completed:",
                len(original)
            )

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as error:

        write_json(
            {
                "model": name,
                "error": str(error),
                "completed": {
                    k: list(v)
                    for k, v in PROBS12.items()
                }
            },
            BACKUP12 / "audit" / "inference_failure.json"
        )

        push12("Preserve external inference failure details")

        raise RuntimeError(
            f"Original inference failed for {name}: "
            f"{error}. Previously completed remote "
            "predictions remain preserved."
        ) from error

    TIMINGS12.append({
        "model": name,
        "runtime_minutes": (
            time.time() - model_start
        ) / 60
    })

    table12(
        pd.DataFrame(TIMINGS12),
        "external_11_inference_timings"
    )

    push12(f"Record external inference completion: {name}")

for dataset in DATA12:
    assert set(PROBS12[dataset]) == set(MODELS12)

# ==================================================================
# STEP 234 — BUILD PRE-SPECIFIED ENSEMBLES
# ==================================================================

section12("STEP 234 — FIXED ENSEMBLE CONSTRUCTION")

for dataset, frame in DATA12.items():

    for name, members in ENSEMBLES12.items():

        probabilities = np.mean(
            [
                PROBS12[dataset][m]
                for m in members
            ],
            axis=0
        )

        PROBS12[dataset][name] = probabilities

        path = PRED12 / f"{dataset}_{name}.npz"

        np.savez_compressed(
            path,
            sample_id=frame["sample_id"].to_numpy(str),
            labels=frame["label_id"].to_numpy(int),
            probabilities=probabilities.astype(
                np.float32
            )
        )

        verify_npz12(path, frame)
        upload12(path)

# ==================================================================
# STEP 235 — COMPREHENSIVE PERFORMANCE METRICS
# ==================================================================

section12("STEP 235 — EXTERNAL PERFORMANCE METRICS")

def ece12(y, probabilities, bins=15):

    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y

    edges = np.linspace(0, 1, bins+1)
    score = 0.0

    for i in range(bins):

        mask = (
            (confidence >= edges[i])
            & (
                (confidence < edges[i+1])
                if i < bins-1
                else (confidence <= edges[i+1])
            )
        )

        if mask.any():

            score += (
                mask.mean()
                * abs(
                    correct[mask].mean()
                    - confidence[mask].mean()
                )
            )

    return float(score)

def metrics12(y, probabilities):

    pred = probabilities.argmax(axis=1)

    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(
            y, pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y, pred,
            average="weighted",
            zero_division=0
        ),
        "nll": log_loss(
            y, probabilities, labels=[0, 1, 2]
        ),
        "brier": np.mean(
            np.sum(
                (
                    probabilities
                    - np.eye(3)[y]
                )**2,
                axis=1
            )
        ),
        "ece": ece12(y, probabilities)
    }

RESULTS12 = []
CLASSES12 = []
CONFUSIONS12 = []

for dataset, frame in DATA12.items():

    y = frame["label_id"].to_numpy(int)

    for name, probabilities in PROBS12[dataset].items():

        values = metrics12(y, probabilities)

        RESULTS12.append({
            "dataset": dataset,
            "model": name,
            "samples": len(y),
            **values
        })

        pred = probabilities.argmax(axis=1)

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y, pred,
                labels=[0, 1, 2],
                zero_division=0
            )
        )

        for i, label in enumerate(LABELS12):

            CLASSES12.append({
                "dataset": dataset,
                "model": name,
                "class": label,
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i]
            })

        cm = confusion_matrix(
            y, pred, labels=[0, 1, 2]
        )

        for i in range(3):
            for j in range(3):

                CONFUSIONS12.append({
                    "dataset": dataset,
                    "model": name,
                    "true": LABELS12[i],
                    "predicted": LABELS12[j],
                    "count": cm[i, j]
                })

RESULT_DF12 = pd.DataFrame(RESULTS12).sort_values(
    ["dataset", "macro_f1"],
    ascending=[True, False]
)

CLASS_DF12 = pd.DataFrame(CLASSES12)
CM_DF12 = pd.DataFrame(CONFUSIONS12)

table12(RESULT_DF12, "external_12_all_results")
table12(CLASS_DF12, "external_13_classwise_results")
table12(CM_DF12, "external_14_confusion_matrices")

print(RESULT_DF12.to_string(index=False))

push12("Preserve complete external performance tables")

# ==================================================================
# STEP 236 — PAIRED STATISTICAL COMPARISONS
# ==================================================================

section12("STEP 236 — PAIRED BOOTSTRAP STATISTICS")

# Multinomial resampling of joint confusion outcomes.
# Equivalent to paired document-level bootstrap
# for metrics depending only on the confusion counts.

def f1_from_cm12(cm):

    tp = np.diagonal(cm, axis1=-2, axis2=-1)

    denom = (
        cm.sum(axis=-1) + cm.sum(axis=-2)
    )

    values = np.divide(
        2 * tp,
        denom,
        out=np.zeros_like(tp, dtype=float),
        where=denom > 0
    )

    return values.mean(axis=-1)

def paired12(y, candidate, reference, seed=42):

    a = candidate.argmax(axis=1)
    b = reference.argmax(axis=1)

    codes = y * 9 + a * 3 + b

    distribution = np.bincount(
        codes, minlength=27
    )

    rng = np.random.default_rng(seed)

    boot = rng.multinomial(
        len(y),
        distribution / len(y),
        size=BOOT12
    ).reshape(BOOT12, 3, 3, 3)

    cm_a = boot.sum(axis=3)
    cm_b = boot.sum(axis=2)

    delta = (
        f1_from_cm12(cm_a)
        - f1_from_cm12(cm_b)
    )

    gained = int(((a == y) & (b != y)).sum())
    lost = int(((a != y) & (b == y)).sum())

    p = (
        binomtest(
            min(gained, lost),
            gained + lost,
            p=.5
        ).pvalue
        if gained + lost else 1.0
    )

    return {
        "delta_macro_f1": (
            f1_score(y, a, average="macro")
            - f1_score(y, b, average="macro")
        ),
        "ci95_low": np.quantile(delta, .025),
        "ci95_high": np.quantile(delta, .975),
        "gained_correct": gained,
        "lost_correct": lost,
        "mcnemar_p": p
    }

REFERENCES12 = [
    "BanglaBERT",
    "base_standard",
    "large_standard",
    "base_large_standard",
    "three_member_standard",
    "base_large_pasnet"
]

PAIRED12 = []

for dataset, frame in DATA12.items():

    y = frame["label_id"].to_numpy(int)

    for reference in REFERENCES12:

        PAIRED12.append({
            "dataset": dataset,
            "candidate": "proposed_prototype_family",
            "reference": reference,
            **paired12(
                y,
                PROBS12[dataset][
                    "proposed_prototype_family"
                ],
                PROBS12[dataset][reference]
            )
        })

PAIRED_DF12 = pd.DataFrame(PAIRED12)

# Holm correction across the 12 prespecified comparisons.

pvalues = PAIRED_DF12["mcnemar_p"].to_numpy()

order = np.argsort(pvalues)

adjusted = np.empty(len(pvalues))

running = 0.0

for rank, idx in enumerate(order):

    value = (
        len(pvalues) - rank
    ) * pvalues[idx]

    running = max(running, value)
    adjusted[idx] = min(1.0, running)

PAIRED_DF12["holm_p"] = adjusted

table12(
    PAIRED_DF12,
    "external_15_paired_statistics"
)

print(PAIRED_DF12.to_string(index=False))

# ==================================================================
# STEP 237 — DOMAIN AND LINGUISTIC FAILURE ANALYSIS
# ==================================================================

section12("STEP 237 — SUBGROUP AND DOMAIN ANALYSIS")

SUBGROUP12 = []
DOMAIN12 = []
ERRORS12 = []

SELECTED12 = [
    "BanglaBERT",
    "base_large_standard",
    "three_member_standard",
    "proposed_prototype_family"
]

for dataset, frame in DATA12.items():

    y = frame["label_id"].to_numpy(int)

    neg = frame[resolve_column12(frame, ["has_negation", "contains_negation", "negation"])].astype(bool).to_numpy()
    con = frame[resolve_column12(frame, ["has_contrast", "contains_contrast", "contrast"])].astype(bool).to_numpy()

    length = frame["word_count"].to_numpy()

    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "negation": neg,
        "contrast": con,
        "both": neg & con,
        "any_operator": neg | con,
        "no_operator": ~(neg | con),
        "short_1_to_10_words": length <= 10,
        "medium_11_to_20_words": (
            (length > 10) & (length <= 20)
        ),
        "long_over_20_words": length > 20
    }

    for name in SELECTED12:

        probs = PROBS12[dataset][name]
        pred = probs.argmax(axis=1)

        for subgroup, mask in masks.items():

            if mask.sum() < 20:
                continue

            SUBGROUP12.append({
                "dataset": dataset,
                "model": name,
                "subgroup": subgroup,
                "samples": int(mask.sum()),
                "classes_present": len(
                    np.unique(y[mask])
                ),
                **metrics12(y[mask], probs[mask])
            })

        for domain in frame["domain"].dropna().unique():

            mask = (
                frame["domain"].to_numpy() == domain
            )

            if mask.sum() < 30:
                continue

            DOMAIN12.append({
                "dataset": dataset,
                "model": name,
                "domain": domain,
                "samples": int(mask.sum()),
                "classes_present": len(
                    np.unique(y[mask])
                ),
                **metrics12(y[mask], probs[mask])
            })

        wrong = pred != y

        neutral_error = wrong & (
            (y == 1) | (pred == 1)
        )

        ERRORS12.append({
            "dataset": dataset,
            "model": name,
            "total_errors": int(wrong.sum()),
            "neutral_related_errors": int(
                neutral_error.sum()
            ),
            "neutral_related_error_fraction": (
                neutral_error.sum()
                / max(wrong.sum(), 1)
            ),
            "high_confidence_errors": int(
                (
                    wrong & (probs.max(axis=1) >= .90)
                ).sum()
            )
        })

SUB_DF12 = pd.DataFrame(SUBGROUP12)
DOMAIN_DF12 = pd.DataFrame(DOMAIN12)
ERROR_DF12 = pd.DataFrame(ERRORS12)

table12(SUB_DF12, "external_16_subgroup_results")
table12(DOMAIN_DF12, "external_17_domain_results")
table12(ERROR_DF12, "external_18_error_analysis")

print(ERROR_DF12.to_string(index=False))

# Also preserve SentNoB's original published split results.

SENT_SPLITS12 = []

frame = DATA12["SentNoB"]
y = frame["label_id"].to_numpy(int)

for split in frame["original_split"].unique():

    mask = (
        frame["original_split"].to_numpy() == split
    )

    for name, probs in PROBS12["SentNoB"].items():

        SENT_SPLITS12.append({
            "original_split": split,
            "model": name,
            "samples": int(mask.sum()),
            **metrics12(y[mask], probs[mask])
        })

table12(
    pd.DataFrame(SENT_SPLITS12),
    "external_19_sentnob_original_splits"
)

push12("Preserve domain and linguistic error analysis")

# ==================================================================
# STEP 238 — MODEL COMPLEMENTARITY AND ORACLE ANALYSIS
# ==================================================================

section12("STEP 238 — MODEL COMPLEMENTARITY")

ORACLE12 = []
AGREEMENT12 = []

for dataset, frame in DATA12.items():

    y = frame["label_id"].to_numpy(int)

    predictions = np.stack([
        PROBS12[dataset][name].argmax(axis=1)
        for name in MODELS12
    ])

    correct = predictions == y[None, :]

    ORACLE12.append({
        "dataset": dataset,
        "samples": len(y),
        "models": len(MODELS12),
        "all_models_correct": int(
            correct.all(axis=0).sum()
        ),
        "all_models_wrong": int(
            (~correct.any(axis=0)).sum()
        ),
        "oracle_accuracy": float(
            correct.any(axis=0).mean()
        ),
        "mixed_correctness": int(
            (
                correct.any(axis=0)
                & ~correct.all(axis=0)
            ).sum()
        )
    })

    for i, a in enumerate(MODELS12):

        for j, b in enumerate(MODELS12):

            AGREEMENT12.append({
                "dataset": dataset,
                "model_a": a,
                "model_b": b,
                "agreement": float(
                    (
                        predictions[i] == predictions[j]
                    ).mean()
                )
            })

ORACLE_DF12 = pd.DataFrame(ORACLE12)
AGREE_DF12 = pd.DataFrame(AGREEMENT12)

table12(ORACLE_DF12, "external_20_oracle_analysis")
table12(AGREE_DF12, "external_21_model_agreement")

print(ORACLE_DF12.to_string(index=False))

# ==================================================================
# STEP 239 — PUBLICATION FIGURES
# ==================================================================

section12("STEP 239 — PUBLICATION FIGURES")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150
})

FIGS12 = []

def figure12(fig, name, caption):

    for ext in ["png", "pdf", "svg"]:

        path = (
            BACKUP12 / "figures" / ext
            / f"{name}.{ext}"
        )

        fig.savefig(
            path,
            dpi=600 if ext == "png" else 150,
            bbox_inches="tight"
        )

        FIGS12.append({
            "figure": name,
            "format": ext,
            "path": str(
                path.relative_to(REPO11)
            ),
            "sha256": sha256(path),
            "bytes": path.stat().st_size
        })

    (
        BACKUP12 / "captions" / f"{name}.txt"
    ).write_text(caption, encoding="utf-8")

    plt.show()
    plt.close(fig)

    push12(f"Preserve external figure: {name}")

# 1–2. External model rankings

for dataset in DATA12:

    part = RESULT_DF12[
        RESULT_DF12["dataset"] == dataset
    ].sort_values("macro_f1")

    fig, ax = plt.subplots(figsize=(9, 7))

    bars = ax.barh(
        part["model"],
        part["macro_f1"]
    )

    for bar, model in zip(bars, part["model"]):

        if model == "proposed_prototype_family":
            bar.set_hatch("///")

    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0, 1)
    ax.set_title(f"External sentiment classification: {dataset}")

    figure12(
        fig,
        f"stage12a_ranking_{dataset}",
        "External macro-F1 of fixed individual "
        "models and prespecified ensembles."
    )

# 3. Cross-domain comparison

fig, ax = plt.subplots(figsize=(13, 6))

pivot = RESULT_DF12.pivot(
    index="model",
    columns="dataset",
    values="macro_f1"
)

pivot.plot.bar(ax=ax)

ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1)
ax.set_xlabel("")
ax.legend(frameon=False)
ax.set_title("Cross-dataset generalization")
ax.tick_params(axis="x", rotation=65)

figure12(
    fig,
    "stage12a_cross_domain_comparison",
    "External macro-F1 across both audited cohorts."
)

# 4–5. Paired confidence intervals

for dataset in DATA12:

    part = PAIRED_DF12[
        PAIRED_DF12["dataset"] == dataset
    ].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = part["delta_macro_f1"].to_numpy()

    ax.errorbar(
        x,
        np.arange(len(part)),
        xerr=[
            x - part["ci95_low"].to_numpy(),
            part["ci95_high"].to_numpy() - x
        ],
        fmt="o",
        capsize=4
    )

    ax.axvline(0, linestyle="--")
    ax.set_yticks(
        np.arange(len(part)),
        part["reference"]
    )

    ax.set_xlabel(
        "Macro-F1 difference: proposed minus reference"
    )

    ax.set_title(f"Paired comparison: {dataset}")

    figure12(
        fig,
        f"stage12a_paired_ci_{dataset}",
        "Paired document-level bootstrap 95% "
        "confidence intervals, 5,000 replicates."
    )

# 6–7. Classwise performance

for dataset in DATA12:

    part = CLASS_DF12[
        (CLASS_DF12["dataset"] == dataset)
        & CLASS_DF12["model"].isin(SELECTED12)
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    part.pivot(
        index="class",
        columns="model",
        values="f1"
    ).reindex(LABELS12).plot.bar(
        ax=ax, rot=0
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Classwise F1")
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Sentiment-class reliability: {dataset}")

    figure12(
        fig,
        f"stage12a_classwise_{dataset}",
        "Class-specific F1 for selected transformer "
        "and ensemble models."
    )

# 8–9. Linguistic subgroup comparisons

for dataset in DATA12:

    part = SUB_DF12[
        SUB_DF12["dataset"] == dataset
    ]

    fig, ax = plt.subplots(figsize=(13, 6))

    part.pivot(
        index="subgroup",
        columns="model",
        values="macro_f1"
    ).plot.bar(ax=ax)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40)
    ax.legend(frameon=False, fontsize=8)

    figure12(
        fig,
        f"stage12a_linguistic_subgroups_{dataset}",
        "Descriptive linguistic subgroup comparisons. "
        "Subgroups may overlap and differ in class balance."
    )

# 10. Calibration comparison

fig, ax = plt.subplots(figsize=(13, 6))

RESULT_DF12.pivot(
    index="model",
    columns="dataset",
    values="ece"
).plot.bar(ax=ax)

ax.set_ylabel("Expected calibration error")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=65)
ax.set_title("External confidence calibration")

figure12(
    fig,
    "stage12a_external_calibration",
    "Unrecalibrated expected calibration error "
    "using fifteen equal-width bins."
)

# 11–12. Proposed ensemble confusion matrices

for dataset in DATA12:

    y = DATA12[dataset]["label_id"].to_numpy(int)

    pred = PROBS12[dataset][
        "proposed_prototype_family"
    ].argmax(axis=1)

    cm = confusion_matrix(
        y, pred,
        labels=[0, 1, 2],
        normalize="true"
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, vmin=0, vmax=1)

    ax.set_xticks(range(3), LABELS12)
    ax.set_yticks(range(3), LABELS12)

    for i in range(3):
        for j in range(3):

            ax.text(
                j, i, f"{cm[i,j]:.2f}",
                ha="center", va="center"
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Proposed ensemble: {dataset}")

    fig.colorbar(im, ax=ax)

    figure12(
        fig,
        f"stage12a_confusion_{dataset}",
        "Row-normalized external confusion matrix "
        "of the proposed ensemble."
    )

# 13–14. Inter-model agreement

for dataset in DATA12:

    part = AGREE_DF12[
        AGREE_DF12["dataset"] == dataset
    ]

    mat = part.pivot(
        index="model_a",
        columns="model_b",
        values="agreement"
    ).reindex(
        index=MODELS12,
        columns=MODELS12
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(
        mat.to_numpy(),
        vmin=0,
        vmax=1
    )

    ax.set_xticks(
        range(len(MODELS12)),
        MODELS12,
        rotation=60,
        ha="right"
    )

    ax.set_yticks(
        range(len(MODELS12)),
        MODELS12
    )

    for i in range(len(MODELS12)):
        for j in range(len(MODELS12)):

            ax.text(
                j, i,
                f"{mat.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

    fig.colorbar(im, ax=ax)

    figure12(
        fig,
        f"stage12a_model_agreement_{dataset}",
        "Pairwise classification agreement of "
        "the nine individual models."
    )

# 15. Domain-specific performance

if not DOMAIN_DF12.empty:

    part = DOMAIN_DF12[
        DOMAIN_DF12["dataset"] == "SentiGOLD"
    ]

    if not part.empty:

        fig, ax = plt.subplots(figsize=(13, 6))

        part.pivot(
            index="domain",
            columns="model",
            values="macro_f1"
        ).plot.bar(ax=ax)

        ax.set_ylim(0, 1)
        ax.set_ylabel("Macro-F1")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=55)

        figure12(
            fig,
            "stage12a_sentigold_domains",
            "Domain-specific external performance "
            "on SentiGOLD."
        )

table12(
    pd.DataFrame(FIGS12),
    "external_22_figure_manifest"
)

# ==================================================================
# STEP 240 — SCIENTIFIC DECISION AND RESULT SYNTHESIS
# ==================================================================

section12("STEP 240 — FINAL SCIENTIFIC ASSESSMENT")

DECISIONS12 = []

for dataset in DATA12:

    part = RESULT_DF12[
        RESULT_DF12["dataset"] == dataset
    ].sort_values(
        "macro_f1", ascending=False
    )

    proposed = float(
        part.loc[
            part["model"] == "proposed_prototype_family",
            "macro_f1"
        ].iloc[0]
    )

    standard = float(
        part.loc[
            part["model"] == "three_member_standard",
            "macro_f1"
        ].iloc[0]
    )

    paired = PAIRED_DF12[
        (PAIRED_DF12["dataset"] == dataset)
        & (
            PAIRED_DF12["reference"]
            == "three_member_standard"
        )
    ].iloc[0]

    DECISIONS12.append({
        "dataset": dataset,
        "best_evaluated_system": part.iloc[0]["model"],
        "best_macro_f1": float(
            part.iloc[0]["macro_f1"]
        ),
        "proposed_macro_f1": proposed,
        "matched_standard_ensemble_macro_f1": standard,
        "difference_percentage_points": (
            proposed - standard
        ) * 100,
        "ci95_low": float(paired["ci95_low"]),
        "ci95_high": float(paired["ci95_high"]),
        "holm_p": float(paired["holm_p"]),
        "positive_macro_f1_ci": bool(
            paired["ci95_low"] > 0
        )
    })

DECISION_DF12 = pd.DataFrame(DECISIONS12)

table12(
    DECISION_DF12,
    "external_23_final_decisions"
)

print(DECISION_DF12.to_string(index=False))

# ==================================================================
# STEP 241 — PRESERVE ALL RESEARCH ARTIFACTS
# ==================================================================

section12("STEP 241 — PERMANENT GITHUB BACKUP")

source_saved12 = False

try:

    history = get_ipython().user_ns.get("In", [])

    source = next(
        x for x in reversed(history)
        if (
            "STAGE 11B–12A — CONSOLIDATED" in x
            and len(x) > 5000
        )
    )

    assert TOKEN11 not in source

    (
        BACKUP12 / "stage11b12a_complete.py"
    ).write_text(source, encoding="utf-8")

    source_saved12 = True

except Exception as error:

    print("Source saving issue:", type(error).__name__)

write_json(
    REMOTE12,
    BACKUP12 / "audit" / "remote_assets.json"
)

ARCHIVE12 = ROOT12 / "stage12a_research_archive.zip"

with zipfile.ZipFile(
    ARCHIVE12, "w", zipfile.ZIP_DEFLATED
) as z:

    for p in BACKUP12.rglob("*"):

        if p.is_file():

            z.write(
                p,
                f"research/{p.relative_to(BACKUP12)}"
            )

    for p in PRED12.glob("*.npz"):

        z.write(
            p, f"predictions/{p.name}"
        )

with zipfile.ZipFile(ARCHIVE12) as z:
    assert z.testzip() is None

upload12(ARCHIVE12)

MANIFEST12 = ROOT12 / "stage12a_remote_manifest.json"

write_json(REMOTE12, MANIFEST12)

upload12(MANIFEST12)

# ==================================================================
# STEP 242 — FINAL CONSOLIDATED REPORT
# ==================================================================

section12("STEP 242 — FINAL EXTERNAL RESEARCH REPORT")

SUMMARY12 = {
    "stage": "11B–12A",
    "external_datasets": list(DATA12),
    "external_samples": sum(
        len(v) for v in DATA12.values()
    ),
    "individual_models": len(MODELS12),
    "ensembles": len(ENSEMBLES12),
    "model_dataset_evaluations": len(RESULT_DF12),
    "bootstrap_replicates": BOOT12,
    "external_inference_completed": True,
    "new_training": False,
    "internal_test_tuning": False,
    "external_threshold_tuning": False,
    "architectural_novelty_verified": False,
    "new_publication_figures": len(FIGS12) // 3,
    "new_figure_files": len(FIGS12),
    "remote_verified_assets": len(REMOTE12),
    "manual_saving_required": False,
    "google_drive_used": False,
    "source_code_saved": source_saved12,
    "dataset_decisions": DECISIONS12,
    "runtime_minutes": round(
        (time.time() - START12) / 60, 2
    )
}

write_json(
    SUMMARY12,
    BACKUP12 / "stage12a_summary.json"
)

write_json(
    REMOTE12,
    BACKUP12 / "audit" / "remote_assets.json"
)

FINAL12 = push12(
    "Complete external evaluation and statistical analysis"
)

print("\nEXTERNAL MODEL PERFORMANCE")
print(RESULT_DF12.to_string(index=False))

print("\nPAIRED STATISTICS")
print(PAIRED_DF12.to_string(index=False))

print("\nLINGUISTIC FAILURE ANALYSIS")
print(ERROR_DF12.to_string(index=False))

print("\nORACLE ANALYSIS")
print(ORACLE_DF12.to_string(index=False))

print("\nFINAL DECISIONS")
print(DECISION_DF12.to_string(index=False))

print("\nRESEARCH SUMMARY")
print(json.dumps(SUMMARY12, indent=2, default=str))

print("\nGITHUB IMAGE INVENTORY")

print({
    ext: len(list(REPO11.rglob(f"*.{ext}")))
    for ext in ["png", "pdf", "svg"]
})

print("\nFinal public commit:", FINAL12)
print("Private release:", REL12["html_url"])
print("GitHub synchronized: True")
print("Manual saving required: NO")
print("Google Drive used: NO")
print("New model training: NO")

section12("STAGE 11B–12A COMPLETE")
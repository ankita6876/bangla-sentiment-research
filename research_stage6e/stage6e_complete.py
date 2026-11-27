
# =====================================================================
# STAGE 6E — ACCELERATED CONFIRMATORY EXPERIMENTS
# STEPS 90–97
# GitHub-first recovery
# Third seed + three matched-capacity controls
# No repeated completed experiments
# Automatic private GitHub backup
# =====================================================================

import os
import re
import gc
import json
import time
import hashlib
import requests
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

START6E = time.time()

WORK6E = Path("/kaggle/working")

ROOT6E = WORK6E / "sentiment_research_stage6e"
ROOT6E.mkdir(parents=True, exist_ok=True)

OWNER6E = "ankita6876"
PUBLIC6E = "bangla-sentiment-research"

COMMIT6D = "fe8be78c4c00a2ed98d88ad6c0f7479c962091e5"

TAG6D = "stage6d-amsc-refinement-v1"
TAG6E = "stage6e-amsc-confirmation-v1"

print("=" * 90)
print("STEP 90 — VERIFIED SOURCE RECOVERY")
print("=" * 90)

url6e = (
    f"https://raw.githubusercontent.com/"
    f"{OWNER6E}/{PUBLIC6E}/{COMMIT6D}/"
    "research_stage6d/stage6d_complete.py"
)

response6e = requests.get(url6e, timeout=120)
response6e.raise_for_status()

SOURCE_D = response6e.text

assert "class AMSCNetV2" in SOURCE_D
assert "def objective_d" in SOURCE_D
assert "def make_model_d" in SOURCE_D

(ROOT6E / "original_stage6d.py").write_text(
    SOURCE_D, encoding="utf-8"
)

print("Stage 6D source recovery: PASS")
print("Verified commit:", COMMIT6D)

# Recover Stage 6C setup, exact dataset membership,
# tokenizer and original architectural components.
# Do not execute Stage 6D training again.

prefix_marker = (
    'section("STEP 81 — FRESH INITIALIZATION PROTOCOL")'
)

assert prefix_marker in SOURCE_D

prefix6e = SOURCE_D.split(prefix_marker, 1)[0]

exec(
    compile(prefix6e, "stage6d_recovery", "exec"),
    globals()
)

import shutil
import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import softmax
from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

section("STEP 91 — LOCK CONFIRMATION PROTOCOL")

S6D = ROOT6E

TABLES = ROOT6E / "tables"
CHECKPOINTS = ROOT6E / "checkpoints"
PRED = ROOT6E / "private_predictions"
AUDIT = ROOT6E / "audit"

BACKUP = REPO / "research_stage6e"

for folder in [
    TABLES, CHECKPOINTS, PRED,
    AUDIT, BACKUP
]:
    folder.mkdir(parents=True, exist_ok=True)

PRETRAINED_NAME = "csebuetnlp/banglabert"

SEEDS_D = [42, 123, 2026]

MODES_D = [
    "standard",
    "fixed_fusion",
    "matched_control"
]

EPOCHS_D = 4
TRAIN_BATCH_D = 32
EVAL_BATCH_D = 64

ENCODER_LR_D = 2e-5
HEAD_LR_D = 2e-4

AUX_WEIGHT_D = 0.10
WARMUP_D = 0.10
WEIGHT_DECAY_D = 0.01

BOOTSTRAPS_D = 5000

STAGE_TAG = TAG6E

RUNS_E = [
    ("standard", 2026),
    ("fixed_fusion", 2026),
    ("matched_control", 42),
    ("matched_control", 123),
    ("matched_control", 2026)
]

CONFIG_D = {
    "stage": "6e",
    "parent_commit": COMMIT6D,
    "parent_release": TAG6D,
    "pretrained_model": PRETRAINED_NAME,
    "dataset_split_seed": 42,
    "training_seeds": SEEDS_D,
    "architectures": MODES_D,
    "new_runs": RUNS_E,
    "epochs": EPOCHS_D,
    "batch_size": TRAIN_BATCH_D,
    "max_length": MAX_LENGTH,
    "encoder_lr": ENCODER_LR_D,
    "new_head_lr": HEAD_LR_D,
    "auxiliary_weight": AUX_WEIGHT_D,
    "warmup": WARMUP_D,
    "weight_decay": WEIGHT_DECAY_D,
    "selection": "validation_macro_f1",
    "initialization": "fresh_pretrained",
    "primary_candidate": "fixed_fusion",
    "test_evaluated": False,
    "novelty_verified": False
}

save_json(
    CONFIG_D,
    AUDIT / "experiment_config.json"
)

shutil.copy2(
    AUDIT / "experiment_config.json",
    BACKUP / "experiment_config.json"
)

def push_d(message):

    git("add", "research_stage6e")

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

# Preserve the complete new executable source.

history6e = (
    get_ipython()
    .history_manager.input_hist_raw
)

sources6e = [
    c for c in history6e
    if "STAGE 6E — ACCELERATED CONFIRMATORY EXPERIMENTS" in c
    and len(c) > 5000
]

assert sources6e

cell6e = sources6e[-1]

assert not re.search(
    r"(?:ghp_|github_pat_|gho_|ghs_)"
    r"[A-Za-z0-9_]{20,}",
    cell6e
)

(AUDIT / "stage6e_complete.py").write_text(
    cell6e, encoding="utf-8"
)

shutil.copy2(
    AUDIT / "stage6e_complete.py",
    BACKUP / "stage6e_complete.py"
)

shutil.copy2(
    ROOT6E / "original_stage6d.py",
    BACKUP / "original_stage6d.py"
)

push_d("Lock Stage 6E confirmation protocol")

# =====================================================================
# STEP 92 — PRIVATE STORAGE AND PREVIOUS RUN RECOVERY
# =====================================================================

section("STEP 92 — RECOVER COMPLETED STAGE 6D RUNS")

storage_start = (
    'section("STEP 74B — PRIVATE ARTIFACT STORAGE")'
)

storage_end = (
    'section("STEP 75 — TRAINING UTILITIES")'
)

assert storage_start in source6c
assert storage_end in source6c

storage_source = (
    source6c.split(storage_start, 1)[1]
    .split(storage_end, 1)[0]
)

# Initialize the new private release.

STAGE_TAG = TAG6E

exec(storage_source, globals())

assert STAGE_TAG == TAG6E

metric_source = (
    source6c.split("def metrics(y, logits):", 1)[1]
    .split("def objective(output, labels):", 1)[0]
)

exec(
    "def metrics(y, logits):" + metric_source,
    globals()
)

# Recover the exact Stage 6D architecture and utilities.

architecture_source = (
    SOURCE_D.split("class AMSCNetV2", 1)[1]
    .split(
        'section("STEP 84 — ARCHITECTURE SMOKE TEST")',
        1
    )[0]
)

exec(
    "class AMSCNetV2" + architecture_source,
    globals()
)

RESULTS_D = []
HISTORY_D = []
LOGITS_D = {}
WEIGHTS_D = {}
REMOTE_D = []

sample_ids_d = (
    DATA["val"]["sample_id"]
    .astype(str)
    .to_numpy(dtype=str)
)

private_api6e = (
    f"https://api.github.com/repos/"
    f"{OWNER}/{PRIVATE_REPO}"
)

old_release_response = SESSION.get(
    private_api6e + "/releases/tags/" + TAG6D,
    timeout=60
)

old_release_response.raise_for_status()

old_release = old_release_response.json()

old_assets = {}

page = 1

while True:

    rr = SESSION.get(
        private_api6e
        + f"/releases/{old_release['id']}/assets",
        params={"per_page": 100, "page": page},
        timeout=60
    )

    rr.raise_for_status()

    items = rr.json()

    for item in items:
        old_assets[item["name"]] = item

    if len(items) < 100:
        break

    page += 1

RECOVERED_E = []

for seed in [42, 123]:

    for mode in ["standard", "fixed_fusion"]:

        run_name = f"{mode}_seed{seed}"

        run_dir = CHECKPOINTS / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = run_dir / "completed.json"

        pred = PRED / f"{run_name}_val.npz"

        metadata_asset = (
            f"{run_name}_completed.json"
        )

        prediction_asset = (
            f"{run_name}_val.npz"
        )

        checkpoint_asset = (
            f"{run_name}_model.pt"
        )

        for name in [
            metadata_asset,
            prediction_asset,
            checkpoint_asset
        ]:
            assert name in old_assets, name

        download_asset(
            old_assets[metadata_asset],
            meta
        )

        completed = json.loads(
            meta.read_text()
        )

        assert completed["result"]["seed"] == seed
        assert completed["result"]["model"] == mode

        download_asset(
            old_assets[prediction_asset],
            pred,
            completed["prediction_sha256"]
        )

        verify_remote_asset(
            old_assets[checkpoint_asset],
            completed["checkpoint_sha256"]
        )

        with np.load(
            pred, allow_pickle=False
        ) as saved:

            assert np.array_equal(
                saved["sample_id"],
                sample_ids_d
            )

            assert np.array_equal(
                saved["labels"],
                Y_VAL
            )

            LOGITS_D[(mode, seed)] = (
                saved["logits"].copy()
            )

            WEIGHTS_D[(mode, seed)] = (
                saved["weights"].copy()
            )

        calculated = metrics(
            Y_VAL,
            LOGITS_D[(mode, seed)]
        )["macro_f1"]

        assert abs(
            calculated
            - completed["result"]["macro_f1"]
        ) < 1e-7

        RESULTS_D.append(
            completed["result"]
        )

        HISTORY_D.extend(
            completed["history"]
        )

        RECOVERED_E.append({
            "model": mode,
            "seed": seed,
            "release": TAG6D,
            "checkpoint_asset": checkpoint_asset,
            "checkpoint_sha256":
                completed["checkpoint_sha256"],
            "prediction_sha256":
                completed["prediction_sha256"]
        })

        print(
            run_name,
            "| F1:", round(calculated, 6),
            "| Recovery: PASS"
        )

assert len(RESULTS_D) == 4

# =====================================================================
# STEP 93 — MATCHED-CAPACITY CONTROL
# =====================================================================

section("STEP 93 — PARAMETER-MATCHED CONTROL")

# The control receives the same contextual token features.
# It uses global pooling and generic MLP branches rather
# than the proposed phrase and directional operations.
# It retains the same fusion weights and auxiliary losses.

def count_parameters_e(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )

reference_e = make_model_d(
    "fixed_fusion", 42
)

target_parameters_e = (
    count_parameters_e(reference_e)
)

base_parameters_e = (
    count_parameters_e(reference_e.encoder)
    + count_parameters_e(
        reference_e.base_classifier
    )
)

target_extra_e = (
    target_parameters_e - base_parameters_e
)

del reference_e

class MatchedControlE(nn.Module):

    def __init__(self, base_model, width):

        super().__init__()

        self.encoder = base_model.electra
        self.base_classifier = base_model.classifier

        hidden = base_model.config.hidden_size

        self.project = nn.Sequential(
            nn.Linear(hidden, DIM),
            nn.LayerNorm(DIM),
            nn.GELU()
        )

        self.global_repr = nn.Sequential(
            nn.Linear(2 * DIM, DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

        self.global_delta = nn.Linear(DIM, 3)

        self.generic_branches = nn.Sequential(
            nn.Linear(2 * DIM, width),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(width, 2 * DIM)
        )

        self.phrase_head = nn.Linear(DIM, 3)

        self.directional_head = nn.Linear(DIM, 3)

        nn.init.zeros_(
            self.global_delta.weight
        )

        nn.init.zeros_(
            self.global_delta.bias
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        valid_mask
    ):

        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).last_hidden_state

        base_logits = self.base_classifier(
            hidden
        )

        x = self.project(hidden)

        pooled = masked_mean(
            x, valid_mask
        )

        global_input = torch.cat(
            [x[:, 0], pooled],
            dim=-1
        )

        g = self.global_repr(
            global_input
        )

        p, d = self.generic_branches(
            global_input
        ).chunk(2, dim=-1)

        zg = (
            base_logits
            + self.global_delta(g)
        )

        zp = self.phrase_head(p)
        zd = self.directional_head(d)

        branches = torch.stack(
            [zg, zp, zd],
            dim=1
        )

        weights = torch.tensor(
            [0.70, 0.15, 0.15],
            device=zg.device,
            dtype=zg.dtype
        ).expand(len(zg), -1)

        logits = (
            branches
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        return {
            "logits": logits,
            "branches": branches,
            "weights": weights
        }

# Calculate the MLP width without repeatedly loading
# the pretrained encoder.

from transformers import AutoConfig

config_e = AutoConfig.from_pretrained(
    PRETRAINED_NAME
)

with torch.device("meta"):

    dummy_base_e = (
        AutoModelForSequenceClassification
        .from_config(config_e)
    )

    # Match the actual three-class classifier.

    dummy_base_e.config.num_labels = 3

# Determine parameter count using only the new layers.

hidden_e = config_e.hidden_size

constant_e = (
    hidden_e * DIM + DIM
    + 2 * DIM
    + 2 * DIM * DIM + DIM
    + DIM * 3 + 3
    + 2 * (DIM * 3 + 3)
    + 2 * DIM
)

per_width_e = 4 * DIM + 1

CONTROL_WIDTH_E = max(
    1,
    round(
        (target_extra_e - constant_e)
        / per_width_e
    )
)

del dummy_base_e

original_make_model_d = make_model_d

def make_model_d(mode, seed):

    if mode != "matched_control":

        return original_make_model_d(
            mode, seed
        )

    seed_all(seed)

    base = (
        AutoModelForSequenceClassification
        .from_pretrained(
            PRETRAINED_NAME,
            num_labels=3
        )
    )

    return MatchedControlE(
        base,
        CONTROL_WIDTH_E
    )

control_e = make_model_d(
    "matched_control", 42
)

control_parameters_e = (
    count_parameters_e(control_e)
)

control_extra_e = (
    control_parameters_e - base_parameters_e
)

mismatch_e = abs(
    control_extra_e - target_extra_e
) / target_extra_e

assert mismatch_e < 0.005, mismatch_e

print(
    "Proposed extra parameters:",
    target_extra_e
)

print(
    "Control extra parameters:",
    control_extra_e
)

print(
    "Parameter mismatch:",
    round(100 * mismatch_e, 5),
    "%"
)

print(
    "Matched control width:",
    CONTROL_WIDTH_E
)

CONFIG_D.update({
    "control_width": CONTROL_WIDTH_E,
    "proposed_extra_parameters":
        target_extra_e,
    "control_extra_parameters":
        control_extra_e,
    "relative_parameter_mismatch":
        mismatch_e
})

save_json(
    CONFIG_D,
    AUDIT / "experiment_config.json"
)

shutil.copy2(
    AUDIT / "experiment_config.json",
    BACKUP / "experiment_config.json"
)

# Smoke test.

control_e = control_e.to(DEVICE)

_, val_loader_e = loaders(42)

batch_e = {
    k: v.to(DEVICE)
    for k, v in next(
        iter(val_loader_e)
    ).items()
}

control_e.train()

out_e = control_e(
    batch_e["input_ids"],
    batch_e["attention_mask"],
    batch_e["valid_mask"]
)

assert out_e["logits"].shape == (
    len(batch_e["labels"]), 3
)

loss_e = objective_d(
    out_e, batch_e["labels"]
)

assert torch.isfinite(loss_e)

loss_e.backward()

for module in [
    control_e.encoder,
    control_e.generic_branches
]:

    grads = [
        p.grad
        for p in module.parameters()
        if p.grad is not None
    ]

    assert grads

    assert all(
        torch.isfinite(g).all()
        for g in grads
    )

    assert any(
        g.abs().sum().item() > 0
        for g in grads
    )

print("Matched control gradient flow: PASS")

del control_e, batch_e, out_e, loss_e
del grads, val_loader_e

gc.collect()
torch.cuda.empty_cache()

push_d("Preserve matched-capacity control protocol")

# =====================================================================
# STEP 94 — EXECUTE ONLY FIVE NEW TRAINING RUNS
# =====================================================================

section("STEP 94 — FIVE CONFIRMATORY TRAINING RUNS")

# Reuse the exact verified Stage 6D training loop.
# No change to optimizer, loss, scheduler, selection
# criterion, precision or training budget.

loop_start = "for seed in SEEDS_D:\n\n    for mode in MODES_D:"

loop_end = (
    'section("STEP 86 — MULTI-SEED COMPARISON")'
)

assert loop_start in SOURCE_D
assert loop_end in SOURCE_D

training_source_e = (
    SOURCE_D.split(loop_start, 1)[1]
    .split(loop_end, 1)[0]
)

# Preserve original indentation by retaining
# the original nested loop and skipping completed runs.

new_loop_e = (
    "for seed in SEEDS_D:\n"
    "    for mode in MODES_D:\n"
    "        if (mode, seed) not in RUNS_E:\n"
    "            continue\n"
    + training_source_e
)

# Compile before executing any new training.

compiled_e = compile(
    new_loop_e,
    "stage6e_verified_training_loop",
    "exec"
)

(AUDIT / "executed_training_loop.py").write_text(
    new_loop_e,
    encoding="utf-8"
)

shutil.copy2(
    AUDIT / "executed_training_loop.py",
    BACKUP / "executed_training_loop.py"
)

push_d("Preserve executable confirmation training loop")

exec(compiled_e, globals())

assert len(RESULTS_D) == 9

assert len(LOGITS_D) == 9

# =====================================================================
# STEP 95 — THREE-SEED COMPARISON AND STATISTICS
# =====================================================================

section("STEP 95 — THREE-SEED CONFIRMATION")

results_e = pd.DataFrame(
    RESULTS_D
).sort_values(
    ["model", "seed"]
)

assert not results_e.duplicated(
    ["model", "seed"]
).any()

assert all(
    results_e.groupby("model").size() == 3
)

summary_e = (
    results_e.groupby("model")
    .agg(
        mean_accuracy=("accuracy", "mean"),
        sd_accuracy=("accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        sd_macro_f1=("macro_f1", "std"),
        mean_ece=("ece", "mean"),
        mean_nll=("nll", "mean"),
        parameters=("parameters", "first")
    )
    .reset_index()
    .sort_values(
        "mean_macro_f1",
        ascending=False
    )
)

seed_comparison_e = (
    results_e.pivot(
        index="seed",
        columns="model",
        values="macro_f1"
    )
    .reset_index()
)

seed_comparison_e[
    "proposed_minus_standard"
] = (
    seed_comparison_e["fixed_fusion"]
    - seed_comparison_e["standard"]
)

seed_comparison_e[
    "proposed_minus_control"
] = (
    seed_comparison_e["fixed_fusion"]
    - seed_comparison_e["matched_control"]
)

print("\nINDIVIDUAL SEED RESULTS")

print(
    results_e.round(5)
    .to_string(index=False)
)

print("\nTHREE-SEED SUMMARY")

print(
    summary_e.round(5)
    .to_string(index=False)
)

print("\nPAIRED SEED DIFFERENCES")

print(
    seed_comparison_e.round(5)
    .to_string(index=False)
)

# Equal-size ensembles.

ENSEMBLES_E = {}
ensemble_rows_e = []

for mode in MODES_D:

    probabilities = np.mean(
        [
            softmax(
                LOGITS_D[(mode, seed)],
                axis=1
            )
            for seed in SEEDS_D
        ],
        axis=0
    )

    logits = np.log(
        probabilities.clip(1e-12, 1)
    )

    ENSEMBLES_E[mode] = logits

    ensemble_rows_e.append({
        "model": mode,
        "seeds": 3,
        **metrics(Y_VAL, logits),
        **calibration(Y_VAL, logits)
    })

    path = PRED / (
        f"{mode}_three_seed_ensemble_val.npz"
    )

    np.savez_compressed(
        path,
        sample_id=sample_ids_d,
        labels=Y_VAL,
        logits=logits
    )

    upload_private(path)

ensemble_e = (
    pd.DataFrame(ensemble_rows_e)
    .sort_values(
        "macro_f1",
        ascending=False
    )
)

print("\nTHREE-SEED ENSEMBLES")

print(
    ensemble_e.round(5)
    .to_string(index=False)
)

# Paired group bootstrap.

def fast_f1_e(y, p):

    cm = np.bincount(
        3 * y + p,
        minlength=9
    ).reshape(3, 3)

    tp = np.diag(cm).astype(float)

    den = cm.sum(0) + cm.sum(1)

    return np.divide(
        2 * tp,
        den,
        out=np.zeros(3),
        where=den > 0
    ).mean()

groups_e = (
    DATA["val"]["group_id"].to_numpy()
)

members_e = [
    np.flatnonzero(groups_e == group)
    for group in np.unique(groups_e)
]

predictions_e = {
    mode: logits.argmax(axis=1)
    for mode, logits in ENSEMBLES_E.items()
}

pairs_e = [
    ("fixed_fusion", "standard"),
    ("fixed_fusion", "matched_control"),
    ("matched_control", "standard")
]

rng_e = np.random.default_rng(2026)

boot_e = {
    pair: [] for pair in pairs_e
}

for _ in range(BOOTSTRAPS_D):

    selected = rng_e.integers(
        0, len(members_e),
        size=len(members_e)
    )

    idx = np.concatenate([
        members_e[i] for i in selected
    ])

    scores = {
        name: fast_f1_e(
            Y_VAL[idx],
            pred[idx]
        )
        for name, pred
        in predictions_e.items()
    }

    for a, b in pairs_e:

        boot_e[(a, b)].append(
            scores[a] - scores[b]
        )

stat_rows_e = []

for a, b in pairs_e:

    pa = predictions_e[a]
    pb = predictions_e[b]

    ca = pa == Y_VAL
    cb = pb == Y_VAL

    gained = int((ca & ~cb).sum())
    lost = int((~ca & cb).sum())

    pvalue = (
        binomtest(
            gained,
            gained + lost,
            p=0.5
        ).pvalue
        if gained + lost
        else 1.0
    )

    stat_rows_e.append({
        "model": a,
        "reference": b,
        "delta_macro_f1":
            fast_f1_e(Y_VAL, pa)
            - fast_f1_e(Y_VAL, pb),
        "ci95_low":
            np.percentile(boot_e[(a, b)], 2.5),
        "ci95_high":
            np.percentile(boot_e[(a, b)], 97.5),
        "gained_correct": gained,
        "lost_correct": lost,
        "net_correct": gained - lost,
        "mcnemar_p": pvalue
    })

stats_e = pd.DataFrame(stat_rows_e)

pvalues_e = stats_e["mcnemar_p"].to_numpy()

order_e = np.argsort(pvalues_e)

adjusted_e = np.empty(len(pvalues_e))

running_e = 0.0

for rank, idx in enumerate(order_e):

    running_e = max(
        running_e,
        (len(pvalues_e) - rank)
        * pvalues_e[idx]
    )

    adjusted_e[idx] = min(
        running_e, 1.0
    )

stats_e["mcnemar_p_holm"] = adjusted_e

print("\nPAIRED STATISTICAL COMPARISONS")

print(
    stats_e.round(5)
    .to_string(index=False)
)

# Linguistic subgroup and class-wise analysis.

subgroup_rows_e = []
class_rows_e = []
confusion_rows_e = []

for mode, logits in ENSEMBLES_E.items():

    for subgroup, mask in MASKS.items():

        if not mask.any():
            continue

        subgroup_rows_e.append({
            "model": mode,
            "subgroup": subgroup,
            "n": int(mask.sum()),
            **metrics(
                Y_VAL[mask],
                logits[mask]
            )
        })

    predicted = logits.argmax(axis=1)

    report = classification_report(
        Y_VAL,
        predicted,
        labels=[0, 1, 2],
        target_names=[
            "Negative",
            "Neutral",
            "Positive"
        ],
        output_dict=True,
        zero_division=0
    )

    for label in [
        "Negative",
        "Neutral",
        "Positive"
    ]:

        class_rows_e.append({
            "model": mode,
            "class": label,
            **report[label]
        })

    cm = confusion_matrix(
        Y_VAL,
        predicted,
        labels=[0, 1, 2]
    )

    for i in range(3):

        for j in range(3):

            confusion_rows_e.append({
                "model": mode,
                "true_class": i,
                "predicted_class": j,
                "count": int(cm[i, j])
            })

subgroups_e = pd.DataFrame(
    subgroup_rows_e
)

classes_e = pd.DataFrame(
    class_rows_e
)

confusions_e = pd.DataFrame(
    confusion_rows_e
)

# Validation-fitted temperature scaling.
# These are descriptive calibration results only.
# Temperatures must remain frozen for test evaluation.

from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

calibration_rows_e = []
temperatures_e = {}

for mode, logits in ENSEMBLES_E.items():

    def nll_e(log_temperature):

        temperature = np.exp(
            log_temperature
        )

        scaled = logits / temperature

        return float(
            np.mean(
                logsumexp(
                    scaled, axis=1
                )
                - scaled[
                    np.arange(len(Y_VAL)),
                    Y_VAL
                ]
            )
        )

    optimization = minimize_scalar(
        nll_e,
        bounds=(-3, 3),
        method="bounded"
    )

    temperature = float(
        np.exp(optimization.x)
    )

    temperatures_e[mode] = temperature

    for variant, value in [
        ("uncalibrated", 1.0),
        ("validation_fitted", temperature)
    ]:

        calibration_rows_e.append({
            "model": mode,
            "variant": variant,
            "temperature": value,
            **calibration(
                Y_VAL,
                logits / value
            )
        })

calibration_e = pd.DataFrame(
    calibration_rows_e
)

save_json(
    temperatures_e,
    AUDIT / "locked_ensemble_temperatures.json"
)

# =====================================================================
# STEP 96 — PUBLICATION TABLES AND FIGURES
# =====================================================================

section("STEP 96 — PUBLICATION ASSETS")

tables_e = {
    "stage6e_01_individual_results": results_e,
    "stage6e_02_multiseed_summary": summary_e,
    "stage6e_03_seed_differences": seed_comparison_e,
    "stage6e_04_ensemble_results": ensemble_e,
    "stage6e_05_paired_statistics": stats_e,
    "stage6e_06_subgroup_results": subgroups_e,
    "stage6e_07_classwise_results": classes_e,
    "stage6e_08_confusion_matrices": confusions_e,
    "stage6e_09_calibration": calibration_e,
    "stage6e_10_training_history":
        pd.DataFrame(HISTORY_D)
}

for name, frame in tables_e.items():

    frame.to_csv(
        TABLES / f"{name}.csv",
        index=False
    )

    for folder in [
        BACKUP / "tables",
        PUB / "tables"
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        frame.to_csv(
            folder / f"{name}.csv",
            index=False
        )

    tex_path = (
        PUB / "tables" / f"{name}.tex"
    )

    tex_path.write_text(
        frame.round(4).to_latex(
            index=False,
            escape=True
        ),
        encoding="utf-8"
    )

    tex_target = (
        BACKUP / "publication_assets"
        / "tables" / tex_path.name
    )

    tex_target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        tex_path, tex_target
    )

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

FIGURES_E = []

def figure_e(
    fig, name, caption, source_table
):

    fig.tight_layout()

    for ext in ["png", "pdf", "svg"]:

        path = (
            PUB / "figures" / ext
            / f"{name}.{ext}"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.savefig(
            path,
            dpi=600 if ext == "png" else 300,
            bbox_inches="tight"
        )

        target = (
            BACKUP / "publication_assets"
            / path.relative_to(PUB)
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(path, target)

    caption_path = (
        PUB / "captions" / f"{name}.txt"
    )

    caption_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    caption_path.write_text(
        caption, encoding="utf-8"
    )

    target = (
        BACKUP / "publication_assets"
        / "captions" / caption_path.name
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        caption_path, target
    )

    FIGURES_E.append({
        "figure_id": name,
        "stage": "6e",
        "png": f"figures/png/{name}.png",
        "pdf": f"figures/pdf/{name}.pdf",
        "svg": f"figures/svg/{name}.svg",
        "caption": caption,
        "source_table": source_table
    })

    plt.close(fig)

names_e = {
    "standard": "BanglaBERT",
    "fixed_fusion": "AMSC-Net V2",
    "matched_control": "Matched MLP control"
}

# Figure 1 — Three-seed comparison.

plot = summary_e.sort_values(
    "mean_macro_f1"
)

fig, ax = plt.subplots(
    figsize=(8, 4.5)
)

ax.barh(
    plot["model"].map(names_e),
    plot["mean_macro_f1"],
    xerr=plot["sd_macro_f1"],
    capsize=4
)

ax.set_xlim(0, 1)
ax.set_xlabel("Mean validation macro-F1")

ax.set_title(
    "Three-seed architectural comparison"
)

figure_e(
    fig,
    "stage6e_01_multiseed_comparison",
    "Mean and sample standard deviation across "
    "three training seeds on a fixed cohort.",
    "stage6e_02_multiseed_summary.csv"
)

# Figure 2 — Seed consistency.

fig, ax = plt.subplots(
    figsize=(7.5, 4.5)
)

for mode in MODES_D:

    part = results_e[
        results_e["model"] == mode
    ].sort_values("seed")

    ax.plot(
        part["seed"].astype(str),
        part["macro_f1"],
        marker="o",
        label=names_e[mode]
    )

ax.set_xlabel("Training seed")
ax.set_ylabel("Validation macro-F1")
ax.legend(frameon=False)

figure_e(
    fig,
    "stage6e_02_seed_consistency",
    "Performance across three independent "
    "training seeds.",
    "stage6e_01_individual_results.csv"
)

# Figure 3 — Equal-size ensemble comparison.

fig, ax = plt.subplots(
    figsize=(8, 4.5)
)

plot = ensemble_e.sort_values(
    "macro_f1"
)

bars = ax.barh(
    plot["model"].map(names_e),
    plot["macro_f1"]
)

ax.bar_label(
    bars, fmt="%.4f", padding=4
)

ax.set_xlim(0, 1.05)
ax.set_xlabel("Validation macro-F1")

figure_e(
    fig,
    "stage6e_03_ensemble_comparison",
    "Comparison of equal-size three-model "
    "probability ensembles.",
    "stage6e_04_ensemble_results.csv"
)

# Figure 4 — Bootstrap effects.

fig, ax = plt.subplots(
    figsize=(9, 4.5)
)

for i, row in stats_e.iterrows():

    ax.plot(
        [row["ci95_low"], row["ci95_high"]],
        [i, i]
    )

    ax.scatter(
        row["delta_macro_f1"], i
    )

ax.axvline(
    0, linestyle="--", color="gray"
)

ax.set_yticks(
    range(len(stats_e)),
    [
        f"{names_e[r.model]} vs "
        f"{names_e[r.reference]}"
        for r in stats_e.itertuples()
    ]
)

ax.set_xlabel("Macro-F1 difference")

figure_e(
    fig,
    "stage6e_04_paired_effects",
    "Exploratory paired group-bootstrap "
    "95% confidence intervals on validation.",
    "stage6e_05_paired_statistics.csv"
)

# Figure 5 — Linguistic subgroup heatmap.

plot = subgroups_e.pivot(
    index="subgroup",
    columns="model",
    values="macro_f1"
)

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

im = ax.imshow(
    plot.to_numpy(),
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    aspect="auto"
)

for i in range(len(plot.index)):

    for j in range(len(plot.columns)):

        value = plot.iloc[i, j]

        ax.text(
            j, i, f"{value:.3f}",
            ha="center", va="center",
            color=(
                "white"
                if value > 0.65
                else "black"
            )
        )

ax.set_xticks(
    range(len(plot.columns)),
    [names_e[x] for x in plot.columns],
    rotation=20,
    ha="right"
)

ax.set_yticks(
    range(len(plot.index)),
    plot.index
)

fig.colorbar(im, ax=ax)

figure_e(
    fig,
    "stage6e_05_linguistic_heatmap",
    "Three-seed ensemble performance across "
    "heuristic linguistic subgroups.",
    "stage6e_06_subgroup_results.csv"
)

# Figure 6 — Class-wise F1.

plot = classes_e.pivot(
    index="class",
    columns="model",
    values="f1-score"
)

plot = plot.rename(
    columns=names_e
)

fig, ax = plt.subplots(
    figsize=(8, 4.5)
)

plot.plot(
    kind="bar", ax=ax
)

ax.set_ylim(0, 1)
ax.set_ylabel("Class-wise F1")
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=0)
ax.legend(frameon=False)

figure_e(
    fig,
    "stage6e_06_classwise_comparison",
    "Class-wise validation F1 for all "
    "three-seed ensembles.",
    "stage6e_07_classwise_results.csv"
)

# Figure 7 — Calibration comparison.

plot = calibration_e.pivot(
    index="model",
    columns="variant",
    values="ece"
)

plot.index = plot.index.map(names_e)

fig, ax = plt.subplots(
    figsize=(8, 4.5)
)

plot.plot(
    kind="barh", ax=ax
)

ax.set_xlabel(
    "Expected calibration error"
)

ax.set_ylabel("")
ax.legend(frameon=False)

figure_e(
    fig,
    "stage6e_07_calibration",
    "Descriptive validation calibration before "
    "and after validation-fitted temperature scaling.",
    "stage6e_09_calibration.csv"
)

# Figure 8 — Parameter efficiency.

fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

for row in summary_e.itertuples():

    ax.scatter(
        row.parameters / 1e6,
        row.mean_macro_f1,
        s=90,
        label=names_e[row.model]
    )

ax.set_xlabel("Parameters (millions)")
ax.set_ylabel("Mean validation macro-F1")
ax.legend(frameon=False)

figure_e(
    fig,
    "stage6e_08_parameter_efficiency",
    "Mean validation macro-F1 relative "
    "to total trainable parameter count.",
    "stage6e_02_multiseed_summary.csv"
)

# =====================================================================
# STEP 97 — VERIFIED BACKUP AND FINAL DECISION
# =====================================================================

section("STEP 97 — FINAL RESEARCH DECISION")

mean_e = summary_e.set_index(
    "model"
)["mean_macro_f1"]

wins_standard_e = int(
    (
        seed_comparison_e[
            "proposed_minus_standard"
        ] > 0
    ).sum()
)

wins_control_e = int(
    (
        seed_comparison_e[
            "proposed_minus_control"
        ] > 0
    ).sum()
)

proposed_e = float(
    mean_e["fixed_fusion"]
)

standard_e = float(
    mean_e["standard"]
)

control_score_e = float(
    mean_e["matched_control"]
)

if (
    proposed_e > standard_e
    and proposed_e > control_score_e
    and wins_standard_e >= 2
    and wins_control_e >= 2
):

    decision_e = (
        "Advance locked fixed-fusion candidate "
        "to independent evaluation after final "
        "protocol and literature review."
    )

else:

    decision_e = (
        "Architecture superiority remains "
        "unconfirmed. Review evidence before "
        "unlocking the test cohort."
    )

report_e = {
    "candidate": "AMSC-Net V2 fixed fusion",
    "proposed_mean_macro_f1": proposed_e,
    "standard_mean_macro_f1": standard_e,
    "matched_control_mean_macro_f1":
        control_score_e,
    "delta_vs_standard":
        proposed_e - standard_e,
    "delta_vs_matched_control":
        proposed_e - control_score_e,
    "wins_vs_standard": wins_standard_e,
    "wins_vs_control": wins_control_e,
    "training_seeds": SEEDS_D,
    "new_training_runs": 5,
    "recovered_training_runs": 4,
    "total_comparison_runs": 9,
    "decision": decision_e,
    "test_evaluated": False,
    "external_evaluation": False,
    "novelty_verified": False
}

save_json(
    report_e,
    AUDIT / "stage6e_summary.json"
)

save_json(
    RECOVERED_E,
    AUDIT / "recovered_stage6d_runs.json"
)

# Audit all new private checkpoint assets.

assets_e = remote_assets()

manifest_e = []

for mode, seed in RUNS_E:

    run_name = f"{mode}_seed{seed}"

    meta = (
        CHECKPOINTS / run_name
        / "completed.json"
    )

    completed = json.loads(
        meta.read_text()
    )

    for name, expected in [
        (
            f"{run_name}_model.pt",
            completed["checkpoint_sha256"]
        ),
        (
            f"{run_name}_val.npz",
            completed["prediction_sha256"]
        ),
        (
            f"{run_name}_completed.json",
            sha256(meta)
        )
    ]:

        assert name in assets_e

        verify_remote_asset(
            assets_e[name],
            expected
        )

        manifest_e.append({
            "name": name,
            "asset_id": assets_e[name]["id"],
            "sha256": expected,
            "size_bytes": assets_e[name]["size"],
            "release": TAG6E
        })

assert len(manifest_e) == 15

save_json(
    {
        "new_release": TAG6E,
        "previous_release": TAG6D,
        "new_assets": manifest_e,
        "recovered_runs": RECOVERED_E,
        "config": CONFIG_D
    },
    AUDIT / "stage6e_remote_manifest.json"
)

# Preserve aggregate audit files.

for path in AUDIT.glob("*.json"):

    shutil.copy2(
        path,
        BACKUP / path.name
    )

# Consolidate all previous publication assets.

for folder in [
    "research_reconstruction_v1",
    "research_stage6b",
    "research_stage6c",
    "research_stage6d",
    "research_stage6e"
]:

    previous = (
        REPO / folder
        / "publication_assets"
    )

    if not previous.exists():
        continue

    for path in previous.rglob("*"):

        if not path.is_file():
            continue

        destination = (
            PUB / path.relative_to(previous)
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if not destination.exists():

            shutil.copy2(
                path, destination
            )

manifest_parts_e = []

for path in REPO.rglob(
    "figure_manifest.csv"
):

    try:

        frame = pd.read_csv(path)

        if "figure_id" in frame.columns:

            manifest_parts_e.append(frame)

    except Exception:

        continue

manifest_parts_e.append(
    pd.DataFrame(FIGURES_E)
)

fig_manifest_e = pd.concat(
    manifest_parts_e,
    ignore_index=True
).drop_duplicates(
    "figure_id",
    keep="last"
)

fig_manifest_e.to_csv(
    PUB / "figure_manifest.csv",
    index=False
)

fig_manifest_e.to_csv(
    BACKUP / "figure_manifest.csv",
    index=False
)

# Private compact reproducibility archive.

compact_e = (
    WORK6E / "sentiment_stage6e_compact.zip"
)

with zipfile.ZipFile(
    compact_e,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for path in ROOT6E.rglob("*"):

        if (
            path.is_file()
            and path.suffix != ".pt"
        ):

            z.write(
                path,
                path.relative_to(ROOT6E)
            )

with zipfile.ZipFile(compact_e) as z:

    assert z.testzip() is None

upload_private(
    compact_e,
    f"stage6e_compact_{sha256(compact_e)[:12]}.zip"
)

publication_e = (
    WORK6E / "sentiment_publication_assets.zip"
)

with zipfile.ZipFile(
    publication_e,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for path in PUB.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(PUB)
            )

with zipfile.ZipFile(publication_e) as z:

    assert z.testzip() is None

upload_private(
    publication_e,
    f"publication_assets_"
    f"{sha256(publication_e)[:12]}.zip"
)

manifest_path_e = (
    AUDIT / "stage6e_remote_manifest.json"
)

upload_private(
    manifest_path_e,
    f"stage6e_manifest_"
    f"{sha256(manifest_path_e)[:12]}.json"
)

FINAL_COMMIT_E = push_d(
    "Complete three-seed AMSC-Net confirmation "
    "and matched-capacity comparison"
)

# =====================================================================
# FINAL REPORT
# =====================================================================

section("FINAL STAGE 6E RESEARCH REPORT")

print("Dataset recovery: PASS")
print("Previous runs recovered: 4")
print("New completed runs: 5")
print("Total comparison runs: 9")
print("Training seeds:", SEEDS_D)
print("Parameter matching: PASS")
print("GitHub synchronized: True")
print("Private checkpoint backup: PASS")
print("Final public commit:", FINAL_COMMIT_E)

print("\nINDIVIDUAL RESULTS")

print(
    results_e.round(5)
    .to_string(index=False)
)

print("\nTHREE-SEED SUMMARY")

print(
    summary_e.round(5)
    .to_string(index=False)
)

print("\nPAIRED SEED DIFFERENCES")

print(
    seed_comparison_e.round(5)
    .to_string(index=False)
)

print("\nENSEMBLE RESULTS")

print(
    ensemble_e.round(5)
    .to_string(index=False)
)

print("\nPAIRED STATISTICS")

print(
    stats_e.round(5)
    .to_string(index=False)
)

print("\nLINGUISTIC SUBGROUPS")

print(
    subgroups_e.pivot(
        index="subgroup",
        columns="model",
        values="macro_f1"
    ).round(4).to_string()
)

print("\nCLASS-WISE PERFORMANCE")

print(
    classes_e.round(5)
    .to_string(index=False)
)

print("\nCALIBRATION")

print(
    calibration_e.round(5)
    .to_string(index=False)
)

print("\nRESEARCH DECISION")

print(
    json.dumps(
        report_e,
        indent=2
    )
)

print("\nPUBLICATION ASSETS")

print(
    "New unique figures:",
    len(FIGURES_E)
)

for ext in ["png", "pdf", "svg"]:

    print(
        ext.upper(), "figures:",
        len(list(
            (PUB / "figures" / ext)
            .glob(f"*.{ext}")
        ))
    )

print(
    "New verified checkpoint assets:",
    len(manifest_e)
)

print(
    "Private release:",
    f"https://github.com/{OWNER}/"
    f"{PRIVATE_REPO}/releases/tag/{TAG6E}"
)

print("Manual file saving required: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("Test evaluated: NO")
print("External evaluation: NO")
print("Novelty conclusively established: NO")

print(
    "Runtime minutes:",
    round(
        (time.time() - START6E) / 60,
        2
    )
)

SESSION.headers.pop(
    "Authorization", None
)

SESSION.close()

GIT_ENV.clear()

TOKEN = None
credential = None

gc.collect()
torch.cuda.empty_cache()

print("\nSTAGE 6E COMPLETE")
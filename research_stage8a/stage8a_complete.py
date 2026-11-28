
# =============================================================================
# STAGE 8A — HIERARCHICAL DUAL-SPACE SENTIMENT NETWORK
# STEPS 141–155
#
# HIERARCHICAL CLASSIFICATION
# NEUTRALITY AND POLARITY REPRESENTATION SPACES
# MULTI-PROTOTYPE LEARNING
# SUPERVISED CONTRASTIVE LEARNING
# OOF DIFFICULTY-AWARE SUPERVISION
# PARAMETER-MATCHED CONTROLS
# PERMANENT GITHUB BACKUP
# =============================================================================

import os
import re
import gc
import json
import time
import math
import shutil
import hashlib
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from getpass import getpass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import softmax
from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)

STAGE8_START = time.time()

assert torch.cuda.is_available(), "Enable GPU T4."

# =============================================================================
# STEP 141 — RECOVER VERIFIED RESEARCH INFRASTRUCTURE
# =============================================================================

print("=" * 90)
print("STEP 141 — RECOVER VERIFIED RESEARCH INFRASTRUCTURE")
print("=" * 90)

OWNER8 = "ankita6876"

PREVIOUS_COMMIT8 = (
    "f0478feef14fbfede7323fbf5030f6c0927716dd"
)

SOURCE_URL8 = (
    "https://raw.githubusercontent.com/"
    "ankita6876/bangla-sentiment-research/"
    f"{PREVIOUS_COMMIT8}/"
    "research_stage7de/stage7de_complete.py"
)

response8 = requests.get(
    SOURCE_URL8,
    timeout=120
)

response8.raise_for_status()

ORIGINAL_SOURCE8 = response8.text

assert "STEP 129" in ORIGINAL_SOURCE8
assert "STEP 130" in ORIGINAL_SOURCE8
assert "STEP 131" in ORIGINAL_SOURCE8

# Capture this complete executable cell for permanent backup.

history8 = get_ipython().history_manager.input_hist_raw

candidates8 = [
    s for s in history8
    if "STAGE 8A — HIERARCHICAL DUAL-SPACE SENTIMENT NETWORK" in s
    and len(s) > 10000
]

assert candidates8, "Run the complete cell."

CURRENT_SOURCE8 = candidates8[-1]

TOKEN8 = getpass(
    "Enter your working GitHub token: "
).strip()

assert TOKEN8
assert TOKEN8 not in CURRENT_SOURCE8

# Recover only the infrastructure, dataset, tokenization,
# OOF predictions and training difficulty.
#
# Stop before the old DGSL architecture is constructed.

stop_marker8 = (
    'section("STEP 131 — DGSL-NET ARCHITECTURE")'
)

assert stop_marker8 in ORIGINAL_SOURCE8

RECOVERY_SOURCE8 = ORIGINAL_SOURCE8.split(
    stop_marker8,
    1
)[0]

# Reuse the entered token without asking again.

old_token8 = '''TOKEN = getpass(
    "Enter your working GitHub token: "
).strip()'''

assert old_token8 in RECOVERY_SOURCE8

RECOVERY_SOURCE8 = RECOVERY_SOURCE8.replace(
    old_token8,
    "TOKEN = TOKEN8",
    1
)

# Preserve the exact historical source hash so the
# existing OOF checkpoints can be recovered.

capture_start8 = RECOVERY_SOURCE8.index(
    "# Preserve complete source before training."
)

capture_end8 = RECOVERY_SOURCE8.index(
    "# STEP 126",
    capture_start8
)

RECOVERY_SOURCE8 = (
    RECOVERY_SOURCE8[:capture_start8]
    + "\nSOURCE = ORIGINAL_SOURCE8\n\n"
    + RECOVERY_SOURCE8[capture_end8:]
)

compile(
    RECOVERY_SOURCE8,
    "<stage7de_verified_recovery>",
    "exec"
)

exec(
    RECOVERY_SOURCE8,
    globals()
)

assert len(TRAIN) == 9204
assert len(VAL) == 1840
assert np.isfinite(OOF_PROBS).all()

assert abs(
    macro_f1(
        Y_TRAIN,
        OOF_PROBS.argmax(1)
    ) - 0.7995833593156781
) < 1e-6

assert abs(
    macro_f1(
        Y_VAL,
        HISTORICAL["prototype_family"].argmax(1)
    ) - 0.8287035909704495
) < 1e-6

print("\nExisting OOF predictions: VERIFIED")
print("OOF retraining required: NO, if all archived runs match")
print("Locked training and validation membership: VERIFIED")

# =============================================================================
# STEP 142 — NEW STAGE CONFIGURATION AND GITHUB STORAGE
# =============================================================================

section("STEP 142 — STAGE 8A CONFIGURATION")

OLD_RELEASE8 = RELEASE
OLD_REMOTE8 = dict(REMOTE)

OLD_OUT8 = OUT

STAGE_TAG = "stage8a-hds-screening-v1"

OUT = WORK / "sentiment_research_stage8a"

AUDIT = OUT / "audit"
TABLES = OUT / "tables"
PRED = OUT / "private_predictions"
CKPT = OUT / "checkpoints"
FIGURES = OUT / "figures"
REC = OUT / "recovered"

BACKUP = REPO / "research_stage8a"

for path in [
    OUT, AUDIT, TABLES, PRED,
    CKPT, FIGURES, REC, BACKUP
]:

    path.mkdir(
        parents=True,
        exist_ok=True
    )

for fmt in ["png", "pdf", "svg"]:

    (FIGURES / fmt).mkdir(
        parents=True,
        exist_ok=True
    )

    (PUB / "figures" / fmt).mkdir(
        parents=True,
        exist_ok=True
    )

# Recover the latest complete publication archive,
# including all 83 existing figures.

publication_assets8 = [
    asset
    for name, asset in OLD_REMOTE8.items()
    if name.startswith(
        "sentiment_stage7de_publication_"
    )
    and name.endswith(".zip")
]

assert publication_assets8

latest_publication8 = max(
    publication_assets8,
    key=lambda a: a["created_at"]
)

archive8 = download_asset(
    latest_publication8,
    REC / "previous_publication.zip"
)

safe_extract(
    archive8,
    PUB
)

# Recover previous DGSL validation predictions
# for descriptive comparison.

previous_dgsl8 = download_asset(
    OLD_REMOTE8["dgsl_full_seed42_val.npz"],
    REC / "previous_dgsl_val.npz"
)

with np.load(
    previous_dgsl8,
    allow_pickle=False
) as z:

    assert np.array_equal(
        z["sample_id"],
        VAL_IDS
    )

    assert np.array_equal(
        z["labels"],
        Y_VAL
    )

    HISTORICAL["previous_dgsl"] = z[
        "probabilities"
    ].copy()

# Create the new private artifact release.

response8 = SESSION.get(
    f"{API}/repos/{OWNER}/{PRIVATE_REPO}/"
    f"releases/tags/{STAGE_TAG}",
    timeout=90
)

if response8.status_code == 404:

    response8 = SESSION.post(
        f"{API}/repos/{OWNER}/{PRIVATE_REPO}/releases",
        json={
            "tag_name": STAGE_TAG,
            "name": "Stage 8A — HDS-Net screening",
            "body": (
                "Hierarchical sentiment classification, "
                "dual-space representations, prototype "
                "and contrastive learning experiments."
            )
        },
        timeout=90
    )

response8.raise_for_status()

RELEASE = response8.json()
REMOTE = assets_of(RELEASE)
REMOTE_MANIFEST = []

def push_public(message):

    git("add", "research_stage8a")

    if git("diff", "--cached", "--name-only"):

        git("commit", "-m", message)

    git(
        "push",
        "origin",
        f"HEAD:{BRANCH}"
    )

    local = git("rev-parse", "HEAD")

    remote = git(
        "ls-remote",
        "origin",
        f"refs/heads/{BRANCH}"
    ).split()[0]

    assert local == remote

    print("GitHub synchronized:", local)

    return local

# Stage-specific experimental configuration.

SEED8 = 42
EPOCHS8 = 5

DIM8 = 192
DROPOUT8 = .15

ENCODER_LR8 = 2e-5
HEAD_LR8 = 2e-4

CONTRASTIVE_WEIGHT8 = .08
PROTOTYPE_WEIGHT8 = .10
HIERARCHY_WEIGHT8 = .30

TEMPERATURE8 = .12
N_PROTOTYPES8 = 3

MODES8 = [
    "standard",
    "matched_flat",
    "hierarchy_only",
    "dual_space",
    "dual_space_prototype",
    "hds_full"
]

SOURCE_HASH8 = hashlib.sha256(
    CURRENT_SOURCE8.encode()
).hexdigest()

for destination in [
    AUDIT / "stage8a_complete.py",
    BACKUP / "stage8a_complete.py",
    PUB / "scripts/stage8a_complete.py"
]:

    destination.write_text(
        CURRENT_SOURCE8,
        encoding="utf-8"
    )

(AUDIT / "verified_stage7de_source.py").write_text(
    ORIGINAL_SOURCE8,
    encoding="utf-8"
)

push_public(
    "Stage 8A: preserve complete HDS research source"
)

print("GPU:", torch.cuda.get_device_name(0))
print("Architecture experiments:", len(MODES8))
print("Epochs per experiment:", EPOCHS8)
print("Private release:", RELEASE["html_url"])

for fmt in ["png", "pdf", "svg"]:

    print(
        "Recovered", fmt.upper(), "figures:",
        len(list(
            (PUB / "figures" / fmt).rglob(f"*.{fmt}")
        ))
    )

# =============================================================================
# STEP 143 — ANALYZE NEUTRALITY AND POLARITY BOTTLENECKS
# =============================================================================

section("STEP 143 — HIERARCHICAL BOTTLENECK ANALYSIS")

def hierarchy_diagnostics(y, probabilities):

    pred = probabilities.argmax(1)

    true_neutral = y == 1
    predicted_neutral = pred == 1

    polar = y != 1

    polarity_pred = np.where(
        probabilities[:, 2] > probabilities[:, 0],
        2,
        0
    )

    return {
        "neutrality_accuracy": float(
            accuracy_score(
                true_neutral,
                predicted_neutral
            )
        ),
        "neutrality_macro_f1": float(
            f1_score(
                true_neutral,
                predicted_neutral,
                average="macro"
            )
        ),
        "conditional_polarity_accuracy": float(
            accuracy_score(
                y[polar],
                polarity_pred[polar]
            )
        ),
        "neutral_to_negative": int(
            ((y == 1) & (pred == 0)).sum()
        ),
        "neutral_to_positive": int(
            ((y == 1) & (pred == 2)).sum()
        ),
        "negative_to_neutral": int(
            ((y == 0) & (pred == 1)).sum()
        ),
        "positive_to_neutral": int(
            ((y == 2) & (pred == 1)).sum()
        ),
        "negative_positive_reversal": int(
            (
                ((y == 0) & (pred == 2))
                | ((y == 2) & (pred == 0))
            ).sum()
        )
    }

OOF_HIERARCHY8 = hierarchy_diagnostics(
    Y_TRAIN,
    OOF_PROBS
)

print(
    json.dumps(
        OOF_HIERARCHY8,
        indent=2
    )
)

# Quantify OOF ambiguity separately for neutrality
# and conditional polarity.

OOF_NEUTRAL8 = OOF_PROBS[:, 1]

OOF_POLARITY8 = (
    OOF_PROBS[:, 2]
    / np.clip(
        OOF_PROBS[:, 0] + OOF_PROBS[:, 2],
        1e-8,
        None
    )
)

NEUTRAL_TARGET8 = (
    Y_TRAIN == 1
).astype(np.float32)

POLAR_TARGET8 = (
    Y_TRAIN == 2
).astype(np.float32)

NEUTRAL_ERROR8 = np.abs(
    NEUTRAL_TARGET8 - OOF_NEUTRAL8
)

POLAR_ERROR8 = np.abs(
    POLAR_TARGET8 - OOF_POLARITY8
)

POLAR_ERROR8[Y_TRAIN == 1] = 0

NEUTRAL_DIFFICULTY8 = (
    .5 * DIFFICULTY
    + .5 * NEUTRAL_ERROR8
).astype(np.float32)

POLAR_DIFFICULTY8 = (
    .5 * DIFFICULTY
    + .5 * POLAR_ERROR8
).astype(np.float32)

DIFFICULTY_ROWS8 = []

for k, label in enumerate(CLASSES):

    mask = Y_TRAIN == k

    DIFFICULTY_ROWS8.append({
        "class": label,
        "n": int(mask.sum()),
        "general_difficulty": DIFFICULTY[mask].mean(),
        "neutrality_difficulty":
            NEUTRAL_DIFFICULTY8[mask].mean(),
        "conditional_polarity_difficulty":
            POLAR_DIFFICULTY8[mask].mean()
            if k != 1 else np.nan
    })

DIFFICULTY_TABLE8 = save_table(
    DIFFICULTY_ROWS8,
    "stage8a_01_hierarchical_difficulty"
)

print(
    DIFFICULTY_TABLE8.round(5).to_string(
        index=False
    )
)

# =============================================================================
# STEP 144 — DUAL-SPACE ARCHITECTURE COMPONENTS
# =============================================================================

section("STEP 144 — HDS-NET ARCHITECTURE")

def masked_mean8(x, mask):

    weights = mask.unsqueeze(-1).to(x.dtype)

    return (
        (x * weights).sum(1)
        / weights.sum(1).clamp_min(1)
    )

class AttentionPool8(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x, valid):

        scores = self.score(
            x
        ).squeeze(-1).float()

        scores = scores.masked_fill(
            ~valid,
            -1e4
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        weights = weights * valid.float()

        weights = (
            weights
            / weights.sum(
                1,
                keepdim=True
            ).clamp_min(1e-8)
        )

        return (
            x.float()
            * weights.unsqueeze(-1)
        ).sum(1)

class MultiPrototype8(nn.Module):

    def __init__(
        self,
        dim,
        classes=2,
        prototypes=3
    ):

        super().__init__()

        self.prototypes = nn.Parameter(
            torch.randn(
                classes,
                prototypes,
                dim
            ) * .02
        )

        self.log_scale = nn.Parameter(
            torch.tensor(2.0)
        )

    def forward(self, h):

        h = F.normalize(
            h.float(),
            dim=-1
        )

        prototypes = F.normalize(
            self.prototypes.float(),
            dim=-1
        )

        similarity = torch.einsum(
            "bd,ckd->bck",
            h,
            prototypes
        )

        scale = (
            F.softplus(self.log_scale)
            + 1
        )

        scores = torch.logsumexp(
            similarity * scale,
            dim=-1
        ) - math.log(
            self.prototypes.shape[1]
        )

        return scores, similarity

class HDSNet(nn.Module):

    def __init__(
        self,
        base,
        mode,
        control_width=256
    ):

        super().__init__()

        self.mode = mode

        self.encoder = base.electra
        self.base_classifier = base.classifier

        hidden = base.config.hidden_size

        if mode == "standard":

            return

        self.projection = nn.Sequential(
            nn.Linear(hidden, DIM8),
            nn.LayerNorm(DIM8),
            nn.GELU()
        )

        self.pool = AttentionPool8(
            DIM8
        )

        pooled_dim = 3 * DIM8

        if mode == "matched_flat":

            self.control = nn.Sequential(
                nn.Linear(
                    pooled_dim,
                    control_width
                ),
                nn.LayerNorm(control_width),
                nn.GELU(),
                nn.Dropout(DROPOUT8),
                nn.Linear(
                    control_width,
                    DIM8
                ),
                nn.GELU(),
                nn.Linear(DIM8, 3)
            )

            return

        self.neutral_space = nn.Sequential(
            nn.Linear(pooled_dim, DIM8),
            nn.LayerNorm(DIM8),
            nn.GELU(),
            nn.Dropout(DROPOUT8)
        )

        self.neutral_head = nn.Linear(
            DIM8,
            2
        )

        if mode != "hierarchy_only":

            self.polar_space = nn.Sequential(
                nn.Linear(
                    pooled_dim,
                    DIM8
                ),
                nn.LayerNorm(DIM8),
                nn.GELU(),
                nn.Dropout(DROPOUT8)
            )

        self.polar_head = nn.Linear(
            DIM8,
            2
        )

        self.use_prototypes = mode in [
            "dual_space_prototype",
            "hds_full"
        ]

        if self.use_prototypes:

            self.neutral_prototypes = MultiPrototype8(
                DIM8,
                2,
                N_PROTOTYPES8
            )

            self.polar_prototypes = MultiPrototype8(
                DIM8,
                2,
                N_PROTOTYPES8
            )

        self.use_dual_space = (
            mode != "hierarchy_only"
        )

        # Learned direct three-class residual
        # retains the original semantic pathway.

        self.residual = nn.Sequential(
            nn.Linear(
                2 * DIM8,
                DIM8
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT8),
            nn.Linear(DIM8, 3)
        )

        nn.init.zeros_(
            self.residual[-1].weight
        )

        nn.init.zeros_(
            self.residual[-1].bias
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        valid_mask,
        ling=None
    ):

        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        base_logits = self.base_classifier(
            hidden
        ).float()

        base_logp = F.log_softmax(
            base_logits,
            dim=-1
        )

        if self.mode == "standard":

            return {
                "logits": base_logits,
                "base_logits": base_logits
            }

        x = self.projection(hidden)

        pooled = torch.cat([
            x[:, 0],
            masked_mean8(
                x,
                valid_mask
            ),
            self.pool(
                x,
                valid_mask
            )
        ], dim=-1)

        if self.mode == "matched_flat":

            logits = (
                base_logits
                + self.control(pooled).float()
            )

            return {
                "logits": logits,
                "base_logits": base_logits
            }

        h_neutral = self.neutral_space(
            pooled
        )

        h_polar = (
            self.polar_space(pooled)
            if self.use_dual_space
            else h_neutral
        )

        # Neutrality classes:
        # 0 = non-neutral, 1 = neutral.

        neutral_logits = self.neutral_head(
            h_neutral
        ).float()

        # Polarity classes:
        # 0 = negative, 1 = positive.

        polar_logits = self.polar_head(
            h_polar
        ).float()

        neutral_similarity = None
        polar_similarity = None

        if self.use_prototypes:

            pn, neutral_similarity = (
                self.neutral_prototypes(
                    h_neutral
                )
            )

            pp, polar_similarity = (
                self.polar_prototypes(
                    h_polar
                )
            )

            neutral_logits = (
                neutral_logits + pn
            )

            polar_logits = (
                polar_logits + pp
            )

        neutral_logp = F.log_softmax(
            neutral_logits,
            dim=-1
        )

        polar_logp = F.log_softmax(
            polar_logits,
            dim=-1
        )

        # Probabilistically consistent hierarchy:
        #
        # P(negative) = P(non-neutral) * P(negative | non-neutral)
        # P(neutral)  = P(neutral)
        # P(positive) = P(non-neutral) * P(positive | non-neutral)

        hierarchical_logp = torch.stack([
            neutral_logp[:, 0] + polar_logp[:, 0],
            neutral_logp[:, 1],
            neutral_logp[:, 0] + polar_logp[:, 1]
        ], dim=-1)

        residual_logits = (
            base_logits
            + self.residual(
                torch.cat([
                    h_neutral,
                    h_polar
                ], dim=-1)
            ).float()
        )

        residual_logp = F.log_softmax(
            residual_logits,
            dim=-1
        )

        # Fixed mixture isolates the learning formulation
        # from the previously unsuccessful adaptive router.

        final_logp = torch.logaddexp(
            hierarchical_logp + math.log(.65),
            residual_logp + math.log(.35)
        )

        return {
            "logits": final_logp,
            "base_logits": base_logits,
            "residual_logits": residual_logits,
            "hierarchical_logp": hierarchical_logp,
            "neutral_logits": neutral_logits,
            "polar_logits": polar_logits,
            "neutral_embedding": h_neutral,
            "polar_embedding": h_polar,
            "neutral_similarity": neutral_similarity,
            "polar_similarity": polar_similarity
        }

def count_parameters8(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def new_model8(
    mode,
    width=256,
    seed=SEED8
):

    seed_all(seed)

    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    return HDSNet(
        base,
        mode,
        width
    )

# =============================================================================
# STEP 145 — PARAMETER-MATCHED CONTROL
# =============================================================================

section("STEP 145 — PARAMETER MATCHING")

probe_base8 = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

probe_standard8 = HDSNet(
    probe_base8,
    "standard"
)

BASE_PARAMS8 = count_parameters8(
    probe_standard8
)

probe_full8 = HDSNet(
    probe_base8,
    "hds_full"
)

TARGET_PARAMS8 = count_parameters8(
    probe_full8
)

probe_control8 = HDSNet(
    probe_base8,
    "matched_flat",
    256
)

CONTROL_PARAMS8 = count_parameters8(
    probe_control8
)

# Linear layers plus LayerNorm and biases.

increment8 = (
    3 * DIM8
    + 1
    + 2
    + DIM8
)

MATCHED_WIDTH8 = max(
    16,
    int(round(
        256
        + (
            TARGET_PARAMS8
            - CONTROL_PARAMS8
        ) / increment8
    ))
)

probe_matched8 = HDSNet(
    probe_base8,
    "matched_flat",
    MATCHED_WIDTH8
)

MATCHED_PARAMS8 = count_parameters8(
    probe_matched8
)

MISMATCH8 = (
    abs(
        TARGET_PARAMS8
        - MATCHED_PARAMS8
    )
    / (
        TARGET_PARAMS8
        - BASE_PARAMS8
    )
)

assert MISMATCH8 < .01

print("Standard parameters:", BASE_PARAMS8)
print("HDS parameters:", TARGET_PARAMS8)
print("Matched parameters:", MATCHED_PARAMS8)
print("Matched width:", MATCHED_WIDTH8)
print("Extra-parameter mismatch:", 100 * MISMATCH8, "%")

del (
    probe_base8,
    probe_standard8,
    probe_full8,
    probe_control8,
    probe_matched8
)

clear_gpu()

# =============================================================================
# STEP 146 — BOUNDARY-AWARE LEARNING OBJECTIVES
# =============================================================================

section("STEP 146 — SPECIALIZED TRAINING OBJECTIVES")

NEUTRAL_DIFFICULTY_GPU8 = torch.tensor(
    NEUTRAL_DIFFICULTY8,
    device=DEVICE
)

POLAR_DIFFICULTY_GPU8 = torch.tensor(
    POLAR_DIFFICULTY8,
    device=DEVICE
)

def weighted_mean8(values, weights):

    return (
        values * weights
    ).sum() / weights.sum().clamp_min(1e-8)

def supervised_contrastive8(
    embeddings,
    labels,
    weights=None,
    temperature=TEMPERATURE8
):

    n = len(labels)

    if n < 3:

        return embeddings.sum() * 0

    z = F.normalize(
        embeddings.float(),
        dim=-1
    )

    similarities = (
        z @ z.T
    ) / temperature

    eye = torch.eye(
        n,
        device=z.device,
        dtype=torch.bool
    )

    positive = (
        labels[:, None]
        == labels[None, :]
    ) & ~eye

    valid = positive.any(1)

    if not valid.any():

        return embeddings.sum() * 0

    similarities = similarities.masked_fill(
        eye,
        -1e4
    )

    log_prob = (
        similarities
        - torch.logsumexp(
            similarities,
            dim=1,
            keepdim=True
        )
    )

    per_anchor = -(
        log_prob
        * positive.float()
    ).sum(1) / positive.sum(1).clamp_min(1)

    if weights is None:

        return per_anchor[valid].mean()

    return weighted_mean8(
        per_anchor[valid],
        weights[valid]
    )

def prototype_margin8(
    similarity,
    labels,
    margin=.15
):

    if similarity is None or len(labels) == 0:

        return None

    best_similarity = similarity.max(
        dim=-1
    ).values

    true_similarity = best_similarity.gather(
        1,
        labels[:, None]
    ).squeeze(1)

    false_mask = ~F.one_hot(
        labels,
        num_classes=2
    ).bool()

    false_similarity = best_similarity.masked_fill(
        ~false_mask,
        -1e4
    ).max(1).values

    return F.relu(
        margin
        + false_similarity
        - true_similarity
    ).mean()

def objective8(output, batch, mode):

    y = batch["labels"]

    primary = F.cross_entropy(
        output["logits"],
        y
    )

    zero = primary * 0

    parts = {
        "primary": primary,
        "hierarchy": zero,
        "contrastive": zero,
        "prototype": zero,
        "residual": zero
    }

    if mode in [
        "standard",
        "matched_flat"
    ]:

        return primary, parts

    neutral_target = (
        y == 1
    ).long()

    polar_mask = y != 1

    polar_target = (
        y[polar_mask] == 2
    ).long()

    neutral_ce = F.cross_entropy(
        output["neutral_logits"],
        neutral_target,
        reduction="none"
    )

    # Difficulty weights are used only in the full model.

    if mode == "hds_full":

        neutral_weights = (
            1
            + NEUTRAL_DIFFICULTY_GPU8[
                batch["index"]
            ]
        )

    else:

        neutral_weights = torch.ones_like(
            neutral_ce
        )

    neutral_loss = weighted_mean8(
        neutral_ce,
        neutral_weights
    )

    polar_loss = zero

    if polar_mask.any():

        polar_ce = F.cross_entropy(
            output["polar_logits"][polar_mask],
            polar_target,
            reduction="none"
        )

        if mode == "hds_full":

            polar_weights = (
                1
                + POLAR_DIFFICULTY_GPU8[
                    batch["index"][polar_mask]
                ]
            )

        else:

            polar_weights = torch.ones_like(
                polar_ce
            )

        polar_loss = weighted_mean8(
            polar_ce,
            polar_weights
        )

    hierarchy_loss = (
        neutral_loss
        + polar_loss
    ) / 2

    residual_loss = F.cross_entropy(
        output["residual_logits"],
        y
    )

    loss = (
        primary
        + HIERARCHY_WEIGHT8 * hierarchy_loss
        + .10 * residual_loss
    )

    parts["hierarchy"] = hierarchy_loss
    parts["residual"] = residual_loss

    if mode in [
        "dual_space_prototype",
        "hds_full"
    ]:

        prototype_neutral = prototype_margin8(
            output["neutral_similarity"],
            neutral_target
        )

        prototype_polar = (
            prototype_margin8(
                output["polar_similarity"][polar_mask],
                polar_target
            )
            if polar_mask.any()
            else zero
        )

        prototype_loss = (
            prototype_neutral
            + prototype_polar
        ) / 2

        loss = (
            loss
            + PROTOTYPE_WEIGHT8 * prototype_loss
        )

        parts["prototype"] = prototype_loss

    if mode == "hds_full":

        contrastive_neutral = supervised_contrastive8(
            output["neutral_embedding"],
            neutral_target,
            neutral_weights
        )

        contrastive_polar = (
            supervised_contrastive8(
                output["polar_embedding"][polar_mask],
                polar_target,
                polar_weights
            )
            if polar_mask.any()
            else zero
        )

        contrastive_loss = (
            contrastive_neutral
            + contrastive_polar
        ) / 2

        loss = (
            loss
            + CONTRASTIVE_WEIGHT8
            * contrastive_loss
        )

        parts["contrastive"] = contrastive_loss

    return loss, parts

# =============================================================================
# STEP 147 — ARCHITECTURE VALIDATION
# =============================================================================

section("STEP 147 — ARCHITECTURE SMOKE TESTS")

smoke8 = to_device(
    next(
        iter(
            make_loader(
                "train",
                indices=np.arange(16),
                difficulty=DIFFICULTY,
                teacher=OOF_PROBS
            )
        )
    )
)

ARCHITECTURE_ROWS8 = []

for mode in MODES8:

    model = new_model8(
        mode,
        MATCHED_WIDTH8
    ).to(DEVICE)

    output = model(
        smoke8["input_ids"],
        smoke8["attention_mask"],
        smoke8["valid_mask"],
        smoke8["ling"]
    )

    loss, parts = objective8(
        output,
        smoke8,
        mode
    )

    assert output["logits"].shape == (
        len(smoke8["labels"]),
        3
    )

    assert torch.isfinite(loss)

    if "hierarchical_logp" in output:

        assert torch.allclose(
            output[
                "hierarchical_logp"
            ].exp().sum(1),
            torch.ones(
                len(smoke8["labels"]),
                device=DEVICE
            ),
            atol=1e-5
        )

        assert torch.allclose(
            output["logits"].exp().sum(1),
            torch.ones(
                len(smoke8["labels"]),
                device=DEVICE
            ),
            atol=1e-5
        )

    loss.backward()

    gradients = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]

    assert gradients

    assert all(
        torch.isfinite(g).all().item()
        for g in gradients
    )

    if mode == "hds_full":

        assert (
            model.neutral_prototypes
                .prototypes.grad is not None
        )

        assert (
            model.polar_prototypes
                .prototypes.grad is not None
        )

    ARCHITECTURE_ROWS8.append({
        "mode": mode,
        "parameters": count_parameters8(model),
        "extra_parameters":
            count_parameters8(model) - BASE_PARAMS8,
        "forward_backward": "PASS"
    })

    print(
        mode,
        "| parameters:",
        count_parameters8(model),
        "| PASS"
    )

    del model, output, loss, gradients, parts

    clear_gpu()

ARCHITECTURE_TABLE8 = save_table(
    ARCHITECTURE_ROWS8,
    "stage8a_02_architecture_audit"
)

# =============================================================================
# STEP 148 — TRAINING AND EVALUATION UTILITIES
# =============================================================================

section("STEP 148 — TRAINING PROTOCOL")

CONFIG8 = {
    "stage": "8A",
    "model": MODEL_NAME,
    "seed": SEED8,
    "epochs": EPOCHS8,
    "dim": DIM8,
    "dropout": DROPOUT8,
    "encoder_lr": ENCODER_LR8,
    "head_lr": HEAD_LR8,
    "batch_size": TRAIN_BATCH,
    "max_length": MAX_LENGTH,
    "matched_width": MATCHED_WIDTH8,
    "hierarchy_weight": HIERARCHY_WEIGHT8,
    "prototype_weight": PROTOTYPE_WEIGHT8,
    "contrastive_weight": CONTRASTIVE_WEIGHT8,
    "contrastive_temperature": TEMPERATURE8,
    "prototypes_per_class": N_PROTOTYPES8,
    "hierarchical_mixture": .65,
    "residual_mixture": .35,
    "source_sha256": SOURCE_HASH8,
    "oof_sha256": sha256(OOF_PATH),
    "train_ids_sha256": hashlib.sha256(
        TRAIN_IDS.tobytes()
    ).hexdigest(),
    "val_ids_sha256": hashlib.sha256(
        VAL_IDS.tobytes()
    ).hexdigest(),
    "test_evaluated": False
}

save_json(
    CONFIG8,
    AUDIT / "stage8a_config.json"
)

save_json(
    CONFIG8,
    BACKUP / "stage8a_config.json"
)

push_public(
    "Stage 8A: lock HDS architecture and training protocol"
)

def evaluate8(model, loader):

    model.eval()

    collected = {}

    keys = [
        "logits",
        "base_logits",
        "hierarchical_logp",
        "neutral_logits",
        "polar_logits",
        "neutral_embedding",
        "polar_embedding"
    ]

    with torch.no_grad():

        for batch in loader:

            batch = to_device(batch)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                output = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["valid_mask"],
                    batch["ling"]
                )

            for key in keys:

                if key in output:

                    collected.setdefault(
                        key, []
                    ).append(
                        output[key]
                        .float()
                        .cpu()
                        .numpy()
                    )

            for key in ["labels", "index"]:

                collected.setdefault(
                    key, []
                ).append(
                    batch[key].cpu().numpy()
                )

    result = {
        key: np.concatenate(values)
        for key, values in collected.items()
    }

    result["probabilities"] = softmax(
        result["logits"].astype(np.float64),
        axis=1
    )

    assert np.array_equal(
        result["labels"],
        Y_VAL
    )

    assert np.array_equal(
        result["index"],
        np.arange(N_VAL)
    )

    return result

def optimizer8(model, total_steps):

    base_params = (
        list(model.encoder.parameters())
        + list(model.base_classifier.parameters())
    )

    base_ids = {
        id(p)
        for p in base_params
    }

    head_params = [
        p
        for p in model.parameters()
        if id(p) not in base_ids
    ]

    groups = [{
        "params": base_params,
        "lr": ENCODER_LR8
    }]

    if head_params:

        groups.append({
            "params": head_params,
            "lr": HEAD_LR8
        })

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=.01
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_steps * .10
        ),
        num_training_steps=total_steps
    )

    return optimizer, scheduler

# =============================================================================
# STEP 149 — SIX MODEL TRAINING EXPERIMENTS
# =============================================================================

section("STEP 149 — HDS MODEL TRAINING")

RESULTS8 = []
HISTORY8 = []
OUTPUTS8 = {}

for mode in MODES8:

    prefix = f"{mode}_seed{SEED8}"

    model_name = prefix + "_model.pt"
    pred_name = prefix + "_val.npz"
    meta_name = prefix + "_completed.json"

    run_config = {
        **CONFIG8,
        "mode": mode
    }

    run_hash = run_fingerprint(
        run_config
    )

    # Resume completed runs directly from GitHub.

    if all(
        name in REMOTE
        for name in [
            model_name,
            pred_name,
            meta_name
        ]
    ):

        meta_path = download_asset(
            REMOTE[meta_name],
            AUDIT / meta_name
        )

        meta = json.loads(
            meta_path.read_text()
        )

        assert meta["run_hash"] == run_hash

        pred_path = download_asset(
            REMOTE[pred_name],
            PRED / pred_name
        )

        assert (
            sha256(pred_path)
            == meta["prediction_sha256"]
        )

        remote_matches(
            model_name,
            meta["checkpoint_sha256"]
        )

        with np.load(
            pred_path,
            allow_pickle=False
        ) as z:

            assert np.array_equal(
                z["sample_id"],
                VAL_IDS
            )

            assert np.array_equal(
                z["labels"],
                Y_VAL
            )

            OUTPUTS8[mode] = {
                key: z[key]
                for key in z.files
                if key != "sample_id"
            }

        RESULTS8.append(
            meta["result"]
        )

        HISTORY8.extend(
            meta["history"]
        )

        print(
            mode,
            "RECOVERED — training skipped"
        )

        continue

    model = new_model8(
        mode,
        MATCHED_WIDTH8
    ).to(DEVICE)

    train_loader = make_loader(
        "train",
        training=True,
        difficulty=DIFFICULTY,
        teacher=OOF_PROBS,
        seed=SEED8
    )

    val_loader = make_loader(
        "val"
    )

    optimizer, scheduler = optimizer8(
        model,
        len(train_loader) * EPOCHS8
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True
    )

    best_f1 = -1
    best_epoch = None
    best_output = None

    checkpoint = CKPT / model_name

    run_history = []

    print("\nTRAINING:", mode)

    for epoch in range(
        1,
        EPOCHS8 + 1
    ):

        model.train()

        totals = {
            "loss": 0.,
            "primary": 0.,
            "hierarchy": 0.,
            "contrastive": 0.,
            "prototype": 0.,
            "residual": 0.
        }

        seen = 0

        for batch in train_loader:

            batch = to_device(batch)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                output = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["valid_mask"],
                    batch["ling"]
                )

                loss, parts = objective8(
                    output,
                    batch,
                    mode
                )

            assert torch.isfinite(loss)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.
            )

            previous_scale = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            if scaler.get_scale() >= previous_scale:

                scheduler.step()

            n = len(batch["labels"])

            totals["loss"] += (
                loss.item() * n
            )

            for key, value in parts.items():

                totals[key] += (
                    value.detach().item() * n
                )

            seen += n

        validation = evaluate8(
            model,
            val_loader
        )

        score = metrics(
            Y_VAL,
            validation["probabilities"]
        )

        row = {
            "mode": mode,
            "seed": SEED8,
            "epoch": epoch,
            **{
                f"train_{key}": value / seen
                for key, value in totals.items()
            },
            **{
                f"val_{key}": value
                for key, value in score.items()
            }
        }

        run_history.append(row)

        print(
            f"{mode} | epoch {epoch}/{EPOCHS8}"
            f" | loss={row['train_loss']:.4f}"
            f" | accuracy={score['accuracy']:.5f}"
            f" | macro-F1={score['macro_f1']:.5f}"
        )

        if score["macro_f1"] > best_f1:

            best_f1 = score["macro_f1"]
            best_epoch = epoch
            best_output = validation

            torch.save(
                cpu_state(model),
                checkpoint
            )

    assert best_output is not None

    result = {
        "model": mode,
        "seed": SEED8,
        "best_epoch": best_epoch,
        "parameters": count_parameters8(model),
        **metrics(
            Y_VAL,
            best_output["probabilities"]
        )
    }

    pred_path = PRED / pred_name

    np.savez_compressed(
        pred_path,
        sample_id=VAL_IDS,
        **best_output
    )

    meta = {
        "run_hash": run_hash,
        "config": run_config,
        "result": result,
        "history": run_history,
        "checkpoint_sha256": sha256(checkpoint),
        "prediction_sha256": sha256(pred_path)
    }

    meta_path = AUDIT / meta_name

    save_json(
        meta,
        meta_path
    )

    # Permanent checkpoint backup.
    # Metadata is uploaded last.

    upload_verified(
        checkpoint,
        model_name
    )

    upload_verified(
        pred_path,
        pred_name
    )

    upload_verified(
        meta_path,
        meta_name
    )

    shutil.copy2(
        meta_path,
        BACKUP / meta_name
    )

    RESULTS8.append(result)
    HISTORY8.extend(run_history)

    OUTPUTS8[mode] = best_output

    save_json(
        RESULTS8,
        BACKUP / "completed_model_results.json"
    )

    push_public(
        f"Stage 8A: completed and verified {mode}"
    )

    checkpoint.unlink()

    del (
        model,
        optimizer,
        scheduler,
        scaler,
        validation,
        output,
        loss,
        parts
    )

    clear_gpu()

    print(
        "BEST RESULT:",
        result
    )

RESULT_TABLE8 = save_table(
    RESULTS8,
    "stage8a_03_model_results"
)

HISTORY_TABLE8 = save_table(
    HISTORY8,
    "stage8a_04_training_history"
)

print(
    RESULT_TABLE8.sort_values(
        "macro_f1",
        ascending=False
    ).round(6).to_string(
        index=False
    )
)

# =============================================================================
# STEP 150 — MODEL AND ENSEMBLE COMPARISONS
# =============================================================================

section("STEP 150 — PERFORMANCE COMPARISONS")

PROBS8 = {
    mode: OUTPUTS8[mode]["probabilities"]
    for mode in MODES8
}

PROBS8.update(HISTORICAL)

# Prespecified exploratory ensembles.

PROBS8["hds_family_ensemble"] = np.mean([
    PROBS8["dual_space"],
    PROBS8["dual_space_prototype"],
    PROBS8["hds_full"]
], axis=0)

PROBS8["prototype_plus_hds"] = (
    PROBS8["prototype_family"]
    + PROBS8["hds_full"]
) / 2

PROBS8["prototype_plus_hds_family"] = (
    PROBS8["prototype_family"]
    + PROBS8["hds_family_ensemble"]
) / 2

ALL_RESULTS8 = save_table(
    [
        {
            "model": name,
            **metrics(Y_VAL, probabilities)
        }
        for name, probabilities in PROBS8.items()
    ],
    "stage8a_05_all_comparisons"
).sort_values(
    "macro_f1",
    ascending=False
)

print(
    ALL_RESULTS8.round(6).to_string(
        index=False
    )
)

# =============================================================================
# STEP 151 — CLASSWISE AND HIERARCHICAL EVALUATION
# =============================================================================

section("STEP 151 — CLASSWISE AND HIERARCHICAL RESULTS")

CLASS_ROWS8 = []
SUBGROUP_ROWS8 = []
HIERARCHY_ROWS8 = []
CONFUSION_ROWS8 = []
BRANCH_ROWS8 = []

for name, probabilities in PROBS8.items():

    pred = probabilities.argmax(1)

    report = classification_report(
        Y_VAL,
        pred,
        labels=[0, 1, 2],
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    for label in CLASSES:

        CLASS_ROWS8.append({
            "model": name,
            "class": label,
            **report[label]
        })

    for subgroup, mask in MASKS.items():

        if mask.any():

            SUBGROUP_ROWS8.append({
                "model": name,
                "subgroup": subgroup,
                "n": int(mask.sum()),
                "accuracy": accuracy_score(
                    Y_VAL[mask],
                    pred[mask]
                ),
                "macro_f1": macro_f1(
                    Y_VAL[mask],
                    pred[mask]
                )
            })

    HIERARCHY_ROWS8.append({
        "model": name,
        **hierarchy_diagnostics(
            Y_VAL,
            probabilities
        )
    })

    cm = confusion_matrix(
        Y_VAL,
        pred,
        labels=[0, 1, 2]
    )

    for i in range(3):

        for j in range(3):

            CONFUSION_ROWS8.append({
                "model": name,
                "true_class": CLASSES[i],
                "predicted_class": CLASSES[j],
                "count": int(cm[i, j])
            })

for mode in MODES8:

    output = OUTPUTS8[mode]

    if "hierarchical_logp" not in output:

        continue

    neutral_prob = softmax(
        output["neutral_logits"],
        axis=1
    )

    polar_prob = softmax(
        output["polar_logits"],
        axis=1
    )

    neutral_target = (
        Y_VAL == 1
    ).astype(int)

    polar_mask = Y_VAL != 1

    polar_target = (
        Y_VAL[polar_mask] == 2
    ).astype(int)

    hierarchy_prob = np.exp(
        output["hierarchical_logp"]
    )

    hierarchy_prob /= hierarchy_prob.sum(
        1,
        keepdims=True
    )

    BRANCH_ROWS8.append({
        "model": mode,
        "neutrality_auroc": roc_auc_score(
            neutral_target,
            neutral_prob[:, 1]
        ),
        "neutrality_f1": f1_score(
            neutral_target,
            neutral_prob.argmax(1),
            average="macro"
        ),
        "conditional_polarity_auroc": roc_auc_score(
            polar_target,
            polar_prob[polar_mask, 1]
        ),
        "conditional_polarity_f1": f1_score(
            polar_target,
            polar_prob[polar_mask].argmax(1),
            average="macro"
        ),
        "hierarchy_only_macro_f1": macro_f1(
            Y_VAL,
            hierarchy_prob.argmax(1)
        ),
        "final_macro_f1": macro_f1(
            Y_VAL,
            PROBS8[mode].argmax(1)
        )
    })

CLASSWISE8 = save_table(
    CLASS_ROWS8,
    "stage8a_06_classwise"
)

SUBGROUPS8 = save_table(
    SUBGROUP_ROWS8,
    "stage8a_07_subgroups"
)

HIERARCHY_TABLE8 = save_table(
    HIERARCHY_ROWS8,
    "stage8a_08_hierarchical_errors"
)

save_table(
    CONFUSION_ROWS8,
    "stage8a_09_confusion_matrices"
)

BRANCH_TABLE8 = save_table(
    BRANCH_ROWS8,
    "stage8a_10_branch_performance"
)

print("\nCLASSWISE F1")

print(
    CLASSWISE8.pivot(
        index="class",
        columns="model",
        values="f1-score"
    )[MODES8].round(5).to_string()
)

print("\nHIERARCHICAL BRANCH PERFORMANCE")

print(
    BRANCH_TABLE8.round(5).to_string(
        index=False
    )
)

# =============================================================================
# STEP 152 — PAIRED STATISTICAL ANALYSIS
# =============================================================================

section("STEP 152 — STATISTICAL COMPARISONS")

rng8 = np.random.default_rng(
    SEED8
)

group_indices8 = [
    np.flatnonzero(
        GROUP_VAL == group
    )
    for group in np.unique(GROUP_VAL)
]

BOOT8 = [
    np.concatenate([
        group_indices8[j]
        for j in rng8.integers(
            0,
            len(group_indices8),
            len(group_indices8)
        )
    ])
    for _ in range(2000)
]

def fast_f1_8(y, pred):

    cm = np.bincount(
        3 * y + pred,
        minlength=9
    ).reshape(3, 3)

    tp = np.diag(cm)

    denominator = (
        cm.sum(0)
        + cm.sum(1)
    )

    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros(3),
        where=denominator > 0
    ).mean()

COMPARISONS8 = [
    ("hds_full", "standard"),
    ("hds_full", "matched_flat"),
    ("hds_full", "hierarchy_only"),
    ("hds_full", "dual_space"),
    ("hds_full", "dual_space_prototype"),
    ("hds_full", "prototype_family"),
    ("dual_space", "hierarchy_only"),
    ("dual_space_prototype", "dual_space"),
    ("hds_family_ensemble", "prototype_family"),
    ("prototype_plus_hds", "prototype_family"),
    ("prototype_plus_hds_family", "prototype_family")
]

STATS_ROWS8 = []

for candidate, reference in COMPARISONS8:

    a = PROBS8[candidate].argmax(1)
    b = PROBS8[reference].argmax(1)

    gained = int(
        (
            (a == Y_VAL)
            & (b != Y_VAL)
        ).sum()
    )

    lost = int(
        (
            (a != Y_VAL)
            & (b == Y_VAL)
        ).sum()
    )

    delta = np.array([
        fast_f1_8(
            Y_VAL[idx],
            a[idx]
        )
        - fast_f1_8(
            Y_VAL[idx],
            b[idx]
        )
        for idx in BOOT8
    ])

    pvalue = (
        binomtest(
            gained,
            gained + lost,
            .5
        ).pvalue
        if gained + lost
        else 1.
    )

    STATS_ROWS8.append({
        "candidate": candidate,
        "reference": reference,
        "delta_macro_f1": (
            fast_f1_8(Y_VAL, a)
            - fast_f1_8(Y_VAL, b)
        ),
        "ci95_low": np.quantile(
            delta, .025
        ),
        "ci95_high": np.quantile(
            delta, .975
        ),
        "gained_correct": gained,
        "lost_correct": lost,
        "mcnemar_p": pvalue
    })

pvalues8 = np.array([
    row["mcnemar_p"]
    for row in STATS_ROWS8
])

order8 = np.argsort(
    pvalues8
)

adjusted8 = np.empty(
    len(pvalues8)
)

running8 = 0.

for rank, idx in enumerate(order8):

    running8 = max(
        running8,
        (len(pvalues8) - rank)
        * pvalues8[idx]
    )

    adjusted8[idx] = min(
        running8,
        1.
    )

for i, row in enumerate(STATS_ROWS8):

    row["holm_p"] = adjusted8[i]

STATS8 = save_table(
    STATS_ROWS8,
    "stage8a_11_paired_statistics"
)

print(
    STATS8.round(6).to_string(
        index=False
    )
)

# =============================================================================
# STEP 153 — CALIBRATION AND REPRESENTATION ANALYSIS
# =============================================================================

section("STEP 153 — CALIBRATION AND REPRESENTATIONS")

def ece8(y, probabilities, bins=15):

    confidence = probabilities.max(1)

    correct = (
        probabilities.argmax(1) == y
    )

    edges = np.linspace(
        0,
        1,
        bins + 1
    )

    result = 0.

    for i in range(bins):

        mask = (
            (confidence >= edges[i])
            & (
                confidence < edges[i + 1]
                if i < bins - 1
                else confidence <= edges[i + 1]
            )
        )

        if mask.any():

            result += (
                mask.mean()
                * abs(
                    correct[mask].mean()
                    - confidence[mask].mean()
                )
            )

    return float(result)

CALIBRATION8 = save_table(
    [
        {
            "model": name,
            "ece": ece8(
                Y_VAL,
                probabilities
            ),
            "nll": log_loss(
                Y_VAL,
                probabilities,
                labels=[0, 1, 2]
            ),
            "brier": np.mean(
                np.sum(
                    (
                        probabilities
                        - np.eye(3)[Y_VAL]
                    )**2,
                    axis=1
                )
            )
        }
        for name, probabilities in PROBS8.items()
    ],
    "stage8a_12_calibration"
)

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

REPRESENTATION_ROWS8 = []

for mode in [
    "hierarchy_only",
    "dual_space",
    "dual_space_prototype",
    "hds_full"
]:

    output = OUTPUTS8[mode]

    neutral_embedding = output[
        "neutral_embedding"
    ]

    polar_embedding = output[
        "polar_embedding"
    ]

    neutral_labels = (
        Y_VAL == 1
    ).astype(int)

    polar_mask = Y_VAL != 1

    polar_labels = (
        Y_VAL[polar_mask] == 2
    ).astype(int)

    REPRESENTATION_ROWS8.append({
        "model": mode,
        "neutrality_silhouette": silhouette_score(
            neutral_embedding,
            neutral_labels,
            metric="cosine"
        ),
        "polarity_silhouette": silhouette_score(
            polar_embedding[polar_mask],
            polar_labels,
            metric="cosine"
        ),
        "three_class_neutral_space_silhouette":
            silhouette_score(
                neutral_embedding,
                Y_VAL,
                metric="cosine"
            ),
        "three_class_polar_space_silhouette":
            silhouette_score(
                polar_embedding,
                Y_VAL,
                metric="cosine"
            )
    })

REPRESENTATIONS8 = save_table(
    REPRESENTATION_ROWS8,
    "stage8a_13_representation_separation"
)

print("\nREPRESENTATION SEPARATION")

print(
    REPRESENTATIONS8.round(6).to_string(
        index=False
    )
)

# Complementarity diagnostic.

family_names8 = [
    "standard",
    "matched_flat",
    "hierarchy_only",
    "dual_space",
    "dual_space_prototype",
    "hds_full"
]

pred_matrix8 = np.column_stack([
    PROBS8[name].argmax(1)
    for name in family_names8
])

correct_matrix8 = (
    pred_matrix8
    == Y_VAL[:, None]
)

ORACLE_ROWS8 = [{
    "models": len(family_names8),
    "all_correct": int(
        correct_matrix8.all(1).sum()
    ),
    "all_wrong": int(
        (~correct_matrix8.any(1)).sum()
    ),
    "mixed_correctness": int(
        (
            correct_matrix8.any(1)
            & ~correct_matrix8.all(1)
        ).sum()
    ),
    "oracle_accuracy": float(
        correct_matrix8.any(1).mean()
    )
}]

ORACLE8 = save_table(
    ORACLE_ROWS8,
    "stage8a_14_complementarity"
)

# =============================================================================
# STEP 154 — PUBLICATION-QUALITY FIGURES
# =============================================================================

section("STEP 154 — PUBLICATION FIGURES AND GITHUB BACKUP")

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

FIGURE_MANIFEST8 = []

def save_figure8(fig, name, caption):

    for fmt in ["png", "pdf", "svg"]:

        path = (
            FIGURES / fmt / f"{name}.{fmt}"
        )

        fig.savefig(
            path,
            dpi=600 if fmt == "png" else None
        )

        shutil.copy2(
            path,
            PUB / "figures" / fmt / path.name
        )

        destination = (
            BACKUP / "figures" / fmt / path.name
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            path,
            destination
        )

        FIGURE_MANIFEST8.append({
            "figure": name,
            "format": fmt,
            "sha256": sha256(path),
            "caption": caption
        })

    for directory in [
        PUB / "captions",
        BACKUP / "captions"
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        (directory / f"{name}.txt").write_text(
            caption,
            encoding="utf-8"
        )

    save_json(
        FIGURE_MANIFEST8,
        BACKUP / "figure_manifest.json"
    )

    # Immediate GitHub synchronization.
    # No figure waits until the end of the notebook.

    push_public(
        "Stage 8A figure: " + name
    )

    plt.show()
    plt.close(fig)

# Figure 1 — Overall performance.

fig, ax = plt.subplots(
    figsize=(11, 7)
)

plot = ALL_RESULTS8.sort_values(
    "macro_f1"
)

ax.barh(
    plot["model"],
    100 * plot["macro_f1"]
)

ax.set_xlim(0, 100)
ax.set_xlabel("Validation macro-F1 (%)")

for i, value in enumerate(
    plot["macro_f1"]
):

    ax.text(
        100 * value + .4,
        i,
        f"{100 * value:.2f}",
        va="center",
        fontsize=9
    )

save_figure8(
    fig,
    "stage8a_01_model_comparison",
    "Validation performance of HDS-Net variants, controls and historical models."
)

# Figure 2 — Learning curves.

fig, ax = plt.subplots(
    figsize=(9, 5)
)

for mode in MODES8:

    part = HISTORY_TABLE8[
        HISTORY_TABLE8["mode"] == mode
    ]

    ax.plot(
        part["epoch"],
        part["val_macro_f1"],
        marker="o",
        label=mode
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation macro-F1")
ax.legend(frameon=False, fontsize=9)

save_figure8(
    fig,
    "stage8a_02_learning_curves",
    "Learning curves under the shared five-epoch training protocol."
)

# Figure 3 — Classwise performance.

fig, ax = plt.subplots(
    figsize=(11, 5)
)

CLASSWISE8[
    CLASSWISE8["model"].isin(MODES8)
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

save_figure8(
    fig,
    "stage8a_03_classwise_performance",
    "Negative, neutral and positive sentiment classification performance."
)

# Figure 4 — Neutrality errors.

fig, ax = plt.subplots(
    figsize=(11, 6)
)

HIERARCHY_TABLE8[
    HIERARCHY_TABLE8["model"].isin(
        MODES8 + ["prototype_family"]
    )
].set_index("model")[[
    "neutral_to_negative",
    "neutral_to_positive",
    "negative_to_neutral",
    "positive_to_neutral"
]].plot(
    kind="bar",
    stacked=True,
    ax=ax,
    rot=30
)

ax.set_ylabel("Number of errors")
ax.legend(frameon=False, fontsize=8)

save_figure8(
    fig,
    "stage8a_04_neutrality_errors",
    "Error decomposition around the neutral sentiment boundary."
)

# Figure 5 — Hierarchical branch performance.

fig, ax = plt.subplots(
    figsize=(10, 5)
)

BRANCH_TABLE8.set_index("model")[[
    "neutrality_f1",
    "conditional_polarity_f1",
    "hierarchy_only_macro_f1",
    "final_macro_f1"
]].plot(
    kind="bar",
    ax=ax,
    rot=25
)

ax.set_ylim(0, 1)
ax.set_ylabel("Macro-F1")
ax.legend(frameon=False, fontsize=8)

save_figure8(
    fig,
    "stage8a_05_branch_performance",
    "Neutrality, conditional polarity and final classification performance."
)

# Figure 6 — Linguistic subgroup heatmap.

pivot8 = SUBGROUPS8[
    SUBGROUPS8["model"].isin(
        MODES8 + ["prototype_family"]
    )
].pivot(
    index="subgroup",
    columns="model",
    values="macro_f1"
)

fig, ax = plt.subplots(
    figsize=(12, 6)
)

im = ax.imshow(
    pivot8.to_numpy(),
    cmap="YlGnBu",
    vmin=0,
    vmax=1
)

ax.set_xticks(
    range(len(pivot8.columns)),
    pivot8.columns,
    rotation=35,
    ha="right"
)

ax.set_yticks(
    range(len(pivot8.index)),
    pivot8.index
)

for i in range(len(pivot8.index)):

    for j in range(len(pivot8.columns)):

        value = pivot8.iloc[i, j]

        ax.text(
            j, i,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color=(
                "white"
                if value > .7
                else "black"
            )
        )

fig.colorbar(
    im,
    ax=ax,
    label="Macro-F1"
)

save_figure8(
    fig,
    "stage8a_06_linguistic_subgroups",
    "Performance across linguistic candidate subgroups."
)

# Figure 7 — Paired effects.

fig, ax = plt.subplots(
    figsize=(12, 8)
)

labels8 = []

for i, row in STATS8.iterrows():

    ax.plot(
        [
            100 * row["ci95_low"],
            100 * row["ci95_high"]
        ],
        [i, i],
        linewidth=2
    )

    ax.scatter(
        100 * row["delta_macro_f1"],
        i
    )

    labels8.append(
        row["candidate"]
        + " vs "
        + row["reference"]
    )

ax.axvline(
    0,
    color="gray",
    linestyle="--"
)

ax.set_yticks(
    range(len(labels8)),
    labels8
)

ax.set_xlabel(
    "Macro-F1 difference (percentage points)"
)

save_figure8(
    fig,
    "stage8a_07_paired_effects",
    "Exploratory paired group-bootstrap confidence intervals."
)

# Figure 8 — Neutrality representation PCA.

embedding8 = OUTPUTS8[
    "hds_full"
]["neutral_embedding"]

pca8 = PCA(
    n_components=2,
    random_state=SEED8
)

xy8 = pca8.fit_transform(
    embedding8
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

for label, name in [
    (0, "Non-neutral"),
    (1, "Neutral")
]:

    mask = (
        (Y_VAL == 1).astype(int)
        == label
    )

    ax.scatter(
        xy8[mask, 0],
        xy8[mask, 1],
        s=10,
        alpha=.45,
        label=name
    )

ax.set_xlabel("Principal component 1")
ax.set_ylabel("Principal component 2")
ax.legend(frameon=False)

save_figure8(
    fig,
    "stage8a_08_neutrality_space",
    "Descriptive PCA projection of the learned neutrality representation."
)

# Figure 9 — Conditional polarity PCA.

mask8 = Y_VAL != 1

embedding8 = OUTPUTS8[
    "hds_full"
]["polar_embedding"][mask8]

xy8 = PCA(
    n_components=2,
    random_state=SEED8
).fit_transform(
    embedding8
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

for label, name in [
    (0, "Negative"),
    (2, "Positive")
]:

    selected = Y_VAL[mask8] == label

    ax.scatter(
        xy8[selected, 0],
        xy8[selected, 1],
        s=10,
        alpha=.45,
        label=name
    )

ax.set_xlabel("Principal component 1")
ax.set_ylabel("Principal component 2")
ax.legend(frameon=False)

save_figure8(
    fig,
    "stage8a_09_polarity_space",
    "Descriptive PCA projection of the conditional polarity representation."
)

# Figure 10 — Representation separation.

fig, ax = plt.subplots(
    figsize=(10, 5)
)

REPRESENTATIONS8.set_index("model")[[
    "neutrality_silhouette",
    "polarity_silhouette"
]].plot(
    kind="bar",
    ax=ax,
    rot=25
)

ax.axhline(
    0,
    color="gray",
    linewidth=1
)

ax.set_ylabel("Cosine silhouette score")
ax.legend(frameon=False)

save_figure8(
    fig,
    "stage8a_10_representation_separation",
    "Representation-space separation for the two classification tasks."
)

# Figure 11 — Confusion matrix.

fig, ax = plt.subplots(
    figsize=(6, 5)
)

cm8 = confusion_matrix(
    Y_VAL,
    PROBS8["hds_full"].argmax(1),
    labels=[0, 1, 2]
)

im = ax.imshow(
    cm8,
    cmap="Blues"
)

ax.set_xticks(
    range(3),
    CLASSES
)

ax.set_yticks(
    range(3),
    CLASSES
)

ax.set_xlabel("Predicted sentiment")
ax.set_ylabel("Recorded sentiment")

for i in range(3):

    for j in range(3):

        ax.text(
            j, i,
            str(cm8[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm8[i, j] > cm8.max() / 2
                else "black"
            )
        )

fig.colorbar(im, ax=ax)

save_figure8(
    fig,
    "stage8a_11_confusion_matrix",
    "HDS-Net validation confusion matrix."
)

# Figure 12 — Calibration.

fig, ax = plt.subplots(
    figsize=(10, 6)
)

plot8 = CALIBRATION8.sort_values(
    "ece",
    ascending=False
)

ax.barh(
    plot8["model"],
    plot8["ece"]
)

ax.set_xlabel(
    "Expected calibration error"
)

save_figure8(
    fig,
    "stage8a_12_calibration",
    "Uncalibrated expected calibration error across compared models."
)

# Figure 13 — Difficulty decomposition.

fig, ax = plt.subplots(
    figsize=(8, 5)
)

DIFFICULTY_TABLE8.set_index("class")[[
    "general_difficulty",
    "neutrality_difficulty",
    "conditional_polarity_difficulty"
]].plot(
    kind="bar",
    ax=ax,
    rot=0
)

ax.set_ylabel("Mean OOF-derived difficulty")
ax.legend(frameon=False, fontsize=8)

save_figure8(
    fig,
    "stage8a_13_hierarchical_difficulty",
    "Training-only difficulty decomposition by recorded sentiment class."
)

# Figure 14 — Complete architecture diagram.

from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch
)

fig, ax = plt.subplots(
    figsize=(14, 11)
)

ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis("off")

def box8(
    x, y, w, h,
    text,
    color="#f1f5f9"
):

    patch = FancyBboxPatch(
        (x, y),
        w, h,
        boxstyle="round,pad=.04",
        facecolor=color,
        edgecolor="#334155",
        linewidth=1.4
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10
    )

def arrow8(
    start,
    end,
    dashed=False
):

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color="#475569",
            linestyle="--" if dashed else "-"
        )
    )

box8(
    5, 10.9, 4, .65,
    "Bangla text"
)

box8(
    4.5, 9.65, 5, .8,
    "BanglaBERT encoder",
    "#dbeafe"
)

box8(
    4.3, 8.3, 5.4, .8,
    "CLS + masked mean + attention pooling"
)

box8(
    .5, 6.6, 5.6, 1,
    "Neutrality representation space\nNeutral vs. non-neutral",
    "#ccfbf1"
)

box8(
    7.9, 6.6, 5.6, 1,
    "Conditional polarity space\nNegative vs. positive",
    "#ede9fe"
)

box8(
    .5, 5.05, 5.6, .9,
    "Neutrality multi-prototype classifier",
    "#ccfbf1"
)

box8(
    7.9, 5.05, 5.6, .9,
    "Polarity multi-prototype classifier",
    "#ede9fe"
)

box8(
    3.9, 3.5, 6.2, .9,
    "Hierarchical probability composition",
    "#fef3c7"
)

box8(
    .3, 2.0, 3.3, 1,
    "Direct semantic\nresidual classifier",
    "#dbeafe"
)

box8(
    4.7, 1.9, 4.6, .9,
    "Fixed probability fusion\n65% hierarchy + 35% residual",
    "#fef3c7"
)

box8(
    4.7, .45, 4.6, .75,
    "Negative / Neutral / Positive"
)

box8(
    10.8, 2.0, 2.9, 1.5,
    "Training only\nOOF difficulty\nContrastive loss\nPrototype margin"
)

arrow8((7, 10.9), (7, 10.45))
arrow8((7, 9.65), (7, 9.1))

arrow8((7, 8.3), (3.3, 7.6))
arrow8((7, 8.3), (10.7, 7.6))

arrow8((3.3, 6.6), (3.3, 5.95))
arrow8((10.7, 6.6), (10.7, 5.95))

arrow8((3.3, 5.05), (5.5, 4.4))
arrow8((10.7, 5.05), (8.5, 4.4))

arrow8((7, 3.5), (7, 2.8))
arrow8((3.6, 2.5), (4.7, 2.5))
arrow8((7, 1.9), (7, 1.2))

# Training-only supervision.

arrow8(
    (12.2, 3.5),
    (11.5, 5.05),
    dashed=True
)

arrow8(
    (10.8, 3.0),
    (9.8, 3.9),
    dashed=True
)

# Direct semantic path, with pooled dual-space features.

ax.plot(
    [4.5, .15, .15, .3],
    [10.05, 10.05, 2.5, 2.5],
    color="#475569",
    linewidth=1.2
)

arrow8(
    (.15, 2.5),
    (.3, 2.5)
)

ax.set_title(
    "Hierarchical Dual-Space Sentiment Network (HDS-Net)",
    fontsize=16,
    fontweight="bold"
)

save_figure8(
    fig,
    "stage8a_14_hds_architecture",
    "Provisional HDS-Net architecture with hierarchical classification and training-only boundary supervision."
)

for fmt in ["png", "pdf", "svg"]:

    name = f"stage8a_14_hds_architecture.{fmt}"

    shutil.copy2(
        FIGURES / fmt / name,
        PUB / "architecture" / name
    )

# =============================================================================
# STEP 155 — FINAL SYNTHESIS AND VERIFIED PERMANENT BACKUP
# =============================================================================

section("STEP 155 — FINAL RESEARCH SYNTHESIS")

scores8 = {
    name: macro_f1(
        Y_VAL,
        probabilities.argmax(1)
    )
    for name, probabilities in PROBS8.items()
}

full8 = scores8["hds_full"]

standard8 = scores8["standard"]
matched8 = scores8["matched_flat"]

historical8 = scores8[
    "prototype_family"
]

vs_standard8 = STATS8[
    (STATS8["candidate"] == "hds_full")
    & (STATS8["reference"] == "standard")
].iloc[0]

vs_matched8 = STATS8[
    (STATS8["candidate"] == "hds_full")
    & (STATS8["reference"] == "matched_flat")
].iloc[0]

neutral_full8 = CLASSWISE8[
    (CLASSWISE8["model"] == "hds_full")
    & (CLASSWISE8["class"] == "Neutral")
]["f1-score"].iloc[0]

neutral_standard8 = CLASSWISE8[
    (CLASSWISE8["model"] == "standard")
    & (CLASSWISE8["class"] == "Neutral")
]["f1-score"].iloc[0]

# Stricter decision rule than Stage 7DE.
# Tiny positive differences do not count as success.

if (
    full8 >= .85
    and full8 > historical8
    and full8 - standard8 >= .02
    and full8 - matched8 >= .01
    and vs_standard8["ci95_low"] > 0
    and vs_matched8["ci95_low"] > 0
):

    DECISION8 = (
        "STRONG SCREENING RESULT — "
        "MULTI-SEED CONFIRMATION REQUIRED"
    )

elif (
    full8 - standard8 >= .01
    and full8 - matched8 >= .005
    and neutral_full8 > neutral_standard8
):

    DECISION8 = (
        "MEANINGFUL SCREENING IMPROVEMENT — "
        "INVESTIGATE AND CONFIRM"
    )

else:

    DECISION8 = (
        "INSUFFICIENT IMPROVEMENT — "
        "DO NOT FINALIZE ARCHITECTURE"
    )

SUMMARY8 = {
    "stage": "8A",
    "architecture": "HDS-Net",
    "training_samples": N_TRAIN,
    "validation_samples": N_VAL,
    "training_seeds": [SEED8],
    "completed_models": len(RESULTS8),
    "scores": scores8,
    "hds_full_macro_f1": full8,
    "delta_vs_standard": full8 - standard8,
    "delta_vs_matched": full8 - matched8,
    "delta_vs_previous_best": full8 - historical8,
    "neutral_f1_full": neutral_full8,
    "neutral_f1_standard": neutral_standard8,
    "neutral_f1_improvement":
        neutral_full8 - neutral_standard8,
    "best_individual_stage8a_model": max(
        MODES8,
        key=lambda name: scores8[name]
    ),
    "best_overall_model": max(
        scores8,
        key=scores8.get
    ),
    "best_overall_macro_f1": max(
        scores8.values()
    ),
    "matched_parameter_difference": MISMATCH8,
    "new_figures": len(
        set(
            record["figure"]
            for record in FIGURE_MANIFEST8
        )
    ),
    "new_tables": len(
        list(TABLES.glob("*.csv"))
    ),
    "decision": DECISION8,
    "test_evaluated": False,
    "external_evaluated": False,
    "novelty_verified": False,
    "limitations": [
        "Single-seed exploratory screening.",
        "Repeated use of the same validation set.",
        "Hierarchy models use additional auxiliary losses.",
        "Full model combines contrastive learning and difficulty weighting.",
        "Separate ablations are needed to isolate these two mechanisms.",
        "Fixed fusion coefficient was chosen before screening.",
        "Neutral sentiment is not necessarily equivalent to objective language.",
        "Historical ensembles have a different inference budget.",
        "No independent external evaluation.",
        "Architectural novelty remains unverified."
    ]
}

save_json(
    SUMMARY8,
    AUDIT / "stage8a_summary.json"
)

save_json(
    SUMMARY8,
    BACKUP / "stage8a_summary.json"
)

# Save aggregate publication tables publicly.

for path in TABLES.glob("*.csv"):

    destination = (
        BACKUP / "tables" / path.name
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        path,
        destination
    )

    tex = (
        PUB / "tables/latex" /
        (path.stem + ".tex")
    )

    if tex.exists():

        destination = (
            BACKUP / "latex_tables" / tex.name
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            tex,
            destination
        )

# Preserve tokenizer and backbone configuration.

TOKENIZER.save_pretrained(
    AUDIT / "tokenizer"
)

AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=3
).save_pretrained(
    AUDIT / "backbone_config"
)

# Preserve all aligned predictions privately.

np.savez_compressed(
    PRED / "stage8a_all_validation_predictions.npz",
    sample_id=VAL_IDS,
    labels=Y_VAL,
    group_id=GROUP_VAL.astype(str),
    **PROBS8
)

# Preserve hierarchical difficulty privately.

np.savez_compressed(
    PRED / "stage8a_training_difficulty.npz",
    sample_id=TRAIN_IDS,
    labels=Y_TRAIN,
    general_difficulty=DIFFICULTY,
    neutrality_difficulty=NEUTRAL_DIFFICULTY8,
    polarity_difficulty=POLAR_DIFFICULTY8
)

pd.DataFrame(
    FIGURE_MANIFEST8
).to_csv(
    PUB / "stage8a_figure_manifest.csv",
    index=False
)

# Complete cumulative publication figure manifest.

CUMULATIVE_FIGURES8 = []

for fmt in ["png", "pdf", "svg"]:

    for path in sorted(
        (PUB / "figures" / fmt).rglob(
            f"*.{fmt}"
        )
    ):

        CUMULATIVE_FIGURES8.append({
            "figure": path.stem,
            "format": fmt,
            "relative_path": str(
                path.relative_to(PUB)
            ),
            "sha256": sha256(path),
            "bytes": path.stat().st_size
        })

pd.DataFrame(
    CUMULATIVE_FIGURES8
).to_csv(
    PUB / "cumulative_figure_manifest.csv",
    index=False
)

# Compact research package excludes large model weights.
# All model weights have already been uploaded individually.

compact8 = (
    WORK / "sentiment_stage8a_compact.zip"
)

with zipfile.ZipFile(
    compact8,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for directory in [
        AUDIT,
        TABLES,
        PRED,
        FIGURES
    ]:

        for path in directory.rglob("*"):

            if path.is_file():

                z.write(
                    path,
                    path.relative_to(OUT)
                )

publication8 = (
    WORK / "sentiment_stage8a_publication.zip"
)

with zipfile.ZipFile(
    publication8,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for path in PUB.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(PUB)
            )

for path in [
    compact8,
    publication8
]:

    with zipfile.ZipFile(path) as z:

        assert z.testzip() is None

    upload_verified(
        path,
        f"{path.stem}_{sha256(path)[:12]}.zip"
    )

# Verify all six model checkpoints and prediction files.

for mode in MODES8:

    prefix = f"{mode}_seed{SEED8}"

    meta_path = download_asset(
        REMOTE[prefix + "_completed.json"],
        AUDIT / (prefix + "_completed.json")
    )

    meta = json.loads(
        meta_path.read_text()
    )

    remote_matches(
        prefix + "_model.pt",
        meta["checkpoint_sha256"]
    )

    remote_matches(
        prefix + "_val.npz",
        meta["prediction_sha256"]
    )

# Remote artifact manifest.

all_remote8 = assets_of(
    RELEASE
)

manifest8 = {
    "release": STAGE_TAG,
    "assets": [
        {
            "name": name,
            "asset_id": asset["id"],
            "size": asset["size"],
            "digest": asset.get("digest")
        }
        for name, asset in all_remote8.items()
    ],
    "verified_this_session": REMOTE_MANIFEST
}

manifest_path8 = (
    AUDIT / "stage8a_remote_manifest.json"
)

save_json(
    manifest8,
    manifest_path8
)

upload_verified(
    manifest_path8,
    f"stage8a_manifest_{sha256(manifest_path8)[:12]}.json"
)

shutil.copy2(
    manifest_path8,
    BACKUP / manifest_path8.name
)

FINAL_COMMIT8 = push_public(
    "Stage 8A: complete HDS screening and permanent artifact backup"
)

# Verify every new figure against tracked GitHub content.

tracked8 = set(
    git(
        "ls-tree",
        "-r",
        "--name-only",
        "HEAD"
    ).splitlines()
)

for record in FIGURE_MANIFEST8:

    relative = (
        "research_stage8a/figures/"
        + record["format"]
        + "/"
        + record["figure"]
        + "."
        + record["format"]
    )

    assert relative in tracked8

    assert sha256(
        REPO / relative
    ) == record["sha256"]

# =============================================================================
# FINAL EXECUTION REPORT
# =============================================================================

section("FINAL STAGE 8A RESEARCH REPORT")

print("\nOOF HIERARCHICAL BOTTLENECKS")

print(
    json.dumps(
        OOF_HIERARCHY8,
        indent=2
    )
)

print("\nTRAINING DIFFICULTY")

print(
    DIFFICULTY_TABLE8.round(5).to_string(
        index=False
    )
)

print("\nARCHITECTURE AUDIT")

print(
    ARCHITECTURE_TABLE8.to_string(
        index=False
    )
)

print("\nMODEL RESULTS")

print(
    RESULT_TABLE8.round(6).to_string(
        index=False
    )
)

print("\nALL MODEL COMPARISONS")

print(
    ALL_RESULTS8.round(6).to_string(
        index=False
    )
)

print("\nCLASSWISE PERFORMANCE")

print(
    CLASSWISE8.pivot(
        index="class",
        columns="model",
        values="f1-score"
    )[
        MODES8 + ["prototype_family"]
    ].round(5).to_string()
)

print("\nHIERARCHICAL ERROR DECOMPOSITION")

print(
    HIERARCHY_TABLE8[
        HIERARCHY_TABLE8["model"].isin(
            MODES8 + ["prototype_family"]
        )
    ].round(5).to_string(
        index=False
    )
)

print("\nHIERARCHICAL BRANCH PERFORMANCE")

print(
    BRANCH_TABLE8.round(5).to_string(
        index=False
    )
)

print("\nLINGUISTIC SUBGROUPS")

print(
    SUBGROUPS8.pivot(
        index="subgroup",
        columns="model",
        values="macro_f1"
    )[
        MODES8 + ["prototype_family"]
    ].round(5).to_string()
)

print("\nPAIRED STATISTICS")

print(
    STATS8.round(6).to_string(
        index=False
    )
)

print("\nREPRESENTATION SEPARATION")

print(
    REPRESENTATIONS8.round(6).to_string(
        index=False
    )
)

print("\nCOMPLEMENTARITY")

print(
    ORACLE8.round(6).to_string(
        index=False
    )
)

print("\nCALIBRATION")

print(
    CALIBRATION8.round(6).to_string(
        index=False
    )
)

print("\nRESEARCH SUMMARY")

print(
    json.dumps(
        SUMMARY8,
        indent=2,
        default=str
    )
)

print("\nPERMANENT PUBLICATION ASSETS")

for fmt in ["png", "pdf", "svg"]:

    print(
        fmt.upper(),
        "figures:",
        len(list(
            (PUB / "figures" / fmt).rglob(
                f"*.{fmt}"
            )
        ))
    )

print("\nAll new figures committed and verified: YES")
print("All six model checkpoints backed up: YES")
print("Private predictions backed up: YES")
print("GitHub synchronized: True")
print("Final public commit:", FINAL_COMMIT8)
print("Private release:", RELEASE["html_url"])

print("Manual saving required: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("Test evaluated: NO")

print(
    "\nRuntime minutes:",
    round(
        (time.time() - STAGE8_START) / 60,
        2
    )
)

print("\nRESEARCH DECISION:", DECISION8)
print("\nSTAGE 8A COMPLETE")

clear_gpu()

# =====================================================================
# STAGE 6D — ACCELERATED ARCHITECTURE REFINEMENT
# STEPS 80–89
# GitHub-first recovery | Fresh pretrained initialization
# Two-seed screening | Automatic permanent backups
# =====================================================================

import os
import re
import gc
import sys
import json
import time
import math
import hashlib
import getpass
import zipfile
import subprocess
from pathlib import Path

import requests
import numpy as np
import pandas as pd

STAGE6D_START = time.time()

WORK6D = Path("/kaggle/working")

OWNER6D = "ankita6876"
REPO6D = "bangla-sentiment-research"
BRANCH6D = "research/reconstruction-20261127-080232"

COMMIT6C = "17df059f688b5e0fb609a6f270ae2c2b3c9c3995"

ROOT6D = WORK6D / "sentiment_research_stage6d"
ROOT6D.mkdir(parents=True, exist_ok=True)

SOURCE6C = ROOT6D / "recovered_stage6c.py"

print("=" * 85)
print("STEP 80 — RECOVER VERIFIED IMPLEMENTATION")
print("=" * 85)

source_url = (
    f"https://raw.githubusercontent.com/"
    f"{OWNER6D}/{REPO6D}/{COMMIT6C}/"
    "research_stage6c/stage6c_complete.py"
)

response = requests.get(
    source_url,
    timeout=120
)

response.raise_for_status()

source6c = response.text

assert "class AMSCNet" in source6c
assert "class MultiScalePhraseBranch" in source6c
assert "class DirectionalCompositionBranch" in source6c
assert "STEP 74" in source6c

SOURCE6C.write_text(
    source6c,
    encoding="utf-8"
)

print("Verified Stage 6C source recovered.")
print("Source commit:", COMMIT6C)

# Recover initialization, GitHub authentication,
# exact data membership, tokenization and architecture.
# Stop before the original Stage 6C training protocol.

stop_marker = (
    'section("STEP 74 — ARCHITECTURE VALIDATION")'
)

assert stop_marker in source6c

recovery_source = source6c.split(
    stop_marker, 1
)[0]

exec(
    compile(
        recovery_source,
        str(SOURCE6C),
        "exec"
    ),
    globals()
)

# =====================================================================
# STEP 81 — LOCK NEW EXPERIMENT PROTOCOL
# =====================================================================

section("STEP 81 — FRESH INITIALIZATION PROTOCOL")

S6D = ROOT6D

TABLES = S6D / "tables"
CHECKPOINTS = S6D / "checkpoints"
PRED = S6D / "private_predictions"
AUDIT = S6D / "audit"

BACKUP = REPO / "research_stage6d"

for folder in [
    TABLES,
    CHECKPOINTS,
    PRED,
    AUDIT,
    BACKUP
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

STAGE_TAG = "stage6d-amsc-refinement-v1"

PRETRAINED_NAME = "csebuetnlp/banglabert"

SEEDS_D = [42, 123]

MODES_D = [
    "standard",
    "fixed_fusion",
    "normalized_fusion"
]

EPOCHS_D = 4
TRAIN_BATCH_D = 32
EVAL_BATCH_D = 64

ENCODER_LR_D = 2e-5
HEAD_LR_D = 2e-4

AUX_WEIGHT_D = 0.10
WARMUP_D = 0.10
WEIGHT_DECAY_D = 0.01

BOOTSTRAPS_D = 2000

# All experiments use the same locked
# seed-42 dataset membership.
# Only training randomness changes.

CONFIG_D = {
    "stage": "6d",
    "source_stage6c_commit": COMMIT6C,
    "pretrained_model": PRETRAINED_NAME,
    "dataset_split_seed": 42,
    "training_seeds": SEEDS_D,
    "architectures": MODES_D,
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

# Preserve this entire executable cell.

history_d = (
    get_ipython()
    .history_manager
    .input_hist_raw
)

source_candidates_d = [
    c for c in history_d
    if "STAGE 6D — ACCELERATED ARCHITECTURE REFINEMENT" in c
    and len(c) > 5000
]

assert source_candidates_d

source_d = source_candidates_d[-1]

assert not re.search(
    r"(?:ghp_|github_pat_|gho_|ghs_)"
    r"[A-Za-z0-9_]{20,}",
    source_d
)

(AUDIT / "stage6d_complete.py").write_text(
    source_d,
    encoding="utf-8"
)

shutil.copy2(
    AUDIT / "stage6d_complete.py",
    BACKUP / "stage6d_complete.py"
)

shutil.copy2(
    SOURCE6C,
    BACKUP / "recovered_stage6c.py"
)

def push_d(message):

    git("add", "research_stage6d")

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

push_d(
    "Lock Stage 6D fresh initialization protocol"
)

# =====================================================================
# STEP 82 — PRIVATE STORAGE AND METRIC UTILITIES
# =====================================================================

section("STEP 82 — PERMANENT ARTIFACT STORAGE")

# Reuse verified upload and integrity functions
# from the successful Stage 6C implementation.

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

exec(storage_source, globals())

# Reuse metric functions without executing
# any original training or evaluation.

metric_source = (
    source6c.split("def metrics(y, logits):", 1)[1]
    .split("def objective(output, labels):", 1)[0]
)

exec(
    "def metrics(y, logits):" + metric_source,
    globals()
)

# =====================================================================
# STEP 83 — REFINED MODEL ARCHITECTURE
# =====================================================================

section("STEP 83 — AMSC-NET V2")

class AMSCNetV2(nn.Module):

    def __init__(self, base_model, mode):

        super().__init__()

        self.mode = mode

        self.encoder = base_model.electra

        self.base_classifier = (
            base_model.classifier
        )

        hidden = (
            base_model.config.hidden_size
        )

        if mode == "standard":

            return

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

        self.global_delta = nn.Linear(
            DIM, 3
        )

        self.phrase = (
            MultiScalePhraseBranch(DIM)
        )

        self.directional = (
            DirectionalCompositionBranch(DIM)
        )

        self.phrase_head = nn.Linear(
            DIM, 3
        )

        self.directional_head = nn.Linear(
            DIM, 3
        )

        if mode == "normalized_fusion":

            self.fusion = nn.Sequential(
                nn.Linear(3 * DIM + 5, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(64, 3)
            )

            nn.init.zeros_(
                self.fusion[-1].weight
            )

            with torch.no_grad():

                self.fusion[-1].bias.copy_(
                    torch.log(
                        torch.tensor([
                            0.70, 0.15, 0.15
                        ])
                    )
                )

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

        if self.mode == "standard":

            weights = torch.zeros(
                len(hidden), 3,
                device=hidden.device
            )

            weights[:, 0] = 1

            return {
                "logits": base_logits,
                "branches": None,
                "weights": weights
            }

        x = self.project(hidden)

        pooled = masked_mean(
            x, valid_mask
        )

        g = self.global_repr(
            torch.cat([
                x[:, 0],
                pooled
            ], dim=-1)
        )

        p = self.phrase(
            x, valid_mask
        )

        d = self.directional(
            x, valid_mask
        )

        zg = (
            base_logits
            + self.global_delta(g)
        )

        zp = self.phrase_head(p)
        zd = self.directional_head(d)

        branch_logits = torch.stack(
            [zg, zp, zd],
            dim=1
        )

        if self.mode == "fixed_fusion":

            weights = torch.tensor(
                [0.70, 0.15, 0.15],
                device=x.device,
                dtype=zg.dtype
            ).expand(len(x), -1)

        else:

            probs = torch.softmax(
                zg.detach().float(),
                dim=-1
            )

            entropy = (
                -(
                    probs
                    * probs.clamp_min(1e-8).log()
                ).sum(
                    dim=-1,
                    keepdim=True
                ) / math.log(3)
            )

            top2 = probs.topk(
                2, dim=-1
            ).values

            uncertainty = (
                1 - top2[:, :1]
                + top2[:, 1:2]
            )

            confidence_uncertainty = (
                1 - probs.max(
                    dim=-1,
                    keepdim=True
                ).values
            )

            dp = js_divergence(
                zg.detach(),
                zp.detach()
            )

            dd = js_divergence(
                zg.detach(),
                zd.detach()
            )

            fusion_input = torch.cat([
                g, p, d,
                entropy.to(g.dtype),
                uncertainty.to(g.dtype),
                confidence_uncertainty.to(g.dtype),
                dp.to(g.dtype),
                dd.to(g.dtype)
            ], dim=-1)

            raw_weights = torch.softmax(
                self.fusion(
                    fusion_input
                ).float(),
                dim=-1
            )

            # A small floor prevents complete
            # elimination of any branch.

            weights = (
                0.94 * raw_weights + 0.02
            ).to(zg.dtype)

        final_logits = (
            branch_logits
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        return {
            "logits": final_logits,
            "branches": branch_logits,
            "weights": weights
        }

def objective_d(output, labels):

    loss = F.cross_entropy(
        output["logits"],
        labels
    )

    if output["branches"] is not None:

        branch_losses = [

            F.cross_entropy(
                output["branches"][:, i],
                labels
            )

            for i in range(3)
        ]

        loss = (
            loss
            + AUX_WEIGHT_D
            * torch.stack(
                branch_losses
            ).mean()
        )

    return loss

def make_model_d(mode, seed):

    seed_all(seed)

    base = (
        AutoModelForSequenceClassification
        .from_pretrained(
            PRETRAINED_NAME,
            num_labels=3
        )
    )

    return AMSCNetV2(
        base,
        mode
    )

@torch.inference_mode()
def evaluate_d(model, loader):

    model.eval()

    logits = []
    labels = []
    weights = []

    for batch in loader:

        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["valid_mask"]
            )

        logits.append(
            output["logits"]
            .float().cpu().numpy()
        )

        labels.append(
            batch["labels"]
            .cpu().numpy()
        )

        weights.append(
            output["weights"]
            .float().cpu().numpy()
        )

    return (
        np.concatenate(logits),
        np.concatenate(labels),
        np.concatenate(weights)
    )

# =====================================================================
# STEP 84 — ARCHITECTURE SMOKE TEST
# =====================================================================

section("STEP 84 — ARCHITECTURE SMOKE TEST")

smoke_d = make_model_d(
    "normalized_fusion", 42
).to(DEVICE)

_, val_loader_d = loaders(42)

batch_d = next(
    iter(val_loader_d)
)

batch_d = {
    k: v.to(DEVICE)
    for k, v in batch_d.items()
}

smoke_d.eval()

with torch.no_grad():

    smoke_output = smoke_d(
        batch_d["input_ids"],
        batch_d["attention_mask"],
        batch_d["valid_mask"]
    )

assert torch.isfinite(
    smoke_output["logits"]
).all()

assert torch.allclose(
    smoke_output["weights"].sum(dim=1),
    torch.ones(
        len(batch_d["labels"]),
        device=DEVICE
    ),
    atol=1e-5
)

print(
    "Initial fusion weights:",
    smoke_output["weights"]
    .mean(dim=0)
    .cpu().numpy()
)

smoke_d.train()

smoke_output = smoke_d(
    batch_d["input_ids"],
    batch_d["attention_mask"],
    batch_d["valid_mask"]
)

smoke_loss = objective_d(
    smoke_output,
    batch_d["labels"]
)

smoke_loss.backward()

for name, module in [
    ("encoder", smoke_d.encoder),
    ("phrase", smoke_d.phrase),
    ("directional", smoke_d.directional),
    ("fusion", smoke_d.fusion)
]:

    gradients = [
        p.grad
        for p in module.parameters()
        if p.grad is not None
    ]

    assert gradients

    assert all(
        torch.isfinite(g).all()
        for g in gradients
    )

    print(name, "gradient flow: PASS")

del smoke_d, smoke_output
del smoke_loss, batch_d, gradients

gc.collect()
torch.cuda.empty_cache()

# =====================================================================
# STEP 85 — SIX COMPLETE TRAINING RUNS
# =====================================================================

section("STEP 85 — MULTI-SEED ARCHITECTURAL SCREENING")

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

for seed in SEEDS_D:

    for mode in MODES_D:

        run_name = (
            f"{mode}_seed{seed}"
        )

        section(
            f"TRAINING — {run_name}"
        )

        run_dir = (
            CHECKPOINTS / run_name
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        ckpt = run_dir / "model.pt"

        meta = (
            run_dir / "completed.json"
        )

        pred = (
            PRED / f"{run_name}_val.npz"
        )

        checkpoint_asset = (
            f"{run_name}_model.pt"
        )

        prediction_asset = (
            f"{run_name}_val.npz"
        )

        metadata_asset = (
            f"{run_name}_completed.json"
        )

        run_config = {
            **CONFIG_D,
            "mode": mode,
            "seed": seed
        }

        run_hash = hashlib.sha256(
            json.dumps(
                run_config,
                sort_keys=True
            ).encode()
        ).hexdigest()

        assets = remote_assets()

        remote_complete = all(
            name in assets
            for name in [
                checkpoint_asset,
                prediction_asset,
                metadata_asset
            ]
        )

        if remote_complete:

            download_asset(
                assets[metadata_asset],
                meta
            )

            completed = json.loads(
                meta.read_text()
            )

            assert (
                completed["run_hash"]
                == run_hash
            )

            download_asset(
                assets[prediction_asset],
                pred,
                completed["prediction_sha256"]
            )

            verify_remote_asset(
                assets[checkpoint_asset],
                completed["checkpoint_sha256"]
            )

            print(
                "Completed run recovered "
                "from GitHub."
            )

        else:

            local_complete = False

            if (
                ckpt.exists()
                and meta.exists()
                and pred.exists()
            ):

                completed = json.loads(
                    meta.read_text()
                )

                local_complete = (
                    completed["run_hash"]
                    == run_hash
                    and sha256(ckpt)
                    == completed[
                        "checkpoint_sha256"
                    ]
                    and sha256(pred)
                    == completed[
                        "prediction_sha256"
                    ]
                )

            if not local_complete:

                model = make_model_d(
                    mode, seed
                ).to(DEVICE)

                train_loader, val_loader = (
                    loaders(seed)
                )

                # The original classifier and
                # encoder use the same LR.
                # New architectural heads use
                # the higher head LR.

                original = []
                added = []

                for name, parameter in (
                    model.named_parameters()
                ):

                    if name.startswith(
                        ("encoder.",
                         "base_classifier.")
                    ):

                        original.append(
                            parameter
                        )

                    else:

                        added.append(
                            parameter
                        )

                groups = [{
                    "params": original,
                    "lr": ENCODER_LR_D
                }]

                if added:

                    groups.append({
                        "params": added,
                        "lr": HEAD_LR_D
                    })

                optimizer = (
                    torch.optim.AdamW(
                        groups,
                        weight_decay=WEIGHT_DECAY_D
                    )
                )

                steps = (
                    len(train_loader)
                    * EPOCHS_D
                )

                scheduler = (
                    get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=int(
                            steps * WARMUP_D
                        ),
                        num_training_steps=steps
                    )
                )

                scaler = torch.amp.GradScaler(
                    "cuda"
                )

                best_f1 = -1
                best_epoch = 0
                run_history = []

                run_start = time.time()

                for epoch in range(
                    1, EPOCHS_D + 1
                ):

                    model.train()

                    loss_sum = 0.0
                    count = 0

                    for batch in train_loader:

                        batch = {
                            k: v.to(DEVICE)
                            for k, v
                            in batch.items()
                        }

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
                                batch["valid_mask"]
                            )

                            loss = objective_d(
                                output,
                                batch["labels"]
                            )

                        assert torch.isfinite(
                            loss
                        )

                        scaler.scale(
                            loss
                        ).backward()

                        scaler.unscale_(
                            optimizer
                        )

                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            1.0
                        )

                        old_scale = (
                            scaler.get_scale()
                        )

                        scaler.step(optimizer)
                        scaler.update()

                        if (
                            scaler.get_scale()
                            >= old_scale
                        ):

                            scheduler.step()

                        n = len(
                            batch["labels"]
                        )

                        loss_sum += (
                            loss.item() * n
                        )

                        count += n

                    logits, y, weights = (
                        evaluate_d(
                            model,
                            val_loader
                        )
                    )

                    assert np.array_equal(
                        y, Y_VAL
                    )

                    scores = metrics(
                        y, logits
                    )

                    row = {
                        "model": mode,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss":
                            loss_sum / count,
                        **scores
                    }

                    run_history.append(row)

                    print(
                        f"Epoch {epoch}/{EPOCHS_D}"
                        f" | Loss {loss_sum/count:.4f}"
                        f" | Acc {scores['accuracy']:.4f}"
                        f" | F1 {scores['macro_f1']:.4f}"
                    )

                    if (
                        scores["macro_f1"]
                        > best_f1
                    ):

                        best_f1 = (
                            scores["macro_f1"]
                        )

                        best_epoch = epoch

                        torch.save(
                            {
                                k: v.detach().cpu()
                                for k, v
                                in model.state_dict().items()
                            },
                            ckpt
                        )

                        np.savez_compressed(
                            pred,
                            sample_id=sample_ids_d,
                            labels=Y_VAL,
                            logits=logits,
                            weights=weights
                        )

                        print(
                            "Best checkpoint saved."
                        )

                saved = np.load(
                    pred,
                    allow_pickle=False
                )

                result = {
                    "model": mode,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "parameters": sum(
                        p.numel()
                        for p in model.parameters()
                    ),
                    **metrics(
                        Y_VAL,
                        saved["logits"]
                    ),
                    **calibration(
                        Y_VAL,
                        saved["logits"]
                    ),
                    "runtime_minutes":
                        (time.time()-run_start)/60
                }

                completed = {
                    "run_hash": run_hash,
                    "config": run_config,
                    "result": result,
                    "history": run_history,
                    "checkpoint_sha256":
                        sha256(ckpt),
                    "prediction_sha256":
                        sha256(pred)
                }

                save_json(
                    completed, meta
                )

                del model, optimizer
                del scheduler, scaler
                del batch, output, loss
                del train_loader, val_loader

                gc.collect()
                torch.cuda.empty_cache()

            REMOTE_D.append(
                upload_private(
                    ckpt,
                    checkpoint_asset
                )
            )

            REMOTE_D.append(
                upload_private(
                    pred,
                    prediction_asset
                )
            )

            REMOTE_D.append(
                upload_private(
                    meta,
                    metadata_asset
                )
            )

            # Delete local weights only after
            # verified remote storage.

            ckpt.unlink()

        saved = np.load(
            pred,
            allow_pickle=False
        )

        assert np.array_equal(
            saved["sample_id"],
            sample_ids_d
        )

        assert np.array_equal(
            saved["labels"],
            Y_VAL
        )

        LOGITS_D[(mode, seed)] = (
            saved["logits"]
        )

        WEIGHTS_D[(mode, seed)] = (
            saved["weights"]
        )

        RESULTS_D.append(
            completed["result"]
        )

        HISTORY_D.extend(
            completed["history"]
        )

        save_table(
            pd.DataFrame(RESULTS_D),
            "stage6d_01_seed_results"
        )

        save_table(
            pd.DataFrame(HISTORY_D),
            "stage6d_02_training_history"
        )

        for path in TABLES.glob("*.csv"):

            target = (
                BACKUP / "tables"
                / path.name
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                path, target
            )

        target = (
            BACKUP / "run_metadata"
            / f"{run_name}.json"
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            meta, target
        )

        push_d(
            f"Preserve completed Stage 6D run: "
            f"{run_name}"
        )

        print(
            run_name,
            "| F1:",
            completed["result"]["macro_f1"],
            "| Permanent backup: PASS"
        )

# =====================================================================
# STEP 86 — MULTI-SEED PERFORMANCE AND ENSEMBLES
# =====================================================================

section("STEP 86 — MULTI-SEED COMPARISON")

results_d = pd.DataFrame(
    RESULTS_D
)

assert len(results_d) == (
    len(SEEDS_D) * len(MODES_D)
)

summary_d = (
    results_d.groupby("model")
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

save_table(
    summary_d,
    "stage6d_03_multiseed_summary"
)

print(
    summary_d.round(5)
    .to_string(index=False)
)

ENSEMBLES_D = {}

ensemble_rows = []

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

    ensemble_logits = np.log(
        probabilities.clip(
            1e-12, 1
        )
    )

    ENSEMBLES_D[mode] = (
        ensemble_logits
    )

    ensemble_rows.append({
        "model": mode,
        "seeds": len(SEEDS_D),
        **metrics(
            Y_VAL,
            ensemble_logits
        ),
        **calibration(
            Y_VAL,
            ensemble_logits
        )
    })

    ensemble_path = (
        PRED / f"{mode}_ensemble_val.npz"
    )

    np.savez_compressed(
        ensemble_path,
        sample_id=sample_ids_d,
        labels=Y_VAL,
        logits=ensemble_logits
    )

    upload_private(
        ensemble_path
    )

ensemble_d = save_table(
    pd.DataFrame(ensemble_rows)
    .sort_values(
        "macro_f1",
        ascending=False
    ),
    "stage6d_04_ensemble_results"
)

print("\nEnsemble comparison:")

print(
    ensemble_d.round(5)
    .to_string(index=False)
)

# Paired differences across training seeds.

seed_comparison_d = (
    results_d.pivot(
        index="seed",
        columns="model",
        values="macro_f1"
    )
    .reset_index()
)

seed_comparison_d[
    "fixed_minus_standard"
] = (
    seed_comparison_d["fixed_fusion"]
    - seed_comparison_d["standard"]
)

seed_comparison_d[
    "normalized_minus_standard"
] = (
    seed_comparison_d["normalized_fusion"]
    - seed_comparison_d["standard"]
)

seed_comparison_d[
    "normalized_minus_fixed"
] = (
    seed_comparison_d["normalized_fusion"]
    - seed_comparison_d["fixed_fusion"]
)

save_table(
    seed_comparison_d,
    "stage6d_05_paired_seed_differences"
)

# =====================================================================
# STEP 87 — STATISTICAL AND LINGUISTIC ANALYSIS
# =====================================================================

section("STEP 87 — ROBUSTNESS AND STATISTICS")

def fast_f1_d(y, p):

    cm = np.bincount(
        3*y + p,
        minlength=9
    ).reshape(3, 3)

    tp = np.diag(cm).astype(float)

    den = (
        cm.sum(axis=0)
        + cm.sum(axis=1)
    )

    return np.divide(
        2*tp,
        den,
        out=np.zeros(3),
        where=den > 0
    ).mean()

groups_d = (
    DATA["val"]["group_id"]
    .to_numpy()
)

members_d = [
    np.flatnonzero(groups_d == g)
    for g in np.unique(groups_d)
]

predictions_d = {
    mode: logits.argmax(axis=1)
    for mode, logits
    in ENSEMBLES_D.items()
}

pairs_d = [
    ("fixed_fusion", "standard"),
    ("normalized_fusion", "standard"),
    ("normalized_fusion", "fixed_fusion")
]

rng_d = np.random.default_rng(42)

bootstrap_d = {
    pair: []
    for pair in pairs_d
}

for _ in range(BOOTSTRAPS_D):

    sampled = rng_d.integers(
        0,
        len(members_d),
        size=len(members_d)
    )

    idx = np.concatenate([
        members_d[i]
        for i in sampled
    ])

    scores = {
        name: fast_f1_d(
            Y_VAL[idx],
            pred[idx]
        )
        for name, pred
        in predictions_d.items()
    }

    for a, b in pairs_d:

        bootstrap_d[(a, b)].append(
            scores[a] - scores[b]
        )

stat_rows_d = []

for a, b in pairs_d:

    pa = predictions_d[a]
    pb = predictions_d[b]

    ca = pa == Y_VAL
    cb = pb == Y_VAL

    gained = int(
        (ca & ~cb).sum()
    )

    lost = int(
        (~ca & cb).sum()
    )

    pvalue = (
        binomtest(
            gained,
            gained + lost,
            p=0.5
        ).pvalue
        if gained + lost
        else 1.0
    )

    values = bootstrap_d[(a, b)]

    stat_rows_d.append({
        "model": a,
        "reference": b,
        "delta_macro_f1":
            fast_f1_d(Y_VAL, pa)
            - fast_f1_d(Y_VAL, pb),
        "ci95_low":
            np.percentile(values, 2.5),
        "ci95_high":
            np.percentile(values, 97.5),
        "gained_correct": gained,
        "lost_correct": lost,
        "mcnemar_p": pvalue
    })

stats_d = pd.DataFrame(
    stat_rows_d
)

pvalues = stats_d[
    "mcnemar_p"
].to_numpy()

order = np.argsort(pvalues)

adjusted = np.empty(
    len(pvalues)
)

running = 0.0

for rank, idx in enumerate(order):

    running = max(
        running,
        (len(pvalues)-rank)
        * pvalues[idx]
    )

    adjusted[idx] = min(
        running, 1.0
    )

stats_d["mcnemar_p_holm"] = (
    adjusted
)

save_table(
    stats_d,
    "stage6d_06_paired_statistics"
)

subgroups_d = []
gates_d = []

for mode, logits in (
    ENSEMBLES_D.items()
):

    mean_weights = np.mean(
        [
            WEIGHTS_D[(mode, seed)]
            for seed in SEEDS_D
        ],
        axis=0
    )

    for subgroup, mask in MASKS.items():

        if not mask.any():
            continue

        subgroups_d.append({
            "model": mode,
            "subgroup": subgroup,
            "n": int(mask.sum()),
            **metrics(
                Y_VAL[mask],
                logits[mask]
            )
        })

        gates_d.append({
            "model": mode,
            "subgroup": subgroup,
            "n": int(mask.sum()),
            "global_weight":
                mean_weights[mask, 0].mean(),
            "phrase_weight":
                mean_weights[mask, 1].mean(),
            "directional_weight":
                mean_weights[mask, 2].mean(),
            "global_weight_sd":
                mean_weights[mask, 0].std(),
            "phrase_weight_sd":
                mean_weights[mask, 1].std(),
            "directional_weight_sd":
                mean_weights[mask, 2].std()
        })

subgroup_d = save_table(
    pd.DataFrame(subgroups_d),
    "stage6d_07_subgroup_results"
)

gate_d = save_table(
    pd.DataFrame(gates_d),
    "stage6d_08_fusion_diagnostics"
)

print("\nPaired statistical comparisons:")

print(
    stats_d.round(5)
    .to_string(index=False)
)

# =====================================================================
# STEP 88 — PUBLICATION ASSETS
# =====================================================================

section("STEP 88 — PUBLICATION FIGURES")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

FIGURES_D = []

def figure_d(
    fig, name, caption, source_table
):

    fig.tight_layout()

    for ext in [
        "png", "pdf", "svg"
    ]:

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

        destination = (
            BACKUP / "publication_assets"
            / path.relative_to(PUB)
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            path, destination
        )

    caption_path = (
        PUB / "captions"
        / f"{name}.txt"
    )

    caption_path.write_text(
        caption,
        encoding="utf-8"
    )

    target = (
        BACKUP / "publication_assets"
        / "captions"
        / caption_path.name
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        caption_path,
        target
    )

    FIGURES_D.append({
        "figure_id": name,
        "stage": "6d",
        "png": f"figures/png/{name}.png",
        "pdf": f"figures/pdf/{name}.pdf",
        "svg": f"figures/svg/{name}.svg",
        "caption": caption,
        "source_table": source_table
    })

    plt.close(fig)

# 1. Multi-seed model performance.

plot = summary_d.sort_values(
    "mean_macro_f1"
)

fig, ax = plt.subplots(
    figsize=(8, 4.5)
)

ax.barh(
    plot["model"],
    plot["mean_macro_f1"],
    xerr=plot["sd_macro_f1"],
    capsize=4,
    color="#398b80"
)

ax.set_xlim(0, 1)

ax.set_xlabel(
    "Mean validation macro-F1"
)

ax.set_title(
    "Fresh initialization: two-seed comparison"
)

figure_d(
    fig,
    "stage6d_01_multiseed_comparison",
    "Mean and sample standard deviation "
    "across two training seeds on a fixed "
    "validation cohort.",
    "stage6d_03_multiseed_summary.csv"
)

# 2. Individual seed consistency.

fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

for mode in MODES_D:

    part = results_d[
        results_d["model"] == mode
    ]

    ax.plot(
        part["seed"].astype(str),
        part["macro_f1"],
        marker="o",
        label=mode
    )

ax.set_xlabel("Training seed")
ax.set_ylabel("Validation macro-F1")

ax.legend(
    frameon=False
)

ax.set_title(
    "Training-seed consistency"
)

figure_d(
    fig,
    "stage6d_02_seed_consistency",
    "Validation performance for each "
    "training seed.",
    "stage6d_01_seed_results.csv"
)

# 3. Training curves.

history_df_d = pd.DataFrame(
    HISTORY_D
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for mode in MODES_D:

    part = (
        history_df_d[
            history_df_d["model"] == mode
        ]
        .groupby("epoch")["macro_f1"]
        .mean()
    )

    ax.plot(
        part.index,
        part.values,
        marker="o",
        label=mode
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Mean validation macro-F1")

ax.legend(
    frameon=False
)

ax.set_title(
    "Fresh initialization training convergence"
)

figure_d(
    fig,
    "stage6d_03_training_curves",
    "Mean validation macro-F1 across "
    "two training seeds at each epoch.",
    "stage6d_02_training_history.csv"
)

# 4. Paired statistical effects.

fig, ax = plt.subplots(
    figsize=(8, 4.5)
)

for i, row in stats_d.iterrows():

    ax.plot(
        [
            row["ci95_low"],
            row["ci95_high"]
        ],
        [i, i],
        color="#398b80"
    )

    ax.scatter(
        row["delta_macro_f1"],
        i,
        color="#398b80"
    )

ax.axvline(
    0,
    linestyle="--",
    color="#64748b"
)

ax.set_yticks(
    range(len(stats_d)),
    [
        f"{r.model} vs {r.reference}"
        for r in stats_d.itertuples()
    ]
)

ax.set_xlabel(
    "Macro-F1 difference"
)

ax.set_title(
    "Paired validation effect estimates"
)

figure_d(
    fig,
    "stage6d_04_paired_effects",
    "Exploratory group-bootstrap confidence "
    "intervals for ensemble differences.",
    "stage6d_06_paired_statistics.csv"
)

# 5. Subgroup heatmap.

plot = subgroup_d.pivot(
    index="subgroup",
    columns="model",
    values="macro_f1"
)

fig, ax = plt.subplots(
    figsize=(8, 5.5)
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
            j, i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=(
                "white"
                if value > 0.65
                else "black"
            )
        )

ax.set_xticks(
    range(len(plot.columns)),
    plot.columns,
    rotation=20,
    ha="right"
)

ax.set_yticks(
    range(len(plot.index)),
    plot.index
)

fig.colorbar(
    im, ax=ax
)

ax.set_title(
    "Linguistic subgroup performance"
)

figure_d(
    fig,
    "stage6d_05_subgroup_heatmap",
    "Ensemble performance on heuristic "
    "linguistic validation subgroups.",
    "stage6d_07_subgroup_results.csv"
)

# 6. Learned normalized fusion weights.

plot = gate_d[
    gate_d["model"]
    == "normalized_fusion"
].set_index("subgroup")

fig, ax = plt.subplots(
    figsize=(9, 5)
)

plot[[
    "global_weight",
    "phrase_weight",
    "directional_weight"
]].plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=[
        "#648cce",
        "#24967b",
        "#9a78ca"
    ]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Mean fusion weight")
ax.set_xlabel("")

ax.tick_params(
    axis="x",
    rotation=25
)

ax.legend(
    frameon=False,
    fontsize=9
)

ax.set_title(
    "Normalized branch contribution"
)

figure_d(
    fig,
    "stage6d_06_fusion_weights",
    "Mean normalized branch weights "
    "by linguistic subgroup.",
    "stage6d_08_fusion_diagnostics.csv"
)

# 7. Ensemble comparison.

plot = ensemble_d.sort_values(
    "macro_f1"
)

fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

bars = ax.barh(
    plot["model"],
    plot["macro_f1"],
    color="#398b80"
)

ax.bar_label(
    bars,
    fmt="%.4f",
    padding=4
)

ax.set_xlim(0, 1.05)

ax.set_xlabel(
    "Validation macro-F1"
)

ax.set_title(
    "Equal-size two-model ensembles"
)

figure_d(
    fig,
    "stage6d_07_ensemble_comparison",
    "Equal-size probability ensemble "
    "comparison across architectures.",
    "stage6d_04_ensemble_results.csv"
)

# 8. Calibration comparison.

fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

ax.barh(
    ensemble_d["model"],
    ensemble_d["ece"],
    color="#648cce"
)

ax.set_xlabel(
    "Expected calibration error"
)

ax.set_title(
    "Uncalibrated ensemble reliability"
)

figure_d(
    fig,
    "stage6d_08_calibration",
    "Expected calibration error of "
    "uncalibrated probability ensembles.",
    "stage6d_04_ensemble_results.csv"
)

# =====================================================================
# STEP 89 — DECISION AND PERMANENT BACKUP
# =====================================================================

section("STEP 89 — RESEARCH DECISION")

mean_scores_d = (
    summary_d.set_index("model")
    ["mean_macro_f1"]
)

baseline_d = float(
    mean_scores_d["standard"]
)

fixed_d = float(
    mean_scores_d["fixed_fusion"]
)

normalized_d = float(
    mean_scores_d["normalized_fusion"]
)

candidate_d = max(
    ["fixed_fusion", "normalized_fusion"],
    key=lambda m: mean_scores_d[m]
)

candidate_score_d = float(
    mean_scores_d[candidate_d]
)

consistent_d = bool(
    (
        seed_comparison_d[candidate_d]
        > seed_comparison_d["standard"]
    ).all()
)

if (
    candidate_score_d > baseline_d
    and consistent_d
):

    decision_d = (
        "Advance strongest architectural candidate "
        "to third-seed and matched-capacity "
        "confirmation."
    )

elif candidate_score_d > baseline_d:

    decision_d = (
        "Positive mean improvement but inconsistent "
        "across seeds. Confirmation required."
    )

else:

    decision_d = (
        "Neither proposed architecture improves "
        "the standard baseline on mean macro-F1."
    )

report_d = {
    "standard_mean_f1": baseline_d,
    "fixed_fusion_mean_f1": fixed_d,
    "normalized_fusion_mean_f1": normalized_d,
    "best_architectural_candidate": candidate_d,
    "candidate_mean_f1": candidate_score_d,
    "candidate_delta_vs_standard":
        candidate_score_d - baseline_d,
    "candidate_wins_both_seeds":
        consistent_d,
    "decision": decision_d,
    "training_seeds": SEEDS_D,
    "test_evaluated": False,
    "external_evaluation": False,
    "novelty_verified": False
}

save_json(
    report_d,
    AUDIT / "stage6d_summary.json"
)

shutil.copy2(
    AUDIT / "stage6d_summary.json",
    BACKUP / "stage6d_summary.json"
)

# Consolidate figure manifests.

manifest_parts_d = []

for path in REPO.rglob(
    "figure_manifest.csv"
):

    try:

        frame = pd.read_csv(path)

        if "figure_id" in frame.columns:

            manifest_parts_d.append(
                frame
            )

    except Exception:

        continue

existing_manifest = (
    PUB / "figure_manifest.csv"
)

if existing_manifest.exists():

    frame = pd.read_csv(
        existing_manifest
    )

    if "figure_id" in frame.columns:

        manifest_parts_d.append(frame)

manifest_parts_d.append(
    pd.DataFrame(FIGURES_D)
)

manifest_d = pd.concat(
    manifest_parts_d,
    ignore_index=True
).drop_duplicates(
    "figure_id",
    keep="last"
)

manifest_d.to_csv(
    PUB / "figure_manifest.csv",
    index=False
)

manifest_d.to_csv(
    BACKUP / "figure_manifest.csv",
    index=False
)

# Restore all previously committed
# publication assets.

for folder in [
    "research_reconstruction_v1",
    "research_stage6b",
    "research_stage6c",
    "research_stage6d"
]:

    previous = (
        REPO / folder
        / "publication_assets"
    )

    if previous.exists():

        for path in previous.rglob("*"):

            if not path.is_file():
                continue

            destination = (
                PUB
                / path.relative_to(previous)
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            if not destination.exists():

                shutil.copy2(
                    path,
                    destination
                )

# Preserve final aggregate tables.

for path in TABLES.glob("*.csv"):

    target = (
        BACKUP / "tables"
        / path.name
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        path, target
    )

for path in (
    PUB / "tables"
).glob("stage6d_*.tex"):

    target = (
        BACKUP / "publication_assets"
        / "tables" / path.name
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        path, target
    )

# Verify every completed experiment
# against the private GitHub release.

assets = remote_assets()

remote_manifest_d = []

for seed in SEEDS_D:

    for mode in MODES_D:

        run_name = (
            f"{mode}_seed{seed}"
        )

        metadata_path = (
            CHECKPOINTS / run_name
            / "completed.json"
        )

        completed = json.loads(
            metadata_path.read_text()
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
                sha256(metadata_path)
            )
        ]:

            assert name in assets

            verify_remote_asset(
                assets[name],
                expected
            )

            remote_manifest_d.append({
                "name": name,
                "asset_id": assets[name]["id"],
                "sha256": expected,
                "size_bytes": assets[name]["size"]
            })

manifest_path_d = (
    AUDIT / "stage6d_remote_manifest.json"
)

save_json(
    {
        "release_tag": STAGE_TAG,
        "repository":
            f"{OWNER}/{PRIVATE_REPO}",
        "assets": remote_manifest_d,
        "config": CONFIG_D
    },
    manifest_path_d
)

manifest_hash_d = sha256(
    manifest_path_d
)

upload_private(
    manifest_path_d,
    f"stage6d_manifest_"
    f"{manifest_hash_d[:12]}.json"
)

shutil.copy2(
    manifest_path_d,
    BACKUP / "stage6d_remote_manifest.json"
)

# Complete reproducibility archive.

compact_d = (
    WORK6D
    / "sentiment_stage6d_compact.zip"
)

with zipfile.ZipFile(
    compact_d,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for path in S6D.rglob("*"):

        if (
            path.is_file()
            and path.suffix != ".pt"
        ):

            z.write(
                path,
                path.relative_to(S6D)
            )

with zipfile.ZipFile(compact_d) as z:

    assert z.testzip() is None

upload_private(
    compact_d,
    f"stage6d_compact_"
    f"{sha256(compact_d)[:12]}.zip"
)

publication_d = (
    WORK6D
    / "sentiment_publication_assets.zip"
)

with zipfile.ZipFile(
    publication_d,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for path in PUB.rglob("*"):

        if path.is_file():

            z.write(
                path,
                path.relative_to(PUB)
            )

with zipfile.ZipFile(publication_d) as z:

    assert z.testzip() is None

upload_private(
    publication_d,
    f"publication_assets_"
    f"{sha256(publication_d)[:12]}.zip"
)

FINAL_COMMIT_D = push_d(
    "Complete Stage 6D multi-seed screening "
    "and publication assets"
)

# =====================================================================
# FINAL REPORT
# =====================================================================

section("FINAL STAGE 6D RESEARCH REPORT")

print("Exact dataset recovery: PASS")
print("Architecture smoke tests: PASS")
print("Fresh pretrained initialization: PASS")
print("Completed runs:", len(RESULTS_D))
print("Training seeds:", SEEDS_D)
print("GitHub synchronized: True")
print("Private checkpoint backup: PASS")
print("Final commit:", FINAL_COMMIT_D)

print("\nINDIVIDUAL SEED RESULTS")

print(
    results_d.round(5)
    .to_string(index=False)
)

print("\nMULTI-SEED SUMMARY")

print(
    summary_d.round(5)
    .to_string(index=False)
)

print("\nPAIRED SEED DIFFERENCES")

print(
    seed_comparison_d.round(5)
    .to_string(index=False)
)

print("\nENSEMBLE COMPARISON")

print(
    ensemble_d.round(5)
    .to_string(index=False)
)

print("\nLINGUISTIC SUBGROUPS")

print(
    subgroup_d.pivot(
        index="subgroup",
        columns="model",
        values="macro_f1"
    ).round(4).to_string()
)

print("\nPAIRED STATISTICS")

print(
    stats_d.round(5)
    .to_string(index=False)
)

print("\nNORMALIZED FUSION DIAGNOSTICS")

print(
    gate_d[
        gate_d["model"]
        == "normalized_fusion"
    ].round(4).to_string(index=False)
)

print("\nRESEARCH DECISION")

print(
    json.dumps(
        report_d,
        indent=2
    )
)

print("\nPUBLICATION ASSETS")

print(
    "New unique figures:",
    len(FIGURES_D)
)

for ext in [
    "png", "pdf", "svg"
]:

    print(
        ext.upper(),
        "figures:",
        len(list(
            (PUB / "figures" / ext)
            .glob(f"*.{ext}")
        ))
    )

print(
    "Verified checkpoint-related assets:",
    len(remote_manifest_d)
)

print(
    "Private release:",
    f"https://github.com/{OWNER}/"
    f"{PRIVATE_REPO}/releases/tag/{STAGE_TAG}"
)

print("Manual file saving required: NO")
print("Google Drive used: NO")
print("Paid API calls: NO")
print("Test evaluated: NO")
print("External evaluation: NO")
print("Novelty conclusively verified: NO")

print(
    "Runtime minutes:",
    round(
        (time.time()-STAGE6D_START)/60,
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

print("\nSTAGE 6D COMPLETE")
for seed in SEEDS_D:
    for mode in MODES_D:
        if (mode, seed) not in RUNS_E:
            continue


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


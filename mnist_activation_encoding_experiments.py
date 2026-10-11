"""Two manual-NumPy experiments: encoder freezing and common-head readout.

Run --help first. Formal runs keep MNIST 54000/6000 and never access test keys.
Default: classroom9, seeds42/43/44, paired freezing for sigmoid/relu/linear.
Use --synthetic-check for a small, explicitly non-scientific end-to-end check.
Every invocation creates a fresh directory; old experiments are read-only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import activation_encoding_core as core
from activation_encoding_readout import extract_features, fit_output_head

ROOT = Path(__file__).resolve().parent
SHANGHAI = timezone(timedelta(hours=8))
ORIGINAL = ROOT / "logs/manual_activation_comparison_glorot2010_N2_seed42_20260930_165626_546101"
SUPPLEMENT = ROOT / "logs/manual_activation_supplement_N2_seed42_20261008_175133_212904"
SNAPSHOT_EPOCHS = {0, 1, 2, 3, 5, 10, 20, 30, 40, 50}
EARLY_STEPS = {1, 2, 5, 10, 20, 50, 100, 200, 500}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run_directory(output_root: Path, protocol_prefix: str = "activation_encoding_N2") -> Path:
    directory = Path(output_root) / f"{protocol_prefix}_{datetime.now(SHANGHAI):%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{k: serializable(v) for k, v in row.items()} for row in rows])


def parameter_dict(p) -> dict[str, np.ndarray]:
    values = {"W1": p.weights[0], "b1": p.biases[0], "W2": p.weights[1], "b2": p.biases[1]}
    if p.alpha is not None:
        values["alpha"] = p.alpha
    return values


def norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values.astype(np.float64)))


def parameter_statistics(p, initial, start=None, paths=None) -> dict:
    row = {}
    for name, value in parameter_dict(p).items():
        initial_value = parameter_dict(initial)[name]
        initial_norm = norm(initial_value)
        row[f"parameter_norm_{name}"] = norm(value)
        row[f"parameter_rms_{name}"] = norm(value) / np.sqrt(value.size)
        row[f"displacement_from_initial_{name}"] = norm(value.astype(np.float64) - initial_value)
        row[f"relative_initial_displacement_{name}"] = row[f"displacement_from_initial_{name}"] / initial_norm if initial_norm else None
        if start is not None:
            origin = parameter_dict(start)[name]
            row[f"epoch_net_displacement_{name}"] = norm(value.astype(np.float64) - origin)
            row[f"epoch_relative_displacement_{name}"] = row[f"epoch_net_displacement_{name}"] / norm(origin) if norm(origin) else None
            row[f"epoch_actual_update_path_{name}"] = paths.get(name, 0.0)
            row[f"epoch_net_over_path_{name}"] = row[f"epoch_net_displacement_{name}"] / paths[name] if paths.get(name, 0.0) else None
    return row


def learning_rate(epoch: int, base: float, policy: str) -> float:
    if policy == "constant":
        return base
    # Shared predeclared schedule, NOT per-model validation-driven scheduling.
    return base * (1.0 if epoch <= 30 else .5 if epoch <= 40 else .25 if epoch <= 45 else .125)


def load_data(cache: Path, synthetic: bool) -> tuple:
    if synthetic:
        rng = np.random.default_rng(20261009)
        x_train = rng.normal(size=(72, 12)).astype(np.float32)
        x_val = rng.normal(size=(24, 12)).astype(np.float32)
        teacher = rng.normal(size=(12, 3))
        return x_train, (x_train @ teacher).argmax(axis=1), x_val, (x_val @ teacher).argmax(axis=1), [12, 9, 3]
    with np.load(cache, allow_pickle=False) as archive:
        # Do NOT enumerate, load, or evaluate x_test/y_test.
        images = archive["x_train"]
        labels = archive["y_train"]
    if images.shape != (60000, 28, 28) or labels.shape != (60000,):
        raise ValueError("Expected the original 60000-image MNIST training archive.")
    if not np.isfinite(images).all() or not np.issubdtype(labels.dtype, np.integer) or images.min() < 0 or images.max() > 255 or labels.min() < 0 or labels.max() > 9:
        raise ValueError("Invalid MNIST training images/labels.")
    x = images.reshape(60000, 784).astype(np.float32) / np.float32(255)
    return x[:54000], labels[:54000].astype(np.int64), x[54000:], labels[54000:].astype(np.int64), [784, 784, 10]


def save_validation(directory: Path, epoch: int, evaluation: dict, labels, synthetic: bool) -> None:
    values = {"epoch": np.asarray(epoch), "probabilities": evaluation["probabilities"], "labels": labels,
              "validation_indices": np.arange(len(labels)), "predictions": evaluation["predictions"],
              "per_sample_cross_entropy": evaluation["per_sample_loss"], "logits": evaluation["logits"]}
    if not synthetic:
        values["mnist_training_indices"] = np.arange(54000, 60000)
    np.savez_compressed(directory / f"epoch_{epoch:03d}.npz", **values)


def plot_training(directory: Path, rows: list[dict]) -> None:
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, key, label in zip(axes.flat, ("validation_loss", "validation_accuracy", "probe_local_jacobian_gain", "relative_initial_displacement_W1"),
                                ("Validation cross-entropy", "Validation accuracy", "Probe local derivative gain", "W1 displacement / initial norm")):
        values = [np.nan if r.get(key) is None else r[key] for r in rows]
        axis.plot(epochs, values)
        axis.set(xlabel="Completed epoch", ylabel=label)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(directory / f"training_diagnostics.{extension}", dpi=600)
    plt.close(fig)


def train_condition(initial, condition: str, x_train, y_train, x_val, y_val, train_probe, val_probe,
                    output: Path, seed: int, args) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir, validation_dir = output / "checkpoints", output / "validation"
    checkpoint_dir.mkdir()
    validation_dir.mkdir()
    (output / "probe_snapshots").mkdir()
    p = core.copy_parameters(initial)
    train_encoder = condition == "trainable"
    rows, probe_rows, step_rows, unit_rows = [], [], [], []
    best_loss, best_epoch = float("inf"), None
    global_step, examples_seen, completed = 0, 0, 0
    failure = None
    initial_inactive = None
    inactive_streak = np.zeros(p.weights[0].shape[0], dtype=np.int64)
    overall_start = time.perf_counter()
    epoch = 0
    core.save_parameters(output / "initial_weights.npz", p)

    def diagnose(epoch: int, step: int, snapshot: bool, epoch_endpoint: bool) -> dict:
        nonlocal initial_inactive, inactive_streak
        start_time = time.perf_counter()
        before_hash = hashlib.sha256(b"".join(a.tobytes() for a in parameter_dict(p).values())).hexdigest()
        scalars, units, raw = core.probe_diagnostics(x_val[val_probe], y_val[val_probe], p)
        train_scalars, _, _ = core.probe_diagnostics(x_train[train_probe], y_train[train_probe], p)
        if snapshot:
            np.savez_compressed(output / "probe_snapshots" / f"epoch_{epoch:03d}_step_{step:06d}.npz", **raw)
        if epoch_endpoint:
            inactive = np.asarray(units["all_probe_zero"], dtype=bool)
            previous = initial_inactive
            inactive_streak = np.where(inactive, inactive_streak + 1, 0) if epoch else np.zeros_like(inactive_streak)
            if p.activation == "relu":
                scalars["probe_new_inactive_units"] = int((inactive & ~previous).sum()) if previous is not None else None
                scalars["probe_recovered_units"] = int((~inactive & previous).sum()) if previous is not None else None
                scalars["probe_max_inactive_epoch_streak"] = int(inactive_streak.max())
            units["inactive_epoch_streak"] = inactive_streak.copy()
            initial_inactive = inactive.copy()
            unit_rows.append((epoch, {k: np.asarray(v).copy() for k, v in units.items()}))
        after_hash = hashlib.sha256(b"".join(a.tobytes() for a in parameter_dict(p).values())).hexdigest()
        if before_hash != after_hash:
            raise RuntimeError("Diagnostic probe mutated parameters.")
        scalars["fixed_train_probe_loss"] = train_scalars["probe_loss"]
        scalars["fixed_train_probe_accuracy"] = train_scalars["probe_accuracy"]
        scalars["probe_seconds"] = time.perf_counter() - start_time
        timepoint = "epoch_endpoint" if epoch_endpoint else "startup_step"
        probe_rows.append({"epoch": epoch, "global_step": step, "timepoint": timepoint, "scope": "fixed_validation_probe", **scalars})
        probe_rows.append({"epoch": epoch, "global_step": step, "timepoint": timepoint, "scope": "fixed_training_probe", **train_scalars})
        return scalars

    try:
        evaluation = core.evaluate_dataset(x_val, y_val, p)
        diagnostics = diagnose(0, 0, True, True)
        save_validation(validation_dir, 0, evaluation, y_val, args.synthetic_check)
        core.save_parameters(checkpoint_dir / "epoch_000.npz", p)
        rows.append({"epoch": 0, "global_step": 0, "examples_seen": 0, "learning_rate": None, "online_train_loss": None,
                     "online_train_accuracy": None, "validation_loss": evaluation["loss"], "validation_accuracy": evaluation["accuracy"],
                     **diagnostics, **parameter_statistics(p, initial)})
        for epoch in range(1, args.epochs + 1):
            start_parameters = core.copy_parameters(p)
            epoch_start = time.perf_counter()
            rate = learning_rate(epoch, args.learning_rate, args.lr_policy)
            order = np.random.default_rng(seed + 1000003 * epoch).permutation(len(y_train))
            loss_sum, correct, seen = 0.0, 0, 0
            paths = {key: 0.0 for key in parameter_dict(p)}
            weighted = {f"{kind}_{layer}": 0.0 for layer in parameter_dict(p) for kind in ("gradient_fro", "gradient_rms")}
            weighted.update(delta_rms_L1=0.0, delta_rms_L2=0.0)
            training_start = time.perf_counter()
            probe_seconds_in_training = 0.0
            for batch_number, start in enumerate(range(0, len(order), args.batch_size), 1):
                indices = order[start:start + args.batch_size]
                with np.errstate(over="raise", invalid="raise"):
                    cache = core.forward(x_train[indices], p)
                    gradients = core.backward(cache, y_train[indices], p)
                    batch_loss = core.cross_entropy_from_logits(cache["logits"], y_train[indices])
                    if not np.isfinite(batch_loss):
                        raise FloatingPointError("Non-finite training loss.")
                    batch_correct = int((cache["probabilities"].argmax(axis=0) == y_train[indices]).sum())
                    batch_stats = {}
                    for layer in parameter_dict(p):
                        gradient_norm = norm(gradients[layer])
                        batch_stats[f"gradient_fro_{layer}"] = gradient_norm
                        batch_stats[f"gradient_rms_{layer}"] = gradient_norm / np.sqrt(gradients[layer].size)
                    for number in (1, 2):
                        batch_stats[f"delta_rms_L{number}"] = norm(gradients[f"delta{number}"]) / np.sqrt(gradients[f"delta{number}"].size)
                    update_norms = core.update_parameters(p, gradients, rate, train_encoder=train_encoder)
                if not all(np.isfinite(value).all() for value in parameter_dict(p).values()):
                    raise FloatingPointError("Non-finite parameter after update.")
                count = len(indices)
                loss_sum += batch_loss * count
                correct += batch_correct
                seen += count
                global_step += 1
                examples_seen += count
                for key, value in batch_stats.items():
                    weighted[key] += value * count
                for key in paths:
                    paths[key] += update_norms[key]
                if batch_number % (10 if epoch <= 5 else 50) == 0 or start + count == len(order) or (epoch == 1 and batch_number in EARLY_STEPS):
                    step_rows.append({"epoch": epoch, "global_step": global_step, "batch_size_actual": count,
                                      "examples_seen": examples_seen, "learning_rate": rate, "pre_update_batch_loss": batch_loss,
                                      "pre_update_batch_accuracy": batch_correct / count, **batch_stats,
                                      **{f"actual_update_norm_{k}": v for k, v in update_norms.items()}})
                if epoch == 1 and batch_number in EARLY_STEPS:
                    probe_t0 = time.perf_counter()
                    diagnose(epoch, global_step, False, False)
                    probe_seconds_in_training += time.perf_counter() - probe_t0
            training_seconds = time.perf_counter() - training_start - probe_seconds_in_training
            evaluation_start = time.perf_counter()
            evaluation = core.evaluate_dataset(x_val, y_val, p)
            validation_seconds = time.perf_counter() - evaluation_start
            diagnostics = diagnose(epoch, global_step, epoch in SNAPSHOT_EPOCHS or epoch == args.epochs, True)
            row = {"epoch": epoch, "global_step": global_step, "examples_seen": examples_seen, "learning_rate": rate,
                   "online_train_loss": loss_sum / seen, "online_train_accuracy": correct / seen,
                   "validation_loss": evaluation["loss"], "validation_accuracy": evaluation["accuracy"],
                   **{k: v / seen for k, v in weighted.items()}, **diagnostics,
                   **parameter_statistics(p, initial, start_parameters, paths), "training_seconds": training_seconds,
                   "validation_seconds": validation_seconds, "startup_probe_seconds": probe_seconds_in_training}
            ratio_denominator = row["gradient_rms_W2"]
            row["gradient_ratio_G1_GN_rms"] = row["gradient_rms_W1"] / ratio_denominator if ratio_denominator else None
            io_start = time.perf_counter()
            core.save_parameters(checkpoint_dir / f"epoch_{epoch:03d}.npz", p)
            save_validation(validation_dir, epoch, evaluation, y_val, args.synthetic_check)
            if evaluation["loss"] < best_loss:
                best_loss, best_epoch = evaluation["loss"], epoch
                core.save_parameters(output / "best_weights.npz", p)
            rows.append(row)
            write_csv(output / "epoch_history.csv", rows)
            write_csv(output / "probe_history.csv", probe_rows)
            write_csv(output / "sampled_step_history.csv", step_rows)
            row["io_seconds"] = time.perf_counter() - io_start
            row["epoch_wall_seconds"] = time.perf_counter() - epoch_start
            completed = epoch
            print(f"seed={seed} {p.activation}/{condition} epoch={epoch}/{args.epochs} CE={evaluation['loss']:.6f} acc={evaluation['accuracy']:.4%} lr={rate:g}", flush=True)
    except FloatingPointError as exc:
        failure = {"reason": str(exc), "epoch_attempted": epoch, "global_step_completed": global_step,
                   "examples_seen_completed": examples_seen}
        write_json(output / "failure.json", failure)
        # Preserve failed state for diagnosis even if it cannot be a valid checkpoint.
        np.savez_compressed(output / "failure_parameters.npz", **parameter_dict(p), activation=np.asarray(p.activation))
    if all(np.isfinite(value).all() for value in parameter_dict(p).values()):
        core.save_parameters(output / "final_weights.npz", p)
    write_csv(output / "epoch_history.csv", rows)
    write_csv(output / "probe_history.csv", probe_rows)
    write_csv(output / "sampled_step_history.csv", step_rows)
    if unit_rows:
        np.savez_compressed(output / "unit_history.npz", epochs=np.array([e for e, _ in unit_rows]),
                            **{key: np.stack([u[key] for _, u in unit_rows]) for key in unit_rows[0][1]})
    result = {"activation": p.activation, "seed": seed, "condition": condition, "directory": str(output.resolve()),
              "scientific_result": not args.synthetic_check, "status": "failed_nonfinite" if failure else "completed",
              "encoder_trainable": train_encoder, "alpha_trainable": train_encoder and p.alpha is not None,
              "encoder_gradient_update_enabled": train_encoder and p.activation != "binary_step",
              "epochs_requested": args.epochs, "epochs_completed": completed, "best_epoch": best_epoch,
              "best_validation_loss": best_loss if best_epoch else None,
              "best_validation_accuracy": next((r["validation_accuracy"] for r in rows if r["epoch"] == best_epoch), None),
              "fixed_budget_validation_loss": rows[-1]["validation_loss"] if not failure and completed == args.epochs else None,
              "fixed_budget_validation_accuracy": rows[-1]["validation_accuracy"] if not failure and completed == args.epochs else None,
              "initial_path": str((output / "initial_weights.npz").resolve()),
              "budget_path": str((checkpoint_dir / f"epoch_{completed:03d}.npz").resolve()) if completed else None,
              "stored_parameter_count": sum(v.size for v in parameter_dict(p).values()),
              "updated_parameter_count": sum(v.size for k, v in parameter_dict(p).items() if k in ("W2", "b2") or (train_encoder and p.activation != "binary_step")),
              "gradient_scope": "epoch fields: sample-weighted average of minibatch norms, not full-epoch gradient; probe fields: one fixed probe batch",
              "saturation_scope": "fixed diagnostic probe only, not universal saturation or permanent dead-neuron criteria",
              "seconds_including_diagnostics_io": time.perf_counter() - overall_start, "failure": failure}
    write_json(output / "result.json", result)
    if rows:
        plot_training(output, rows)
    return result


def encoder_signature(p) -> str:
    activation = p.activation
    alpha = p.alpha
    if activation == "prelu" and alpha is not None and np.all(alpha == np.asarray(.01, dtype=alpha.dtype)):
        activation, alpha = "leaky_relu", None
    digest = hashlib.sha256(activation.encode())
    for value in (p.weights[0], p.biases[0], alpha):
        if value is not None:
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def run_readouts(source_records: list[dict], x_train, y_train, x_val, y_val, output: Path, args) -> list[dict]:
    results, cache = [], {}
    images = None if args.synthetic_check else x_val.reshape(-1, 28, 28)
    for source in source_records:
        if source["condition"] != "trainable" or source["status"] != "completed":
            continue
        seed, activation = source["seed"], source["activation"]
        initial = core.load_parameters(Path(source["initial_path"]))
        if initial.activation != activation or initial.weights[0].shape[1] != x_train.shape[1]:
            raise ValueError("Readout source activation/input dimensions do not match its record/dataset.")
        head_signature = hashlib.sha256(initial.weights[1].tobytes() + initial.biases[1].tobytes()).hexdigest()
        # Both states discard their old decoder and receive the identical initial head.
        for state, path in (("initial", source["initial_path"]), ("budget", source["budget_path"])):
            encoder = core.load_parameters(Path(path))
            signature = encoder_signature(encoder)
            h_train, h_val = None, None
            for scale in args.head_scales:
                destination = output / "readout" / f"seed_{seed}" / activation / state / scale
                destination.parent.mkdir(parents=True, exist_ok=True)
                key = (seed, signature, scale, head_signature)
                if key in cache:
                    destination.mkdir(exist_ok=False)
                    summary = dict(cache[key]["summary"])
                    summary["output_directory"] = str(destination.resolve())
                    write_json(destination / "alias.json", {"identical_encoder_and_head_protocol": True, "alias_of": cache[key]["directory"], "reason": "No independent duplicate fit is needed; this is not an independent replicate."})
                    alias = cache[key]["directory"]
                else:
                    if h_train is None:
                        print(f"Extracting frozen features: seed={seed} {activation}/{state}", flush=True)
                        h_train = extract_features(x_train, encoder)
                        h_val = extract_features(x_val, encoder)
                    try:
                        summary = fit_output_head(h_train, y_train, h_val, y_val, initial.weights[1], initial.biases[1],
                                                  epochs=args.head_epochs, learning_rate=args.head_learning_rate,
                                                  batch_size=args.batch_size, seed=seed, standardize=scale == "standardized",
                                                  output=destination, validation_images=images)
                    except FloatingPointError as exc:
                        summary = {"status": "failed_nonfinite", "reason": str(exc), "final_validation_loss": None,
                                   "final_validation_accuracy": None, "standardize": scale == "standardized"}
                    if summary["status"] == "completed":
                        cache[key] = {"summary": dict(summary), "directory": str(destination.resolve())}
                    alias = None
                record = {**summary, "seed": seed, "activation": activation, "encoder_state": state, "feature_scale": scale,
                          "source_checkpoint": path, "source_encoder_signature": signature, "source_budget_epoch": source["epochs_completed"],
                          "directory": str(destination.resolve()), "alias_of": alias,
                          "scientific_result": not args.synthetic_check, "interpretation": "finite-budget common-head readout, not information amount or proven optimal encoding"}
                results.append(record)
                write_json(destination / "readout_record.json", record)
                write_json(output / "readout_results.json", results)
            del h_train, h_val
    return results


def comparison_outputs(output: Path, paired: list[dict], heads: list[dict]) -> None:
    write_json(output / "paired_results.json", paired)
    write_csv(output / "paired_results.csv", [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in paired])
    differences = []
    for full in paired:
        if full["condition"] != "trainable" or full["fixed_budget_validation_loss"] is None:
            continue
        frozen = next((p for p in paired if p["activation"] == full["activation"] and p["seed"] == full["seed"] and p["condition"] == "frozen"), None)
        if frozen and frozen["fixed_budget_validation_loss"] is not None:
            differences.append({"activation": full["activation"], "seed": full["seed"],
                                "delta_ce_trainable_minus_frozen": full["fixed_budget_validation_loss"] - frozen["fixed_budget_validation_loss"],
                                "delta_accuracy_trainable_minus_frozen": full["fixed_budget_validation_accuracy"] - frozen["fixed_budget_validation_accuracy"]})
    write_csv(output / "paired_differences.csv", differences)
    write_json(output / "readout_results.json", heads)
    if heads:
        write_csv(output / "readout_results.csv", [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in heads])
    readout_differences = []
    for current in heads:
        if current["encoder_state"] != "budget" or current["status"] != "completed":
            continue
        baseline = next((h for h in heads if h["seed"] == current["seed"] and h["activation"] == current["activation"]
                         and h["feature_scale"] == current["feature_scale"] and h["encoder_state"] == "initial"
                         and h["status"] == "completed"), None)
        if baseline:
            readout_differences.append({"activation": current["activation"], "seed": current["seed"],
                                        "feature_scale": current["feature_scale"],
                                        "delta_ce_budget_minus_initial": current["final_validation_loss"] - baseline["final_validation_loss"],
                                        "delta_accuracy_budget_minus_initial": current["final_validation_accuracy"] - baseline["final_validation_accuracy"],
                                        "initial_head_gradient_threshold_met": baseline["gradient_threshold_met"],
                                        "budget_head_gradient_threshold_met": current["gradient_threshold_met"],
                                        "scientific_result": current["scientific_result"]})
    write_csv(output / "readout_encoder_differences.csv", readout_differences)
    if readout_differences:
        scales = list(dict.fromkeys(row["feature_scale"] for row in readout_differences))
        names = list(dict.fromkeys(row["activation"] for row in readout_differences))
        fig, axes = plt.subplots(1, len(scales), figsize=(6.1 * len(scales), 4.4), squeeze=False)
        for axis, scale in zip(axes.flat, scales):
            for index, name in enumerate(names):
                values = [r["delta_ce_budget_minus_initial"] for r in readout_differences if r["activation"] == name and r["feature_scale"] == scale]
                if values:
                    offsets = np.linspace(-.08, .08, len(values)) if len(values) > 1 else np.zeros(1)
                    axis.scatter(index + offsets, values, s=23, color="#0072B2")
                    axis.plot([index - .17, index + .17], [np.mean(values)] * 2, color="black", linewidth=1)
            axis.axhline(0, color="gray", linewidth=.8)
            axis.set(xticks=range(len(names)), xticklabels=names, ylabel="Validation CE: trained encoder - initial encoder",
                     title=f"Common-head readout: {scale}")
            axis.set_xlim(-.5, len(names) - .5)
            axis.tick_params(axis="x", labelrotation=45, labelsize=8)
            axis.spines[["top", "right"]].set_visible(False)
        label = "SYNTHETIC CHECK — NOT MNIST.\n" if not readout_differences[0]["scientific_result"] else ""
        fig.text(.5, .015, label + "Dots: paired seeds; bars: means, not confidence intervals.\nFinite-budget readout; not a measure of information amount.",
                 ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, .13, 1, 1))
        for extension in ("png", "svg"):
            fig.savefig(output / f"readout_encoder_effect.{extension}", dpi=600, bbox_inches="tight")
        plt.close(fig)
    if not differences:
        return
    fig, axis = plt.subplots(figsize=(8, 4.5))
    names = list(dict.fromkeys(row["activation"] for row in differences))
    for index, name in enumerate(names):
        values = [r["delta_ce_trainable_minus_frozen"] for r in differences if r["activation"] == name]
        axis.scatter(np.full(len(values), index), values, label=name)
        axis.plot([index - .15, index + .15], [np.mean(values)] * 2, color="black")
    axis.axhline(0, color="gray", linewidth=.8)
    axis.set(xticks=range(len(names)), xticklabels=names, ylabel="Validation CE: trainable - frozen", title="Paired seed differences at the same training budget")
    axis.set_xlim(-.5, len(names) - .5)
    axis.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .02, "Each dot is one paired training seed; lower favors a trainable encoder. Validation-only exploratory comparison.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output / f"paired_encoder_effect.{extension}", dpi=600)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("paired", "readout", "all"), default="all")
    parser.add_argument("--paired-run", type=Path, help="Read-only source of paired_results.json for --stage readout.")
    parser.add_argument("--activations", default="classroom9")
    parser.add_argument("--freeze-activations", default="sigmoid,relu,linear")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=.05)
    parser.add_argument("--lr-policy", choices=("shared_step", "constant"), default="shared_step")
    parser.add_argument("--head-epochs", type=int, default=100)
    parser.add_argument("--head-learning-rate", type=float, default=.05)
    parser.add_argument("--head-scales", default="raw,standardized")
    parser.add_argument("--mnist-cache", type=Path, default=Path.home() / ".keras" / "datasets" / "mnist.npz")
    parser.add_argument("--output-parent", type=Path, default=ROOT / "logs")
    parser.add_argument("--synthetic-check", action="store_true")
    args = parser.parse_args()
    args.activations = list(core.CLASSROOM9) if args.activations == "classroom9" else ["binary_step" if a.strip() == "step" else a.strip() for a in args.activations.split(",")]
    args.freeze_activations = [a.strip() for a in args.freeze_activations.split(",") if a.strip()]
    args.head_scales = [s.strip() for s in args.head_scales.split(",")]
    try:
        args.seeds = [int(s.strip()) for s in args.seeds.split(",")]
    except ValueError:
        parser.error("Seeds must be comma-separated integers.")
    if not args.activations or len(set(args.activations)) != len(args.activations) or set(args.activations) - set(core.CLASSROOM9):
        parser.error("Use unique classroom9 activation names.")
    if set(args.freeze_activations) - set(core.CLASSROOM9) or len(set(args.freeze_activations)) != len(args.freeze_activations):
        parser.error("Invalid/duplicate frozen activation names.")
    if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique nonnegative integers.")
    if min(args.epochs, args.head_epochs, args.batch_size) <= 0 or not np.isfinite(args.learning_rate) or not np.isfinite(args.head_learning_rate) or min(args.learning_rate, args.head_learning_rate) <= 0:
        parser.error("Budgets and learning rates must be positive and finite.")
    if set(args.head_scales) - {"raw", "standardized"} or not args.head_scales or len(set(args.head_scales)) != len(args.head_scales):
        parser.error("Head scales: raw,standardized (no duplicates).")
    if args.stage == "readout" and args.paired_run is None:
        parser.error("--stage readout requires --paired-run.")
    if args.paired_run is not None and args.stage != "readout":
        parser.error("--paired-run is only used with --stage readout; it does not resume/retrain old runs.")
    if args.synthetic_check:
        args.seeds, args.epochs, args.head_epochs = [42], 2, 2
        args.batch_size = min(args.batch_size, 16)
    return args


def main() -> None:
    args = parse_args()
    plt.rcParams.update({"font.family": "DejaVu Serif", "svg.fonttype": "none", "pdf.fonttype": 42})
    monitored = [Path(__file__), ROOT / "activation_encoding_core.py", ROOT / "activation_encoding_readout.py"]
    if not args.synthetic_check:
        monitored.append(args.mnist_cache)
    source_records = None
    source_hashes = {}
    if args.stage == "readout":
        source_json = args.paired_run / "paired_results.json"
        source_protocol = args.paired_run / "protocol.json"
        monitored.extend((source_json, source_protocol))
        source_hashes = {str(path.resolve()): sha256(path) for path in (source_json, source_protocol)}
        old_protocol = json.loads(source_protocol.read_text(encoding="utf-8"))
        if old_protocol.get("protocol") != "classroom11_encoding_v1":
            raise ValueError("Readout requires a paired run from this encoding runner, not historical adaptive-LR logs.")
        if bool(old_protocol["scientific_result"]) == args.synthetic_check:
            raise ValueError("Cannot mix synthetic and formal datasets in readout.")
        if old_protocol.get("N") != 2:
            raise ValueError("Readout source must use the project's N=2 architecture convention.")
        expected_split = "synthetic only" if args.synthetic_check else "original final 6000 of MNIST training archive"
        if old_protocol.get("validation_split") != expected_split:
            raise ValueError("Readout source uses a different validation split.")
        source_records = json.loads(source_json.read_text(encoding="utf-8"))
        source_records = [row for row in source_records if row["activation"] in args.activations and row["seed"] in args.seeds]
        if not any(row["condition"] == "trainable" and row["status"] == "completed" for row in source_records):
            raise ValueError("No completed trainable source matches --activations and --seeds.")
        identities = [(row["seed"], row["activation"], row["condition"]) for row in source_records]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate source conditions are not independent seed replicates.")
        if not args.synthetic_check:
            source_cache = str(Path(old_protocol["settings"]["mnist_cache"]).resolve())
            if sha256(args.mnist_cache) != old_protocol["input_sha256"].get(source_cache):
                raise ValueError("MNIST cache differs from the source paired run; no mixed-data readout is allowed.")
        for row in source_records:
            monitored.append(Path(row["initial_path"]))
            if row.get("budget_path"):
                monitored.append(Path(row["budget_path"]))
    original_initial = ORIGINAL / "sigmoid/initial_weights.npz"
    if original_initial.exists():
        monitored.append(original_initial)
    before = {str(path.resolve()): sha256(path) for path in dict.fromkeys(monitored)}
    if any(before[path] != fingerprint for path, fingerprint in source_hashes.items()):
        raise RuntimeError("Source metadata changed while it was being read.")
    x_train, y_train, x_val, y_val, sizes = load_data(args.mnist_cache, args.synthetic_check)
    if args.stage == "readout":
        if old_protocol["architecture"] != sizes or old_protocol["train_samples"] != len(y_train) or old_protocol["validation_samples"] != len(y_val):
            raise ValueError("Source architecture or dataset split differs from the current readout dataset.")
        for source in source_records:
            if source["status"] != "completed":
                continue
            if source["epochs_completed"] != old_protocol["settings"]["epochs"]:
                raise ValueError("A completed source does not match its declared fixed budget.")
            actual = core.load_parameters(Path(source["initial_path"]))
            reference = core.initialize_parameters(sizes, source["seed"], source["activation"])
            if not all(np.array_equal(value, parameter_dict(reference)[name]) for name, value in parameter_dict(actual).items()):
                raise ValueError("Readout source did not use the declared shared Xavier initialization.")
    val_probe = np.random.default_rng(42 + 31415).choice(len(y_val), min(300, len(y_val)), replace=False)
    train_probe = np.random.default_rng(42 + 27182).choice(len(y_train), min(300, len(y_train)), replace=False)
    output = new_run_directory(args.output_parent, "activation_encoding_synthetic_check" if args.synthetic_check else "activation_encoding_N2")
    np.save(output / "fixed_validation_probe_indices.npy", val_probe)
    np.save(output / "fixed_training_probe_indices.npy", train_probe)
    (output / "source_snapshot").mkdir()
    for path in monitored[:3]:
        shutil.copy2(path, output / "source_snapshot" / path.name)
    protocol = {"protocol": "classroom11_encoding_v1", "scientific_result": not args.synthetic_check, "stage": args.stage,
                "architecture": sizes, "N": 2, "train_samples": len(y_train), "validation_samples": len(y_val),
                "validation_split": "original final 6000 of MNIST training archive" if not args.synthetic_check else "synthetic only",
                "test_policy": "not accessed or evaluated", "settings": vars(args),
                "classroom11_mapping": {"hidden_functions": list(core.CLASSROOM9), "VI": "ReLU derivative / persistent probe inactivity diagnostic", "X": "shared output Softmax; not another hidden activation"},
                "training_definition": "manual NumPy mini-batch SGD; fixed budget; no early stopping or validation-driven LR reduction",
                "lr_schedule": "base through 30; base/2 through 40; base/4 through 45; base/8 thereafter" if args.lr_policy == "shared_step" else "constant base",
                "dtype": {"paired_parameters": "float32", "readout_parameters": "float64", "reported_cross_entropy": "float64 logits/logsumexp"},
                "selection": "fixed-budget final primary; minimum validation CE auxiliary",
                "readout_definition": "initial and fixed-budget encoder; same newly initialized head; training-only scaling; raw/standardized separate protocols",
                "probe_role": "fixed diagnostic only; never updates parameters or affects selection; same indices across seeds",
                "ce_definition": "float64 logsumexp of logits, NOT clipped probability CE; differs numerically from historical clipped reporting",
                "parameter_update_path": "actual float32 after-minus-before measured in float64, not eta*mean gradient proxy",
                "input_sha256": before, "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "historical_data_policy": "All historical logs/checkpoints remain read-only; no merging into old adaptive-LR statistics."}
    if args.stage == "readout":
        protocol["paired_source_protocol"] = old_protocol
        protocol["paired_source_directory"] = str(args.paired_run.resolve())
    write_json(output / "protocol.json", protocol)
    paired = source_records or []
    if args.stage in ("paired", "all"):
        for seed in args.seeds:
            template = core.initialize_parameters(sizes, seed, "linear")
            if seed == 42 and not args.synthetic_check and original_initial.exists():
                with np.load(original_initial, allow_pickle=False) as archive:
                    if not all(np.array_equal(value, archive[f"{kind}_{i}"]) for kind, values in (("W", template.weights), ("b", template.biases)) for i, value in enumerate(values, 1)):
                        raise ValueError("Seed42 Xavier initialization does not exactly match the historical reference.")
            for activation in args.activations:
                initial = core.initialize_parameters(sizes, seed, activation)
                if not all(np.array_equal(a, b) for a, b in zip([*template.weights, *template.biases], [*initial.weights, *initial.biases])):
                    raise RuntimeError("Shared initialization check failed.")
                conditions = ["trainable"] + (["frozen"] if activation in args.freeze_activations and activation != "binary_step" else [])
                for condition in conditions:
                    directory = output / "paired" / f"seed_{seed}" / activation / condition
                    paired.append(train_condition(initial, condition, x_train, y_train, x_val, y_val, train_probe, val_probe, directory, seed, args))
                    write_json(output / "paired_results.json", paired)
    heads = run_readouts(paired, x_train, y_train, x_val, y_val, output, args) if args.stage in ("readout", "all") else []
    comparison_outputs(output, paired, heads)
    after = {str(path.resolve()): sha256(path) for path in dict.fromkeys(monitored)}
    if before != after:
        raise RuntimeError("An input source changed while running.")
    write_json(output / "verification.json", {"input_sha256_unchanged": True, "new_test_evaluation": False,
                                               "scientific_result": not args.synthetic_check, "paired_runs": len(paired),
                                               "readout_records": len(heads), "independent_head_fits": sum(r.get("alias_of") is None for r in heads)})
    print(f"OUTPUT: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()


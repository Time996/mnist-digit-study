"""Manual softmax readouts of fixed hidden representations.

This module does not load MNIST, access a test split, or train an encoder.
The experiment runner supplies N x input arrays to ``extract_features`` and
N x hidden arrays to ``fit_output_head``. Raw and train-standardized features
are two explicitly different finite-budget readout protocols, not measures
of all information retained by an encoding. Only W2 and b2 are updated.

The passed initial W2/b2 are copied, with no optimizer state carried over.
Every epoch uses default_rng(seed + 1000003 * epoch), a constant learning
rate, and the full precommitted budget. Validation selects a diagnostic best
checkpoint only; it never changes training, scaling, stopping, or the LR.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from activation_encoding_core import Parameters


SHUFFLE_MULTIPLIER = 1_000_003
STYLE = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _array_hash(array: np.ndarray) -> str:
    """Hash values, shape and dtype without allocating one huge byte string."""
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.dtype.str.encode("ascii"))
    if values.ndim == 0:
        digest.update(values.tobytes())
    else:
        for start in range(0, len(values), 1024):
            digest.update(np.ascontiguousarray(values[start:start + 1024]).tobytes())
    return digest.hexdigest()


def _parameter_hashes(parameters: Parameters) -> list[str]:
    arrays = [*parameters.weights, *parameters.biases]
    if parameters.alpha is not None:
        arrays.append(parameters.alpha)
    return [_array_hash(value) for value in arrays]


def extract_features(x: np.ndarray, p: Parameters, batch_size: int = 1024) -> np.ndarray:
    """Return N x hidden features; verify forward extraction changes no state.

    ``x`` follows the core's public sample-row convention. The core internally
    uses column activations, so only its returned hidden activation is transposed.
    No backward function is called, including for PReLU alpha.
    """
    from activation_encoding_core import forward

    batch_size = _positive_integer(batch_size, "batch_size")
    values = np.asarray(x)
    if values.ndim != 2 or len(values) == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("x must be a nonempty numeric N x input matrix.")
    if values.shape[1] != p.weights[0].shape[1]:
        raise ValueError("Input width does not match the encoder.")
    before = _parameter_hashes(p)
    features = None
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size]
        if not np.all(np.isfinite(batch)):
            raise ValueError("Input features must be finite.")
        hidden = np.asarray(forward(batch, p)["h"])
        if hidden.shape != (p.weights[0].shape[0], len(batch)) or not np.all(np.isfinite(hidden)):
            raise FloatingPointError("Encoder returned invalid hidden activations.")
        if features is None:
            features = np.empty((len(values), hidden.shape[0]), dtype=hidden.dtype)
        features[start:start + len(batch)] = hidden.T
    if before != _parameter_hashes(p):
        raise RuntimeError("Feature extraction mutated encoder/head parameters or PReLU slopes.")
    return features


def fit_standardizer(features: np.ndarray, standardize: bool = True) -> dict[str, np.ndarray]:
    """Fit population moments ONLY on training rows; zero-SD units use scale 1."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Training features must be a nonempty finite N x hidden matrix.")
    train_mean = np.mean(values, axis=0, dtype=np.float64)
    std = np.std(values, axis=0, dtype=np.float64, ddof=0)
    if not np.all(np.isfinite(train_mean)) or not np.all(np.isfinite(std)):
        raise FloatingPointError("Training feature moments overflowed.")
    zero_std = std == 0
    mean = train_mean.copy() if standardize else np.zeros(values.shape[1], dtype=np.float64)
    scale = np.where(zero_std, 1.0, std) if standardize else np.ones(values.shape[1], dtype=np.float64)
    return {"mean": mean, "std": std, "scale": scale,
            "train_mean": train_mean, "zero_std_mask": zero_std}


def transform_features(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply previously fitted moments without fitting or modifying any input."""
    values = np.asarray(features, dtype=np.float64)
    mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if values.ndim != 2 or mean.shape != (values.shape[1],) or scale.shape != mean.shape:
        raise ValueError("Feature matrix and one-dimensional mean/scale widths must match.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Features and moments must be finite, with positive scales.")
    transformed = (values - mean) / scale
    if not np.all(np.isfinite(transformed)):
        raise FloatingPointError("Feature standardization overflowed.")
    return transformed


def head_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Float64 column-convention logits: W2 @ H.T + b2 (classes x batch)."""
    h = np.asarray(features, dtype=np.float64)
    w, b = np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    if h.ndim != 2 or w.ndim != 2 or w.shape[1] != h.shape[1] or b.shape != (w.shape[0], 1):
        raise ValueError("Head dimensions must be H:N x hidden, W2:classes x hidden, b2:classes x 1.")
    with np.errstate(over="ignore", invalid="ignore"):
        logits = w @ h.T + b
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("Non-finite head logits; no clipping or silent stabilization of parameters is used.")
    return logits


def _log_softmax(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # logsumexp on shifted logits avoids exp overflow AND common-offset cancellation.
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=0, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=0, keepdims=True))
    log_probabilities = shifted - log_normalizer
    return log_probabilities, np.exp(log_probabilities)


def _labels(labels: np.ndarray, count: int, classes: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Labels must be a one-dimensional integer array with one label per row.")
    if np.any(values < 0) or np.any(values >= classes):
        raise ValueError("Labels are outside the head's class range.")
    return values.astype(np.int64, copy=False)


def head_gradients(features: np.ndarray, labels: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> dict:
    """Manual mean-CE gradients, online metrics and probabilities for one batch."""
    logits = head_logits(features, weight, bias)
    count = logits.shape[1]
    if count == 0:
        raise ValueError("A head gradient batch must not be empty.")
    y = _labels(labels, count, logits.shape[0])
    log_probabilities, probabilities = _log_softmax(logits)
    delta = probabilities.copy()
    delta[y, np.arange(count)] -= 1.0
    dw = delta @ np.asarray(features, dtype=np.float64) / count
    db = np.mean(delta, axis=1, keepdims=True, dtype=np.float64)
    if not np.all(np.isfinite(dw)) or not np.all(np.isfinite(db)):
        raise FloatingPointError("Non-finite output-head gradient.")
    return {"loss": float(-np.mean(log_probabilities[y, np.arange(count)], dtype=np.float64)),
            "accuracy": float(np.mean(np.argmax(logits, axis=0) == y)),
            "gradient_W2": dw, "gradient_b2": db, "delta": delta,
            "probabilities": probabilities.T}


def evaluate_head(
    features: np.ndarray, labels: np.ndarray, weight: np.ndarray, bias: np.ndarray,
    *, batch_size: int = 1024, return_probabilities: bool = False,
    compute_gradients: bool = False,
) -> dict:
    """Stable endpoint metrics; optionally the true complete-dataset mean gradient.

    Gradients are accumulated as sums BEFORE computing RMS. This is not an
    average of mini-batch gradient RMS values measured at changing parameters.
    Returned probabilities, when requested, have shape N x classes.
    """
    batch_size = _positive_integer(batch_size, "batch_size")
    h = np.asarray(features)
    if h.ndim != 2 or len(h) == 0:
        raise ValueError("Evaluation features must be nonempty N x hidden.")
    w, b = np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    y = _labels(labels, len(h), w.shape[0])
    ce = np.empty(len(h), dtype=np.float64)
    predictions = np.empty(len(h), dtype=np.int64)
    probabilities = np.empty((len(h), w.shape[0]), dtype=np.float64) if return_probabilities else None
    dw_sum, db_sum = np.zeros_like(w), np.zeros_like(b)
    for start in range(0, len(h), batch_size):
        end = min(start + batch_size, len(h))
        logits = head_logits(h[start:end], w, b)
        log_probabilities, p = _log_softmax(logits)
        ce[start:end] = -log_probabilities[y[start:end], np.arange(end - start)]
        predictions[start:end] = np.argmax(logits, axis=0)
        if probabilities is not None:
            probabilities[start:end] = p.T
        if compute_gradients:
            delta = p.copy()
            delta[y[start:end], np.arange(end - start)] -= 1.0
            dw_sum += delta @ np.asarray(h[start:end], dtype=np.float64)
            db_sum += np.sum(delta, axis=1, keepdims=True, dtype=np.float64)
    result = {"loss": float(np.mean(ce, dtype=np.float64)),
              "accuracy": float(np.mean(predictions == y)),
              "predictions": predictions, "per_sample_ce": ce}
    if probabilities is not None:
        result["probabilities"] = probabilities
    if compute_gradients:
        dw, db = dw_sum / len(h), db_sum / len(h)
        if not np.all(np.isfinite(dw)) or not np.all(np.isfinite(db)):
            raise FloatingPointError("Non-finite complete-training-set gradient.")
        result.update(gradient_W2=dw, gradient_b2=db,
                      gradient_rms_W2=_rms(dw), gradient_rms_b2=_rms(db),
                      gradient_rms=_combined_rms(dw, db))
    return result


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _combined_rms(weight_values: np.ndarray, bias_values: np.ndarray) -> float:
    return float(np.sqrt((np.sum(weight_values ** 2, dtype=np.float64) +
                          np.sum(bias_values ** 2, dtype=np.float64)) /
                         (weight_values.size + bias_values.size)))


def _relative_displacement(current: np.ndarray, initial: np.ndarray) -> float | None:
    initial_norm = float(np.linalg.norm(initial))
    return float(np.linalg.norm(current - initial) / initial_norm) if initial_norm > 0 else None


def _write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    names = list(rows[0]) if rows and fields is None else (fields or [])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _save_head(path: Path, weight: np.ndarray, bias: np.ndarray, moments: dict,
               epoch: int, standardize: bool) -> None:
    np.savez_compressed(path, W2=weight, b2=bias, mean=moments["mean"],
                        std=moments["std"], scale=moments["scale"],
                        epoch=np.asarray(epoch), standardize=np.asarray(standardize))


def _save_predictions(path: Path, result: dict, labels: np.ndarray, epoch: int) -> None:
    p = result["probabilities"]
    np.savez_compressed(path, probabilities=p, labels=labels,
                        predictions=result["predictions"], per_sample_ce=result["per_sample_ce"],
                        p_true=p[np.arange(len(labels)), labels], p_max=np.max(p, axis=1),
                        epoch=np.asarray(epoch))


def _save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    figure.savefig(output / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_history(rows: list[dict], output: Path, protocol: str) -> None:
    epochs = [row["epoch"] for row in rows]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(7.8, 5.6))
        for field, label, color in [("endpoint_train_loss", "Endpoint train", "#0072B2"),
                                    ("validation_loss", "Validation", "#D55E00")]:
            axes[0, 0].plot(epochs, [row[field] for row in rows], label=label, color=color)
        for field, label, color in [("endpoint_train_accuracy", "Endpoint train", "#0072B2"),
                                    ("validation_accuracy", "Validation", "#D55E00")]:
            axes[0, 1].plot(epochs, [100 * (1 - row[field]) for row in rows], label=label, color=color)
        for field, label, color in [("full_train_gradient_rms_W2", "W2", "#0072B2"),
                                    ("full_train_gradient_rms_b2", "b2", "#D55E00")]:
            values = np.asarray([row[field] for row in rows])
            positive = values > 0
            axes[1, 0].plot(np.asarray(epochs)[positive], values[positive], label=label, color=color)
            if np.any(~positive):
                axes[1, 0].text(0.02, 0.05 if label == "W2" else 0.14,
                                f"{label}: exact zeros omitted on log axis", transform=axes[1, 0].transAxes, fontsize=8)
        axes[1, 0].set_yscale("log")
        axes[1, 1].plot(epochs, [row["delta_W2_fro"] for row in rows], label="W2", color="#0072B2")
        axes[1, 1].plot(epochs, [row["delta_b2_fro"] for row in rows], label="b2", color="#D55E00")
        titles = ["(a) Cross entropy", "(b) Classification error",
                  "(c) Complete-train gradient RMS", "(d) Displacement from head initialization"]
        ylabels = ["Mean CE (logsumexp)", "Error (%)", "RMS at fixed endpoint", "Frobenius norm of difference"]
        for axis, title, ylabel in zip(axes.flat, titles, ylabels):
            axis.set_title(title, loc="left")
            axis.set(xlabel="Readout training epoch", ylabel=ylabel)
            axis.grid(axis="y", color="0.88", linewidth=0.5)
            axis.legend(frameon=False)
        figure.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.93, wspace=0.37, hspace=0.53)
        figure.text(0.5, 0.025, f"Fixed representation; {protocol}; fixed LR and budget. Diagnostic best is validation-selected.",
                    ha="center", fontsize=8)
        _save_figure(figure, output, "readout_dynamics")


def _validation_images(images: np.ndarray | None, samples: int) -> np.ndarray | None:
    if images is None:
        return None
    values = np.asarray(images)
    if values.ndim == 2 and values.shape == (samples, 784):
        values = values.reshape(samples, 28, 28)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3 or len(values) != samples or not np.all(np.isfinite(values)):
        raise ValueError("validation_images must be finite N x height x width (or N x 784), aligned with validation rows.")
    return values


def _save_errors(result: dict, labels: np.ndarray, images: np.ndarray | None,
                 output: Path, checkpoint: str, epoch: int, protocol: str) -> None:
    p, predictions = result["probabilities"], result["predictions"]
    indices = np.flatnonzero(predictions != labels)
    sorted_p = np.sort(p, axis=1)
    rows = [{"validation_index": int(i), "true_label": int(labels[i]),
             "prediction": int(predictions[i]), "p_true": float(p[i, labels[i]]),
             "predicted_class_probability": float(p[i, predictions[i]]),
             "top1_minus_top2": float(sorted_p[i, -1] - sorted_p[i, -2]),
             "cross_entropy": float(result["per_sample_ce"][i]),
             "checkpoint": checkpoint, "epoch": epoch} for i in indices]
    fields = ["validation_index", "true_label", "prediction", "p_true",
              "predicted_class_probability", "top1_minus_top2", "cross_entropy", "checkpoint", "epoch"]
    _write_csv(output / f"{checkpoint}_validation_errors.csv", rows, fields)
    selected = indices[:25]  # Stable index order, not hand-picked examples.
    with plt.rc_context(STYLE):
        if len(selected) == 0:
            figure, axis = plt.subplots(figsize=(7.8, 2.1))
            axis.axis("off")
            axis.text(0.5, 0.5, "No validation errors at this checkpoint.", ha="center", va="center")
        else:
            columns = min(5, len(selected))
            row_count = (len(selected) + columns - 1) // columns
            figure, axes = plt.subplots(row_count, columns, figsize=(7.8, 2.05 * row_count + 1.2), squeeze=False)
            for axis in axes.flat:
                axis.axis("off")
            for axis, i in zip(axes.flat, selected):
                if images is not None:
                    vmax = 1.0 if np.max(images[i]) <= 1 else 255.0
                    axis.imshow(images[i], cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
                else:
                    axis.axis("on")
                    colors = ["0.75"] * p.shape[1]
                    colors[int(labels[i])], colors[int(predictions[i])] = "#0072B2", "#D55E00"
                    axis.bar(np.arange(p.shape[1]), p[i], color=colors, width=0.8)
                    axis.set(ylim=(0, 1), xticks=np.arange(p.shape[1]), yticks=[0, 1])
                    axis.tick_params(labelsize=6)
                axis.set_title(f"{i}: {labels[i]} → {predictions[i]}\n"
                               f"p(y)={p[i, labels[i]]:.3f}\n"
                               f"p(pred)={p[i, predictions[i]]:.3f}", fontsize=8)
            figure.subplots_adjust(left=0.045, right=0.98, bottom=0.13, top=0.84, hspace=0.95, wspace=0.42)
        gallery = "raw validation images" if images is not None else "class-probability bars (no images supplied)"
        figure.suptitle(f"{checkpoint.capitalize()} readout, epoch {epoch}: first {len(selected)} of {len(indices)} validation errors", fontsize=10, y=0.98)
        figure.text(0.5, 0.035, f"{gallery}; ascending validation index; {protocol}.\n"
                    "Titles: validation index: y → pred. p(y) = dataset-label probability; p(pred) = predicted-class probability.",
                    ha="center", fontsize=8)
        _save_figure(figure, output, f"{checkpoint}_validation_errors")


def fit_output_head(
    htrain: np.ndarray, ytrain: np.ndarray, hval: np.ndarray, yval: np.ndarray,
    initial_weight: np.ndarray, initial_bias: np.ndarray,
    *, epochs: int, learning_rate: float, batch_size: int, seed: int,
    standardize: bool, output: Path, validation_images: np.ndarray | None = None,
    gradient_rms_threshold: float = 1e-5,
) -> dict:
    """Train only a copied NumPy softmax head and write a NEW experiment folder.

    The returned best accuracy belongs to the minimum-validation-CE head,
    including epoch 0 if no later epoch improves. Final and best stay distinct.
    The predeclared gradient threshold is diagnostic only and never ends a run.
    It uses RMS over concatenated full-training dW2/db2 entries; meeting it
    does not certify convergence, an optimum, or an optimal encoding/readout.
    """
    epochs, batch_size = _positive_integer(epochs, "epochs"), _positive_integer(batch_size, "batch_size")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not isinstance(standardize, (bool, np.bool_)):
        raise ValueError("standardize must explicitly be True or False.")
    if not np.isfinite(learning_rate) or learning_rate <= 0 or not np.isfinite(gradient_rms_threshold) or gradient_rms_threshold <= 0:
        raise ValueError("learning_rate and gradient_rms_threshold must be finite and positive.")
    seed, standardize = int(seed), bool(standardize)
    learning_rate, gradient_rms_threshold = float(learning_rate), float(gradient_rms_threshold)
    train, val = np.asarray(htrain), np.asarray(hval)
    if train.ndim != 2 or val.ndim != 2 or min(train.shape) == 0 or min(val.shape) == 0 or train.shape[1] != val.shape[1]:
        raise ValueError("Train and validation features must be nonempty N x hidden matrices of matching width.")
    w0 = np.array(initial_weight, dtype=np.float64, copy=True)
    b0 = np.array(initial_bias, dtype=np.float64, copy=True)
    if b0.ndim == 1:
        b0 = b0.reshape(-1, 1)
    if w0.ndim != 2 or w0.shape[0] < 2 or w0.shape[1] != train.shape[1] or b0.shape != (w0.shape[0], 1):
        raise ValueError("Initial W2/b2 dimensions are invalid.")
    if not np.all(np.isfinite(w0)) or not np.all(np.isfinite(b0)):
        raise ValueError("Initial head parameters must be finite.")
    yt, yv = _labels(ytrain, len(train), w0.shape[0]), _labels(yval, len(val), w0.shape[0])
    images = _validation_images(validation_images, len(val))
    moments = fit_standardizer(train, standardize)
    ht = transform_features(train, moments["mean"], moments["scale"])
    hv = transform_features(val, moments["mean"], moments["scale"])
    protocol = "standardized" if standardize else "raw"
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)  # Never resume/overwrite an existing run.
    predictions_dir = directory / "validation_epochs"
    predictions_dir.mkdir(exist_ok=False)
    np.savez_compressed(directory / "standardization.npz", **moments, standardize=np.asarray(standardize))
    script_path = Path(__file__).resolve()
    config = {
        "experiment": "Fixed-representation, manually trained softmax readout", "protocol": protocol,
        "standardize": standardize, "dtype": "float64 head/logits/CE/gradient accumulation",
        "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed,
        "train_samples": len(ht), "validation_samples": len(hv), "hidden_width": ht.shape[1], "classes": w0.shape[0],
        "optimizer": "manual NumPy mini-batch gradient descent; no momentum, regularization or inherited state",
        "shuffle_control": "default_rng(seed + 1000003*epoch).permutation(train_samples)",
        "schedule": "constant LR, fixed epoch budget; no validation-driven LR or early stopping",
        "updated_parameters": "output W2 and b2 only; encoder W1/b1/alpha are never passed to this trainer",
        "scaling_fit_split": "training features ONLY; population SD ddof=0; zero SD uses scale 1",
        "head_initialization": "exact caller-provided initial W2/b2, copied and converted to float64",
        "model_selection": "minimum validation CE over epoch0 and all completed epochs; strict improvement",
        "primary_budget_endpoint": "final checkpoint, saved separately from validation-selected best",
        "loss": "mean cross entropy from float64 logits and shifted logsumexp, without probability clipping",
        "gradient_threshold": gradient_rms_threshold,
        "gradient_threshold_definition": "RMS of concatenated complete-training-set mean dW2/db2, evaluated at final head",
        "gradient_threshold_role": "predeclared diagnostic only; no training/stopping consequence; not an optimality certificate",
        "mini_batch_gradient_definition": "sample-count-weighted mean of RMS/norms at changing pre-update batch parameters",
        "full_train_gradient_definition": "sum of per-sample gradient contributions at one fixed endpoint, divided by train count BEFORE RMS",
        "interpretation_limit": "finite-budget affine-softmax readout; raw and standardized are different protocols, not pure encoding information",
        "test_evaluated": False, "test_arrays_accessed_by_module": False,
        "input_sha256": {"train_features": _array_hash(train), "validation_features": _array_hash(val),
                         "train_labels": _array_hash(yt), "validation_labels": _array_hash(yv),
                         "initial_W2": _array_hash(w0), "initial_b2": _array_hash(b0)},
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    _write_json(directory / "config.json", config)
    w, b = w0.copy(), b0.copy()
    _save_head(directory / "initial_head.npz", w, b, moments, 0, standardize)
    rows = []
    started = perf_counter()
    best_epoch, best_loss, best_accuracy = 0, float("inf"), None
    best_w, best_b, best_validation = w.copy(), b.copy(), None
    final_train, final_validation = None, None
    try:
        for epoch in range(0, epochs + 1):
            epoch_started = perf_counter()
            online_loss, online_correct, online_seen = 0.0, 0.0, 0
            rms_w_sum, rms_b_sum, fro_w_sum, fro_b_sum, delta_sum = 0.0, 0.0, 0.0, 0.0, 0.0
            if epoch > 0:
                order = np.random.default_rng(seed + SHUFFLE_MULTIPLIER * epoch).permutation(len(ht))
                for start in range(0, len(order), batch_size):
                    indices = order[start:start + batch_size]
                    result = head_gradients(ht[indices], yt[indices], w, b)
                    count = len(indices)
                    dw, db = result["gradient_W2"], result["gradient_b2"]
                    online_loss += result["loss"] * count
                    online_correct += result["accuracy"] * count
                    online_seen += count
                    rms_w_sum += _rms(dw) * count
                    rms_b_sum += _rms(db) * count
                    fro_w_sum += float(np.linalg.norm(dw)) * count
                    fro_b_sum += float(np.linalg.norm(db)) * count
                    delta_sum += _rms(result["delta"]) * count
                    with np.errstate(over="ignore", invalid="ignore"):
                        w -= learning_rate * dw
                        b -= learning_rate * db
                    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                        raise FloatingPointError(f"Non-finite head parameter at epoch {epoch}, batch start {start}.")
            final_train = evaluate_head(ht, yt, w, b, compute_gradients=True)
            final_validation = evaluate_head(hv, yv, w, b, return_probabilities=True)
            if not np.isfinite(final_train["loss"]) or not np.isfinite(final_validation["loss"]):
                raise FloatingPointError(f"Non-finite endpoint loss at epoch {epoch}.")
            _save_predictions(predictions_dir / f"epoch_{epoch:04d}.npz", final_validation, yv, epoch)
            row = {
                "epoch": epoch, "learning_rate": learning_rate,
                "online_train_loss": online_loss / online_seen if online_seen else None,
                "online_train_accuracy": online_correct / online_seen if online_seen else None,
                "endpoint_train_loss": final_train["loss"], "endpoint_train_accuracy": final_train["accuracy"],
                "validation_loss": final_validation["loss"], "validation_accuracy": final_validation["accuracy"],
                "gradient_rms_W2": rms_w_sum / online_seen if online_seen else None,
                "gradient_rms_b2": rms_b_sum / online_seen if online_seen else None,
                "gradient_fro_W2": fro_w_sum / online_seen if online_seen else None,
                "gradient_fro_b2": fro_b_sum / online_seen if online_seen else None,
                "delta_rms_L2": delta_sum / online_seen if online_seen else None,
                "full_train_gradient_rms_W2": final_train["gradient_rms_W2"],
                "full_train_gradient_rms_b2": final_train["gradient_rms_b2"],
                "full_train_gradient_rms": final_train["gradient_rms"],
                "delta_W2_fro": float(np.linalg.norm(w - w0)), "delta_b2_fro": float(np.linalg.norm(b - b0)),
                "relative_delta_W2": _relative_displacement(w, w0), "relative_delta_b2": _relative_displacement(b, b0),
                "epoch_seconds": perf_counter() - epoch_started, "elapsed_seconds": perf_counter() - started,
            }
            rows.append(row)
            _write_csv(directory / "training_log.csv", rows)
            if final_validation["loss"] < best_loss:
                best_epoch, best_loss, best_accuracy = epoch, final_validation["loss"], final_validation["accuracy"]
                best_w, best_b = w.copy(), b.copy()
                best_validation = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in final_validation.items()}
                _save_head(directory / "best_head.npz", best_w, best_b, moments, epoch, standardize)
        _save_head(directory / "final_head.npz", w, b, moments, epochs, standardize)
    except Exception as error:
        _write_json(directory / "failure_diagnostics.json", {
            "status": "failed", "reason": str(error), "completed_epochs_in_log": [row["epoch"] for row in rows],
            "last_attempted_epoch": epoch, "elapsed_seconds": perf_counter() - started,
            "test_evaluated": False,
        })
        _save_head(directory / "failed_head.npz", w, b, moments, epoch, standardize)
        raise
    _save_predictions(directory / "initial_validation_predictions.npz",
                      evaluate_head(hv, yv, w0, b0, return_probabilities=True), yv, 0)
    _save_predictions(directory / "best_validation_predictions.npz", best_validation, yv, best_epoch)
    _save_predictions(directory / "final_validation_predictions.npz", final_validation, yv, epochs)
    # Explicit alias for consumers: this is the CE-selected head, NOT the final head.
    _save_predictions(directory / "validation_predictions.npz", best_validation, yv, best_epoch)
    np.savez_compressed(directory / "final_full_train_gradient.npz",
                        gradient_W2=final_train["gradient_W2"], gradient_b2=final_train["gradient_b2"],
                        gradient_rms=np.asarray(final_train["gradient_rms"]),
                        threshold=np.asarray(gradient_rms_threshold), epoch=np.asarray(epochs))
    threshold_met = bool(final_train["gradient_rms"] <= gradient_rms_threshold)
    summary = {
        "status": "completed", "protocol": protocol, "standardize": standardize, "epochs_completed": epochs,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "best_checkpoint_validation_accuracy": best_accuracy,
        "best_validation_accuracy": best_accuracy,  # CE-selected accuracy, never a peak-accuracy mixture.
        "final_validation_loss": final_validation["loss"], "final_validation_accuracy": final_validation["accuracy"],
        "final_endpoint_train_loss": final_train["loss"], "final_endpoint_train_accuracy": final_train["accuracy"],
        "final_full_train_gradient_rms": final_train["gradient_rms"],
        "final_full_train_gradient_rms_W2": final_train["gradient_rms_W2"],
        "final_full_train_gradient_rms_b2": final_train["gradient_rms_b2"],
        "gradient_rms_threshold": gradient_rms_threshold, "gradient_threshold_met": threshold_met,
        "optimization_status": "finite_budget_readout; diagnostic gradient threshold " + ("met" if threshold_met else "not met"),
        "convergence_claim": "None: this first-order threshold is not a certificate of an optimal readout, convergence or generalization.",
        "head_updated_parameter_count": w.size + b.size, "encoder_updated_parameter_count": 0,
        "zero_std_training_units": int(np.sum(moments["zero_std_mask"])),
        "final_delta_W2_fro": float(np.linalg.norm(w - w0)), "final_relative_delta_W2": _relative_displacement(w, w0),
        "final_delta_b2_fro": float(np.linalg.norm(b - b0)),
        "elapsed_training_and_diagnostic_seconds": perf_counter() - started,
        "test_evaluated": False, "output_directory": str(directory.resolve()),
        "validation_predictions_alias": "validation_predictions.npz is best_validation_predictions, not final",
    }
    _write_json(directory / "summary.json", summary)
    _plot_history(rows, directory, protocol)
    _save_errors(best_validation, yv, images, directory, "best", best_epoch, protocol)
    _save_errors(final_validation, yv, images, directory, "final", epochs, protocol)
    _write_json(directory / "artifact_manifest.json", {
        "head_checkpoints": ["initial_head.npz", "best_head.npz", "final_head.npz"],
        "saved_head_arrays": "W2, b2, mean, std, scale, epoch, standardize; mean/scale reproduce transforms",
        "validation_epoch_predictions": "validation_epochs/epoch_0000.npz through the fixed budget endpoint",
        "error_gallery_selection": "first min(25,total errors) in ascending validation row order",
        "error_gallery_type": "raw supplied validation images" if images is not None else "probability bars; no validation images supplied",
        "probability_semantics": "p_true is dataset-label probability; predicted_class_probability is p(argmax)",
        "figures": "600 dpi PNG and editable-text SVG; native matplotlib, no generated/altered source images",
        "input_values_unchanged": all(config["input_sha256"][key] == _array_hash(value) for key, value in
                                      [("train_features", train), ("validation_features", val),
                                       ("train_labels", yt), ("validation_labels", yv)]),
        "interpretation": "No optimum or causal encoding-versus-optimization contribution is claimed.",
    })
    return summary


if __name__ == "__main__":
    print(__doc__)
    print("This is an importable readout module; use the experiment runner. No data or training was executed.")


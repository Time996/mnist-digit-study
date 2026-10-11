"""Manual NumPy mathematics for the classroom N=2 activation experiment.

Public inputs are samples x features; internal activations are column vectors.
This module loads no datasets, starts no training and uses no autodifferentiation.
The caller owns fresh experiment directories and diagnostic/model-selection rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASSROOM9 = (
    "binary_step", "linear", "sigmoid", "tanh", "relu", "leaky_relu",
    "prelu", "elu", "swish",
)
LEAKY_ALPHA = 0.01
PRELU_ALPHA_INIT = 0.01
LOW_DERIVATIVE_THRESHOLD = 0.05
ACTIVATION_FORMULAS = {
    "binary_step": "1[z>0]; derivative zero, including the chosen convention at zero; no STE",
    "linear": "z; derivative 1",
    "sigmoid": "1/(1+exp(-z)); derivative sigmoid(z)*(1-sigmoid(z))",
    "tanh": "tanh(z); derivative 1-tanh(z)^2",
    "relu": "max(z,0); derivative 1[z>0], convention zero at zero",
    "leaky_relu": "z if z>0 else 0.01*z; derivative 1 if z>0 else 0.01",
    "prelu": "z if z>0 else alpha_j*z; unconstrained per-unit alpha, initially 0.01",
    "elu": "z if z>0 else expm1(z); alpha=1",
    "swish": "z*sigmoid(z); beta=1",
}


@dataclass
class Parameters:
    activation: str
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    alpha: np.ndarray | None = None


def _name(activation: str) -> str:
    name = "binary_step" if activation == "step" else activation
    if name not in CLASSROOM9:
        raise ValueError(f"Unsupported activation: {activation!r}")
    return name


def _finite(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"Non-finite {name}.")


def _validate_parameters(p: Parameters) -> tuple[int, int, int]:
    name = _name(p.activation)
    if len(p.weights) != 2 or len(p.biases) != 2:
        raise ValueError("N=2 requires exactly two weights and two biases.")
    w1, w2 = p.weights
    if w1.ndim != 2 or w2.ndim != 2:
        raise ValueError("Weights must be matrices.")
    hidden, inputs = w1.shape
    classes, second_hidden = w2.shape
    if inputs < 1 or hidden < 1 or classes < 2 or second_hidden != hidden:
        raise ValueError("Invalid input/hidden/output dimensions.")
    if p.biases[0].shape != (hidden, 1) or p.biases[1].shape != (classes, 1):
        raise ValueError("Biases must have shape (layer width, 1).")
    if w1.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Use float32 or float64 parameters.")
    arrays = [*p.weights, *p.biases]
    if (name == "prelu") != (p.alpha is not None):
        raise ValueError("Only PReLU must have an alpha parameter.")
    if p.alpha is not None:
        if p.alpha.shape != (hidden, 1):
            raise ValueError("PReLU alpha must be per-unit, shape (hidden, 1).")
        arrays.append(p.alpha)
    for index, array in enumerate(arrays):
        if array.dtype != w1.dtype:
            raise ValueError("All parameter arrays must share one dtype.")
        _finite(f"parameter array {index}", array)
    return inputs, hidden, classes


def _input(x: np.ndarray, p: Parameters) -> np.ndarray:
    inputs, _, _ = _validate_parameters(p)
    values = np.asarray(x)
    if values.ndim != 2 or values.shape[1] != inputs or len(values) == 0:
        raise ValueError("Input must be a nonempty samples x features matrix.")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("Input must be real numeric data.")
    with np.errstate(over="raise", invalid="raise"):
        values = values.astype(p.weights[0].dtype, copy=False)
    _finite("input", values)
    return values


def _labels(y: np.ndarray, samples: int, classes: int) -> np.ndarray:
    values = np.asarray(y)
    if values.shape != (samples,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Labels must be a one-dimensional integer array.")
    if samples == 0 or np.any(values < 0) or np.any(values >= classes):
        raise ValueError("Labels are empty or outside the output class range.")
    return values.astype(np.int64, copy=False)


def initialize_parameters(
    sizes: list[int], seed: int, activation: str, dtype=np.float32,
) -> Parameters:
    """Identical seeded Xavier W/b for every function; alpha consumes no RNG."""
    name = _name(activation)
    if len(sizes) != 3 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in sizes):
        raise ValueError("Use three positive integer sizes for N=2.")
    if sizes[-1] < 2:
        raise ValueError("Softmax requires at least two classes.")
    chosen_dtype = np.dtype(dtype)
    if chosen_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Use float32 or float64.")
    rng = np.random.default_rng(seed)
    weights, biases = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        weights.append(rng.uniform(-bound, bound, (fan_out, fan_in)).astype(chosen_dtype))
        biases.append(np.zeros((fan_out, 1), dtype=chosen_dtype))
    alpha = np.full((sizes[1], 1), PRELU_ALPHA_INIT, dtype=chosen_dtype) if name == "prelu" else None
    return Parameters(name, weights, biases, alpha)


def copy_parameters(p: Parameters) -> Parameters:
    _validate_parameters(p)
    return Parameters(_name(p.activation), [w.copy() for w in p.weights],
                      [b.copy() for b in p.biases], None if p.alpha is None else p.alpha.copy())


def save_parameters(path: str | Path, p: Parameters) -> None:
    """Save a complete checkpoint; the caller must guard historical directories."""
    sizes = _validate_parameters(p)
    values = {
        "N": np.asarray(2), "layer_sizes": np.asarray(sizes),
        "activation": np.asarray(_name(p.activation)),
        "W_1": p.weights[0], "b_1": p.biases[0],
        "W_2": p.weights[1], "b_2": p.biases[1],
    }
    if p.alpha is not None:
        values["alpha_H1"] = p.alpha
    # A stream prevents NumPy from silently appending an extra suffix.
    with Path(path).open("wb") as stream:
        np.savez(stream, **values)


def load_parameters(path: str | Path) -> Parameters:
    with np.load(path, allow_pickle=False) as values:
        p = Parameters(
            _name(str(values["activation"].item())),
            [values["W_1"].copy(), values["W_2"].copy()],
            [values["b_1"].copy(), values["b_2"].copy()],
            values["alpha_H1"].copy() if "alpha_H1" in values else None,
        )
        sizes = _validate_parameters(p)
        if "N" in values and int(values["N"].item()) != 2:
            raise ValueError("Checkpoint is not N=2.")
        if "layer_sizes" in values and not np.array_equal(values["layer_sizes"], sizes):
            raise ValueError("Checkpoint architecture metadata is inconsistent.")
    return p


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Branch-stable sigmoid without clipping its mathematical argument."""
    values = np.asarray(z)
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        values = values.astype(np.float64)
    _finite("sigmoid argument", values)
    result = np.empty_like(values)
    positive = values >= 0
    with np.errstate(under="ignore"):
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponential = np.exp(values[~positive])
        result[~positive] = exponential / (1.0 + exponential)
    return result


def hidden_activation(z: np.ndarray, p: Parameters) -> np.ndarray:
    name = _name(p.activation)
    if name == "binary_step":
        result = (z > 0).astype(z.dtype)
    elif name == "linear":
        result = z.copy()
    elif name == "sigmoid":
        result = stable_sigmoid(z)
    elif name == "tanh":
        result = np.tanh(z)
    elif name == "relu":
        result = np.maximum(z, 0)
    elif name in ("leaky_relu", "prelu"):
        alpha = LEAKY_ALPHA if name == "leaky_relu" else p.alpha
        result = np.where(z > 0, z, alpha * z)
    elif name == "elu":
        result = z.copy()
        negative = z <= 0
        with np.errstate(under="ignore"):
            result[negative] = np.expm1(z[negative])
    else:
        result = z * stable_sigmoid(z)
    _finite("hidden activation", result)
    return result


def hidden_derivative(z: np.ndarray, h: np.ndarray, p: Parameters) -> np.ndarray:
    name = _name(p.activation)
    if name == "binary_step":
        result = np.zeros_like(z)
    elif name == "linear":
        result = np.ones_like(z)
    elif name == "sigmoid":
        result = h * (1 - h)
    elif name == "tanh":
        result = 1 - h * h
    elif name == "relu":
        result = (z > 0).astype(z.dtype)
    elif name in ("leaky_relu", "prelu"):
        alpha = LEAKY_ALPHA if name == "leaky_relu" else p.alpha
        result = np.where(z > 0, 1.0, alpha).astype(z.dtype)
    elif name == "elu":
        result = np.ones_like(z)
        negative = z <= 0
        with np.errstate(under="ignore"):
            result[negative] = np.exp(z[negative])
    else:
        sigmoid = stable_sigmoid(z)
        result = sigmoid + z * sigmoid * (1 - sigmoid)
    _finite("hidden derivative", result)
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("Logits must have shape classes x nonempty batch.")
    _finite("logits", values)
    values64 = values.astype(np.float64)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        shifted = values64 - values64.max(axis=0, keepdims=True)
        exponential = np.exp(shifted)
        result = exponential / exponential.sum(axis=0, keepdims=True)
    result = result.astype(values.dtype if values.dtype in (np.float32, np.float64) else np.float64)
    _finite("softmax probabilities", result)
    return result


def per_sample_loss_from_logits(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Float64 logsumexp CE, with neither probability clipping nor a loss cap."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Logits must be classes x batch.")
    labels = _labels(y, values.shape[1], values.shape[0])
    _finite("loss logits", values)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        shifted = values - values.max(axis=0, keepdims=True)
        losses = np.log(np.exp(shifted).sum(axis=0)) - shifted[labels, np.arange(len(labels))]
    _finite("cross entropy", losses)
    return losses


def cross_entropy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    return float(per_sample_loss_from_logits(logits, y).mean(dtype=np.float64))


def forward(x: np.ndarray, p: Parameters) -> dict[str, np.ndarray]:
    values = _input(x, p)
    a0 = values.T
    with np.errstate(over="raise", invalid="raise"):
        z = p.weights[0] @ a0 + p.biases[0]
        _finite("hidden preactivation", z)
        h = hidden_activation(z, p)
        logits = p.weights[1] @ h + p.biases[1]
    return {"a0": a0, "z": z, "h": h, "logits": logits,
            "probabilities": softmax(logits)}


def backward(cache: dict, y: np.ndarray, p: Parameters) -> dict:
    """Manual chain rule; all gradients refer to the same pre-update parameters."""
    inputs, hidden, classes = _validate_parameters(p)
    samples = cache["a0"].shape[1]
    labels = _labels(y, samples, classes)
    expected = {"a0": (inputs, samples), "z": (hidden, samples), "h": (hidden, samples),
                "logits": (classes, samples), "probabilities": (classes, samples)}
    for name, shape in expected.items():
        if cache[name].shape != shape:
            raise ValueError(f"Cache {name} has an invalid shape.")
        _finite(f"cache {name}", cache[name])
    delta2 = cache["probabilities"].copy()
    delta2[labels, np.arange(samples)] -= 1
    with np.errstate(over="raise", invalid="raise"):
        upstream = p.weights[1].T @ delta2
        derivative = hidden_derivative(cache["z"], cache["h"], p)
        if _name(p.activation) == "binary_step":
            # Exact zero is the selected true-derivative convention, not a STE.
            delta1 = np.zeros_like(cache["z"])
            dw1 = np.zeros_like(p.weights[0])
            db1 = np.zeros_like(p.biases[0])
        else:
            delta1 = upstream * derivative
            dw1 = delta1 @ cache["a0"].T / samples
            db1 = delta1.mean(axis=1, keepdims=True)
        dw2 = delta2 @ cache["h"].T / samples
        db2 = delta2.mean(axis=1, keepdims=True)
        # U, not delta1: delta1 already contains alpha on the negative branch.
        dalpha = (upstream * np.minimum(cache["z"], 0)).mean(axis=1, keepdims=True) if p.alpha is not None else None
    gradients = {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2, "alpha": dalpha,
                 "U": upstream, "delta1": delta1, "delta2": delta2, "derivative": derivative}
    for name, value in gradients.items():
        if value is not None:
            _finite(f"gradient/diagnostic {name}", value)
    return gradients


def parameter_dict(p: Parameters) -> dict[str, np.ndarray]:
    """References, not copies; useful for explicitly labelled parameter diagnostics."""
    values = {"W1": p.weights[0], "b1": p.biases[0], "W2": p.weights[1], "b2": p.biases[1]}
    if p.alpha is not None:
        values["alpha"] = p.alpha
    return values


def frobenius(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    _finite("norm argument", array)
    result = float(np.linalg.norm(array.ravel()))
    if not np.isfinite(result):
        raise FloatingPointError("Non-finite float64 Frobenius norm.")
    return result


def rms(values: np.ndarray) -> float:
    array = np.asarray(values)
    if not array.size:
        raise ValueError("RMS is undefined for an empty array.")
    return frobenius(array) / np.sqrt(array.size)


def update_parameters(p: Parameters, gradients: dict, lr: float, train_encoder: bool = True) -> dict[str, float]:
    """Atomic SGD update; return real float64 after-minus-before norms.

    Freezing the encoder freezes alpha too. The alpha key is zero when absent,
    and does not imply an extra parameter exists for non-PReLU models.
    """
    _validate_parameters(p)
    if not np.isfinite(lr) or lr < 0:
        raise ValueError("Learning rate must be finite and nonnegative.")
    arrays = parameter_dict(p)
    proposed = {}
    update_norms = dict.fromkeys(("W1", "b1", "W2", "b2", "alpha"), 0.0)
    for name, array in arrays.items():
        gradient = gradients[name]
        if gradient is None or gradient.shape != array.shape:
            raise ValueError(f"Gradient {name} has an invalid shape.")
        _finite(f"update gradient {name}", gradient)
        if not train_encoder and name in ("W1", "b1", "alpha"):
            continue
        with np.errstate(over="raise", invalid="raise"):
            candidate = (array - lr * gradient).astype(array.dtype, copy=False)
        _finite(f"proposed {name}", candidate)
        update_norms[name] = frobenius(candidate.astype(np.float64) - array.astype(np.float64))
        proposed[name] = candidate
    # No parameter has been changed until every proposed value passes checks.
    for name, candidate in proposed.items():
        arrays[name][...] = candidate
    return update_norms


def evaluate_dataset(x: np.ndarray, y: np.ndarray, p: Parameters, batch_size: int = 1024) -> dict:
    values = _input(x, p)
    classes = p.weights[1].shape[0]
    labels = _labels(y, len(values), classes)
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    probabilities = np.empty((len(values), classes), dtype=p.weights[0].dtype)
    logits = np.empty_like(probabilities)
    losses = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))
        cache = forward(values[start:end], p)
        probabilities[start:end] = cache["probabilities"].T
        logits[start:end] = cache["logits"].T
        losses[start:end] = per_sample_loss_from_logits(cache["logits"], labels[start:end])
    predictions = probabilities.argmax(axis=1)
    return {"loss": float(losses.mean(dtype=np.float64)),
            "accuracy": float(np.mean(predictions == labels)),
            "probabilities": probabilities, "logits": logits,
            "per_sample_loss": losses, "predictions": predictions}


def _distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _finite(prefix, array)
    quantiles = np.quantile(array, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {f"{prefix}_mean": float(array.mean()), f"{prefix}_std": float(array.std(ddof=0)),
            f"{prefix}_rms": rms(array), f"{prefix}_min": float(array.min()),
            f"{prefix}_max": float(array.max()),
            **{f"{prefix}_{key}": float(value) for key, value in zip(("p01", "p05", "p50", "p95", "p99"), quantiles)}}


def probe_diagnostics(x: np.ndarray, y: np.ndarray, p: Parameters) -> tuple[dict, dict, dict]:
    """Non-mutating fixed-probe diagnostics; not training-epoch aggregates.

    Every backward value uses one whole probe batch. Unit arrays preserve unit
    identities; pooled SD is descriptive dispersion, never a confidence interval.
    A unit zero on this finite probe is not proven dead on the full dataset.
    None represents a non-applicable saturation measure or an undefined ratio.
    """
    cache = forward(x, p)
    labels = _labels(y, cache["h"].shape[1], p.weights[1].shape[0])
    gradients = backward(cache, labels, p)
    h, z, derivative = cache["h"], cache["z"], gradients["derivative"]
    scalars = {"probe_samples": int(len(labels)),
               "probe_loss": cross_entropy_from_logits(cache["logits"], labels),
               "probe_accuracy": float(np.mean(cache["probabilities"].argmax(axis=0) == labels))}
    for name, values in (("z", z), ("h", h), ("derivative", derivative)):
        scalars.update(_distribution(values, f"probe_{name}"))
    scalars.update({
        "probe_derivative_mean_abs": float(np.mean(np.abs(derivative.astype(np.float64)))),
        "probe_derivative_zero_fraction": float(np.mean(derivative == 0)),
        "probe_derivative_negative_fraction": float(np.mean(derivative < 0)),
        "probe_low_derivative_fraction": float(np.mean(np.abs(derivative) < LOW_DERIVATIVE_THRESHOLD)),
        "probe_h_zero_fraction": float(np.mean(h == 0)),
        "probe_h_positive_fraction": float(np.mean(h > 0)),
        "probe_h_negative_fraction": float(np.mean(h < 0)),
        "probe_z_positive_fraction": float(np.mean(z > 0)),
    })
    for name in ("U", "delta1", "delta2"):
        scalars[f"probe_{name}_rms"] = rms(gradients[name])
    upstream_rms = scalars["probe_U_rms"]
    output_rms = scalars["probe_delta2_rms"]
    scalars["probe_upstream_weight_gain"] = upstream_rms / output_rms if output_rms > 0 else None
    scalars["probe_local_jacobian_gain"] = scalars["probe_delta1_rms"] / upstream_rms if upstream_rms > 0 else None
    scalars["probe_delta1_delta2_ratio"] = scalars["probe_delta1_rms"] / output_rms if output_rms > 0 else None
    for name in ("W1", "b1", "W2", "b2", "alpha"):
        if gradients[name] is not None:
            gradient64 = gradients[name].astype(np.float64)
            prefix = f"probe_gradient_{name}"
            scalars.update({f"{prefix}_fro": frobenius(gradient64),
                            f"{prefix}_rms": rms(gradient64),
                            f"{prefix}_mean": float(gradient64.mean()),
                            f"{prefix}_std": float(gradient64.std(ddof=0))})
    h64, z64 = h.astype(np.float64), z.astype(np.float64)
    unit_arrays = {
        "unit_index": np.arange(h.shape[0]),
        "h_mean": h64.mean(axis=1), "h_std": h64.std(axis=1, ddof=0),
        "h_rms": np.sqrt(np.mean(h64 * h64, axis=1)),
        "z_mean": z64.mean(axis=1), "z_std": z64.std(axis=1, ddof=0),
        "h_zero_fraction": np.mean(h == 0, axis=1),
        "h_positive_fraction": np.mean(h > 0, axis=1),
        "h_negative_fraction": np.mean(h < 0, axis=1),
        "z_positive_fraction": np.mean(z > 0, axis=1),
        "derivative_zero_fraction": np.mean(derivative == 0, axis=1),
        "derivative_negative_fraction": np.mean(derivative < 0, axis=1),
        "derivative_mean_abs": np.mean(np.abs(derivative.astype(np.float64)), axis=1),
        "all_probe_zero": np.all(h == 0, axis=1),
    }
    scalars["probe_all_zero_unit_count"] = int(unit_arrays["all_probe_zero"].sum())
    scalars["probe_all_zero_unit_fraction"] = float(unit_arrays["all_probe_zero"].mean())
    name = _name(p.activation)
    saturation = (h < 0.05) | (h > 0.95) if name == "sigmoid" else (np.abs(h) > 0.95 if name == "tanh" else None)
    scalars["probe_bounded_saturation_ratio"] = float(saturation.mean()) if saturation is not None else None
    if saturation is not None:
        unit_arrays["bounded_saturation_ratio"] = saturation.mean(axis=1)
    raw_arrays = {key: value.copy() for key, value in {
        "z": z, "h": h, "derivative": derivative, "U": gradients["U"],
        "delta1": gradients["delta1"], "delta2": gradients["delta2"],
        "logits": cache["logits"], "probabilities": cache["probabilities"], "labels": labels,
        "gradient_W1": gradients["W1"], "gradient_b1": gradients["b1"],
        "gradient_W2": gradients["W2"], "gradient_b2": gradients["b2"],
    }.items()}
    if p.alpha is not None:
        scalars.update(_distribution(p.alpha, "probe_alpha"))
        scalars["probe_alpha_negative_fraction"] = float(np.mean(p.alpha < 0))
        scalars["probe_alpha_gt_one_fraction"] = float(np.mean(p.alpha > 1))
        scalars["probe_alpha_zero_fraction"] = float(np.mean(p.alpha == 0))
        scalars["probe_negative_slope_exposure_fraction"] = float(np.mean((p.alpha < 0) & (z < 0)))
        unit_arrays["alpha"] = p.alpha.ravel().copy()
        unit_arrays["gradient_alpha"] = gradients["alpha"].ravel().copy()
        raw_arrays["alpha"] = p.alpha.copy()
        raw_arrays["gradient_alpha"] = gradients["alpha"].copy()
    for key, value in scalars.items():
        if value is not None and not np.isfinite(value):
            raise FloatingPointError(f"Non-finite scalar diagnostic {key}.")
    for key, value in unit_arrays.items():
        _finite(f"unit diagnostic {key}", value)
    return scalars, unit_arrays, raw_arrays


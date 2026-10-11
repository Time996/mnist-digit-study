from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist


# ============================================================
# Configuration
# ============================================================

SEED = 42

EPOCHS = 50
BATCH_SIZE = 64

# Ciresan et al. (2010) learning-rate schedule:
# eta_e = max(MIN_LR, INITIAL_LR * LR_DECAY_FACTOR ** (e - 1)).
# Only the schedule is borrowed here; mini-batching and the other training
# choices still follow this project's manual sigmoid ANN experiment.
INITIAL_LR = 1e-3
LR_DECAY_FACTOR = 0.9
MIN_LR = 1e-6

EARLY_STOPPING_PATIENCE = 10

VALIDATION_SIZE = 6000
PROBE_SIZE = 300

INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10

# Width-only inspiration from Ciresan et al. (2010), whose ID4 MLP used
# hidden widths 2500 -> 2000 -> 1500 -> 1000 -> 500. For this experiment,
# N=6 reproduces that hidden-width sequence; other N values crop or extend
# the same 500-neuron staircase. The rest of this experiment is not a
# reproduction of that paper: input, activation, initialization, batching,
# data augmentation, and optimization remain those of this project.
PYRAMID_WIDTH_STEP = 500
PAPER_ANCHOR_N = 6

SATURATION_LOW = 0.05
SATURATION_HIGH = 0.95

IMPROVEMENT_THRESHOLD = 1e-6

HISTOGRAM_INTERVAL = 5

np.random.seed(SEED)


# ============================================================
# User chooses N
# ============================================================

print()
print("=" * 80)
print("MANUAL N-LAYER ANN")
print("SIGMOID + GD + TRAINING DYNAMICS OBSERVATION")
print("=" * 80)

N = int(
    input(
        "请输入 ANN 的总层数 N "
        "(包括输出层，不包括输入层): "
    )
)

if N < 1:
    raise ValueError("N 必须至少为 1")


# ============================================================
# Architecture
# ============================================================

def make_layer_sizes(
    trainable_layers,
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    width_step=PYRAMID_WIDTH_STEP
):

    if trainable_layers == 1:
        return [
            input_size,
            output_size
        ]

    # Hidden layer l uses h_l = width_step * (N - l), l = 1, ..., N - 1.
    # N=6 therefore gives 2500, 2000, 1500, 1000, 500 exactly.
    hidden_sizes = [
        width_step
        *
        (
            trainable_layers
            -
            layer
        )
        for layer in range(
            1,
            trainable_layers
        )
    ]

    return [
        input_size,
        *hidden_sizes,
        output_size
    ]


layer_sizes = make_layer_sizes(N)

print()
print("=" * 80)
print("NETWORK ARCHITECTURE")
print("=" * 80)

print(
    " -> ".join(
        str(size)
        for size in layer_sizes
    )
)

print(f"Trainable layers: {N}")
print(f"Hidden layers: {N - 1}")

if N > 1:
    print(
        "Width policy: Ciresan-inspired pyramid, "
        f"h_l = {PYRAMID_WIDTH_STEP} * (N - l)"
    )

    if N > PAPER_ANCHOR_N:
        print(
            f"Note: N>{PAPER_ANCHOR_N} extends the width rule "
            "beyond the paper's ID4 architecture."
        )

print("=" * 80)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

EXPERIMENT_NAME = (
    f"manual_n_ann_sigmoid_ciresan_pyramid_expdecay_observation_"
    f"N{N}_{timestamp}"
)

EXPERIMENT_DIR = (
    LOGS_DIR
    /
    EXPERIMENT_NAME
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MISCLASSIFIED_DIR = (
    EXPERIMENT_DIR
    /
    "misclassified"
)

MISCLASSIFIED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ACTIVATION_HIST_DIR = (
    EXPERIMENT_DIR
    /
    "activation_histograms"
)

ACTIVATION_HIST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BEST_WEIGHTS_PATH = (
    EXPERIMENT_DIR
    /
    "best_weights.npz"
)

FINAL_WEIGHTS_PATH = (
    EXPERIMENT_DIR
    /
    "final_weights.npz"
)

TRAINING_LOG_PATH = (
    EXPERIMENT_DIR
    /
    "training_log.csv"
)

RESULT_PATH = (
    EXPERIMENT_DIR
    /
    "result.txt"
)

CONFIG_PATH = (
    EXPERIMENT_DIR
    /
    "experiment_config.txt"
)

LOSS_CURVE_PATH = (
    EXPERIMENT_DIR
    /
    "loss_curve.png"
)

ACCURACY_CURVE_PATH = (
    EXPERIMENT_DIR
    /
    "accuracy_curve.png"
)

LR_CURVE_PATH = (
    EXPERIMENT_DIR
    /
    "learning_rate_curve.png"
)

GRADIENT_NORM_PATH = (
    EXPERIMENT_DIR
    /
    "gradient_norm_curve.png"
)

GRADIENT_RMS_PATH = (
    EXPERIMENT_DIR
    /
    "gradient_rms_curve.png"
)

DELTA_RMS_PATH = (
    EXPERIMENT_DIR
    /
    "backprop_delta_rms_curve.png"
)

GRADIENT_RATIO_PATH = (
    EXPERIMENT_DIR
    /
    "gradient_ratio_curve.png"
)

ACTIVATION_MEAN_STD_PATH = (
    EXPERIMENT_DIR
    /
    "activation_mean_std_curve.png"
)

ACTIVATION_STD_PATH = (
    EXPERIMENT_DIR
    /
    "activation_std_curve.png"
)

SATURATION_PATH = (
    EXPERIMENT_DIR
    /
    "activation_saturation_curve.png"
)

MISCLASSIFIED_TXT_PATH = (
    EXPERIMENT_DIR
    /
    "misclassified.txt"
)

MISCLASSIFIED_GRID_PATH = (
    EXPERIMENT_DIR
    /
    "misclassified_grid.png"
)


# ============================================================
# Load MNIST
# ============================================================

(
    x_train,
    y_train
), (
    x_test,
    y_test
) = mnist.load_data()


x_train = (
    x_train
    .reshape(
        len(x_train),
        INPUT_SIZE
    )
    .astype(np.float32)
    /
    255.0
)

x_test = (
    x_test
    .reshape(
        len(x_test),
        INPUT_SIZE
    )
    .astype(np.float32)
    /
    255.0
)


# ============================================================
# Validation Split
# ============================================================

x_val = x_train[
    -VALIDATION_SIZE:
]

y_val = y_train[
    -VALIDATION_SIZE:
]

x_train = x_train[
    :-VALIDATION_SIZE
]

y_train = y_train[
    :-VALIDATION_SIZE
]


# ============================================================
# Fixed 300-sample Probe Set
# ============================================================

probe_rng = np.random.default_rng(SEED)

probe_indices = probe_rng.choice(
    len(x_val),
    size=PROBE_SIZE,
    replace=False
)

x_probe = x_val[
    probe_indices
]

y_probe = y_val[
    probe_indices
]


print()
print("=" * 80)
print("DATA")
print("=" * 80)

print(
    "Train:",
    x_train.shape
)

print(
    "Validation:",
    x_val.shape
)

print(
    "Test:",
    x_test.shape
)

print(
    "Fixed activation probe:",
    x_probe.shape
)

print("=" * 80)


# ============================================================
# One-hot
# ============================================================

def one_hot(
    labels,
    num_classes=10
):

    output = np.zeros(
        (
            len(labels),
            num_classes
        ),
        dtype=np.float32
    )

    output[
        np.arange(len(labels)),
        labels
    ] = 1.0

    return output


# ============================================================
# Activation Functions
# ============================================================

def sigmoid(z):

    z = np.clip(
        z,
        -50,
        50
    )

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(-z)
        )
    )


def sigmoid_derivative_from_activation(a):

    return (
        a
        *
        (
            1.0
            -
            a
        )
    )


def softmax(z):

    z = (
        z
        -
        np.max(
            z,
            axis=0,
            keepdims=True
        )
    )

    exp_z = np.exp(z)

    return (
        exp_z
        /
        np.sum(
            exp_z,
            axis=0,
            keepdims=True
        )
    )


# ============================================================
# Cross Entropy
# ============================================================

def cross_entropy(
    probabilities,
    y_true
):

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    m = probabilities.shape[1]

    return (
        -np.sum(
            y_true
            *
            np.log(
                probabilities.T
            )
        )
        /
        m
    )


# ============================================================
# Xavier Initialization
# ============================================================

weights = []
biases = []

for layer in range(N):

    fan_in = layer_sizes[
        layer
    ]

    fan_out = layer_sizes[
        layer + 1
    ]

    limit = np.sqrt(
        6.0
        /
        (
            fan_in
            +
            fan_out
        )
    )

    W = np.random.uniform(
        -limit,
        limit,
        size=(
            fan_out,
            fan_in
        )
    ).astype(
        np.float32
    )

    b = np.zeros(
        (
            fan_out,
            1
        ),
        dtype=np.float32
    )

    weights.append(W)
    biases.append(b)


# ============================================================
# Network Information
# ============================================================

parameter_count = 0

print()
print("=" * 80)
print("WEIGHT MATRIX DIMENSIONS")
print("=" * 80)

for layer in range(N):

    print(
        f"W{layer + 1}: "
        f"{weights[layer].shape}"
    )

    print(
        f"b{layer + 1}: "
        f"{biases[layer].shape}"
    )

    parameter_count += (
        weights[layer].size
        +
        biases[layer].size
    )

print("=" * 80)

print(
    "Total parameters:",
    f"{parameter_count:,}"
)


# ============================================================
# Forward Propagation
# ============================================================

def forward(X):

    A = X.T

    activations = [
        A
    ]

    for layer in range(
        N - 1
    ):

        Z = (
            weights[layer]
            @
            A
            +
            biases[layer]
        )

        A = sigmoid(Z)

        activations.append(A)

    Z_output = (
        weights[-1]
        @
        A
        +
        biases[-1]
    )

    A_output = softmax(
        Z_output
    )

    activations.append(
        A_output
    )

    return activations


# ============================================================
# Backpropagation
# ============================================================

def backward(
    activations,
    y_batch
):

    m = y_batch.shape[0]

    dW = [
        None
    ] * N

    db = [
        None
    ] * N

    delta_rms = np.zeros(
        N,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Output
    # delta^N = y_hat - y
    # --------------------------------------------------------

    y_hat = activations[-1]

    delta = (
        y_hat
        -
        y_batch.T
    )

    delta_rms[-1] = np.sqrt(
        np.mean(
            delta.astype(
                np.float64
            ) ** 2
        )
    )

    dW[-1] = (
        delta
        @
        activations[-2].T
        /
        m
    )

    db[-1] = (
        np.sum(
            delta,
            axis=1,
            keepdims=True
        )
        /
        m
    )


    # --------------------------------------------------------
    # Hidden Layers
    # --------------------------------------------------------

    for layer in range(
        N - 2,
        -1,
        -1
    ):

        sigmoid_prime = (
            sigmoid_derivative_from_activation(
                activations[
                    layer + 1
                ]
            )
        )

        delta = (
            weights[
                layer + 1
            ].T
            @
            delta
        ) * sigmoid_prime

        delta_rms[layer] = np.sqrt(
            np.mean(
                delta.astype(
                    np.float64
                ) ** 2
            )
        )

        dW[layer] = (
            delta
            @
            activations[layer].T
            /
            m
        )

        db[layer] = (
            np.sum(
                delta,
                axis=1,
                keepdims=True
            )
            /
            m
        )

    return (
        dW,
        db,
        delta_rms
    )


# ============================================================
# Gradient Descent
# ============================================================

def update_parameters(
    dW,
    db,
    learning_rate
):

    for layer in range(N):

        weights[layer] -= (
            learning_rate
            *
            dW[layer]
        )

        biases[layer] -= (
            learning_rate
            *
            db[layer]
        )


# ============================================================
# Prediction
# ============================================================

def predict(
    X,
    batch_size=1000
):

    predictions = []
    probabilities = []

    for start in range(
        0,
        len(X),
        batch_size
    ):

        end = (
            start
            +
            batch_size
        )

        activations = forward(
            X[start:end]
        )

        probs = (
            activations[-1].T
        )

        pred = np.argmax(
            probs,
            axis=1
        )

        predictions.append(
            pred
        )

        probabilities.append(
            probs
        )

    return (
        np.concatenate(
            predictions
        ),
        np.concatenate(
            probabilities
        )
    )


# ============================================================
# Evaluate Accuracy
# ============================================================

def evaluate_accuracy(
    X,
    y
):

    pred, _ = predict(X)

    return np.mean(
        pred == y
    )


# ============================================================
# Evaluate Loss
# ============================================================

def evaluate_loss(
    X,
    y,
    batch_size=1000
):

    total_loss = 0.0
    total_samples = 0

    for start in range(
        0,
        len(X),
        batch_size
    ):

        end = (
            start
            +
            batch_size
        )

        X_batch = X[
            start:end
        ]

        labels = y[
            start:end
        ]

        y_batch = one_hot(
            labels
        )

        activations = forward(
            X_batch
        )

        batch_loss = cross_entropy(
            activations[-1],
            y_batch
        )

        batch_samples = len(
            X_batch
        )

        total_loss += (
            batch_loss
            *
            batch_samples
        )

        total_samples += (
            batch_samples
        )

    return (
        total_loss
        /
        total_samples
    )


# ============================================================
# Probe Activation Statistics
# ============================================================

def activation_statistics(
    X
):

    activations = forward(X)

    means = []
    stds = []
    saturation_ratios = []


    # Only hidden layers
    for layer in range(
        1,
        N
    ):

        A = activations[
            layer
        ]

        mean_value = np.mean(A)

        std_value = np.std(A)

        saturated = (
            (A < SATURATION_LOW)
            |
            (A > SATURATION_HIGH)
        )

        saturation_ratio = (
            np.mean(
                saturated
            )
        )

        means.append(
            float(
                mean_value
            )
        )

        stds.append(
            float(
                std_value
            )
        )

        saturation_ratios.append(
            float(
                saturation_ratio
            )
        )

    return (
        activations,
        means,
        stds,
        saturation_ratios
    )


# ============================================================
# Activation Histogram
# ============================================================

def save_activation_histogram(
    activations,
    epoch
):

    if N <= 1:
        return

    plt.figure(
        figsize=(
            9,
            6
        )
    )

    for layer in range(
        1,
        N
    ):

        values = (
            activations[layer]
            .reshape(-1)
        )

        plt.hist(
            values,
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.3,
            label=f"Hidden {layer}"
        )

    plt.xlabel(
        "Sigmoid Activation"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        f"Activation Distribution - Epoch {epoch}"
    )

    plt.xlim(
        0,
        1
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ACTIVATION_HIST_DIR
        /
        f"activation_hist_epoch_{epoch:03d}.png",
        dpi=220
    )

    plt.close()


# ============================================================
# Save / Load Weights
# ============================================================

def save_weights(path):

    save_dict = {
        "N":
            np.array(
                [N]
            ),

        "layer_sizes":
            np.array(
                layer_sizes
            )
    }

    for layer in range(N):

        save_dict[
            f"W_{layer + 1}"
        ] = weights[layer]

        save_dict[
            f"b_{layer + 1}"
        ] = biases[layer]

    np.savez(
        path,
        **save_dict
    )


def load_weights(path):

    data = np.load(path)

    for layer in range(N):

        weights[layer][...] = (
            data[
                f"W_{layer + 1}"
            ]
        )

        biases[layer][...] = (
            data[
                f"b_{layer + 1}"
            ]
        )


# ============================================================
# History
# ============================================================

train_loss_history = []
val_loss_history = []

train_acc_history = []
val_acc_history = []

learning_rate_history = []

gradient_norm_history = [
    []
    for _ in range(N)
]

gradient_rms_history = [
    []
    for _ in range(N)
]

delta_rms_history = [
    []
    for _ in range(N)
]

activation_mean_history = [
    []
    for _ in range(
        max(
            N - 1,
            0
        )
    )
]

activation_std_history = [
    []
    for _ in range(
        max(
            N - 1,
            0
        )
    )
]

saturation_history = [
    []
    for _ in range(
        max(
            N - 1,
            0
        )
    )
]

gradient_ratio_history = []


# ============================================================
# Scheduler State
# ============================================================

learning_rate = (
    INITIAL_LR
)

best_val_loss = float(
    "inf"
)

best_epoch = 0

early_stop_wait = 0


# ============================================================
# Training
# ============================================================

print()
print("=" * 80)
print("START TRAINING")
print("=" * 80)


for epoch in range(
    1,
    EPOCHS + 1
):

    current_epoch_lr = (
        learning_rate
    )

    indices = np.random.permutation(
        len(x_train)
    )

    x_train_shuffled = (
        x_train[
            indices
        ]
    )

    y_train_shuffled = (
        y_train[
            indices
        ]
    )


    epoch_loss_sum = 0.0
    epoch_samples = 0

    epoch_correct = 0

    batch_count = 0

    epoch_gradient_norms = np.zeros(
        N,
        dtype=np.float64
    )

    epoch_gradient_rms = np.zeros(
        N,
        dtype=np.float64
    )

    epoch_delta_rms = np.zeros(
        N,
        dtype=np.float64
    )


    # ========================================================
    # Mini-batch Training
    # ========================================================

    for start in range(
        0,
        len(x_train_shuffled),
        BATCH_SIZE
    ):

        end = (
            start
            +
            BATCH_SIZE
        )

        X_batch = (
            x_train_shuffled[
                start:end
            ]
        )

        y_labels = (
            y_train_shuffled[
                start:end
            ]
        )

        y_batch = one_hot(
            y_labels
        )

        batch_size_actual = len(
            X_batch
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        activations = forward(
            X_batch
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        batch_loss = cross_entropy(
            activations[-1],
            y_batch
        )

        epoch_loss_sum += (
            batch_loss
            *
            batch_size_actual
        )


        # ----------------------------------------------------
        # Online Train Accuracy
        # ----------------------------------------------------

        batch_pred = np.argmax(
            activations[-1],
            axis=0
        )

        epoch_correct += np.sum(
            batch_pred
            ==
            y_labels
        )

        epoch_samples += (
            batch_size_actual
        )


        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------

        (
            dW,
            db,
            batch_delta_rms
        ) = backward(
            activations,
            y_batch
        )


        # ----------------------------------------------------
        # Gradient Statistics
        # ----------------------------------------------------

        for layer in range(N):

            grad_norm = np.linalg.norm(
                dW[layer]
            )

            grad_rms = (
                grad_norm
                /
                np.sqrt(
                    dW[layer].size
                )
            )

            epoch_gradient_norms[
                layer
            ] += grad_norm

            epoch_gradient_rms[
                layer
            ] += grad_rms

            epoch_delta_rms[
                layer
            ] += batch_delta_rms[
                layer
            ]


        # ----------------------------------------------------
        # GD Update
        # ----------------------------------------------------

        update_parameters(
            dW,
            db,
            learning_rate
        )

        batch_count += 1


    # ========================================================
    # Epoch Statistics
    # ========================================================

    train_loss = (
        epoch_loss_sum
        /
        epoch_samples
    )

    train_accuracy = (
        epoch_correct
        /
        epoch_samples
    )

    epoch_gradient_norms /= (
        batch_count
    )

    epoch_gradient_rms /= (
        batch_count
    )

    epoch_delta_rms /= (
        batch_count
    )


    # ========================================================
    # Validation
    # ========================================================

    val_loss = evaluate_loss(
        x_val,
        y_val
    )

    val_accuracy = evaluate_accuracy(
        x_val,
        y_val
    )


    # ========================================================
    # Fixed Probe Activation Statistics
    # ========================================================

    (
        probe_activations,
        activation_means,
        activation_stds,
        saturation_ratios
    ) = activation_statistics(
        x_probe
    )


    # ========================================================
    # Gradient Ratio
    #
    # RMS gradient is used because W1 and WN
    # have different numbers of parameters.
    # ========================================================

    gradient_ratio = (
        epoch_gradient_rms[0]
        /
        (
            epoch_gradient_rms[-1]
            +
            1e-30
        )
    )


    # ========================================================
    # History
    # ========================================================

    train_loss_history.append(
        train_loss
    )

    val_loss_history.append(
        val_loss
    )

    train_acc_history.append(
        train_accuracy
    )

    val_acc_history.append(
        val_accuracy
    )

    learning_rate_history.append(
        current_epoch_lr
    )

    gradient_ratio_history.append(
        gradient_ratio
    )


    for layer in range(N):

        gradient_norm_history[
            layer
        ].append(
            epoch_gradient_norms[
                layer
            ]
        )

        gradient_rms_history[
            layer
        ].append(
            epoch_gradient_rms[
                layer
            ]
        )

        delta_rms_history[
            layer
        ].append(
            epoch_delta_rms[
                layer
            ]
        )


    for layer in range(
        N - 1
    ):

        activation_mean_history[
            layer
        ].append(
            activation_means[
                layer
            ]
        )

        activation_std_history[
            layer
        ].append(
            activation_stds[
                layer
            ]
        )

        saturation_history[
            layer
        ].append(
            saturation_ratios[
                layer
            ]
        )


    # ========================================================
    # Activation Histogram Snapshots
    # ========================================================

    if (
        N > 1
        and
        (
            epoch == 1
            or
            epoch % HISTOGRAM_INTERVAL == 0
        )
    ):

        save_activation_histogram(
            probe_activations,
            epoch
        )


    # ========================================================
    # Console Output
    # ========================================================

    print()
    print("-" * 80)

    print(
        f"Epoch "
        f"{epoch:02d}/{EPOCHS}"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.6f}"
        f" | "
        f"Val Loss: "
        f"{val_loss:.6f}"
    )

    print(
        f"Online Train Accuracy: "
        f"{train_accuracy * 100:.4f}%"
        f" | "
        f"Val Accuracy: "
        f"{val_accuracy * 100:.4f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_epoch_lr:.8f}"
    )


    print()
    print("Weight Gradient Frobenius Norm")

    print(
        " | ".join(
            [
                f"G{layer + 1}: "
                f"{epoch_gradient_norms[layer]:.3e}"
                for layer in range(N)
            ]
        )
    )


    print()
    print("Weight Gradient RMS")

    print(
        " | ".join(
            [
                f"G{layer + 1}: "
                f"{epoch_gradient_rms[layer]:.3e}"
                for layer in range(N)
            ]
        )
    )


    print()
    print("Backprop Delta RMS")

    print(
        " | ".join(
            [
                f"D{layer + 1}: "
                f"{epoch_delta_rms[layer]:.3e}"
                for layer in range(N)
            ]
        )
    )


    print()

    print(
        f"Gradient Ratio "
        f"G1_RMS / G{N}_RMS: "
        f"{gradient_ratio:.3e}"
    )


    if N > 1:

        print()
        print("Probe Activation Statistics")

        for layer in range(
            N - 1
        ):

            print(
                f"H{layer + 1}: "
                f"mean="
                f"{activation_means[layer]:.4f}"
                f" | std="
                f"{activation_stds[layer]:.4f}"
                f" | saturation="
                f"{saturation_ratios[layer] * 100:.2f}%"
            )


    # ========================================================
    # Best Model
    # ========================================================

    if (
        val_loss
        <
        best_val_loss
        -
        IMPROVEMENT_THRESHOLD
    ):

        best_val_loss = (
            val_loss
        )

        best_epoch = (
            epoch
        )

        early_stop_wait = 0

        save_weights(
            BEST_WEIGHTS_PATH
        )

        print()

        print(
            f"Best validation loss improved "
            f"to {best_val_loss:.6f}"
        )

        print(
            "Best weights saved."
        )

    else:

        early_stop_wait += 1


    # ========================================================
    # Paper-Style Exponential Learning-Rate Decay
    #
    # The current epoch used current_epoch_lr. This update prepares
    # eta_(e+1) = max(MIN_LR, eta_e * LR_DECAY_FACTOR).
    # ========================================================

    learning_rate = max(
        MIN_LR,
        current_epoch_lr
        *
        LR_DECAY_FACTOR
    )


    # ========================================================
    # Early Stopping
    # ========================================================

    if (
        early_stop_wait
        >=
        EARLY_STOPPING_PATIENCE
    ):

        print()

        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# Save Final Weights
# ============================================================

save_weights(
    FINAL_WEIGHTS_PATH
)


# ============================================================
# Restore Best Weights
# ============================================================

print()
print("=" * 80)
print("RESTORING BEST MODEL")
print("=" * 80)

load_weights(
    BEST_WEIGHTS_PATH
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)


# ============================================================
# Test
# ============================================================

test_predictions, test_probabilities = predict(
    x_test
)

test_accuracy = np.mean(
    test_predictions
    ==
    y_test
)

test_loss = evaluate_loss(
    x_test,
    y_test
)

wrong_indices = np.where(
    test_predictions
    !=
    y_test
)[0]

wrong_count = len(
    wrong_indices
)

correct_count = (
    len(y_test)
    -
    wrong_count
)

error_rate = (
    wrong_count
    /
    len(y_test)
)


print()
print("=" * 80)
print("FINAL TEST RESULT")
print("=" * 80)

print(
    "Architecture:",
    " -> ".join(
        str(size)
        for size in layer_sizes
    )
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Test loss: "
    f"{test_loss:.6f}"
)

print(
    f"Accuracy: "
    f"{test_accuracy * 100:.4f}%"
)

print(
    f"Correct: "
    f"{correct_count}"
)

print(
    f"Wrong: "
    f"{wrong_count}"
)

print(
    f"Error rate: "
    f"{error_rate * 100:.4f}%"
)

print("=" * 80)


# ============================================================
# Training Log
# ============================================================

epochs_axis = np.arange(
    1,
    len(
        train_loss_history
    )
    +
    1
)

log_data = {

    "epoch":
        epochs_axis,

    "train_loss":
        train_loss_history,

    "val_loss":
        val_loss_history,

    "online_train_accuracy":
        train_acc_history,

    "val_accuracy":
        val_acc_history,

    "learning_rate":
        learning_rate_history,

    "gradient_ratio_G1_GN_rms":
        gradient_ratio_history
}


for layer in range(N):

    log_data[
        f"gradient_fro_W{layer + 1}"
    ] = gradient_norm_history[
        layer
    ]

    log_data[
        f"gradient_rms_W{layer + 1}"
    ] = gradient_rms_history[
        layer
    ]

    log_data[
        f"delta_rms_L{layer + 1}"
    ] = delta_rms_history[
        layer
    ]


for layer in range(
    N - 1
):

    log_data[
        f"activation_mean_H{layer + 1}"
    ] = activation_mean_history[
        layer
    ]

    log_data[
        f"activation_std_H{layer + 1}"
    ] = activation_std_history[
        layer
    ]

    log_data[
        f"saturation_ratio_H{layer + 1}"
    ] = saturation_history[
        layer
    ]


training_log_df = pd.DataFrame(
    log_data
)

training_log_df.to_csv(
    TRAINING_LOG_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Loss Curve
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_axis,
    train_loss_history,
    label="Train Loss"
)

plt.plot(
    epochs_axis,
    val_loss_history,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    f"Loss - N={N}"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    LOSS_CURVE_PATH,
    dpi=250
)

plt.close()


# ============================================================
# Accuracy Curve
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_axis,
    train_acc_history,
    label="Online Train Accuracy"
)

plt.plot(
    epochs_axis,
    val_acc_history,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    f"Accuracy - N={N}"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    ACCURACY_CURVE_PATH,
    dpi=250
)

plt.close()


# ============================================================
# Learning Rate
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_axis,
    learning_rate_history
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")

plt.title(
    f"Learning Rate - N={N}"
)

plt.yscale("log")

plt.tight_layout()

plt.savefig(
    LR_CURVE_PATH,
    dpi=250
)

plt.close()


# ============================================================
# Gradient Frobenius Norm
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for layer in range(N):

    plt.plot(
        epochs_axis,
        gradient_norm_history[
            layer
        ],
        label=f"W{layer + 1}"
    )

plt.xlabel("Epoch")
plt.ylabel("Gradient Frobenius Norm")

plt.title(
    f"Weight Gradient Norm - N={N}"
)

plt.yscale("log")

plt.legend()
plt.tight_layout()

plt.savefig(
    GRADIENT_NORM_PATH,
    dpi=250
)

plt.close()


# ============================================================
# Gradient RMS
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for layer in range(N):

    plt.plot(
        epochs_axis,
        gradient_rms_history[
            layer
        ],
        label=f"W{layer + 1}"
    )

plt.xlabel("Epoch")
plt.ylabel("Gradient RMS")

plt.title(
    f"Weight Gradient RMS - N={N}"
)

plt.yscale("log")

plt.legend()
plt.tight_layout()

plt.savefig(
    GRADIENT_RMS_PATH,
    dpi=250
)

plt.close()


# ============================================================
# Backprop Delta RMS
# ============================================================

plt.figure(
    figsize=(9, 6)
)

for layer in range(N):

    plt.plot(
        epochs_axis,
        delta_rms_history[
            layer
        ],
        label=f"Layer {layer + 1}"
    )

plt.xlabel("Epoch")
plt.ylabel("Backprop Delta RMS")

plt.title(
    f"Backpropagated Gradient Signal - N={N}"
)

plt.yscale("log")

plt.legend()
plt.tight_layout()

plt.savefig(
    DELTA_RMS_PATH,
    dpi=250
)

plt.close()


# ============================================================
# Gradient Ratio
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_axis,
    gradient_ratio_history
)

plt.xlabel("Epoch")

plt.ylabel(
    f"G1 RMS / G{N} RMS"
)

plt.title(
    f"Gradient Propagation Ratio - N={N}"
)

plt.yscale("log")

plt.tight_layout()

plt.savefig(
    GRADIENT_RATIO_PATH,
    dpi=250
)

plt.close()


# ============================================================
# Activation Mean ± Std
# ============================================================

if N > 1:

    plt.figure(
        figsize=(10, 6)
    )

    for layer in range(
        N - 1
    ):

        means = np.array(
            activation_mean_history[
                layer
            ]
        )

        stds = np.array(
            activation_std_history[
                layer
            ]
        )

        plt.plot(
            epochs_axis,
            means,
            label=f"Hidden {layer + 1}"
        )

        plt.fill_between(
            epochs_axis,
            np.clip(
                means - stds,
                0,
                1
            ),
            np.clip(
                means + stds,
                0,
                1
            ),
            alpha=0.12
        )

    plt.xlabel("Epoch")

    plt.ylabel(
        "Sigmoid Activation"
    )

    plt.title(
        f"Activation Mean ± Std - N={N}"
    )

    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ACTIVATION_MEAN_STD_PATH,
        dpi=250
    )

    plt.close()


# ============================================================
# Activation Standard Deviation
# ============================================================

if N > 1:

    plt.figure(
        figsize=(9, 6)
    )

    for layer in range(
        N - 1
    ):

        plt.plot(
            epochs_axis,
            activation_std_history[
                layer
            ],
            label=f"Hidden {layer + 1}"
        )

    plt.xlabel("Epoch")

    plt.ylabel(
        "Activation Standard Deviation"
    )

    plt.title(
        f"Activation Std - N={N}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ACTIVATION_STD_PATH,
        dpi=250
    )

    plt.close()


# ============================================================
# Saturation Ratio
# ============================================================

if N > 1:

    plt.figure(
        figsize=(9, 6)
    )

    for layer in range(
        N - 1
    ):

        plt.plot(
            epochs_axis,
            np.array(
                saturation_history[
                    layer
                ]
            )
            *
            100,
            label=f"Hidden {layer + 1}"
        )

    plt.xlabel("Epoch")

    plt.ylabel(
        "Saturated Activations (%)"
    )

    plt.title(
        f"Sigmoid Saturation Ratio - N={N}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        SATURATION_PATH,
        dpi=250
    )

    plt.close()


# ============================================================
# Misclassified Text
# ============================================================

with open(
    MISCLASSIFIED_TXT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MISCLASSIFIED MNIST TEST IMAGES\n"
    )

    f.write(
        "=" * 80
        +
        "\n\n"
    )

    f.write(
        f"Accuracy: "
        f"{test_accuracy * 100:.4f}%\n"
    )

    f.write(
        f"Wrong: "
        f"{wrong_count}\n\n"
    )

    for idx in wrong_indices:

        true_label = int(
            y_test[idx]
        )

        pred_label = int(
            test_predictions[idx]
        )

        confidence = float(
            test_probabilities[
                idx,
                pred_label
            ]
        )

        f.write(
            f"Index: {idx}, "
            f"True: {true_label}, "
            f"Predicted: {pred_label}, "
            f"Confidence: "
            f"{confidence * 100:.2f}%\n"
        )


# ============================================================
# Misclassified Images
# ============================================================

for idx in wrong_indices:

    true_label = int(
        y_test[idx]
    )

    pred_label = int(
        test_predictions[idx]
    )

    confidence = float(
        test_probabilities[
            idx,
            pred_label
        ]
    )

    filename = (
        f"index_{idx}"
        f"_true_{true_label}"
        f"_pred_{pred_label}"
        f"_conf_{confidence * 100:.2f}.png"
    )

    plt.figure(
        figsize=(3, 3)
    )

    plt.imshow(
        x_test[
            idx
        ].reshape(
            28,
            28
        ),
        cmap="gray"
    )

    plt.title(
        f"Index {idx}\n"
        f"True {true_label} | "
        f"Pred {pred_label}\n"
        f"Conf {confidence * 100:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        MISCLASSIFIED_DIR
        /
        filename,
        dpi=160
    )

    plt.close()


# ============================================================
# Misclassified Grid
# ============================================================

if wrong_count > 0:

    show_count = min(
        wrong_count,
        60
    )

    cols = 6

    rows = int(
        np.ceil(
            show_count
            /
            cols
        )
    )

    plt.figure(
        figsize=(
            12,
            rows * 2
        )
    )

    for position, idx in enumerate(
        wrong_indices[
            :show_count
        ],
        start=1
    ):

        pred_label = (
            test_predictions[idx]
        )

        true_label = (
            y_test[idx]
        )

        confidence = (
            test_probabilities[
                idx,
                pred_label
            ]
        )

        plt.subplot(
            rows,
            cols,
            position
        )

        plt.imshow(
            x_test[
                idx
            ].reshape(
                28,
                28
            ),
            cmap="gray"
        )

        plt.title(
            f"{idx}\n"
            f"T:{true_label} "
            f"P:{pred_label}\n"
            f"{confidence * 100:.1f}%",
            fontsize=8
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        MISCLASSIFIED_GRID_PATH,
        dpi=200
    )

    plt.close()


# ============================================================
# Configuration
# ============================================================

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MANUAL N-LAYER ANN\n"
    )

    f.write(
        "SIGMOID + TRAINING DYNAMICS OBSERVATION\n"
    )

    f.write(
        "=" * 80
        +
        "\n\n"
    )

    f.write(
        f"N: {N}\n"
    )

    f.write(
        f"Seed: {SEED}\n"
    )

    f.write(
        f"Maximum epochs: {EPOCHS}\n"
    )

    f.write(
        f"Architecture: "
        +
        " -> ".join(
            str(size)
            for size in layer_sizes
        )
        +
        "\n"
    )

    f.write(
        "Width policy: Ciresan-inspired pyramid; "
        f"h_l = {PYRAMID_WIDTH_STEP} * (N - l), "
        "l = 1, ..., N - 1\n"
    )

    f.write(
        f"Paper anchor: N={PAPER_ANCHOR_N} gives hidden widths "
        "2500 -> 2000 -> 1500 -> 1000 -> 500\n"
    )

    f.write(
        "Width reference: Ciresan et al. (2010), "
        "Deep Big Simple Neural Nets Excel on Handwritten Digit "
        "Recognition, arXiv:1003.0358, network ID4\n"
    )

    f.write(
        "Paper reproduction: false; hidden-width pattern and learning-rate "
        "schedule are borrowed\n"
    )

    f.write(
        f"Parameters: "
        f"{parameter_count:,}\n"
    )

    f.write(
        f"Train samples: "
        f"{len(x_train)}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(x_val)}\n"
    )

    f.write(
        f"Probe samples: "
        f"{PROBE_SIZE}\n"
    )

    f.write(
        f"Batch size: "
        f"{BATCH_SIZE}\n"
    )

    f.write(
        f"Initial LR: "
        f"{INITIAL_LR}\n"
    )

    f.write(
        "LR schedule: exponential decay every epoch\n"
    )

    f.write(
        f"LR decay factor: "
        f"{LR_DECAY_FACTOR}\n"
    )

    f.write(
        f"Minimum LR: "
        f"{MIN_LR}\n"
    )

    f.write(
        f"Early stopping patience: "
        f"{EARLY_STOPPING_PATIENCE}\n"
    )

    f.write(
        f"Saturation threshold: "
        f"<{SATURATION_LOW} or "
        f">{SATURATION_HIGH}\n"
    )


# ============================================================
# Result
# ============================================================

result_lines = [

    "=" * 80,

    "MANUAL N-LAYER ANN - TRAINING DYNAMICS",

    "=" * 80,

    "",

    "ARCHITECTURE",

    "-" * 80,

    " -> ".join(
        str(size)
        for size in layer_sizes
    ),

    f"N: {N}",

    f"Hidden layers: {N - 1}",

    "Width policy: Ciresan-inspired pyramid; "
    f"h_l = {PYRAMID_WIDTH_STEP} * (N - l)",

    "Paper reproduction: false; hidden-width pattern and "
    "learning-rate schedule borrowed",

    f"Parameters: {parameter_count:,}",

    "",

    "TRAINING",

    "-" * 80,

    f"Epochs completed: "
    f"{len(train_loss_history)}",

    f"Best epoch: "
    f"{best_epoch}",

    f"Best validation loss: "
    f"{best_val_loss:.6f}",

    f"Initial LR: "
    f"{INITIAL_LR}",

    f"LR decay factor: "
    f"{LR_DECAY_FACTOR}",

    f"Last used LR: "
    f"{learning_rate_history[-1]:.8f}",

    "",

    "TEST",

    "-" * 80,

    f"Test loss: "
    f"{test_loss:.6f}",

    f"Accuracy: "
    f"{test_accuracy * 100:.4f}%",

    f"Correct: "
    f"{correct_count}",

    f"Wrong: "
    f"{wrong_count}",

    f"Error rate: "
    f"{error_rate * 100:.4f}%",

    "",

    "=" * 80
]

result_text = "\n".join(
    result_lines
)

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        result_text
    )


print()
print(result_text)

print()
print("Saved to:")

print(
    EXPERIMENT_DIR
)


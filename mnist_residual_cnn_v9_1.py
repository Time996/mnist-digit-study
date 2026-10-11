from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

# ============================================================
# Configuration
# ============================================================

SEED = 42

EPOCHS = 100

BATCH_SIZE = 64

# The validation set is a symbolic monitor only. One error changes its
# accuracy by 1%, so it must not select the official test checkpoint.
VALIDATION_SIZE = 100

INITIAL_LR = 1e-3

AUGMENTATION_LEVELS = (
    # (translation, rotation, scale variation,
    #  elastic probability, elastic scale min, elastic scale max)
    (0.00, 0.0, 0.00, 0.00, 0.00, 0.00),
    (0.05, 5.0, 0.03, 0.10, 0.15, 0.25),
    (0.10, 10.0, 0.06, 0.20, 0.20, 0.30),
    (0.15, 15.0, 0.08, 0.30, 0.25, 0.40),
    (0.20, 15.0, 0.10, 0.40, 0.30, 0.50)
)

AUGMENTATION_STAGE_EPOCHS = 8

LR_DECAY_PER_EPOCH = 0.98

EMA_MOMENTUM = 0.999

STABLE_TRAIN_ACCURACY = 0.985

STABLE_EPOCHS_REQUIRED = 3

MAX_CLEAN_EPOCHS = 15

# ============================================================
# Reproducibility
# ============================================================

tf.keras.utils.set_random_seed(
    SEED
)

try:

    tf.config.experimental.enable_op_determinism()

except Exception:

    pass

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

MODEL_DIR = (
        BASE_DIR /
        "models"
)

LOGS_DIR = (
        BASE_DIR /
        "logs"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

EXPERIMENT_NAME = (
    f"residual_cnn_v9_1_small_validation_elastic_curriculum_"
    f"{timestamp}"
)

EXPERIMENT_DIR = (
        LOGS_DIR /
        EXPERIMENT_NAME
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MISCLASSIFIED_DIR = (
        EXPERIMENT_DIR /
        "misclassified"
)

MISCLASSIFIED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
        EXPERIMENT_DIR /
        "best_val_loss_model.keras"
)

BEST_ACCURACY_MODEL_PATH = (
        EXPERIMENT_DIR /
        "best_val_accuracy_model.keras"
)

FINAL_MODEL_PATH = (
        MODEL_DIR /
        f"{EXPERIMENT_NAME}.keras"
)

TRAINING_LOG_PATH = (
        EXPERIMENT_DIR /
        "training_log.csv"
)

RESULT_PATH = (
        EXPERIMENT_DIR /
        "result.txt"
)

SUMMARY_PATH = (
        EXPERIMENT_DIR /
        "model_summary.txt"
)

CONFIG_PATH = (
        EXPERIMENT_DIR /
        "experiment_config.txt"
)

LOSS_CURVE_PATH = (
        EXPERIMENT_DIR /
        "loss_curve.png"
)

ACCURACY_CURVE_PATH = (
        EXPERIMENT_DIR /
        "accuracy_curve.png"
)

CONFUSION_CSV_PATH = (
        EXPERIMENT_DIR /
        "confusion_matrix.csv"
)

CONFUSION_PNG_PATH = (
        EXPERIMENT_DIR /
        "confusion_matrix.png"
)

MISCLASSIFIED_TXT_PATH = (
        EXPERIMENT_DIR /
        "misclassified.txt"
)

MISCLASSIFIED_GRID_PATH = (
        EXPERIMENT_DIR /
        "misclassified_grid.png"
)

PER_CLASS_RECALL_PATH = (
        EXPERIMENT_DIR /
        "per_class_recall.csv"
)

CONFUSION_PAIRS_PATH = (
        EXPERIMENT_DIR /
        "confusion_pairs.csv"
)

HIGH_CONFIDENCE_ERRORS_PATH = (
        EXPERIMENT_DIR /
        "high_confidence_errors.csv"
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
) = tf.keras.datasets.mnist.load_data()

x_train = (
        x_train.astype(
            "float32"
        )
        /
        255.0
)

x_test = (
        x_test.astype(
            "float32"
        )
        /
        255.0
)

x_train = (
    x_train[..., None]
)

x_test = (
    x_test[..., None]
)

# Explicitly retain the final 100 training examples for validation, matching
# the end-of-array convention formerly used by Keras validation_split.
x_validation = x_train[-VALIDATION_SIZE:]
y_validation = y_train[-VALIDATION_SIZE:]
x_train = x_train[:-VALIDATION_SIZE]
y_train = y_train[:-VALIDATION_SIZE]

print()
print("=" * 80)

print(
    "Training set:",
    x_train.shape
)

print(
    "Validation set:",
    x_validation.shape
)

print(
    "Test set:",
    x_test.shape
)

print("=" * 80)


# ============================================================
# Data Augmentation
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="MNISTResearch"
)
class GradualElasticCurriculumAugmentation(layers.Layer):

    def __init__(
            self,
            augmentation_levels,
            **kwargs
    ):

        super().__init__(**kwargs)

        self.augmentation_levels = tuple(
            (
                float(translation),
                float(rotation_degrees),
                float(scale_variation),
                float(elastic_probability),
                float(elastic_scale_min),
                float(elastic_scale_max)
            )
            for (
                translation,
                rotation_degrees,
                scale_variation,
                elastic_probability,
                elastic_scale_min,
                elastic_scale_max
            )
            in augmentation_levels
        )

        if self.augmentation_levels[0] != (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ):
            raise ValueError(
                "The first augmentation level must be the clean stage."
            )

        self.augmentation_stages = []

        for stage, (
            translation,
            rotation_degrees,
            scale_variation,
            elastic_probability,
            elastic_scale_min,
            elastic_scale_max
        ) in enumerate(
            self.augmentation_levels[1:],
            start=1
        ):
            self.augmentation_stages.append(
                tf.keras.Sequential(
                    [
                        layers.RandomTranslation(
                            height_factor=translation,
                            width_factor=translation,
                            fill_mode="constant",
                            fill_value=0.0
                        ),
                        layers.RandomRotation(
                            factor=rotation_degrees / 360.0,
                            fill_mode="constant",
                            fill_value=0.0
                        ),
                        layers.RandomZoom(
                            height_factor=(
                                -scale_variation,
                                scale_variation
                            ),
                            width_factor=None,
                            fill_mode="constant",
                            fill_value=0.0
                        ),
                        layers.RandomElasticTransform(
                            factor=(
                                elastic_probability,
                                elastic_probability
                            ),
                            scale=(
                                elastic_scale_min,
                                elastic_scale_max
                            ),
                            interpolation="bilinear",
                            fill_mode="constant",
                            fill_value=0.0,
                            value_range=(0.0, 1.0)
                        )
                    ],
                    name=f"random_affine_scale_elastic_stage_{stage}"
                )
            )

    def build(self, input_shape):

        self.stage = self.add_weight(
            name="stage",
            shape=(),
            dtype=tf.int32,
            initializer="zeros",
            trainable=False
        )

        super().build(input_shape)

    def call(self, inputs, training=None):

        if training is False or training is None:
            return inputs

        branch_fns = [
            lambda: tf.identity(inputs)
        ]

        branch_fns.extend(
            [
                lambda augmentation=augmentation: augmentation(
                    inputs,
                    training=True
                )
                for augmentation in self.augmentation_stages
            ]
        )

        return tf.switch_case(
            branch_index=tf.cast(self.stage, tf.int32),
            branch_fns=branch_fns
        )

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "augmentation_levels": [
                    list(level)
                    for level in self.augmentation_levels
                ]
            }
        )

        return config


data_augmentation = GradualElasticCurriculumAugmentation(
    augmentation_levels=AUGMENTATION_LEVELS,
    name="gradual_elastic_curriculum_augmentation"
)


# ============================================================
# Residual Block
# ============================================================

def residual_block(
        x,
        filters
):
    shortcut = x

    x = layers.Conv2D(
        filters=
        filters,

        kernel_size=
        3,

        padding=
        "same",

        use_bias=
        False
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    x = layers.ReLU()(
        x
    )

    x = layers.Conv2D(
        filters=
        filters,

        kernel_size=
        3,

        padding=
        "same",

        use_bias=
        False
    )(x)

    x = layers.BatchNormalization()(
        x
    )

    if (
            shortcut.shape[-1]
            !=
            filters
    ):
        shortcut = layers.Conv2D(
            filters=
            filters,

            kernel_size=
            1,

            padding=
            "same",

            use_bias=
            False
        )(
            shortcut
        )

        shortcut = layers.BatchNormalization()(
            shortcut
        )

    x = layers.Add()(
        [
            x,
            shortcut
        ]
    )

    x = layers.ReLU()(
        x
    )

    return x


# ============================================================
# Build V9.1 from V8.5 with longer curriculum stages and 100-example monitor
# ============================================================

inputs = layers.Input(
    shape=(
        28,
        28,
        1
    ),
    name=
    "input"
)

# Training-only augmentation

x = data_augmentation(
    inputs
)

# Stage 1

x = residual_block(
    x,
    32
)

x = layers.MaxPooling2D(
    pool_size=
    2
)(
    x
)

x = layers.Dropout(
    0.20
)(
    x
)

# Stage 2

x = residual_block(
    x,
    64
)

x = layers.MaxPooling2D(
    pool_size=
    2
)(
    x
)

x = layers.Dropout(
    0.25
)(
    x
)

# Stage 3

x = residual_block(
    x,
    96
)

x = layers.Dropout(
    0.25
)(
    x
)

# Dual global pooling

gap = layers.GlobalAveragePooling2D(
    name=
    "global_average_pooling"
)(
    x
)

gmp = layers.GlobalMaxPooling2D(
    name=
    "global_max_pooling"
)(
    x
)

x = layers.Concatenate(
    name=
    "global_pool_concat"
)(
    [
        gap,
        gmp
    ]
)

x = layers.Dense(
    96,
    activation=
    "relu",
    name=
    "dense_96"
)(
    x
)

x = layers.Dropout(
    0.20
)(
    x
)

outputs = layers.Dense(
    10,
    activation=
    "softmax",
    name=
    "output"
)(
    x
)

model = models.Model(
    inputs=
    inputs,

    outputs=
    outputs,

    name=
        "Residual_CNN_V9_1_Long_Curriculum_Final_EMA"
)

# ============================================================
# Compile
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=
    INITIAL_LR,

    use_ema=
    True,

    ema_momentum=
    EMA_MOMENTUM
)

model.compile(
    optimizer=
    optimizer,

    loss=
    "sparse_categorical_crossentropy",

    metrics=
    [
        "accuracy"
    ]
)

# ============================================================
# Model Summary
# ============================================================

model.summary()

with open(
        SUMMARY_PATH,
        "w",
        encoding=
        "utf-8"
) as f:
    model.summary(
        print_fn=
        lambda line:
        f.write(
            line
            +
            "\n"
        )
    )

# ============================================================
# Save Experiment Config
# ============================================================

with open(
        CONFIG_PATH,
        "w",
        encoding=
        "utf-8"
) as f:
    f.write(
        "MNIST Residual CNN V9.1 - Long Curriculum + 100 Monitor + EMA\n"
    )

    f.write(
        "=" * 80
        +
        "\n\n"
    )

    f.write(
        f"Seed: "
        f"{SEED}\n"
    )

    f.write(
        f"Epochs: "
        f"{EPOCHS}\n"
    )

    f.write(
        f"Batch size: "
        f"{BATCH_SIZE}\n"
    )

    f.write(
        f"Training samples: "
        f"{len(x_train)}\n"
    )

    f.write(
        f"Validation samples: "
        f"{len(x_validation)}\n"
    )

    f.write(
        "Official checkpoint policy: "
        "fixed epoch-100 EMA model\n"
    )

    f.write(
        f"Initial learning rate: "
        f"{INITIAL_LR}\n"
    )

    f.write(
        f"Augmentation levels: "
        f"{AUGMENTATION_LEVELS}\n"
    )

    f.write(
        f"Epochs per intermediate augmentation stage: "
        f"{AUGMENTATION_STAGE_EPOCHS}\n"
    )

    f.write(
        f"LR decay per epoch: "
        f"{LR_DECAY_PER_EPOCH}\n"
    )

    f.write(
        f"EMA momentum: "
        f"{EMA_MOMENTUM}\n"
    )

    f.write(
        f"Stable online training accuracy: "
        f"{STABLE_TRAIN_ACCURACY}\n"
    )

    f.write(
        f"Stable epochs required: "
        f"{STABLE_EPOCHS_REQUIRED}\n"
    )

    f.write(
        f"Maximum clean epochs: "
        f"{MAX_CLEAN_EPOCHS}\n"
    )

    f.write(
        "Early stopping: disabled (fixed observation horizon)\n"
    )

    f.write(
        "\nArchitecture:\n"
    )

    f.write(
        "Residual32 -> Pool -> Dropout0.20\n"
    )

    f.write(
        "Residual64 -> Pool -> Dropout0.25\n"
    )

    f.write(
        "Residual96 -> Dropout0.25\n"
    )

    f.write(
        "GAP + GMP -> Dense96 -> Dropout0.20 -> Softmax10\n"
    )


# ============================================================
# Callbacks
# ============================================================

class GradualAugmentationCurriculum(
    tf.keras.callbacks.Callback
):

    def __init__(
            self,
            augmentation_layer,
            augmentation_levels,
            accuracy_threshold,
            consecutive_epochs,
            maximum_clean_epochs,
            epochs_per_stage
    ):

        super().__init__()

        self.augmentation_layer = augmentation_layer
        self.augmentation_levels = tuple(augmentation_levels)
        self.accuracy_threshold = accuracy_threshold
        self.consecutive_epochs = consecutive_epochs
        self.maximum_clean_epochs = maximum_clean_epochs
        self.epochs_per_stage = epochs_per_stage
        self.stable_count = 0
        self.current_stage = 0
        self.epochs_in_current_stage = 0
        self.stage_start_epochs = {0: 1}

    def on_train_begin(self, logs=None):

        del logs

        self.current_stage = 0
        self.epochs_in_current_stage = 0
        self.augmentation_layer.stage.assign(0)

        print()
        print("Curriculum stage 0: clean training, augmentation disabled.")

    def _start_stage(self, new_stage, start_epoch, reason):

        self.current_stage = new_stage
        self.epochs_in_current_stage = 0
        self.augmentation_layer.stage.assign(new_stage)
        self.stage_start_epochs[new_stage] = start_epoch

        (
            translation,
            rotation,
            scale_variation,
            elastic_probability,
            elastic_scale_min,
            elastic_scale_max
        ) = self.augmentation_levels[new_stage]

        print()
        print(
            f"Curriculum stage {new_stage} begins at epoch "
            f"{start_epoch}: translation ±{translation * 100:.0f}%, "
            f"rotation ±{rotation:.0f} degrees, "
            f"scale {1.0 - scale_variation:.2f}–"
            f"{1.0 + scale_variation:.2f}, "
            f"elastic probability {elastic_probability:.0%}, "
            f"elastic scale {elastic_scale_min:.2f}–"
            f"{elastic_scale_max:.2f} ({reason})."
        )

    def on_epoch_end(self, epoch, logs=None):

        logs = logs if logs is not None else {}

        (
            translation,
            rotation,
            scale_variation,
            elastic_probability,
            elastic_scale_min,
            elastic_scale_max
        ) = self.augmentation_levels[self.current_stage]

        logs["augmentation_stage"] = float(self.current_stage)
        logs["translation_percent"] = translation * 100.0
        logs["rotation_degrees"] = rotation
        logs["scale_min"] = 1.0 - scale_variation
        logs["scale_max"] = 1.0 + scale_variation
        logs["elastic_probability"] = elastic_probability
        logs["elastic_scale_min"] = elastic_scale_min
        logs["elastic_scale_max"] = elastic_scale_max

        if self.current_stage == 0:

            online_train_accuracy = float(
                logs.get("accuracy", 0.0)
            )

            if online_train_accuracy >= self.accuracy_threshold:
                self.stable_count += 1
            else:
                self.stable_count = 0

            stable = (
                    self.stable_count
                    >=
                    self.consecutive_epochs
            )

            clean_limit_reached = (
                    epoch + 1
                    >=
                    self.maximum_clean_epochs
            )

            if stable or clean_limit_reached:
                reason = (
                    "online training-accuracy stability criterion"
                    if stable
                    else
                    "maximum clean-epoch limit"
                )

                self._start_stage(
                    new_stage=1,
                    start_epoch=epoch + 2,
                    reason=reason
                )

            return

        self.epochs_in_current_stage += 1

        final_stage = len(self.augmentation_levels) - 1

        if (
                self.current_stage < final_stage
                and
                self.epochs_in_current_stage >= self.epochs_per_stage
        ):
            self._start_stage(
                new_stage=self.current_stage + 1,
                start_epoch=epoch + 2,
                reason=f"completed {self.epochs_per_stage} epochs at prior level"
            )


curriculum_callback = GradualAugmentationCurriculum(
    augmentation_layer=data_augmentation,
    augmentation_levels=AUGMENTATION_LEVELS,
    accuracy_threshold=STABLE_TRAIN_ACCURACY,
    consecutive_epochs=STABLE_EPOCHS_REQUIRED,
    maximum_clean_epochs=MAX_CLEAN_EPOCHS,
    epochs_per_stage=AUGMENTATION_STAGE_EPOCHS
)

checkpoint_callback = (
    tf.keras.callbacks.ModelCheckpoint(
        filepath=
        BEST_MODEL_PATH,

        monitor=
        "val_loss",

        save_best_only=
        True,

        save_weights_only=
        False,

        verbose=
        1
    )
)


def learning_rate_for_epoch(epoch, current_learning_rate):
    del current_learning_rate

    return (
            INITIAL_LR
            *
            LR_DECAY_PER_EPOCH ** epoch
    )


learning_rate_callback = (
    tf.keras.callbacks.LearningRateScheduler(
        learning_rate_for_epoch,
        verbose=1
    )
)

accuracy_checkpoint_callback = (
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_ACCURACY_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
)

# Validation metrics and checkpoints use EMA weights. The callback must
# appear before ModelCheckpoint so the saved best model contains EMA weights.
ema_swap_callback = (
    tf.keras.callbacks.SwapEMAWeights(
        swap_on_epoch=True
    )
)

csv_logger_callback = (
    tf.keras.callbacks.CSVLogger(
        TRAINING_LOG_PATH,
        append=
        False
    )
)

# ============================================================
# Train
# ============================================================

print()
print("=" * 80)

print(
    "TRAINING RESIDUAL CNN V9.1 LONG ELASTIC CURRICULUM"
)

print("=" * 80)

print("Augmentation schedule:")

for stage, (
        translation,
        rotation,
        scale_variation,
        elastic_probability,
        elastic_scale_min,
        elastic_scale_max
) in enumerate(AUGMENTATION_LEVELS):
    print(
        f"Stage {stage}: translation ±{translation * 100:.0f}%, "
        f"rotation ±{rotation:.0f} degrees, "
        f"scale {1.0 - scale_variation:.2f}–"
        f"{1.0 + scale_variation:.2f}, "
        f"elastic probability {elastic_probability:.0%}, "
        f"elastic scale {elastic_scale_min:.2f}–"
        f"{elastic_scale_max:.2f}"
    )

history = model.fit(

    x_train,
    y_train,

    epochs=
    EPOCHS,

    batch_size=
    BATCH_SIZE,

    validation_data=
    (
        x_validation,
        y_validation
    ),

    shuffle=
    True,

    callbacks=
    [
        learning_rate_callback,
        curriculum_callback,
        ema_swap_callback,
        checkpoint_callback,
        accuracy_checkpoint_callback,
        csv_logger_callback
    ],

    verbose=
    1
)

# ============================================================
# Fixed-Horizon Official Model
# ============================================================

print()
print("=" * 80)

print(
    "USING FIXED EPOCH-100 EMA MODEL"
)

print("=" * 80)

# With use_ema=True, Keras finalizes the model variables to EMA values at the
# end of fit. SwapEMAWeights also preserves EMA values on the final epoch.
# The 100-example validation monitor does not choose this official model.

# ============================================================
# Evaluate
# ============================================================

test_loss, test_accuracy = (
    model.evaluate(
        x_test,
        y_test,
        verbose=
        1
    )
)

probabilities = model.predict(
    x_test,
    batch_size=
    BATCH_SIZE,
    verbose=
    1
)

predictions = np.argmax(
    probabilities,
    axis=
    1
)

wrong_indices = np.where(
    predictions
    !=
    y_test
)[0]

wrong_count = len(
    wrong_indices
)

correct_count = (
        len(
            y_test
        )
        -
        wrong_count
)

error_rate = (
        wrong_count
        /
        len(
            y_test
        )
)

print()
print("=" * 80)

print(
    "TEST RESULT"
)

print("=" * 80)

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
# Save Final Model
# ============================================================

model.save(
    FINAL_MODEL_PATH
)

# ============================================================
# Training Curves
# ============================================================

epochs_axis = np.arange(
    1,
    len(history.history["loss"]) + 1
)

plt.figure(
    figsize=(
        8,
        5
    )
)

plt.plot(
    epochs_axis,
    history.history[
        "loss"
    ],
    label=
    "Train Loss"
)

plt.plot(
    epochs_axis,
    history.history[
        "val_loss"
    ],
    label=
    "Validation Loss"
)

for stage, start_epoch in sorted(
        curriculum_callback.stage_start_epochs.items()
):
    if stage > 0:
        plt.axvline(
            start_epoch,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6
        )

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Residual CNN V9.1 Long Elastic Curriculum Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    LOSS_CURVE_PATH,
    dpi=
    250
)

plt.close()

plt.figure(
    figsize=(
        8,
        5
    )
)

plt.plot(
    epochs_axis,
    history.history[
        "accuracy"
    ],
    label=
    "Train Accuracy"
)

plt.plot(
    epochs_axis,
    history.history[
        "val_accuracy"
    ],
    label=
    "Validation Accuracy"
)

for stage, start_epoch in sorted(
        curriculum_callback.stage_start_epochs.items()
):
    if stage > 0:
        plt.axvline(
            start_epoch,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6
        )

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Residual CNN V9.1 Long Elastic Curriculum Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ACCURACY_CURVE_PATH,
    dpi=
    250
)

plt.close()

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

cm_df = pd.DataFrame(
    cm,

    index=[
        f"True_{i}"
        for i
        in range(
            10
        )
    ],

    columns=[
        f"Pred_{i}"
        for i
        in range(
            10
        )
    ]
)

cm_df.to_csv(
    CONFUSION_CSV_PATH,
    encoding=
    "utf-8-sig"
)

plt.figure(
    figsize=(
        10,
        8
    )
)

plt.imshow(
    cm
)

plt.title(
    "Residual CNN V9.1 Long Elastic Curriculum Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    range(
        10
    )
)

plt.yticks(
    range(
        10
    )
)

for i in range(
        10
):

    for j in range(
            10
    ):
        plt.text(
            j,
            i,
            cm[
                i,
                j
            ],
            ha=
            "center",
            va=
            "center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    CONFUSION_PNG_PATH,
    dpi=
    250
)

plt.close()

# ============================================================
# Misclassified List
# ============================================================

with open(
        MISCLASSIFIED_TXT_PATH,
        "w",
        encoding=
        "utf-8"
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

    for idx in (
            wrong_indices
    ):
        true_label = int(
            y_test[
                idx
            ]
        )

        pred_label = int(
            predictions[
                idx
            ]
        )

        confidence = float(
            probabilities[
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
# Save Misclassified Images
# ============================================================

for idx in (
        wrong_indices
):
    true_label = int(
        y_test[
            idx
        ]
    )

    pred_label = int(
        predictions[
            idx
        ]
    )

    confidence = float(
        probabilities[
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
        figsize=(
            3,
            3
        )
    )

    plt.imshow(
        x_test[
            idx
        ].squeeze(),
        cmap=
        "gray"
    )

    plt.title(
        f"Index {idx}\n"
        f"True {true_label} | "
        f"Pred {pred_label}\n"
        f"Conf "
        f"{confidence * 100:.2f}%"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        MISCLASSIFIED_DIR /
        filename,
        dpi=
        160
    )

    plt.close()

# ============================================================
# Misclassified Grid
# ============================================================

if wrong_count > 0:

    cols = 6

    rows = int(
        np.ceil(
            wrong_count
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

    for grid_position, idx in enumerate(
            wrong_indices,
            start=
            1
    ):
        true_label = int(
            y_test[
                idx
            ]
        )

        pred_label = int(
            predictions[
                idx
            ]
        )

        confidence = float(
            probabilities[
                idx,
                pred_label
            ]
        )

        plt.subplot(
            rows,
            cols,
            grid_position
        )

        plt.imshow(
            x_test[
                idx
            ].squeeze(),
            cmap=
            "gray"
        )

        plt.title(
            f"{idx}\n"
            f"T:{true_label} "
            f"P:{pred_label}\n"
            f"{confidence * 100:.1f}%",
            fontsize=
            8
        )

        plt.axis(
            "off"
        )

    plt.tight_layout()

    plt.savefig(
        MISCLASSIFIED_GRID_PATH,
        dpi=
        200
    )

    plt.close()

# ============================================================
# Per-Class Recall
# ============================================================

per_class_rows = []

for digit in range(
        10
):
    true_count = np.sum(
        y_test
        ==
        digit
    )

    correct_digit = np.sum(
        (
                y_test
                ==
                digit
        )
        &
        (
                predictions
                ==
                digit
        )
    )

    recall = (
            correct_digit
            /
            true_count
    )

    per_class_rows.append({

        "Digit":
            digit,

        "Total":
            int(
                true_count
            ),

        "Correct":
            int(
                correct_digit
            ),

        "Errors":
            int(
                true_count
                -
                correct_digit
            ),

        "Recall":
            recall,

        "Recall_Percent":
            recall
            *
            100
    })

pd.DataFrame(
    per_class_rows
).to_csv(
    PER_CLASS_RECALL_PATH,
    index=
    False,
    encoding=
    "utf-8-sig"
)

# ============================================================
# Confusion Pairs
# ============================================================

confusion_rows = []

for true_label in range(
        10
):

    for pred_label in range(
            10
    ):

        if (
                true_label
                ==
                pred_label
        ):
            continue

        count = int(
            cm[
                true_label,
                pred_label
            ]
        )

        if count > 0:
            confusion_rows.append({

                "True":
                    true_label,

                "Predicted":
                    pred_label,

                "Count":
                    count
            })

confusion_pairs_df = pd.DataFrame(
    confusion_rows
)

if not confusion_pairs_df.empty:
    confusion_pairs_df = (
        confusion_pairs_df
        .sort_values(
            by=
            "Count",
            ascending=
            False
        )
    )

confusion_pairs_df.to_csv(
    CONFUSION_PAIRS_PATH,
    index=
    False,
    encoding=
    "utf-8-sig"
)

# ============================================================
# High-Confidence Errors
# ============================================================

high_confidence_rows = []

for idx in (
        wrong_indices
):

    true_label = int(
        y_test[
            idx
        ]
    )

    pred_label = int(
        predictions[
            idx
        ]
    )

    confidence = float(
        probabilities[
            idx,
            pred_label
        ]
    )

    if confidence >= 0.90:
        high_confidence_rows.append({

            "Index":
                int(
                    idx
                ),

            "True":
                true_label,

            "Predicted":
                pred_label,

            "Confidence":
                confidence,

            "Confidence_Percent":
                confidence
                *
                100
        })

high_confidence_df = pd.DataFrame(
    high_confidence_rows
)

if not high_confidence_df.empty:
    high_confidence_df = (
        high_confidence_df
        .sort_values(
            by=
            "Confidence",
            ascending=
            False
        )
    )

high_confidence_df.to_csv(
    HIGH_CONFIDENCE_ERRORS_PATH,
    index=
    False,
    encoding=
    "utf-8-sig"
)

# ============================================================
# Best Epoch
# ============================================================

val_losses = history.history[
    "val_loss"
]

best_epoch = int(
    np.argmin(
        val_losses
    )
    +
    1
)

best_val_loss = float(
    np.min(
        val_losses
    )
)

val_accuracies = history.history[
    "val_accuracy"
]

best_accuracy_epoch = int(
    np.argmax(
        val_accuracies
    )
    +
    1
)

best_val_accuracy = float(
    np.max(
        val_accuracies
    )
)

# ============================================================
# Result File
# ============================================================

result_lines = []

result_lines.append(
    "=" * 80
)

result_lines.append(
    "MNIST RESIDUAL CNN V9.1 - LONG CURRICULUM + 100 MONITOR + EMA"
)

result_lines.append(
    "=" * 80
)

result_lines.append(
    ""
)

result_lines.append(
    f"Experiment:"
)

result_lines.append(
    EXPERIMENT_NAME
)

result_lines.append(
    ""
)

result_lines.append(
    "AUGMENTATION"
)

result_lines.append(
    "-" * 80
)

for stage, (
        translation,
        rotation,
        scale_variation,
        elastic_probability,
        elastic_scale_min,
        elastic_scale_max
) in enumerate(AUGMENTATION_LEVELS):
    result_lines.append(
        f"Stage {stage}: translation ±{translation * 100:.0f}%, "
        f"rotation ±{rotation:.0f} degrees, "
        f"scale {1.0 - scale_variation:.2f}–"
        f"{1.0 + scale_variation:.2f}, "
        f"elastic probability {elastic_probability:.0%}, "
        f"elastic scale {elastic_scale_min:.2f}–"
        f"{elastic_scale_max:.2f}"
    )

result_lines.append(
    ""
)

result_lines.append(
    "TRAINING"
)

result_lines.append(
    "-" * 80
)

result_lines.append(
    f"Seed: "
    f"{SEED}"
)

result_lines.append(
    f"Batch size: "
    f"{BATCH_SIZE}"
)

result_lines.append(
    f"Training samples: "
    f"{len(x_train)}"
)

result_lines.append(
    f"Validation samples: "
    f"{len(x_validation)}"
)

result_lines.append(
    "Official checkpoint policy: fixed epoch-100 EMA model"
)

result_lines.append(
    f"Initial learning rate: "
    f"{INITIAL_LR}"
)

result_lines.append(
    f"LR decay per epoch: "
    f"{LR_DECAY_PER_EPOCH}"
)

result_lines.append(
    f"EMA momentum: "
    f"{EMA_MOMENTUM}"
)

result_lines.append(
    f"Epochs per intermediate augmentation stage: "
    f"{AUGMENTATION_STAGE_EPOCHS}"
)

result_lines.append("Stage start epochs:")

for stage, start_epoch in sorted(
        curriculum_callback.stage_start_epochs.items()
):
    result_lines.append(
        f"  Stage {stage}: epoch {start_epoch}"
    )

result_lines.append(
    f"Epochs completed: "
    f"{len(history.history['loss'])}"
)

result_lines.append(
    f"Best validation-loss epoch: "
    f"{best_epoch}"
)

result_lines.append(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

result_lines.append(
    f"Best validation-accuracy epoch: "
    f"{best_accuracy_epoch}"
)

result_lines.append(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.4f}%"
)

result_lines.append(
    ""
)

result_lines.append(
    "TEST"
)

result_lines.append(
    "-" * 80
)

result_lines.append(
    f"Test loss: "
    f"{test_loss:.6f}"
)

result_lines.append(
    f"Accuracy: "
    f"{test_accuracy * 100:.4f}%"
)

result_lines.append(
    f"Correct: "
    f"{correct_count}"
)

result_lines.append(
    f"Wrong: "
    f"{wrong_count}"
)

result_lines.append(
    f"Error rate: "
    f"{error_rate * 100:.4f}%"
)

result_lines.append(
    ""
)

result_lines.append(
    "MODEL"
)

result_lines.append(
    "-" * 80
)

result_lines.append(
    "Best validation-accuracy model (diagnostic only):"
)

result_lines.append(
    str(
        BEST_ACCURACY_MODEL_PATH
    )
)

result_lines.append(
    ""
)

result_lines.append(
    "Best validation-loss model (diagnostic only):"
)

result_lines.append(
    str(
        BEST_MODEL_PATH
    )
)

result_lines.append(
    ""
)

result_lines.append(
    f"Final epoch-100 EMA model used for test and saved:"
)

result_lines.append(
    str(
        FINAL_MODEL_PATH
    )
)

result_lines.append(
    ""
)

result_lines.append(
    "=" * 80
)

result_text = "\n".join(
    result_lines
)

with open(
        RESULT_PATH,
        "w",
        encoding=
        "utf-8"
) as f:
    f.write(
        result_text
    )

# ============================================================
# Final Print
# ============================================================

print()
print(
    result_text
)

print()
print(
    "Logs:"
)

print(
    EXPERIMENT_DIR
)

print()
print(
    "Final epoch-100 EMA model used for test:"
)

print(
    FINAL_MODEL_PATH
)




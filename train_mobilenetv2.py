import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------

DATASET_PATH = "asl_split_dataset"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VALIDATION_PATH = os.path.join(DATASET_PATH, "validation")
TEST_PATH = os.path.join(DATASET_PATH, "test")

IMG_SIZE = 128
BATCH_SIZE = 32

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

MODEL_NAME = "asl_mobilenetv2.keras"

PHASE1_MODEL = "phase1_best.keras"
PHASE2_MODEL = "phase2_best.keras"

SEED = 42


# --------------------------------------------
# 2. CHECK DATASET FOLDERS
# --------------------------------------------

for folder in [TRAIN_PATH, VALIDATION_PATH, TEST_PATH]:
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

print("\nDataset folders found successfully!")


# --------------------------------------------
# 3. LOAD DATASETS
# --------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------
# 4. VERIFY CLASS NAMES
# --------------------------------------------

class_names = train_ds.class_names

if len(class_names) != 26:
    raise ValueError(
        f"Expected 26 classes, but found {len(class_names)}"
    )

if validation_ds.class_names != class_names:
    raise ValueError("Validation class names do not match!")

if test_ds.class_names != class_names:
    raise ValueError("Test class names do not match!")

print("\nClasses:", class_names)

print("\nDataset loaded successfully!")
print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Validation batches:", tf.data.experimental.cardinality(validation_ds).numpy())
print("Testing batches:", tf.data.experimental.cardinality(test_ds).numpy())


# --------------------------------------------
# 5. PREFETCH DATASETS
# --------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# --------------------------------------------
# 6. DATA AUGMENTATION
# --------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15)
])


# --------------------------------------------
# 7. LOAD MOBILENETV2
# --------------------------------------------

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# --------------------------------------------
# 8. BUILD THE MODEL
# --------------------------------------------

inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = data_augmentation(inputs)

x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    26,
    activation="softmax"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# --------------------------------------------
# 9. COMPILE MODEL
# --------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------
# 10. PHASE 1 - INITIAL TRAINING
# --------------------------------------------

print("\n====================================")
print("PHASE 1: INITIAL TRAINING")
print("====================================")

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    PHASE1_MODEL,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history_initial = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# --------------------------------------------
# 11. PHASE 2 - FINE TUNING
# --------------------------------------------

print("\n====================================")
print("PHASE 2: FINE TUNING")
print("====================================")

base_model.trainable = True

# Freeze the earlier layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


early_stopping_fine = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True
)


checkpoint_fine = ModelCheckpoint(
    PHASE2_MODEL,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


history_fine = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        early_stopping_fine,
        checkpoint_fine
    ]
)


# --------------------------------------------
# 12. SELECT THE BEST VALIDATION MODEL
# --------------------------------------------

print("\n====================================")
print("SELECTING BEST MODEL")
print("====================================")

phase1_best = load_model(PHASE1_MODEL)
phase2_best = load_model(PHASE2_MODEL)

phase1_loss, phase1_accuracy = phase1_best.evaluate(
    validation_ds,
    verbose=0
)

phase2_loss, phase2_accuracy = phase2_best.evaluate(
    validation_ds,
    verbose=0
)

print("Phase 1 validation accuracy:", phase1_accuracy)
print("Phase 2 validation accuracy:", phase2_accuracy)

if phase2_accuracy >= phase1_accuracy:
    best_model = phase2_best
    print("\nPhase 2 model selected.")
else:
    best_model = phase1_best
    print("\nPhase 1 model selected.")

best_model.save(MODEL_NAME)

print("Best model saved as:", MODEL_NAME)


# --------------------------------------------
# 13. INDEPENDENT TEST EVALUATION
# --------------------------------------------

print("\n====================================")
print("FINAL TEST EVALUATION")
print("====================================")

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)


# --------------------------------------------
# 14. GENERATE PREDICTIONS
# --------------------------------------------

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_ds
])

predictions = best_model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# --------------------------------------------
# 15. CALCULATE EVALUATION METRICS
# --------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

correct_predictions = np.sum(
    y_true == y_pred
)

incorrect_predictions = np.sum(
    y_true != y_pred
)

total_test_images = len(y_true)


print("\n====================================")
print("TEST RESULTS")
print("====================================")

print("Test images:", total_test_images)

print("Correct predictions:", correct_predictions)

print("Incorrect predictions:", incorrect_predictions)

print("Test accuracy: {:.2f}%".format(
    accuracy * 100
))

print("Test loss: {:.4f}".format(
    test_loss
))


# --------------------------------------------
# 16. CLASSIFICATION REPORT
# --------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(26)),
    target_names=class_names,
    zero_division=0
)

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(report)

with open(
    "classification_report.txt",
    "w"
) as file:

    file.write(report)


# --------------------------------------------
# 17. CONFUSION MATRIX
# --------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(26))
)

plt.figure(figsize=(18, 15))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("ASL Alphabet Recognition - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

print("\nConfusion matrix saved as confusion_matrix.png")

plt.show()


# --------------------------------------------
# 18. SAVE CLASS NAMES
# --------------------------------------------

with open(
    "class_names_mobilenetv2.txt",
    "w"
) as file:

    file.write("\n".join(class_names))


# --------------------------------------------
# 19. FINAL MESSAGE
# --------------------------------------------

print("\n====================================")
print("TRAINING AND EVALUATION COMPLETED")
print("====================================")

print("Model:", MODEL_NAME)
print("Classification report: classification_report.txt")
print("Confusion matrix: confusion_matrix.png")

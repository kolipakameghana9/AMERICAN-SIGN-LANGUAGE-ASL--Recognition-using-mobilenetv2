import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =========================================================
# ASL A-Z RECOGNITION - IMPROVED TRAINING
# =========================================================

print("==============================================")
print("       ASL A-Z MODEL TRAINING")
print("==============================================")

# =========================================================
# 1. SETTINGS
# =========================================================

DATASET_PATH = "processed_dataset"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30

print("\nDataset:", DATASET_PATH)
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)


# =========================================================
# 2. CHECK DATASET
# =========================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR!")
    print("processed_dataset folder was not found.")
    print("Make sure you are running from ASL_recognition.")
    exit()

print("\nDataset found successfully!")


# =========================================================
# 3. LOAD TRAINING DATA
# =========================================================

print("\nLoading training images...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =========================================================
# 4. LOAD VALIDATION DATA
# =========================================================

print("Loading validation images...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================================================
# 5. GET CLASS NAMES
# =========================================================

class_names = train_dataset.class_names

print("\n==============================================")
print("Classes:")
print(class_names)

print("\nNumber of classes:", len(class_names))
print("==============================================")


# =========================================================
# 6. IMPROVE DATA PIPELINE
# =========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(
    1000
).prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)


# =========================================================
# 7. DATA AUGMENTATION
# =========================================================
#
# IMPORTANT:
# We DO NOT use horizontal flipping.
#
# Horizontal flipping can change the meaning/orientation
# of some ASL hand signs.
#
# =========================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomRotation(
        0.10
    ),

    layers.RandomZoom(
        height_factor=0.10,
        width_factor=0.10
    ),

    layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    ),

    layers.RandomContrast(
        0.15
    ),

    layers.RandomBrightness(
        0.15
    )

], name="data_augmentation")


# =========================================================
# 8. BUILD IMPROVED CNN MODEL
# =========================================================

print("\nBuilding improved CNN model...")

model = models.Sequential([

    # Data augmentation
    data_augmentation,

    # Normalize pixel values
    layers.Rescaling(1.0 / 255),

    # -----------------------------------------
    # BLOCK 1
    # -----------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -----------------------------------------
    # BLOCK 2
    # -----------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -----------------------------------------
    # BLOCK 3
    # -----------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -----------------------------------------
    # BLOCK 4
    # -----------------------------------------

    layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -----------------------------------------
    # CLASSIFICATION
    # -----------------------------------------

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    # 26 ASL letters
    layers.Dense(
        26,
        activation="softmax"
    )

])


# =========================================================
# 9. SHOW MODEL
# =========================================================

print("\nModel architecture:")

model.summary()


# =========================================================
# 10. COMPILE MODEL
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# =========================================================
# 11. EARLY STOPPING
# =========================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    restore_best_weights=True,

    verbose=1

)


# =========================================================
# 12. SAVE BEST MODEL
# =========================================================

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    "asl_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)


# =========================================================
# 13. START TRAINING
# =========================================================

print("\n==============================================")
print("           STARTING TRAINING")
print("==============================================")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]

)


# =========================================================
# 14. FINAL EVALUATION
# =========================================================

print("\n==============================================")
print("          FINAL MODEL EVALUATION")
print("==============================================")

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print(
    f"\nValidation Accuracy: {accuracy * 100:.2f}%"
)


# =========================================================
# 15. SAVE FINAL MODEL
# =========================================================

model.save(
    "asl_model.keras"
)

print("\nModel saved successfully!")

print(
    "File: asl_model.keras"
)


# =========================================================
# 16. SAVE CLASS NAMES
# =========================================================

with open(
    "class_names.txt",
    "w"
) as file:

    for name in class_names:

        file.write(
            name + "\n"
        )

print(
    "Class names saved successfully!"
)

print(
    "File: class_names.txt"
)


# =========================================================
# 17. TRAINING FINISHED
# =========================================================

print("\n==============================================")
print("        ASL TRAINING COMPLETED!")
print("==============================================")

print("\nYour improved model is ready.")

print("\nNext step:")
print("Run: python realtime.py")
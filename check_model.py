import tensorflow as tf
import numpy as np
import os

print("====================================")
print("ASL MODEL DIAGNOSTIC TEST")
print("====================================")

# Load model
print("\nLoading model...")

model = tf.keras.models.load_model("asl_model.keras")

print("Model loaded successfully!")

# Load class names
with open("class_names.txt", "r") as file:
    class_names = [line.strip() for line in file]

print("\nClasses:")
print(class_names)

# Check dataset
dataset_path = "processed_dataset"

print("\nChecking processed dataset...")

if not os.path.exists(dataset_path):
    print("ERROR: processed_dataset folder not found!")
    exit()

# Letters to test
test_letters = [
    "A", "B", "C", "D", "E",
    "F", "I", "O", "P", "S", "X"
]

print("\n====================================")
print("TESTING DIFFERENT LETTER IMAGES")
print("====================================")

for letter in test_letters:

    folder = os.path.join(
        dataset_path,
        letter
    )

    if not os.path.exists(folder):
        print(f"{letter}: folder not found")
        continue

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if len(images) == 0:
        print(f"{letter}: NO IMAGES")
        continue

    image_path = os.path.join(
        folder,
        images[0]
    )

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(128, 128)
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_letter = class_names[
        predicted_index
    ]

    confidence = (
        prediction[0][predicted_index] * 100
    )

    print(
        f"Actual: {letter}  "
        f"Predicted: {predicted_letter}  "
        f"Confidence: {confidence:.2f}%"
    )

print("\n====================================")
print("TEST COMPLETED")
print("====================================")
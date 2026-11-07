import tensorflow as tf
import numpy as np
import cv2
import os

# Load model
model = tf.keras.models.load_model("asl_model.keras")

# Load class names
with open("class_names.txt", "r") as file:
    class_names = [line.strip() for line in file]

print("Classes:", class_names)
print("\nTesting saved dataset images...\n")

for actual_letter in class_names:

    folder = os.path.join("processed_dataset", actual_letter)

    if not os.path.exists(folder):
        continue

    images = os.listdir(folder)

    if len(images) == 0:
        continue

    image_name = images[0]
    image_path = os.path.join(folder, image_name)

    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.resize(image, (128, 128))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32)

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    predicted_letter = class_names[predicted_index]

    confidence = prediction[predicted_index] * 100

    print(
        f"Actual: {actual_letter}   "
        f"Predicted: {predicted_letter}   "
        f"Confidence: {confidence:.2f}%"
    )
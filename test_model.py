import tensorflow as tf
import cv2
import numpy as np

# ==============================
# 1. LOAD TRAINED MODEL
# ==============================

print("Loading model...")

model = tf.keras.models.load_model("asl_model.keras")

print("Model loaded successfully!")

# ==============================
# 2. LOAD CLASS NAMES
# ==============================

with open("class_names.txt", "r") as file:
    class_names = [line.strip() for line in file]

print("Classes:", class_names)

# ==============================
# 3. IMAGE TO TEST
# ==============================

image_path = "processed_dataset/A/A_1.jpg"

print("\nTesting image:")
print(image_path)

# ==============================
# 4. READ IMAGE USING OPENCV
# ==============================

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not read the image.")
    exit()

print("Image loaded successfully!")

# ==============================
# 5. RESIZE IMAGE
# ==============================

image = cv2.resize(image, (128, 128))

# OpenCV reads images as BGR.
# Convert BGR to RGB.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to NumPy array
image = np.array(image, dtype=np.float32)

# Add batch dimension
image = np.expand_dims(image, axis=0)

# ==============================
# 6. PREDICT
# ==============================

print("Making prediction...")

predictions = model.predict(image, verbose=0)

# Find highest probability
predicted_index = np.argmax(predictions[0])

predicted_letter = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100

# ==============================
# 7. DISPLAY RESULT
# ==============================

print()
print("==============================")
print("       ASL PREDICTION")
print("==============================")

print("Image:", image_path)
print("Predicted Letter:", predicted_letter)
print(f"Confidence: {confidence:.2f}%")

print("==============================")
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("asl_model.keras")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print("Model loaded successfully!")
print("Classes:", class_names)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

# Prediction smoothing
prediction_history = deque(maxlen=5)

print("Camera started.")
print("Put your hand inside the blue box.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # =========================
    # BLUE BOX
    # =========================
    x1, y1 = 105, 105
    x2, y2 = 495, 395

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3
    )

    # =========================
    # CROP HAND AREA
    # =========================
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    # Resize to model input size
    img = cv2.resize(roi, (128, 128))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to numpy array
    img = np.array(img, dtype=np.float32)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # =========================
    # PREDICTION
    # =========================
    predictions = model.predict(img, verbose=0)

    confidence = float(np.max(predictions[0]))

    predicted_index = int(np.argmax(predictions[0]))

    predicted_letter = class_names[predicted_index]

    # =========================
    # SMOOTH PREDICTION
    # =========================
    prediction_history.append(predicted_letter)

    most_common = max(
        set(prediction_history),
        key=prediction_history.count
    )

    # =========================
    # DISPLAY
    # =========================

    cv2.putText(
        frame,
        "Letter: " + most_common,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "Confidence: {:.2f}%".format(confidence * 100),
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow("Real-Time ASL Recognition", frame)

    # Quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# =========================
# CLOSE
# =========================
cap.release()
cv2.destroyAllWindows()

print("Program stopped.")
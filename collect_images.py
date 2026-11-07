import cv2
import os

# Ask for letter
letter = input("Enter the letter (A-Z): ").upper()

# Create folder
save_folder = os.path.join("dataset", letter)
os.makedirs(save_folder, exist_ok=True)

# Start camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera could not be opened.")
    exit()

count = 0

print()
print("Camera started!")
print("Put your hand inside the blue box.")
print("LEFT CLICK = Capture image")
print("Q = Quit")
print()


# Mouse function
def capture_image(event, x, y, flags, param):
    global count

    if event == cv2.EVENT_LBUTTONDOWN:

        filename = os.path.join(
            save_folder,
            f"{letter}_{count + 1}.jpg"
        )

        success = cv2.imwrite(filename, frame)

        if success:
            count += 1
            print(f"Saved: {filename}")
        else:
            print("ERROR: Image could not be saved.")


# Create camera window
cv2.namedWindow("ASL Image Collection")

# Connect mouse to camera window
cv2.setMouseCallback(
    "ASL Image Collection",
    capture_image
)


while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Blue rectangle
    cv2.rectangle(
        frame,
        (100, 100),
        (500, 400),
        (255, 0, 0),
        2
    )

    # Letter
    cv2.putText(
        frame,
        f"Letter: {letter}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Image count
    cv2.putText(
        frame,
        f"Images: {count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Instructions
    cv2.putText(
        frame,
        "LEFT CLICK = Capture | Q = Quit",
        (20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Show camera
    cv2.imshow(
        "ASL Image Collection",
        frame
    )

    # Keyboard only for Q
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break


camera.release()
cv2.destroyAllWindows()

print()
print("Finished!")
print(f"Total images collected for {letter}: {count}")
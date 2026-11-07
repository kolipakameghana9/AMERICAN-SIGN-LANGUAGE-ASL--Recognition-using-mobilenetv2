import cv2
import os

# Main folders
input_folder = "dataset"
output_folder = "processed_dataset"

# Image size for the model
IMG_SIZE = 128

# A to Z
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Create processed_dataset folder
os.makedirs(output_folder, exist_ok=True)

total_images = 0

for letter in letters:

    # Input folder
    input_letter_folder = os.path.join(input_folder, letter)

    # Output folder
    output_letter_folder = os.path.join(output_folder, letter)

    # Create output letter folder
    os.makedirs(output_letter_folder, exist_ok=True)

    print(f"\nProcessing {letter}...")

    # Check if folder exists
    if not os.path.exists(input_letter_folder):
        print(f"Folder not found: {input_letter_folder}")
        continue

    # Get all files
    files = os.listdir(input_letter_folder)

    count = 0

    for filename in files:

        # Only process image files
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            input_path = os.path.join(
                input_letter_folder,
                filename
            )

            # Read image
            image = cv2.imread(input_path)

            if image is None:
                print(f"Could not read: {filename}")
                continue

            # Crop the hand area
            # Original collection box was approximately:
            # x = 100 to 500
            # y = 100 to 400

            cropped = image[105:395, 105:495]

            # Resize image to 128 x 128
            resized = cv2.resize(
                cropped,
                (IMG_SIZE, IMG_SIZE)
            )

            # Output path
            output_path = os.path.join(
                output_letter_folder,
                filename
            )

            # Save processed image
            cv2.imwrite(
                output_path,
                resized
            )

            count += 1
            total_images += 1

    print(f"{letter}: {count} images processed")


print("\n================================")
print("PREPROCESSING COMPLETED!")
print("================================")
print(f"Total images processed: {total_images}")
print(f"Images saved in: {output_folder}")
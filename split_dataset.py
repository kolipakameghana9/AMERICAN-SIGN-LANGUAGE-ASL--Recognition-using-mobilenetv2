
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("processed_dataset")
OUTPUT_DIR = Path("asl_split_dataset")

SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

if not SOURCE_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {SOURCE_DIR}")

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. Stopping to avoid overwriting files."
    )

class_folders = sorted(
    folder for folder in SOURCE_DIR.iterdir() if folder.is_dir()
)

if len(class_folders) != 26:
    raise ValueError(
        f"Expected 26 class folders (A-Z), found {len(class_folders)}."
    )

random_generator = random.Random(SEED)

total_train = 0
total_validation = 0
total_test = 0

for class_folder in class_folders:
    images = sorted(
        file for file in class_folder.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No images found in {class_folder.name}")

    random_generator.shuffle(images)

    train_end = int(len(images) * 0.70)
    validation_end = train_end + int(len(images) * 0.15)

    split_data = {
        "train": images[:train_end],
        "validation": images[train_end:validation_end],
        "test": images[validation_end:]
    }

    for split_name, split_images in split_data.items():
        destination = OUTPUT_DIR / split_name / class_folder.name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(image_path, destination / image_path.name)

    train_count = len(split_data["train"])
    validation_count = len(split_data["validation"])
    test_count = len(split_data["test"])

    total_train += train_count
    total_validation += validation_count
    total_test += test_count

    print(
        f"{class_folder.name}: "
        f"Train={train_count}, "
        f"Validation={validation_count}, "
        f"Test={test_count}"
    )

print("\nDATASET SPLIT COMPLETED")
print(f"Training images:   {total_train}")
print(f"Validation images: {total_validation}")
print(f"Testing images:    {total_test}")
print(f"Total images:      {total_train + total_validation + total_test}")
print(f"\nOutput folder: {OUTPUT_DIR}")

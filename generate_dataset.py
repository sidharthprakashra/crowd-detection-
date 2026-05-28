from ultralytics import YOLO
import os
import pandas as pd
import cv2

# ======================================================
# LOAD YOLOv8 PRETRAINED MODEL
# ======================================================

model = YOLO("yolov8s.pt")

# ======================================================
# CONFIG
# ======================================================

base_folder = "training_images"

dataset = []

valid_ext = (".jpg", ".jpeg", ".png")

print("\nGenerating crowd dataset...\n")

# ======================================================
# YOLO PERSON COUNT
# ======================================================

def yolo_count(image):

    results = model.predict(
        image,
        conf=0.25,
        imgsz=960,
        verbose=False
    )

    count = 0

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0])

            # PERSON CLASS
            if cls == 0:
                count += 1

    return count

# ======================================================
# FOLDER TRAVERSAL
# ======================================================

for day_type in os.listdir(base_folder):

    day_path = os.path.join(base_folder, day_type)

    if not os.path.isdir(day_path):
        continue

    for location in os.listdir(day_path):

        location_path = os.path.join(day_path, location)

        if not os.path.isdir(location_path):
            continue

        for time_slot in os.listdir(location_path):

            time_path = os.path.join(location_path, time_slot)

            if not os.path.isdir(time_path):
                continue

            for img_name in os.listdir(time_path):

                if not img_name.lower().endswith(valid_ext):
                    continue

                img_path = os.path.join(time_path, img_name)

                image = cv2.imread(img_path)

                if image is None:
                    continue

                try:

                    count = yolo_count(image)

                    dataset.append({
                        "day_type": day_type,
                        "location": location,
                        "time_slot": time_slot,
                        "crowd_count": count
                    })

                    print(
                        f"{img_name} -> "
                        f"Count={count}"
                    )

                except Exception as e:

                    print(f"ERROR: {img_name}")
                    print(e)

# ======================================================
# SAVE CSV
# ======================================================

df = pd.DataFrame(dataset)

df.to_csv("crowd_dataset.csv", index=False)

print("\nDataset saved: crowd_dataset.csv")
print(df.head())
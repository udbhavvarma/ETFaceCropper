import os
import uuid
import cv2
import gdown
import pandas as pd
import mediapipe as mp
from PIL import Image

# Initialize MediaPipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def init_folders(download_folder="downloaded_images", cropped_folder="cropped_faces"):
    """
    Create folders for downloads and cropped faces if they don't exist.
    """
    os.makedirs(download_folder, exist_ok=True)
    os.makedirs(cropped_folder, exist_ok=True)


def load_file_ids(csv_path):
    """
    Load Google Drive file IDs from a CSV file with a 'file_id' column.
    """
    df = pd.read_csv(csv_path)
    return df['file_id'].dropna().tolist()


def download_file(file_id, download_folder="downloaded_images"):
    """
    Download a file from Google Drive by its file ID to the download_folder.
    Returns the local file path.
    """
    output_path = os.path.join(download_folder, f"{uuid.uuid4().hex}.jpg")
    gdown.download(id=file_id, output=output_path, quiet=False, use_cookies=False)
    return output_path


def crop_faces_mediapipe(
    image_path,
    cropped_folder="cropped_faces",
    top_padding_ratio=0.6,
    bottom_padding_ratio=0.4,
    left_padding_ratio=0.4,
    right_padding_ratio=0.4,
    base_name=None
):
    """
    Detect and crop faces from an image using MediaPipe, applying padding ratios.
    Saves each face crop into cropped_folder and returns list of paths.
    If base_name is provided, uses that for naming; otherwise uses image filename.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image {image_path}")
        return []

    h, w, _ = img.shape
    results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.detections:
        print(f"❌ No face found in {os.path.basename(image_path)}")
        return []

    cropped_paths = []
    original_name = os.path.splitext(os.path.basename(image_path))[0]
    for i, detection in enumerate(results.detections):
        bbox = detection.location_data.relative_bounding_box
        xmin = int(bbox.xmin * w)
        ymin = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Compute padding
        pad_top = int(height * top_padding_ratio)
        pad_bottom = int(height * bottom_padding_ratio)
        pad_left = int(width * left_padding_ratio)
        pad_right = int(width * right_padding_ratio)

        # Clamp to image bounds
        x1 = max(xmin - pad_left, 0)
        y1 = max(ymin - pad_top, 0)
        x2 = min(xmin + width + pad_right, w)
        y2 = min(ymin + height + pad_bottom, h)

        cropped = img[y1:y2, x1:x2]
        cropped_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        name = base_name or original_name
        out_name = f"{name}.jpg"
        output_path = os.path.join(cropped_folder, out_name)
        cropped_image.save(output_path)
        cropped_paths.append(output_path)

    return cropped_paths


def process_csv(
    csv_path="file_ids.csv",
    download_folder="downloaded_images",
    cropped_folder="cropped_faces"
):
    """
    Full pipeline: initializes folders, loads IDs, downloads images, and crops faces.
    """
    init_folders(download_folder, cropped_folder)
    file_ids = load_file_ids(csv_path)

    for file_id in file_ids:
        print(f"\n📥 Downloading File ID: {file_id}")
        local_path = download_file(file_id, download_folder)
        if local_path:
            cropped_images = crop_faces_mediapipe(
                local_path,
                cropped_folder=cropped_folder,
                base_name=file_id
            )
            print(f"✅ Cropped {len(cropped_images)} face(s) from {file_id}")

    print("\n🎉 All Done!")


if __name__ == "__main__":
    process_csv()

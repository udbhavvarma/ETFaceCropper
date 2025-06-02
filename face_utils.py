import os
import uuid
import time
import random
import requests
import cv2
# Configure OpenCV to run in headless mode
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'  # Enable OpenEXR support if needed
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # Disable Media Foundation backend
import pandas as pd
import mediapipe as mp
from PIL import Image
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


def get_direct_download_link(file_id):
    """Convert Google Drive file ID to direct download link."""
    return f"https://drive.google.com/uc?id={file_id}&export=download"

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True
)
def download_file(file_id, download_folder="downloaded_images"):
    """
    Download a file from Google Drive using multiple methods with retries.
    Returns the local file path if successful, None otherwise.
    """
    os.makedirs(download_folder, exist_ok=True)
    output_path = os.path.join(download_folder, f"{uuid.uuid4().hex}.jpg")
    
    # Add random delay to avoid hitting rate limits
    time.sleep(random.uniform(1, 3))
    
    try:
        # # Method 1: Try direct download with gdown first
        # try:
        #     gdown.download(
        #         id=file_id,
        #         output=output_path,
        #         quiet=False,
        #         use_cookies=True,  # Use cookies to handle large files
        #         fuzzy=True
        #     )
        #     if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        #         return output_path
        # except Exception as e:
        #     print(f"Gdown method failed, trying alternative methods... ({str(e)})")
        
        # Method 2: Try direct download with requests
        direct_link = get_direct_download_link(file_id)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        with requests.get(direct_link, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
            
    except Exception as e:
        print(f"Failed to download file {file_id}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise  # Re-raise to trigger retry
    
    return None


def crop_faces_mediapipe(
    image_path: str,
    cropped_folder: str = "cropped_faces",
    top_padding_ratio: float = 0.2,
    bottom_padding_ratio: float = 0.2,
    left_padding_ratio: float = 0.2,
    right_padding_ratio: float = 0.2,
    base_name: str = None,
    min_face_size: int = 1,  # Minimum face size in pixels
    min_confidence: float = 0.2  # Lower confidence threshold
) -> List[str]:
    """
    Detect and crop faces from an image using MediaPipe with improved error handling.
    
    Args:
        image_path: Path to the input image
        cropped_folder: Directory to save cropped faces
        top_padding_ratio: Padding ratio for top of face
        bottom_padding_ratio: Padding ratio for bottom of face
        left_padding_ratio: Padding ratio for left of face
        right_padding_ratio: Padding ratio for right of face
        base_name: Base name for output files
        min_face_size: Minimum face size in pixels (width or height)
        min_confidence: Minimum confidence score for face detection (0-1)
        
    Returns:
        List of paths to cropped face images
    """
    # Validate input parameters
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return []
        
    if not os.access(image_path, os.R_OK):
        print(f"❌ No read permissions for: {image_path}")
        return []
    
    # Create output directory if it doesn't exist
    os.makedirs(cropped_folder, exist_ok=True)
    
    # Initialize face detector
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(
        model_selection=1,  # 1 for general, 2 for close-up faces
        min_detection_confidence=min_confidence
    )
    
    cropped_paths = []
    original_name = base_name or os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Read and validate image
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"❌ Could not read image (possibly corrupted): {image_path}")
                return []
                
            if img.size == 0:
                print(f"❌ Empty image: {image_path}")
                return []
                
        except Exception as e:
            print(f"❌ Error reading image {image_path}: {str(e)}")
            return []

        # Convert to RGB for MediaPipe
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {str(e)}")
            return []
        
        # Detect faces
        try:
            results = face_detector.process(rgb_img)
        except Exception as e:
            print(f"❌ Face detection failed for {image_path}: {str(e)}")
            return []
            
        if not results.detections:
            print(f"ℹ️ No faces detected in {os.path.basename(image_path)}")
            return []

        # Process each detected face
        for i, detection in enumerate(results.detections):
            try:
                # Get confidence score
                confidence = detection.score[0] if detection.score else 0
                
                # Skip low confidence detections
                if confidence < min_confidence:
                    print(f"ℹ️ Low confidence face ({confidence:.2f}) in {os.path.basename(image_path)}")
                    continue
                
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                xmin = int(bbox.xmin * w)
                ymin = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Skip faces that are too small
                if width < min_face_size or height < min_face_size:
                    print(f"ℹ️ Face too small ({width}x{height}px) in {os.path.basename(image_path)}")
                    continue
                
                # Apply padding
                pad_top = int(height * top_padding_ratio)
                pad_bottom = int(height * bottom_padding_ratio)
                pad_left = int(width * left_padding_ratio)
                pad_right = int(width * right_padding_ratio)
                
                # Calculate crop coordinates with bounds checking
                x1 = max(xmin - pad_left, 0)
                y1 = max(ymin - pad_top, 0)
                x2 = min(xmin + width + pad_right, w)
                y2 = min(ymin + height + pad_bottom, h)
                
                # Ensure valid crop dimensions
                if x2 <= x1 or y2 <= y1:
                    print(f"⚠️ Invalid crop dimensions in {os.path.basename(image_path)}")
                    continue
                
                # Crop the face
                try:
                    cropped = img[y1:y2, x1:x2]
                    if cropped.size == 0:
                        print(f"⚠️ Empty crop for face {i} in {os.path.basename(image_path)}")
                        continue
                        
                    # Convert to RGB and save
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    cropped_image = Image.fromarray(cropped_rgb)
                    
                    # Generate unique filename
                    out_name = f"{original_name}_face{i+1}_{int(confidence*100)}.jpg"
                    output_path = os.path.join(cropped_folder, out_name)
                    
                    # Handle filename conflicts
                    counter = 1
                    while os.path.exists(output_path):
                        name, ext = os.path.splitext(out_name)
                        output_path = os.path.join(cropped_folder, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    # Save with quality settings
                    cropped_image.save(output_path, quality=95, optimize=True, subsampling=0, qtables='web_high')
                    cropped_paths.append(output_path)
                    
                except Exception as e:
                    print(f"⚠️ Error processing face {i+1} in {os.path.basename(image_path)}: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"⚠️ Error in face processing loop for {os.path.basename(image_path)}: {str(e)}")
                continue
                
        return cropped_paths
        
    except Exception as e:
        print(f"❌ Unexpected error processing {image_path}: {str(e)}")
        return []
        
    finally:
        # Clean up resources
        try:
            face_detector.close()
        except:
            pass


def process_csv(
    csv_path="file_ids.csv",
    download_folder="downloaded_images",
    cropped_folder="cropped_faces",
    error_log_file="failed_downloads.csv"
):
    """
    Full pipeline: initializes folders, loads IDs, downloads images, and crops faces.
    Tracks failed downloads and saves them to a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing file IDs
        download_folder (str): Directory to save downloaded images
        cropped_folder (str): Directory to save cropped faces
        error_log_file (str): Path to save the error log CSV
    """
    init_folders(download_folder, cropped_folder)
    file_ids = load_file_ids(csv_path)
    
    # Initialize tracking for failed downloads
    failed_downloads = []
    processed_count = 0
    success_count = 0
    failure_count = 0
    
    print(f"\n🚀 Starting to process {len(file_ids)} files...")
    
    for file_id in file_ids:
        processed_count += 1
        print(f"\n📥 [{processed_count}/{len(file_ids)}] Processing File ID: {file_id}")
        
        try:
            local_path = download_file(file_id, download_folder)
            if local_path:
                cropped_images = crop_faces_mediapipe(
                    local_path,
                    cropped_folder=cropped_folder,
                    base_name=file_id
                )
                if cropped_images:
                    success_count += 1
                    print(f"✅ Successfully processed {file_id} - Cropped {len(cropped_images)} face(s)")
                else:
                    failure_count += 1
                    failed_downloads.append({
                        'file_id': file_id,
                        'error': 'No faces detected',
                        'timestamp': pd.Timestamp.now().isoformat()
                    })
                    print(f"⚠️  No faces detected in {file_id}")
            else:
                failure_count += 1
                failed_downloads.append({
                    'file_id': file_id,
                    'error': 'Download failed',
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                print(f"❌ Failed to download {file_id}")
                
        except Exception as e:
            failure_count += 1
            error_msg = str(e)
            failed_downloads.append({
                'file_id': file_id,
                'error': error_msg,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            print(f"❌ Error processing {file_id}: {error_msg}")
    
    # Save failed downloads to CSV
    if failed_downloads:
        try:
            failed_df = pd.DataFrame(failed_downloads)
            failed_df.to_csv(error_log_file, index=False)
            print(f"\n📝 Saved {len(failed_downloads)} failed downloads to {error_log_file}")
        except Exception as e:
            print(f"⚠️  Failed to save error log: {str(e)}")
    
    # Print summary
    print("\n" + "="*50)
    print("📊 Processing Summary:")
    print(f"✅ Successfully processed: {success_count} files")
    print(f"❌ Failed to process: {failure_count} files")
    print(f"📋 Error log saved to: {error_log_file if failed_downloads else 'No errors to log'}")
    print("="*50 + "\n")
    
    if failure_count == 0:
        print("🎉 All files processed successfully!")
    else:
        print(f"⚠️  Completed with {failure_count} error(s). Check the error log for details.")


if __name__ == "__main__":
    process_csv()

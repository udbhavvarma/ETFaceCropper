import os
import uuid
import time
import random
import requests
import cv2
from PIL import Image, ImageDraw
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
                
                # Create a circular mask for the face
                try:
                    # Create a black image with the same dimensions as the cropped area
                    mask = Image.new('L', (x2-x1, y2-y1), 0)
                    
                    # Create a white circle on the mask
                    draw = ImageDraw.Draw(mask)
                    radius = min(x2-x1, y2-y1) // 2
                    center = ((x2-x1) // 2, (y2-y1) // 2)
                    draw.ellipse((center[0]-radius, center[1]-radius, 
                                 center[0]+radius, center[1]+radius), 
                                fill=255)
                    
                    # Crop and convert the original image to RGB
                    cropped = img[y1:y2, x1:x2]
                    if cropped.size == 0:
                        print(f"⚠️ Empty crop for face {i} in {os.path.basename(image_path)}")
                        continue
                        
                    # Convert to PIL Image and apply the circular mask
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    cropped_image = Image.fromarray(cropped_rgb)
                    
                    # Create a new image with transparent background
                    result = Image.new('RGBA', (x2-x1, y2-y1), (0, 0, 0, 0))
                    # Paste the cropped image using the mask
                    result.paste(cropped_image, (0, 0), mask=mask)
                    
                    # Generate filename using only the base name
                    out_name = f"{original_name}.png"
                    output_path = os.path.join(cropped_folder, out_name)
                    
                    # If file exists, skip instead of renaming to avoid duplicates
                    if os.path.exists(output_path):
                        print(f"⚠️ File {out_name} already exists, skipping...")
                        continue
                    
                    # Save with transparency
                    result.save(output_path, format='PNG', optimize=True)
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
    error_log_file="failed_downloads.csv",
    progress_file="progress.json",
    chunk_size=50,
    request_delay=1.0,
    max_retries=3
):
    """
    Full pipeline: initializes folders, loads IDs, downloads images, and crops faces.
    Handles large datasets with chunked processing and resume capability.
    
    Args:
        csv_path (str): Path to the CSV file containing file IDs
        download_folder (str): Directory to save downloaded images
        cropped_folder (str): Directory to save cropped faces
        error_log_file (str): Path to save the error log CSV
        progress_file (str): Path to save/load progress
        chunk_size (int): Number of files to process before saving progress
        request_delay (float): Delay between requests in seconds
        max_retries (int): Maximum number of retry attempts for failed downloads
    """
    import json
    from tqdm import tqdm
    import time
    from pathlib import Path
    
    # Initialize folders
    init_folders(download_folder, cropped_folder)
    
    # Load or initialize progress
    progress_path = Path(progress_file)
    if progress_path.exists():
        try:
            with open(progress_path, 'r') as f:
                progress = json.load(f)
            print(f"\n🔍 Resuming from previous progress (processed {progress.get('processed', 0)} files)")
        except Exception as e:
            print(f"⚠️  Could not load progress file: {e}. Starting fresh.")
            progress = {'processed': 0, 'failed_downloads': [], 'file_ids': []}
    else:
        progress = {'processed': 0, 'failed_downloads': [], 'file_ids': []}
    
    # Load file IDs and filter out already processed ones
    all_file_ids = load_file_ids(csv_path)
    
    # If we have existing progress, skip already processed files
    if progress['file_ids'] and len(progress['file_ids']) == len(all_file_ids):
        file_ids = all_file_ids[progress['processed']:]
    else:
        file_ids = all_file_ids
        progress = {'processed': 0, 'failed_downloads': [], 'file_ids': all_file_ids}
    
    total_files = len(file_ids)
    
    if total_files == 0:
        print("✅ No new files to process")
        if progress['failed_downloads']:
            print(f"⚠️  There were {len(progress['failed_downloads'])} failed downloads from previous runs")
        return
    
    # Initialize tracking
    stats = {
        'processed': progress['processed'],
        'success': 0,
        'failure': 0,
        'total_faces': 0,
        'start_time': time.time(),
        'last_save': time.time()
    }
    
    # Load existing failed downloads if any
    failed_downloads = progress.get('failed_downloads', [])
    
    print(f"\n🚀 Starting to process {total_files} files in chunks of {chunk_size}...")
    print("Press Ctrl+C to stop processing and save progress\n")
    
    def save_progress():
        """Save current progress to resume later"""
        progress = {
            'processed': stats['processed'],
            'file_ids': all_file_ids,
            'failed_downloads': failed_downloads,
            'success_count': stats['success'],
            'failure_count': stats['failure']
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    # Process files in chunks
    try:
        for chunk_start in range(0, len(file_ids), chunk_size):
            chunk = file_ids[chunk_start:chunk_start + chunk_size]
            chunk_processed = 0
            
            # Initialize progress bar for current chunk
            with tqdm(
                chunk,
                desc=f"Chunk {chunk_start//chunk_size + 1}/{(len(file_ids)-1)//chunk_size + 1}",
                unit="file",
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            ) as pbar:
                for file_id in pbar:
                    current_file = f"{file_id}"
                    pbar.set_postfix({
                        'Current': current_file[:20] + ('...' if len(current_file) > 20 else ''),
                        'Success': stats['success'],
                        'Failed': stats['failure'],
                        'Faces': stats['total_faces']
                    })
                    
                    # Process file with retries
                    for attempt in range(max_retries + 1):
                        try:
                            # Add delay between requests
                            if attempt > 0:
                                time.sleep(request_delay * (2 ** attempt))  # Exponential backoff
                                print(f"\n🔄 Retry {attempt + 1}/{max_retries} for {file_id}")
                            
                            local_path = download_file(file_id, download_folder)
                            if local_path:
                                cropped_images = crop_faces_mediapipe(
                                    local_path,
                                    cropped_folder=cropped_folder,
                                    base_name=file_id
                                )
                                if cropped_images:
                                    stats['success'] += 1
                                    stats['total_faces'] += len(cropped_images)
                                    chunk_processed += 1
                                    break  # Success, exit retry loop
                                else:
                                    if attempt == max_retries:  # Only fail after all retries
                                        stats['failure'] += 1
                                        failed_downloads.append({
                                            'file_id': file_id,
                                            'error': 'No faces detected',
                                            'timestamp': pd.Timestamp.now().isoformat(),
                                            'attempts': attempt + 1
                                        })
                            else:
                                if attempt == max_retries:
                                    stats['failure'] += 1
                                    failed_downloads.append({
                                        'file_id': file_id,
                                        'error': 'Download failed',
                                        'timestamp': pd.Timestamp.now().isoformat(),
                                        'attempts': attempt + 1
                                    })
                                    
                        except Exception as e:
                            if attempt == max_retries:  # Only fail after all retries
                                stats['failure'] += 1
                                failed_downloads.append({
                                    'file_id': file_id,
                                    'error': str(e),
                                    'timestamp': pd.Timestamp.now().isoformat(),
                                    'attempts': attempt + 1
                                })
                                print(f"\n❌ Failed after {max_retries} attempts: {file_id} - {str(e)}")
                            continue
                        else:
                            break  # Successfully processed, exit retry loop
                    
                    stats['processed'] += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Success': stats['success'],
                        'Failed': stats['failure'],
                        'Faces': stats['total_faces']
                    })
                    
                    # Small delay between files to be gentle on the server
                    time.sleep(request_delay)
            
            # Save progress after each chunk
            save_progress()
            
            # Print chunk summary
            print(f"\n📦 Chunk {chunk_start//chunk_size + 1} complete: "
                  f"{chunk_processed}/{len(chunk)} files processed successfully")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user. Saving progress...")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        print("Saving current progress...")
    
    # Calculate and display final statistics
    elapsed_time = time.time() - stats['start_time']
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*60)
    print("✅ Processing Summary")
    print("="*60)
    print(f"📊 Total Files: {total_files}")
    print(f"   ├── ✅ Success: {stats['success']} ({stats['success']/total_files*100:.1f}%)")
    print(f"   ├── ❌ Failed: {stats['failure']} ({stats['failure']/total_files*100:.1f}%)")
    print(f"   └── 😊 Faces Detected: {stats['total_faces']}")
    print(f"\n⏱️  Time Elapsed: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    if stats['success'] > 0:
        avg_time = elapsed_time / stats['success']
        print(f"⏳ Avg. time per file: {avg_time:.1f}s")
    print("="*60 + "\n")
    
    # Save final progress and failed downloads
    save_progress()
    
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

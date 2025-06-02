import streamlit as st
import os
import io
import zipfile
import tempfile
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from face_utils import init_folders, load_file_ids, download_file, crop_faces_mediapipe

# Initialize session state
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'download_ready' not in st.session_state:
    st.session_state.download_ready = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False


def zip_folder(folder_path):
    """Zip the contents of a folder in-memory and return a BytesIO buffer."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zf.write(file_path, arcname=file)
    buffer.seek(0)
    return buffer


def process_single_image(file_id: str, download_folder: str, cropped_folder: str) -> Dict:
    """Process a single image and return the result."""
    result = {
        'file_id': file_id,
        'status': 'processing',
        'message': '',
        'faces_detected': 0,
        'error': ''
    }
    
    try:
        # Download the file
        local_path = download_file(file_id, download_folder)
        if not local_path or not os.path.exists(local_path):
            result['status'] = 'failed'
            result['error'] = 'Download failed or file not found'
            return result
        
        # Process the image
        cropped_paths = crop_faces_mediapipe(
            local_path,
            cropped_folder=cropped_folder,
            base_name=file_id
        )
        
        if not cropped_paths:
            result['status'] = 'processed'
            result['message'] = 'No faces detected'
        else:
            result['status'] = 'processed'
            result['faces_detected'] = len(cropped_paths)
            result['message'] = f'Found {len(cropped_paths)} face(s)'
            
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result

def main():
    st.title("Face Cropper Service")
    st.markdown(
        "Upload a CSV containing a `file_id` column (Google Drive IDs). "
        "The app will download each image, detect & crop faces, and package the results for you."
    )

    # File uploader
    csv_uploader = st.file_uploader("Upload CSV", type=["csv"])
    
    # Reset state if new file is uploaded
    if csv_uploader and not st.session_state.processing_started:
        st.session_state.processing_done = False
        st.session_state.download_ready = False
        st.session_state.results = []
        st.session_state.current_index = 0
        st.session_state.processing_started = True
        st.session_state.file_ids = []
        st.session_state.tmpdir = None
    
    if not csv_uploader:
        return
    
    # Initialize temporary directory and load file IDs if not already done
    if not hasattr(st.session_state, 'tmpdir') or st.session_state.tmpdir is None:
        st.session_state.tmpdir = tempfile.TemporaryDirectory()
        tmpdir = st.session_state.tmpdir.name
        
        # Save uploaded CSV
        csv_path = os.path.join(tmpdir, "input.csv")
        with open(csv_path, "wb") as f:
            f.write(csv_uploader.getvalue())
        
        # Load file IDs
        st.session_state.file_ids = load_file_ids(csv_path)
        st.session_state.total_files = len(st.session_state.file_ids)
        
        if st.session_state.total_files == 0:
            st.error("No file IDs found in the CSV")
            return
    else:
        tmpdir = st.session_state.tmpdir.name
    
    # Initialize paths
    download_folder = os.path.join(tmpdir, "downloads")
    cropped_folder = os.path.join(tmpdir, "cropped_faces")
    
    # Initialize folders
    init_folders(download_folder, cropped_folder)
    
    # Create placeholders for UI elements
    progress_bar = st.progress(0)
    progress_container = st.container()
    
    # Initialize progress display
    with progress_container:
        st.subheader("Processing Progress")
        progress_cols = st.columns(3)
        with progress_cols[0]:
            processed_metric = st.metric("Processed", f"{len(st.session_state.results)}/{st.session_state.total_files}")
        with progress_cols[1]:
            success_count = len([r for r in st.session_state.results if r.get('status') == 'processed'])
            success_metric = st.metric("Success", str(success_count))
        with progress_cols[2]:
            failed_count = len([r for r in st.session_state.results if r.get('status') in ['failed', 'error']])
            failed_metric = st.metric("Failed", str(failed_count))
        
        st.subheader("Current Status")
        current_status = st.empty()
        
        st.subheader("Recent Activity")
        activity_log = st.empty()
    
    # Process files one at a time, continuing from where we left off
    if not st.session_state.processing_done and st.session_state.current_index < len(st.session_state.file_ids):
        file_id = st.session_state.file_ids[st.session_state.current_index]
        short_id = file_id[:15] + '...' if len(file_id) > 15 else file_id
        current_status.text(f'Processing: {short_id}')
        
        # Process the current image
        result = process_single_image(file_id, download_folder, cropped_folder)
        st.session_state.results.append(result)
        
        # Update progress
        progress = (st.session_state.current_index + 1) / st.session_state.total_files
        progress_bar.progress(progress)
        
        # Update metrics
        with progress_container:
            success_count = len([r for r in st.session_state.results if r.get('status') == 'processed'])
            failed_count = len([r for r in st.session_state.results if r.get('status') in ['failed', 'error']])
            
            processed_metric.metric("Processed", f"{len(st.session_state.results)}/{st.session_state.total_files}")
            success_metric.metric("Success", str(success_count))
            failed_metric.metric("Failed", str(failed_count))
            
            # Show recent activity
            recent_activity = []
            for r in st.session_state.results[-5:]:  # Show last 5 results
                status_icon = "✅" if r['status'] == 'processed' else "❌"
                recent_activity.append(f"{status_icon} {r['file_id'][:20]}...: {r.get('message', r.get('error', ''))}")
            
            activity_log.text("\n".join(recent_activity))
        
        # Move to next file
        st.session_state.current_index += 1
        
        # Check if we're done
        if st.session_state.current_index >= len(st.session_state.file_ids):
            st.session_state.processing_done = True
            st.session_state.download_ready = True
            current_status.text("Processing complete!")
            
            # Create zip file
            zip_buffer = zip_folder(cropped_folder)
            st.session_state.zip_buffer = zip_buffer
        else:
            # Force a rerun to process the next file
            st.rerun()
    
    # Show download button when processing is done
    if st.session_state.download_ready and 'zip_buffer' in st.session_state:
        st.success("Processing complete! Click below to download the results.")
        st.download_button(
            label="Download Cropped Faces",
            data=st.session_state.zip_buffer,
            file_name="cropped_faces.zip",
            mime="application/zip"
        )
        
        # Add a button to start a new session
        if st.button("Process another CSV"):
            st.session_state.clear()
            st.rerun()
            st.success(f"Found {len(cropped_files)} cropped faces.")
            
            # Show sample of cropped faces
            with st.expander("Show Sample Cropped Faces", expanded=True):
                cols = st.columns(4)
                for i, img_file in enumerate(cropped_files[:8]):  # Show up to 8 images
                    try:
                        img_path = os.path.join(cropped_folder, img_file)
                        img = Image.open(img_path)
                        cols[i % 4].image(img, width=150, caption=img_file[:15] + '...')
                    except Exception as e:
                        st.error(f"Error displaying {img_file}: {str(e)}")
            
            # Create zip buffer
            zip_buffer = zip_folder(cropped_folder)
            
            # Download button
            st.download_button(
                label='📥 Download All Cropped Faces',
                data=zip_buffer,
                file_name='cropped_faces.zip',
                mime='application/zip',
                use_container_width=True,
                help='Download all cropped faces as a ZIP file'
            )
        else:
            st.warning("No faces were detected in any of the processed images.")
        
        # Show error log if any
        if failed_count > 0:
            with st.expander("View Error Log", expanded=False):
                st.write("The following files had errors:")
                error_results = [r for r in results if r['status'] in ['failed', 'error']]
                st.table(pd.DataFrame(error_results)[['file_id', 'error']])

if __name__ == "__main__":
    # Add necessary imports at the top of the file
    from PIL import Image
    import pandas as pd
    
    # Set page config
    st.set_page_config(
        page_title="Face Cropper Service",
        page_icon="✂️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()
                        
            

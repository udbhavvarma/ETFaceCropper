import streamlit as st
import os
import io
import zipfile
import tempfile
import pandas as pd
from typing import List, Dict, Optional, Tuple

from face_utils import init_folders, load_file_ids, download_file, crop_faces_mediapipe


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


def process_images(csv_path: str, download_folder: str, cropped_folder: str, progress_bar=None) -> Tuple[List[Dict], int]:
    """Process all images from the CSV and return results summary."""
    import time
    from tqdm import tqdm
    
    results = []
    total_processed = 0
    
    try:
        ids = load_file_ids(csv_path)
        total_files = len(ids)
        
        # Initialize progress bar if not provided
        if progress_bar is None:
            progress_bar = tqdm(
                total=total_files,
                desc="Processing",
                unit="file",
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )
        
        for idx, file_id in enumerate(ids, start=1):
            result = {
                'file_id': file_id,
                'status': 'processing',
                'message': '',
                'faces_detected': 0,
                'error': ''
            }
            
            # Update progress bar
            progress_bar.set_postfix({
                'Current': file_id[:20] + ('...' if len(file_id) > 20 else ''),
                'Success': len([r for r in results if r.get('status') == 'processed']),
                'Failed': len([r for r in results if r.get('status') == 'failed'])
            })
            
            try:
                # Download the file
                local_path = download_file(file_id, download_folder)
                if not local_path or not os.path.exists(local_path):
                    result['status'] = 'failed'
                    result['error'] = 'Download failed or file not found'
                    results.append(result)
                    progress_bar.update(1)
                    continue
                
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
                    total_processed += 1
                
            except Exception as e:
                result['status'] = 'error'
                result['error'] = str(e)
                
            results.append(result)
            
    except Exception as e:
        return results, total_processed
    
    return results, total_processed

def main():
    st.title("Face Cropper Service")
    st.markdown(
        "Upload a CSV containing a `file_id` column (Google Drive IDs). "
        "The app will download each image, detect & crop faces, and package the results for you."
    )

    # File uploader
    csv_uploader = st.file_uploader("Upload CSV", type=["csv"])
    if not csv_uploader:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize paths
        csv_path = os.path.join(tmpdir, "input.csv")
        download_folder = os.path.join(tmpdir, "downloads")
        cropped_folder = os.path.join(tmpdir, "cropped_faces")
        
        # Save uploaded CSV
        with open(csv_path, "wb") as f:
            f.write(csv_uploader.getvalue())
        
        # Initialize folders
        init_folders(download_folder, cropped_folder)
        
        # Load file IDs
        file_ids = load_file_ids(csv_path)
        total_files = len(file_ids)
        
        if total_files == 0:
            st.error("No file IDs found in the CSV")
            return
        
        # Create placeholders for UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_placeholder = st.empty()
        
        # Initialize results
        results = []
        processed_count = 0
        success_count = 0
        failed_count = 0
        
        # Create containers for progress display
        progress_container = st.container()
        stats_container = st.container()
        
        # Initialize progress display
        with progress_container:
            st.subheader("Processing Progress")
            progress_cols = st.columns(3)
            with progress_cols[0]:
                processed_metric = st.metric("Processed", "0/0")
            with progress_cols[1]:
                success_metric = st.metric("Success", "0")
            with progress_cols[2]:
                failed_metric = st.metric("Failed", "0")
            
            st.subheader("Current Status")
            current_status = st.empty()
            
            st.subheader("Recent Activity")
            activity_log = st.empty()
        
        # Process each file
        for idx, file_id in enumerate(file_ids, 1):
            # Update progress
            progress = idx / total_files
            progress_bar.progress(progress)
            
            # Create result entry
            result = {
                'file_id': file_id,
                'status': 'processing',
                'message': '',
                'faces_detected': 0,
                'error': ''
            }
            
            # Update status
            short_id = file_id[:15] + '...' if len(file_id) > 15 else file_id
            current_status.text(f'Processing: {short_id}')
            
            # Update progress metrics
            with progress_container:
                processed_metric.metric("Processed", f"{processed_count}/{total_files}")
                success_metric.metric("Success", str(success_count))
                failed_metric.metric("Failed", str(failed_count))
                
                # Download the file
                local_path = download_file(file_id, download_folder)
                if not local_path or not os.path.exists(local_path):
                    result['status'] = 'failed'
                    result['error'] = 'Download failed or file not found'
                    failed_count += 1
                else:
                    # Process the image
                    try:
                        cropped_paths = crop_faces_mediapipe(
                            local_path,
                            cropped_folder=cropped_folder,
                            base_name=file_id
                        )
                        
                        if cropped_paths:
                            result['status'] = 'success'
                            result['message'] = f'Found {len(cropped_paths)} face(s)'
                            result['faces_detected'] = len(cropped_paths)
                            success_count += 1
                        else:
                            result['status'] = 'processed'
                            result['message'] = 'No faces detected'
                    except Exception as e:
                        result['status'] = 'error'
                        result['error'] = str(e)
                        failed_count += 1
                
                # Update processed count
                processed_count += 1
                results.append(result)
                
                # Update activity log
                with progress_container:
                    recent_results = results[-5:]  # Show last 5 results
                    activity_messages = []
                    for r in recent_results:
                        if r['status'] == 'success':
                            activity_messages.append(f"✅ {r['file_id'][:15]}... - {r['message']}")
                        elif r['status'] in ['failed', 'error']:
                            activity_messages.append(f"❌ {r['file_id'][:15]}... - {r.get('error', 'Unknown error')}")
                        else:
                            activity_messages.append(f"ℹ️ {r['file_id'][:15]}... - {r.get('message', 'Processed')}")
                    
                    # Display all activity messages in one go
                    activity_log.text("\n".join(activity_messages))
                

        
        # Processing complete
        progress_bar.empty()
        status_text.success("✅ Processing complete!")
        
        # Show summary
        st.balloons()
        st.success(f"✅ Processing complete! Processed {processed_count} files.")
        
        # Create zip of cropped faces if any were created
        cropped_files = [f for f in os.listdir(cropped_folder) if f.endswith('.png')]
        if cropped_files:
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
    main()
                        
            

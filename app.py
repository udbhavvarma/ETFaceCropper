import streamlit as st
import os
import io
import zipfile
import tempfile
import pandas as pd
import time
from typing import List, Dict, Optional, Tuple
from face_utils import init_folders, load_file_ids, download_file, crop_faces_mediapipe, TOTAL_FACES_CROPPED


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


def process_images(csv_path: str, download_folder: str, cropped_folder: str, circular_crop: bool = True) -> Tuple[List[Dict], int]:
    """Process all images from the CSV and return results summary."""
    results = []
    total_processed = 0
    
    try:
        ids = load_file_ids(csv_path)
        total_files = len(ids)
        
        # Create a status container
        status_placeholder = st.empty()
        
        for idx, file_id in enumerate(ids, start=1):
            result = {
                'file_id': file_id,
                'status': 'pending',
                'message': '',
                'faces_detected': 0,
                'error': ''
            }
            
            # Update status
            status_placeholder.markdown(
                f"""
                **Processing Progress**  
                - Files: {idx}/{total_files}  
                - Faces Cropped: {TOTAL_FACES_CROPPED}
                """
            )
            
            try:
                # Download the file
                local_path = download_file(file_id, download_folder)
                if not local_path or not os.path.exists(local_path):
                    result['status'] = 'failed'
                    result['error'] = 'Download failed or file not found'
                    results.append(result)
                    continue
                
                # Process the image with circular crop option
                cropped_paths = crop_faces_mediapipe(
                    local_path,
                    cropped_folder=cropped_folder,
                    base_name=file_id,
                    circular_crop=circular_crop
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

    # Add circular crop toggle
    circular_crop = st.sidebar.checkbox("Use Circular Crop", value=True, 
                                      help="Crop faces in a circular shape with transparent background")
    
    # Display current face count
    st.sidebar.metric("Total Faces Cropped", TOTAL_FACES_CROPPED)
    
    csv_uploader = st.file_uploader("Upload CSV", type=["csv"])
    if not csv_uploader:
        return

    with st.spinner("Processing..."):
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
            
            # Process all images
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process images and get results
            results, total_processed = process_images(
                csv_path, 
                download_folder, 
                cropped_folder,
                circular_crop=circular_crop
            )
            
            # Create results dataframe
            df_results = pd.DataFrame(results)
            
            # Show summary
            st.subheader("Processing Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Files", len(df_results))
            with col2:
                st.metric("Successfully Processed", total_processed)
            with col3:
                failed = len(df_results[df_results['status'] == 'failed'])
                st.metric("Failed", failed)
            
            # Show detailed results
            with st.expander("View Detailed Results"):
                st.dataframe(df_results[['file_id', 'status', 'message', 'faces_detected']])
            
            # Prepare download
            if total_processed > 0:
                with st.spinner("Preparing download..."):
                    zip_buffer = zip_folder(cropped_folder)
                    st.download_button(
                        label=f"Download {total_processed} Cropped Faces",
                        data=zip_buffer,
                        file_name="cropped_faces.zip",
                        mime="application/zip"
                    )
            else:
                st.warning("No faces were detected in any of the processed images.")
            
            # Show error log if any
            if not df_results[df_results['status'] == 'failed'].empty:
                with st.expander("View Error Log"):
                    st.write("The following files could not be processed:")
                    st.dataframe(df_results[df_results['status'] == 'failed'][['file_id', 'error']])



if __name__ == "__main__":
    main()

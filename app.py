import streamlit as st
import os
import io
import zipfile
import tempfile

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


def main():
    st.title("Face Cropper Service")
    st.markdown(
        "Upload a CSV containing a `file_id` column (Google Drive IDs)."  
        "For example, in the google drive link https://drive.google.com/open?id=1234567890, the file_id is 1234567890"
        "The app will download each image, detect & crop faces, and package the results for you."
    )

    csv_uploader = st.file_uploader("Upload CSV", type=["csv"])
    if csv_uploader:
        # Use a temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "input.csv")
            with open(csv_path, "wb") as f:
                f.write(csv_uploader.getvalue())

            download_folder = os.path.join(tmpdir, "downloads")
            cropped_folder = os.path.join(tmpdir, "cropped_faces")
            init_folders(download_folder, cropped_folder)

            ids = load_file_ids(csv_path)
            status = st.empty()
            for idx, file_id in enumerate(ids, start=1):
                status.text(f"({idx}/{len(ids)}) Downloading & cropping {file_id}")
                local_path = download_file(file_id, download_folder)
                crop_faces_mediapipe(
                    local_path,
                    cropped_folder=cropped_folder
                )

            status.text("All done! Preparing download...")
            zip_buffer = zip_folder(cropped_folder)
            st.download_button(
                label="Download All Cropped Faces",
                data=zip_buffer,
                file_name="cropped_faces.zip",
                mime="application/zip"
            )


if __name__ == "__main__":
    main()

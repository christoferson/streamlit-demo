import streamlit as st
import base64
import io
import pandas as pd
from PIL import Image

from cmn.view.mime_constants import mime_mapping_image, mime_mapping_document


def image_to_base64(image: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def process_uploaded_file(uploaded_file):
    """
    Read an uploaded Streamlit file and return
    (bytes, file_key, file_type, preview_widget_fn).

    preview_widget_fn is a zero-arg callable that renders a Streamlit preview,
    or None if no preview is needed.
    """
    if uploaded_file is None:
        return None, None, None, None

    file_type = uploaded_file.type
    file_key  = (uploaded_file.name
                 .replace(".", "_")
                 .replace(" ", "_"))

    if file_type in mime_mapping_image:
        raw   = uploaded_file.read()
        image = Image.open(io.BytesIO(raw))
        b64   = image_to_base64(image, mime_mapping_image[file_type].upper())
        preview = lambda: st.image(image, caption="Uploaded image", width="stretch")
        return raw, file_key, file_type, preview

    if file_type in mime_mapping_document:
        fmt = mime_mapping_document[file_type]

        if fmt == "csv":
            raw = base64.b64encode(uploaded_file.read())
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
                preview = lambda: st.dataframe(df)
            except Exception as e:
                preview = lambda: st.warning(f"CSV preview failed: {e}")
            return raw, file_key, file_type, preview

        if fmt == "pdf":
            raw = uploaded_file.read()
            preview = lambda: st.markdown(f"**{uploaded_file.name}**")
            return raw, file_key, file_type, preview

        if fmt in ("txt", "md"):
            raw = base64.b64encode(uploaded_file.read())
            preview = lambda: st.markdown(f"**{uploaded_file.name}**")
            return raw, file_key, file_type, preview

    st.warning(f"Unsupported file type: {file_type}")
    return None, None, None, None


def render_file_uploader(uploader_key: int):
    """
    Render the file uploader widget and process the uploaded file.

    Returns (uploaded_file, file_bytes, file_key, file_type, file_preview).
    Renders the preview immediately if one exists.
    """
    uploaded_file = st.file_uploader(
        "Attach file",
        type=["PNG", "JPEG", "TXT", "CSV", "PDF", "MD"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key=f"uploader_{uploader_key}",
    )

    file_bytes, file_key, file_type, file_preview = process_uploaded_file(uploaded_file)

    # Render preview immediately below the uploader
    if file_preview:
        file_preview()

    return uploaded_file, file_bytes, file_key, file_type, file_preview
import streamlit as st
import pandas as pd
import numpy as np
import base64

mime_mapping_document = {
    "application/vnd.ms-excel": "csv",
}

# #'pdf'|'csv'|'doc'|'docx'|'xls'|'xlsx'|'html'|'txt'|'md',
uploaded_file = st.file_uploader(
        "Attach Image",
        type=["CSV", "XLSX"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

if uploaded_file:
    if uploaded_file.type in mime_mapping_document:
        uploaded_file_key = uploaded_file.name.replace(".", "_").replace(" ", "_")
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        bedrock_file_type = mime_mapping_document[uploaded_file_type]
        print(f"-------{bedrock_file_type}")
        if "csv" == bedrock_file_type:
            uploaded_file_bytes = base64.b64encode(uploaded_file.read())
            uploaded_file.seek(0)
            try:
                uploaded_file_df = pd.read_csv(uploaded_file, encoding = "utf-8")
                st.write(uploaded_file_df)
            except Exception as err:
                st.chat_message("system").write(type(err).__name__)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    #if uploaded_file.name.endswidth(".csv"):
        #xx
    if uploaded_file.type in mime_mapping_document:
        uploaded_file_key = uploaded_file.name.replace(".", "_").replace(" ", "_")
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        bedrock_file_type = mime_mapping_document[uploaded_file_type]
        #print(f"-------{bedrock_file_type}")
        if "csv" == bedrock_file_type:
            try:
                uploaded_file_df = pd.read_csv(uploaded_file, encoding = "utf-8")
                st.write("Preview")
                st.dataframe(uploaded_file_df)
                st.write("Contents")
                st.write(uploaded_file_df)
                st.write("Describe")
                st.write(uploaded_file_df.describe())

                uploaded_file_columns = uploaded_file_df.columns.to_list()
                x_axis_selection = st.selectbox("X Axis", uploaded_file_columns)
                y_axis_selection = st.selectbox("Y Axis", uploaded_file_columns)

                pie_data = uploaded_file_df[x_axis_selection].value_counts()
                fig, ax = plt.subplots()
                ax.pie(pie_data, labels=pie_data.index, startangle=90)
                ax.axis = ('equal')
                st.pyplot(fig)

                #ax1 = uploaded_file_df.plot.scatter(
                #        x=x_axis_selection,
                #      y=y_axis_selection,
                #      c='DarkBlue')
                #st.pyplot(ax1)
            except Exception as err:
                print(err)
                st.chat_message("system").write(type(err).__name__)
import streamlit as st
import os
import boto3
import settings

#AWS_REGION = os.environ["AWS_REGION"]
AWS_REGION = settings.AWS_REGION

bedrock = boto3.client("bedrock", region_name=AWS_REGION)

response = bedrock.list_foundation_models(
    byOutputModality="TEXT"
)


prompt = st.chat_input(f"Say something AWS_REGION={settings.AWS_REGION}")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")

    with st.chat_message("user"):
        st.write(f"Input: {prompt}")

    model_ids = []
    for item in response["modelSummaries"]:
        model_ids.append(item['modelId'])
        print(item['modelId'])

    with st.chat_message("assistant"):
        st.write(f"Reply: {model_ids}")
import streamlit as st
import os
import boto3
import settings
import cmn_auth

###### AUTH START #####

if not cmn_auth.check_password():
   st.stop()

######  AUTH END #####

#AWS_REGION = os.environ["AWS_REGION"]
AWS_REGION = settings.AWS_REGION

bedrock = boto3.client("bedrock", region_name=AWS_REGION)

st.title("💬 Chatbot 1")

prompt = st.chat_input(f"Say something AWS_REGION={settings.AWS_REGION}")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")

    with st.chat_message("user"):
        st.write(f"Input: {prompt}")

    model_ids = []
    response = bedrock.list_foundation_models(byOutputModality="TEXT")
    for item in response["modelSummaries"]:
        model_ids.append(item['modelId'])
        print(item['modelId'])

    with st.chat_message("assistant"):
        st.write(f"Reply: {model_ids}")
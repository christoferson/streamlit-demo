import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
import pyperclip
import io
import base64
from PIL import Image

from botocore.exceptions import ClientError

AWS_REGION = cmn_settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

st.set_page_config(
    page_title="Image Query",
    page_icon="🧊",
    layout="wide", # "centered" or "wide"
    initial_sidebar_state="collapsed", #"auto", "expanded", or "collapsed"
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.markdown(
    """
    <style>
    button[kind="primary"] {
        background: none!important;
        border: none;
        padding: 0!important;
        margin: 0;
        color: black !important;
        text-decoration: none;
        cursor: pointer;
        border: none !important;
    }
    button[kind="primary"]:hover {
        text-decoration: none;
        color: black !important;
    }
    button[kind="primary"]:focus {
        outline: none !important;
        box-shadow: none !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def copy_button_clicked(text):
    pyperclip.copy(text)
    #st.session_state.button = not st.session_state.button

def image_to_base64(image,mime_type:str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

mime_mapping = {
        "image/png": "PNG",
        "image/jpeg": "JPEG"
    }

opt_model_id_list = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "meta.llama3-2-11b-instruct-v1:0", #OK
    "meta.llama3-2-90b-instruct-v1:0", #A client error occurred: Invocation of model ID meta.llama3-2-90b-instruct-v1:0 with on-demand throughput isn’t supported. Retry your request with the ID or ARN of an inference profile that contains this model.
    "us.amazon.nova-pro-v1:0",
]

mime_mapping_image = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

SYSTEM_PROMPT = """You are an advanced AI assistant specialized in image analysis and interpretation. Your capabilities include:

1. Detailed Image Description:
- Provide comprehensive descriptions of image content, including objects, people, scenes, colors, and compositions.
- Identify and describe key elements, focal points, and notable details within the image.

2. Visual Question Answering:
- Accurately answer specific questions about the contents, context, or details of the provided image.
- Provide clear, concise responses based solely on the visual information present in the image.

3. Text Extraction and OCR:
- Identify and extract any visible text within the image, including signs, labels, captions, or documents.
- Transcribe the extracted text accurately, maintaining original formatting where possible.

4. Object Detection and Recognition:
- Identify and list all significant objects, people, or elements present in the image.
- Provide approximate counts or quantities of repeated objects when relevant.

5. Scene Understanding and Context Analysis:
- Interpret the overall context, setting, or environment depicted in the image.
- Infer potential locations, time periods, or situations based on visual cues.

6. Emotional and Aesthetic Analysis:
- Describe the mood, atmosphere, or emotional tone conveyed by the image.
- Comment on artistic elements, composition, or photographic techniques if applicable.

7. Comparative Analysis:
- When multiple images are provided, compare and contrast their contents, highlighting similarities and differences.

Guidelines for your responses:
- Provide clear, detailed, and objective descriptions or answers.
- Use precise language and technical terms when appropriate, but explain them if they might be unfamiliar.
- If any part of the image or query is unclear, state your uncertainty and provide the best possible interpretation.
- Respect privacy by avoiding identification of specific individuals unless explicitly requested.
- When extracting text, clearly differentiate between direct quotes and paraphrased content.
- If asked about elements not present in the image, clearly state that they are not visible or detectable.

Always base your responses solely on the content of the provided image(s) and the specific query. If additional context or clarification is needed, please ask for it before proceeding with your analysis."""

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value=SYSTEM_PROMPT, key="system_msg")

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_runtime_us_west_2 = boto3.client('bedrock-runtime', region_name="us-west-2")
rekognition = boto3.client('rekognition', region_name=AWS_REGION)

st.title("💬 Image Query 8.1.6")


col1, col2 = st.columns([2, 1])

with col2:

    uploaded_file = st.file_uploader(
        "Image Upload",
        type=["PNG", "JPEG"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    uploaded_file_name = None
    if uploaded_file:
        uploaded_file_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        #uploaded_file_base64 = image_to_base64(image, mime_mapping[uploaded_file_type])
        st.image(
            image, caption='upload images',
            use_column_width=True
        )
        print(uploaded_file_type)


        # response = rekognition.detect_labels(
        #     Image={'Bytes': uploaded_file_bytes},
        #     #MaxLabels=123,
        #     #MinConfidence=...,
        #     Features=[
        #         'IMAGE_PROPERTIES',
        #     ],
        #     Settings={
        #         'ImageProperties': {
        #             'MaxDominantColors': 5
        #         }
        #     }
        # )

        # print(response)
        # st.write(response)

        # img_properties = response['ImageProperties']
        # fg_dominant_colors = img_properties['Foreground']['DominantColors']
        # st.write(fg_dominant_colors)

# Add a button to trigger Rekognition analysis
        if st.button("Analyze Image"):
            response = rekognition.detect_labels(
                Image={'Bytes': uploaded_file_bytes},
                Features=['IMAGE_PROPERTIES'],
                Settings={
                    'ImageProperties': {
                        'MaxDominantColors': 5
                    }
                }
            )

            print(response)
            st.write(response)

            img_properties = response['ImageProperties']
            fg_dominant_colors = img_properties['Foreground']['DominantColors']
            st.write(fg_dominant_colors)

        uploaded_file_base64 = image_to_base64(image, mime_mapping[uploaded_file_type])

######


with col1:

    if "menu_image_query_messages" not in st.session_state:
        st.session_state["menu_image_query_messages"] = []

    idx = 1
    for msg in st.session_state.menu_image_query_messages:
        idx = idx + 1
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], list):
                # For messages with list content (like those with images)
                for content_item in msg["content"]:
                    if isinstance(content_item, dict) and "text" in content_item:
                        st.write(content_item["text"])
            else:
                # For simple text messages
                st.write(msg["content"])

    if prompt := st.chat_input():

        message_history = st.session_state.menu_image_query_messages.copy()

        message_user_latest = {"role": "user", "content": [{ "text": prompt }]}
        if uploaded_file_name:
            content = message_user_latest['content']
            if uploaded_file_type in mime_mapping_image:
                content.append(
                    {
                        "image": {
                            "format": mime_mapping_image[uploaded_file_type],
                            "source": {
                                "bytes": uploaded_file_bytes, # If the image dimension is not supported we will get validation error
                            }
                        },
                    }
                )
            else:
                st.write(f"Not supported file type: {uploaded_file.type}")
        message_history.append(message_user_latest)

        st.chat_message("user").write(prompt)

        system_prompts = [{"text" : opt_system_msg}]
    
        inference_config = {
            "temperature": opt_temperature,
            "maxTokens": opt_max_tokens,
            "topP": opt_top_p,
            #stopSequences 
        }

        additional_model_fields = {}
        

        with st.spinner('Processing...'):

            try:
                
                if "anthropic.claude-3-5-sonnet-20241022-v2:0" == opt_model_id or "us.anthropic.claude-3-5-sonnet-20241022-v2:0" == opt_model_id:
                    response = bedrock_runtime_us_west_2.converse_stream(
                        modelId=opt_model_id,
                        messages=message_history,
                        system=system_prompts,
                        inferenceConfig=inference_config,
                        additionalModelRequestFields=additional_model_fields
                    )
                else:
                    response = bedrock_runtime.converse_stream(
                        modelId=opt_model_id,
                        messages=message_history,
                        system=system_prompts,
                        inferenceConfig=inference_config,
                        additionalModelRequestFields=additional_model_fields
                    )
                

                #with st.chat_message("assistant", avatar=setAvatar("assistant")):
                result_text = ""
                with st.chat_message("assistant"):
                    result_container = st.container(border=True)
                    result_area = st.empty()
                    stream = response.get('stream')
                    for event in stream:
                        
                        if 'messageStart' in event:
                            #opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens} role= {event['messageStart']['role']}"
                            #result_container.write(opts)                    
                            pass

                        if 'contentBlockDelta' in event:
                            text = event['contentBlockDelta']['delta']['text']
                            result_text += f"{text}"
                            result_area.write(result_text)

                        if 'messageStop' in event:
                            #'stopReason': 'end_turn'|'tool_use'|'max_tokens'|'stop_sequence'|'content_filtered'
                            stop_reason = event['messageStop']['stopReason']
                            if stop_reason == 'end_turn':
                                pass
                            else:
                                stop_reason_display = stop_reason
                                if stop_reason == 'max_tokens':
                                    stop_reason_display = "Insufficient Tokens. Increaes MaxToken Settings."
                                result_text_error = f"{result_text}\n\n:red[Generation Stopped: {stop_reason_display}]"
                                result_area.write(result_text_error)

                        if 'metadata' in event:
                            metadata = event['metadata']
                            if 'usage' in metadata:
                                input_token_count = metadata['usage']['inputTokens']
                                output_token_count = metadata['usage']['outputTokens']
                                total_token_count = metadata['usage']['totalTokens']
                            if 'metrics' in event['metadata']:
                                latency = metadata['metrics']['latencyMs']
                            #stats = f"| token.in={input_token_count} token.out={output_token_count} token={total_token_count} latency={latency} provider={opt_fm.provider}"
                            stats = f"| token.in={input_token_count} token.out={output_token_count} token={total_token_count} latency={latency} "
                            result_container.write(stats)

                        if "internalServerException" in event:
                            exception = event["internalServerException"]
                            result_text += f"\n\{exception}"
                            result_area.write(result_text)
                        if "modelStreamErrorException" in event:
                            exception = event["modelStreamErrorException"]
                            result_text += f"\n\{exception}"
                            result_area.write(result_text)
                        if "throttlingException" in event:
                            exception = event["throttlingException"]
                            result_text += f"\n\{exception}"
                            result_area.write(result_text)
                        if "validationException" in event:
                            exception = event["validationException"]
                            result_text += f"\n\{exception}"
                            result_area.write(result_text)

                #st.session_state.menu_image_query_messages.append({"role": "user", "content": prompt})
                #st.session_state.menu_image_query_messages.append({"role": "assistant", "content": result_text})

                # When storing messages in session state, modify the format:
                st.session_state.menu_image_query_messages.append({"role": "user", "content": [{"text": prompt}]})
                st.session_state.menu_image_query_messages.append({"role": "assistant", "content": [{"text": result_text}]})
            
            except ClientError as err:
                message = err.response["Error"]["Message"]
                logger.error("A client error occurred: %s", message)
                st.error(f"A client error occurred: {message}")

            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}"
                logger.error(error_message)
                st.error(error_message)
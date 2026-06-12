import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
#import pyperclip
import io
import base64
from PIL import Image
from cmn.bedrock_models import FoundationModel

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
    #pyperclip.copy(text)
    #st.session_state.button = not st.session_state.button
    pass

def image_to_base64(image,mime_type:str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

mime_mapping = {
        "image/png": "PNG",
        "image/jpeg": "JPEG"
    }

opt_model_id_list = [
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    # "anthropic.claude-3-5-sonnet-20241022-v2:0",
    # "anthropic.claude-3-5-sonnet-20240620-v1:0",
    # "anthropic.claude-3-sonnet-20240229-v1:0",
    # "anthropic.claude-3-haiku-20240307-v1:0",
    # "meta.llama3-2-11b-instruct-v1:0", #OK
    # "meta.llama3-2-90b-instruct-v1:0", #A client error occurred: Invocation of model ID meta.llama3-2-90b-instruct-v1:0 with on-demand throughput isn’t supported. Retry your request with the ID or ARN of an inference profile that contains this model.
    # "us.amazon.nova-pro-v1:0",
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

opt_fm: FoundationModel = FoundationModel.find(opt_model_id)

opt_fm_max_tokens = opt_fm.InferenceParameter.get("MaxTokensToSample")
opt_fm_temperature = opt_fm.InferenceParameter.get("Temperature")
opt_fm_top_p = opt_fm.InferenceParameter.get("TopP")
opt_fm_top_k = opt_fm.InferenceParameter.get("TopK")

with st.sidebar:
    if opt_fm_temperature.isSupported():
        opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    else:
        opt_temperature = 0.0
    if opt_fm_top_p.isSupported():
        opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    else:
        opt_top_p = 0.0
    if opt_fm_top_k.isSupported():
        opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    else:
        opt_top_k = 0
    opt_max_tokens = st.slider(label="Max Tokens", min_value=opt_fm_max_tokens.MinValue, max_value=opt_fm_max_tokens.MaxValue, value=opt_fm_max_tokens.DefaultValue, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value=SYSTEM_PROMPT, key="system_msg")

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_runtime_us_west_2 = boto3.client('bedrock-runtime', region_name="us-west-2")
rekognition = boto3.client('rekognition', region_name=AWS_REGION)

st.markdown("💬 :blue[Image Query 8.1.6]")

# Create main layout with right sidebar
main_col, right_sidebar_col = st.columns([3, 1])

# Right sidebar for image upload
with right_sidebar_col:
    st.markdown("##### Image")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["PNG", "JPEG"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    uploaded_file_name = None
    actual_image_format = None
    uploaded_file_bytes = None
    uploaded_file_type = None
    uploaded_file_base64 = None

    if uploaded_file:
        uploaded_file_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        # Detect actual image format from PIL
        actual_image_format = image.format.lower() if image.format else None
        st.image(
            image, caption='Uploaded Image',
            use_container_width=True,
        )
        print(f"Uploaded MIME type: {uploaded_file_type}, Actual format: {actual_image_format}")

        # Base64 conversion (currently unused but kept for potential future use)
        if uploaded_file_type in mime_mapping:
            uploaded_file_base64 = image_to_base64(image, mime_mapping[uploaded_file_type])

    # Rekognition analysis section
    if uploaded_file:
    
        st.markdown("**Image Analysis**")
        if st.button("Analyze with Rekognition", use_container_width=True):
            with st.spinner("Analyzing..."):
                response = rekognition.detect_labels(
                    Image={'Bytes': uploaded_file_bytes},
                    Features=['IMAGE_PROPERTIES'],
                    Settings={
                        'ImageProperties': {
                            'MaxDominantColors': 5
                        }
                    }
                )

                st.success("Analysis complete!")
                with st.expander("View Details", expanded=False):
                    st.write(response)

                    if 'ImageProperties' in response:
                        img_properties = response['ImageProperties']
                        if 'Foreground' in img_properties:
                            fg_dominant_colors = img_properties['Foreground']['DominantColors']
                            st.markdown("**Dominant Colors:**")
                            st.write(fg_dominant_colors)

    # Clear conversation history button
    st.divider()
    st.markdown("##### Conversation")
    message_count = len(st.session_state.get("menu_image_query_messages", []))
    st.caption(f"{message_count} messages")

    if st.button(":material/delete_history: Clear History",
                    use_container_width=True,
                    type="secondary",
                    disabled=(message_count == 0)):
        st.session_state.menu_image_query_messages = []
        st.rerun()

# Initialize session state
if "menu_image_query_messages" not in st.session_state:
    st.session_state["menu_image_query_messages"] = []

# Main chat area - display messages in a container
with main_col:
    # Container for chat history
    chat_container = st.container(height=400)

    with chat_container:
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

    # Chat input at the bottom of the main column
    prompt = st.chat_input()

# Process the prompt if provided
if prompt:
    message_history = st.session_state.menu_image_query_messages.copy()

    message_user_latest = {"role": "user", "content": [{ "text": prompt }]}
    if uploaded_file_name and actual_image_format:
        content = message_user_latest['content']
        # Use actual detected format instead of MIME type
        if actual_image_format in ["png", "jpeg", "jpg", "gif", "webp"]:
            # Normalize jpg to jpeg for Bedrock
            bedrock_format = "jpeg" if actual_image_format == "jpg" else actual_image_format
            content.append(
                {
                    "image": {
                        "format": bedrock_format,
                        "source": {
                            "bytes": uploaded_file_bytes,
                        }
                    },
                }
            )
        else:
            st.write(f"Not supported image format: {actual_image_format}")
    message_history.append(message_user_latest)

    # Display user message in the chat container
    with main_col:
        with chat_container:
            st.chat_message("user").write(prompt)

    system_prompts = [{"text" : opt_system_msg}]

    inference_config = {
        "maxTokens": opt_max_tokens,
        #stopSequences
    }

    if opt_fm_temperature.isSupported():
        inference_config["temperature"] = opt_temperature

    if opt_fm_top_p.isSupported():
        inference_config["topP"] = opt_top_p

    additional_model_fields = {}

    if opt_fm_top_k.isSupported():
        additional_model_fields[opt_fm_top_k.Name] = opt_top_k

    # If additional_model_fields is an empty dictionary, set it to None
    if additional_model_fields == {}:
        additional_model_fields = None

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

            # Display streaming response in the chat container
            result_text = ""
            with main_col:
                with chat_container:
                    with st.chat_message("assistant"):
                        result_container = st.container(border=True)
                        result_area = st.empty()
                        stream = response.get('stream')
                        for event in stream:

                            if 'messageStart' in event:
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

            # When storing messages in session state, modify the format:
            st.session_state.menu_image_query_messages.append({"role": "user", "content": [{"text": prompt}]})
            st.session_state.menu_image_query_messages.append({"role": "assistant", "content": [{"text": result_text}]})
            st.rerun()

        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            st.error(f"A client error occurred: {message}")

        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            logger.error(error_message)
            st.error(error_message)
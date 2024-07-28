import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
import io
import base64
from PIL import Image
import cv2
import numpy as np
from collections import Counter


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

rekognition = boto3.client('rekognition', region_name=AWS_REGION)


def get_top_colors(image, n=3):
    # Convert the image to a flattened 1D array of RGB values
    pixels = np.float32(image).reshape((-1, 3))

    # Create a dictionary to store the count of each RGB value
    color_counts = Counter(map(tuple, pixels))

    # Sort the dictionary by value (count) in descending order
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

    # Get the top n colors
    top_colors = sorted_colors[:n]

    return top_colors

def image_to_base64(image,mime_type:str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

mime_mapping = {
        "image/png": "PNG",
        "image/jpeg": "JPEG"
    }

opt_model_id_list = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

st.title("💬 Image Color Compare")


col1, col2, col3 = st.columns([3, 2, 2])

######

with col2:

    uploaded_file  = st.file_uploader(
        "Image File - 1",
        type=["PNG", "JPEG"],
        accept_multiple_files=False,
        #label_visibility="collapsed",
    )

    uploaded_file_name = None
    uploaded_file_bytes = None
    if uploaded_file:
        uploaded_file_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        uploaded_file_base64 = image_to_base64(image, mime_mapping[uploaded_file_type])
        st.image(
            image, caption='upload images',
            use_column_width=True
        )

    if uploaded_file_bytes and uploaded_file_bytes != None:
        
        uploaded_file_fetch_image_properties = st.checkbox("Get Image Properties", key="uploaded_file_fetch_image_properties")

        if uploaded_file_fetch_image_properties:
            response = rekognition.detect_labels(
                Image={'Bytes': uploaded_file_bytes},
                #MaxLabels=123,
                #MinConfidence=...,
                Features=[
                    'IMAGE_PROPERTIES',
                ],
                Settings={
                    'ImageProperties': {
                        'MaxDominantColors': 5
                    }
                }
            )
            img_properties = response['ImageProperties']
            fg_dominant_colors = img_properties['Foreground']['DominantColors']
            #st.write(fg_dominant_colors)

            for fg_dominant_color in fg_dominant_colors:
                fg_dc_red = fg_dominant_color['Red']
                fg_dc_blue = fg_dominant_color['Blue']
                fg_dc_green = fg_dominant_color['Green']
                fg_dc_hex = fg_dominant_color['HexCode']
                vfg_dc_css_color = fg_dominant_color['CSSColor']
                vfg_dc_simple_color = fg_dominant_color['SimplifiedColor']
                vfg_dc_pixel_percent = fg_dominant_color['PixelPercent']
                st.write(f"RGB {fg_dc_red} {fg_dc_blue} {fg_dc_green} Hex {fg_dc_hex} Color {vfg_dc_css_color} {vfg_dc_pixel_percent}")

    #print(cv2.__version__)

    if uploaded_file_bytes and uploaded_file_bytes != None:

        uploaded_file_fetch_opencv = st.checkbox("Get Image Properties", key="uploaded_file_fetch_opencv")

        if uploaded_file_fetch_opencv:
            # Read the uploaded image using OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file_bytes), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Check if the image was read successfully
            if image is None:
                st.error("Error: Could not read the uploaded image file.")
            else:
                # Get the top 3 dominant colors
                top_colors = get_top_colors(image, n=3)

                # Print the top 3 dominant colors
                if top_colors:
                    st.write("Top 3 dominant RGB colors:")
                    for i, (color, count) in enumerate(top_colors, start=1):
                        st.write(f"{i}. RGB({color[0]}, {color[1]}, {color[2]}): {count}")
                else:
                    st.warning("No dominant colors found in the image.")

######

with col3:

    uploaded_file_2 = st.file_uploader(
        "Image File - 2",
        type=["PNG", "JPEG"],
        accept_multiple_files=False,
        #label_visibility="collapsed",
    )

    uploaded_file_2_name = None
    uploaded_file_2_bytes = None
    if uploaded_file_2:
        uploaded_file_2_bytes = uploaded_file_2.getvalue()
        image_2 = Image.open(uploaded_file_2)
        uploaded_file_2_name = uploaded_file_2.name
        uploaded_file_2_type = uploaded_file_2.type
        uploaded_file_2_base64 = image_to_base64(image_2, mime_mapping[uploaded_file_2_type])
        st.image(
            image_2, caption='upload images',
            use_column_width=True
        )


    if uploaded_file_2_bytes and uploaded_file_2_bytes != None:
        
        uploaded_file_2_fetch_image_properties = st.checkbox("Get Image Properties", key="uploaded_file_2_fetch_image_properties")

        if uploaded_file_2_fetch_image_properties:
            response = rekognition.detect_labels(
                Image={'Bytes': uploaded_file_2_bytes},
                #MaxLabels=123,
                #MinConfidence=...,
                Features=[
                    'IMAGE_PROPERTIES',
                ],
                Settings={
                    'ImageProperties': {
                        'MaxDominantColors': 5
                    }
                }
            )
            img_properties = response['ImageProperties']
            fg_dominant_colors = img_properties['Foreground']['DominantColors']
            #st.write(fg_dominant_colors)

            for fg_dominant_color in fg_dominant_colors:
                fg_dc_red = fg_dominant_color['Red']
                fg_dc_blue = fg_dominant_color['Blue']
                fg_dc_green = fg_dominant_color['Green']
                fg_dc_hex = fg_dominant_color['HexCode']
                vfg_dc_css_color = fg_dominant_color['CSSColor']
                vfg_dc_simple_color = fg_dominant_color['SimplifiedColor']
                vfg_dc_pixel_percent = fg_dominant_color['PixelPercent']
                st.write(f"RGB {fg_dc_red} {fg_dc_blue} {fg_dc_green} Hex {fg_dc_hex} Color {vfg_dc_css_color} {vfg_dc_pixel_percent}")


    #####

with col1:

    if "menu_image_query_messages" not in st.session_state:
        st.session_state["menu_image_query_messages"] = []

    idx = 1
    for msg in st.session_state.menu_image_query_messages:
        idx = idx + 1
        content = msg["content"]
        with st.chat_message(msg["role"]):
            st.write(content)
            if "assistant" == msg["role"]:
                #assistant_cmd_panel_col1, assistant_cmd_panel_col2, assistant_cmd_panel_col3 = st.columns([0.07,0.23,0.7], gap="small")
                #with assistant_cmd_panel_col2:
                #st.button(key=f"copy_button_{idx}", label='📄', type='primary', on_click=copy_button_clicked, args=[content])
                pass


    if prompt := st.chat_input():

        message_history = st.session_state.menu_image_query_messages.copy()
        content =  [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        }
                    ]

        if uploaded_file_name:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": uploaded_file_type,
                        "data": uploaded_file_base64,
                    },
                }
            )

        if uploaded_file_2_name:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": uploaded_file_2_type,
                        "data": uploaded_file_2_base64,
                    },
                }
            )

        message_history.append({"role": "user", "content": content})
        #st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        #user_message =  {"role": "user", "content": f"{prompt}"}
        #messages = [st.session_state.messages]
        print(f"messages={st.session_state.menu_image_query_messages}")

        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": opt_temperature,
            "top_p": opt_top_p,
            "top_k": opt_top_k,
            "max_tokens": opt_max_tokens,
            "system": opt_system_msg,
            "messages": message_history #st.session_state.messages
        }
        json.dumps(request, indent=3)

        try:
            #bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            response = bedrock_runtime.invoke_model_with_response_stream(
                modelId = opt_model_id, #bedrock_model_id, 
                contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
                accept = "application/json",
                body = json.dumps(request))

            #with st.chat_message("assistant", avatar=setAvatar("assistant")):
            result_text = ""
            with st.chat_message("assistant"):
                result_container = st.container(border=True)
                result_area = st.empty()
                stream = response["body"]
                for event in stream:
                    
                    if event["chunk"]:

                        chunk = json.loads(event["chunk"]["bytes"])

                        if chunk['type'] == 'message_start':
                            opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens}"
                            #result_text += f"{opts}\n\n"
                            #result_area.write(result_text)
                            #result_container.write(opts)
                            #pass

                        elif chunk['type'] == 'message_delta':
                            #print(f"\nStop reason: {chunk['delta']['stop_reason']}")
                            #print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
                            #print(f"Output tokens: {chunk['usage']['output_tokens']}")
                            pass

                        elif chunk['type'] == 'content_block_delta':
                            if chunk['delta']['type'] == 'text_delta':
                                text = chunk['delta']['text']
                                #await msg.stream_token(f"{text}")
                                result_text += f"{text}"
                                result_area.write(result_text)

                        elif chunk['type'] == 'message_stop':
                            invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                            input_token_count = invocation_metrics["inputTokenCount"]
                            output_token_count = invocation_metrics["outputTokenCount"]
                            latency = invocation_metrics["invocationLatency"]
                            lag = invocation_metrics["firstByteLatency"]
                            stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                            #result_container.write(stats)

                            invocation_metrics = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                            #result_container.markdown(f""":blue[{invocation_metrics}]""")
                            #result_area.markdown(f"{invocation_metrics} {result_text} ")
                            result_text_final = f"""{result_text}  \n\n:blue[{invocation_metrics}]"""
                            #result_text += f"{reference_chunk_list_text}"
                            #result_area.write(f"{result_text_final}")
                            #result_container.markdown()
                            result_area.write(f"{result_text_final}")

                    elif "internalServerException" in event:
                        exception = event["internalServerException"]
                        result_text += f"\n\{exception}"
                        result_area.write(result_text)
                    elif "modelStreamErrorException" in event:
                        exception = event["modelStreamErrorException"]
                        result_text += f"\n\{exception}"
                        result_area.write(result_text)
                    elif "modelTimeoutException" in event:
                        exception = event["modelTimeoutException"]
                        result_text += f"\n\{exception}"
                        result_area.write(result_text)
                    elif "throttlingException" in event:
                        exception = event["throttlingException"]
                        result_text += f"\n\{exception}"
                        result_area.write(result_text)
                    elif "validationException" in event:
                        exception = event["validationException"]
                        result_text += f"\n\{exception}"
                        result_area.write(result_text)
                    else:
                        result_text += f"\n\nUnknown Token"
                        result_area.write(result_text)

                #st.button(key='copy_button', label='📄', type='primary', on_click=copy_button_clicked, args=[result_text])
                

            st.session_state.menu_image_query_messages.append({"role": "user", "content": prompt})
            st.session_state.menu_image_query_messages.append({"role": "assistant", "content": result_text})

            
            
        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            print("A client error occured: " + format(message))
            st.chat_message("system").write(message)

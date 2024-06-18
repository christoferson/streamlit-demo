import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
import base64
import io
from PIL import Image
import os
from io import BytesIO
import sys
import subprocess
from contextlib import closing
from tempfile import gettempdir
import random

from pydub import AudioSegment
from pydub.playback import play

from botocore.exceptions import BotoCoreError, ClientError

AWS_REGION = cmn_settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

####################################################################################

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
polly = boto3.client("polly", region_name=AWS_REGION)

####################################################################################


### Utilities

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_image(base64_str) -> Image:
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))

#####################

st.set_page_config(
    page_title="Image Generator",
    page_icon="🧊",
    layout="centered", # "centered" or "wide"
    initial_sidebar_state="expanded", #"auto", "expanded", or "collapsed"
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.logo(icon_image="images/logo.png", image="images/logo_text.png")

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

# Play Audio Button stAudio
st.markdown(
    """
    <style>
    #.stAudio {
    #    max-width: 70px;
    #    max-height: 50px;
    #}
    #audio::-webkit-media-controls-time-remaining-display,
    #audio::-webkit-media-controls-current-time-display {
    #    max-width: 50%;
    #    max-height: 20px;
    #}
    </style>
    """,
    unsafe_allow_html=True,
)


opt_model_id_list = [
    "stability.stable-diffusion-xl-v1"
]

opt_style_preset_list = [
    "anime", 
    "photographic"
]

opt_negative_prompt_list = [
    "ugly", "tiling", "out of frame",
    "disfigured", "deformed", "bad anatomy", "cut off", "low contrast", 
    "underexposed", "overexposed", "bad art", "beginner", "amateur", "blurry", "draft", "grainy"
]

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_config_scale = st.slider(label="Config Scale", min_value=0, max_value=35, value=10, step=1, key="config_scale")
    opt_steps = st.slider(label="Steps", min_value=10, max_value=50, value=30, step=1, key="steps")
    opt_style_preset = st.selectbox(label="Style Presets", options=opt_style_preset_list, index = 0, key="style_preset")
    opt_negative_prompt = st.multiselect(label="Negative Prompt", options=opt_negative_prompt_list, default=opt_negative_prompt_list, key="negative_prompt")
    opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")



st.title("💬 Image Generator")
st.write("Text to Image")

if "menu_img_gen_messages" not in st.session_state:
    st.session_state["menu_img_gen_messages"] = [
        #{"role": "user", "content": "Hello there."},
        #{"role": "assistant", "content": "How can I help you?"}
    ]

#if "audio_stream" not in st.session_state:
#    st.session_state["audio_stream"] = ""

idx = 1
for msg in st.session_state.menu_img_gen_messages:
    idx = idx + 1
    content = msg["content"]
    with st.chat_message(msg["role"]):
        if "user" == msg["role"]:
            st.write(content)
        if "assistant" == msg["role"]:
            #assistant_cmd_panel_col1, assistant_cmd_panel_col2, assistant_cmd_panel_col3 = st.columns([0.07,0.23,0.7], gap="small")
            #with assistant_cmd_panel_col2:
            #st.button(key=f"copy_button_{idx}", label='📄', type='primary', on_click=copy_button_clicked, args=[content])
            st.image(content)


if prompt := st.chat_input():

    #st.session_state["audio_stream"] = ""

    message_history = st.session_state.menu_img_gen_messages.copy()
    message_history.append({"role": "user", "content": prompt})
    #st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    #user_message =  {"role": "user", "content": f"{prompt}"}
    #messages = [st.session_state.messages]
    #print(f"messages={st.session_state.messages}")

    request = {
            "text_prompts": (
                [{"text": prompt, "weight": 1.0}]
                + [{"text": negprompt, "weight": -1.0} for negprompt in opt_negative_prompt]
            ),
            "cfg_scale": opt_config_scale,
            #"clip_guidance_preset"
            #"height": "1024",
            #"width": "1024",
            "seed": random.randint(0, 4294967295), # The seed determines the initial noise setting.0-4294967295,0
            #"start_schedule": config["start_schedule"],
            "steps": opt_steps, # Generation step determines how many times the image is sampled. 10-50,50
            "style_preset": opt_style_preset,
            "samples": 1,
        }
    json.dumps(request, indent=3)

    try:
        #bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        response = bedrock_runtime.invoke_model(
            modelId = opt_model_id, #bedrock_model_id, 
            contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
            accept = "application/json",
            body = json.dumps(request))
        
        response_body = json.loads(response.get("body").read())
        response_image_base64 = response_body["artifacts"][0].get("base64")
        response_image:Image = base64_to_image(response_image_base64)

        #file_extension = ".png"
        #OUTPUT_IMG_PATH = os.path.join("./output/{}-{}{}".format("img", idx, file_extension))
        #print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)
        #response_image.save(OUTPUT_IMG_PATH)

        with st.chat_message("assistant"):
            st.image(response_image)
            
        st.session_state.menu_img_gen_messages.append({"role": "user", "content": prompt})
        st.session_state.menu_img_gen_messages.append({"role": "assistant", "content": response_image})

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)


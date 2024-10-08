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
import random
from datetime import datetime

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
    page_icon="🖌️",
    layout="wide", #"centered", # "centered" or "wide"
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


###########################

def update_menu_img_gen_splash():
    if st.session_state["menu_img_gen_splash"] == True:
        st.session_state["menu_img_gen_splash"] = False
    else:
        st.session_state["menu_img_gen_splash"] = True

if "menu_img_gen_splash" not in st.session_state:
    st.session_state["menu_img_gen_splash"] = True

if st.session_state["menu_img_gen_splash"] == True:
    st.markdown("Sample Prompt 1")
    st.markdown("Sample Prompt 2")
    st.button("Close", type="primary", on_click=update_menu_img_gen_splash)
else:
    st.button("Again", type="primary", on_click=update_menu_img_gen_splash)

###########################

opt_model_id_list = [
    "stability.stable-diffusion-xl-v1"
]

opt_style_preset_list = [
    "anime",
    "photographic",
    "3d-model",
    "cinematic",
    "fantasy-art",
    "enhance",
    "isometric",
    "line-art"
]

opt_negative_prompt_list = [
    "ugly", "tiling", "out of frame",
    "disfigured", "deformed", "bad anatomy", "cut off", "low contrast", 
    "underexposed", "overexposed", "bad art", "beginner", "amateur", "blurry", "draft", "grainy"
]

opt_dimensions_list = [
    "1024x1024", "1152x896", "1216x832", "1344x768", "1536x640", "640x1536", "768x1344", "832x1216", "896x1152"
]

opt_style_preset_help = """
A style preset that guides the image model towards a particular style.
"""

opt_config_scale_help = """
Determines how much the final image portrays the prompt. Use a lower number to increase randomness in the generation.
"""

opt_steps_help = """
Generation step determines how many times the image is sampled. More steps can result in a more accurate result.
"""

opt_model_id = "stability.stable-diffusion-xl-v1"
opt_negative_prompt = opt_negative_prompt_list

with st.sidebar:
    #opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_style_preset = st.selectbox(label=":blue[**Style Presets**]", options=opt_style_preset_list, index = 0, key="style_preset", help=opt_style_preset_help)
    opt_config_scale = st.slider(label=":blue[**Config Scale**] - Loose vs Strict", min_value=0, max_value=35, value=10, step=1, key="config_scale", help=opt_config_scale_help)
    opt_steps = st.slider(label=":blue[**Steps**]", min_value=10, max_value=50, value=30, step=1, key="steps", help=opt_steps_help)
    opt_dimensions = st.selectbox(label=":blue[**Dimensions - Width x Height**]", options=opt_dimensions_list, index = 0, key="dimensions")
    #opt_negative_prompt = st.multiselect(label="Negative Prompt", options=opt_negative_prompt_list, default=opt_negative_prompt_list, key="negative_prompt")
    #opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")
    opt_seed = st.slider(label=":blue[**Seed**]", min_value=-1, max_value=4294967295, value=-1, step=1, key="seed")
    opt_negative_prompt_csv = st.text_area(label=":blue[**Negative Prompts**]", value="", placeholder="Things you don't want to see in the generated image. Input comma separated values. e.g. ugly,disfigured,low contrast,underexposed,overexposed,blurry,grainy", max_chars=256, key="negative_prompts")



#st.title("💬 🖌️ 🖼️ Image Generator")
st.markdown("🖼️ Image Generator 3")
#st.write("Text to Image")

if "menu_img_gen_messages" not in st.session_state:
    st.session_state["menu_img_gen_messages"] = [
        #{"role": "user", "content": "Hello there."},
        #{"role": "assistant", "content": "How can I help you?"}
    ]

idx = 1
for msg in st.session_state.menu_img_gen_messages:
    idx = idx + 1
    content = msg["content"]
    with st.chat_message(msg["role"]):
        if "user" == msg["role"]:
            st.write(content)
        if "assistant" == msg["role"]:
            st.image(content)
            st.markdown(f":blue[**style**] {msg['style']} :blue[**seed**] {msg['seed']} :blue[**scale**] {msg['scale']} :blue[**steps**] {msg['steps']} :blue[**width**] {msg['width']} :blue[**height**] {msg['height']}")

if prompt := st.chat_input():

    st.session_state["menu_img_gen_splash"] = False

    message_history = st.session_state.menu_img_gen_messages.copy()
    message_history.append({"role": "user", "content": prompt})
    #st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    opt_dimensions_width = int(opt_dimensions.split("x")[0])
    opt_dimensions_height = int(opt_dimensions.split("x")[1])
    #print(f"width: {opt_dimensions_width}  height: {opt_dimensions_height}")

    opt_negative_prompt_elements = opt_negative_prompt_list
    if "" != opt_negative_prompt_csv:
        opt_negative_prompt_elements = opt_negative_prompt_csv.split(",")
    print(opt_negative_prompt_elements)
        

    seed = opt_seed
    if seed < 0:
        seed = random.randint(0, 4294967295)
    
    logger.info(f"prompt={prompt} negative={opt_negative_prompt_csv}")

    request = {
            "text_prompts": (
                #[{"text": prompt, "weight": 1.0}] + [{"text": negprompt, "weight": -1.0} for negprompt in opt_negative_prompt]
                [{"text": prompt, "weight": 1.0}] + [{"text": negprompt, "weight": -1.0} for negprompt in opt_negative_prompt_elements]
            ),
            "cfg_scale": opt_config_scale,
            #"clip_guidance_preset"
            "width": opt_dimensions_width,
            "height": opt_dimensions_height,
            "seed": seed, #random.randint(0, 4294967295),
            #"start_schedule": config["start_schedule"],
            "steps": opt_steps, # Generation step determines how many times the image is sampled. 10-50,50
            "style_preset": opt_style_preset,
            "samples": 1,
        }
    #json.dumps(request, indent=3)

    with st.spinner('Generating Image...'):

        try:

            response = bedrock_runtime.invoke_model(
                modelId = opt_model_id,
                contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
                accept = "application/json",
                body = json.dumps(request))
            
            response_body = json.loads(response.get("body").read())
            finish_reason = response_body.get("artifacts")[0].get("finishReason")
            if finish_reason == 'ERROR' or finish_reason == 'CONTENT_FILTERED':
                st.chat_message("system").write(f"Image generation error. Error code is {finish_reason}")
            else:
                response_image_base64 = response_body["artifacts"][0].get("base64")
                response_image:Image = base64_to_image(response_image_base64)

                #file_extension = ".png"
                #OUTPUT_IMG_PATH = os.path.join("./output/{}-{}{}".format("img", idx, file_extension))
                #print("OUTPUT_IMG_PATH: " + OUTPUT_IMG_PATH)
                #response_image.save(OUTPUT_IMG_PATH)

                with st.chat_message("assistant"):
                    current_datetime = datetime.now()
                    current_datetime_str = current_datetime.strftime("%Y/%m/%d, %H:%M:%S")
                    st.image(response_image)
                    st.markdown(f":blue[**style**] {opt_style_preset} :blue[**seed**] {seed} :blue[**scale**] {opt_config_scale} :blue[**steps**] {opt_steps} :blue[**width**] {opt_dimensions_width} :blue[**height**] {opt_dimensions_height} :green[**{current_datetime_str}**]")

                st.session_state.menu_img_gen_messages.append({"role": "user", "content": prompt})
                st.session_state.menu_img_gen_messages.append({"role": "assistant", 
                    "content": response_image, 
                    "style": opt_style_preset,
                    "seed": seed,
                    "scale": opt_config_scale,
                    "steps": opt_steps,
                    "width": opt_dimensions_width,
                    "height": opt_dimensions_height,
                })

        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            print("A client error occured: " + format(message))
            st.chat_message("system").write(message)


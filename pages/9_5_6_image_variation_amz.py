import streamlit as st
import boto3
import cmn_settings
import cmn_constants
import json
import logging
import cmn_auth
import base64
import io
import PIL
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

####################################################################################


### Utilities

def base64_to_image(base64_str) -> Image:
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))


def image_to_base64(image,mime_type:str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

mime_mapping = {
    "image/png": "PNG",
    "image/jpeg": "JPEG"
}

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

st.markdown(cmn_constants.css_btn_primary, unsafe_allow_html=True)

variation_prompts_init = [
    "Trendy casual sneakers, comfortable design, vibrant colors, for young adults",
    "Stylish high heel shoes, modern design, comfortable fit, for fashion-forward women",
    "Rugged hiking boots, waterproof material, excellent traction, for outdoor enthusiasts",
    "High-end designer shoes, premium materials, unique aesthetic, for fashion connoisseurs",
    "Slip-resistant work boots, with steel toe protection for construction or industrial settings",
    "Stylish and comfortable sandals for women, perfect for a beach vacation or resort wear",
]

opt_model_id_list = [ "amazon.titan-image-generator-v2:0" ]

opt_style_preset_list = [
    "anime",
    "photographic",
    "3d-model",
    "cinematic",
    "fantasy-art",
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

opt_model_id = "amazon.titan-image-generator-v2:0"
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


st.markdown("🖼️ Image Variation 1")

#if "menu_img_variation_messages" not in st.session_state:
#    st.session_state["menu_img_variation_messages"] = [
#        #{"role": "user", "content": "Hello there."},
#        #{"role": "assistant", "content": "How can I help you?"}
#    ]


#idx = 1
#for msg in st.session_state.menu_img_variation_messages:
#    idx = idx + 1
#    content = msg["content"]
#    with st.chat_message(msg["role"]):
#        if "user" == msg["role"]:
#            st.write(content)
#        if "assistant" == msg["role"]:
#            st.image(content)
#            st.markdown(f":blue[**style**] {msg['style']} :blue[**seed**] {msg['seed']} :blue[**scale**] {msg['scale']} :blue[**steps**] {msg['steps']} :blue[**width**] {msg['width']} :blue[**height**] {msg['height']}")


uploaded_file = st.file_uploader(
    "Base Image",
    type=["PNG", "JPEG"],
    accept_multiple_files=False,
    label_visibility="collapsed",
    key="menu_img_variation_init_image"
)

uploaded_file_name = None
if uploaded_file:
    uploaded_file_bytes = uploaded_file.getvalue()
    uploaded_file_image = Image.open(uploaded_file)
    uploaded_file_name = uploaded_file.name
    uploaded_file_type = uploaded_file.type
    uploaded_file_base64 = image_to_base64(uploaded_file_image, mime_mapping[uploaded_file_type])

    # Get the original dimensions
    original_width, original_height = uploaded_file_image.size
    new_width = original_width
    new_height = original_height

    # Check if the original dimensions are multiples of 64
    if original_width % 64 != 0 or original_height % 64 != 0:
        # Calculate the new dimensions that are multiples of 64
        new_width = (original_width + 63) // 64 * 64
        new_height = (original_height + 63) // 64 * 64

        # Resize the image
        uploaded_file_image = uploaded_file_image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
        uploaded_file_base64 = image_to_base64(uploaded_file_image, mime_mapping[uploaded_file_type])



    st.image(uploaded_file_image, caption=f"Base Image {original_width}x{original_height} {new_width}x{new_height}",
        use_column_width="auto" #"auto", "always", "never", or bool
    )
    print(uploaded_file_type)
    #uploaded_file_bytes = uploaded_file.read()
    #uploaded_file_base64 = base64.b64encode(uploaded_file_bytes).decode("utf-8")
    #uploaded_file_base64 = base64.b64encode(uploaded_file_bytes)

# Join the elements with a newline character
variation_prompts_init_str = "\n".join(variation_prompts_init)

# Display the text area with the joined string
variation_prompts_str = st.text_area(":blue[**Variation Prompts**]", value=variation_prompts_init_str, height=170, max_chars=2000,
                                     placeholder="Enter each variation as a separate line")

generate_btn = st.button("Generate", type="primary", disabled=uploaded_file_name==None)

#if prompt := st.chat_input(disabled=uploaded_file_name==None):

if generate_btn:

    # Split the string into lines
    variation_prompts_lines = variation_prompts_str.split("\n")

    # Take only the first 6 lines
    variation_prompts = [line.strip() for line in variation_prompts_lines[:6] if line.strip()]

    #message_history = st.session_state.menu_img_variation_messages.copy()
    #message_history.append({"role": "user", "content": prompt})
    #st.chat_message("user").write(prompt)

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
    
    #logger.info(f"prompt={prompt} negative={opt_negative_prompt_csv}")

    #json.dumps(request, indent=3)


    with st.spinner('Generating Image...'):

        with st.container():

            # Print commmon request properties
            current_datetime = datetime.now()
            current_datetime_str = current_datetime.strftime("%Y/%m/%d, %H:%M:%S")
            st.markdown(f":blue[**style**] {opt_style_preset} :blue[**seed**] {seed} :blue[**scale**] {opt_config_scale} :blue[**steps**] {opt_steps} :blue[**width**] {opt_dimensions_width} :blue[**height**] {opt_dimensions_height} :green[**{current_datetime_str}**]")
            
            cols = st.columns(3)
            
            try:

                for i, variation_prompt in enumerate(variation_prompts):

                    # Determine the column to place the image in
                    col_index = i % 3
                        
                    request = {
                            "taskType": "TEXT_IMAGE",
                            "textToImageParams": {
                                "text": variation_prompt,
                                "conditionImage": uploaded_file_base64,
                                "controlStrength": 0.7, #opt_config_scale,
                                "controlMode": "CANNY_EDGE"
                            },
                            "imageGenerationConfig": {
                                "numberOfImages": 1,
                                "height": opt_dimensions_height,
                                "width": opt_dimensions_width,
                                "cfgScale": opt_config_scale,
                                "quality": "standard",
                                "seed": seed,
                            }
                        }

                    response = bedrock_runtime.invoke_model(
                        modelId = opt_model_id,
                        contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
                        accept = "application/json",
                        body = json.dumps(request))
                    
                    response_body = json.loads(response.get("body").read())
                    finish_reason = response_body.get("error")

                    # Display the image and metadata in the corresponding column
                    with cols[col_index]:

                        if finish_reason == 'ERROR' or finish_reason == 'CONTENT_FILTERED':
                            st.markdown(f"Image generation error. Error code is {finish_reason}")
                        else:
                            #response_image_base64 = response_body["images"][0].get("base64")
                            response_image_base64 = response_body.get("images")[0]
                            response_image:Image = base64_to_image(response_image_base64)
                            st.image(response_image)
                            st.markdown(f":orange[*{variation_prompt}*]")
                            

                    #st.session_state.menu_img_variation_messages.append({"role": "user", "content": prompt})
                    #st.session_state.menu_img_variation_messages.append({"role": "assistant", 
                    #    "content": response_image, 
                    #    "style": opt_style_preset,
                    #    "seed": seed,
                    #    "scale": opt_config_scale,
                    #    "steps": opt_steps,
                    #    "width": opt_dimensions_width,
                    #    "height": opt_dimensions_height,
                    #})

            except ClientError as err:
                message = err.response["Error"]["Message"]
                logger.error("A client error occurred: %s", message)
                print("A client error occured: " + format(message))
                st.chat_message("system").write(message)



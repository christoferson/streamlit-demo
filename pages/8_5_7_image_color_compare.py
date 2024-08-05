import streamlit as st
import boto3
import cmn_settings
import cmn_constants
import json
import logging
import cmn_auth
import io
import base64
from PIL import Image, ImageFilter
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

st.markdown(cmn_constants.css_btn_primary, unsafe_allow_html=True)

rekognition = boto3.client('rekognition', region_name=AWS_REGION)

######################


def stream_result_data(stream):

    result_text = ""

    try: 

        for event in stream:
            
            if "chunk" in event:

                chunk = json.loads(event["chunk"]["bytes"])

                if chunk['type'] == 'message_start':
                    #opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens}"
                    pass

                elif chunk['type'] == 'message_delta':
                    pass

                elif chunk['type'] == 'content_block_delta':
                    if chunk['delta']['type'] == 'text_delta':
                        text = chunk['delta']['text']
                        result_text += f"{text}"
                        yield text

                elif chunk['type'] == 'message_stop':
                    invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                    input_token_count = invocation_metrics["inputTokenCount"]
                    output_token_count = invocation_metrics["outputTokenCount"]
                    latency = invocation_metrics["invocationLatency"]
                    lag = invocation_metrics["firstByteLatency"]
                    #stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                    #result_area.markdown(result_text)
                    #st.session_state["translate_result"] = result_text
                    #yield result_text
                    #yield output_token_count
                    pass


            elif "internalServerException" in event:
                exception = event["internalServerException"]
                yield exception
            elif "modelStreamErrorException" in event:
                exception = event["modelStreamErrorException"]
                yield exception
            elif "modelTimeoutException" in event:
                exception = event["modelTimeoutException"]
                yield exception
            elif "throttlingException" in event:
                exception = event["throttlingException"]
                yield exception
            elif "validationException" in event:
                exception = event["validationException"]
                yield exception
            else:
                yield "Unknown Token"

        st.session_state["translate_result"] = result_text
    
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        yield message


######################

def rgb_to_hex(rgb):
    """
    Convert an RGB color tuple to a hexadecimal color code.

    Args:
        rgb (tuple): A tuple of three integers representing the RGB color values.
            Each value should be in the range 0-255.

    Returns:
        str: A hexadecimal color code in the format "#RRGGBB".
    """
    r, g, b = rgb
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def is_close_to_white(color, threshold=230):
    return all(c >= threshold for c in color)

def opencv_get_top_colors(image, n=7, exclude_background=True, white_threshold=200):
    # Check if the image is valid
    if image is None or image.size == 0:
        return []

    # Convert the image to a flattened 1D array of RGB values
    pixels = np.float32(image).reshape((-1, 3))

    # Exclude background colors if requested
    if exclude_background:
        try:
            # Use GrabCut algorithm for foreground segmentation
            mask = np.zeros(image.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (1, 1, image.shape[1]-2, image.shape[0]-2)  # Slightly smaller rectangle
            cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Check if any foreground pixels were found
            if np.sum(mask) == 0:
                raise ValueError("No foreground pixels found")

            pixels = pixels[mask.ravel() == 1]
        except Exception as e:
            print(f"GrabCut failed: {str(e)}. Using entire image.")
            # If GrabCut fails, use the entire image
            pixels = np.float32(image).reshape((-1, 3))

    # Filter out colors close to white
    non_white_pixels = [tuple(pixel) for pixel in pixels if not is_close_to_white(pixel, white_threshold)]

    # Create a dictionary to store the count of each RGB value
    color_counts = Counter(non_white_pixels)

    # Sort the dictionary by value (count) in descending order
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

    # Get the top n colors
    top_colors = sorted_colors[:n]

    return top_colors


#####

from PIL import Image, ImageFilter
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_foreground_mask(image):
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    mask = edges.point(lambda x: 255 if x > 10 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(5))
    return mask

def get_dominant_colors(image, mask, n_colors=5, white_threshold=230):
    img_array = np.array(image)
    mask_array = np.array(mask)
    foreground_pixels = img_array[mask_array == 255]
    non_white_pixels = foreground_pixels[(foreground_pixels < white_threshold).any(axis=1)]

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(non_white_pixels)

    colors = kmeans.cluster_centers_.astype(int)
    label_counts = Counter(kmeans.labels_)
    sorted_colors = sorted(zip(colors, label_counts.values()), key=lambda x: x[1], reverse=True)

    total_pixels = sum(label_counts.values())
    color_info = [
        {
            "rgb": tuple(color),
            "hex": rgb_to_hex(color),
            "percentage": count / total_pixels * 100
        }
        for color, count in sorted_colors
    ]

    return color_info

def pillow_get_dominant_foreground_colors(image:Image, n_colors=5, white_threshold=230):

    # Get foreground mask
    mask = get_foreground_mask(image)

    # Get dominant colors
    dominant_colors = get_dominant_colors(image, mask, n_colors, white_threshold)

    return dominant_colors

######################

def get_brightness(img):
    """
    Estimate the brightness of an image on a scale of 0-100.
    :param img: PIL Image object
    :return: float, estimated brightness (0-100)
    """
    gray_img = img.convert('L')
    brightness = np.mean(np.array(gray_img))
    # Normalize to 0-100 scale
    return (brightness / 255) * 100

def get_sharpness(img):
    """
    Estimate the sharpness of an image on a scale of 0-100.
    :param img: PIL Image object
    :return: float, estimated sharpness (0-100)
    """
    blurred = img.filter(ImageFilter.BLUR)
    diff = np.array(img) - np.array(blurred)
    sharpness = np.var(diff)
    # Normalize to 0-100 scale (assuming max sharpness around 10000)
    return min(sharpness / 100, 100)

def get_contrast(img):
    """
    Estimate the contrast of an image on a scale of 0-100.
    :param img: PIL Image object
    :return: float, estimated contrast (0-100)
    """
    gray_img = img.convert('L')
    contrast = np.std(np.array(gray_img))
    # Normalize to 0-100 scale (assuming max contrast around 100)
    return min(contrast, 100)

##################

@st.cache_data(show_spinner='Loading Image Properties')
def bedrock_rekognition_get_image_properties(uploaded_file_bytes):
    response = rekognition.detect_labels(
        Image={'Bytes': uploaded_file_bytes},
        #MaxLabels=123,
        #MinConfidence=...,
        Features=[
            'IMAGE_PROPERTIES',
        ],
        Settings={
            'ImageProperties': {
                'MaxDominantColors': 7
            }
        }
    )
    return response


###########################


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


col2, col3 = st.columns([2, 2])

######

with col2:

    uploaded_file  = st.file_uploader(
        "Image File - 1",
        type=["PNG", "JPEG"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    uploaded_file_name = None
    uploaded_file_bytes = None
    uploaded_file_image = None
    uploaded_file_1_quality = None
    uploaded_file_1_foreground_colors = []
    if uploaded_file:
        uploaded_file_bytes = uploaded_file.getvalue()
        uploaded_file_image = Image.open(uploaded_file)
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        uploaded_file_base64 = image_to_base64(uploaded_file_image, mime_mapping[uploaded_file_type])
        st.image(
            uploaded_file_image,
            use_column_width=True,
        )

    if uploaded_file_bytes and uploaded_file_bytes != None:
        
        uploaded_file_fetch_image_properties = st.checkbox(":rainbow[**Get Image Properties (Rekognition)**]", key="uploaded_file_fetch_image_properties")

        if uploaded_file_fetch_image_properties:
            #response = rekognition.detect_labels(
            #    Image={'Bytes': uploaded_file_bytes},
            #    #MaxLabels=123,
            #    #MinConfidence=...,
            #    Features=[
            #        'IMAGE_PROPERTIES',
            #    ],
            #    Settings={
            #        'ImageProperties': {
            #            'MaxDominantColors': 7
            #        }
            #    }
            #)
            response = bedrock_rekognition_get_image_properties(uploaded_file_bytes)
            img_properties = response['ImageProperties']
            img_quality = img_properties['Quality']
            img_quality_brightness = img_quality['Brightness']
            img_quality_sharpness = img_quality['Sharpness']
            img_quality_contrast = img_quality['Contrast']
            st.markdown(f":blue[**Brightness:**] {img_quality_brightness:.2f} :blue[**Sharpness:**] {img_quality_sharpness:.2f} :blue[**Contrast:**] {img_quality_contrast:.2f}")
            uploaded_file_1_quality = f"Brightness: {img_quality_brightness:.2f} Sharpness: {img_quality_sharpness:.2f} Contrast: {img_quality_contrast:.2f}"

            fg_dominant_colors = img_properties['Foreground']['DominantColors']

            uploaded_file_1_foreground_colors = []
            for fg_dominant_color in fg_dominant_colors:
                fg_dc_red = fg_dominant_color['Red']
                fg_dc_blue = fg_dominant_color['Blue']
                fg_dc_green = fg_dominant_color['Green']
                fg_dc_hex = fg_dominant_color['HexCode']
                vfg_dc_css_color = fg_dominant_color['CSSColor']
                vfg_dc_simple_color = fg_dominant_color['SimplifiedColor']
                vfg_dc_pixel_percent = fg_dominant_color['PixelPercent']
                st.markdown(f"RGB({fg_dc_red},{fg_dc_green},{fg_dc_blue}) {fg_dc_hex.upper()} {vfg_dc_css_color} {vfg_dc_pixel_percent:.2f}% <span style='font-size:28px;color:rgb({fg_dc_red}, {fg_dc_green}, {fg_dc_blue})'>■</span>", unsafe_allow_html=True)
                uploaded_file_1_foreground_colors.append(f"RGB({fg_dc_red},{fg_dc_green},{fg_dc_blue}) {fg_dc_hex.upper()} {vfg_dc_css_color} {vfg_dc_pixel_percent:.2f}%")

###

    if uploaded_file_bytes and uploaded_file_bytes != None:

        uploaded_file_fetch_pillow = st.checkbox(":rainbow[Get Image Properties (Pillow)]", key="uploaded_file_fetch_pillow")

        if uploaded_file_fetch_pillow:

            img_quality_brightness = get_brightness(uploaded_file_image)
            img_quality_sharpness = get_sharpness(uploaded_file_image)
            img_quality_contrast = get_contrast(uploaded_file_image)

            st.markdown(f":blue[**Brightness:**] {img_quality_brightness:.2f} :blue[**Sharpness:**] {img_quality_sharpness:.2f} :blue[**Contrast:**] {img_quality_contrast:.2f}")

            dominant_colors = pillow_get_dominant_foreground_colors(uploaded_file_image, n_colors=7, white_threshold=200)

            for dominant_color in dominant_colors:
                color = dominant_color['rgb']
                st.markdown(f"RGB{dominant_color['rgb']} {dominant_color['hex'].upper()} {dominant_color['percentage']:.2f}% <span style='font-size:28px;color:rgb({color[0]}, {color[1]}, {color[2]})'>■</span>", unsafe_allow_html=True)
                uploaded_file_1_foreground_colors.append(f"RGB{dominant_color['rgb']} {dominant_color['hex'].upper()} {dominant_color['percentage']:.2f}%")

######

with col3:

    uploaded_file_2 = st.file_uploader(
        "Image File - 2",
        type=["PNG", "JPEG"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    uploaded_file_2_name = None
    uploaded_file_2_bytes = None
    uploaded_file_2_image = None
    uploaded_file_2_quality = None
    uploaded_file_2_foreground_colors = []
    if uploaded_file_2:
        uploaded_file_2_bytes = uploaded_file_2.getvalue()
        uploaded_file_2_image = Image.open(uploaded_file_2)
        image_2 = Image.open(uploaded_file_2)
        uploaded_file_2_name = uploaded_file_2.name
        uploaded_file_2_type = uploaded_file_2.type
        uploaded_file_2_base64 = image_to_base64(image_2, mime_mapping[uploaded_file_2_type])
        st.image(
            image_2,
            use_column_width=True
        )


    if uploaded_file_2_bytes and uploaded_file_2_bytes != None:
        
        uploaded_file_2_fetch_image_properties = st.checkbox(":rainbow[Get Image Properties (Rekognition)]", key="uploaded_file_2_fetch_image_properties")

        if uploaded_file_2_fetch_image_properties:
            #response = rekognition.detect_labels(
            #    Image={'Bytes': uploaded_file_2_bytes},
            #    #MaxLabels=123,
            #    #MinConfidence=...,
            #    Features=[
            #        'IMAGE_PROPERTIES',
            #    ],
            #    Settings={
            #        'ImageProperties': {
            #            'MaxDominantColors': 7
            #        }
            #    }
            #)
            response = bedrock_rekognition_get_image_properties(uploaded_file_2_bytes)
            img_properties = response['ImageProperties']
            fg_dominant_colors = img_properties['Foreground']['DominantColors']
            img_quality = img_properties['Quality']
            img_quality_brightness = img_quality['Brightness']
            img_quality_sharpness = img_quality['Sharpness']
            img_quality_contrast = img_quality['Contrast']
            st.markdown(f":blue[**Brightness:**] {img_quality_brightness:.2f} :blue[**Sharpness:**] {img_quality_sharpness:.2f} :blue[**Contrast:**] {img_quality_contrast:.2f}")
            uploaded_file_2_quality = f"Brightness: {img_quality_brightness:.2f} Sharpness: {img_quality_sharpness:.2f} Contrast: {img_quality_contrast:.2f}"

            uploaded_file_2_foreground_colors = []
            for fg_dominant_color in fg_dominant_colors:
                fg_dc_red = fg_dominant_color['Red']
                fg_dc_blue = fg_dominant_color['Blue']
                fg_dc_green = fg_dominant_color['Green']
                fg_dc_hex = fg_dominant_color['HexCode']
                vfg_dc_css_color = fg_dominant_color['CSSColor']
                vfg_dc_simple_color = fg_dominant_color['SimplifiedColor']
                vfg_dc_pixel_percent = fg_dominant_color['PixelPercent']
                st.markdown(f"RGB({fg_dc_red},{fg_dc_green},{fg_dc_blue}) {fg_dc_hex.upper()} {vfg_dc_css_color} {vfg_dc_pixel_percent:.2f}% <span style='font-size:28px;color:rgb({fg_dc_red}, {fg_dc_green}, {fg_dc_blue})'>■</span>", unsafe_allow_html=True)
                uploaded_file_2_foreground_colors.append(f"RGB({fg_dc_red},{fg_dc_green},{fg_dc_blue}) {fg_dc_hex.upper()} {vfg_dc_css_color} {vfg_dc_pixel_percent:.2f}%")

    if uploaded_file_2_bytes and uploaded_file_2_bytes != None:

        uploaded_file_2_fetch_pillow = st.checkbox(":rainbow[Get Image Properties (Pillow)]", key="uploaded_file_2_fetch_pillow")

        if uploaded_file_2_fetch_pillow:

            img_quality_brightness = get_brightness(uploaded_file_2_image)
            img_quality_sharpness = get_sharpness(uploaded_file_2_image)
            img_quality_contrast = get_contrast(uploaded_file_2_image)

            st.markdown(f":blue[**Brightness:**] {img_quality_brightness:.2f} :blue[**Sharpness:**] {img_quality_sharpness:.2f} :blue[**Contrast:**] {img_quality_contrast:.2f}")


            dominant_colors = pillow_get_dominant_foreground_colors(uploaded_file_2_image, n_colors=7, white_threshold=200)

            for dominant_color in dominant_colors:
                color = dominant_color['rgb']
                st.markdown(f"RGB{dominant_color['rgb']} {dominant_color['hex'].upper()} {dominant_color['percentage']:.2f}% <span style='font-size:28px;color:rgb({color[0]}, {color[1]}, {color[2]})'>■</span>", unsafe_allow_html=True)
                uploaded_file_2_foreground_colors.append(f"RGB{dominant_color['rgb']} {dominant_color['hex'].upper()} {dominant_color['percentage']:.2f}%")

#############################


#col1


#############################

st.divider()

compare_prompt = """I have provided you 2 images to compare their color and design. Please do the following:
1. For each image, list out the dominant colors in the foreground or of the subject. Use Hex Codes, RGB, and Css Color Names to describe each color.
2. Compare the 2 images and describe how similar or different are they.
3. In the scale of 1 to 10, 1 is totally different and 10 identical, rate and compare the dominant colors of the foreground or subject of the 2 images.
Think step by step and make sure to verify that the color codes are as accurate as possible.
"""

compare_btn = st.button("Compare", type="primary", disabled=uploaded_file_name==None or uploaded_file_2_name==None)

if compare_btn:
    
    content =  [
                    {
                        "type": "text",
                        "text": f"{compare_prompt}"
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

    message_history = []
    message_history.append({"role": "user", "content": content})

    request = {
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": opt_temperature,
        "top_p": opt_top_p,
        "top_k": opt_top_k,
        "max_tokens": opt_max_tokens,
        "system": opt_system_msg,
        "messages": message_history #st.session_state.messages
    }

    try:
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId = opt_model_id, #bedrock_model_id, 
            contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
            accept = "application/json",
            body = json.dumps(request))

        #with st.chat_message("assistant", avatar=setAvatar("assistant")):
        #result_text = ""
        result_container = st.container(border=True)
        #result_area = st.empty()
        stream = response["body"]
        result_container.write_stream(stream_result_data(stream))
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)

st.divider()

uploaded_file_1_foreground_colors_meta = '\n'.join(str(item) for item in uploaded_file_1_foreground_colors)
uploaded_file_2_foreground_colors_meta = '\n'.join(str(item) for item in uploaded_file_2_foreground_colors)

compare_with_meta_prompt = f"""I have provided you 2 images as well as metadata regarding the image quality and dominant foreground colors.
<image_1_quality>
{uploaded_file_1_quality}
</image_1_quality>
<image_1_foreground_colors>
{uploaded_file_1_foreground_colors_meta}
</image_1_foreground_colors>

<image_2_quality>
{uploaded_file_2_quality}
</image_2_quality>
<image_2_foreground_colors>
{uploaded_file_2_foreground_colors_meta}
</image_2_foreground_colors>


Please do the following:
1. For each image, list out the dominant colors in the foreground or subject. Use Hex Codes, RGB, and Css Color Names to describe each color. In case the indicated color in the metadata is incorrect or inaccurate, please revise and output the correct color.
2. Check if the image quality are good and there are no bad characteristics like Underexposed, Overexposed, Dim, Low contrast, Blurred, Grainy/Noisy, Oversaturated, Lens flare, Vignetting, Poor white balance etc. Also, if the metadata is available, apply quantitative analysis using the provided brighness, contrast, and sharpness.
3. Verify the provided dominant foreground colors of each image if they are accurate and then re-order them as necessary in descending order. Limit the output to the top 5 colors.
4. Comparison Section: While also looking back at the prior information from previous steps, compare the 2 images and describe how similar or different are they.
5. Conclusion Section: As a conclusion, consider all prior information and then in the scale of 1 to 10, 1 is totally different and 10 identical, rate and compare the dominant colors of the foreground or subject of the 2 images.
Think step by step and make sure to verify that the color codes are as accurate as possible.
Show each step as a separate and independent paragraph with a clear header label. Prefix each section title with ***. 
Provide sufficient space in between the sections or steps by adding 3 newlines after the sections or paragraphs.
Provide the output in both English and Japanese language.
"""

compare_with_meta_btn = st.button("Compare with Meta", type="primary", disabled=uploaded_file_name==None or uploaded_file_2_name==None)

if compare_with_meta_btn:
    
    content =  [
                    {
                        "type": "text",
                        "text": f"{compare_with_meta_prompt}"
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

    message_history = []
    message_history.append({"role": "user", "content": content})

    request = {
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": opt_temperature,
        "top_p": opt_top_p,
        "top_k": opt_top_k,
        "max_tokens": opt_max_tokens,
        "system": opt_system_msg,
        "messages": message_history #st.session_state.messages
    }

    try:
        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId = opt_model_id, #bedrock_model_id, 
            contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
            accept = "application/json",
            body = json.dumps(request))

        #with st.chat_message("assistant", avatar=setAvatar("assistant")):
        #result_text = ""
        result_container = st.container(border=True)
        #result_area = st.empty()
        stream = response["body"]
        result_container.write_stream(stream_result_data(stream))
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)
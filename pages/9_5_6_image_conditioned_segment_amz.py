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
    "Colorful, realistic shoe based on this sketch. Add vibrant colors, textures, and details to make it look like a real, stylish sneaker. Highly detailed. 4K.",
    #"Transform this shoe sketch into a vibrant, photorealistic sneaker. Add rich textures, bold colors, and intricate details. Maintain the original shape while bringing it to life. 4K resolution, highly detailed.",
    #"Color this shoe outline with a modern, trendy palette. Incorporate sleek materials, subtle gradients, and eye-catching accents. Make it look like a premium, fashionable sneaker. Ultra-high definition, sharp details.",
    "Breathe life into this shoe sketch with a explosion of colors. Create a hyper-realistic sneaker with visible fabric textures, reflective elements, and complex stitching patterns. Photorealistic quality, 4K detail.",
    #"Render this shoe drawing as a cutting-edge athletic sneaker. Use a mix of mesh, synthetic leather, and rubber textures. Add dynamic color blocking and futuristic design elements. Highly detailed, studio-quality image.",
    "Convert this basic shoe outline into a luxury sneaker design. Incorporate premium materials like suede, full-grain leather, and metallic accents. Use a sophisticated color palette. Photorealistic rendering, ultra-high resolution.",
    "Transform this sketch into a street-style sneaker icon. Add bold graffiti-inspired colors, unique texture combinations, and urban-themed details. Keep the original silhouette but make it pop with vivid realism. 4K quality.",
    "Bring this shoe sketch to life as a retro-inspired sneaker. Use vintage color schemes, classic materials, and nostalgic design elements. Add wear and texture for authenticity. Highly detailed, photorealistic rendering.",
    #"Color this outline to create a futuristic, tech-inspired sneaker. Incorporate holographic elements, neon accents, and sleek, modern textures. Make it look like it's from the year 2050. Ultra-high definition, sharp details.",
    "Render this shoe sketch as a high-performance running sneaker. Add breathable mesh textures, supportive overlays, and a dynamic sole design. Use a gradient color scheme for a sense of motion. 4K resolution, highly detailed.",
    "Transform this basic shoe drawing into an avant-garde fashion sneaker. Use unconventional color combinations, mixed materials, and artistic details. Make it worthy of a high-fashion runway. Photorealistic quality, intricate details."
]

variation_prompts_init = [
    "Transform this sketch into a cyberpunk-inspired sneaker. Blend neon colors with matte black, add holographic accents and LED-like details. Make it look futuristic yet wearable. 4K resolution, hyper-detailed.",
    "Render this outline as an eco-friendly, sustainable sneaker. Use textures resembling recycled materials, natural dyes, and organic shapes. Add earthy tones with pops of vibrant green. Highly detailed, photorealistic quality.",
    "Bring this shoe sketch to life as a luxurious basketball sneaker. Combine premium leather with high-tech performance materials. Use a bold color scheme with metallic gold accents. Ultra-high definition, visible textures.",
    "Convert this drawing into a whimsical, fairy-tale inspired shoe. Add fantastical elements like tiny flowers, sparkling dust, and iridescent materials. Use pastel colors with magical shimmer. Dreamlike quality, intricate 4K details.",
    "Transform this outline into a rugged, all-terrain hiking boot. Add durable textures like waterproof leather and ripstop fabric. Use earth tones with bright accents for visibility. Hyper-realistic, showing every stitch and tread.",
    "Render this sketch as a 1950s vintage-style sneaker with a modern twist. Combine classic canvas textures with contemporary color blocking. Add retro-futuristic details. High-resolution, with a slightly worn, authentic look.",
    "Bring this shoe to life as a high-fashion platform sneaker. Mix luxe materials like patent leather and suede. Use a monochromatic color scheme with a pop of neon. Avant-garde style, photorealistic rendering.",
    "Color this outline as a glow-in-the-dark party sneaker. Use fluorescent pigments and light-reactive materials. Add fun, youthful patterns and textures. Ultra-vibrant in normal light, with visible glow effect. 4K detail.",
]


variation_prompts_init = [
    "Transform this basic sketch into a steampunk-inspired boot. Incorporate textures of aged leather, polished brass, and intricate clockwork. Use a rich, vintage color palette. Highly detailed with visible rivets and gears."
    "Render this drawing as a ballet-inspired athletic shoe. Blend the elegance of satin with the functionality of performance mesh. Use soft, pastel tones with ribbon-like laces. Graceful yet sporty, in high-definition detail.",
    "Convert this outline into a comic book style sneaker. Use bold, primary colors with black outlines and halftone pattern fills. Add 'pow' and 'zoom' text elements as part of the design. Highly stylized yet 3D-looking.",
    "Bring this shoe sketch to life as a sushi-inspired sneaker. Use textures and colors reminiscent of nori, rice, and various fish. Add chopstick-like laces and wasabi-colored accents. Photorealistic yet whimsical, 4K detail.",
    "Transform this basic shape into a glacier-inspired winter boot. Use icy blue tones, crystalline textures, and frosty white accents. Add the appearance of carved ice and snow-like fur trim. Ultra-high definition, cold and crisp looking.",
    "Render this outline as a Moroccan-inspired slip-on. Incorporate intricate mosaic patterns, rich jewel tones, and the texture of embroidered fabrics. Add metallic thread-like details. Ornate and luxurious, with photorealistic quality.",
    "Color this sketch as a firefighter-inspired safety boot. Use durable, heat-resistant textures, reflective strips, and heavy-duty hardware. Combine black and yellow with a pop of fire-engine red. Rugged and functional, highly detailed.",
]

variation_prompts_init = [
    "Transform this outline into a bioluminescent deep-sea inspired sneaker. Use dark blues and blacks with glowing neon accents. Add textures resembling fish scales and jellyfish tentacles. Ethereal and mysterious, in 4K detail."
    "Render this sketch as a Bauhaus-inspired minimalist shoe. Use primary colors in geometric blocks. Incorporate textures of smooth leather and canvas. Clean lines, functional aesthetics, highly detailed yet simple.",
    "Bring this shoe to life as a baroque-style formal sneaker. Add ornate gold filigree patterns over rich velvet textures. Use deep, royal colors like burgundy and navy. Opulent and extravagant, with photorealistic detailing.",
    "Convert this drawing into a circuit board sneaker. Use green and silver tones, add metallic pathways and electronic component-like details. Make it look like a wearable piece of technology. Ultra-high definition, futuristic.",
    "Color this outline as a candy-inspired platform shoe. Use swirling patterns of bright, saturated colors. Add textures that look like hard candy, gummy bears, and cotton candy. Playful and mouth-watering, in vivid 4K."
    "Transform this basic shape into a zen garden slip-on. Use sandy textures, incorporate patterns like raked sand and placed rocks. Add bamboo-like elements. Serene earth tones, with highly detailed, calming aesthetics.",
    "Render this sketch as a post-apocalyptic survival boot. Combine distressed leather with repurposed materials like tire treads and metal scraps. Use muted, dusty colors with rust accents. Weathered, rugged, highly detailed.",
    "Bring this outline to life as a Art Nouveau-inspired evening shoe. Use flowing, organic lines and nature-inspired motifs. Add iridescent peacock feather patterns and delicate floral elements. Elegant and artistic, in 4K detail.",
    "Color this drawing as a tropical rainforest hiking shoe. Use vibrant greens and earthy browns, add textures resembling leaves, bark, and exotic flowers. Include subtle animal print accents. Lush and wild, photorealistic quality.",
]


variation_prompts_init = [
    "Transform this sketch into a constellation-themed sneaker. Use deep space blues and purples with glittering star-like accents. Add zodiac symbols and galactic swirls. Cosmic and dreamy, with ultra-high definition details.",
    "Render this outline as a Mondrian-inspired color block shoe. Use primary colors in geometric sections divided by bold black lines. Clean, artistic, and striking, with visible canvas-like textures. Highly detailed, modernist aesthetic.",
    "Bring this shoe to life as a stealth tactical boot. Use matte black textures with subtle dark grey patterns for camouflage effect. Add technical details like reinforced toe caps and ankle supports. Sleek, functional, 4K resolution.",
    "Convert this drawing into a rococo-style mule. Use pastel colors, add intricate floral patterns and cherub motifs. Incorporate textures of silk and delicate lace. Ornate and romantic, with photorealistic detailing.",
    #"Color this sketch as a volcano-inspired trail runner. Use gradients of black, red, and orange to mimic cooling lava. Add textures resembling volcanic rock and ash. Dynamic and intense, with highly detailed, textured surfaces.",
    "Transform this basic shape into a stained-glass inspired formal shoe. Use jewel tones separated by black lines to mimic lead caming. Add a glossy finish to resemble polished glass. Artistic and luminous, in vivid 4K detail.",
    "Render this outline as a quantum physics-themed sneaker. Use cool blues and purples, add patterns representing particle waves and atomic structures. Incorporate holographic elements. Cerebral and cutting-edge, ultra-high definition.",
    "Bring this shoe sketch to life as a Día de los Muertos celebration boot. Use vibrant colors, incorporate intricate sugar skull patterns and marigold motifs. Add textures resembling embroidered fabric. Festive and cultural, highly detailed.",
    "Color this drawing as a bonsai-inspired minimalist sandal. Use natural wood tones and textures, add delicate branch-like straps and leaf-shaped accents. Zen and organic, with 4K photorealistic wood grain details.",
    "Transform this outline into a Van Gogh 'Starry Night' inspired sneaker. Use swirling blues and yellows, add textured brushstroke patterns. Make it look like a wearable masterpiece. Highly detailed, 4K resolution.",
]

variation_prompts_init = [
    "Render this sketch as a bioluminescent fungi boot. Use earthy browns with glowing neon accents. Add textures resembling mushroom caps and mycelium networks. Ethereal and organic, ultra-high definition.",
    "Bring this shoe to life as a Fabergé egg-inspired formal heel. Use pastel colors, add intricate gold filigree and jewel-like embellishments. Ornate and luxurious, with photorealistic detailing.",
    "Convert this drawing into a solar system sneaker. Use deep space black with planetary rings and orbits. Add textures of different planet surfaces. Cosmic and educational, in vivid 4K detail.",
    "Color this outline as a Venetian carnival masquerade shoe. Use rich jewel tones, add intricate mask-like patterns and feather textures. Mysterious and elegant, highly detailed.",
    "Transform this basic shape into a coral reef sandal. Use vibrant underwater colors, add textures of coral, sea anemones, and fish scales. Vibrant and alive, with ultra-high definition marine details.",
    "Render this sketch as a steampunk aviator boot. Combine distressed leather with brass gears and gauges. Use warm browns and metallic accents. Retro-futuristic and functional, in 4K detail.",
    "Bring this outline to life as a Mondrian-inspired color block heel. Use primary colors in geometric sections divided by bold black lines. Clean, artistic, and striking, with visible canvas-like textures.",
    "Color this drawing as a Northern Lights-inspired trail runner. Use dark blues and greens with streaks of purple and pink aurora effects. Add textures resembling snow and ice. Magical and dynamic, photorealistic quality.",
    "Transform this sketch into a Día de Muertos celebration sneaker. Use vibrant colors, incorporate sugar skull patterns and marigold motifs. Festive and cultural, highly detailed.",
]


variation_prompts_init = [
    "Render this outline as a cyberpunk street racer boot. Use neon colors over matte black, add holographic accents and LED-like details. Futuristic and edgy, with ultra-high definition textures.",
    "Bring this shoe to life as a Zen garden-inspired slip-on. Use sandy textures, incorporate patterns like raked sand and placed rocks. Serene earth tones, with highly detailed, calming aesthetics.",
    "Convert this drawing into a stained-glass cathedral window shoe. Use jewel tones separated by black lines to mimic lead caming. Add a glossy finish to resemble polished glass. Artistic and luminous, in vivid 4K detail.",
    "Color this sketch as a tropical bird-inspired running shoe. Use vibrant feather patterns, add textures resembling different types of plumage. Exotic and eye-catching, with photorealistic feather details.",
    "Transform this basic shape into a Art Deco cinema-inspired evening shoe. Use geometric patterns in gold and black, add textures reminiscent of velvet theater curtains. Glamorous and nostalgic, highly detailed.",
    "Render this outline as a quantum computing-themed sneaker. Use cool blues and silvers, add patterns representing qubits and entanglement. Incorporate circuit-like elements. Futuristic and complex, ultra-high definition.",
    "Bring this shoe sketch to life as a Moroccan tile-inspired sandal. Use intricate geometric patterns in rich blues and golds. Add textures resembling glazed ceramic. Ornate and cultural, with 4K photorealistic detail.",
    "Color this drawing as a bonsai-inspired minimalist boot. Use natural wood tones and textures, add delicate branch-like patterns and leaf-shaped accents. Zen and organic, with 4K photorealistic wood grain details.",
    "Transform this outline into a climate change awareness sneaker. Use gradients from cool blues to warm reds, incorporate melting ice cap patterns and rising sea level motifs. Impactful and thought-provoking, highly detailed.",
]

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
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
    "underexposed", "overexposed", "bad art", "beginner", "amateur", "blurry", "draft", "grainy",
    "distorted shape", "additional lines", "changing the original sketch structure"
]

opt_dimensions_list = [
    "1024x1024", "1152x768"
]

opt_style_preset_help = """
A style preset that guides the image model towards a particular style.
"""

opt_similarity_strength_help = """
Specifies how similar the generated image should be to the input image(s) Use a lower value to introduce more randomness in the generation. Accepted range is between 0.2 and 1.0 (both inclusive), while a default of 0.7 is used if this parameter is missing in the request.
"""

opt_config_scale_help = """
Determines how much the final image portrays the prompt. Use a lower number to increase randomness in the generation.
"""

opt_steps_help = """
Generation step determines how many times the image is sampled. More steps can result in a more accurate result.
"""

opt_model_id = "amazon.titan-image-generator-v2:0"
opt_negative_prompt = opt_negative_prompt_list
opt_negative_prompt_csv_init = "ugly, tiling, out of frame, disfigured, deformed, bad anatomy, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, blurry, draft, grainy, distorted shape, additional lines, changing the original sketch structure"

with st.sidebar:
    #opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    #opt_style_preset = st.selectbox(label=":blue[**Style Presets**]", options=opt_style_preset_list, index = 0, key="style_preset", help=opt_style_preset_help)
    opt_similarity_strength = st.slider(label=":blue[**Similarity Strength**] - Loose vs Strict", min_value=0.2, max_value=1.0, value=0.7, step=0.1, key="opt_similarity_strength", help=opt_similarity_strength_help)
    opt_config_scale = st.slider(label=":blue[**Config Scale**] - Loose vs Strict", min_value=1.1, max_value=8.0, value=8.0, step=0.1, key="config_scale", help=opt_config_scale_help)
    #opt_steps = st.slider(label=":blue[**Steps**]", min_value=10, max_value=50, value=30, step=1, key="steps", help=opt_steps_help)
    opt_dimensions = st.selectbox(label=":blue[**Dimensions - Width x Height**]", options=opt_dimensions_list, index = 1, key="dimensions")
    #opt_negative_prompt = st.multiselect(label="Negative Prompt", options=opt_negative_prompt_list, default=opt_negative_prompt_list, key="negative_prompt")
    #opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")
    opt_seed = st.slider(label=":blue[**Seed**]", min_value=-1, max_value=4294967295, value=-1, step=1, key="seed")
    opt_negative_prompt_csv = st.text_area(label=":blue[**Negative Prompts**]", value=opt_negative_prompt_csv_init, placeholder="Things you don't want to see in the generated image. Input comma separated values. e.g. ugly,disfigured,low contrast,underexposed,overexposed,blurry,grainy", max_chars=256, key="negative_prompts")


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


    with st.expander("Image", expanded=True):
        st.image(uploaded_file_image, caption=f"Base Image {original_width}x{original_height} {new_width}x{new_height}",
            use_column_width="auto" #"auto", "always", "never", or bool
        )
    print(uploaded_file_type)
    #uploaded_file_bytes = uploaded_file.read()
    #uploaded_file_base64 = base64.b64encode(uploaded_file_bytes).decode("utf-8")
    #uploaded_file_base64 = base64.b64encode(uploaded_file_bytes)

# Join the elements with a newline character
variation_prompts_init_str = "\n".join(variation_prompts_init)

with st.expander("Prompts", expanded=True):
    # Display the text area with the joined string
    variation_prompts_str = st.text_area(":blue[**Variation Prompts**]", value=variation_prompts_init_str, height=170, max_chars=2000,
                                        placeholder="Enter each variation as a separate line")

generate_btn = st.button("Generate", type="primary")

#if prompt := st.chat_input(disabled=uploaded_file_name==None):

if generate_btn:

    # Split the string into lines
    variation_prompts_lines = variation_prompts_str.split("\n")

    # Take only the first 6 lines
    variation_prompts = [line.strip() for line in variation_prompts_lines[:12] if line.strip()]

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
            st.markdown(f":blue[similarity] {opt_similarity_strength} :blue[**seed**] {seed} :blue[**scale**] {opt_config_scale} :blue[**width**] {opt_dimensions_width} :blue[**height**] {opt_dimensions_height} :green[**{current_datetime_str}**]")
            
            cols = st.columns(3)
            
            try:

                for i, variation_prompt in enumerate(variation_prompts):

                    # Determine the column to place the image in
                    col_index = i % 3
                        
                    request = {
                            "taskType": "TEXT_IMAGE",
                            "textToImageParams": {
                                "text": variation_prompt,
                                "negativeText": opt_negative_prompt_csv,
                                "conditionImage": uploaded_file_base64,
                                "controlMode": "CANNY_EDGE",
                                #"controlStrength": 0.7
                            },
                            "imageGenerationConfig": {
                                "numberOfImages": 1,
                                "height": opt_dimensions_height, #1024,
                                "width": opt_dimensions_width, #1024,
                                "cfgScale": opt_config_scale,
                                #"quality": "premium", #"standard" || "premium"
                                "seed": seed,
                            }
                        }
                    
                    print(json.dumps(request, indent=2))

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



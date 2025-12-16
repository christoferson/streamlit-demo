import streamlit as st
import boto3
import json
import logging
import base64
import io
import os
from PIL import Image
from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime

# Configuration
AWS_REGION = "us-east-1"  # Update with your region

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

####################################################################################

# Initialize Bedrock client
bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

####################################################################################

### Utilities

def base64_to_image(base64_str) -> Image:
    return Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))

def image_to_base64(image, mime_type: str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_image_size_kb(image, format_type):
    """Calculate the size of an image in KB for a given format"""
    buffer = io.BytesIO()
    image.save(buffer, format=format_type.upper())
    return len(buffer.getvalue()) / 1024

def estimate_output_size(width, height, format_type):
    """Estimate output size in MB based on dimensions and format"""
    pixels = width * height
    if format_type.lower() == 'png':
        estimated_bytes = pixels * 4
    elif format_type.lower() == 'jpeg':
        estimated_bytes = pixels * 0.75
    else:  # webp
        estimated_bytes = pixels * 0.5
    return estimated_bytes / (1024 * 1024)

mime_mapping = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP"
}

# Maximum payload size in bytes (16 MB)
MAX_PAYLOAD_SIZE = 16777216
MAX_PAYLOAD_SIZE_MB = MAX_PAYLOAD_SIZE / (1024 * 1024)

#####################

st.set_page_config(
    page_title="Stability AI Image Services",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.title("🎨 Stability AI Image Generation Services")
st.markdown("Transform your images with AI-powered Control Structure, Style Guide, and Style Transfer")

# Service options with detailed configurations
services = {
    "Control Structure": {
        "model_id": "us.stability.stable-image-control-structure-v1:0",
        "description": "Generate images while maintaining the structure of an input image. Perfect for recreating scenes or rendering characters.",
        "required_fields": ["prompt", "image"],
        "optional_fields": ["control_strength", "negative_prompt", "seed", "output_format", "style_preset"],
        "supports_style_preset": True,
        "supports_control_strength": True,
        "image_label": "Structure Image",
        "image_help": "Upload an image whose structure you want to maintain"
    },
    "Style Guide": {
        "model_id": "us.stability.stable-image-style-guide-v1:0",
        "description": "Extract stylistic elements from an input image and use them to guide creation of a new image based on your prompt.",
        "required_fields": ["prompt", "image"],
        "optional_fields": ["aspect_ratio", "negative_prompt", "seed", "output_format", "fidelity", "style_preset"],
        "supports_style_preset": True,
        "supports_fidelity": True,
        "supports_aspect_ratio": True,
        "image_label": "Style Reference Image",
        "image_help": "Upload an image whose style you want to extract"
    },
    "Style Transfer": {
        "model_id": "us.stability.stable-style-transfer-v1:0",
        "description": "Apply visual characteristics from a style image to a target image while preserving the original composition.",
        "required_fields": ["init_image", "style_image"],
        "optional_fields": ["prompt", "negative_prompt", "seed", "output_format", "composition_fidelity", "style_strength", "change_strength"],
        "supports_style_preset": False,
        "supports_composition_fidelity": True,
        "supports_style_strength": True,
        "supports_change_strength": True,
        "needs_two_images": True,
        "image_label": "Content Image",
        "image_help": "Upload the image you want to restyle"
    }
}

output_formats = ["png", "jpeg", "webp"]

style_presets = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture"
]

aspect_ratios = ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Service Settings")

    selected_service = st.selectbox(
        "Select Service",
        options=list(services.keys()),
        index=0,
        help="Choose the AI image service you want to use"
    )

    service_config = services[selected_service]

    st.info(service_config["description"])

    # Show available fields
    with st.expander("📋 Available Parameters"):
        st.write("**Required:**")
        for field in service_config["required_fields"]:
            st.write(f"✅ {field}")
        st.write("**Optional:**")
        for field in service_config["optional_fields"]:
            st.write(f"⚙️ {field}")

    st.divider()

    # Output format
    output_format = st.selectbox(
        "Output Format",
        options=output_formats,
        index=0,
        help="Choose the output image format"
    )

    st.divider()

    # Prompt (required for Control Structure and Style Guide, optional for Style Transfer)
    if "prompt" in service_config["required_fields"] or "prompt" in service_config["optional_fields"]:
        st.markdown("##### 📝 Text Prompts")

        if selected_service == "Control Structure":
            default_prompt = "surreal structure with motion generated sparks lighting the scene"
        elif selected_service == "Style Guide":
            default_prompt = "wide shot of modern metropolis"
        else:  # Style Transfer
            default_prompt = "statue"

        prompt = st.text_area(
            "Prompt" + (" (Optional)" if "prompt" in service_config["optional_fields"] else ""),
            value=default_prompt,
            height=100,
            max_chars=10000,
            help="Describe what you wish to see. Use (word:weight) to control emphasis. Values 0-1.0 de-emphasize, 1.1-2.0 emphasize."
        )

        negative_prompt = st.text_area(
            "Negative Prompt (Optional)",
            value="blurry, low quality, distorted, artifacts",
            height=80,
            max_chars=10000,
            help="Describe what you don't want to see"
        )
    else:
        prompt = None
        negative_prompt = None

    st.divider()

    # Service-specific controls
    st.markdown("##### 🎨 Creative Controls")

    # Control Strength (Control Structure)
    if service_config.get("supports_control_strength", False):
        control_strength = st.slider(
            "Control Strength",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="How much influence the structure image has on generation (0=least, 1=maximum)"
        )
    else:
        control_strength = None

    # Fidelity (Style Guide)
    if service_config.get("supports_fidelity", False):
        fidelity = st.slider(
            "Style Fidelity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="How closely the output style resembles the input style"
        )
    else:
        fidelity = None

    # Composition Fidelity (Style Transfer)
    if service_config.get("supports_composition_fidelity", False):
        composition_fidelity = st.slider(
            "Composition Fidelity",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="How closely the output composition matches the input"
        )
    else:
        composition_fidelity = None

    # Style Strength (Style Transfer)
    if service_config.get("supports_style_strength", False):
        style_strength = st.slider(
            "Style Strength",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="How much the style image influences the output (0=identical to input, 1=maximum style)"
        )
    else:
        style_strength = None

    # Change Strength (Style Transfer)
    if service_config.get("supports_change_strength", False):
        change_strength = st.slider(
            "Change Strength",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="How much the original image should change"
        )
    else:
        change_strength = None

    # Aspect Ratio (Style Guide)
    if service_config.get("supports_aspect_ratio", False):
        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            options=aspect_ratios,
            index=0,
            help="Controls the aspect ratio of the generated image"
        )
    else:
        aspect_ratio = None

    # Style Preset
    if service_config.get("supports_style_preset", False):
        style_preset = st.selectbox(
            "Style Preset (Optional)",
            options=style_presets,
            index=0,
            help="Guide the image towards a particular style"
        )
    else:
        style_preset = None

    st.divider()

    # Seed
    st.markdown("##### 🎲 Randomization")
    seed = st.number_input(
        "Seed (0 for random)",
        min_value=0,
        max_value=4294967294,
        value=0,
        help="Use a specific seed for reproducible results"
    )

# Main content
if service_config.get("needs_two_images", False):
    # Style Transfer needs two images
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📤 Content Image")
        uploaded_file = st.file_uploader(
            service_config["image_label"],
            type=["png", "jpg", "jpeg", "webp"],
            key="content_image",
            help=service_config["image_help"]
        )

        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
            width, height = uploaded_image.size
            st.image(uploaded_image, caption=f"Content ({width}x{height})", use_container_width=True)

    with col2:
        st.subheader("🎨 Style Image")
        style_file = st.file_uploader(
            "Style Reference Image",
            type=["png", "jpg", "jpeg", "webp"],
            key="style_image",
            help="Upload an image whose style you want to apply"
        )

        if style_file:
            style_image = Image.open(style_file)
            style_width, style_height = style_image.size
            st.image(style_image, caption=f"Style ({style_width}x{style_height})", use_container_width=True)

    with col3:
        st.subheader("✨ Generated Result")
        result_placeholder = st.empty()
else:
    # Single image services
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            service_config["image_label"],
            type=["png", "jpg", "jpeg", "webp"],
            help=service_config["image_help"]
        )

        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
            width, height = uploaded_image.size
            total_pixels = width * height

            st.image(uploaded_image, caption=f"Input Image ({width}x{height}, {total_pixels:,} pixels)", use_container_width=True)

    with col2:
        st.subheader("✨ Generated Result")
        result_placeholder = st.empty()

# Validation function
def validate_image(image, image_name="Image"):
    """Validate image meets requirements"""
    width, height = image.size
    total_pixels = width * height
    errors = []
    warnings = []

    # Check minimum dimensions
    if width < 64 or height < 64:
        errors.append(f"❌ {image_name} sides must be at least 64px")

    # Check total pixels
    if total_pixels > 9437184:
        errors.append(f"❌ {image_name} total pixels cannot exceed 9,437,184")

    # Check aspect ratio
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 2.5:
        errors.append(f"❌ {image_name} aspect ratio must be between 1:2.5 and 2.5:1")

    return errors, warnings

# Generate button and processing
if service_config.get("needs_two_images", False):
    can_generate = uploaded_file is not None and style_file is not None

    if can_generate:
        # Validate both images
        errors1, _ = validate_image(uploaded_image, "Content image")
        errors2, _ = validate_image(style_image, "Style image")
        all_errors = errors1 + errors2

        if all_errors:
            for error in all_errors:
                st.error(error)
            can_generate = False
else:
    can_generate = uploaded_file is not None

    if can_generate:
        errors, warnings = validate_image(uploaded_image)

        if errors:
            for error in errors:
                st.error(error)
            can_generate = False
        elif warnings:
            for warning in warnings:
                st.warning(warning)

# Generate button
if can_generate:
    generate_button = st.button(
        f"🚀 Generate with {selected_service}", 
        type="primary", 
        use_container_width=True
    )

    if generate_button:
        with st.spinner(f"Generating with {selected_service}... This may take a moment."):
            try:
                # Prepare images
                file_type = uploaded_file.type.split('/')[-1]
                if file_type == 'jpg':
                    file_type = 'jpeg'

                image_base64 = image_to_base64(uploaded_image, file_type.upper())

                # Build parameters based on service
                params = {"output_format": output_format}

                if service_config.get("needs_two_images", False):
                    # Style Transfer
                    style_file_type = style_file.type.split('/')[-1]
                    if style_file_type == 'jpg':
                        style_file_type = 'jpeg'
                    style_image_base64 = image_to_base64(style_image, style_file_type.upper())

                    params["init_image"] = image_base64
                    params["style_image"] = style_image_base64

                    if prompt and prompt.strip():
                        params["prompt"] = prompt
                    if composition_fidelity is not None:
                        params["composition_fidelity"] = composition_fidelity
                    if style_strength is not None:
                        params["style_strength"] = style_strength
                    if change_strength is not None:
                        params["change_strength"] = change_strength
                else:
                    # Control Structure or Style Guide
                    params["image"] = image_base64

                    if prompt:
                        params["prompt"] = prompt

                    if control_strength is not None:
                        params["control_strength"] = control_strength

                    if fidelity is not None:
                        params["fidelity"] = fidelity

                    if aspect_ratio and aspect_ratio != "1:1":
                        params["aspect_ratio"] = aspect_ratio

                # Common optional parameters
                if negative_prompt and negative_prompt.strip():
                    params["negative_prompt"] = negative_prompt

                if seed > 0:
                    params["seed"] = seed

                if style_preset and style_preset != "None":
                    params["style_preset"] = style_preset

                # Log request
                logger.info(f"Generating with service: {service_config['model_id']}")
                logger.info(f"Parameters: {json.dumps({k: v for k, v in params.items() if 'image' not in k.lower()}, indent=2)}")

                # Display request info
                with st.expander("🔍 Request Parameters", expanded=False):
                    display_params = {k: v for k, v in params.items() if 'image' not in k.lower()}
                    st.json(display_params)

                # Make API call
                start_time = datetime.now()

                response = bedrock_runtime.invoke_model(
                    modelId=service_config["model_id"],
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(params)
                )

                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()

                # Parse response
                response_body = json.loads(response.get("body").read())

                # Check for errors
                if "error" in response_body:
                    st.error(f"❌ Error: {response_body['error']}")
                else:
                    # Get generated image
                    generated_image_base64 = response_body["images"][0]
                    generated_image = base64_to_image(generated_image_base64)
                    gen_width, gen_height = generated_image.size

                    # Calculate output size
                    output_size_kb = get_image_size_kb(generated_image, output_format)
                    output_size_mb = output_size_kb / 1024

                    # Display result
                    with result_placeholder.container():
                        st.image(generated_image, caption=f"Generated Image ({gen_width}x{gen_height})", use_container_width=True)

                        # Metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Output Size", f"{gen_width}x{gen_height}")
                        with col_b:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        with col_c:
                            st.metric("File Size", f"{output_size_mb:.2f} MB")

                        st.success(f"✅ Generation completed successfully!")

                        # Download button
                        buffer = io.BytesIO()
                        generated_image.save(buffer, format=output_format.upper())
                        buffer.seek(0)

                        st.download_button(
                            label="⬇️ Download Generated Image",
                            data=buffer,
                            file_name=f"generated_{selected_service.lower().replace(' ', '_')}.{output_format}",
                            mime=f"image/{output_format}",
                            use_container_width=True
                        )

                        # Display metadata
                        with st.expander("📋 Generation Details"):
                            st.write(f"**Service:** {selected_service}")
                            st.write(f"**Model ID:** {service_config['model_id']}")
                            st.write(f"**Processing Time:** {processing_time:.2f} seconds")
                            if "seeds" in response_body:
                                st.write(f"**Seed Used:** {response_body['seeds'][0]}")
                            if "finish_reasons" in response_body:
                                st.write(f"**Finish Reason:** {response_body['finish_reasons'][0]}")
                            st.write(f"**Output Format:** {output_format}")
                            st.write(f"**Output File Size:** {output_size_kb:.2f} KB ({output_size_mb:.2f} MB)")

                            st.write("**Parameters Used:**")
                            for key, value in params.items():
                                if 'image' not in key.lower():
                                    st.write(f"- {key}: {value}")

            except ClientError as err:
                error_message = err.response["Error"]["Message"]
                logger.error("A client error occurred: %s", error_message)

                try:
                    error_detail = json.loads(error_message)
                    detail_msg = error_detail.get("detail", error_message)
                except:
                    detail_msg = error_message

                st.error(f"❌ AWS Error: {detail_msg}")

                if "payload size" in detail_msg.lower():
                    st.info("""
                    **Payload Size Error Solutions:**
                    1. Try JPEG format (most compressed)
                    2. Use a different seed value
                    3. Use smaller input images
                    """)

            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                st.error(f"❌ Error: {str(e)}")

elif uploaded_file:
    st.warning("⚠️ Please fix validation errors before generating")
else:
    with result_placeholder.container():
        if service_config.get("needs_two_images", False):
            st.info("👈 Upload both content and style images to get started")
        else:
            st.info("👈 Upload an image to get started")

# Footer with information
st.divider()

with st.expander("ℹ️ About Stability AI Image Services"):

    tab1, tab2, tab3, tab4 = st.tabs(["Control Structure", "Style Guide", "Style Transfer", "Tips & Tricks"])

    with tab1:
        st.markdown("""
        ### Control Structure
        Generate images while maintaining the structure of an input image.

        **Perfect for:**
        - Recreating scenes with different styles
        - Rendering characters from models
        - Maintaining composition while changing content

        **Key Parameters:**
        - **Prompt:** Describe what you want to see (required)
        - **Control Strength:** How much the structure influences generation (0-1, default 0.7)
        - **Style Preset:** Guide towards specific artistic styles

        **Image Requirements:**
        - Minimum side: 64px
        - Maximum pixels: 9,437,184
        - Aspect ratio: 1:2.5 to 2.5:1
        - Formats: JPEG, PNG, WebP

        **Prompt Weighting:**
        Use `(word:weight)` to control emphasis:
        - 0-1.0: De-emphasize
        - 1.1-2.0: Emphasize
        - Example: `(blue:0.3) and (green:1.8)` = more green than blue
        """)

    with tab2:
        st.markdown("""
        ### Style Guide
        Extract stylistic elements from an input image and apply them to a new creation.

        **Perfect for:**
        - Creating new images in the style of existing artwork
        - Maintaining brand visual consistency
        - Artistic style transfer with new content

        **Key Parameters:**
        - **Prompt:** Describe the new content you want (required)
        - **Fidelity:** How closely to match the input style (0-1, default 0.5)
        - **Aspect Ratio:** Control output dimensions
        - **Style Preset:** Additional style guidance

        **Image Requirements:**
        - Minimum side: 64px
        - Maximum pixels: 9,437,184
        - Aspect ratio: 1:2.5 to 2.5:1
        - Formats: JPEG, PNG, WebP

        **Aspect Ratios:**
        16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21
        """)

    with tab3:
        st.markdown("""
        ### Style Transfer
        Apply visual characteristics from a style image to a content image.

        **Perfect for:**
        - Consistent content across multiple assets
        - Applying artistic styles to photos
        - Transforming existing content while preserving composition

        **Key Parameters:**
        - **Composition Fidelity:** Preserve original composition (0-1, default 0.9)
        - **Style Strength:** How much style to apply (0-1, default 1.0)
        - **Change Strength:** How much to change original (0.1-1, default 0.9)
        - **Prompt:** Optional guidance for the transformation

        **Image Requirements (both images):**
        - Minimum side: 64px
        - Maximum pixels: 9,437,184
        - Aspect ratio: 1:2.5 to 2.5:1
        - Formats: JPEG, PNG, WebP

        **Difference from Style Guide:**
        - Style Guide: Creates NEW content in a style
        - Style Transfer: Transforms EXISTING content with a style
        """)

    with tab4:
        st.markdown("""
        ### 💡 Tips & Tricks

        **Writing Better Prompts:**
        - Be specific and descriptive
        - Include colors, subjects, and elements clearly
        - Use prompt weighting for fine control: `(word:weight)`
        - Experiment with negative prompts to avoid unwanted elements

        **Avoiding Payload Size Errors:**
        - ✅ Use JPEG format (most compressed)
        - ✅ Use WebP for good quality and compression
        - ⚠️ Avoid PNG for large outputs
        - ✅ Try different seeds if you hit size limits

        **Optimizing Results:**
        - **Control Structure:** Higher control_strength = more faithful to structure
        - **Style Guide:** Higher fidelity = closer to reference style
        - **Style Transfer:** Adjust composition_fidelity to balance preservation vs transformation

        **Style Presets:**
        Experiment with presets like:
        - `photographic` - Realistic photos
        - `anime` - Anime/manga style
        - `digital-art` - Digital artwork
        - `cinematic` - Movie-like quality
        - `enhance` - General enhancement

        **Seed Usage:**
        - Use seed=0 for random results
        - Save successful seeds for reproducibility
        - Try different seeds if results aren't satisfactory

        **Image Quality:**
        - Start with high-quality input images
        - Ensure good lighting and clarity
        - Avoid heavily compressed or low-resolution sources
        """)

st.caption("Powered by Stability AI via Amazon Bedrock | ⚠️ Max response size: 16MB")
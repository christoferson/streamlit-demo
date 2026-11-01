import streamlit as st
import boto3
import json
import logging
import base64
import io
from PIL import Image, ImageDraw
from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime
import numpy as np

# Configuration
AWS_REGION = "us-east-1"  # Update with your region

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

####################################################################################

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

mime_mapping = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP"
}

#####################

st.set_page_config(
    page_title="Stability AI Edit Tools",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.title("✂️ Stability AI Image Edit Tools")
st.markdown("Professional image editing powered by AI")

# Edit tool configurations
edit_tools = {
    "Remove Background": {
        "model_id": "us.stability.stable-image-remove-background-v1:0",
        "description": "Isolate subjects from the background with precision.",
        "icon": "🎭",
        "min_pixels": 4096,
        "max_pixels": 9437184,
        "min_side": 64,
        "max_aspect_ratio": 2.5,
        "requires_mask": False,
        "requires_prompt": False,
        "supports_output_format": True,
        "supports_seed": False,
        "supports_style_preset": False,
        "supports_negative_prompt": False,
        "supports_grow_mask": False,
        "available_fields": ["image", "output_format"]
    },
    "Inpaint": {
        "model_id": "us.stability.stable-image-inpaint-v1:0",
        "description": "Fill in or replace specified areas with new content based on a mask.",
        "icon": "🖌️",
        "min_pixels": 4096,
        "max_pixels": 9437184,
        "min_side": 64,
        "max_aspect_ratio": 2.5,
        "requires_mask": True,
        "requires_prompt": True,
        "supports_output_format": True,
        "supports_seed": True,
        "supports_style_preset": True,
        "supports_negative_prompt": True,
        "supports_grow_mask": True,
        "available_fields": ["image", "mask", "prompt", "negative_prompt", "style_preset", "seed", "output_format", "grow_mask"]
    },
    "Outpaint": {
        "model_id": "us.stability.stable-outpaint-v1:0",
        "description": "Extend images beyond their original boundaries in any direction.",
        "icon": "🖼️",
        "min_pixels": 4096,
        "max_pixels": 9437184,
        "min_side": 64,
        "max_aspect_ratio": 2.5,
        "requires_mask": False,
        "requires_prompt": False,
        "supports_output_format": True,
        "supports_seed": True,
        "supports_style_preset": True,
        "supports_negative_prompt": False,
        "supports_prompt": True,
        "supports_creativity": True,
        "supports_directions": True,
        "available_fields": ["image", "prompt", "style_preset", "seed", "output_format", "creativity", "left", "right", "up", "down"]
    },
    "Search and Recolor": {
        "model_id": "us.stability.stable-image-search-recolor-v1:0",
        "description": "Change the color of specific objects using prompts without a mask.",
        "icon": "🎨",
        "min_pixels": 4096,
        "max_pixels": 9437184,
        "min_side": 64,
        "max_aspect_ratio": 2.5,
        "requires_mask": False,
        "requires_prompt": True,
        "requires_select_prompt": True,
        "supports_output_format": True,
        "supports_seed": True,
        "supports_style_preset": True,
        "supports_negative_prompt": True,
        "supports_grow_mask": True,
        "available_fields": ["image", "prompt", "select_prompt", "negative_prompt", "style_preset", "seed", "output_format", "grow_mask"]
    },
    "Search and Replace": {
        "model_id": "us.stability.stable-image-search-replace-v1:0",
        "description": "Replace objects in images using search prompts without a mask.",
        "icon": "🔄",
        "min_pixels": 4096,
        "max_pixels": 9437184,
        "min_side": 64,
        "max_aspect_ratio": 2.5,
        "requires_mask": False,
        "requires_prompt": True,
        "requires_search_prompt": True,
        "supports_output_format": True,
        "supports_seed": True,
        "supports_style_preset": True,
        "supports_negative_prompt": True,
        "supports_grow_mask": True,
        "available_fields": ["image", "prompt", "search_prompt", "negative_prompt", "style_preset", "seed", "output_format", "grow_mask"]
    },
    "Erase": {
        "model_id": "us.stability.stable-image-erase-object-v1:0",
        "description": "Remove unwanted elements using masks while maintaining background consistency.",
        "icon": "🧹",
        "min_pixels": 4096,
        "max_pixels": 9437184,
        "min_side": 64,
        "max_aspect_ratio": 2.5,
        "requires_mask": True,
        "requires_prompt": False,
        "supports_output_format": True,
        "supports_seed": True,
        "supports_style_preset": False,
        "supports_negative_prompt": False,
        "supports_grow_mask": True,
        "available_fields": ["image", "mask", "seed", "output_format", "grow_mask"]
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

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Edit Tool Settings")

    # Tool selection
    selected_tool = st.selectbox(
        "Select Edit Tool",
        options=list(edit_tools.keys()),
        index=0,
        help="Choose the editing operation you want to perform"
    )

    tool_config = edit_tools[selected_tool]

    # Display tool info
    st.markdown(f"### {tool_config['icon']} {selected_tool}")
    st.info(tool_config["description"])

    # Show available fields
    with st.expander("📋 Available Parameters"):
        st.write("This tool supports:")
        for field in tool_config["available_fields"]:
            st.write(f"✅ {field}")

    st.divider()

    # Output format
    if tool_config.get("supports_output_format", False):
        st.markdown("##### 📤 Output Settings")
        default_format = 0 if selected_tool == "Remove Background" else 0
        output_format = st.selectbox(
            "Output Format",
            options=output_formats,
            index=default_format,
            help="PNG recommended for transparency"
        )
    else:
        output_format = "png"

    st.divider()

    # Prompt (if required or supported)
    if tool_config.get("requires_prompt", False) or tool_config.get("supports_prompt", False):
        st.markdown("##### 📝 Prompts")

        if selected_tool == "Inpaint":
            prompt = st.text_area(
                "Prompt *",
                value="",
                placeholder="Describe what you want to see in the inpainted area",
                height=100,
                max_chars=10000,
                help="What you wish to see in the output image"
            )
        elif selected_tool == "Outpaint":
            prompt = st.text_area(
                "Prompt (Optional)",
                value="",
                placeholder="Describe the scene to guide outpainting",
                height=100,
                max_chars=10000,
                help="Optional: Guide the outpainting process"
            )
        elif selected_tool == "Search and Recolor":
            prompt = st.text_area(
                "Color Prompt *",
                value="",
                placeholder="e.g., 'pink jacket', 'blue car'",
                height=80,
                max_chars=10000,
                help="Describe the new color/appearance"
            )
        elif selected_tool == "Search and Replace":
            prompt = st.text_area(
                "Replacement Prompt *",
                value="",
                placeholder="e.g., 'jacket', 'sunglasses'",
                height=80,
                max_chars=10000,
                help="What to replace the object with"
            )
        else:
            prompt = None
    else:
        prompt = None

    # Select prompt (for Search and Recolor)
    if tool_config.get("requires_select_prompt", False):
        select_prompt = st.text_input(
            "Select Prompt *",
            value="",
            placeholder="e.g., 'jacket', 'car', 'wall'",
            max_chars=10000,
            help="Short description of what to search for"
        )
    else:
        select_prompt = None

    # Search prompt (for Search and Replace)
    if tool_config.get("requires_search_prompt", False):
        search_prompt = st.text_input(
            "Search Prompt *",
            value="",
            placeholder="e.g., 'sweater', 'hat', 'background'",
            max_chars=10000,
            help="What to search for and replace"
        )
    else:
        search_prompt = None

    # Negative prompt
    if tool_config.get("supports_negative_prompt", False):
        negative_prompt = st.text_area(
            "Negative Prompt (Optional)",
            value="",
            placeholder="Things you don't want to see",
            height=80,
            max_chars=10000,
            help="Describe what you don't want to see"
        )
    else:
        negative_prompt = None

    st.divider()

    # Style preset
    if tool_config.get("supports_style_preset", False):
        st.markdown("##### 🎨 Style")
        style_preset = st.selectbox(
            "Style Preset (Optional)",
            options=style_presets,
            index=0,
            help="Guide the image towards a particular style"
        )
    else:
        style_preset = None

    # Creativity (for Outpaint)
    if tool_config.get("supports_creativity", False):
        st.markdown("##### 🎨 Creativity")
        creativity = st.slider(
            "Creativity",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher values = more creative content"
        )
    else:
        creativity = None

    # Outpaint directions
    if tool_config.get("supports_directions", False):
        st.markdown("##### 📐 Outpaint Directions")
        st.caption("At least one direction must be > 0")

        col_lr, col_ud = st.columns(2)
        with col_lr:
            left = st.number_input("Left (px)", min_value=0, max_value=2000, value=0, step=50)
            right = st.number_input("Right (px)", min_value=0, max_value=2000, value=0, step=50)
        with col_ud:
            up = st.number_input("Up (px)", min_value=0, max_value=2000, value=0, step=50)
            down = st.number_input("Down (px)", min_value=0, max_value=2000, value=0, step=50)
    else:
        left = right = up = down = None

    st.divider()

    # Grow mask
    if tool_config.get("supports_grow_mask", False):
        st.markdown("##### 🔍 Mask Settings")
        grow_mask = st.slider(
            "Grow Mask",
            min_value=0,
            max_value=20,
            value=5,
            help="Expands mask edges for smoother transitions"
        )
    else:
        grow_mask = None

    # Seed
    if tool_config.get("supports_seed", False):
        st.markdown("##### 🎲 Randomization")
        seed = st.number_input(
            "Seed (0 for random)",
            min_value=0,
            max_value=4294967294,
            value=0,
            help="Use a specific seed for reproducible results"
        )
    else:
        seed = None

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image to edit",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="main_image",
        help=f"Min side: {tool_config.get('min_side', 64)}px"
    )

    # Mask upload (if required)
    mask_file = None
    if tool_config.get("requires_mask", False):
        st.markdown("---")
        st.markdown("##### 🎭 Mask Image")
        st.caption("Black = no change, White = full effect")

        mask_file = st.file_uploader(
            "Upload mask (or use alpha channel)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key="mask_image",
            help="Black and white image defining the edit area"
        )

        # Option to draw mask
        if uploaded_file and not mask_file:
            st.info("💡 Tip: You can also use an image with an alpha channel as the mask")

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        original_width, original_height = uploaded_image.size
        total_pixels = original_width * original_height

        st.image(uploaded_image, caption=f"Original Image ({original_width}x{original_height})", use_container_width=True)

        # Show mask if uploaded
        if mask_file:
            mask_image = Image.open(mask_file)
            st.image(mask_image, caption=f"Mask Image", use_container_width=True)

        # Validation
        validation_errors = []
        validation_warnings = []

        # Check minimum side
        if "min_side" in tool_config:
            if original_width < tool_config["min_side"] or original_height < tool_config["min_side"]:
                validation_errors.append(f"❌ Image sides must be at least {tool_config['min_side']}px")

        # Check pixel count
        if "min_pixels" in tool_config and total_pixels < tool_config["min_pixels"]:
            validation_errors.append(f"❌ Total pixels must be at least {tool_config['min_pixels']:,}")

        if "max_pixels" in tool_config and total_pixels > tool_config["max_pixels"]:
            validation_errors.append(f"❌ Total pixels must not exceed {tool_config['max_pixels']:,}")

        # Check aspect ratio
        if "max_aspect_ratio" in tool_config:
            aspect_ratio = max(original_width, original_height) / min(original_width, original_height)
            if aspect_ratio > tool_config["max_aspect_ratio"]:
                validation_errors.append(f"❌ Aspect ratio must be between 1:{tool_config['max_aspect_ratio']} and {tool_config['max_aspect_ratio']}:1")

        # Check required fields
        if tool_config.get("requires_prompt") and not prompt:
            validation_errors.append("❌ Prompt is required for this tool")

        if tool_config.get("requires_select_prompt") and not select_prompt:
            validation_errors.append("❌ Select prompt is required")

        if tool_config.get("requires_search_prompt") and not search_prompt:
            validation_errors.append("❌ Search prompt is required")

        if tool_config.get("requires_mask") and not mask_file:
            # Check if image has alpha channel
            if uploaded_image.mode not in ('RGBA', 'LA') and not (uploaded_image.mode == 'P' and 'transparency' in uploaded_image.info):
                validation_errors.append("❌ Mask image required (or use image with alpha channel)")

        # Check outpaint directions
        if tool_config.get("supports_directions"):
            if left == 0 and right == 0 and up == 0 and down == 0:
                validation_errors.append("❌ At least one outpaint direction must be > 0")

        # Display validation results
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            can_process = False
        else:
            st.success("✅ Ready to process")
            can_process = True

            for warning in validation_warnings:
                st.warning(warning)

        # Display image info
        with st.expander("📊 Image Information"):
            st.write(f"**Dimensions:** {original_width} x {original_height}")
            st.write(f"**Total Pixels:** {total_pixels:,}")
            st.write(f"**Aspect Ratio:** {original_width/original_height:.2f}:1")
            st.write(f"**Format:** {uploaded_file.type}")
            st.write(f"**File Size:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")

            if uploaded_image.mode in ('RGBA', 'LA') or (uploaded_image.mode == 'P' and 'transparency' in uploaded_image.info):
                st.write(f"**Has Alpha Channel:** ✅ Yes")
            else:
                st.write(f"**Has Alpha Channel:** ❌ No")

with col2:
    st.subheader("✨ Edited Result")

    if uploaded_file and can_process:
        process_button = st.button(
            f"{tool_config['icon']} Process Image", 
            type="primary", 
            use_container_width=True
        )

        if process_button:
            with st.spinner(f"Processing with {selected_tool}... This may take a moment."):
                try:
                    # Convert image to base64
                    file_type = uploaded_file.type.split('/')[-1]
                    if file_type == 'jpg':
                        file_type = 'jpeg'

                    image_base64 = image_to_base64(uploaded_image, file_type.upper())

                    # Build request parameters
                    params = {
                        "image": image_base64
                    }

                    # Add mask if provided
                    if mask_file:
                        mask_type = mask_file.type.split('/')[-1]
                        if mask_type == 'jpg':
                            mask_type = 'jpeg'
                        mask_img = Image.open(mask_file)
                        mask_base64 = image_to_base64(mask_img, mask_type.upper())
                        params["mask"] = mask_base64

                    # Add tool-specific parameters
                    if prompt:
                        params["prompt"] = prompt

                    if select_prompt:
                        params["select_prompt"] = select_prompt

                    if search_prompt:
                        params["search_prompt"] = search_prompt

                    if negative_prompt and negative_prompt.strip():
                        params["negative_prompt"] = negative_prompt

                    if style_preset and style_preset != "None":
                        params["style_preset"] = style_preset

                    if creativity is not None:
                        params["creativity"] = creativity

                    if seed is not None and seed > 0:
                        params["seed"] = seed

                    if grow_mask is not None:
                        params["grow_mask"] = grow_mask

                    if output_format:
                        params["output_format"] = output_format

                    # Add outpaint directions
                    if left is not None:
                        params["left"] = left
                    if right is not None:
                        params["right"] = right
                    if up is not None:
                        params["up"] = up
                    if down is not None:
                        params["down"] = down

                    # Log request
                    logger.info(f"Processing with tool: {selected_tool}")
                    logger.info(f"Model ID: {tool_config['model_id']}")
                    logger.info(f"Parameters: {json.dumps({k: v for k, v in params.items() if k not in ['image', 'mask']}, indent=2)}")

                    # Display request info
                    with st.expander("🔍 Request Parameters", expanded=False):
                        st.json({k: v for k, v in params.items() if k not in ['image', 'mask']})

                    # Make API call
                    start_time = datetime.now()

                    response = bedrock_runtime.invoke_model(
                        modelId=tool_config["model_id"],
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
                        # Get processed image
                        processed_image_base64 = response_body["images"][0]
                        processed_image = base64_to_image(processed_image_base64)
                        processed_width, processed_height = processed_image.size

                        # Calculate output size
                        actual_size_kb = get_image_size_kb(processed_image, output_format)
                        actual_size_mb = actual_size_kb / 1024

                        # Display result with transparency support
                        if processed_image.mode in ('RGBA', 'LA'):
                            # Create checkered background
                            checker_size = 20
                            checker = Image.new('RGB', (processed_width, processed_height), 'white')
                            for i in range(0, processed_width, checker_size):
                                for j in range(0, processed_height, checker_size):
                                    if (i // checker_size + j // checker_size) % 2:
                                        checker.paste((200, 200, 200), (i, j, i + checker_size, j + checker_size))

                            display_image = Image.alpha_composite(checker.convert('RGBA'), processed_image)
                            st.image(display_image, caption=f"Processed Image ({processed_width}x{processed_height})", use_container_width=True)
                        else:
                            st.image(processed_image, caption=f"Processed Image ({processed_width}x{processed_height})", use_container_width=True)

                        # Success metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        with col_b:
                            st.metric("Output Size", f"{actual_size_mb:.2f} MB")
                        with col_c:
                            has_transparency = processed_image.mode in ('RGBA', 'LA')
                            st.metric("Transparency", "✅ Yes" if has_transparency else "❌ No")

                        st.success(f"✅ Processing completed successfully!")

                        # Download button
                        buffer = io.BytesIO()
                        processed_image.save(buffer, format=output_format.upper())
                        buffer.seek(0)

                        st.download_button(
                            label=f"⬇️ Download as {output_format.upper()}",
                            data=buffer,
                            file_name=f"{selected_tool.lower().replace(' ', '_')}_{uploaded_file.name.rsplit('.', 1)[0]}.{output_format}",
                            mime=f"image/{output_format}",
                            use_container_width=True
                        )

                        # Display metadata
                        with st.expander("📋 Processing Details"):
                            st.write(f"**Tool:** {selected_tool}")
                            st.write(f"**Model ID:** {tool_config['model_id']}")
                            st.write(f"**Processing Time:** {processing_time:.2f} seconds")
                            if "seeds" in response_body:
                                st.write(f"**Seed Used:** {response_body['seeds'][0]}")
                            if "finish_reasons" in response_body:
                                st.write(f"**Finish Reason:** {response_body['finish_reasons'][0]}")
                            st.write(f"**Output Format:** {output_format}")
                            st.write(f"**Output File Size:** {actual_size_kb:.2f} KB ({actual_size_mb:.2f} MB)")
                            st.write(f"**Output Dimensions:** {processed_width}x{processed_height}")

                        # Side-by-side comparison
                        with st.expander("🔄 Side-by-Side Comparison", expanded=True):
                            comp_col1, comp_col2 = st.columns(2)
                            with comp_col1:
                                st.image(uploaded_image, caption="Original", use_container_width=True)
                            with comp_col2:
                                if processed_image.mode in ('RGBA', 'LA'):
                                    st.image(display_image, caption="Processed", use_container_width=True)
                                else:
                                    st.image(processed_image, caption="Processed", use_container_width=True)

                except ClientError as err:
                    error_message = err.response["Error"]["Message"]
                    logger.error("A client error occurred: %s", error_message)

                    try:
                        error_detail = json.loads(error_message)
                        detail_msg = error_detail.get("detail", error_message)
                    except:
                        detail_msg = error_message

                    st.error(f"❌ AWS Error: {detail_msg}")

                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}")
                    st.error(f"❌ Error: {str(e)}")

    elif uploaded_file and not can_process:
        st.warning("⚠️ Please fix the validation errors before processing")
    else:
        st.info("👈 Upload an image to get started")

# Footer with information
st.divider()

with st.expander("ℹ️ About Stability AI Edit Tools"):

    tabs = st.tabs(["Remove Background", "Inpaint", "Outpaint", "Search & Recolor", "Search & Replace", "Erase"])

    with tabs[0]:
        st.markdown("""
        ### 🎭 Remove Background
        Isolate subjects from the background with precision.

        **Use Cases:**
        - Product photography
        - Portrait editing
        - E-commerce listings
        - Graphic design assets

        **Parameters:**
        - ✅ image (required)
        - ✅ output_format (optional)
        """)

    with tabs[1]:
        st.markdown("""
        ### 🖌️ Inpaint
        Fill in or replace specified areas with new content based on a mask.

        **Use Cases:**
        - Remove unwanted objects
        - Add new elements
        - Fix imperfections
        - Creative editing

        **Parameters:**
        - ✅ image (required)
        - ✅ mask (required or use alpha channel)
        - ✅ prompt (required)
        - ✅ negative_prompt (optional)
        - ✅ style_preset (optional)
        - ✅ seed (optional)
        - ✅ grow_mask (optional, 0-20)
        - ✅ output_format (optional)
        """)

    with tabs[2]:
        st.markdown("""
        ### 🖼️ Outpaint
        Extend images beyond their original boundaries in any direction.

        **Use Cases:**
        - Expand canvas
        - Change aspect ratio
        - Add context
        - Create panoramas

        **Parameters:**
        - ✅ image (required)
        - ✅ left, right, up, down (at least one > 0)
        - ✅ prompt (optional)
        - ✅ style_preset (optional)
        - ✅ creativity (optional, 0.1-1.0)
        - ✅ seed (optional)
        - ✅ output_format (optional)
        """)

    with tabs[3]:
        st.markdown("""
        ### 🎨 Search and Recolor
        Change the color of specific objects using prompts without a mask.

        **Use Cases:**
        - Product color variations
        - Fashion editing
        - Design exploration
        - Quick color changes

        **Parameters:**
        - ✅ image (required)
        - ✅ prompt (required) - new color/appearance
        - ✅ select_prompt (required) - what to recolor
        - ✅ negative_prompt (optional)
        - ✅ style_preset (optional)
        - ✅ seed (optional)
        - ✅ grow_mask (optional)
        - ✅ output_format (optional)
        """)

    with tabs[4]:
        st.markdown("""
        ### 🔄 Search and Replace
        Replace objects in images using search prompts without a mask.

        **Use Cases:**
        - Object replacement
        - Product variations
        - Scene modification
        - Creative transformations

        **Parameters:**
        - ✅ image (required)
        - ✅ prompt (required) - what to add
        - ✅ search_prompt (required) - what to replace
        - ✅ negative_prompt (optional)
        - ✅ style_preset (optional)
        - ✅ seed (optional)
        - ✅ grow_mask (optional)
        - ✅ output_format (optional)
        """)

    with tabs[5]:
        st.markdown("""
        ### 🧹 Erase
        Remove unwanted elements using masks while maintaining background consistency.

        **Use Cases:**
        - Remove objects
        - Clean up photos
        - Simplify compositions
        - Fix distractions

        **Parameters:**
        - ✅ image (required)
        - ✅ mask (required or use alpha channel)
        - ✅ seed (optional)
        - ✅ grow_mask (optional, 0-20)
        - ✅ output_format (optional)
        """)

st.caption("Powered by Stability AI via Amazon Bedrock")
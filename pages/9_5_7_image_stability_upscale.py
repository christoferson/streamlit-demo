import streamlit as st
import boto3
import json
import logging
import base64
import io
from PIL import Image
from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime

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

def estimate_output_size(width, height, format_type):
    """Estimate output size in MB based on dimensions and format"""
    pixels = width * height
    if format_type.lower() == 'png':
        # PNG is typically larger, estimate ~4 bytes per pixel
        estimated_bytes = pixels * 4
    elif format_type.lower() == 'jpeg':
        # JPEG is compressed, estimate ~0.5-1 bytes per pixel
        estimated_bytes = pixels * 0.75
    else:  # webp
        # WebP is efficient, estimate ~0.3-0.7 bytes per pixel
        estimated_bytes = pixels * 0.5

    return estimated_bytes / (1024 * 1024)  # Convert to MB

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
    page_title="Stability AI Upscale",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.title("🔍 Stability AI Image Upscale Services")
st.markdown("Enhance your images with AI-powered upscaling technology")

# Model options
upscale_models = {
    "Creative Upscale": {
        "model_id": "us.stability.stable-creative-upscale-v1:0",
        "description": "Upscales images to 4K (20-40x) with creative enhancement. Best for highly degraded images.",
        "min_pixels": 4096,
        "max_pixels": 1048576,
        "min_side": 64,
        "supports_prompt": True,
        "supports_creativity": True,
        "typical_upscale": "20-40x"
    },
    "Conservative Upscale": {
        "model_id": "us.stability.stable-conservative-upscale-v1:0",
        "description": "Upscales images to 4K (20-40x) while preserving original details. Minimal alterations.",
        "min_pixels": 4096,
        "max_pixels": 9437184,
        "min_side": 64,
        "supports_prompt": True,
        "supports_creativity": True,
        "typical_upscale": "20-40x"
    },
    "Fast Upscale": {
        "model_id": "us.stability.stable-fast-upscale-v1:0",
        "description": "Quick 4x upscale using predictive AI. Ideal for social media and compressed images.",
        "min_pixels": 1024,
        "max_pixels": 1048576,
        "min_side": 32,
        "max_side": 1536,
        "supports_prompt": False,
        "supports_creativity": False,
        "typical_upscale": "4x"
    }
}

output_formats = ["jpeg", "webp", "png"]  # Reordered with JPEG first (most compressed)

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
    st.header("⚙️ Upscale Settings")

    selected_model = st.selectbox(
        "Upscale Model",
        options=list(upscale_models.keys()),
        index=0,
        help="Choose the upscaling model based on your needs"
    )

    model_config = upscale_models[selected_model]

    st.info(model_config["description"])

    st.divider()

    # Output format - moved to top with warning
    output_format = st.selectbox(
        "Output Format",
        options=output_formats,
        index=0,
        help="⚠️ Use JPEG or WebP to avoid payload size errors. PNG creates larger files."
    )

    if output_format == "png":
        st.warning("⚠️ PNG format may exceed 16MB limit for large upscales. Consider JPEG or WebP.")

    st.divider()

    # Prompt (only for Creative and Conservative)
    if model_config["supports_prompt"]:
        prompt = st.text_area(
            "Prompt",
            value="A highly detailed, photorealistic image",
            height=100,
            max_chars=10000,
            help="Describe what you wish to see in the upscaled image"
        )

        negative_prompt = st.text_area(
            "Negative Prompt (Optional)",
            value="blurry, low quality, distorted, artifacts",
            height=80,
            max_chars=10000,
            help="Describe what you don't want to see"
        )

    # Creativity (only for Creative and Conservative)
    if model_config["supports_creativity"]:
        creativity = st.slider(
            "Creativity",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="Higher values add more creative details during upscaling"
        )

    # Style preset (only for Creative and Conservative)
    if model_config["supports_prompt"]:
        style_preset = st.selectbox(
            "Style Preset (Optional)",
            options=style_presets,
            index=0,
            help="Guide the image towards a particular style"
        )

    # Seed
    seed = st.number_input(
        "Seed (0 for random)",
        min_value=0,
        max_value=4294967294,
        value=0,
        help="Use a specific seed for reproducible results. Try different seeds if you get size errors."
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image to upscale",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        help=f"Min side: {model_config['min_side']}px, Pixel range: {model_config['min_pixels']}-{model_config['max_pixels']}"
    )

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        original_width, original_height = uploaded_image.size
        total_pixels = original_width * original_height

        st.image(uploaded_image, caption=f"Original Image ({original_width}x{original_height}, {total_pixels:,} pixels)", use_container_width=True)

        # Validation
        validation_errors = []
        validation_warnings = []

        if original_width < model_config["min_side"] or original_height < model_config["min_side"]:
            validation_errors.append(f"❌ Image sides must be at least {model_config['min_side']}px")

        if "max_side" in model_config:
            if original_width > model_config["max_side"] or original_height > model_config["max_side"]:
                validation_errors.append(f"❌ Image sides must not exceed {model_config['max_side']}px")

        if total_pixels < model_config["min_pixels"]:
            validation_errors.append(f"❌ Total pixels must be at least {model_config['min_pixels']:,}")

        if total_pixels > model_config["max_pixels"]:
            validation_errors.append(f"❌ Total pixels must not exceed {model_config['max_pixels']:,}")

        # Check aspect ratio for Conservative Upscale
        if selected_model == "Conservative Upscale":
            aspect_ratio = max(original_width, original_height) / min(original_width, original_height)
            if aspect_ratio > 2.5:
                validation_errors.append("❌ Aspect ratio must be between 1:2.5 and 2.5:1")

        # Estimate output size and warn if potentially too large
        if selected_model in ["Creative Upscale", "Conservative Upscale"]:
            # These typically upscale to 4K
            estimated_width = min(3840, original_width * 30)  # Rough estimate
            estimated_height = min(2160, original_height * 30)
        else:  # Fast Upscale
            estimated_width = original_width * 4
            estimated_height = original_height * 4

        estimated_size_mb = estimate_output_size(estimated_width, estimated_height, output_format)

        if estimated_size_mb > MAX_PAYLOAD_SIZE_MB:
            validation_warnings.append(
                f"⚠️ Estimated output size (~{estimated_size_mb:.1f}MB) may exceed 16MB limit. "
                f"Consider using JPEG format or a smaller input image."
            )
        elif estimated_size_mb > MAX_PAYLOAD_SIZE_MB * 0.8:
            validation_warnings.append(
                f"⚠️ Estimated output size (~{estimated_size_mb:.1f}MB) is close to 16MB limit. "
                f"If upscaling fails, try JPEG format or different seed."
            )

        if validation_errors:
            for error in validation_errors:
                st.error(error)
            can_upscale = False
        else:
            st.success("✅ Image meets requirements")
            can_upscale = True

            # Show warnings
            for warning in validation_warnings:
                st.warning(warning)

        # Display image info
        with st.expander("📊 Image Information"):
            st.write(f"**Dimensions:** {original_width} x {original_height}")
            st.write(f"**Total Pixels:** {total_pixels:,}")
            st.write(f"**Aspect Ratio:** {original_width/original_height:.2f}:1")
            st.write(f"**Format:** {uploaded_file.type}")
            st.write(f"**File Size:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")
            st.write(f"**Estimated Upscaled Size:** ~{estimated_width}x{estimated_height}")
            st.write(f"**Estimated Output Size ({output_format.upper()}):** ~{estimated_size_mb:.1f} MB")

            # Size comparison table
            st.write("**Format Size Estimates:**")
            for fmt in ["jpeg", "webp", "png"]:
                size = estimate_output_size(estimated_width, estimated_height, fmt)
                status = "✅" if size < MAX_PAYLOAD_SIZE_MB else "❌"
                st.write(f"- {fmt.upper()}: ~{size:.1f} MB {status}")

with col2:
    st.subheader("✨ Upscaled Result")

    if uploaded_file and can_upscale:
        upscale_button = st.button("🚀 Upscale Image", type="primary", use_container_width=True)

        if upscale_button:
            with st.spinner(f"Upscaling with {selected_model}... This may take a moment."):
                try:
                    # Convert image to base64
                    file_type = uploaded_file.type.split('/')[-1]
                    if file_type == 'jpg':
                        file_type = 'jpeg'

                    image_base64 = image_to_base64(uploaded_image, file_type.upper())

                    # Build request parameters
                    params = {
                        "image": image_base64,
                        "output_format": output_format
                    }

                    # Add model-specific parameters
                    if model_config["supports_prompt"]:
                        params["prompt"] = prompt
                        if negative_prompt.strip():
                            params["negative_prompt"] = negative_prompt
                        if style_preset != "None":
                            params["style_preset"] = style_preset

                    if model_config["supports_creativity"]:
                        params["creativity"] = creativity

                    if seed > 0:
                        params["seed"] = seed

                    # Log request
                    logger.info(f"Upscaling with model: {model_config['model_id']}")
                    logger.info(f"Parameters: {json.dumps({k: v for k, v in params.items() if k != 'image'}, indent=2)}")

                    # Make API call
                    start_time = datetime.now()

                    response = bedrock_runtime.invoke_model(
                        modelId=model_config["model_id"],
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
                        # Get upscaled image
                        upscaled_image_base64 = response_body["images"][0]
                        upscaled_image = base64_to_image(upscaled_image_base64)
                        upscaled_width, upscaled_height = upscaled_image.size

                        # Calculate actual output size
                        actual_size_kb = get_image_size_kb(upscaled_image, output_format)
                        actual_size_mb = actual_size_kb / 1024

                        # Display result
                        st.image(upscaled_image, caption=f"Upscaled Image ({upscaled_width}x{upscaled_height})", use_container_width=True)

                        # Success metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Original Size", f"{original_width}x{original_height}")
                        with col_b:
                            st.metric("Upscaled Size", f"{upscaled_width}x{upscaled_height}")
                        with col_c:
                            upscale_factor = (upscaled_width * upscaled_height) / (original_width * original_height)
                            st.metric("Upscale Factor", f"{upscale_factor:.1f}x")
                        with col_d:
                            st.metric("Output Size", f"{actual_size_mb:.1f} MB")

                        st.success(f"✅ Upscaling completed in {processing_time:.2f} seconds")

                        # Download button
                        buffer = io.BytesIO()
                        upscaled_image.save(buffer, format=output_format.upper())
                        buffer.seek(0)

                        st.download_button(
                            label="⬇️ Download Upscaled Image",
                            data=buffer,
                            file_name=f"upscaled_{uploaded_file.name.rsplit('.', 1)[0]}.{output_format}",
                            mime=f"image/{output_format}",
                            use_container_width=True
                        )

                        # Display metadata
                        with st.expander("📋 Processing Details"):
                            st.write(f"**Model:** {selected_model}")
                            st.write(f"**Processing Time:** {processing_time:.2f} seconds")
                            if "seeds" in response_body:
                                st.write(f"**Seed Used:** {response_body['seeds'][0]}")
                            if "finish_reasons" in response_body:
                                st.write(f"**Finish Reason:** {response_body['finish_reasons'][0]}")
                            st.write(f"**Output Format:** {output_format}")
                            st.write(f"**Output File Size:** {actual_size_kb:.2f} KB ({actual_size_mb:.2f} MB)")
                            if model_config["supports_creativity"]:
                                st.write(f"**Creativity:** {creativity}")

                except ClientError as err:
                    error_message = err.response["Error"]["Message"]
                    logger.error("A client error occurred: %s", error_message)

                    # Check if it's a payload size error
                    if "payload size" in error_message.lower() or "exceeds the maximum" in error_message.lower():
                        st.error("❌ **Payload Size Error**: The upscaled image is too large (>16MB)")
                        st.info("""
                        **Solutions to try:**
                        1. ✅ Change output format to **JPEG** (most compressed)
                        2. ✅ Try a different **seed** value (randomness affects compression)
                        3. ✅ Use a smaller input image
                        4. ✅ Try **Fast Upscale** instead (4x vs 20-40x)
                        5. ✅ Use **WebP** format (good compression, smaller than PNG)
                        """)
                    else:
                        st.error(f"❌ AWS Error: {error_message}")

                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}")
                    st.error(f"❌ Error: {str(e)}")

    elif uploaded_file and not can_upscale:
        st.warning("⚠️ Please fix the validation errors before upscaling")
    else:
        st.info("👈 Upload an image to get started")

# Footer with information
st.divider()

with st.expander("ℹ️ About Stability AI Upscale Services & Payload Limits"):
    st.markdown("""
    ### Creative Upscale
    - Upscales images to 4K resolution (20-40x)
    - Best for highly degraded images
    - Performs heavy reimagining with creative enhancements
    - Input: 64x64 to 1 megapixel

    ### Conservative Upscale
    - Upscales images to 4K resolution (20-40x)
    - Preserves all aspects of the original image
    - Minimizes alterations
    - Input: 64x64 to ~9.4 megapixels

    ### Fast Upscale
    - Quick 4x resolution enhancement
    - Lightweight and fast
    - Ideal for social media and compressed images
    - Input: 32-1536px per side, 1024 to 1 megapixel

    ### 🚨 Important: 16MB Payload Limit

    Amazon Bedrock has a **16MB maximum response size limit**. Large upscaled images may exceed this.

    **Solutions:**
    - ✅ **Use JPEG format** - Most compressed, typically 3-5x smaller than PNG
    - ✅ **Use WebP format** - Modern format with excellent compression
    - ✅ **Avoid PNG format** - Lossless format creates large files
    - ✅ **Try different seeds** - Different seeds produce different compression ratios
    - ✅ **Use smaller input images** - Smaller inputs = smaller outputs
    - ✅ **Use Fast Upscale** - 4x upscale instead of 20-40x

    **Format Comparison (for 4K image):**
    - PNG: ~15-25 MB ❌ (may exceed limit)
    - WebP: ~3-8 MB ✅
    - JPEG: ~2-5 MB ✅

    ### Tips for Best Results
    - Use **Creative Upscale** for old, low-quality, or heavily compressed images
    - Use **Conservative Upscale** when you want to maintain the original look
    - Use **Fast Upscale** for quick enhancements and social media content
    - Provide detailed prompts for Creative and Conservative upscaling
    - Adjust creativity slider to control the level of enhancement
    - **Always start with JPEG or WebP format** to avoid size issues
    """)

st.caption("Powered by Stability AI via Amazon Bedrock | ⚠️ Max response size: 16MB")
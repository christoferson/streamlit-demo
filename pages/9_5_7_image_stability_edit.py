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
        "description": "Isolate subjects from the background with precision. Perfect for product photos, portraits, and creating transparent backgrounds.",
        "icon": "🎭",
        "min_pixels": 4096,
        "max_pixels": 9437184,
        "min_side": 64,
        "max_aspect_ratio": 2.5,
        "supports_output_format": True,
        "available_fields": ["image", "output_format"],
        "use_cases": [
            "Product photography",
            "Portrait editing",
            "Creating transparent PNGs",
            "E-commerce listings",
            "Graphic design assets"
        ]
    },
    # Placeholder for future edit tools
    "Inpaint (Coming Soon)": {
        "model_id": None,
        "description": "Remove or replace objects within an image using AI-powered inpainting.",
        "icon": "🖌️",
        "available": False
    },
    "Outpaint (Coming Soon)": {
        "model_id": None,
        "description": "Extend images beyond their original boundaries with AI-generated content.",
        "icon": "🖼️",
        "available": False
    },
    "Search and Replace (Coming Soon)": {
        "model_id": None,
        "description": "Find and replace specific objects or elements in your images.",
        "icon": "🔍",
        "available": False
    }
}

output_formats = ["png", "jpeg", "webp"]

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Edit Tool Settings")

    # Tool selection
    available_tools = [tool for tool, config in edit_tools.items() if config.get("available", True)]
    selected_tool = st.selectbox(
        "Select Edit Tool",
        options=available_tools,
        index=0,
        help="Choose the editing operation you want to perform"
    )

    tool_config = edit_tools[selected_tool]

    # Display tool info
    st.markdown(f"### {tool_config['icon']} {selected_tool}")
    st.info(tool_config["description"])

    # Show available fields
    if "available_fields" in tool_config:
        with st.expander("📋 Available Parameters"):
            st.write("This tool supports:")
            for field in tool_config["available_fields"]:
                st.write(f"✅ {field}")

    st.divider()

    # Output format (if supported)
    if tool_config.get("supports_output_format", False):
        st.markdown("##### 📤 Output Settings")
        output_format = st.selectbox(
            "Output Format",
            options=output_formats,
            index=0,  # PNG default for transparency
            help="PNG recommended for transparent backgrounds"
        )

        if output_format != "png" and selected_tool == "Remove Background":
            st.warning("⚠️ PNG format recommended to preserve transparency")
    else:
        output_format = "png"

    st.divider()

    # Use cases
    if "use_cases" in tool_config:
        with st.expander("💡 Use Cases"):
            for use_case in tool_config["use_cases"]:
                st.write(f"• {use_case}")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image to edit",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        help=f"Min side: {tool_config.get('min_side', 64)}px"
    )

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        original_width, original_height = uploaded_image.size
        total_pixels = original_width * original_height

        st.image(uploaded_image, caption=f"Original Image ({original_width}x{original_height}, {total_pixels:,} pixels)", use_container_width=True)

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

        # Display validation results
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            can_process = False
        else:
            st.success("✅ Image meets requirements")
            can_process = True

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

            # Check if image has transparency
            if uploaded_image.mode in ('RGBA', 'LA') or (uploaded_image.mode == 'P' and 'transparency' in uploaded_image.info):
                st.write(f"**Has Transparency:** ✅ Yes")
            else:
                st.write(f"**Has Transparency:** ❌ No")

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

                    # Add output format if supported
                    if tool_config.get("supports_output_format", False):
                        params["output_format"] = output_format

                    # Log request
                    logger.info(f"Processing with tool: {selected_tool}")
                    logger.info(f"Model ID: {tool_config['model_id']}")
                    logger.info(f"Parameters: {json.dumps({k: v for k, v in params.items() if k != 'image'}, indent=2)}")

                    # Display request info
                    with st.expander("🔍 Request Parameters", expanded=False):
                        st.json({k: v for k, v in params.items() if k != 'image'})

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
                        # Create a checkered background for transparency visualization
                        if processed_image.mode in ('RGBA', 'LA'):
                            # Create checkered background
                            checker_size = 20
                            checker = Image.new('RGB', (processed_width, processed_height), 'white')
                            for i in range(0, processed_width, checker_size):
                                for j in range(0, processed_height, checker_size):
                                    if (i // checker_size + j // checker_size) % 2:
                                        checker.paste((200, 200, 200), (i, j, i + checker_size, j + checker_size))

                            # Composite the image over the checker
                            display_image = Image.alpha_composite(checker.convert('RGBA'), processed_image)
                            st.image(display_image, caption=f"Processed Image ({processed_width}x{processed_height}) - Transparency shown with checkered background", use_container_width=True)
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

                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)

                        with col_dl1:
                            # Download in selected format
                            buffer = io.BytesIO()
                            processed_image.save(buffer, format=output_format.upper())
                            buffer.seek(0)

                            st.download_button(
                                label=f"⬇️ Download as {output_format.upper()}",
                                data=buffer,
                                file_name=f"edited_{uploaded_file.name.rsplit('.', 1)[0]}.{output_format}",
                                mime=f"image/{output_format}",
                                use_container_width=True
                            )

                        with col_dl2:
                            # Always offer PNG download for transparency
                            if output_format != "png" and processed_image.mode in ('RGBA', 'LA'):
                                buffer_png = io.BytesIO()
                                processed_image.save(buffer_png, format='PNG')
                                buffer_png.seek(0)

                                st.download_button(
                                    label="⬇️ Download as PNG (with transparency)",
                                    data=buffer_png,
                                    file_name=f"edited_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                                    mime="image/png",
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
                            st.write(f"**Color Mode:** {processed_image.mode}")

                            st.write("**Parameters Used:**")
                            for key, value in params.items():
                                if key != "image":
                                    st.write(f"- {key}: {value}")

                        # Side-by-side comparison
                        with st.expander("🔄 Side-by-Side Comparison", expanded=True):
                            comp_col1, comp_col2 = st.columns(2)
                            with comp_col1:
                                st.image(uploaded_image, caption="Original", use_container_width=True)
                            with comp_col2:
                                if processed_image.mode in ('RGBA', 'LA'):
                                    st.image(display_image, caption="Processed (with transparency)", use_container_width=True)
                                else:
                                    st.image(processed_image, caption="Processed", use_container_width=True)

                except ClientError as err:
                    error_message = err.response["Error"]["Message"]
                    logger.error("A client error occurred: %s", error_message)

                    # Parse error details if available
                    try:
                        error_detail = json.loads(error_message)
                        detail_msg = error_detail.get("detail", error_message)
                    except:
                        detail_msg = error_message

                    st.error(f"❌ AWS Error: {detail_msg}")

                    # Provide helpful suggestions
                    if "aspect ratio" in detail_msg.lower():
                        st.info("💡 Try cropping your image to a more standard aspect ratio (closer to square)")
                    elif "pixel" in detail_msg.lower():
                        st.info("💡 Try resizing your image to meet the pixel requirements")

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

    tab1, tab2, tab3 = st.tabs(["Remove Background", "Best Practices", "Technical Details"])

    with tab1:
        st.markdown("""
        ### 🎭 Remove Background

        Isolate subjects from the background with AI-powered precision. This tool automatically detects the main subject in your image and removes the background, creating a transparent PNG.

        **Perfect For:**
        - 🛍️ Product photography for e-commerce
        - 👤 Portrait editing and profile pictures
        - 🎨 Creating graphic design assets
        - 📱 Social media content
        - 🖼️ Professional presentations

        **Technical Specifications:**
        - **Input Size:** 64px minimum per side
        - **Max Pixels:** 9,437,184 (~3072x3072)
        - **Aspect Ratio:** Between 1:2.5 and 2.5:1
        - **Supported Formats:** JPEG, PNG, WebP
        - **Output:** Transparent PNG (recommended) or JPEG/WebP

        **Tips for Best Results:**
        - Use high-contrast images where the subject is clearly distinguishable
        - Ensure good lighting on your subject
        - Avoid cluttered backgrounds when possible
        - Use PNG output format to preserve transparency
        - Higher resolution inputs generally produce better results
        """)

    with tab2:
        st.markdown("""
        ### 💡 Best Practices

        **Image Preparation:**
        1. **Resolution:** Use the highest quality image available
        2. **Lighting:** Ensure even lighting on the subject
        3. **Contrast:** Clear distinction between subject and background works best
        4. **Focus:** Subject should be in sharp focus
        5. **Framing:** Center your subject with some margin around edges

        **Output Format Selection:**
        - **PNG:** Best for transparency, larger file size
        - **WebP:** Good compression with transparency support
        - **JPEG:** Smallest file size, but no transparency (adds white background)

        **Common Use Cases:**

        **E-commerce:**
        - Product photos with clean backgrounds
        - Consistent white or transparent backgrounds
        - Multiple product angles

        **Marketing:**
        - Social media graphics
        - Advertisement materials
        - Promotional content

        **Professional:**
        - Headshots and portraits
        - Team photos
        - Corporate materials

        **Creative:**
        - Composite images
        - Graphic design elements
        - Digital art projects
        """)

    with tab3:
        st.markdown("""
        ### 🔧 Technical Details

        **Model Information:**
        - **Model ID:** `us.stability.stable-image-remove-background-v1:0`
        - **Provider:** Stability AI
        - **Platform:** Amazon Bedrock

        **API Parameters:**
        ```json
        {
            "image": "base64_encoded_image",
            "output_format": "png|jpeg|webp"
        }
        ```

        **Response Format:**
        ```json
        {
            "images": ["base64_encoded_result"],
            "seeds": [seed_value],
            "finish_reasons": [status]
        }
        ```

        **Constraints:**
        - Minimum side length: 64 pixels
        - Maximum total pixels: 9,437,184
        - Aspect ratio: 1:2.5 to 2.5:1
        - Supported input formats: JPEG, PNG, WebP
        - Supported output formats: JPEG, PNG, WebP

        **Performance:**
        - Typical processing time: 2-5 seconds
        - Depends on image size and complexity
        - Automatic subject detection
        - High-precision edge detection

        **Transparency Handling:**
        - Output preserves alpha channel in PNG/WebP
        - JPEG output adds white background (no transparency support)
        - Checkered background used for visualization in this app
        """)

st.caption("Powered by Stability AI via Amazon Bedrock")
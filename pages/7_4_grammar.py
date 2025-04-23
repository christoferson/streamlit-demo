import streamlit as st
import boto3
import settings
import json
import logging
import cmn_auth
import cmn_constants
from cmn.bedrock_models import FoundationModel

from botocore.exceptions import ClientError

AWS_REGION = settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analysis_stats' not in st.session_state:
    st.session_state.analysis_stats = None

st.markdown(cmn_constants.css_button_primary, unsafe_allow_html=True)


opt_model_id_list = [
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.nova-pro-v1:0",
]


system_message = """You are a text analysis assistant. When analyzing text:
1. Always respond in the same language as the input text
2. Maintain the writing style and tone appropriate for that language
3. Provide culturally appropriate suggestions
4. Use language-specific grammar and writing conventions
"""

def analyze_text(text: str) -> str:
    """
    Create a prompt that asks the LLM to analyze the text for improvements
    Ensures response is in the same language as input
    """
    analysis_prompt = f"""Analyze the following text and provide detailed feedback. 
IMPORTANT: Your response MUST be in the same language as the input text.

Analysis points:
1. Spelling and grammar
2. Clarity and readability
3. Structure and flow
4. Contradictions
5. Word choice
6. Overall coherence
7. Tone and voice
8. Cultural appropriateness
9. Technical formatting
10. Content depth
11. Redundancy
12. Active/passive voice
13. Conciseness
14. Terminology
15. Writing style

Text to analyze:
{text}

Remember: Provide your analysis in the SAME LANGUAGE as the input text."""

    return analysis_prompt

    
with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")

opt_fm:FoundationModel = FoundationModel.find(opt_model_id)

opt_fm_max_tokens = opt_fm.InferenceParameter.get("MaxTokensToSample")
opt_fm_top_k = opt_fm.InferenceParameter.get("TopK")

with st.sidebar:

    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value=system_message, key="system_msg")

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)



# "Text Analysis" (most aligned with your current naming pattern)
# "Text Review"
# "Text Improve"
# "Text Check"
# "Text Enhance"

# Main content
st.markdown("Text Analysis")

# Add this somewhere appropriate in your UI
if st.button("Clear All", type="secondary"):
    st.session_state.analysis_complete = False
    st.session_state.analysis_result = None
    st.session_state.analysis_stats = None
    st.rerun()

# Text input area
user_text = st.text_area(
    "Enter text to analyze",
    height=200,
    placeholder="Paste or type your text here..."
)

# Add analyze button
# Create parallel buttons
col1, col2 = st.columns(2)
with col1:
    analyze_button = st.button("Analyze Text", type="secondary")
with col2:
    quick_revise_button = st.button("Quick Revise", type="secondary")


if user_text and analyze_button:  # Only analyze when button is clicked
    with st.spinner('Analyzing text...'):
        try:
            # Create the analysis prompt
            prompt = analyze_text(user_text)

            # Prepare the message history
            message_history = []
            message_user_latest = {"role": "user", "content": [{"text": prompt}]}

            # Configure the model parameters
            system_prompts = [{"text": opt_system_msg}]
            inference_config = {
                "temperature": opt_temperature,
                "maxTokens": opt_max_tokens,
                "topP": opt_top_p,
            }

            # Add additional model fields if supported
            additional_model_fields = {}
            if opt_fm_top_k.isSupported():
                additional_model_fields[opt_fm_top_k.Name] = opt_top_k
            if not additional_model_fields:
                additional_model_fields = None

            # Make the API call
            response = bedrock_runtime.converse_stream(
                modelId=opt_model_id,
                messages=[message_user_latest],
                system=system_prompts,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields
            )

            # Process and display the response
            with st.container():
                st.subheader("Analysis Results")

                result_text = ""
                result_area = st.empty()

                for event in response.get('stream'):
                    if 'contentBlockDelta' in event:
                        text = event['contentBlockDelta']['delta']['text']
                        result_text += text
                        result_area.markdown(result_text)

                    # Handle metadata and usage statistics
                    if 'metadata' in event:
                        metadata = event['metadata']
                        if 'usage' in metadata:
                            input_token_count = metadata['usage']['inputTokens']
                            output_token_count = metadata['usage']['outputTokens']
                            total_token_count = metadata['usage']['totalTokens']

                            # Log usage statistics
                            st.caption(f"Analysis statistics: {total_token_count} total tokens used")

            st.session_state.analysis_result = result_text
            st.session_state.analysis_stats = f"Analysis statistics: {total_token_count} total tokens used"
            st.session_state.analysis_complete = True

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            logger.error(f"Analysis error: {str(e)}", exc_info=True)


# Display stored analysis results if they exist
if st.session_state.analysis_result:
    with st.container():
        st.subheader("Analysis Results")
        st.markdown(st.session_state.analysis_result)
        if st.session_state.analysis_stats:
            st.caption(st.session_state.analysis_stats)



# Handle Quick Revision
if user_text and quick_revise_button:
    with st.spinner('Applying quick revision...'):
        try:
            revision_prompt = f"""Please revise and improve the following text, applying necessary corrections for grammar, clarity, structure, and style. 
Provide ONLY the revised text without any explanations.

Text to revise:
{user_text}

Remember: Keep the response in the SAME LANGUAGE as the input text."""

            message_user_latest = {"role": "user", "content": [{"text": revision_prompt}]}

            # Configure the model parameters for revision
            revision_system_msg = """You are a text revision assistant. Your task is to:
1. Improve the given text while maintaining its original meaning
2. Always respond in the same language as the input text
3. Provide ONLY the revised text without explanations
4. Maintain appropriate tone and style for the context
"""
            system_prompts = [{"text": revision_system_msg}]
            inference_config = {
                "temperature": opt_temperature,
                "maxTokens": opt_max_tokens,
                "topP": opt_top_p,
            }

            # Add additional model fields if supported
            additional_model_fields = {}
            if opt_fm_top_k.isSupported():
                additional_model_fields[opt_fm_top_k.Name] = opt_top_k
            if not additional_model_fields:
                additional_model_fields = None

            # Make the API call
            response = bedrock_runtime.converse_stream(
                modelId=opt_model_id,
                messages=[message_user_latest],
                system=system_prompts,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields
            )

            # Process and display the revised text
            with st.container():
                st.subheader("Quick Revision")
                result_text = ""
                result_area = st.empty()

                for event in response.get('stream'):
                    if 'contentBlockDelta' in event:
                        text = event['contentBlockDelta']['delta']['text']
                        result_text += text
                        result_area.markdown(result_text)

                    # Handle metadata and usage statistics
                    if 'metadata' in event:
                        metadata = event['metadata']
                        if 'usage' in metadata:
                            input_token_count = metadata['usage']['inputTokens']
                            output_token_count = metadata['usage']['outputTokens']
                            total_token_count = metadata['usage']['totalTokens']

                            # Log usage statistics
                            st.caption(f"Quick revision statistics: {total_token_count} total tokens used")

        except Exception as e:
            st.error(f"An error occurred during quick revision: {str(e)}")
            logger.error(f"Quick revision error: {str(e)}", exc_info=True)

# Show sequential revision button after analysis
if st.session_state.analysis_complete:
    st.divider()
    if st.button("Apply Suggested Changes", type="primary", key="sequential_revise"):
        with st.spinner('Applying changes...'):
            try:
                revision_prompt = f"""Please revise and improve the following text, applying necessary corrections for grammar, clarity, structure, and style. 
Provide ONLY the revised text without any explanations.

Text to revise:
{user_text}

Remember: Keep the response in the SAME LANGUAGE as the input text."""

                message_user_latest = {"role": "user", "content": [{"text": revision_prompt}]}

                # Configure the model parameters for revision
                revision_system_msg = """You are a text revision assistant. Your task is to:
1. Improve the given text while maintaining its original meaning
2. Always respond in the same language as the input text
3. Provide ONLY the revised text without explanations
4. Maintain appropriate tone and style for the context
"""
                system_prompts = [{"text": revision_system_msg}]
                inference_config = {
                    "temperature": opt_temperature,
                    "maxTokens": opt_max_tokens,
                    "topP": opt_top_p,
                }

                # Add additional model fields if supported
                additional_model_fields = {}
                if opt_fm_top_k.isSupported():
                    additional_model_fields[opt_fm_top_k.Name] = opt_top_k
                if not additional_model_fields:
                    additional_model_fields = None

                # Make the API call
                response = bedrock_runtime.converse_stream(
                    modelId=opt_model_id,
                    messages=[message_user_latest],
                    system=system_prompts,
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields
                )

                # Process and display the revised text
                with st.container():
                    st.subheader("Revised Text")
                    result_text = ""
                    result_area = st.empty()

                    for event in response.get('stream'):
                        if 'contentBlockDelta' in event:
                            text = event['contentBlockDelta']['delta']['text']
                            result_text += text
                            result_area.markdown(result_text)

                        # Handle metadata and usage statistics
                        if 'metadata' in event:
                            metadata = event['metadata']
                            if 'usage' in metadata:
                                input_token_count = metadata['usage']['inputTokens']
                                output_token_count = metadata['usage']['outputTokens']
                                total_token_count = metadata['usage']['totalTokens']

                                # Log usage statistics
                                st.caption(f"Revision statistics: {total_token_count} total tokens used")

            except Exception as e:
                st.error(f"An error occurred during revision: {str(e)}")
                logger.error(f"Revision error: {str(e)}", exc_info=True)

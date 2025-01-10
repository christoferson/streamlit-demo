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


st.markdown(cmn_constants.css_button_primary, unsafe_allow_html=True)


opt_model_id_list = [
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.nova-pro-v1:0",
]

system_message = """Your task is to take the text provided and rewrite it into a clear, grammatically correct version while preserving the original meaning as closely as possible. Correct any spelling mistakes, punctuation errors, verb tense issues, word choice problems, and other grammatical mistakes.
"""



    
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

def analyze_text(text: str) -> str:
    """
    Create a prompt that asks the LLM to analyze the text for improvements
    """
    analysis_prompt = f"""Please analyze the following text and provide detailed feedback on:
1. Spelling and grammar errors
2. Clarity and readability
3. Potential improvements in structure and flow
4. Any contradicting statements
5. Suggestions for better word choices
6. Overall coherence and effectiveness

Text to analyze:
{text}

Please provide your analysis in a well-structured format, maintaining the same language as the input text."""

    return analysis_prompt


# "Text Analysis" (most aligned with your current naming pattern)
# "Text Review"
# "Text Improve"
# "Text Check"
# "Text Enhance"

# Main content
st.markdown("Text Analysis")

# Text input area
user_text = st.text_area(
    "Enter text to analyze",
    height=200,
    placeholder="Paste or type your text here..."
)

# Add analyze button
analyze_button = st.button("Analyze Text", type="secondary")

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

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
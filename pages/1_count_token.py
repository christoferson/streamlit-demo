import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError

# Page configuration
st.set_page_config(
    page_title="Amazon Bedrock Token Counter",
    page_icon="🔢",
    layout="wide"
)

# Title and description
st.title("Bedrock Token Counter")
st.markdown("""
This tool helps you estimate token usage before sending requests to Amazon Bedrock foundation models.
Using the CountTokens API doesn't incur charges.
""")

# Sidebar for AWS configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # AWS Region selection
    regions = [
        "us-east-1",
        "us-east-2",
        "us-west-2",
        "ap-northeast-1",
        "ap-south-1",
        "ap-southeast-1",
        "ap-southeast-2",
        "eu-central-1",
        "eu-central-2",
        "eu-west-1",
        "eu-west-2",
        "sa-east-1"
    ]
    selected_region = st.selectbox("AWS Region", regions, index=0)

    # Model selection
    st.subheader("Model Selection")

    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
    models = [
        "global.anthropic.claude-fable-5",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "global.anthropic.claude-sonnet-4-20250514-v1:0",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "global.anthropic.claude-opus-4-6-v1",
        "global.anthropic.claude-opus-4-7",
        "global.anthropic.claude-opus-4-8",

        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-haiku-20240307-v1:0",
        "us.anthropic.claude-3-sonnet-20240229-v1:0",
        "us.anthropic.claude-3-opus-20240229-v1:0",
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-opus-4-20250514-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "amazon.nova-pro-v1:0",
        "global.amazon.nova-2-lite-v1:0",

        "cohere.command-r-v1:0",
        "cohere.command-r-plus-v1:0",
        "meta.llama2-13b-chat-v1",
        "meta.llama2-70b-chat-v1",
        "meta.llama3-8b-instruct-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "us.meta.llama3-2-11b-instruct-v1:0",
        "us.meta.llama3-2-90b-instruct-v1:0",
        "mistral.mistral-small-2402-v1:0",
        "mistral.mistral-large-2402-v1:0",
        "us.mistral.pixtral-large-2502-v1:0",
        "us.amazon.nova-premier-v1:0",
        "us.meta.llama4-scout-17b-instruct-v1:0",
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "us.writer.palmyra-x4-v1:0",
        "us.writer.palmyra-x5-v1:0",
        "qwen.qwen3-next-80b-a3b",
        "qwen.qwen3-vl-235b-a22b",
        "openai.gpt-oss-safeguard-20b",
        "openai.gpt-oss-safeguard-120b",
        "google.gemma-3-4b-it",
        "google.gemma-3-12b-it",
        "google.gemma-3-27b-it",
        "nvidia.nemotron-nano-9b-v2",
        "nvidia.nemotron-nano-12b-v2",
        "us.amazon.nova-2-lite-v1:0",
    ]

    default_model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    default_idx = models.index(default_model) if default_model in models else 0

    selected_model_id = st.selectbox("Model ID", models, index=default_idx)

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client(region):
    try:
        return boto3.client("bedrock-runtime", region_name=region)
    except NoCredentialsError:
        st.error("❌ AWS credentials not found. Please configure your AWS credentials.")
        return None
    except Exception as e:
        st.error(f"❌ Error initializing Bedrock client: {str(e)}")
        return None

bedrock_runtime = get_bedrock_client(selected_region)

# Main content - tabs for different request types
tab1, tab2 = st.tabs(["InvokeModel", "Converse"])

# Tab 1: InvokeModel Request
with tab1:
    st.markdown("Count Tokens for InvokeModel")
    st.markdown("Enter your prompt and model parameters to count tokens.")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_message = st.text_area(
            "User Message",
            value="What is the capital of France?",
            height=150,
            help="Enter the message you want to send to the model"
        )

    with col2:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=4096,
            value=500,
            help="Maximum number of tokens to generate"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            help="Controls randomness in the output"
        )

    if st.button("Count Tokens", type="primary"):
        if bedrock_runtime:
            try:
                with st.spinner("Counting tokens..."):
                    input_to_count = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ]
                    })

                    response = bedrock_runtime.count_tokens(
                        modelId=selected_model_id,
                        input={
                            "invokeModel": {
                                "body": input_to_count
                            }
                        }
                    )

                    st.success("✅ Token count completed!")

                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Input Tokens", response["inputTokens"])

                    # Show the request body
                    with st.expander("📄 View Request Body"):
                        st.json(json.loads(input_to_count))

            except ClientError as e:
                st.error(f"❌ AWS Error: {e.response['Error']['Message']}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 2: Converse Request
with tab2:
    st.markdown("Count Tokens for Converse")
    st.markdown("Build a conversation with multiple messages and system prompts.")

    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You're an expert in geography.",
        height=80,
        help="Set the context and behavior for the assistant"
    )

    # Conversation messages
    st.markdown("Conversation Messages")

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is its population?"}
        ]

    # Display and edit messages
    for idx, msg in enumerate(st.session_state.messages):
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            role = st.selectbox(
                f"Role {idx+1}",
                ["user", "assistant"],
                index=0 if msg["role"] == "user" else 1,
                key=f"role_{idx}"
            )
            st.session_state.messages[idx]["role"] = role

        with col2:
            content = st.text_area(
                f"Message {idx+1}",
                value=msg["content"], height=20,
                key=f"msg_{idx}"
            )
            st.session_state.messages[idx]["content"] = content

        with col3:
            st.write("")
            st.write("")
            if st.button(":material/delete_forever:", type="tertiary", key=f"del_{idx}"):
                st.session_state.messages.pop(idx)
                st.rerun()

    # Add new message button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Add Message"):
            st.session_state.messages.append({"role": "user", "content": ""})
            st.rerun()

    # Count tokens button
    if st.button("Count Tokens", type="primary", key="count_converse"):
        if bedrock_runtime:
            try:
                with st.spinner("Counting tokens..."):
                    # Build the input structure
                    input_to_count = {
                        "messages": [
                            {
                                "role": msg["role"],
                                "content": [{"text": msg["content"]}]
                            }
                            for msg in st.session_state.messages
                        ],
                        "system": [{"text": system_prompt}]
                    }

                    response = bedrock_runtime.count_tokens(
                        modelId=selected_model_id,
                        input={
                            "converse": input_to_count
                        }
                    )

                    st.success("✅ Token count completed!")

                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Input Tokens", response["inputTokens"])

                    # Show the request body
                    with st.expander("📄 View Request Body"):
                        st.json(input_to_count)

            except ClientError as e:
                st.error(f"❌ AWS Error: {e.response['Error']['Message']}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: gray;'>
#     <p>Amazon Bedrock Token Counter | Built with Streamlit</p>
#     <p>⚠️ Make sure your AWS credentials are properly configured</p>
# </div>
# """, unsafe_allow_html=True)
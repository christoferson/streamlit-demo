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
st.title("🔢 Amazon Bedrock Token Counter")
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
    st.subheader("Supported Models")
    models = {
        "Claude 3.5 Haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "Claude 3.5 Sonnet v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "Claude 3.7 Sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "Claude Opus 4": "anthropic.claude-opus-4-20250514-v1:0",
        "Claude Sonnet 4": "anthropic.claude-sonnet-4-20250514-v1:0"
    }

    selected_model_name = st.selectbox("Model", list(models.keys()))
    selected_model_id = models[selected_model_name]

    st.info(f"**Model ID:** `{selected_model_id}`")

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
tab1, tab2, tab3 = st.tabs(["📝 InvokeModel Request", "💬 Converse Request", "📚 Documentation"])

# Tab 1: InvokeModel Request
with tab1:
    st.header("Count Tokens for InvokeModel Request")
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

    if st.button("🔢 Count Tokens (InvokeModel)", type="primary"):
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
    st.header("Count Tokens for Converse Request")
    st.markdown("Build a conversation with multiple messages and system prompts.")

    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You're an expert in geography.",
        height=100,
        help="Set the context and behavior for the assistant"
    )

    # Conversation messages
    st.subheader("Conversation Messages")

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
                value=msg["content"],
                height=80,
                key=f"msg_{idx}"
            )
            st.session_state.messages[idx]["content"] = content

        with col3:
            st.write("")
            st.write("")
            if st.button("🗑️", key=f"del_{idx}"):
                st.session_state.messages.pop(idx)
                st.rerun()

    # Add new message button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("➕ Add Message"):
            st.session_state.messages.append({"role": "user", "content": ""})
            st.rerun()

    # Count tokens button
    if st.button("🔢 Count Tokens (Converse)", type="primary"):
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

# Tab 3: Documentation
with tab3:
    st.header("📚 Documentation")

    st.markdown("""
    ### About Token Counting

    The CountTokens API helps you estimate token usage before sending requests to foundation models. 
    This allows you to:

    - ✅ Estimate costs before sending inference requests
    - ✅ Optimize prompts to fit within token limits
    - ✅ Plan for token usage in your applications

    **Note:** Using the CountTokens API doesn't incur charges.

    ### Supported Regions

    - US East (N. Virginia) - `us-east-1`
    - US East (Ohio) - `us-east-2`
    - US West (Oregon) - `us-west-2`
    - Asia Pacific (Tokyo) - `ap-northeast-1`
    - Asia Pacific (Mumbai) - `ap-south-1`
    - Asia Pacific (Singapore) - `ap-southeast-1`
    - Asia Pacific (Sydney) - `ap-southeast-2`
    - Europe (Frankfurt) - `eu-central-1`
    - Europe (Zurich) - `eu-central-2`
    - Europe (Ireland) - `eu-west-1`
    - Europe (London) - `eu-west-2`
    - South America (São Paulo) - `sa-east-1`

    ### Prerequisites

    1. **AWS Credentials**: Configure your AWS credentials using one of these methods:
       - AWS CLI: `aws configure`
       - Environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
       - IAM role (if running on AWS)

    2. **IAM Permissions**: Your IAM identity needs:
       - `bedrock:CountTokens` - Allows usage of CountTokens
       - `bedrock:InvokeModel` - Allows usage of InvokeModel and Converse

    ### Example IAM Policy

    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:CountTokens",
                    "bedrock:InvokeModel"
                ],
                "Resource": "arn:aws:bedrock:*::foundation-model/*"
            }
        ]
    }
    ```

    ### Resources

    - [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
    - [Amazon Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
    - [Boto3 Bedrock Runtime Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Amazon Bedrock Token Counter | Built with Streamlit</p>
    <p>⚠️ Make sure your AWS credentials are properly configured</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
import re

# Page configuration
st.set_page_config(
    page_title="Amazon Bedrock Prompt Manager",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 Amazon Bedrock Prompt Manager")
st.markdown("""
Call and test your saved Amazon Bedrock prompts with custom variables.
""")

# Available models by region
AVAILABLE_MODELS = {
    "Amazon Nova": {
        "amazon.nova-pro-v1:0": "Nova Pro",
        "amazon.nova-lite-v1:0": "Nova Lite",
        "amazon.nova-micro-v1:0": "Nova Micro",
    },
    "Anthropic Claude": {
        "anthropic.claude-3-5-sonnet-20241022-v2:0": "Claude 3.5 Sonnet v2",
        "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude 3.5 Sonnet",
        "anthropic.claude-3-5-haiku-20241022-v1:0": "Claude 3.5 Haiku",
        "anthropic.claude-3-opus-20240229-v1:0": "Claude 3 Opus",
        "anthropic.claude-3-sonnet-20240229-v1:0": "Claude 3 Sonnet",
        "anthropic.claude-3-haiku-20240307-v1:0": "Claude 3 Haiku",
    },
    "Meta Llama": {
        "meta.llama3-1-405b-instruct-v1:0": "Llama 3.1 405B Instruct",
        "meta.llama3-1-70b-instruct-v1:0": "Llama 3.1 70B Instruct",
        "meta.llama3-1-8b-instruct-v1:0": "Llama 3.1 8B Instruct",
        "meta.llama3-70b-instruct-v1:0": "Llama 3 70B Instruct",
        "meta.llama3-8b-instruct-v1:0": "Llama 3 8B Instruct",
    },
    "Mistral AI": {
        "mistral.mistral-large-2407-v1:0": "Mistral Large 2 (24.07)",
        "mistral.mistral-large-2402-v1:0": "Mistral Large (24.02)",
        "mistral.mistral-small-2402-v1:0": "Mistral Small",
    },
    "Cohere": {
        "cohere.command-r-plus-v1:0": "Command R+",
        "cohere.command-r-v1:0": "Command R",
    },
    "AI21 Labs": {
        "ai21.jamba-1-5-large-v1:0": "Jamba 1.5 Large",
        "ai21.jamba-1-5-mini-v1:0": "Jamba 1.5 Mini",
    }
}

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

    st.info("💡 Make sure your AWS credentials are configured")

# Initialize Bedrock clients
@st.cache_resource
def get_bedrock_clients(region):
    try:
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        bedrock_agent = boto3.client("bedrock-agent", region_name=region)
        return bedrock_runtime, bedrock_agent
    except NoCredentialsError:
        st.error("❌ AWS credentials not found. Please configure your AWS credentials.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error initializing Bedrock clients: {str(e)}")
        return None, None

bedrock_runtime, bedrock_agent = get_bedrock_clients(selected_region)

def extract_variables_from_text(text):
    """Extract {{variable}} patterns from text"""
    if not text:
        return []
    pattern = r'\{\{(\w+)\}\}'
    return list(set(re.findall(pattern, text)))

def replace_variables(text, variables):
    """Replace {{variable}} with actual values"""
    if not text:
        return text
    result = text
    for key, value in variables.items():
        result = result.replace(f"{{{{{key}}}}}", value)
    return result

def get_model_display_name(model_id):
    """Get friendly display name for model ID"""
    for category, models in AVAILABLE_MODELS.items():
        if model_id in models:
            return f"{models[model_id]} ({category})"
    return model_id

def find_model_id(search_string):
    """Try to find a valid model ID from a string"""
    if not search_string:
        return None

    # Check if it's already a valid model ID
    for category, models in AVAILABLE_MODELS.items():
        if search_string in models:
            return search_string

    # Try to extract from ARN or other formats
    if "amazon.nova" in search_string.lower():
        if "pro" in search_string.lower():
            return "amazon.nova-pro-v1:0"
        elif "lite" in search_string.lower():
            return "amazon.nova-lite-v1:0"
        elif "micro" in search_string.lower():
            return "amazon.nova-micro-v1:0"

    return None

def get_prompt_details(bedrock_agent, prompt_id, version=None):
    """Get prompt details from Bedrock"""
    try:
        params = {
            "promptIdentifier": prompt_id
        }
        if version:
            params["promptVersion"] = version

        response = bedrock_agent.get_prompt(**params)
        return response
    except Exception as e:
        st.error(f"Error getting prompt details: {str(e)}")
        return None

def invoke_model_with_prompt(bedrock_runtime, model_id, system_prompt, user_message, config):
    """Invoke model using Converse API"""
    try:
        messages = [
            {
                "role": "user",
                "content": [{"text": user_message}]
            }
        ]

        system = [{"text": system_prompt}] if system_prompt else []

        inference_config = {
            "temperature": config.get("temperature", 0.7),
            "topP": config.get("topP", 0.9),
            "maxTokens": config.get("maxTokens", 512)
        }

        # Add stop sequences if present
        additional_params = {}
        if config.get("stopSequences"):
            additional_params["stopSequences"] = config["stopSequences"]

        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=messages,
            system=system,
            inferenceConfig=inference_config,
            **additional_params
        )

        return response
    except Exception as e:
        raise e

# Main tabs
tab1, tab2, tab3 = st.tabs(["🚀 Invoke Prompt", "📋 List Prompts", "📖 Documentation"])

# Tab 1: Invoke Prompt
with tab1:
    st.header("Invoke a Bedrock Prompt")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Prompt identifier input
        st.subheader("Prompt Identification")

        identifier_type = st.radio(
            "Identifier Type",
            ["Prompt ARN", "Prompt ID"],
            horizontal=True
        )

        if identifier_type == "Prompt ARN":
            prompt_identifier = st.text_input(
                "Prompt ARN",
                value="",
                placeholder="arn:aws:bedrock:region:account-id:prompt/PROMPT_ID",
                help="Full ARN of the prompt"
            )
        else:
            prompt_identifier = st.text_input(
                "Prompt ID",
                value="",
                placeholder="Enter your prompt ID (e.g., YBI8XTXN6U)",
                help="Just the prompt ID"
            )

        # Version selection
        prompt_version = st.text_input(
            "Prompt Version (optional)",
            value="",
            placeholder="Leave empty for DRAFT or enter version number",
            help="Enter a version number (e.g., '1') or leave empty for DRAFT"
        )

    with col2:
        st.subheader("Actions")
        load_prompt = st.button("📥 Load Prompt Details", type="secondary", use_container_width=True)

    # Load prompt details
    if load_prompt and prompt_identifier:
        if bedrock_agent:
            with st.spinner("Loading prompt details..."):
                prompt_details = get_prompt_details(bedrock_agent, prompt_identifier, prompt_version)

                if prompt_details:
                    st.session_state.prompt_details = prompt_details
                    st.success("✅ Prompt loaded successfully!")
                    st.rerun()

    # Display prompt details if loaded
    if 'prompt_details' in st.session_state:
        prompt_data = st.session_state.prompt_details

        # Display prompt info
        with st.expander("📋 Prompt Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Name:** {prompt_data.get('name', 'N/A')}")
                st.markdown(f"**ID:** `{prompt_data.get('id', 'N/A')}`")
            with col2:
                st.markdown(f"**Version:** {prompt_data.get('version', 'DRAFT')}")
                created = prompt_data.get('createdAt', 'N/A')
                st.markdown(f"**Created:** {created}")
            with col3:
                st.markdown(f"**Description:** {prompt_data.get('description', 'No description')}")

        # Extract prompt variants
        variants = prompt_data.get('variants', [])
        if variants:
            variant = variants[0]  # Use first variant

            # Get model configuration
            raw_model_id = variant.get('modelId', '')
            detected_model_id = find_model_id(raw_model_id)

            inference_config = variant.get('inferenceConfiguration', {}).get('text', {})

            # Display model configuration with dropdown
            st.subheader("⚙️ Model Configuration")

            col1, col2 = st.columns([3, 1])

            with col1:
                # Show original model ID if found
                if raw_model_id and raw_model_id != 'N/A':
                    st.info(f"📌 Original Model ID from prompt: `{raw_model_id}`")

                # Model selection dropdown
                st.markdown("**Select Model to Use:**")

                # Flatten models for dropdown
                model_options = {}
                for category, models in AVAILABLE_MODELS.items():
                    for model_id, model_name in models.items():
                        display_name = f"{model_name} - {category}"
                        model_options[display_name] = model_id

                # Set default selection
                default_index = 0
                if detected_model_id:
                    # Find index of detected model
                    for idx, (display_name, model_id) in enumerate(model_options.items()):
                        if model_id == detected_model_id:
                            default_index = idx
                            break

                selected_display = st.selectbox(
                    "Choose Model",
                    options=list(model_options.keys()),
                    index=default_index,
                    help="Select the model to use for inference"
                )

                selected_model_id = model_options[selected_display]

                st.success(f"✅ Using Model ID: `{selected_model_id}`")

            with col2:
                st.metric("Temperature", inference_config.get('temperature', 0.7))
                st.metric("Top P", inference_config.get('topP', 0.9))
                st.metric("Max Tokens", inference_config.get('maxTokens', 512))

            # Advanced settings
            with st.expander("🔧 Advanced Model Settings"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    override_temp = st.checkbox("Override Temperature")
                    if override_temp:
                        temperature = st.slider("Temperature", 0.0, 1.0, 
                                              inference_config.get('temperature', 0.7), 0.1)
                    else:
                        temperature = inference_config.get('temperature', 0.7)

                with col2:
                    override_topp = st.checkbox("Override Top P")
                    if override_topp:
                        top_p = st.slider("Top P", 0.0, 1.0, 
                                        inference_config.get('topP', 0.9), 0.05)
                    else:
                        top_p = inference_config.get('topP', 0.9)

                with col3:
                    override_tokens = st.checkbox("Override Max Tokens")
                    if override_tokens:
                        max_tokens = st.number_input("Max Tokens", 1, 4096, 
                                                    inference_config.get('maxTokens', 512))
                    else:
                        max_tokens = inference_config.get('maxTokens', 512)

            # Get template configuration
            template_config = variant.get('templateConfiguration', {})
            text_config = template_config.get('text', {})

            # Extract system and user messages
            system_text = ""
            user_text = ""

            if 'text' in text_config:
                # Simple text template
                user_text = text_config.get('text', '')
            else:
                # Chat template with system and messages
                system_prompts = text_config.get('systemPrompt', [])
                if system_prompts:
                    system_text = system_prompts[0].get('text', '')

                messages = text_config.get('messages', [])
                for msg in messages:
                    if msg.get('role') == 'user':
                        content = msg.get('content', [])
                        if content:
                            user_text = content[0].get('text', '')

            # Display templates
            st.subheader("📝 Prompt Templates")

            col1, col2 = st.columns(2)

            with col1:
                if system_text:
                    st.markdown("**System Instructions:**")
                    st.text_area("System", system_text, height=150, disabled=True, key="system_display")

            with col2:
                if user_text:
                    st.markdown("**User Message Template:**")
                    st.text_area("User", user_text, height=150, disabled=True, key="user_display")

            # Extract variables
            all_text = system_text + " " + user_text
            variables = extract_variables_from_text(all_text)

            if variables:
                st.subheader("🔧 Prompt Variables")
                st.markdown(f"Found {len(variables)} variable(s): {', '.join([f'`{{{{{v}}}}}`' for v in variables])}")

                # Initialize session state for variables
                if 'prompt_variables' not in st.session_state:
                    st.session_state.prompt_variables = {var: "" for var in variables}

                # Create input fields for each variable
                cols = st.columns(2)
                for idx, var in enumerate(variables):
                    with cols[idx % 2]:
                        value = st.text_input(
                            f"{{{{ {var} }}}}",
                            value=st.session_state.prompt_variables.get(var, ""),
                            key=f"var_{var}",
                            placeholder=f"Enter value for {var}"
                        )
                        st.session_state.prompt_variables[var] = value

                # Add custom variables
                with st.expander("➕ Add Custom Variables"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        new_var_name = st.text_input("Variable Name", key="new_var_name", 
                                                     placeholder="variable_name")
                    with col2:
                        new_var_value = st.text_input("Variable Value", key="new_var_value",
                                                      placeholder="value")
                    with col3:
                        st.write("")
                        st.write("")
                        if st.button("Add"):
                            if new_var_name:
                                st.session_state.prompt_variables[new_var_name] = new_var_value
                                st.rerun()
            else:
                # No variables found, initialize empty dict
                if 'prompt_variables' not in st.session_state:
                    st.session_state.prompt_variables = {}
                st.info("ℹ️ No variables found in this prompt template.")

            # Invoke button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                invoke_button = st.button("🚀 Invoke Prompt", type="primary", use_container_width=True)

            if invoke_button:
                # Validate all variables have values
                missing_vars = [var for var, val in st.session_state.prompt_variables.items() if not val and var in variables]

                if missing_vars:
                    st.error(f"❌ Please provide values for: {', '.join([f'`{{{{{v}}}}}`' for v in missing_vars])}")
                else:
                    if bedrock_runtime:
                        try:
                            with st.spinner("Invoking model..."):
                                # Replace variables in templates
                                final_system = replace_variables(system_text, st.session_state.prompt_variables)
                                final_user = replace_variables(user_text, st.session_state.prompt_variables)

                                # Show what will be sent
                                with st.expander("📤 Final Prompt (after variable substitution)", expanded=False):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if final_system:
                                            st.markdown("**System:**")
                                            st.code(final_system, language=None)
                                    with col2:
                                        st.markdown("**User:**")
                                        st.code(final_user, language=None)
                                    st.markdown(f"**Model:** `{selected_model_id}`")
                                    st.markdown(f"**Temperature:** {temperature} | **Top P:** {top_p} | **Max Tokens:** {max_tokens}")

                                # Prepare config
                                config = {
                                    "temperature": temperature,
                                    "topP": top_p,
                                    "maxTokens": max_tokens,
                                    "stopSequences": inference_config.get('stopSequences', [])
                                }

                                # Invoke model
                                response = invoke_model_with_prompt(
                                    bedrock_runtime,
                                    selected_model_id,
                                    final_system,
                                    final_user,
                                    config
                                )

                                st.success("✅ Response received!")

                                # Display response
                                st.subheader("📥 Model Response")

                                output = response.get('output', {})
                                message = output.get('message', {})
                                content = message.get('content', [])

                                if content:
                                    response_text = content[0].get('text', '')

                                    # Display in a nice box
                                    st.markdown("---")
                                    st.markdown(response_text)
                                    st.markdown("---")

                                # Show usage metrics
                                usage = response.get('usage', {})
                                if usage:
                                    st.subheader("📊 Token Usage")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Input Tokens", usage.get('inputTokens', 0))
                                    with col2:
                                        st.metric("Output Tokens", usage.get('outputTokens', 0))
                                    with col3:
                                        st.metric("Total Tokens", usage.get('totalTokens', 0))

                                # Show full response
                                with st.expander("🔍 Full Response Details"):
                                    st.json(response)

                        except ClientError as e:
                            error_code = e.response['Error']['Code']
                            error_message = e.response['Error']['Message']
                            st.error(f"❌ AWS Error ({error_code}): {error_message}")

                            if error_code == "ValidationException":
                                st.warning(f"""
                                💡 **Troubleshooting Model Access**

                                The model `{selected_model_id}` may not be available.

                                **Common Issues:**
                                1. Model not available in region `{selected_region}`
                                2. Model access not requested in Bedrock console
                                3. Insufficient IAM permissions

                                **Next Steps:**
                                1. Go to AWS Bedrock Console → Model Access
                                2. Request access to the model
                                3. Wait for approval (usually instant for most models)
                                4. Try a different model from the dropdown
                                """)
                            elif error_code == "ResourceNotFoundException":
                                st.warning("💡 Model not found. Try selecting a different model.")
                            elif error_code == "AccessDeniedException":
                                st.warning("💡 Check your IAM permissions for bedrock:InvokeModel")

                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            with st.expander("🐛 Debug Information"):
                                st.exception(e)
    else:
        st.info("👆 Enter a Prompt ARN or ID above and click 'Load Prompt Details' to get started.")

        # Show example
        with st.expander("💡 Example: How to find your Prompt ID"):
            st.markdown("""
            1. Go to AWS Bedrock Console
            2. Navigate to **Prompts** in the left sidebar
            3. Click on your prompt
            4. Copy the **Prompt ID** or **ARN** from the details page

            **Example Prompt ID:** `YBI8XTXN6U`

            **Example ARN:** `arn:aws:bedrock:us-east-1:123456789012:prompt/YBI8XTXN6U`
            """)

# Tab 2: List Prompts
with tab2:
    st.header("List Available Prompts")

    if st.button("🔄 Refresh Prompt List", use_container_width=True):
        if bedrock_agent:
            try:
                with st.spinner("Loading prompts..."):
                    response = bedrock_agent.list_prompts(maxResults=50)

                    prompts = response.get('promptSummaries', [])

                    if prompts:
                        st.success(f"✅ Found {len(prompts)} prompt(s)")

                        for prompt in prompts:
                            with st.expander(f"📝 {prompt.get('name', 'Unnamed')}"):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown(f"**ID:** `{prompt.get('id', 'N/A')}`")
                                    st.markdown(f"**Description:** {prompt.get('description', 'No description')}")

                                with col2:
                                    created = prompt.get('createdAt', 'N/A')
                                    updated = prompt.get('updatedAt', 'N/A')
                                    st.markdown(f"**Created:** {created}")
                                    st.markdown(f"**Updated:** {updated}")

                                # Show ARN
                                st.markdown("**ARN:**")
                                st.code(prompt.get('arn', ''), language=None)

                                # Copy ID button
                                st.markdown("**Prompt ID:**")
                                st.code(prompt.get('id', ''), language=None)

                                st.info(f"💡 Copy the ID above and paste it in the 'Invoke Prompt' tab")
                    else:
                        st.info("No prompts found in this region.")
                        st.markdown("""
                        **To create a prompt:**
                        1. Go to AWS Bedrock Console
                        2. Click on **Prompts** in the left sidebar
                        3. Click **Create prompt**
                        4. Configure your prompt with variables like `{{variable_name}}`
                        5. Save and return here to invoke it
                        """)

            except ClientError as e:
                st.error(f"❌ AWS Error: {e.response['Error']['Message']}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 3: Documentation
with tab3:
    st.header("📖 Documentation")

    st.markdown("""
    ### Quick Start Guide

    1. **Load Your Prompt**
       - Enter your Prompt ID or ARN
       - Click "Load Prompt Details"

    2. **Select Model**
       - Choose from the dropdown (auto-detected if possible)
       - All major Bedrock models are available

    3. **Fill Variables**
       - Enter values for `{{variables}}` in your prompt

    4. **Invoke**
       - Click "Invoke Prompt" to get results

    ### Available Models

    This tool supports all major Bedrock models:

    **Amazon Nova** (Recommended for general use)
    - Nova Pro - Best balance of performance and cost
    - Nova Lite - Fast and cost-effective
    - Nova Micro - Ultra-fast for simple tasks

    **Anthropic Claude** (Best for complex reasoning)
    - Claude 3.5 Sonnet v2 - Latest and most capable
    - Claude 3.5 Haiku - Fast and efficient
    - Claude 3 Opus - Most capable Claude 3 model

    **Meta Llama** (Open source)
    - Llama 3.1 405B - Largest open model
    - Llama 3.1 70B - Great performance
    - Llama 3.1 8B - Fast and efficient

    **Mistral AI** (European alternative)
    - Mistral Large - Most capable
    - Mistral Small - Cost-effective

    **Cohere** (Enterprise-focused)
    - Command R+ - Best for RAG
    - Command R - Efficient alternative

    ### Model Access

    If you get a validation error:

    1. Go to **AWS Bedrock Console**
    2. Click **Model Access** in left sidebar
    3. Click **Manage model access**
    4. Select the models you want
    5. Click **Request model access**
    6. Wait for approval (usually instant)

    ### Python Code Example

    ```python
    import boto3

    # Initialize clients
    bedrock_agent = boto3.client('bedrock-agent', region_name='us-east-1')
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

    # Get prompt
    prompt = bedrock_agent.get_prompt(promptIdentifier='YOUR_PROMPT_ID')

    # Extract details
    variant = prompt['variants'][0]
    template = variant['templateConfiguration']['text']

    # Get messages
    system = template.get('systemPrompt', [{}])[0].get('text', '')
    user = template.get('messages', [{}])[0].get('content', [{}])[0].get('text', '')

    # Replace variables
    user = user.replace('{{topic}}', 'AI').replace('{{language}}', 'English')

    # Invoke
    response = bedrock_runtime.converse(
        modelId='amazon.nova-pro-v1:0',
        messages=[{"role": "user", "content": [{"text": user}]}],
        system=[{"text": system}],
        inferenceConfig={"temperature": 0.7, "topP": 0.9, "maxTokens": 512}
    )

    print(response['output']['message']['content'][0]['text'])
    ```

    ### Required IAM Permissions

    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock-agent:GetPrompt",
                    "bedrock-agent:ListPrompts",
                    "bedrock:InvokeModel"
                ],
                "Resource": "*"
            }
        ]
    }
    ```

    ### Tips & Best Practices

    ✅ **Use DRAFT for testing**, create versions for production

    ✅ **Start with Nova Pro** - great balance of quality and cost

    ✅ **Use Claude for complex tasks** - best reasoning capabilities

    ✅ **Monitor token usage** - optimize prompts to reduce costs

    ✅ **Test different temperatures** - lower for factual, higher for creative

    ### Resources

    - [Bedrock Prompt Management](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html)
    - [Model Access Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
    - [Pricing Calculator](https://aws.amazon.com/bedrock/pricing/)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Amazon Bedrock Prompt Manager | Built with Streamlit</p>
    <p>✨ Select any model from the dropdown - no hardcoded values</p>
</div>
""", unsafe_allow_html=True)
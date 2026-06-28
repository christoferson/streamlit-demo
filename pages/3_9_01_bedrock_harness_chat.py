import streamlit as st
import boto3
import cmn_settings
import json
import logging
import uuid
from botocore.exceptions import ClientError

AWS_REGION = cmn_settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Bedrock Agent Harness",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.markdown("## 🤖 :blue[Bedrock Agent Harness Chat]")

# Initialize Bedrock AgentCore clients
bedrock_agentcore = boto3.client('bedrock-agentcore', region_name=AWS_REGION)
bedrock_agentcore_control = boto3.client('bedrock-agentcore-control', region_name=AWS_REGION)

# Function to list harnesses
@st.cache_data(ttl=300)  # Cache for 5 minutes
def list_harnesses():
    """List all available harnesses in the account"""
    try:
        response = bedrock_agentcore_control.list_harnesses()
        harnesses = response.get('harnesses', [])

        # Extract harness info
        harness_list = []
        for harness in harnesses:
            harness_arn = harness.get('arn', '')
            harness_name = harness.get('harnessName', 'Unknown')
            harness_id = harness.get('harnessId', '')
            status = harness.get('status', 'Unknown')

            # Include all harnesses (user can see status)
            harness_list.append({
                'arn': harness_arn,
                'name': harness_name,
                'id': harness_id,
                'status': status,
                'display': f"{harness_name} ({harness_id}) - {status}"
            })

        return harness_list
    except ClientError as e:
        logger.error(f"Error listing harnesses: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing harnesses: {e}")
        return []

# Initialize session state for harness ARN
if "selected_harness_arn" not in st.session_state:
    st.session_state.selected_harness_arn = None

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # List and select harness
    with st.spinner("Loading harnesses..."):
        available_harnesses = list_harnesses()

    if available_harnesses:
        # Create selectbox options
        harness_options = {h['display']: h['arn'] for h in available_harnesses}

        # Auto-select first harness if none selected
        if st.session_state.selected_harness_arn is None:
            first_arn = list(harness_options.values())[0]
            st.session_state.selected_harness_arn = first_arn
            logger.info(f"Auto-initialized harness to first available: {first_arn}")

        # Find the display name for the selected ARN
        selected_index = 0
        for idx, (display, arn) in enumerate(harness_options.items()):
            if arn == st.session_state.selected_harness_arn:
                selected_index = idx
                break

        selected_display = st.selectbox(
            "Select Harness",
            options=list(harness_options.keys()),
            index=selected_index,
            help="Choose from available harnesses in your account",
            key="harness_selectbox"
        )

        # Get the ARN for the displayed harness
        current_harness_arn = harness_options[selected_display]

        # Show harness details
        selected_harness = next(h for h in available_harnesses if h['display'] == selected_display)
        with st.container(border=True):
            st.caption(f"**Name:** {selected_harness['name']}")
            st.caption(f"**ID:** {selected_harness['id']}")
            #st.caption(f"**Status:** {selected_harness['status']}")
            st.caption(f"**ARN:** `{selected_harness['arn']}`")

            # Show current selection status
            is_selected = st.session_state.selected_harness_arn == current_harness_arn

            if is_selected:
                st.success("✓ Active")
            else:
                if st.button("Select This Harness", width='stretch', type="primary"):
                    st.session_state.selected_harness_arn = current_harness_arn
                    logger.info(f"Harness selected: {current_harness_arn}")
                    st.rerun()

        # Refresh button for harness list
        if st.button("🔄 Refresh List", width='stretch', type="secondary"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.error("No harnesses found. Please check your permissions or region.")
        st.info("Required permission: `bedrock-agentcore-control:ListHarnesses`")
        st.session_state.selected_harness_arn = None

    st.divider()

    # Session configuration
    st.subheader("Session Settings")

    # Actor ID input
    if "harness_user_id" not in st.session_state:
        st.session_state.harness_user_id = "default-user"

    user_id = st.text_input(
        "Actor ID",
        value=st.session_state.harness_user_id,
        help="Scopes/overrides the memory boundary to this user ID",
        key="user_id_input"
    )

    # Update session state if user changes the ID
    if user_id != st.session_state.harness_user_id:
        st.session_state.harness_user_id = user_id

    if "harness_session_id" not in st.session_state:
        # Generate a new session ID (minimum 2 characters)
        st.session_state.harness_session_id = str(uuid.uuid4())

    session_id_display = st.session_state.harness_session_id
    st.text_input("Runtime Session ID", value=session_id_display, disabled=True)

    if st.button("🔄 New Session", width='stretch'):
        st.session_state.harness_session_id = str(uuid.uuid4())
        st.session_state.harness_messages = []
        st.rerun()

    st.divider()

    # Agent parameters
    st.subheader("Agent Parameters")

    enable_trace = st.checkbox("Enable Trace", value=False, help="Show agent reasoning trace")
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=4096, value=2048, step=100)

# Initialize session state for messages
if "harness_messages" not in st.session_state:
    st.session_state.harness_messages = []

# Create main layout
main_col, right_sidebar_col = st.columns([3, 1])

# Main chat area
with main_col:
    # Container for chat history
    chat_container = st.container(height=500)

    with chat_container:
        for msg in st.session_state.harness_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

                # Show trace if available
                if "trace" in msg and msg["trace"]:
                    with st.expander("🔍 View Trace"):
                        st.json(msg["trace"])

    # Chat input
    prompt = st.chat_input("Ask the agent...")

# Right sidebar for session info
with right_sidebar_col:
    st.markdown("##### Session Info")

    with st.container(border=True):
        st.metric("Messages", len(st.session_state.harness_messages))
        st.caption(f"**Actor:** {st.session_state.get('harness_user_id', 'N/A')}")

        if st.session_state.harness_session_id:
            st.caption(f"Active session")
        else:
            st.caption("No active session")
        st.caption(f"Arn: {st.session_state.selected_harness_arn}")

    st.divider()

    st.markdown("##### Actions")

    if st.button("Clear History",
                 width='stretch',
                 type="secondary",
                 disabled=len(st.session_state.harness_messages) == 0):
        st.session_state.harness_messages = []
        st.rerun()

# Process user input
if prompt:
    # Add user message to history
    st.session_state.harness_messages.append({"role": "user", "content": prompt})

    # Display user message
    with main_col:
        with chat_container:
            st.chat_message("user").write(prompt)
    # Check if harness is selected
    if not st.session_state.selected_harness_arn:
        st.error("Please select a harness first")
        st.stop()

    logger.info(f"Invoking harness - ARN: {st.session_state.selected_harness_arn} | Session: {st.session_state.harness_session_id} | Actor: {st.session_state.harness_user_id}")

    with st.spinner("Processing..."):
        try:
            # Call Bedrock AgentCore invoke_harness
            response = bedrock_agentcore.invoke_harness(
                harnessArn=st.session_state.selected_harness_arn,
                runtimeSessionId=st.session_state.harness_session_id,
                actorId=st.session_state.harness_user_id,
                messages=[
                    {
                        'role': 'user',
                        'content': [{'text': prompt}]
                    }
                ]
            )

            # Process streaming response
            result_text = ""
            trace_data = []

            event_stream = response.get('stream', [])

            # Create placeholders for streaming
            with main_col:
                with chat_container:
                    with st.chat_message("assistant"):
                        result_container = st.container(border=True)
                        result_area = st.empty()

                        for event in event_stream:
                            # Handle contentBlockDelta events
                            if 'contentBlockDelta' in event:
                                delta = event['contentBlockDelta'].get('delta', {})
                                if 'text' in delta:
                                    text_chunk = delta['text']
                                    result_text += text_chunk
                                    result_area.write(result_text)

                            # Handle messageStop event
                            elif 'messageStop' in event:
                                stop_reason = event['messageStop'].get('stopReason', 'end_turn')
                                if stop_reason != 'end_turn':
                                    result_container.caption(f"Stop reason: {stop_reason}")

                            # Handle trace events if enabled
                            elif 'trace' in event and enable_trace:
                                trace_data.append(event['trace'])

                            # Handle metadata
                            elif 'metadata' in event:
                                metadata = event['metadata']
                                if 'usage' in metadata:
                                    usage = metadata['usage']
                                    stats = f"| Input: {usage.get('inputTokens', 0)} | Output: {usage.get('outputTokens', 0)}"
                                    result_container.caption(stats)

            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": result_text or "No response received"
            }

            if trace_data and enable_trace:
                assistant_message["trace"] = trace_data

            st.session_state.harness_messages.append(assistant_message)
            st.rerun()

        except ClientError as err:
            error_message = err.response["Error"]["Message"]
            logger.error("AWS Client Error: %s", error_message)
            st.error(f"❌ AWS Error: {error_message}")

            # Show more details in expander
            with st.expander("Error Details"):
                st.json(err.response)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            logger.error(error_message)
            st.error(f"❌ {error_message}")



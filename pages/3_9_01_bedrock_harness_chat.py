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

def get_harness_details(harness_id):
    """Get detailed information about a harness including memory ID"""
    try:
        response = bedrock_agentcore_control.get_harness(harnessId=harness_id)
        harness_details = response.get('harness', {})

        # Extract memory ID from nested structure
        memory_id = 'Not configured'
        memory_arn = None

        memory_config = harness_details.get('memory', {})
        if memory_config:
            managed_memory = memory_config.get('managedMemoryConfiguration', {})
            if managed_memory:
                memory_arn = managed_memory.get('arn', '')
                # Extract memory ID from ARN: arn:aws:bedrock-agentcore:region:account:memory/MEMORY_ID
                if memory_arn and 'memory/' in memory_arn:
                    memory_id = memory_arn.split('memory/')[-1]

        # Save memory ID to session state
        if memory_id != 'Not configured':
            st.session_state.harness_memory_id = memory_id

        return {
            'harnessId': harness_details.get('harnessId', 'N/A'),
            'harnessName': harness_details.get('harnessName', 'N/A'),
            'status': harness_details.get('status', 'N/A'),
            'memoryId': memory_id,
            'memoryArn': memory_arn or 'N/A',
            'maxIterations': harness_details.get('maxIterations', 'N/A'),
            'timeoutSeconds': harness_details.get('timeoutSeconds', 'N/A'),
            'createdAt': harness_details.get('createdAt', 'N/A'),
            'updatedAt': harness_details.get('updatedAt', 'N/A'),
            'description': harness_details.get('description', 'No description'),
            'full_response': harness_details
        }
    except ClientError as e:
        logger.error(f"Error getting harness details: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting harness details: {e}")
        return None

def list_memory_events(memory_id, session_id, actor_id, include_payloads=True, max_results=50):
    """List memory events for the given memory ID (sessionId and actorId are required)"""
    try:
        params = {
            'memoryId': memory_id,
            'sessionId': session_id,
            'actorId': actor_id,
            'includePayloads': include_payloads,
            'maxResults': max_results
        }

        response = bedrock_agentcore.list_events(**params)

        events = response.get('events', [])
        next_token = response.get('nextToken')

        return {
            'events': events,
            'nextToken': next_token,
            'count': len(events)
        }
    except ClientError as e:
        logger.error(f"Error listing memory events: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error listing memory events: {e}")
        return None

# Initialize session state for harness ARN and memory ID
if "selected_harness_arn" not in st.session_state:
    st.session_state.selected_harness_arn = None
if "harness_memory_id" not in st.session_state:
    st.session_state.harness_memory_id = None

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

            # Button to show detailed information
            if st.button("📋 Show Details", width='stretch', type="secondary", key=f"details_{selected_harness['id']}"):
                with st.spinner("Loading harness details..."):
                    details = get_harness_details(selected_harness['id'])

                    if details:
                        st.markdown("---")
                        st.markdown("**Detailed Information:**")
                        st.caption(f"**Memory ID:** `{details['memoryId']}`")
                        st.caption(f"**Memory ARN:** `{details['memoryArn']}`")
                        st.caption(f"**Status:** {details['status']}")
                        st.caption(f"**Max Iterations:** {details['maxIterations']}")
                        st.caption(f"**Timeout:** {details['timeoutSeconds']}s")
                        st.caption(f"**Created:** {details['createdAt']}")
                        st.caption(f"**Updated:** {details['updatedAt']}")
                        if details['description'] != 'No description':
                            st.caption(f"**Description:** {details['description']}")

                        # Show full response in expander
                        with st.expander("🔍 View Full Response"):
                            st.json(details['full_response'])
                    else:
                        st.error("Failed to load harness details")

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

    # Memory events button
    if st.session_state.harness_memory_id:
        st.divider()
        st.markdown("##### Memory")
        st.caption(f"Memory ID: `{st.session_state.harness_memory_id}`")

        if st.button("📜 View Memory Events", width='stretch', type="secondary"):
            st.session_state.show_memory_events = True
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

# Memory Events Dialog
if st.session_state.get('show_memory_events', False):
    @st.dialog("Memory Events", width="large")
    def show_memory_events_dialog():
        st.markdown("##### Memory Events")

        # Show current context
        st.caption(f"**Memory ID:** `{st.session_state.harness_memory_id}`")
        st.caption(f"**Session ID:** `{st.session_state.harness_session_id}`")
        st.caption(f"**Actor ID:** `{st.session_state.harness_user_id}`")

        st.divider()

        # Options
        col1, col2, col3 = st.columns(3)

        with col1:
            include_payloads = st.checkbox("Include Payloads", value=True)
        with col2:
            skip_unknown = st.checkbox("Skip Unknown Role", value=True)
        with col3:
            max_results = st.slider("Max Results", min_value=10, max_value=100, value=50, step=10)

        if st.button("🔍 Load Events", type="primary"):
            with st.spinner("Loading memory events..."):
                result = list_memory_events(
                    memory_id=st.session_state.harness_memory_id,
                    session_id=st.session_state.harness_session_id,
                    actor_id=st.session_state.harness_user_id,
                    include_payloads=include_payloads,
                    max_results=max_results
                )

                if result:
                    st.success(f"✅ Loaded {result['count']} events")

                    if result['nextToken']:
                        st.info(f"More events available. Use nextToken: `{result['nextToken']}`")

                    # Display events (reversed to show earliest first)
                    if result['events']:
                        for idx, event in enumerate(reversed(result['events']), 1):
                            # Extract key information
                            event_id = event.get('eventId', 'N/A')
                            timestamp = event.get('eventTimestamp', 'N/A')
                            payloads = event.get('payload', [])
                            branch = event.get('branch', {})
                            metadata = event.get('metadata', {})

                            # Determine event type and role from payload
                            event_type = 'EVENT'
                            role_display = None
                            content_preview = ''

                            if payloads and len(payloads) > 0:
                                first_payload = payloads[0]
                                if 'conversational' in first_payload:
                                    conv = first_payload['conversational']

                                    # Extract content.text which contains JSON string
                                    content = conv.get('content', {})
                                    text_field = content.get('text', '')

                                    # Parse the JSON string to get message details
                                    try:
                                        message_data = json.loads(text_field)
                                        message = message_data.get('message', {})
                                        role = message.get('role', 'UNKNOWN').upper()
                                        role_display = role
                                        event_type = 'CONVERSATIONAL'

                                        # Extract text from content array
                                        content_array = message.get('content', [])
                                        text = ''
                                        if content_array and len(content_array) > 0:
                                            if isinstance(content_array[0], dict) and 'text' in content_array[0]:
                                                text = content_array[0]['text']
                                        content_preview = text[:80] + ('...' if len(text) > 80 else '')
                                    except (json.JSONDecodeError, TypeError):
                                        # Fallback if not JSON or parsing fails
                                        role_display = 'UNKNOWN'
                                        event_type = 'CONVERSATIONAL'
                                        text = text_field
                                        content_preview = text[:80] + ('...' if len(text) > 80 else '')
                                elif 'blob' in first_payload:
                                    event_type = 'BLOB'
                                    content_preview = str(first_payload['blob'])[:50]

                            # Skip unknown role if option is enabled
                            if skip_unknown and (role_display == 'UNKNOWN' or role_display is None):
                                continue

                            # Create expander with meaningful title
                            if role_display:
                                # Use emoji for USER/ASSISTANT
                                if role_display == 'USER':
                                    title = f"👤 {idx}. USER"
                                elif role_display == 'ASSISTANT':
                                    title = f"🤖 {idx}. ASSISTANT"
                                elif role_display == 'TOOL':
                                    title = f"🔧 {idx}. TOOL"
                                else:
                                    title = f"📝 {idx}. {role_display}"
                            else:
                                title = f"📄 {idx}. {event_type}"

                            if timestamp != 'N/A':
                                if isinstance(timestamp, str):
                                    title += f" - {timestamp}"
                                else:
                                    title += f" - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

                            if content_preview:
                                title += f"\n{content_preview}"

                            with st.expander(title, expanded=(idx == 1)):
                                # Display payload prominently at the top
                                if payloads:
                                    for pidx, payload in enumerate(payloads):
                                        if 'conversational' in payload:
                                            conv = payload['conversational']

                                            # Extract content.text which contains JSON string
                                            content = conv.get('content', {})
                                            text_field = content.get('text', '')

                                            # Parse the JSON string to get message details
                                            try:
                                                message_data = json.loads(text_field)
                                                message = message_data.get('message', {})
                                                role = message.get('role', 'unknown').upper()

                                                # Extract text from content array
                                                content_array = message.get('content', [])
                                                text = ''
                                                if content_array and len(content_array) > 0:
                                                    if isinstance(content_array[0], dict) and 'text' in content_array[0]:
                                                        text = content_array[0]['text']

                                                # Extract metadata from message_data top level
                                                message_id = message_data.get('message_id', 'N/A')
                                                created_at = message_data.get('created_at', 'N/A')
                                                updated_at = message_data.get('updated_at', 'N/A')

                                                # Extract metadata (usage and metrics) from message object (not message_data)
                                                metadata_obj = message.get('metadata', {})
                                                usage = metadata_obj.get('usage', {})
                                                metrics = metadata_obj.get('metrics', {})
                                            except (json.JSONDecodeError, TypeError):
                                                # Fallback if not JSON or parsing fails
                                                role = 'UNKNOWN'
                                                text = text_field
                                                message_id = 'N/A'
                                                created_at = 'N/A'
                                                updated_at = 'N/A'
                                                usage = {}
                                                metrics = {}

                                            # Color-coded role badge
                                            if role == 'USER':
                                                st.markdown("### 👤 USER")
                                            elif role == 'ASSISTANT':
                                                st.markdown("### 🤖 ASSISTANT")
                                            elif role == 'TOOL':
                                                st.markdown("### 🔧 TOOL")
                                            else:
                                                st.markdown(f"### 📝 {role}")

                                            # Display message in a nice container
                                            with st.container(border=True):
                                                st.markdown(text)

                                            # Show message metadata and metrics
                                            if role == 'ASSISTANT':
                                                # Display metrics in columns for assistant messages
                                                metric_col1, metric_col2, metric_col3 = st.columns(3)

                                                with metric_col1:
                                                    if usage and any(usage.values()):
                                                        input_tokens = usage.get('inputTokens', 0)
                                                        output_tokens = usage.get('outputTokens', 0)
                                                        total_tokens = usage.get('totalTokens', 0)
                                                        st.caption(f"🔢 **Tokens:** In={input_tokens} | Out={output_tokens} | Total={total_tokens}")

                                                with metric_col2:
                                                    if metrics and any(metrics.values()):
                                                        latency = metrics.get('latencyMs', 0)
                                                        ttfb = metrics.get('timeToFirstByteMs', 0)
                                                        st.caption(f"⏱️ **Latency:** {latency}ms | TTFB={ttfb}ms")

                                                with metric_col3:
                                                    if message_id != 'N/A':
                                                        st.caption(f"🆔 Msg ID: {message_id}")
                                            elif message_id != 'N/A':
                                                st.caption(f"Message ID: {message_id} | Created: {created_at}")

                                        elif 'blob' in payload:
                                            st.markdown("### 📦 BLOB Data")
                                            with st.container(border=True):
                                                st.code(str(payload['blob']))

                                st.divider()

                                # Metadata section
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Event Details:**")
                                    st.caption(f"Event ID: `{event_id}`")
                                    st.caption(f"Timestamp: {timestamp}")

                                with col2:
                                    if branch:
                                        st.markdown("**Branch:**")
                                        st.caption(f"Name: {branch.get('name', 'N/A')}")
                                        if branch.get('rootEventId'):
                                            st.caption(f"Root: `{branch['rootEventId'][:12]}...`")

                                # Display metadata if present
                                if metadata:
                                    st.markdown("**Metadata:**")
                                    metadata_display = {}
                                    for key, value in metadata.items():
                                        if isinstance(value, dict) and 'stringValue' in value:
                                            metadata_display[key] = value['stringValue']
                                        else:
                                            metadata_display[key] = str(value)
                                    st.json(metadata_display)

                                # Show raw JSON in nested expander
                                with st.expander("🔍 View Raw JSON"):
                                    st.json(event)
                    else:
                        st.warning("No events found matching the criteria")
                else:
                    st.error("Failed to load memory events")

        if st.button("Close"):
            st.session_state.show_memory_events = False
            st.rerun()

    show_memory_events_dialog()

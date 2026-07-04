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

# Model configuration map — add new models here only
HARNESS_MODEL_CONFIGS = {
    "global.anthropic.claude-sonnet-4-6": {
        "temperature_supported": True,
        "temperature_default": 0.1,
    },
    "global.anthropic.claude-sonnet-5": {
        "temperature_supported": False,
    },
    "global.anthropic.claude-opus-4-8": {
        "temperature_supported": False,
    },
}
HARNESS_MODEL_IDS = list(HARNESS_MODEL_CONFIGS.keys())
HARNESS_MODEL_DEFAULT = HARNESS_MODEL_IDS[0]

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


# ============================================================================
# API Client Methods
# ============================================================================

def list_harnesses(bedrock_agentcore_control):
    """List all available harnesses in the account"""
    try:
        response = bedrock_agentcore_control.list_harnesses()
        harnesses = response.get('harnesses', [])

        harness_list = []
        for harness in harnesses:
            harness_list.append({
                'arn': harness.get('arn', ''),
                'name': harness.get('harnessName', 'Unknown'),
                'id': harness.get('harnessId', ''),
                'status': harness.get('status', 'Unknown'),
                'display': f"{harness.get('harnessName', 'Unknown')} ({harness.get('harnessId', '')}) - {harness.get('status', 'Unknown')}"
            })

        return harness_list
    except (ClientError, Exception) as e:
        logger.error(f"Error listing harnesses: {e}")
        return []


def get_harness_details(bedrock_agentcore_control, harness_id):
    """Get detailed information about a harness including memory ID"""
    try:
        response = bedrock_agentcore_control.get_harness(harnessId=harness_id)
        harness_details = response.get('harness', {})

        # Extract memory ID from nested structure
        memory_id = 'Not configured'
        memory_arn = None

        memory_config = harness_details.get('memory', {})
        if memory_config:
            # Try agentCoreMemoryConfiguration first (newer format)
            agent_core_memory = memory_config.get('agentCoreMemoryConfiguration', {})
            if agent_core_memory:
                memory_arn = agent_core_memory.get('arn', '')
            else:
                # Fallback to managedMemoryConfiguration (older format)
                managed_memory = memory_config.get('managedMemoryConfiguration', {})
                if managed_memory:
                    memory_arn = managed_memory.get('arn', '')

            if memory_arn and 'memory/' in memory_arn:
                memory_id = memory_arn.split('memory/')[-1]

        # Extract model config from harness
        model_id = None
        model_temperature = None
        model_max_tokens = None
        model_top_p = None
        model_api_format = None
        model_config = harness_details.get('model', {})
        if model_config:
            bedrock_cfg = model_config.get('bedrockModelConfig', {})
            if bedrock_cfg:
                model_id = bedrock_cfg.get('modelId')
                model_temperature = bedrock_cfg.get('temperature')
                model_max_tokens = bedrock_cfg.get('maxTokens')
                model_top_p = bedrock_cfg.get('topP')
                model_api_format = bedrock_cfg.get('apiFormat')

        # Extract system prompt
        system_prompt_blocks = harness_details.get('systemPrompt', [])
        system_prompt_text = '\n\n'.join(
            block.get('text', '') for block in system_prompt_blocks if block.get('text')
        ) or None

        return {
            'harnessId': harness_details.get('harnessId', 'N/A'),
            'harnessName': harness_details.get('harnessName', 'N/A'),
            'status': harness_details.get('status', 'N/A'),
            'memoryId': memory_id,
            'memoryArn': memory_arn or 'N/A',
            'modelId': model_id,
            'modelTemperature': model_temperature,
            'modelMaxTokens': model_max_tokens,
            'modelTopP': model_top_p,
            'modelApiFormat': model_api_format,
            'systemPrompt': system_prompt_text,
            'maxIterations': harness_details.get('maxIterations', 'N/A'),
            'maxTokens': harness_details.get('maxTokens', 'N/A'),
            'timeoutSeconds': harness_details.get('timeoutSeconds', 'N/A'),
            'createdAt': harness_details.get('createdAt', 'N/A'),
            'updatedAt': harness_details.get('updatedAt', 'N/A'),
            'description': harness_details.get('description', 'No description'),
            'full_response': harness_details
        }
    except (ClientError, Exception) as e:
        logger.error(f"Error getting harness details: {e}")
        return None


def list_sessions(bedrock_agentcore, memory_id, actor_id, max_results=20, next_token=None, has_events_only=False):
    """List sessions for a given memory and actor"""
    try:
        params = {
            'memoryId': memory_id,
            'actorId': actor_id,
            'maxResults': max_results,
        }
        if next_token:
            params['nextToken'] = next_token
        if has_events_only:
            params['filter'] = {'eventFilter': 'HAS_EVENTS'}

        response = bedrock_agentcore.list_sessions(**params)
        logger.info(f"list_sessions response keys: {list(response.keys())}")
        sessions = response.get('sessionSummaries', [])
        return {
            'sessions': sessions,
            'nextToken': response.get('nextToken'),
            'count': len(sessions)
        }
    except (ClientError, Exception) as e:
        logger.error(f"Error listing sessions: {e}")
        return None


def list_memory_events(bedrock_agentcore, memory_id, session_id, actor_id, include_payloads=True, max_results=50, next_token=None):
    """List memory events for the given memory ID"""
    try:
        params = {
            'memoryId': memory_id,
            'sessionId': session_id,
            'actorId': actor_id,
            'includePayloads': include_payloads,
            'maxResults': max_results
        }

        if next_token:
            params['nextToken'] = next_token

        response = bedrock_agentcore.list_events(**params)

        return {
            'events': response.get('events', []),
            'nextToken': response.get('nextToken'),
            'count': len(response.get('events', []))
        }
    except (ClientError, Exception) as e:
        logger.error(f"Error listing memory events: {e}")
        return None


def invoke_harness_stream(bedrock_agentcore, harness_arn, session_id, actor_id, messages, model_id=None, temperature=None, system_prompt=None, max_tokens=None):
    """Invoke harness with streaming response"""
    params = {
        'harnessArn': harness_arn,
        'runtimeSessionId': session_id,
        'actorId': actor_id,
        'messages': messages,
    }
    if model_id:
        bedrock_model_config = {'modelId': model_id}
        model_cfg = HARNESS_MODEL_CONFIGS.get(model_id, {})
        if temperature is not None and model_cfg.get("temperature_supported", False):
            bedrock_model_config['temperature'] = temperature
        if max_tokens is not None:
            bedrock_model_config['maxTokens'] = max_tokens
        params['model'] = {'bedrockModelConfig': bedrock_model_config}
    if system_prompt:
        params['systemPrompt'] = [{'text': system_prompt}]
    return bedrock_agentcore.invoke_harness(**params)


def invoke_runtime_command(bedrock_agentcore, harness_arn, session_id, command):
    """Invoke a shell command in the harness runtime environment via InvokeAgentRuntimeCommand"""
    try:
        response = bedrock_agentcore.invoke_agent_runtime_command(
            agentRuntimeArn=harness_arn,
            runtimeSessionId=session_id,
            body={"command": command}
        )
        stdout = ""
        stderr = ""
        exit_code = -1
        for event in response.get("stream", []):
            chunk = event.get("chunk", {})
            if "contentDelta" in chunk:
                delta = chunk["contentDelta"]
                stdout += delta.get("stdout", "")
                stderr += delta.get("stderr", "")
            elif "contentStop" in chunk:
                exit_code = chunk["contentStop"].get("exitCode", -1)
        return {
            'stdout': stdout,
            'stderr': stderr,
            'exitCode': exit_code
        }
    except (ClientError, Exception) as e:
        logger.error(f"Error invoking runtime command: {e}")
        return None


def end_agent_session(session_id):
    """End/terminate an agent session"""
    try:
        # Use bedrock-agent-runtime client to end session
        bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
        bedrock_agent_runtime.end_session(sessionIdentifier=session_id)
        logger.info(f"Session {session_id} terminated")
        return True
    except (ClientError, Exception) as e:
        logger.warning(f"Error ending session (may already be expired): {e}")
        return False


def list_long_term_memory_events(bedrock_agentcore, memory_id, actor_id, session_id, namespace):
    """List long-term memory records for a specific actor and namespace"""
    try:
        # Format namespace with actor_id and session_id
        formatted_namespace = namespace.format(actorId=actor_id, sessionId=session_id)

        logger.info(f"Listing memory records - Memory ID: {memory_id}, Actor ID: {actor_id}, Namespace: {formatted_namespace}")

        response = bedrock_agentcore.list_memory_records(
            memoryId=memory_id,
            namespace=formatted_namespace
        )

        logger.info(f"Response: {response}")

        # API returns 'memoryRecordSummaries' not 'memoryRecords'
        records = response.get('memoryRecordSummaries', [])

        return {
            'records': records,
            'nextToken': response.get('nextToken'),
            'count': len(records)
        }
    except (ClientError, Exception) as e:
        logger.error(f"Error listing long-term memory events: {e}")
        return None


def get_memory_namespaces(bedrock_agentcore_control, memory_id):
    """Get configured namespace paths from memory resource"""
    try:
        response = bedrock_agentcore_control.get_memory(memoryId=memory_id)
        logger.info(f"get_memory response: {response}")

        memory = response.get('memory', {})
        logger.info(f"memory object: {memory}")

        # Try to get strategies from different possible locations
        strategies = memory.get('strategies', [])

        if not strategies:
            # If no strategies, check for retrievalConfig (alternative structure)
            retrieval_config = memory.get('retrievalConfig', {})
            if retrieval_config:
                # extractConfig has namespaces as keys
                namespace_info = []
                for namespace_path, config in retrieval_config.items():
                    strategy_id = config.get('strategyId', 'Unknown')
                    namespace_info.append({
                        'strategy_type': strategy_id,
                        'namespaces': [namespace_path]
                    })
                return namespace_info

        namespace_info = []
        for strategy in strategies:
            # Strategy is a flat dict with strategyId, name, type, namespaces
            strategy_name = strategy.get('name', 'Unknown')
            strategy_type = strategy.get('type', 'Unknown')
            strategy_id = strategy.get('strategyId', 'Unknown')
            namespaces = strategy.get('namespaces', [])
            status = strategy.get('status', 'Unknown')

            namespace_info.append({
                'strategy_name': strategy_name,
                'strategy_type': strategy_type,
                'strategy_id': strategy_id,
                'namespaces': namespaces,
                'status': status
            })

        return namespace_info
    except (ClientError, Exception) as e:
        logger.error(f"Error getting memory namespaces: {e}")
        return None


def query_long_term_memory(bedrock_agentcore, memory_id, actor_id, session_id, namespace, query_text, max_results=10):
    """Query/search long-term memory records with a query string"""
    try:
        # Format namespace with actor_id and session_id
        formatted_namespace = namespace.format(actorId=actor_id, sessionId=session_id)

        response = bedrock_agentcore.query_memory(
            memoryId=memory_id,
            namespace=formatted_namespace,
            queryText=query_text,
            maxResults=max_results
        )

        # API returns 'memoryRecordSummaries' not 'memoryRecords'
        records = response.get('memoryRecordSummaries', [])

        return {
            'records': records,
            'nextToken': response.get('nextToken'),
            'count': len(records)
        }
    except (ClientError, Exception) as e:
        logger.error(f"Error querying long-term memory: {e}")
        return None


# ============================================================================
# Parser Methods
# ============================================================================

def parse_conversational_event(conv_payload):
    """Parse a conversational payload from memory events"""
    result = {
        'role': 'UNKNOWN',
        'text': '',
        'message_id': 'N/A',
        'created_at': 'N/A',
        'updated_at': 'N/A',
        'usage': {},
        'metrics': {}
    }

    try:
        content = conv_payload.get('content', {})
        text_field = content.get('text', '')
        message_data = json.loads(text_field)
        message = message_data.get('message', {})

        result['role'] = message.get('role', 'UNKNOWN').upper()

        content_array = message.get('content', [])
        if content_array and len(content_array) > 0:
            if isinstance(content_array[0], dict) and 'text' in content_array[0]:
                result['text'] = content_array[0]['text']

        result['message_id'] = message_data.get('message_id', 'N/A')
        result['created_at'] = message_data.get('created_at', 'N/A')
        result['updated_at'] = message_data.get('updated_at', 'N/A')

        metadata_obj = message.get('metadata', {})
        result['usage'] = metadata_obj.get('usage', {})
        result['metrics'] = metadata_obj.get('metrics', {})

    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse conversational event: {e}")
        result['text'] = text_field if 'text_field' in locals() else ''

    return result


def parse_event_payload(event):
    """Parse an event and extract payload information"""
    result = {
        'event_type': 'EVENT',
        'role': None,
        'content_preview': '',
        'parsed_data': None
    }

    payloads = event.get('payload', [])
    if not payloads or len(payloads) == 0:
        return result

    first_payload = payloads[0]

    if 'conversational' in first_payload:
        conv = first_payload['conversational']
        result['event_type'] = 'CONVERSATIONAL'
        parsed = parse_conversational_event(conv)
        result['role'] = parsed['role']
        result['parsed_data'] = parsed
        text = parsed['text']
        result['content_preview'] = text[:80] + ('...' if len(text) > 80 else '')

    elif 'blob' in first_payload:
        result['event_type'] = 'BLOB'
        result['content_preview'] = str(first_payload['blob'])[:50]

    return result


def format_event_title(idx, event_info, timestamp):
    """Format event title for display"""
    role = event_info['role']
    event_type = event_info['event_type']
    content_preview = event_info['content_preview']

    if role:
        if role == 'USER':
            title = f"👤 {idx}. USER"
        elif role == 'ASSISTANT':
            title = f"🤖 {idx}. ASSISTANT"
        elif role == 'TOOL':
            title = f"🔧 {idx}. TOOL"
        else:
            title = f"📝 {idx}. {role}"
    else:
        title = f"📄 {idx}. {event_type}"

    if timestamp != 'N/A':
        if isinstance(timestamp, str):
            title += f" - {timestamp}"
        else:
            title += f" - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    if content_preview:
        title += f"\n{content_preview}"

    return title


# ============================================================================
# UI Render Methods
# ============================================================================

def render_harness_details_dialog(harness_details):
    """Render harness details in a dialog"""
    if harness_details:
        st.subheader("Harness Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Basic Information:**")
            st.markdown(f":blue[ID: `{harness_details['harnessId']}`]")
            st.markdown(f":blue[Name: {harness_details['harnessName']}]")
            st.markdown(f":blue[Status: {harness_details['status']}]")

        with col2:
            st.markdown("**Configuration:**")
            st.markdown(f":blue[Max Iterations: {harness_details['maxIterations']}]")
            st.markdown(f":blue[Max Tokens: {harness_details['maxTokens']}]")
            st.markdown(f":blue[Timeout: {harness_details['timeoutSeconds']}s]")

        st.markdown("**Memory Configuration:**")
        st.markdown(f":blue[Memory ID: `{harness_details['memoryId']}`]")

        st.markdown("**Default Model & Inference:**")
        with st.container(border=True):
            st.markdown(f":blue[Model ID: `{harness_details['modelId'] or 'Not configured'}`]")
            parts = []
            if harness_details['modelTemperature'] is not None:
                parts.append(f"Temperature: {harness_details['modelTemperature']}")
            if harness_details['modelMaxTokens'] is not None:
                parts.append(f"Max Tokens: {harness_details['modelMaxTokens']}")
            if harness_details['modelTopP'] is not None:
                parts.append(f"Top P: {harness_details['modelTopP']}")
            if harness_details['modelApiFormat'] is not None:
                parts.append(f"API Format: {harness_details['modelApiFormat']}")
            if parts:
                st.markdown(f":blue[{' | '.join(parts)}]")
            else:
                st.markdown(":blue[Inference params: harness defaults]")

        system_prompt = harness_details.get('systemPrompt')
        if system_prompt:
            st.markdown("**System Prompt:**")
            with st.container(border=True):
                st.markdown(system_prompt)
        else:
            st.markdown(":blue[System Prompt: not configured]")

        if harness_details.get('description') != 'No description':
            st.markdown("**Description:**")
            st.info(harness_details['description'])

        st.markdown("**Timestamps:**")
        st.markdown(f":blue[Created: {harness_details['createdAt']}]")
        st.markdown(f":blue[Updated: {harness_details['updatedAt']}]")

        with st.expander("📄 Full Response", expanded=False):
            st.json(harness_details['full_response'])
    else:
        st.error("Failed to load harness details")


def render_conversational_payload(parsed_data):
    """Render a parsed conversational payload"""
    role = parsed_data['role']
    text = parsed_data['text']
    message_id = parsed_data['message_id']
    created_at = parsed_data['created_at']
    usage = parsed_data['usage']
    metrics = parsed_data['metrics']

    if role == 'USER':
        st.markdown("### 👤 USER")
    elif role == 'ASSISTANT':
        st.markdown("### 🤖 ASSISTANT")
    elif role == 'TOOL':
        st.markdown("### 🔧 TOOL")
    else:
        st.markdown(f"### 📝 {role}")

    with st.container(border=True):
        st.markdown(text)

    if role == 'ASSISTANT':
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            if usage and any(usage.values()):
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)
                total_tokens = usage.get('totalTokens', 0)
                st.markdown(f":blue[**Tokens:** In={input_tokens} | Out={output_tokens} | Total={total_tokens}]")

        with metric_col2:
            if metrics and any(metrics.values()):
                latency = metrics.get('latencyMs', 0)
                ttfb = metrics.get('timeToFirstByteMs', 0)
                st.markdown(f":blue[**Latency:** {latency}ms | TTFB={ttfb}ms]")

        with metric_col3:
            if message_id != 'N/A':
                st.markdown(f":blue[Msg ID: {message_id}]")
    elif message_id != 'N/A':
        st.markdown(f":blue[Message ID: {message_id} | Created: {created_at}]")


def render_memory_event(idx, event, skip_unknown=True):
    """Render a single memory event. Returns True if rendered, False if skipped"""
    event_id = event.get('eventId', 'N/A')
    timestamp = event.get('eventTimestamp', 'N/A')
    payloads = event.get('payload', [])
    branch = event.get('branch', {})
    metadata = event.get('metadata', {})

    event_info = parse_event_payload(event)
    role = event_info['role']

    if skip_unknown and (role == 'UNKNOWN' or role is None):
        return False

    title = format_event_title(idx, event_info, timestamp)

    with st.expander(title, expanded=(idx == 1)):
        if payloads:
            for pidx, payload in enumerate(payloads):
                if 'conversational' in payload:
                    parsed = event_info['parsed_data']
                    render_conversational_payload(parsed)
                elif 'blob' in payload:
                    st.markdown("### 📦 BLOB Data")
                    with st.container(border=True):
                        st.code(str(payload['blob']))

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Event Details:**")
            st.markdown(f":blue[Event ID: `{event_id}`]")
            st.markdown(f":blue[Timestamp: {timestamp}]")

        with col2:
            if branch:
                st.markdown("**Branch:**")
                st.markdown(f":blue[Name: {branch.get('name', 'N/A')}]")
                if branch.get('rootEventId'):
                    st.markdown(f":blue[Root: `{branch['rootEventId'][:12]}...`]")

        if metadata:
            with st.expander("🔍 Event Metadata", expanded=False):
                st.json(metadata)

    return True


def render_memory_events_dialog(list_memory_events_fn, memory_id, session_id, actor_id):
    """Render memory events dialog"""
    st.subheader("Memory Events")

    with st.container(border=True):
        st.markdown(f":violet[**Memory ID:** `{memory_id}`]")
        st.markdown(f":violet[**Session ID:** `{session_id}`]")
        st.markdown(f":violet[**Actor ID:** `{actor_id}`]")

    # Initialize session state for accumulated events and next token
    if 'memory_events_accumulated' not in st.session_state:
        st.session_state.memory_events_accumulated = []
    if 'memory_events_next_token' not in st.session_state:
        st.session_state.memory_events_next_token = None

    col1, col2, col3 = st.columns(3)

    with col1:
        include_payloads = st.checkbox("Include Payloads", value=True)
    with col2:
        skip_unknown = st.checkbox("Skip Unknown Role", value=True)
    with col3:
        max_results = st.slider("Max Results", min_value=10, max_value=100, value=50, step=10)

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🔍 Load Events", type="primary", use_container_width=True):
            with st.spinner("Loading memory events..."):
                # Clear accumulated events for fresh load
                st.session_state.memory_events_accumulated = []
                st.session_state.memory_events_next_token = None

                result = list_memory_events_fn(
                    memory_id=memory_id,
                    session_id=session_id,
                    actor_id=actor_id,
                    include_payloads=include_payloads,
                    max_results=max_results
                )

                if result:
                    st.session_state.memory_events_accumulated = result['events']
                    st.session_state.memory_events_next_token = result.get('nextToken')

    with col_btn2:
        if st.button("📥 Load Next Events",
                     type="secondary",
                     use_container_width=True,
                     disabled=not st.session_state.memory_events_next_token):
            with st.spinner("Loading more events..."):
                result = list_memory_events_fn(
                    memory_id=memory_id,
                    session_id=session_id,
                    actor_id=actor_id,
                    include_payloads=include_payloads,
                    max_results=max_results,
                    next_token=st.session_state.memory_events_next_token
                )

                if result:
                    # Append new events
                    st.session_state.memory_events_accumulated.extend(result['events'])
                    st.session_state.memory_events_next_token = result.get('nextToken')

    # Display accumulated events
    if st.session_state.memory_events_accumulated:
        st.success(f"✅ Loaded {len(st.session_state.memory_events_accumulated)} total events")

        if st.session_state.memory_events_next_token:
            st.info("More events available. Click 'Load Next Events' to continue.")

        rendered_count = 0
        for idx, event in enumerate(reversed(st.session_state.memory_events_accumulated), 1):
            if render_memory_event(idx, event, skip_unknown):
                rendered_count += 1

        if rendered_count == 0:
            st.info("No events to display (all filtered out)")
    elif st.session_state.memory_events_accumulated is not None and len(st.session_state.memory_events_accumulated) == 0:
        st.info("No events found")


def render_long_term_memory_dialog(list_long_term_memory_fn, query_long_term_memory_fn, get_namespaces_fn, memory_id, actor_id, session_id):
    """Render long-term memory records dialog"""
    st.subheader("Long-Term Memory Events")

    with st.container(border=True):
        st.markdown(f":violet[**Memory ID:** `{memory_id}`]")
        st.markdown(f":violet[**Actor ID:** `{actor_id}`]")
        st.markdown(f":violet[**Session ID:** `{session_id}`]")

    # Initialize cached namespace info
    if 'cached_namespace_info' not in st.session_state:
        st.session_state.cached_namespace_info = None

    # Full width namespace listing
    if st.button("📋 List Configured Namespaces", type="secondary", use_container_width=True):
        with st.spinner("Loading namespaces..."):
            st.session_state.cached_namespace_info = get_namespaces_fn(memory_id=memory_id)

    if st.session_state.cached_namespace_info:
        with st.container(border=True):
            st.markdown("**Configured Namespaces**")
            for info in st.session_state.cached_namespace_info:
                ns_list = " | ".join([f"`{ns}`" for ns in info['namespaces']])
                st.markdown(f":blue[**{info['strategy_name']}** ({info['strategy_type']}) → {ns_list} — {info['status']}]")

    # Built-in namespace options (based on actual harness configuration)
    namespace_options = {
        "Facts": "/facts/{actorId}/",
        "Session Summaries": "/summaries/{actorId}/{sessionId}/",
        "Semantic Facts (Alt)": "/actors/{actorId}/semantic",
        "Session Summaries (Alt)": "/actors/{actorId}/{sessionId}/summary",
        "App Facts": "/app/{actorId}/facts",
        "App Preferences": "/app/{actorId}/preferences",
        "Episodic": "/episodic/{actorId}",
        "Custom Namespace": "custom"
    }

    selected_namespace_type = st.selectbox(
        "Namespace Type",
        options=list(namespace_options.keys()),
        index=0,
        help="Select the type of long-term memory to view"
    )

    namespace = namespace_options[selected_namespace_type]

    # Allow custom namespace input
    if selected_namespace_type == "Custom Namespace":
        namespace = st.text_input(
            "Custom Namespace Path",
            value="/actors/{actorId}/custom",
            help="Use {actorId} and {sessionId} placeholders for dynamic substitution"
        )
    else:
        st.markdown(f":blue[**Namespace:** `{namespace}`]")

    # Initialize cached results
    if 'cached_memory_records' not in st.session_state:
        st.session_state.cached_memory_records = None

    # Search or List tabs
    tab1, tab2 = st.tabs(["📚 List All", "🔍 Search"])

    with tab1:
        if st.button("📚 Load All Records", type="primary", key="load_all"):
            with st.spinner("Loading long-term memory events..."):
                st.session_state.cached_memory_records = list_long_term_memory_fn(
                    memory_id=memory_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    namespace=namespace
                )

        if st.session_state.cached_memory_records:
            _render_memory_records_result(st.session_state.cached_memory_records)

    with tab2:
        query_text = st.text_input(
            "Search Query",
            placeholder="Enter search keywords...",
            help="Search for specific content in long-term memory"
        )

        max_results = st.slider("Max Results", min_value=5, max_value=50, value=10, step=5)

        if st.button("🔍 Search", type="primary", key="search", disabled=not query_text):
            with st.spinner(f"Searching for '{query_text}'..."):
                st.session_state.cached_memory_records = query_long_term_memory_fn(
                    memory_id=memory_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    namespace=namespace,
                    query_text=query_text,
                    max_results=max_results
                )

        if st.session_state.cached_memory_records:
            _render_memory_records_result(st.session_state.cached_memory_records)


def _render_memory_records_result(result):
    """Helper to render memory records result"""
    if result:
        st.success(f"✅ Found {result['count']} memory records")

        if result['nextToken']:
            st.info(f"More records available. Use nextToken: `{result['nextToken']}`")

        if result['records']:
            for idx, record in enumerate(result['records'], 1):
                content = record.get('content', {})
                text = content.get('text', 'N/A')
                record_id = record.get('memoryRecordId', 'N/A')
                strategy_id = record.get('memoryStrategyId', 'N/A')
                created_at = record.get('createdAt')
                relevance_score = record.get('relevanceScore')

                # Build compact metadata line
                meta_parts = [f"Record ID: {record_id}"]
                if strategy_id != 'N/A':
                    meta_parts.append(f"Strategy: {strategy_id}")
                if created_at:
                    meta_parts.append(f"Created: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                if relevance_score is not None:
                    meta_parts.append(f"Relevance Score: {relevance_score:.4f}")

                with st.expander(f"{idx}. {text[:80]}{'...' if len(text) > 80 else ''}", expanded=False):
                    st.markdown(text)
                    st.markdown(f":blue[{' | '.join(meta_parts)}]")

                    with st.expander("View Raw Record"):
                        import json as json_module
                        record_json = json_module.loads(json_module.dumps(record, default=str))
                        st.json(record_json)
        else:
            st.info("No memory records found")
    else:
        st.error("Failed to retrieve memory records")


def render_custom_environment_dialog(invoke_runtime_command_fn, harness_arn, session_id, actor_id):
    """Render Run Command / InvokeAgentRuntimeCommand dialog"""
    st.subheader("Run Command")

    with st.container(border=True):
        st.markdown(f":violet[**Harness ARN:** `{harness_arn}`]")
        st.markdown(f":violet[**Session ID:** `{session_id}`]")
        st.markdown(f":violet[**Actor ID:** `{actor_id}`]")

    st.markdown("""
Direct shell access to the harness microVM — no model reasoning, no token cost.
Use it to prepare the environment, inspect the VM, run scripts, or act on agent output.
""")

    preset_commands = {
        "Python version": "python --version",
        "Python packages": "pip list",
        "Bash version": "bash --version",
        "OS info": "cat /etc/os-release",
        "Architecture": "uname -m",
        "Working directory": "pwd",
        "Environment variables": "env",
        "Disk usage": "df -h",
        "Running processes": "ps aux",
        "Custom command": ""
    }

    selected_preset = st.selectbox("Preset Commands", options=list(preset_commands.keys()), index=0)

    default_cmd = preset_commands[selected_preset]
    command = st.text_input("Command", value=default_cmd, placeholder="e.g. python --version")

    if st.button("▶ Run Command", type="primary", use_container_width=True, disabled=not command.strip()):
        with st.spinner(f"Running: `{command}` ..."):
            result = invoke_runtime_command_fn(command=command.strip())

        if result is None:
            st.error("Command invocation failed. Check logs for details.")
        else:
            exit_code = result.get('exitCode', -1)
            stdout = result.get('stdout', '')
            stderr = result.get('stderr', '')

            if exit_code == 0:
                st.success(f"Exit code: {exit_code}")
            else:
                st.warning(f"Exit code: {exit_code}")

            if stdout:
                st.markdown("**stdout**")
                st.code(stdout, language="bash")

            if stderr:
                st.markdown("**stderr**")
                st.code(stderr, language="bash")

            if not stdout and not stderr:
                st.info("Command produced no output.")


def render_sessions_dialog(list_sessions_fn, memory_id, actor_id, current_session_id):
    """Render sessions list dialog"""
    st.subheader("Sessions")

    with st.container(border=True):
        st.markdown(f":violet[**Memory ID:** `{memory_id}`]")
        st.markdown(f":violet[**Actor ID:** `{actor_id}`]")

    has_events_only = st.checkbox("Only sessions with events", value=False)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Load Sessions", type="primary", use_container_width=True):
            with st.spinner("Loading sessions..."):
                result = list_sessions_fn(has_events_only=has_events_only)
            logger.info(f"list_sessions result: {result}")
            if result is not None:
                st.session_state.sessions_accumulated = result['sessions']
                st.session_state.sessions_next_token = result.get('nextToken')
                st.session_state.sessions_loaded = True
            else:
                st.error("Failed to load sessions. Check logs for details.")

    with col2:
        if st.session_state.sessions_next_token:
            if st.button("Load Next", type="secondary", use_container_width=True):
                with st.spinner("Loading more..."):
                    result = list_sessions_fn(
                        next_token=st.session_state.sessions_next_token,
                        has_events_only=has_events_only
                    )
                if result is not None:
                    st.session_state.sessions_accumulated.extend(result['sessions'])
                    st.session_state.sessions_next_token = result.get('nextToken')

    sessions = st.session_state.sessions_accumulated
    if sessions:
        sorted_sessions = sorted(sessions, key=lambda s: s.get('createdAt') or '', reverse=True)
        st.markdown(f":blue[**{len(sorted_sessions)} session(s)**]")
        for session in sorted_sessions:
            session_id = session.get('sessionId', 'N/A')
            created_at_raw = session.get('createdAt')
            created_at = created_at_raw.strftime('%Y-%m-%d %H:%M:%S') if created_at_raw else 'N/A'
            is_current = session_id == current_session_id
            color = "violet" if is_current else "blue"
            st.markdown(f":{color}[**`{session_id}`** | {created_at}]")
    elif st.session_state.get("sessions_loaded"):
        st.info("No sessions found for this actor and memory.")


def render_session_info_panel(session_id, actor_id, memory_id):
    """Render session information panel in sidebar"""
    #st.divider()
    st.markdown("##### Session Info")

    with st.container(border=True):
        st.markdown(f":violet[**Session ID:** `{session_id}`]")
        st.markdown(f":violet[**Actor ID:** `{actor_id}`]")
        if memory_id:
            st.markdown(f":violet[**Memory ID:** `{memory_id}`]")


# ============================================================================
# Main Application
# ============================================================================

# Initialize Bedrock AgentCore clients
bedrock_agentcore = boto3.client('bedrock-agentcore', region_name=AWS_REGION)
bedrock_agentcore_control = boto3.client('bedrock-agentcore-control', region_name=AWS_REGION)

# Cached wrapper for list_harnesses
@st.cache_data(ttl=300)
def list_harnesses_cached():
    """List all available harnesses in the account (cached for 5 minutes)"""
    return list_harnesses(bedrock_agentcore_control)

# Initialize session state
if "selected_harness_arn" not in st.session_state:
    st.session_state.selected_harness_arn = None
if "harness_memory_id" not in st.session_state:
    st.session_state.harness_memory_id = None
if "harness_session_id" not in st.session_state:
    st.session_state.harness_session_id = str(uuid.uuid4())
if "harness_user_id" not in st.session_state:
    st.session_state.harness_user_id = "default-user"
if "harness_messages" not in st.session_state:
    st.session_state.harness_messages = []
if "latest_tool_use" not in st.session_state:
    st.session_state.latest_tool_use = []
if "latest_usage_stats" not in st.session_state:
    st.session_state.latest_usage_stats = None
if "harness_model_id" not in st.session_state:
    st.session_state.harness_model_id = HARNESS_MODEL_DEFAULT
if "harness_override_system_prompt" not in st.session_state:
    st.session_state.harness_override_system_prompt = False
if "harness_system_prompt" not in st.session_state:
    st.session_state.harness_system_prompt = ""
if "harness_configured_system_prompt" not in st.session_state:
    st.session_state.harness_configured_system_prompt = ""
if "harness_configured_max_tokens" not in st.session_state:
    st.session_state.harness_configured_max_tokens = None
if "sessions_accumulated" not in st.session_state:
    st.session_state.sessions_accumulated = []
if "sessions_next_token" not in st.session_state:
    st.session_state.sessions_next_token = None
if "sessions_loaded" not in st.session_state:
    st.session_state.sessions_loaded = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.spinner("Loading harnesses..."):
        available_harnesses = list_harnesses_cached()

    if available_harnesses:
        harness_options = {h['display']: h['arn'] for h in available_harnesses}

        if st.session_state.selected_harness_arn is None:
            first_arn = list(harness_options.values())[0]
            st.session_state.selected_harness_arn = first_arn
            logger.info(f"Auto-initialized harness to first available: {first_arn}")

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

        current_harness_arn = harness_options[selected_display]

        selected_harness = next(h for h in available_harnesses if h['display'] == selected_display)
        with st.container(border=True):
            st.markdown(f":blue[**Name:** {selected_harness['name']}]")
            st.markdown(f":blue[**ID:** {selected_harness['id']}]")
            st.markdown(f":blue[**ARN:** `{selected_harness['arn']}`]")

            is_selected = st.session_state.selected_harness_arn == current_harness_arn

            if is_selected:
                st.success("✓ Active")
            else:
                if st.button("Select This Harness", use_container_width=True, type="primary"):
                    st.session_state.selected_harness_arn = current_harness_arn
                    logger.info(f"Harness selected: {current_harness_arn}")
                    st.rerun()

            if st.button("📋 Show Details", use_container_width=True, type="secondary", key=f"details_{selected_harness['id']}"):
                with st.spinner("Loading harness details..."):
                    details = get_harness_details(bedrock_agentcore_control, selected_harness['id'])
                    logger.info(f"Details fetched: {details}")
                    if details:
                        logger.info(f"Memory ID from details: {details['memoryId']}")
                        if details['memoryId'] != 'Not configured':
                            st.session_state.harness_memory_id = details['memoryId']
                            logger.info(f"Memory ID set in session state: {st.session_state.harness_memory_id}")
                        else:
                            logger.warning("Memory ID is 'Not configured'")
                        if details.get('modelId'):
                            st.session_state.harness_model_id = details['modelId']
                            logger.info(f"Model ID set in session state: {st.session_state.harness_model_id}")
                        if details.get('systemPrompt'):
                            st.session_state.harness_configured_system_prompt = details['systemPrompt']
                        if details.get('maxTokens') not in (None, 'N/A'):
                            st.session_state.harness_configured_max_tokens = details['maxTokens']
                    else:
                        logger.error("Details is None")

                @st.dialog("Harness Details", width="large")
                def show_details_dialog():
                    render_harness_details_dialog(details)

                show_details_dialog()

    else:
        st.warning("No harnesses found in your account")

    # Inference parameters
    st.markdown("##### Inference Parameters")
    with st.container(border=True):
        selected_model = st.selectbox(
            "Model",
            options=HARNESS_MODEL_IDS,
            index=HARNESS_MODEL_IDS.index(st.session_state.harness_model_id) if st.session_state.harness_model_id in HARNESS_MODEL_IDS else 0,
            key="harness_model_selectbox"
        )
        st.session_state.harness_model_id = selected_model
        model_cfg = HARNESS_MODEL_CONFIGS[selected_model]
        if model_cfg.get("temperature_supported", False):
            override_temp = st.checkbox("Override Temperature", value=False, key="harness_override_temperature")
            if override_temp:
                temp_default = model_cfg.get("temperature_default", 0.1)
                st.slider("Temperature", min_value=0.0, max_value=1.0, value=temp_default, step=0.05, key="harness_temperature")
            else:
                st.session_state["harness_temperature"] = None
        else:
            st.session_state["harness_temperature"] = None
            st.markdown(":orange[Temperature not supported for this model]")

        mt_default = st.session_state.harness_configured_max_tokens or 4096
        override_mt = st.checkbox("Override Max Tokens", value=False, key="harness_override_max_tokens")
        if override_mt:
            st.number_input("Max Tokens", min_value=1, max_value=64000, value=mt_default, step=256, key="harness_max_tokens")
        else:
            st.session_state["harness_max_tokens"] = None

    st.markdown("##### System Prompt")
    st.markdown("""<style>
    button[data-testid="harness_reset_system_prompt"] {
        background: none; border: none; color: #4da6ff;
        padding: 0; font-size: 0.8rem; cursor: pointer; text-decoration: underline;
    }
    </style>""", unsafe_allow_html=True)
    with st.container(border=True):
        configured = st.session_state.harness_configured_system_prompt
        cb_col, btn_col = st.columns([4, 1])
        override_sp = cb_col.checkbox("Override System Prompt", value=st.session_state.harness_override_system_prompt, key="harness_override_system_prompt_cb")
        st.session_state.harness_override_system_prompt = override_sp
        if override_sp:
            if configured:
                if btn_col.button("↩ reset", key="harness_reset_system_prompt", type="tertiary"):
                    st.session_state["harness_system_prompt_input"] = configured
            if "harness_system_prompt_input" not in st.session_state:
                st.session_state["harness_system_prompt_input"] = configured or ""
            elif not st.session_state["harness_system_prompt_input"] and configured:
                st.session_state["harness_system_prompt_input"] = configured
            st.text_area(
                "System Prompt",
                height=150,
                key="harness_system_prompt_input",
                label_visibility="collapsed",
                placeholder="Enter system prompt override...",
            )
            st.session_state.harness_system_prompt = st.session_state["harness_system_prompt_input"]
        else:
            st.session_state.harness_system_prompt = ""
            st.markdown(":blue[Using harness default system prompt]")

    # Debug: Show current memory ID state
    logger.info(f"Current memory ID state: {st.session_state.get('harness_memory_id', 'Not set')}")

    render_session_info_panel(
        st.session_state.harness_session_id,
        st.session_state.harness_user_id,
        st.session_state.harness_memory_id
    )

    # Memory query buttons (in Session Info region, but need to be after harness selection)
    if available_harnesses and st.session_state.harness_memory_id:
        if st.button("📜 View Memory Events", use_container_width=True, type="secondary"):
            @st.dialog("Memory Events", width="large")
            def show_memory_events_dialog():
                def list_events_wrapper(**kwargs):
                    return list_memory_events(bedrock_agentcore, **kwargs)

                render_memory_events_dialog(
                    list_events_wrapper,
                    st.session_state.harness_memory_id,
                    st.session_state.harness_session_id,
                    st.session_state.harness_user_id
                )

            show_memory_events_dialog()

        if st.button("📚 View Long-Term Memory", use_container_width=True, type="secondary"):
            @st.dialog("Long-Term Memory Events", width="large")
            def show_long_term_memory_dialog():
                def list_long_term_memory_wrapper(**kwargs):
                    return list_long_term_memory_events(bedrock_agentcore, **kwargs)

                def query_long_term_memory_wrapper(**kwargs):
                    return query_long_term_memory(bedrock_agentcore, **kwargs)

                def get_namespaces_wrapper(**kwargs):
                    return get_memory_namespaces(bedrock_agentcore_control, **kwargs)

                render_long_term_memory_dialog(
                    list_long_term_memory_wrapper,
                    query_long_term_memory_wrapper,
                    get_namespaces_wrapper,
                    st.session_state.harness_memory_id,
                    st.session_state.harness_user_id,
                    st.session_state.harness_session_id
                )

            show_long_term_memory_dialog()

        if st.button("🗂️ View Sessions", use_container_width=True, type="secondary"):
            @st.dialog("Sessions", width="large")
            def show_sessions_dialog():
                def list_sessions_wrapper(**kwargs):
                    return list_sessions(
                        bedrock_agentcore,
                        st.session_state.harness_memory_id,
                        st.session_state.harness_user_id,
                        **kwargs
                    )

                render_sessions_dialog(
                    list_sessions_wrapper,
                    st.session_state.harness_memory_id,
                    st.session_state.harness_user_id,
                    st.session_state.harness_session_id
                )

            show_sessions_dialog()

    # Display latest usage stats
    if st.session_state.latest_usage_stats:
        #st.divider()
        st.markdown("##### Latest Response Stats")
        usage = st.session_state.latest_usage_stats
        st.markdown(f":blue[In: {usage.get('inputTokens', 0)} | Out: {usage.get('outputTokens', 0)} | Total: {usage.get('totalTokens', 0)}]")

    # Display latest tool use
    if st.session_state.latest_tool_use:
        #st.divider()
        st.markdown("##### Latest Tool Use")
        with st.container(border=True):
            for idx, tool in enumerate(st.session_state.latest_tool_use, 1):
                st.markdown(f":blue[**{idx}. {tool['name']}**]")
                if tool['id'] != 'N/A':
                    st.markdown(f":blue[ID: `{tool['id']}`]")
                if tool['input']:
                    with st.expander("View Input"):
                        try:
                            st.json(json.loads(tool['input']))
                        except:
                            st.code(tool['input'])

    if st.session_state.selected_harness_arn:
        if st.button("⚡ Run Command", use_container_width=True, type="secondary"):
            @st.dialog("Run Command", width="large")
            def show_custom_environment_dialog():
                def invoke_runtime_command_wrapper(**kwargs):
                    return invoke_runtime_command(
                        bedrock_agentcore,
                        st.session_state.selected_harness_arn,
                        st.session_state.harness_session_id,
                        **kwargs
                    )

                render_custom_environment_dialog(
                    invoke_runtime_command_wrapper,
                    st.session_state.selected_harness_arn,
                    st.session_state.harness_session_id,
                    st.session_state.harness_user_id
                )

            show_custom_environment_dialog()

    st.divider()
    if st.button("🔄 New Session", use_container_width=True, type="secondary"):
        # End current session if exists
        if st.session_state.harness_session_id:
            with st.spinner("Ending current session..."):
                success = end_agent_session(st.session_state.harness_session_id)
                if success:
                    st.toast("✅ Session ended")
                else:
                    st.toast("⚠️ Session may already be expired")

        # Clear messages, tool use, stats, and generate new session ID
        st.session_state.harness_messages = []
        st.session_state.latest_tool_use = []
        st.session_state.latest_usage_stats = None
        st.session_state.harness_session_id = str(uuid.uuid4())
        st.rerun()

# Main chat area
if not st.session_state.selected_harness_arn:
    st.info("👈 Please select a harness from the sidebar to start chatting")
else:
    for message in st.session_state.harness_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.harness_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        messages = [{'role': 'user', 'content': [{'text': prompt}]}]

        with st.chat_message("assistant"):
            status_area = st.empty()
            result_area = st.empty()
            full_response = ""
            tool_uses = []
            current_tool_input = ""

            try:
                with st.spinner("Waiting for response..."):
                    response = invoke_harness_stream(
                        bedrock_agentcore,
                        st.session_state.selected_harness_arn,
                        st.session_state.harness_session_id,
                        st.session_state.harness_user_id,
                        messages,
                        model_id=st.session_state.harness_model_id,
                        temperature=st.session_state.get("harness_temperature"),
                        max_tokens=st.session_state.get("harness_max_tokens"),
                        system_prompt=st.session_state.harness_system_prompt or None,
                    )

                event_stream = response.get('stream', [])

                for event in event_stream:
                    # Handle contentBlockStart for tool use
                    if 'contentBlockStart' in event:
                        start = event['contentBlockStart'].get('start', {})
                        if 'toolUse' in start:
                            tool_use = start['toolUse']
                            tool_name = tool_use.get('name', 'Unknown')
                            tool_use_id = tool_use.get('toolUseId', 'N/A')
                            tool_uses.append({
                                'name': tool_name,
                                'id': tool_use_id,
                                'input': None
                            })
                            current_tool_input = ""
                            # Show initial status
                            status_area.markdown(f":blue[Calling **{tool_name}** ...]")

                    # Handle contentBlockDelta events
                    elif 'contentBlockDelta' in event:
                        delta = event['contentBlockDelta'].get('delta', {})
                        if 'text' in delta:
                            text_chunk = delta['text']
                            full_response += text_chunk
                            result_area.markdown(full_response + "▌")
                        elif 'toolUse' in delta:
                            # Collect tool input
                            tool_delta = delta['toolUse']
                            if 'input' in tool_delta and tool_uses:
                                input_chunk = tool_delta['input']
                                current_tool_input += input_chunk
                                if tool_uses[-1]['input'] is None:
                                    tool_uses[-1]['input'] = input_chunk
                                else:
                                    tool_uses[-1]['input'] += input_chunk

                                # Update status with input preview
                                try:
                                    input_preview = json.loads(current_tool_input)
                                    input_str = json.dumps(input_preview, indent=None)
                                    if len(input_str) > 100:
                                        input_str = input_str[:100] + "..."
                                    status_area.markdown(f":blue[Calling **{tool_uses[-1]['name']}** {input_str}]")
                                except:
                                    # Not valid JSON yet, show what we have
                                    preview = current_tool_input[:100]
                                    if len(current_tool_input) > 100:
                                        preview += "..."
                                    status_area.markdown(f":blue[Calling **{tool_uses[-1]['name']}** {preview}]")

                    # Handle contentBlockStop for tool use
                    elif 'contentBlockStop' in event:
                        # Clear status when tool completes
                        status_area.empty()

                    # Handle messageStop event
                    elif 'messageStop' in event:
                        stop_reason = event['messageStop'].get('stopReason', 'end_turn')
                        if stop_reason == 'tool_use':
                            # Clear status for tool_use stop
                            status_area.empty()

                    # Handle metadata
                    elif 'metadata' in event:
                        metadata = event['metadata']
                        if 'usage' in metadata:
                            # Store usage stats for sidebar display
                            st.session_state.latest_usage_stats = metadata['usage']

                # Clear status and final update without cursor
                status_area.empty()
                result_area.markdown(full_response or "No response received")

                # Store tool use info for latest exchange
                st.session_state.latest_tool_use = tool_uses

                # Add assistant message to history
                st.session_state.harness_messages.append({
                    "role": "assistant",
                    "content": full_response or "No response received"
                })

                st.rerun()

            except ClientError as e:
                error_message = f"❌ Error: {e.response['Error']['Message']}"
                logger.error(error_message)
                st.error(error_message)
            except Exception as e:
                error_message = f"❌ Unexpected error: {str(e)}"
                logger.error(error_message)
                st.error(error_message)

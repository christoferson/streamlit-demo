import streamlit as st
import boto3
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Driver Risk Assistant",
    page_icon="🚗",
    layout="wide"
)

model_id_list = [
    "amazon.nova-pro-v1:0",
    "google.gemma-3-27b-it",
    "qwen.qwen3-next-80b-a3b",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
]

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client():
    return boto3.client('bedrock-runtime', region_name='us-east-1')

bedrock_runtime = get_bedrock_client()

# Title and description
st.title("🚗 Driver Risk Assistant")
st.markdown("**Real-time telemetry analysis using Amazon Bedrock with JSON Mode**")
# Create tabs
tab1, tab2 = st.tabs(["🔄 Streaming Mode", "📋 Structured Output (Tool Use)"])

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_id = st.selectbox(
        "Model",
        model_id_list,
        index=0,
        help="Select Claude model"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 50)

    st.divider()
    st.markdown("### About")
    st.info("Compare streaming responses vs structured output with enforced JSON schema using tool use.")

# Shared telemetry inputs function
def render_telemetry_inputs():
    col1, col2 = st.columns(2)

    with col1:
        speed = st.number_input("Speed (mph)", min_value=0, max_value=150, value=85, key=f"speed_{st.session_state.get('tab_key', 'default')}")
        speed_limit = st.number_input("Speed Limit (mph)", min_value=0, max_value=100, value=65, key=f"speed_limit_{st.session_state.get('tab_key', 'default')}")
        hard_braking = st.number_input("Hard Braking Events", min_value=0, max_value=20, value=3, key=f"hard_braking_{st.session_state.get('tab_key', 'default')}")
        rapid_acceleration = st.number_input("Rapid Acceleration Events", min_value=0, max_value=20, value=2, key=f"rapid_accel_{st.session_state.get('tab_key', 'default')}")

    with col2:
        sharp_turns = st.number_input("Sharp Turns", min_value=0, max_value=20, value=1, key=f"sharp_turns_{st.session_state.get('tab_key', 'default')}")
        driver_rating = st.selectbox("Driver Rating", [1, 2, 3, 4, 5], index=1, key=f"driver_rating_{st.session_state.get('tab_key', 'default')}")
        trip_duration = st.number_input("Trip Duration (minutes)", min_value=1, max_value=300, value=45, key=f"trip_duration_{st.session_state.get('tab_key', 'default')}")
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"], key=f"time_of_day_{st.session_state.get('tab_key', 'default')}")

    return {
        "speed_mph": speed,
        "speed_limit_mph": speed_limit,
        "speed_over_limit": speed - speed_limit,
        "hard_braking_events": hard_braking,
        "rapid_acceleration_events": rapid_acceleration,
        "sharp_turns": sharp_turns,
        "driver_rating": driver_rating,
        "trip_duration_minutes": trip_duration,
        "time_of_day": time_of_day
    }

# Display results function
def display_results(risk_assessment, latency_info, token_info, mode="streaming"):
    # Success message
    if mode == "streaming":
        st.success(f"✅ Analysis complete - Total: {latency_info['total']:.0f}ms | First Token: {latency_info['first_token']:.0f}ms")
    else:
        st.success(f"✅ Analysis complete in {latency_info['total']:.0f}ms")

    # Create columns for key metrics
    if mode == "streaming":
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    else:
        metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        risk_level = risk_assessment.get('risk_level', 'unknown').upper()
        color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk_level, "⚪")
        st.metric("Risk Level", f"{color} {risk_level}")

    with metric_col2:
        risk_score = risk_assessment.get('risk_score', 0)
        st.metric("Risk Score", f"{risk_score}/100")

    with metric_col3:
        st.metric("Total Time", f"{latency_info['total']:.0f}ms")

    if mode == "streaming":
        with metric_col4:
            st.metric("First Token", f"{latency_info['first_token']:.0f}ms")

    # Display insight
    st.subheader("💡 Risk Insight")
    st.info(risk_assessment.get('insight', 'No insight provided'))

    # Display factors and recommendations in columns
    factor_col, rec_col = st.columns(2)

    with factor_col:
        st.subheader("⚠️ Risk Factors")
        factors = risk_assessment.get('factors', [])
        if factors:
            for factor in factors:
                st.markdown(f"- {factor}")
        else:
            st.write("No specific factors identified")

    with rec_col:
        st.subheader("✅ Recommendations")
        recommendations = risk_assessment.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.write("No recommendations provided")

    # Display full JSON response
    with st.expander("🔍 View Full JSON Response"):
        st.json(risk_assessment)

    # Display API metadata
    with st.expander("📊 API Response Metadata"):
        metadata = {
            "model_id": model_id,
            "mode": mode,
            "total_latency_ms": round(latency_info['total'], 2),
            "input_tokens": token_info['input'],
            "output_tokens": token_info['output'],
            "total_tokens": token_info['input'] + token_info['output'],
        }

        if mode == "streaming":
            metadata["first_token_latency_ms"] = round(latency_info['first_token'], 2)

        st.json(metadata)

        # Calculate approximate cost
        if "claude-3-haiku" in model_id:
            input_cost = token_info['input'] * 0.00025 / 1000
            output_cost = token_info['output'] * 0.00125 / 1000
        elif "claude-3-5-sonnet" in model_id:
            input_cost = token_info['input'] * 0.003 / 1000
            output_cost = token_info['output'] * 0.015 / 1000
        else:
            input_cost = token_info['input'] * 0.0003 / 1000
            output_cost = token_info['output'] * 0.0006 / 1000

        total_cost = input_cost + output_cost
        st.metric("Estimated Cost", f"${total_cost:.6f}")

# TAB 1: Streaming Mode
with tab1:
    st.session_state['tab_key'] = 'streaming'
    st.subheader("📊 Telemetry Input")
    telemetry_data_stream = render_telemetry_inputs()

    st.subheader("📝 Prompt Template")
    prompt_template_stream = st.text_area(
        "System Prompt",
        value="""You are a driver risk assessment system. Analyze the telemetry data and provide a structured risk assessment in JSON format.

The output must include:
- risk_level: "low", "medium", or "high"
- risk_score: numeric value from 0-100
- insight: brief natural language explanation (1-2 sentences)
- factors: array of contributing risk factors
- recommendations: array of safety recommendations

Telemetry Data:
{telemetry}

Provide your assessment in valid JSON format only, with no additional text.""",
        height=300,
        key="prompt_stream"
    )

    if st.button("🔍 Generate Risk Assessment (Streaming)", type="primary", use_container_width=True, key="btn_stream"):
        formatted_prompt = prompt_template_stream.format(
            telemetry=json.dumps(telemetry_data_stream, indent=2)
        )

        with st.expander("📋 View Formatted Telemetry Data"):
            st.json(telemetry_data_stream)

        st.subheader("🔄 Streaming Response")
        stream_placeholder = st.empty()

        start_time = datetime.now()
        accumulated_text = ""
        first_token_time = None
        input_tokens = 0
        output_tokens = 0

        try:
            request_params = {
                "modelId": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": formatted_prompt}]
                    }
                ],
                "inferenceConfig": {
                    "temperature": temperature,
                    "maxTokens": max_tokens
                }
            }

            response = bedrock_runtime.converse_stream(**request_params)

            for event in response['stream']:
                if 'contentBlockDelta' in event:
                    delta = event['contentBlockDelta']['delta']
                    if 'text' in delta:
                        chunk = delta['text']
                        accumulated_text += chunk

                        if first_token_time is None:
                            first_token_time = datetime.now()

                        stream_placeholder.code(accumulated_text, language="json")

                elif 'metadata' in event:
                    usage = event['metadata'].get('usage', {})
                    input_tokens = usage.get('inputTokens', 0)
                    output_tokens = usage.get('outputTokens', 0)

            end_time = datetime.now()
            total_latency = (end_time - start_time).total_seconds() * 1000
            first_token_latency = (first_token_time - start_time).total_seconds() * 1000 if first_token_time else 0

            output_text = accumulated_text
            if "```json" in output_text:
                output_text = output_text.split("```json")[1].split("```")[0].strip()
            elif "```" in output_text:
                output_text = output_text.split("```")[1].split("```")[0].strip()

            risk_assessment = json.loads(output_text)

            display_results(
                risk_assessment,
                {"total": total_latency, "first_token": first_token_latency},
                {"input": input_tokens, "output": output_tokens},
                mode="streaming"
            )

        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Parsing Error: {str(e)}")
            st.code(accumulated_text, language="text")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# TAB 2: Structured Output with Tool Use
with tab2:
    st.session_state['tab_key'] = 'structured'
    st.subheader("📊 Telemetry Input")
    telemetry_data_struct = render_telemetry_inputs()

    st.subheader("📐 JSON Schema Definition")

    # Define the tool/function schema
    tool_schema = {
        "name": "generate_risk_assessment",
        "description": "Generate a structured driver risk assessment based on telemetry data",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Overall risk classification"
                    },
                    "risk_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Numeric risk score from 0-100"
                    },
                    "insight": {
                        "type": "string",
                        "description": "Brief natural language explanation of the risk assessment"
                    },
                    "factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of contributing risk factors"
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of safety recommendations"
                    }
                },
                "required": ["risk_level", "risk_score", "insight", "factors", "recommendations"]
            }
        }
    }

    # Display the schema
    with st.expander("🔍 View Tool Schema", expanded=True):
        st.json(tool_schema)

    # Allow editing of the schema
    schema_editable = st.checkbox("Edit Schema", value=False)
    if schema_editable:
        schema_text = st.text_area(
            "Edit JSON Schema",
            value=json.dumps(tool_schema, indent=2),
            height=400,
            key="schema_edit"
        )
        try:
            tool_schema = json.loads(schema_text)
        except json.JSONDecodeError:
            st.error("Invalid JSON schema")

    st.subheader("📝 User Prompt")
    user_prompt_struct = st.text_area(
        "Prompt",
        value="""Analyze the following driver telemetry data and provide a risk assessment:

{telemetry}

Consider factors like speeding, aggressive driving behaviors, time of day, and driver rating.""",
        height=200,
        key="prompt_struct"
    )

    if st.button("🔍 Generate Risk Assessment (Structured)", type="primary", use_container_width=True, key="btn_struct"):
        formatted_prompt = user_prompt_struct.format(
            telemetry=json.dumps(telemetry_data_struct, indent=2)
        )

        with st.expander("📋 View Formatted Telemetry Data"):
            st.json(telemetry_data_struct)

        with st.spinner("Analyzing telemetry data with structured output..."):
            start_time = datetime.now()

            try:
                # Use tool/function calling for structured output
                response = bedrock_runtime.converse(
                    modelId=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": formatted_prompt}]
                        }
                    ],
                    inferenceConfig={
                        "temperature": temperature,
                        "maxTokens": max_tokens
                    },
                    toolConfig={
                        "tools": [
                            {
                                "toolSpec": tool_schema
                            }
                        ],
                        "toolChoice": {
                            "tool": {
                                "name": "generate_risk_assessment"
                            }
                        }
                    }
                )

                end_time = datetime.now()
                total_latency = (end_time - start_time).total_seconds() * 1000

                # Extract structured output from tool use
                content = response['output']['message']['content']

                # Find the tool use block
                tool_use_block = None
                for block in content:
                    if 'toolUse' in block:
                        tool_use_block = block['toolUse']
                        break

                if tool_use_block:
                    risk_assessment = tool_use_block['input']

                    # Get token usage
                    input_tokens = response['usage']['inputTokens']
                    output_tokens = response['usage']['outputTokens']

                    st.success("✅ Structured output generated with enforced schema!")

                    # Show that schema was enforced
                    st.info("🎯 **Schema Enforcement**: The model output is guaranteed to match the defined JSON schema structure.")

                    display_results(
                        risk_assessment,
                        {"total": total_latency, "first_token": 0},
                        {"input": input_tokens, "output": output_tokens},
                        mode="structured"
                    )

                    # Show the raw tool use response
                    with st.expander("🔧 View Raw Tool Use Response"):
                        st.json({
                            "toolUseId": tool_use_block.get('toolUseId'),
                            "name": tool_use_block.get('name'),
                            "input": tool_use_block.get('input')
                        })
                else:
                    st.error("No tool use found in response")
                    st.json(response)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Amazon Bedrock | Streaming & Structured Output Comparison</p>
</div>
""", unsafe_allow_html=True)
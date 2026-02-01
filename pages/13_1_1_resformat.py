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

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client():
    return boto3.client('bedrock-runtime', region_name='us-east-1')

bedrock_runtime = get_bedrock_client()

# Title and description
st.title("🚗 Driver Risk Assistant")
st.markdown("**Real-time telemetry analysis using Amazon Bedrock with JSON Mode**")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_id = st.selectbox(
        "Model",
        [
            "global.anthropic.claude-haiku-4-5-20251001-v1:0",

            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",

        ],
        index=0,
        help="Titan Text Premier is the latest Titan model with JSON mode support"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 2000, 500, 50)

    st.divider()
    st.markdown("### About")
    st.info("This demo uses Amazon Bedrock's JSON mode to generate structured risk assessments from driver telemetry data.")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Telemetry Input")

    # Telemetry inputs
    speed = st.number_input("Speed (mph)", min_value=0, max_value=150, value=85)
    speed_limit = st.number_input("Speed Limit (mph)", min_value=0, max_value=100, value=65)

    hard_braking = st.number_input("Hard Braking Events", min_value=0, max_value=20, value=3)
    rapid_acceleration = st.number_input("Rapid Acceleration Events", min_value=0, max_value=20, value=2)

    sharp_turns = st.number_input("Sharp Turns", min_value=0, max_value=20, value=1)
    driver_rating = st.selectbox("Driver Rating", [1, 2, 3, 4, 5], index=1)

    trip_duration = st.number_input("Trip Duration (minutes)", min_value=1, max_value=300, value=45)
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

with col2:
    st.subheader("📝 Prompt Template")

    prompt_template = st.text_area(
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
        height=500
    )

# Generate button
if st.button("🔍 Generate Risk Assessment", type="primary", use_container_width=True):

    # Prepare telemetry data
    telemetry_data = {
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

    # Format prompt
    formatted_prompt = prompt_template.format(
        telemetry=json.dumps(telemetry_data, indent=2)
    )

    # Display telemetry summary
    with st.expander("📋 View Formatted Telemetry Data"):
        st.json(telemetry_data)

    # Streaming response
    st.subheader("🔄 Streaming Response")

    # Create placeholder for streaming text
    stream_placeholder = st.empty()

    start_time = datetime.now()
    accumulated_text = ""
    first_token_time = None
    input_tokens = 0
    output_tokens = 0

    try:
        # Prepare base request
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

        # Call Bedrock with streaming
        response = bedrock_runtime.converse_stream(**request_params)

        # Process streaming response
        for event in response['stream']:
            if 'messageStart' in event:
                # Message started
                pass

            elif 'contentBlockStart' in event:
                # Content block started
                pass

            elif 'contentBlockDelta' in event:
                # New content chunk
                delta = event['contentBlockDelta']['delta']
                if 'text' in delta:
                    chunk = delta['text']
                    accumulated_text += chunk

                    # Record first token time
                    if first_token_time is None:
                        first_token_time = datetime.now()

                    # Update streaming display
                    stream_placeholder.code(accumulated_text, language="json")

            elif 'messageStop' in event:
                # Message completed
                stop_reason = event['messageStop'].get('stopReason', 'unknown')

            elif 'metadata' in event:
                # Extract token usage
                usage = event['metadata'].get('usage', {})
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)

        end_time = datetime.now()
        total_latency = (end_time - start_time).total_seconds() * 1000
        first_token_latency = (first_token_time - start_time).total_seconds() * 1000 if first_token_time else 0

        # Parse JSON (handle potential markdown code blocks)
        output_text = accumulated_text
        if "```json" in output_text:
            output_text = output_text.split("```json")[1].split("```")[0].strip()
        elif "```" in output_text:
            output_text = output_text.split("```")[1].split("```")[0].strip()

        risk_assessment = json.loads(output_text)

        # Display results
        st.success(f"✅ Analysis complete - Total: {total_latency:.0f}ms | First Token: {first_token_latency:.0f}ms")

        # Create columns for key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            risk_level = risk_assessment.get('risk_level', 'unknown').upper()
            color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk_level, "⚪")
            st.metric("Risk Level", f"{color} {risk_level}")

        with metric_col2:
            risk_score = risk_assessment.get('risk_score', 0)
            st.metric("Risk Score", f"{risk_score}/100")

        with metric_col3:
            st.metric("Total Time", f"{total_latency:.0f}ms")

        with metric_col4:
            st.metric("First Token", f"{first_token_latency:.0f}ms")

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
                "total_latency_ms": round(total_latency, 2),
                "first_token_latency_ms": round(first_token_latency, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "streaming_enabled": True
            }
            st.json(metadata)

            # Calculate approximate cost
            if "claude-3-haiku" in model_id:
                input_cost = input_tokens * 0.00025 / 1000
                output_cost = output_tokens * 0.00125 / 1000
            elif "claude-3-5-sonnet" in model_id:
                input_cost = input_tokens * 0.003 / 1000
                output_cost = output_tokens * 0.015 / 1000
            else:
                input_cost = input_tokens * 0.0003 / 1000
                output_cost = output_tokens * 0.0006 / 1000

            total_cost = input_cost + output_cost
            st.metric("Estimated Cost", f"${total_cost:.6f}")

    except json.JSONDecodeError as e:
        st.error(f"❌ JSON Parsing Error: {str(e)}")
        st.code(accumulated_text, language="text")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Amazon Bedrock | Streaming API | Real-time Response</p>
</div>
""", unsafe_allow_html=True)
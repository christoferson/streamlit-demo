import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
import os
from io import BytesIO
import sys
import subprocess
from contextlib import closing
from tempfile import gettempdir

from pydub import AudioSegment
from pydub.playback import play

from botocore.exceptions import BotoCoreError, ClientError
import cmn.cloudwatch_metrics_lib
from cmn.bedrock_models import FOUNDATION_MODELS, FoundationModel, Provider
import random

AWS_REGION = cmn_settings.AWS_REGION
AWS_BEDROCK_GUARDRAIL_IDENTIFIER = cmn_settings.AWS_BEDROCK_GUARDRAIL_IDENTIFIER
AWS_BEDROCK_GUARDRAIL_VERSION = cmn_settings.AWS_BEDROCK_GUARDRAIL_VERSION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

dummy_user_list = ["Tom", "Fred"]

####################################################################################

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_runtime_us_west_2 = boto3.client('bedrock-runtime', region_name="us-west-2")
polly = boto3.client("polly", region_name=AWS_REGION)

####################################################################################

st.set_page_config(
    page_title="Knowledge Base",
    page_icon="🧊",
    layout="centered", # "centered" or "wide"
    initial_sidebar_state="expanded", #"auto", "expanded", or "collapsed"
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.logo(icon_image="images/logo.png", image="images/logo_text.png")

st.markdown(
    """
    <style>
    button[kind="primary"] {
        background: none!important;
        border: none;
        padding: 0!important;
        margin: 0;
        color: black !important;
        text-decoration: none;
        cursor: pointer;
        border: none !important;
    }
    button[kind="primary"]:hover {
        text-decoration: none;
        color: black !important;
    }
    button[kind="primary"]:focus {
        outline: none !important;
        box-shadow: none !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Play Audio Button stAudio
st.markdown(
    """
    <style>
    #.stAudio {
    #    max-width: 70px;
    #    max-height: 50px;
    #}
    #audio::-webkit-media-controls-time-remaining-display,
    #audio::-webkit-media-controls-current-time-display {
    #    max-width: 50%;
    #    max-height: 20px;
    #}
    </style>
    """,
    unsafe_allow_html=True,
)




def recite_button_clicked(text):
    try:
        # Request speech synthesis
        response = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId="Joanna")
    except (BotoCoreError, ClientError) as error:
        print(error)
        return

    if "AudioStream" in response:
        # Note: Closing the stream is important because the service throttles on the
        # number of parallel connections. Here we are using contextlib.closing to
        # ensure the close method of the stream object will be called automatically
        # at the end of the with statement's scope.
            with closing(response["AudioStream"]) as stream:
                output = os.path.join(gettempdir(), "speech.mp3")
                try:
                    # Open a file for writing the output as a binary stream
                    sound = stream.read()
                    with open(output, "wb") as file:
                        file.write(sound)
                    
                    st.session_state['audio_stream'] = sound

                    #data = open(output, 'rb').read()
                    #song = AudioSegment.from_file(BytesIO(data), format="mp3")
                    #play(song)
                except IOError as error:
                    # Could not write to file, exit gracefully
                    print(error)
                    #sys.exit(-1)
                    st.session_state['audio_stream'] = ""
                    return
                
                print("**********************************************************************")
                try:                 
                    print(f"/n/n---------------------------------------------------------/n{output}")   
                    #data = open(output, 'rb').read()
                    #print("------")
                    #song = AudioSegment.from_file(BytesIO(data), format="mp3")
                    #print("------")
                    #play(song)
                except IOError as error:
                    print(error)
                    return                

    else:
        # The response didn't contain audio data, exit gracefully
        print("Could not stream audio")
        #sys.exit(-1)
        return
       

# Models come from FOUNDATION_MODELS (cmn/bedrock_models.py), limited to the
# providers whose native request format invoke_model() implements, and to fast
# and light models: no Opus / Fable, no large-tier models
opt_model_providers = (Provider.ANTHROPIC, Provider.AMAZON, Provider.META, Provider.MISTRAL)
opt_model_excluded = ("opus", "fable", "nova-pro", "70b", "large")
opt_model_list = [
    model for model in FOUNDATION_MODELS.listed_models()
    if model.provider in opt_model_providers
    and not any(excluded in model.model_id for excluded in opt_model_excluded)
]

with st.sidebar:
    opt_fm: FoundationModel = st.selectbox(label="Model", options=opt_model_list, index=0, format_func=lambda model: model.display_name, key="model_id")

opt_model_id = opt_fm.model_id

opt_fm_max_tokens = opt_fm.inference_parameters.max_tokens
opt_fm_temperature = opt_fm.inference_parameters.temperature
opt_fm_top_p = opt_fm.inference_parameters.top_p
opt_fm_top_k = opt_fm.inference_parameters.top_k

# Sliders only for the parameters the model accepts; the rest keep their default
with st.sidebar:
    if opt_fm_temperature.is_supported:
        opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    else:
        opt_temperature = opt_fm_temperature.default_value
    if opt_fm_top_p.is_supported:
        opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    else:
        opt_top_p = opt_fm_top_p.default_value
    if opt_fm_top_k.is_supported:
        opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    else:
        opt_top_k = opt_fm_top_k.default_value
    opt_max_tokens = st.slider(label="Max Tokens", min_value=opt_fm_max_tokens.min_value, max_value=opt_fm_max_tokens.max_value, value=opt_fm_max_tokens.default_value, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")

# Invocation metrics of the last response, shown at the bottom of the sidebar
with st.sidebar:
    metrics_area = st.empty()

if st.session_state.get("last_metrics"):
    with metrics_area.container(border=True):
        for line in st.session_state["last_metrics"]:
            st.write(line)

st.title("💬 Chatbot 3")
st.write("Ask LLM Questions")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        #{"role": "user", "content": "Hello there."},
        #{"role": "assistant", "content": "How can I help you?"}
    ]

#if "audio_stream" not in st.session_state:
#    st.session_state["audio_stream"] = ""

st.chat_message("system").markdown(""":red[Hi, Enter your questions.
                                It can be a simple question or complex one.]""")

idx = 1
for msg in st.session_state.messages:
    idx = idx + 1
    content = msg["content"]
    with st.chat_message(msg["role"]):
        st.write(content)
        #if "assistant" == msg["role"]:
            #assistant_cmd_panel_col1, assistant_cmd_panel_col2, assistant_cmd_panel_col3 = st.columns([0.07,0.23,0.7], gap="small")
            #with assistant_cmd_panel_col2:
            

def invoke_model(prompt, message_history, opt_model_id, opt_temperature, opt_top_p, opt_top_k, opt_max_tokens, opt_system_msg):
    """
    Invokes the selected model with the given parameters and returns the response.
    Handles different model providers (Anthropic Claude, Amazon Nova, etc.)
    """

    # Base request structure
    request = {
        "temperature": opt_temperature,
        "top_p": opt_top_p,
        "max_tokens": opt_max_tokens
    }

    # Handle different model types
    if "anthropic.claude" in opt_model_id:
        # Anthropic Claude models
        request.update({
            "anthropic_version": "bedrock-2023-05-31",
            "top_k": opt_top_k,
            "system": opt_system_msg,
            "messages": message_history
        })
        # Newer Claude models reject some sampling parameters (e.g. Claude 5.x accepts none)
        opt_fm = FOUNDATION_MODELS.find(opt_model_id)
        if opt_fm:
            params = opt_fm.inference_parameters
            for param in (params.temperature, params.top_p, params.top_k):
                if not param.is_supported:
                    request.pop(param.name, None)
    elif "amazon.nova" in opt_model_id:
        # Amazon Nova models require a different message format
        nova_messages = []

        for msg in message_history:
            role = msg["role"]
            content_text = msg["content"]

            # Format content as an array with text object
            nova_content = [{"text": content_text}]

            nova_messages.append({
                "role": role,
                "content": nova_content
            })

        request = {
            "messages": nova_messages,
            "inferenceConfig": {
                "temperature": opt_temperature,
                "topP": opt_top_p,
                "maxTokens": opt_max_tokens
            }
        }

        # Add system message if provided
        if opt_system_msg:
            request["system"] = [{"text": opt_system_msg}]
    elif "meta.llama" in opt_model_id:
        # Meta Llama models
        # Construct conversation history in Llama format
        # Llama 4 uses different header / end-of-turn tokens than Llama 3
        if "llama4" in opt_model_id:
            header_start, header_end, end_of_turn = "<|header_start|>", "<|header_end|>", "<|eot|>"
        else:
            header_start, header_end, end_of_turn = "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"

        formatted_prompt = "<|begin_of_text|>"

        # Add system message if provided
        if opt_system_msg:
            formatted_prompt += f"{header_start}system{header_end}\n\n{opt_system_msg}{end_of_turn}"

        # Add message history
        for msg in message_history:
            role = msg["role"]
            content = msg["content"]

            formatted_prompt += f"{header_start}{role}{header_end}\n\n{content}{end_of_turn}"

        # Add the assistant header for the response
        formatted_prompt += f"{header_start}assistant{header_end}\n\n"

        # Create the Llama-specific request format
        request = {
            "prompt": formatted_prompt,
            "max_gen_len": opt_max_tokens,
            "temperature": opt_temperature,
            "top_p": opt_top_p
        }

        # Add top_k if specified
        #if opt_top_k > 0:
        #    request["top_k"] = opt_top_k
    elif "mistral" in opt_model_id:
        # Mistral models
        formatted_prompt = "<s>[INST] "

        # Add system message if provided
        if opt_system_msg:
            formatted_prompt += f"{opt_system_msg}\n"

        # Add message history
        for msg in message_history:
            if msg["role"] == "user":
                formatted_prompt += f"{msg['content']}"
            elif msg["role"] == "assistant":
                formatted_prompt += f" [/INST] {msg['content']} </s><s>[INST] "

        # Add final instruction token if needed
        if formatted_prompt.endswith("<s>[INST] "):
            formatted_prompt = formatted_prompt[:-len("<s>[INST] ")]
        elif not formatted_prompt.endswith(" [/INST]"):
            formatted_prompt += " [/INST]"

        request = {
            "prompt": formatted_prompt,
            "max_tokens": opt_max_tokens,
            "temperature": opt_temperature,
            "top_p": opt_top_p,
            "stop": ["</s>"]
        }
    else:
        # Default format for other models
        request.update({
            "top_k": opt_top_k,
            "system": opt_system_msg,
            "messages": message_history
        })

    # Determine which client to use based on region (FoundationModel.region override)
    if FOUNDATION_MODELS.find(opt_model_id).region == "us-west-2":
        client = bedrock_runtime_us_west_2
    else:
        client = bedrock_runtime

    # Invoke model with appropriate parameters
    response = client.invoke_model_with_response_stream(
        modelId=opt_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request),
        #guardrailIdentifier=AWS_BEDROCK_GUARDRAIL_IDENTIFIER,
        #guardrailVersion=AWS_BEDROCK_GUARDRAIL_VERSION
    )

    return response

def process_invoke_response(response, opt_model_id, opt_temperature, opt_top_p, opt_top_k, opt_max_tokens):
    """
    Process the streaming response from different model families.

    Args:
        response: The streaming response from the model
        opt_model_id: The model ID
        opt_temperature: Temperature parameter
        opt_top_p: Top P parameter
        opt_top_k: Top K parameter
        opt_max_tokens: Maximum tokens parameter

    Returns:
        result_text: The generated text
    """
    result_text = ""

    st.session_state["last_metrics"] = []
    metrics_container = metrics_area.container(border=True)

    def write_metrics(line):
        st.session_state["last_metrics"].append(line)
        metrics_container.write(line)

    with st.chat_message("assistant"):
        result_area = st.empty()

        # Display model parameters
        params = FOUNDATION_MODELS.find(opt_model_id).inference_parameters
        sent_params = [
            f"{name}={value}"
            for name, value, param in (
                ("temperature", opt_temperature, params.temperature),
                ("top_p", opt_top_p, params.top_p),
                ("top_k", opt_top_k, params.top_k),
            )
            if param.is_supported
        ]
        opts = "| " + " ".join(sent_params + [f"max_tokens={opt_max_tokens}"])
        write_metrics(opts)

        stream = response["body"]

        # Process based on model family
        if "anthropic.claude" in opt_model_id or "us.anthropic.claude" in opt_model_id:
            # Process Anthropic Claude models
            for event in stream:
                if event["chunk"]:
                    chunk = json.loads(event["chunk"]["bytes"])

                    if chunk['type'] == 'message_start':
                        # Already displayed model parameters
                        pass

                    elif chunk['type'] == 'message_delta':
                        # Process message delta if needed
                        pass

                    elif chunk['type'] == 'content_block_delta':
                        if chunk['delta']['type'] == 'text_delta':
                            text = chunk['delta']['text']
                            result_text += text
                            result_area.write(result_text)

                    elif chunk['type'] == 'message_stop':
                        if 'amazon-bedrock-invocationMetrics' in chunk:
                            invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                            input_token_count = invocation_metrics["inputTokenCount"]
                            output_token_count = invocation_metrics["outputTokenCount"]
                            latency = invocation_metrics["invocationLatency"]
                            lag = invocation_metrics["firstByteLatency"]
                            stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                            write_metrics(stats)

                # Handle exceptions
                elif any(key in event for key in ["internalServerException", "modelStreamErrorException", 
                                                "modelTimeoutException", "throttlingException", "validationException"]):
                    for exception_type in ["internalServerException", "modelStreamErrorException", 
                                          "modelTimeoutException", "throttlingException", "validationException"]:
                        if exception := event.get(exception_type):
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)
                            break

        elif "amazon.nova" in opt_model_id:
            # Process Amazon Nova models
            for event in stream:
                if event["chunk"]:
                    chunk = json.loads(event["chunk"]["bytes"])

                    # Debug the first few chunks to understand the structure
                    # print(f"NOVA CHUNK: {json.dumps(chunk, indent=2)}")

                    # Handle Nova's specific response format
                    if "contentBlockDelta" in chunk and "delta" in chunk["contentBlockDelta"]:
                        if "text" in chunk["contentBlockDelta"]["delta"]:
                            text = chunk["contentBlockDelta"]["delta"]["text"]
                            result_text += text
                            result_area.write(result_text)

                    # Handle completion message with metrics
                    elif "amazon-bedrock-invocationMetrics" in chunk:
                        invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                        input_token_count = invocation_metrics.get("inputTokenCount", "N/A")
                        output_token_count = invocation_metrics.get("outputTokenCount", "N/A")
                        latency = invocation_metrics.get("invocationLatency", "N/A")
                        lag = invocation_metrics.get("firstByteLatency", "N/A")
                        stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                        write_metrics(stats)

                # Handle exceptions
                elif any(key in event for key in ["internalServerException", "modelStreamErrorException", 
                                                "modelTimeoutException", "throttlingException", "validationException"]):
                    for exception_type in ["internalServerException", "modelStreamErrorException", 
                                          "modelTimeoutException", "throttlingException", "validationException"]:
                        if exception := event.get(exception_type):
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)
                            break

        elif "meta.llama" in opt_model_id:
            # Process Meta Llama models
            for event in stream:
                if event["chunk"]:
                    chunk = json.loads(event["chunk"]["bytes"])

                    # Based on the documentation, Llama models return text in the "generation" field
                    if "generation" in chunk:
                        text = chunk["generation"]
                        result_text += text
                        result_area.write(result_text)

                    # Handle metrics if available
                    if "amazon-bedrock-invocationMetrics" in chunk:
                        invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                        input_token_count = invocation_metrics.get("inputTokenCount", "N/A")
                        output_token_count = invocation_metrics.get("outputTokenCount", "N/A")
                        latency = invocation_metrics.get("invocationLatency", "N/A")
                        lag = invocation_metrics.get("firstByteLatency", "N/A")
                        stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                        write_metrics(stats)

                # Handle exceptions
                elif any(key in event for key in ["internalServerException", "modelStreamErrorException", 
                                                "modelTimeoutException", "throttlingException", "validationException"]):
                    for exception_type in ["internalServerException", "modelStreamErrorException", 
                                          "modelTimeoutException", "throttlingException", "validationException"]:
                        if exception := event.get(exception_type):
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)
                            break
        elif "mistral" in opt_model_id:
            # Process Mistral models
            for event in stream:
                if event["chunk"]:
                    chunk = json.loads(event["chunk"]["bytes"])

                    # Extract text from Mistral's response format
                    if "outputs" in chunk:
                        text = chunk["outputs"][0].get("text", "")
                        result_text += text
                        result_area.write(result_text)

                    # Handle metrics if available
                    if "amazon-bedrock-invocationMetrics" in chunk:
                        invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                        input_token_count = invocation_metrics.get("inputTokenCount", "N/A")
                        output_token_count = invocation_metrics.get("outputTokenCount", "N/A")
                        latency = invocation_metrics.get("invocationLatency", "N/A")
                        lag = invocation_metrics.get("firstByteLatency", "N/A")
                        stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                        write_metrics(stats)

                # Handle exceptions
                elif any(key in event for key in ["internalServerException", "modelStreamErrorException", 
                                                "modelTimeoutException", "throttlingException", "validationException"]):
                    for exception_type in ["internalServerException", "modelStreamErrorException", 
                                        "modelTimeoutException", "throttlingException", "validationException"]:
                        if exception := event.get(exception_type):
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)
                            break
        else:
            # Generic handler for other models
            for event in stream:
                if event["chunk"]:
                    chunk = json.loads(event["chunk"]["bytes"])

                    # Debug the first few chunks to understand the structure
                    # print(f"GENERIC CHUNK: {json.dumps(chunk, indent=2)}")

                    # Try to extract text from common patterns
                    if "text" in chunk:
                        text = chunk["text"]
                        result_text += text
                        result_area.write(result_text)
                    elif "content" in chunk:
                        text = chunk["content"]
                        result_text += text
                        result_area.write(result_text)
                    elif "delta" in chunk and isinstance(chunk["delta"], dict):
                        if "text" in chunk["delta"]:
                            text = chunk["delta"]["text"]
                            result_text += text
                            result_area.write(result_text)

                    # Handle metrics if available
                    if "amazon-bedrock-invocationMetrics" in chunk:
                        invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                        input_token_count = invocation_metrics.get("inputTokenCount", "N/A")
                        output_token_count = invocation_metrics.get("outputTokenCount", "N/A")
                        latency = invocation_metrics.get("invocationLatency", "N/A")
                        lag = invocation_metrics.get("firstByteLatency", "N/A")
                        stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                        write_metrics(stats)

                # Handle exceptions
                elif any(key in event for key in ["internalServerException", "modelStreamErrorException", 
                                                "modelTimeoutException", "throttlingException", "validationException"]):
                    for exception_type in ["internalServerException", "modelStreamErrorException", 
                                          "modelTimeoutException", "throttlingException", "validationException"]:
                        if exception := event.get(exception_type):
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)
                            break



    return result_text


if prompt := st.chat_input():
    
    st.session_state["audio_stream"] = ""

    message_history = st.session_state.messages.copy()
    message_history.append({"role": "user", "content": prompt})
    #st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    
    try:

        # Use the invoke_model function to handle different model types
        response = invoke_model(
            prompt=prompt,
            message_history=message_history,
            opt_model_id=opt_model_id,
            opt_temperature=opt_temperature,
            opt_top_p=opt_top_p,
            opt_top_k=opt_top_k,
            opt_max_tokens=opt_max_tokens,
            opt_system_msg=opt_system_msg
        )

        # Process the response based on model family
        result_text = process_invoke_response(
            response=response,
            opt_model_id=opt_model_id,
            opt_temperature=opt_temperature,
            opt_top_p=opt_top_p,
            opt_top_k=opt_top_k,
            opt_max_tokens=opt_max_tokens
        )
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": result_text})


    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)

    
    try:
        cmn.cloudwatch_metrics_lib.cloudwatch_put_metric(
            metric_namespace='App/Chat', 
            metric_name='UserInvocation', 
            metric_value=1,
            dimensions=[{
                'Name': 'User',
                'Value': random.choice(dummy_user_list),
            }])
    except ClientError as err:
            # Usage metric is optional; needs cloudwatch:PutMetricData
            logger.warning("Skipped usage metric: %s", err.response["Error"]["Message"])

if "audio_stream" in st.session_state and st.session_state["audio_stream"] != "":
    audio_bytes = BytesIO(st.session_state['audio_stream'])
    st.audio(audio_bytes, format='audio/mp3', autoplay=False)
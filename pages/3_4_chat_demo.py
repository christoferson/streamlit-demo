import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
import pyperclip
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
       

opt_model_id_list = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-premier-v1:0", #NG
    "deepseek.r1-v1:0", #NG
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0", #NG
    "meta.llama3-2-90b-instruct-v1:0", #NG
    "meta.llama3-3-70b-instruct-v1:0", #NG
    "meta.llama4-maverick-17b-instruct-v1:0", #NG
    "meta.llama4-scout-17b-instruct-v1:0", #NG
    "writer.palmyra-x4-v1:0", #NG
    "writer.palmyra-x5-v1:0", #NG
    "mistral.mistral-large-2402-v1:0",
    "mistral.pixtral-large-2502-v1:0", #NG
    "cohere.command-r-plus-v1:0",
]

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    # Adjust Top P range for Cohere models
    if "cohere" in opt_model_id:
        opt_top_p_max = 0.99
    else:
        opt_top_p_max = 1.0

    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=opt_top_p_max, value=opt_top_p_max, step=0.1, key="top_p")

    
    # Adjust Top K range based on model
    if "mistral" in opt_model_id:
        top_k_max = 200  # Mistral's limit
    else:
        top_k_max = 500  # Default limit

    opt_top_k = st.slider(label="Top K", min_value=0, max_value=top_k_max, value=min(250, top_k_max), step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")

with st.sidebar:
    if "cohere" in opt_model_id:
        return_likelihoods = st.selectbox(
            "Return Likelihoods",
            ["NONE", "GENERATION", "ALL"],
            index=0
        )
        truncate = st.selectbox(
            "Truncate",
            ["NONE", "START", "END"],
            index=2
        )

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

        # Ensure topK is within valid range for Nova (1-128)
        nova_top_k = min(max(1, opt_top_k), 128)

        request = {
            "messages": nova_messages,
            "inferenceConfig": {
                "temperature": opt_temperature,
                "topP": opt_top_p,
                "topK": nova_top_k,
                "maxTokens": opt_max_tokens
            }
        }

        # Add system message if provided
        if opt_system_msg:
            request["system"] = [{"text": opt_system_msg}]
    elif "meta.llama" in opt_model_id:
        # Meta Llama models
        # Construct conversation history in Llama format
        formatted_prompt = "<|begin_of_text|>"

        # Add system message if provided
        if opt_system_msg:
            formatted_prompt += f"<|start_header_id|>system<|end_header_id|>\n{opt_system_msg}\n<|eot_id|>\n"

        # Add message history
        for msg in message_history:
            role = msg["role"]
            content = msg["content"]

            formatted_prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>\n"

        # Add the assistant header for the response
        formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

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

        # Ensure top_k is within Mistral's limits
        mistral_top_k = min(opt_top_k, 200)

        request = {
            "prompt": formatted_prompt,
            "max_tokens": opt_max_tokens,
            "temperature": opt_temperature,
            "top_p": opt_top_p,
            "top_k": mistral_top_k,
            "stop": ["</s>"]
        }
    elif "cohere" in opt_model_id:
        # Cohere models
        message_text = ""

        # Add system message if provided
        if opt_system_msg:
            message_text += f"{opt_system_msg}\n\n"

        # Add message history
        for msg in message_history:
            if msg["role"] == "user":
                message_text += f"{msg['content']}\n"
            elif msg["role"] == "assistant":
                message_text += f"{msg['content']}\n\n"

        request = {
            "message": message_text,
            "max_tokens": opt_max_tokens,
            "temperature": opt_temperature,
            "p": min(opt_top_p, 0.99),  # Ensure p is <= 0.99
            "k": min(opt_top_k, 500),  # Ensure within Cohere's limits
            #"stream": True,
            #"return_likelihoods": "NONE",  # Can be "NONE", "GENERATION", or "ALL"
            #"truncate": "END",  # Can be "NONE", "START", or "END"
            #"stop_sequences": [],  # Optional list of sequences where generation should stop
        }

        #request.update({
        #    "return_likelihoods": return_likelihoods,
        #    "truncate": truncate,
        #    # Add other Cohere-specific parameters as needed
        #})
    elif "deepseek" in opt_model_id:
        # Deepseek models
        # Implement Deepseek-specific request format
        pass
    elif "writer.palmyra" in opt_model_id:
        # Palmyra models
        # Implement Palmyra-specific request format
        pass
    else:
        # Default format for other models
        request.update({
            "top_k": opt_top_k,
            "system": opt_system_msg,
            "messages": message_history
        })

    # Determine which client to use based on region
    if "anthropic.claude-3-5-sonnet-20241022-v2:0" == opt_model_id:
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

    with st.chat_message("assistant"):
        result_container = st.container(border=True)
        result_area = st.empty()

        # Display model parameters
        opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens}"
        result_container.write(opts)

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
                            result_container.write(stats)

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
                        result_container.write(stats)

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
                        result_container.write(stats)

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
                        result_container.write(stats)

                # Handle exceptions
                elif any(key in event for key in ["internalServerException", "modelStreamErrorException", 
                                                "modelTimeoutException", "throttlingException", "validationException"]):
                    for exception_type in ["internalServerException", "modelStreamErrorException", 
                                        "modelTimeoutException", "throttlingException", "validationException"]:
                        if exception := event.get(exception_type):
                            result_text += f"\n\n{exception}"
                            result_area.write(result_text)
                            break
        elif "cohere" in opt_model_id:
            # Process Cohere models
            for event in stream:
                if event["chunk"]:
                    chunk = json.loads(event["chunk"]["bytes"])

                    # Debug print to see the structure
                    print(f"Cohere chunk: {chunk}")

                    # Extract text from Cohere's response format
                    if "text" in chunk:  # Changed from "generations" to "text"
                        text = chunk["text"]
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
                        result_container.write(stats)
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
                        result_container.write(stats)

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
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            print("A client error occured: " + format(message))

if "audio_stream" in st.session_state and st.session_state["audio_stream"] != "":
    audio_bytes = BytesIO(st.session_state['audio_stream'])
    st.audio(audio_bytes, format='audio/mp3', autoplay=False)
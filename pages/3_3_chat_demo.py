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


def copy_button_clicked(text):
    pyperclip.copy(text)
    #st.session_state.button = not st.session_state.button

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
    "amazon.nova-pro-v1:0",
    "amazon.nova-premier-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0", #ok
    "deepseek.r1-v1:0",
    "meta.llama4-maverick-17b-instruct-v1:0",
    "meta.llama4-scout-17b-instruct-v1:0",
    "writer.palmyra-x4-v1:0",
    "writer.palmyra-x5-v1:0"
]

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")



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
        if "assistant" == msg["role"]:
            #assistant_cmd_panel_col1, assistant_cmd_panel_col2, assistant_cmd_panel_col3 = st.columns([0.07,0.23,0.7], gap="small")
            #with assistant_cmd_panel_col2:
            st.button(key=f"copy_button_{idx}", label='📄', type='primary', on_click=copy_button_clicked, args=[content])


if prompt := st.chat_input():
    
    st.session_state["audio_stream"] = ""

    message_history = st.session_state.messages.copy()
    message_history.append({"role": "user", "content": prompt})
    #st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    #user_message =  {"role": "user", "content": f"{prompt}"}
    #messages = [st.session_state.messages]
    #print(f"messages={st.session_state.messages}")

    request = {
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": opt_temperature,
        "top_p": opt_top_p,
        "top_k": opt_top_k,
        "max_tokens": opt_max_tokens,
        "system": opt_system_msg,
        "messages": message_history #st.session_state.messages
    }
    #json.dumps(request, indent=3)
    try:

        if "anthropic.claude-3-5-sonnet-20241022-v2:0" == opt_model_id:
            #bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            #st.chat_message("system").write(f"Invoke anthropic.claude-3-5-sonnet-20241022-v2:0 bedrock_runtime_us_west_2")
            response = bedrock_runtime_us_west_2.invoke_model_with_response_stream(
                modelId = opt_model_id, #bedrock_model_id, 
                contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
                accept = "application/json",
                body = json.dumps(request),
                #trace="ENABLED",
                #guardrailIdentifier=AWS_BEDROCK_GUARDRAIL_IDENTIFIER,
                #guardrailVersion=AWS_BEDROCK_GUARDRAIL_VERSION
                ) 
            # The provided request is not valid
            
        else:
            #bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            response = bedrock_runtime.invoke_model_with_response_stream(
                modelId = opt_model_id, #bedrock_model_id, 
                contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
                accept = "application/json",
                body = json.dumps(request),
                #trace="ENABLED",
                guardrailIdentifier=AWS_BEDROCK_GUARDRAIL_IDENTIFIER,
                guardrailVersion=AWS_BEDROCK_GUARDRAIL_VERSION)

        #with st.chat_message("assistant", avatar=setAvatar("assistant")):
        result_text = ""
        with st.chat_message("assistant"):
            result_container = st.container(border=True)
            result_area = st.empty()
            stream = response["body"]
            for event in stream:
                
                if event["chunk"]:

                    chunk = json.loads(event["chunk"]["bytes"])

                    if chunk['type'] == 'message_start':
                        opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens}"
                        #result_text += f"{opts}\n\n"
                        #result_area.write(result_text)
                        result_container.write(opts)
                        #pass

                    elif chunk['type'] == 'message_delta':
                        #print(f"\nStop reason: {chunk['delta']['stop_reason']}")
                        #print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
                        #print(f"Output tokens: {chunk['usage']['output_tokens']}")
                        pass

                    elif chunk['type'] == 'content_block_delta':
                        if chunk['delta']['type'] == 'text_delta':
                            text = chunk['delta']['text']
                            #await msg.stream_token(f"{text}")
                            result_text += f"{text}"
                            result_area.write(result_text)

                    elif chunk['type'] == 'message_stop':
                        invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                        input_token_count = invocation_metrics["inputTokenCount"]
                        output_token_count = invocation_metrics["outputTokenCount"]
                        latency = invocation_metrics["invocationLatency"]
                        lag = invocation_metrics["firstByteLatency"]
                        stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                        #await msg.stream_token(f"\n\n{stats}")
                        #result_text += f"\n\n{stats}"
                        #result_area.write(result_text)
                        result_container.write(stats)

                elif event["internalServerException"]:
                    exception = event["internalServerException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                elif event["modelStreamErrorException"]:
                    exception = event["modelStreamErrorException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                elif event["modelTimeoutException"]:
                    exception = event["modelTimeoutException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                elif event["throttlingException"]:
                    exception = event["throttlingException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                elif event["validationException"]:
                    exception = event["validationException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                else:
                    result_text += f"\n\nUnknown Token"
                    result_area.write(result_text)

            col1, col2, col3 = st.columns([1,1,5])

            with col1:
                st.button(key='copy_button', label='📄', type='primary', on_click=copy_button_clicked, args=[result_text])
            with col2:
                if "audio_stream" not in st.session_state or st.session_state["audio_stream"] == "":
                    st.button(key='recite_button', label='▶️', type='primary', on_click=recite_button_clicked, args=[result_text])
            with col3:
                st.markdown('3')
            
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
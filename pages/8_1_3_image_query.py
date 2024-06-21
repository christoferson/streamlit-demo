import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
import pyperclip
import io
import base64
from PIL import Image

from botocore.exceptions import ClientError

AWS_REGION = cmn_settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

st.set_page_config(
    page_title="Image Query",
    page_icon="🧊",
    layout="wide", # "centered" or "wide"
    initial_sidebar_state="collapsed", #"auto", "expanded", or "collapsed"
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

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

def copy_button_clicked(text):
    pyperclip.copy(text)
    #st.session_state.button = not st.session_state.button

def image_to_base64(image,mime_type:str):
    buffer = io.BytesIO()
    image.save(buffer, format=mime_type)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

mime_mapping = {
        "image/png": "PNG",
        "image/jpeg": "JPEG"
    }

opt_model_id_list = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

st.title("💬 Image Query 1")
st.write("Ask LLM Questions")


col1, col2 = st.columns([2, 1])

with col2:

    uploaded_file  = st.file_uploader(
        "",
        type=["PNG", "JPEG"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    uploaded_file_name = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        uploaded_file_name = uploaded_file.name
        uploaded_file_type = uploaded_file.type
        uploaded_file_base64 = image_to_base64(image, mime_mapping[uploaded_file_type])
        st.image(
            image, caption='upload images',
            use_column_width=True
        )

######


with col1:

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    idx = 1
    for msg in st.session_state.messages:
        idx = idx + 1
        content = msg["content"]
        with st.chat_message(msg["role"]):
            st.write(content)
            if "assistant" == msg["role"]:
                #assistant_cmd_panel_col1, assistant_cmd_panel_col2, assistant_cmd_panel_col3 = st.columns([0.07,0.23,0.7], gap="small")
                #with assistant_cmd_panel_col2:
                #st.button(key=f"copy_button_{idx}", label='📄', type='primary', on_click=copy_button_clicked, args=[content])
                pass


    if prompt := st.chat_input():

        message_history = st.session_state.messages.copy()
        content =  [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        }
                    ]

        if uploaded_file_name:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": uploaded_file_type,
                        "data": uploaded_file_base64,
                    },
                }
            )
        message_history.append({"role": "user", "content": content})
        #st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        #user_message =  {"role": "user", "content": f"{prompt}"}
        #messages = [st.session_state.messages]
        print(f"messages={st.session_state.messages}")

        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": opt_temperature,
            "top_p": opt_top_p,
            "top_k": opt_top_k,
            "max_tokens": opt_max_tokens,
            "system": opt_system_msg,
            "messages": message_history #st.session_state.messages
        }
        json.dumps(request, indent=3)

        try:
            #bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            response = bedrock_runtime.invoke_model_with_response_stream(
                modelId = opt_model_id, #bedrock_model_id, 
                contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
                accept = "application/json",
                body = json.dumps(request))

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
                            #result_container.write(opts)
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
                            #result_container.write(stats)

                            invocation_metrics = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                            #result_container.markdown(f""":blue[{invocation_metrics}]""")
                            #result_area.markdown(f"{invocation_metrics} {result_text} ")
                            result_text_final = f"""{result_text}  \n\n:blue[{invocation_metrics}]"""
                            #result_text += f"{reference_chunk_list_text}"
                            #result_area.write(f"{result_text_final}")
                            #result_container.markdown()
                            result_area.write(f"{result_text_final}")

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

                #st.button(key='copy_button', label='📄', type='primary', on_click=copy_button_clicked, args=[result_text])
                

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": result_text})

            
            
        except ClientError as err:
            message = err.response["Error"]["Message"]
            logger.error("A client error occurred: %s", message)
            print("A client error occured: " + format(message))
            st.chat_message("system").write(message)

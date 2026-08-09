import streamlit as st
import boto3
import cmn_settings
import json
import logging
import cmn_auth
import pyperclip

from botocore.exceptions import ClientError

AWS_REGION = cmn_settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

st.set_page_config(
    page_title="Translate",
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

opt_model_id_list = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

system_message = """You are a highly skilled translator with expertise in many languages. 
Your task is to identify the language of the text I provide and accurately translate it into the specified target language while preserving the meaning, tone, and nuance of the original text. 
Please maintain proper grammar, spelling, and punctuation in the translated version.
With regards to the target language, if it is not explicitly specified, assume target is English. If the source is already in English, then assume target is Japanese. 
With regards to the output format, omit preambles and just provide the translated text.
"""



with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    #opt_system_msg = st.text_area(label="System Message", value=system_message, key="system_msg")

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

if "translate_input" not in st.session_state:
    st.session_state["translate_input"] = """It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair."""

if "translate_result" not in st.session_state:
    st.session_state["translate_result"] = "Default text"

def write_translate_result_if_present():
    text = st.session_state["translate_result"]
    result_container = col2_container.container()
    result_area = result_container.empty()
    result_area.write(text)
    col1, col2, col3 = result_container.columns([1,1,15])
    with col1:
        st.button(key='copy_button', label='📄', type='primary', on_click=on_button_copy_clicked)

def on_text_area_translate_input_changed():
    #st.session_state["translate_input"] = "Default text"
    #print(st.session_state["translate_input"])
    write_translate_result_if_present()



st.title("💬 Translate v3")
st.markdown("Enter text to translate")

col1, col2 = st.columns(2)

col2_container = col2.container()
col2_container.caption(":blue[Result]")


col1.text_area(
        ":blue[Input]",
        #value=st.session_state["translate_result"],
        on_change=on_text_area_translate_input_changed,
        key='translate_input',
        height = 500,
    )

def on_button_copy_clicked():
    text = st.session_state["translate_result"]
    pyperclip.copy(text)
    write_translate_result_if_present()

def on_button_clear_clicked():
    st.session_state["translate_input"] = ""
    result_area = col2_container.empty()

def on_button_clicked():
    #st.session_state["translate_input"] = "Default text"
    #print(st.session_state["translate_input"])
    if st.session_state["translate_input"] == "":
        result_area = col2_container.empty()
        result_area.markdown(":red[Enter Source Text to Translate]")
        return

    #result_area = col2_container.empty()
    result_container = col2_container.container()
    result_area = result_container.empty()
    
    #with col2:
    result_area.write("...")


    prompt = st.session_state["translate_input"]

    message_history = []
    message_history.append({"role": "user", "content": prompt})
    request = {
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": opt_temperature,
            "top_p": opt_top_p,
            "top_k": opt_top_k,
            "max_tokens": opt_max_tokens,
            "system": system_message,
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

        result_text = ""
        stream = response["body"]
        for event in stream:
            
            if event["chunk"]:

                chunk = json.loads(event["chunk"]["bytes"])

                if chunk['type'] == 'message_start':
                    opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens}"
                    pass

                elif chunk['type'] == 'message_delta':
                    pass

                elif chunk['type'] == 'content_block_delta':
                    if chunk['delta']['type'] == 'text_delta':
                        text = chunk['delta']['text']
                        #await msg.stream_token(f"{text}")
                        result_text += f"{text}"
                        #text_area_result.write(result_text)
                        result_area.write(result_text)

                elif chunk['type'] == 'message_stop':
                    invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                    input_token_count = invocation_metrics["inputTokenCount"]
                    output_token_count = invocation_metrics["outputTokenCount"]
                    latency = invocation_metrics["invocationLatency"]
                    lag = invocation_metrics["firstByteLatency"]
                    stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                    #result_container.write(stats)
                    st.session_state["translate_result"] = result_text
                    result_area.write(result_text)
                    col1, col2, col3 = result_container.columns([1,1,15])
                    with col1:
                        st.button(key='copy_button', label='📄', type='primary', on_click=on_button_copy_clicked)

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
        #col2_container.write(system_message)
        
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)


button_panel = col1.columns([1,1,1,1,1,1,1,1,1], gap="small") #gap ("small", "medium", or "large")
button_panel[8].button("Translate", on_click=on_button_clicked)
button_panel[7].button("⎚ Clear", on_click=on_button_clear_clicked)



        
    
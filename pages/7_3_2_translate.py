import streamlit as st
import boto3
import cmn_settings
import cmn_st_audio
import json
import logging
import cmn_constants
from io import BytesIO
import textwrap as tw
#from streamlit_extras.stylable_container import stylable_container
#import streamlit.components.v1 as components
from streamlit.components.v1 import html

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

st.logo(icon_image="images/logo.png", image="images/logo_text.png")

#st.markdown(cmn_constants.css_sidebar,unsafe_allow_html=True)

#st.markdown(cmn_constants.css_sidebar_2,unsafe_allow_html=True)

opt_model_id_list = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

system_message = """You are a highly skilled translator with expertise in many languages. 
Your task is to identify the language of the text I provide and accurately translate it into the specified target language while preserving the meaning, tone, and nuance of the original text. 
Please maintain proper grammar, spelling, and punctuation in the translated version.
In case the the target language is not explicitly specified, If the source is English, then assume the target is Japanese, otherwise assume target is English.
With regards to the output format, omit preambles and just provide the translated text.
"""

error_translate_input_empty = ":red[Enter Source Text to Translate]"

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

if "translate_input" not in st.session_state:
    if "menu_translate_translate_input" in st.session_state:
        st.session_state["translate_input"] = st.session_state["menu_translate_translate_input"]
    else:
        st.session_state["translate_input"] = """It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair."""

if "translate_result" not in st.session_state:
    st.session_state["translate_result"] = None

st.title("💬 Translate v 7.3.1")

col1, col2 = st.columns(2)

col2_container = col2.container()
col2_container.caption(":blue[Result]")
result_container = col2_container.container()
#if "translate_result" in st.session_state and st.session_state["translate_result"] != None:
#    result_display_columns = result_container.slider("columns", value=45, min_value=50, max_value=80, step=1, label_visibility="collapsed")
#    result_text = st.session_state["translate_result"]
#    result_text_wrapped = "\n".join(
#        tw.wrap(result_text, width=result_display_columns, drop_whitespace=True, replace_whitespace=False)
#    )
#    #result_area.markdown(result_text)
#    result_container.code(result_text_wrapped, language="markdown")
#    markdown_display = result_container.checkbox("Markdown", value=True)
#    if markdown_display:
#        result_container.markdown(result_text)
#    else:
#        result_container.code(result_text, language="wiki")

#if "translate_result_error" in st.session_state and st.session_state["translate_result_error"] != None:
#    result_container.markdown(st.session_state["translate_result_error"])

if "translate_result" in st.session_state and st.session_state["translate_result"] != None:
    translate_result_text = st.session_state["translate_result"]
    if translate_result_text != "" and translate_result_text != error_translate_input_empty:
        if 'menu_translate_stream_displayed' in st.session_state:
            del st.session_state['menu_translate_stream_displayed']
        else:
            result_container.markdown(translate_result_text)
            result_container.download_button(key="save_button", label='📩 Save', type='primary', file_name="result.txt", data=st.session_state["translate_result"], mime='text/csv', use_container_width=False)
        #pass


#result_columns = result_container.columns([1,1,1,1,1,1,1,1,1,1,1,1,1], gap="small")
#if "translate_result" in st.session_state and st.session_state["translate_result"] != None:
#    translate_result_text = st.session_state["translate_result"]
#    if translate_result_text != "" and translate_result_text != error_translate_input_empty:
#        result_columns[0].download_button(key="save_button", label='📩 Save', type='primary', file_name="result.txt", data=st.session_state["translate_result"], mime='text/csv', use_container_width=True)


#result_columns = result_container.columns([1,1,1,1,1,1,1,1,1,1,1,1,1], gap="small")
#if "translate_result" in st.session_state and st.session_state["translate_result"] != None:
#    result_columns[0].button(key='copy_button', label='📄 Copy', type='primary', use_container_width=True)
#    result_columns[1].download_button(key="save_button", label='📩 Save', type='primary', file_name="result.txt", data=st.session_state["translate_result"], mime='text/csv', use_container_width=True)
#    result_columns[2].button(key="recite_button", label='Play', type='primary', args=[st.session_state["translate_result"]], help="Text to Speech")
#if "audio_stream" in st.session_state and st.session_state["audio_stream"] != None:
#    audio_bytes = BytesIO(st.session_state['audio_stream'])
#    if "result_text_language" in st.session_state and st.session_state['result_text_language'] != None:
#        result_container.markdown(st.session_state['result_text_language'])
#    result_container.audio(audio_bytes, format='audio/mp3', autoplay=False)

def on_text_area_translate_input_changed():
    st.session_state["menu_translate_translate_input"] = st.session_state["translate_input"]


col1.text_area(
        ":blue[Input]",
        #value=st.session_state["translate_input"],
        on_change=on_text_area_translate_input_changed,
        key='translate_input',
        height = 500,
        help="Specify the Target Language or Custom Instructions at the end. e.g. <input_text> --> Formal English"
    )

def stream_result_data(stream):

    result_text = ""

    try: 

        for event in stream:
            
            if "chunk" in event:

                chunk = json.loads(event["chunk"]["bytes"])

                if chunk['type'] == 'message_start':
                    #opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens}"
                    pass

                elif chunk['type'] == 'message_delta':
                    pass

                elif chunk['type'] == 'content_block_delta':
                    if chunk['delta']['type'] == 'text_delta':
                        text = chunk['delta']['text']
                        result_text += f"{text}"
                        yield text

                elif chunk['type'] == 'message_stop':
                    invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                    input_token_count = invocation_metrics["inputTokenCount"]
                    output_token_count = invocation_metrics["outputTokenCount"]
                    latency = invocation_metrics["invocationLatency"]
                    lag = invocation_metrics["firstByteLatency"]
                    #stats = f"| token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                    #result_area.markdown(result_text)
                    #st.session_state["translate_result"] = result_text
                    #yield result_text
                    yield output_token_count


            elif "internalServerException" in event:
                exception = event["internalServerException"]
                yield exception
            elif "modelStreamErrorException" in event:
                exception = event["modelStreamErrorException"]
                yield exception
            elif "modelTimeoutException" in event:
                exception = event["modelTimeoutException"]
                yield exception
            elif "throttlingException" in event:
                exception = event["throttlingException"]
                yield exception
            elif "validationException" in event:
                exception = event["validationException"]
                yield exception
            else:
                yield "Unknown Token"

        st.session_state["translate_result"] = result_text
    
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        yield message

def on_button_clear_clicked():
    st.session_state["translate_input"] = ""
    st.session_state["translate_result"] = None
    #st.session_state['result_text_language'] = None
    #st.session_state['audio_stream'] = None

def on_button_translate_clicked():
    if "translate_input" not in  st.session_state or st.session_state["translate_input"] == "":
        st.session_state["translate_result_error"] = error_translate_input_empty
        result_container.markdown(error_translate_input_empty)
        return

    if "translate_result_error" in st.session_state:
        del st.session_state["translate_result_error"]
    
    if "translate_result" in st.session_state:
        del st.session_state["translate_result"]

    #st.session_state['result_text_language'] = None
    #st.session_state['audio_stream'] = None

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
            "messages": message_history
        }
    
    try:

        response = bedrock_runtime.invoke_model_with_response_stream(
            modelId = opt_model_id, #bedrock_model_id, 
            contentType = "application/json", #guardrailIdentifier  guardrailVersion=DRAFT, trace=ENABLED | DISABLED
            accept = "application/json",
            body = json.dumps(request))
        
        stream = response["body"]
        result_container.write_stream(stream_result_data(stream))

        if "translate_result" in st.session_state and st.session_state["translate_result"] != None:
            translate_result_text = st.session_state["translate_result"]
            if translate_result_text != "" and translate_result_text != error_translate_input_empty:
                result_container.download_button(key="save_button", label='📩 Save', type='primary', file_name="result.txt", data=st.session_state["translate_result"], mime='text/csv', use_container_width=False)
        
        st.session_state["menu_translate_stream_displayed"] = True
        
        #result_columns = result_container.columns([1,1,1,1,1,1,1,1,1,1,1,1,1], gap="small")
        #if "translate_result" in st.session_state and st.session_state["translate_result"] != None:
        #    translate_result_text = st.session_state["translate_result"]
        #    if translate_result_text != "" and translate_result_text != error_translate_input_empty:
        #        result_columns[0].download_button(key="save_button", label='📩 Save', type='primary', file_name="result.txt", data=st.session_state["translate_result"], mime='text/csv', use_container_width=True)


        
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)


button_panel = col1.columns([1,1,1,1,1,1,1,1,1], gap="small") #gap ("small", "medium", or "large")
button_panel[8].button("Translate", on_click=on_button_translate_clicked, use_container_width=True)
button_panel[7].button("⎚ Clear", on_click=on_button_clear_clicked, use_container_width=True)

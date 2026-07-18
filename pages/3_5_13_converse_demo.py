import streamlit as st
import boto3
import cmn_settings
import cmn_constants
import cmn_security
import json
import logging
from io import BytesIO

from cmn.bedrock_models import FoundationModel
from datetime import datetime

from botocore.exceptions import ClientError, ReadTimeoutError

AWS_REGION = cmn_settings.AWS_REGION
MAX_MESSAGES = 80 * 2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse_stream.html

# include up to 20 images. Each image's size, height, and width must be no more than 3.75 MB, 8,000 px, and 8,000 px, respectively.
#  include up to five documents. Each document's size must be no more than 5 MB.
# can only include images and documents if the role is user.

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

####################################################################################

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_runtime_us_west_2 = boto3.client('bedrock-runtime', region_name="us-west-2")
cloudwatch_logs = boto3.client('logs', region_name=AWS_REGION)
polly = boto3.client("polly", region_name=AWS_REGION)

####################################################################################

st.set_page_config(
    page_title="Converse",
    page_icon="🧊",
    layout="wide", # "centered" or "wide" — wide: page hosts chat + info panel side by side
    initial_sidebar_state="expanded", #"auto", "expanded", or "collapsed"
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    },
)


st.logo(icon_image="images/logo.png", image="images/logo_text.png")

st.markdown(cmn_constants.css_button_primary, unsafe_allow_html=True)

# Create main layout with right sidebar
main_col, right_sidebar_col = st.columns([3, 1])

def trim_conversation_history(messages):
    """Ensure conversation doesn't exceed maximum length"""
    messages_len = len(messages)
    if messages_len > MAX_MESSAGES:
        # Remove oldest messages while keeping pairs together
        return messages[-(MAX_MESSAGES * 2):]
    return messages

def validate_conversation_structure(conversation):
    """Validate the structure of uploaded conversation data"""
    if not isinstance(conversation, list):
        raise ValueError("Invalid format: expected a list of messages")

    if len(conversation) > MAX_MESSAGES:
        raise ValueError(f"Conversation exceeds maximum length of {MAX_MESSAGES} messages")

    for message in conversation:
        if not isinstance(message, dict):
            raise ValueError("Invalid message format: expected a dictionary")

        if "role" not in message or "content" not in message:
            raise ValueError("Message missing required fields: 'role' and 'content'")

        if message["role"] not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {message['role']}")

        if not isinstance(message["content"], list):
            raise ValueError("Invalid content format: expected a list")

        for content_item in message["content"]:
            if not isinstance(content_item, dict) or "text" not in content_item:
                raise ValueError("Invalid content item: expected 'text' key")

def strip_binary_content(messages):
    """
    Replace image/document byte payloads with text placeholders so the
    conversation is JSON-serializable for download. Attachment bytes are
    session-only and not restored on upload.
    """
    stripped = []
    for msg in messages:
        content = []
        for item in msg["content"]:
            if "image" in item:
                content.append({"text": "[image attachment omitted]"})
            elif "document" in item:
                doc_name = item["document"].get("name", "unknown")
                content.append({"text": f"[document attachment omitted: {doc_name}]"})
            else:
                content.append(item)
        stripped.append({"role": msg["role"], "content": content})
    return stripped


def get_conversation_download():
    """Download conversation with security signature"""
    if not st.session_state.menu_converse_messages:
        raise ValueError("No conversation to download")

    security = cmn_security.ConversationSecurity()
    secure_package = security.secure_conversation(
        strip_binary_content(
            trim_conversation_history(st.session_state.menu_converse_messages)
        )
    )
    return BytesIO(json.dumps(secure_package, indent=2).encode())

def clear_conversation():
    """Clear conversation state"""
    st.session_state.menu_converse_messages = []

def upload_conversation(uploaded_file):
    """Upload and verify conversation integrity"""
    try:
        security = cmn_security.ConversationSecurity()

        # Read and parse the uploaded file
        content = uploaded_file.getvalue().decode("utf-8")
        conversation_package = json.loads(content)

        # Verify integrity
        if not security.verify_conversation(conversation_package):
            st.error("Security verification failed. The conversation may have been tampered with.")
            return False

        # Validate structure
        validate_conversation_structure(conversation_package["messages"])

        # Update session state
        st.session_state.menu_converse_messages = conversation_package["messages"]

        st.success("Conversation uploaded and verified successfully!")
        return True

    except json.JSONDecodeError:
        st.error("Invalid JSON file format")
    except ValueError as e:
        st.error(f"Error in conversation structure: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Upload error: {str(e)}", exc_info=True)
    return False

# Get the top 10 users with the most tokens in a given period (e.g., past 3 months):
# fields @timestamp, @message
# | parse @message '{"user_name":"*","input_tokens":*,"output_tokens":*,"total_tokens":*,"event_time":"*"}' as user_name, input_tokens, output_tokens, total_tokens, event_time
# | filter ispresent(total_tokens) and event_time > '2023-03-01T00:00:00' and event_time < '2023-06-01T00:00:00'
# | stats sum(total_tokens) as total_tokens_used by user_name
# | sort total_tokens_used desc
# | limit 10
#Get the total tokens used by a particular user for the whole period:
# fields @timestamp, @message
# | parse @message '{"user_name":"*","input_tokens":*,"output_tokens":*,"total_tokens":*,"event_time":"*"}' as user_name, input_tokens, output_tokens, total_tokens, event_time
# | filter user_name = 'specific_user_name' and ispresent(total_tokens)
# | stats sum(total_tokens) as total_tokens_used by user_name
def push_to_cloudwatch(user_name, input_tokens, output_tokens, total_tokens):
    log_group_name = "/app/openai/metrics/invocations"
    log_stream_name = datetime.now().strftime("%Y-%m-%d")

    # Ensure the log group exists
    try:
        cloudwatch_logs.create_log_group(logGroupName=log_group_name)
    except cloudwatch_logs.exceptions.ResourceAlreadyExistsException as e:
        #logger.error(f"Failed to create log group to CloudWatch: {str(e)}")
        pass

    # Ensure the log stream exists
    try:
        cloudwatch_logs.create_log_stream(logGroupName=log_group_name, logStreamName=log_stream_name)
    except cloudwatch_logs.exceptions.ResourceAlreadyExistsException as e:
        #logger.error(f"Failed to create log stream to CloudWatch: {str(e)}")
        pass

    # Create the log entry as a JSON object
    log_entry = {
        'timestamp': int(datetime.now().timestamp() * 1000),
        'message': json.dumps({
            'user_name': user_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'event_time': datetime.now().isoformat()
        })
    }

    # Push the log entry to CloudWatch
    try:
        cloudwatch_logs.put_log_events(
            logGroupName=log_group_name,
            logStreamName=log_stream_name,
            logEvents=[log_entry]
        )
    except Exception as e:
        logger.error(f"Failed to push logs to CloudWatch: {str(e)}")


def delete_message_pair(index):
    if index < len(st.session_state.menu_converse_messages):
        del st.session_state.menu_converse_messages[index:index+2]

def get_download_data():
    """Get conversation download data, returns None if no data"""
    if not st.session_state.menu_converse_messages:
        return None
    try:
        return get_conversation_download()
    except Exception as e:
        st.error(f"Error preparing conversation: {str(e)}")
        return None

def on_button_clear_clicked():
    """Handle clear conversation button click"""
    clear_conversation()
    st.rerun()

mime_mapping = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

mime_mapping_image = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

#'pdf'|'csv'|'doc'|'docx'|'xls'|'xlsx'|'html'|'txt'|'md',
mime_mapping_document = {
    "text/plain": "txt",
    "application/vnd.ms-excel": "csv",
    "text/csv": "csv",
    "application/pdf": "pdf",
    "text/markdown": "md",
}


def build_attachment_blocks(files):
    """
    Convert chat_input attachments into Converse content blocks.
    Returns (blocks, unsupported_names). Attachments stay in the message
    they arrived with, so the full history replay keeps them visible to
    the model on every subsequent turn.
    """
    blocks = []
    unsupported = []
    for f in files:
        if f.type in mime_mapping_image:
            blocks.append({
                "image": {
                    "format": mime_mapping_image[f.type],
                    "source": {"bytes": f.getvalue()},
                }
            })
        elif f.type in mime_mapping_document:
            blocks.append({
                "document": {
                    "format": mime_mapping_document[f.type],
                    "name": f.name.replace(".", "_").replace(" ", "_"),
                    "source": {"bytes": f.getvalue()},
                }
            })
        else:
            unsupported.append(f"{f.name} ({f.type})")
    return blocks, unsupported
       
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
opt_model_id_list = [
    "global.anthropic.claude-fable-5",
    "us.anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-5",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "global.anthropic.claude-opus-4-6-v1",
    "global.anthropic.claude-opus-4-7",
    "global.anthropic.claude-opus-4-8",

    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    #"anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-opus-4-20250514-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "amazon.nova-pro-v1:0",
    "global.amazon.nova-2-lite-v1:0",
    #"global.amazon.nova-2-pro-v1:0",
    
    
    "cohere.command-r-v1:0", # The model returned the following errors: Malformed input request: #: extraneous key [top_k] is not permitted, please reformat your input and try again.
    "cohere.command-r-plus-v1:0",
    "meta.llama2-13b-chat-v1", # Llama 2 Chat 13B
    "meta.llama2-70b-chat-v1", # Llama 2 Chat 70B
    "meta.llama3-8b-instruct-v1:0", # Llama 3 8b Instruct
    "meta.llama3-70b-instruct-v1:0",  # Llama 3 70b Instruct
    "us.meta.llama3-2-11b-instruct-v1:0", # Vision
    "us.meta.llama3-2-90b-instruct-v1:0", # Vision
    #"mistral.mistral-7b-instruct-v0:2", # Mistral 7B Instruct Does not support system message
    #"mistral.mixtral-8x7b-instruct-v0:1", # Mixtral 8X7B Instruct Does not support system message
    "mistral.mistral-small-2402-v1:0", # Mistral Small
    "mistral.mistral-large-2402-v1:0", # Mistral Large
    "us.mistral.pixtral-large-2502-v1:0",
    "us.amazon.nova-premier-v1:0",
    "us.meta.llama4-scout-17b-instruct-v1:0",
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    "us.writer.palmyra-x4-v1:0",
    "us.writer.palmyra-x5-v1:0",
    "qwen.qwen3-next-80b-a3b",
    "qwen.qwen3-vl-235b-a22b",
    "openai.gpt-oss-safeguard-20b", #KeyError: 'text'
    "openai.gpt-oss-safeguard-120b", #KeyError: 'text'
    "google.gemma-3-4b-it",
    "google.gemma-3-12b-it",
    "google.gemma-3-27b-it",
    "nvidia.nemotron-nano-9b-v2",
    "nvidia.nemotron-nano-12b-v2",
    "us.amazon.nova-2-lite-v1:0",
]

opt_model_id_list_default = "global.anthropic.claude-sonnet-5"
opt_model_id_list_default_idx = opt_model_id_list.index(opt_model_id_list_default)


# "You are a question and answering chatbot"
# - Maintaining a helpful and courteous tone throughout the interaction.
opt_system_msg_int = """You are a smart and helpful Assistant. Your tasks include:
- Providing detailed, step-by-step, and professional responses to user queries.
- Ensuring accuracy and depth in your explanations.
- Adapting your language and complexity to suit the user's level of understanding.
- Offering relevant examples or analogies to clarify complex concepts.
- Citing reliable sources when appropriate to support your information.
Please think through each step carefully before responding to ensure clarity, completeness, and coherence in your answer. 
If any part of the query is unclear, don't hesitate to ask for clarification to provide the most accurate and helpful response possible.
"""

if "menu_converse_messages" not in st.session_state:
    st.session_state["menu_converse_messages"] = []

with st.sidebar:
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = opt_model_id_list_default_idx, key="model_id")

opt_fm:FoundationModel = FoundationModel.find(opt_model_id)

opt_fm_max_tokens = opt_fm.InferenceParameter.get("MaxTokensToSample")
opt_fm_temperature = opt_fm.InferenceParameter.get("Temperature")
opt_fm_top_p = opt_fm.InferenceParameter.get("TopP")
opt_fm_top_k = opt_fm.InferenceParameter.get("TopK")

with st.sidebar:
    if opt_fm_temperature.isSupported():
        opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    else:
        opt_temperature = 0.0
    if opt_fm_top_p.isSupported():
        opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    else:
        opt_top_p = 0.0
    if opt_fm_top_k.isSupported():
        opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    else:
        opt_top_k = 0
    opt_max_tokens = st.slider(label="Max Tokens", min_value=opt_fm_max_tokens.MinValue, max_value=opt_fm_max_tokens.MaxValue, value=opt_fm_max_tokens.DefaultValue, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value=opt_system_msg_int, key="system_msg")

with st.sidebar:
    st.divider()
    st.markdown(":blue[**Conversation**]")
    #st.markdown(f"{len(st.session_state.menu_converse_messages)}/{MAX_MESSAGES}")
    uploaded_conversation = st.file_uploader(
        ":green[**Upload Conversation**]",
        type=["json"],
        help="Upload a previously downloaded conversation file"
    )

    if uploaded_conversation is not None:
        if st.button("Load Conversation", type="secondary", icon=":material/upload:", use_container_width=False):
            if upload_conversation(uploaded_conversation):
                st.rerun()

#st.markdown("#### 💬 :blue[Converse 3-5-12]")

with main_col:
    header_flex = st.container(horizontal=True, width="stretch", horizontal_alignment="left", vertical_alignment="bottom", border=False)
    header_flex.markdown("#### 💬 :blue[Converse 3-5-12]")
    header_flex.space("stretch")
    show_guide = header_flex.toggle(":violet[**Guide**]", value=False, key="show_guide_toggle")

    if show_guide:
        with st.container(border=True):

            # Guide header with language toggle
            guide_header = st.container(horizontal=True, width="stretch", horizontal_alignment="left", vertical_alignment="center", border=False)
            guide_header.markdown("##### 📖 :blue[User Guide]")
            guide_header.space("stretch")
            lang_english = guide_header.toggle(":gray[**EN | 日本語**]", value=True, key="guide_lang_toggle")

            if lang_english:
                st.markdown("""
                **Getting Started**

                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.

                **Uploading Files**

                Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
                Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

                **Managing Conversations**

                Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium,
                totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.

                **Model Settings**

                Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos
                qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet.

                **Tips & Tricks**

                - 💡 Lorem ipsum dolor sit amet, consectetur adipiscing elit
                - 💡 Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua
                - 💡 Ut enim ad minim veniam, quis nostrud exercitation ullamco
                - 💡 Duis aute irure dolor in reprehenderit in voluptate velit
                - 💡 Excepteur sint occaecat cupidatat non proident deserunt
                """)
            else:
                st.markdown("""
                **はじめに**

                AIアシスタントとの会話を開始できます。適切なモデルを選択し、
                パラメータを調整することで最適な結果を得ることができます。
                様々な主要な大規模言語モデルに対応しており、幅広いシナリオに対応しています。

                **ファイルのアップロード**

                画像（PNG、JPG、JPEG）およびドキュメント（TXT、CSV、PDF、MD）のアップロードに対応しています。
                ファイルサイズは5MB以内、画像サイズは8000×8000ピクセル以内に制限されています。

                **会話の管理**

                会話履歴はJSONファイルとしてダウンロード保存でき、後から再アップロードして復元できます。
                システムは最大160件の直近メッセージを保持します。

                **モデル設定**

                サイドバーで温度、Top P、Top K、最大トークン数などのパラメータを調整できます。
                モデルによってサポートされるパラメータの範囲が異なりますので、用途に応じて調整してください。

                **使い方のヒント**

                - 💡 タスクに適したモデルをサイドバーから選択してください
                - 💡 温度を下げることでより安定した出力結果が得られます
                - 💡 ドキュメントをアップロードしてその内容について直接質問できます
                - 💡 後で参照できるよう定期的に会話履歴をダウンロードしてください
                - 💡 クリアボタンを使用して新しい会話を開始できます
                """)

    idx = 1
    for msg in st.session_state.menu_converse_messages:
        idx = idx + 1
        contents = msg["content"]
        with st.chat_message(msg["role"]):
            content = contents[0]
            content_text = content["text"]
            if "user" == msg["role"]:
                st.write(f"{content_text}")
                # Render any attachments carried in this message
                for extra in contents[1:]:
                    if "document" in extra:
                        st.caption(f":green[Document: {extra['document']['name']}]")
                    elif "image" in extra:
                        img_bytes = extra["image"]["source"].get("bytes")
                        if img_bytes:
                            st.image(img_bytes, width=200)

                del_idx = idx - 2
                if st.button(f"Delete ({del_idx})", key=f"delete_button_{del_idx}"):
                    delete_message_pair(del_idx)
                    st.rerun()

            if "assistant" == msg["role"]:
                st.markdown(f"{content_text}")


    # File attachments now come through st.chat_input (accept_file="multiple").
    # Build the accepted-type list from the model's capabilities.
    chat_input_file_types = []
    if opt_fm.isFeatureSupported("document_chat"):
        chat_input_file_types.extend(["txt", "csv", "pdf", "md"])
    if opt_fm.isFeatureSupported("vision"):
        chat_input_file_types.extend(["png", "jpg", "jpeg", "gif", "webp"])

# Chat input at the bottom of viewport
with st.bottom:
    with st.container(horizontal=True, width="stretch", horizontal_alignment="right", vertical_alignment="center", border=False, height="content", gap="xxsmall", autoscroll=False):
        st.markdown(
            f":violet[**{len(st.session_state.menu_converse_messages)}/{MAX_MESSAGES}**]",
            help="Messages in conversation history",
        )
        if st.button(":material/delete_history:", type="tertiary", help="Clear Conversation"):
            on_button_clear_clicked()

        download_data = get_download_data()
        if download_data:
            st.download_button(
                label=":material/archive:",
                data=download_data,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                type="tertiary",
                help="Download Conversation",
                key="download_conversation_button"
            )
        else:
            st.button(":material/archive:", type="tertiary", help="Download Conversation (no data)", disabled=True)

        action = st.menu_button("Export", options=["CSV", "JSON", "PDF"], type="tertiary", help="Export Conversation in different formats")
        # options = ["North", "East", "South", "West"]
        # selection = st.segmented_control(
        #     "", options, selection_mode="single",
        # )
        st.space(size="small")

    chat_input_prompt = st.chat_input(
        accept_audio=True,
        accept_file="multiple" if chat_input_file_types else False,
        file_type=chat_input_file_types or None,
        placeholder="Type your message here — attach files with 📎...",
    )

with main_col:
    if chat_input_prompt and chat_input_prompt.text:
        prompt = chat_input_prompt.text

        # menu_converse_messages = st.session_state.menu_converse_messages
        # menu_converse_messages_len = len(menu_converse_messages)
        # if menu_converse_messages_len > MAX_MESSAGES:
        #     del menu_converse_messages[0 : (menu_converse_messages_len - MAX_MESSAGES) * 2]
        #st.write(f"""{mime_mapping_image[uploaded_file_type]}""")
        #st.session_state["audio_stream"] = ""

        message_history = st.session_state.menu_converse_messages.copy()
        message_user_latest = {"role": "user", "content": [{ "text": prompt }]}

        attached_names = []
        if chat_input_prompt.files:
            attachment_blocks, unsupported = build_attachment_blocks(chat_input_prompt.files)
            message_user_latest["content"].extend(attachment_blocks)
            attached_names = [f.name for f in chat_input_prompt.files]
            if unsupported:
                st.warning(f"Unsupported file type(s) skipped: {', '.join(unsupported)}")

        message_history.append(message_user_latest)
        with st.chat_message("user"):
            st.write(prompt)
            for f in chat_input_prompt.files or []:
                if f.type in mime_mapping_image:
                    st.image(f.getvalue(), caption=f.name, width=300)
                elif f.type in mime_mapping_document:
                    st.caption(f":green[Document: {f.name}]")

        system_prompts = [{"text" : opt_system_msg}]

        inference_config = {
            #"temperature": opt_temperature,
            "maxTokens": opt_max_tokens,
            #"topP": opt_top_p,
            #stopSequences
        }

        if opt_fm_temperature.isSupported():
            inference_config["temperature"] = opt_temperature

        if opt_fm_top_p.isSupported():
            inference_config["topP"] = opt_top_p

        additional_model_fields = {}

        if opt_fm_top_k.isSupported():
            additional_model_fields[opt_fm_top_k.Name] = opt_top_k

        # If additional_model_fields is an empty dictionary, set it to None
        if additional_model_fields == {}:
            additional_model_fields = None

        # additional_model_fields = {"top_k": opt_top_k}
        # if opt_model_id.startswith("cohere"):
        #     additional_model_fields = None
        # if opt_model_id.startswith("meta") or opt_model_id.startswith("us.meta"):
        #     additional_model_fields = None
        # if opt_model_id.startswith("mistral"):
        #     additional_model_fields = None



        #print(json.dumps(inference_config, indent=3))
        #print(json.dumps(system_prompts, indent=3))

        with st.spinner('Processing...'):

            try:
                use_us_west_2 = "anthropic.claude-3-5-sonnet-20241022-v2:0" == opt_model_id or "us.anthropic.claude-3-5-sonnet-20241022-v2:0" == opt_model_id or "writer.palmyra" in opt_model_id
                if use_us_west_2:
                        response = bedrock_runtime_us_west_2.converse_stream(
                        modelId=opt_model_id,
                        messages=message_history,
                        system=system_prompts,
                        inferenceConfig=inference_config,
                        additionalModelRequestFields=additional_model_fields
                    )
                else:
                    response = bedrock_runtime.converse_stream(
                        modelId=opt_model_id,
                        messages=message_history,
                        system=system_prompts,
                        inferenceConfig=inference_config,
                        additionalModelRequestFields=additional_model_fields
                    )


                #with st.chat_message("assistant", avatar=setAvatar("assistant")):
                result_text = ""
                with st.chat_message("assistant"):
                    result_container = st.container(border=True)
                    result_area = st.empty()
                    stream = response.get('stream')
                    for event in stream:

                        if 'messageStart' in event:
                            #opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens} role= {event['messageStart']['role']}"
                            #result_container.write(opts)
                            pass

                        if 'contentBlockDelta' in event:
                            # .get(): newer models can emit non-text deltas
                            # (e.g. reasoning blocks) which have no 'text' key
                            text = event['contentBlockDelta']['delta'].get('text', '')
                            result_text += f"{text}"
                            result_area.write(result_text)

                        if 'messageStop' in event:
                            #'stopReason': 'end_turn'|'tool_use'|'max_tokens'|'stop_sequence'|'content_filtered'
                            stop_reason = event['messageStop']['stopReason']
                            if stop_reason == 'end_turn':
                                pass
                            else:
                                stop_reason_display = stop_reason
                                if stop_reason == 'max_tokens':
                                    stop_reason_display = "Insufficient Tokens. Increaes MaxToken Settings."
                                result_text_error = f"{result_text}\n\n:red[Generation Stopped: {stop_reason_display}]"
                                result_area.write(result_text_error)

                        if 'metadata' in event:
                            metadata = event['metadata']
                            if 'usage' in metadata:
                                input_token_count = metadata['usage']['inputTokens']
                                output_token_count = metadata['usage']['outputTokens']
                                total_token_count = metadata['usage']['totalTokens']
                            if 'metrics' in event['metadata']:
                                latency = metadata['metrics']['latencyMs']
                            stats = f"| token.in={input_token_count} token.out={output_token_count} token={total_token_count} latency={latency} provider={opt_fm.provider}"
                            result_container.write(stats)
                            user_name = "default_user"  # Replace with actual user name if available
                            try:
                                push_to_cloudwatch(user_name, input_token_count, output_token_count, total_token_count)
                            except Exception as e:
                                logger.error(f"Failed to push logs to CloudWatch: {str(e)}")

                        if "internalServerException" in event:
                            exception = event["internalServerException"]
                            result_text += f"\n{exception}"
                            result_area.write(result_text)
                        if "modelStreamErrorException" in event:
                            exception = event["modelStreamErrorException"]
                            result_text += f"\n{exception}"
                            result_area.write(result_text)
                        if "throttlingException" in event:
                            exception = event["throttlingException"]
                            result_text += f"\n{exception}"
                            result_area.write(result_text)
                        if "validationException" in event:
                            exception = event["validationException"]
                            result_text += f"\n{exception}"
                            result_area.write(result_text)

                    #col1, col2, col3 = st.columns([1,1,5])

                    #with col1:
                        #st.button(key='copy_button', label='📄', type='primary', on_click=copy_button_clicked, args=[result_text])
                    #    pass
                    #with col2:
                    #    if "audio_stream" not in st.session_state or st.session_state["audio_stream"] == "":
                    #        st.button(key='recite_button', label='▶️', type='primary', on_click=recite_button_clicked, args=[result_text])
                    #with col3:
                    #    #st.markdown('3')
                    #    pass

                message_assistant_latest = {"role": "assistant", "content": [{ "text": result_text }]}

                st.session_state.menu_converse_messages.append(message_user_latest)
                st.session_state.menu_converse_messages.append(message_assistant_latest)


                # Trim message History
                menu_converse_messages = st.session_state.menu_converse_messages
                menu_converse_messages_len = len(menu_converse_messages)
                if menu_converse_messages_len > MAX_MESSAGES:
                    del menu_converse_messages[0 : (menu_converse_messages_len - MAX_MESSAGES) * 2] #make sure we remove both the user and assistant responses

                # Rerun so bottom-panel counters and sidebar stats reflect
                # the exchange that was just appended
                st.rerun()

            except ClientError as err:
                message = err.response["Error"]["Message"]
                logger.error("A client error occurred: %s", message)
                print("A client error occured: " + format(message))
                st.chat_message("system").write(message)
            except ReadTimeoutError as err:
                logger.error("A client error occurred: %s", err)  # Log the error directly
                print("A client error occurred: " + str(err))     # Print the error as a string
                st.chat_message("system").write(str(err))         # Use str() to display in chat message

#if "audio_stream" in st.session_state and st.session_state["audio_stream"] != "":
#    audio_bytes = BytesIO(st.session_state['audio_stream'])
#    st.audio(audio_bytes, format='audio/mp3', autoplay=False)

# Right sidebar content
# Pin the Info Panel so it stays visible while the conversation scrolls.
# The COLUMN must be made sticky (not the inner container): columns stretch
# to the row height, so align-self:flex-start shrinks it to content height
# and leaves the rest of the row for it to slide along. :has() targets the
# column that contains our keyed marker container.
st.html(
    """
<style>
[data-testid="stColumn"]:has(.st-key-info_panel_sticky) {
    position: sticky;
    top: 3.75rem; /* clear the Streamlit header */
    align-self: flex-start;
    height: fit-content;
    /* Keep the panel within the visible area: viewport minus the header
       (3.75rem top offset) and the st.bottom block (command panel + chat
       input). Scroll the panel's own content if taller. */
    max-height: calc(100vh - 3.75rem - 10rem - 10vh);
    overflow-y: auto;
}
</style>
"""
)

with right_sidebar_col, st.container(key="info_panel_sticky"):
    st.markdown("### Info Panel")

    with st.container(border=True):
        st.markdown("**Statistics**")
        st.metric("Messages", len(st.session_state.menu_converse_messages))
        st.metric("Limit", MAX_MESSAGES)

        if st.session_state.menu_converse_messages:
            usage_pct = (len(st.session_state.menu_converse_messages) / MAX_MESSAGES) * 100
            st.progress(usage_pct / 100)
            st.caption(f"{usage_pct:.1f}% used")

    with st.container(border=True):
        st.markdown("**Current Model**")
        st.caption(opt_model_id)
        st.markdown("**Temperature**")
        st.caption(f"{opt_temperature:.1f}")
        st.markdown("**Max Tokens**")
        st.caption(f"{opt_max_tokens}")

    with st.container(border=True):
        st.markdown("**Quick Info**")
        st.caption("Model provider:")
        st.caption(f"**{opt_fm.provider}**")

        features = []
        if opt_fm.isFeatureSupported("vision"):
            features.append("Vision")
        if opt_fm.isFeatureSupported("document_chat"):
            features.append("Documents")

        if features:
            st.caption("Features:")
            for feature in features:
                st.caption(f"✓ {feature}")
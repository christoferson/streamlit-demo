import streamlit as st
import boto3
import settings
import json
import logging
import traceback
import cmn_auth
import cmn_settings
import app_bedrock_lib

from botocore.exceptions import ClientError

AWS_REGION = cmn_settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###### AUTH START #####

if not cmn_auth.check_password():
   st.stop()

######  AUTH END #####

opt_model_id_list = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

opt_bedrock_kb_list = app_bedrock_lib.list_knowledge_bases()
print(opt_bedrock_kb_list)

with st.sidebar:
    opt_kb_id = st.selectbox(label="Knowledge Base ID", options=opt_bedrock_kb_list, index = 0, key="kb_id")
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1, key="temperature")
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k")
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

st.title("💬 Chatbot - Knowledge Base")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        #{"role": "user", "content": "Hello there."},
        #{"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_prompt := st.chat_input():

    message_history = st.session_state.messages.copy()
    message_history.append({"role": "user", "content": user_prompt})
    #st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(user_prompt)

    #user_message =  {"role": "user", "content": f"{prompt}"}
    #messages = [st.session_state.messages]
    print(f"messages={st.session_state.messages}")

    try:

        knowledge_base_id = opt_kb_id.split(" ", 1)[0]
        #knowledge_base_id = opt_kb_id
        kb_retrieve_document_count = 3

        prompt = f"""\n\nHuman: {user_prompt[0:900]}
        Assistant:
        """

        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId = knowledge_base_id,
            retrievalQuery={
                'text': prompt,
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': kb_retrieve_document_count
                }
            }
        )

        context_info = ""
        #reference_elements = []
        for i, retrievalResult in enumerate(response['retrievalResults']):
            uri = retrievalResult['location']['s3Location']['uri']
            text = retrievalResult['content']['text']
            excerpt = text[0:75]
            score = retrievalResult['score']
            print(f"{i} RetrievalResult: {score} {uri} {excerpt}")
            #await msg.stream_token(f"\n{i} RetrievalResult: {score} {uri} {excerpt}\n")
            context_info += f"{text}\n" #context_info += f"<p>${text}</p>\n" #context_info += f"${text}\n"
            #await step.stream_token(f"\n[{i+1}] score={score} uri={uri} len={len(text)} text={excerpt}\n")
            #await step.stream_token(f"\n[{i+1}] score={score} uri={uri} len={len(text)}\n")
            #reference_elements.append(cl.Text(name=f"[{i+1}] {uri}", content=text, display="inline"))
        
        #await step.stream_token(f"\n")
        #step.elements = reference_elements

    except Exception as e:
        logging.error(traceback.format_exc())
        #await msg.stream_token(f"{e}")

    #####
    
    kb_system_message = f"""Use the following context information to answer the user question.
    <context>{context_info}</context>
    Human: {user_prompt}
    """

    kb_system_message = f"""Please answer the question with the provided context while following instructions provided.
    <context>{context_info}
    </context>
    <instructions>
    - Do not reformat, or convert any numeric values. Inserting commas is allowed for readability. - If the unit of measure of the question does not match that of the source document, convert the input to the same unit of measure as the source first before deciding. - When encountering geographic locations in user questions, first establish the administrative division specified in the source document. If the input location differs from this specified division (e.g., the source document uses prefectures or provinces while the user mentions cities), determine the corresponding administrative division (e.g., prefecture or province) where the mentioned city is situated. Use this determined administrative division to answer the question in accordance with the criteria outlined in the source document.  - Present source data in tabular form as markdown.
    </instructions>
    <question>{user_prompt}</question>
    """


    kb_messages = []
    kb_messages.append({"role": "user", "content": kb_system_message})

    request = {
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": opt_temperature,
        "top_p": opt_top_p,
        "top_k": opt_top_k,
        "max_tokens": opt_max_tokens,
        "system": kb_system_message, #opt_system_msg,
        "messages": kb_messages, #message_history #st.session_state.messages
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

        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.messages.append({"role": "assistant", "content": result_text})
        
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)

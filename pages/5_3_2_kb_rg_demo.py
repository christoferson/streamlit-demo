import streamlit as st
import boto3
import settings
import json
import logging
import traceback
import cmn_auth
import cmn_settings
import cmn_constants
import app_bedrock_lib
from datetime import datetime

from botocore.exceptions import ClientError

AWS_REGION = cmn_settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

st.set_page_config(
    page_title="Knowledge Base",
    page_icon="📓",
    layout="centered", # "centered" or "wide"
    initial_sidebar_state="expanded", #"auto", "expanded", or "collapsed"
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.logo(icon_image="images/logo.png", image="images/logo_text.png")



opt_model_id_list = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

#opt_bedrock_kb_list = app_bedrock_lib.list_knowledge_bases_with_options(["hr-titan", "legal-titan", "concur-titan"])
opt_bedrock_kb_list = app_bedrock_lib.list_knowledge_bases_with_options(["hr-titan"])
print(opt_bedrock_kb_list)

def knowledge_base_format_func(text):
    if "hr-titan" in text:
        return "HR"
    elif "legal-titan" in text:
        return "Legal"
    elif "concur-titan" in text:
        return "Concur"
    else:
        return text

class MetadataCondition:
    def __init__(self, operator, key, value):
        self.operator = operator
        self.key = key
        self.value = value

def medatata_create_filter_condition(application_options):

    filter = None

    #metadata_year = application_options.get("metadata_year")
    metadata_category = application_options.get("metadata_category")

    conditions:list[MetadataCondition] = []

    #filter_bucket_path_prefix = f"s3://{AWS_KB_BUCKET}/"
    #if metadata_year != "ALL":
    #    filter_bucket_path_prefix += f"{metadata_year}/"
    #    #conditions.append(MetadataCondition("startsWith", "x-amz-bedrock-kb-source-uri", filter_bucket_path_prefix))
    #    conditions.append(MetadataCondition("stringContains", "x-amz-bedrock-kb-source-uri", f"/{metadata_year}/"))
    #else:
    #    conditions.append(MetadataCondition("startsWith", "x-amz-bedrock-kb-source-uri", filter_bucket_path_prefix))

    if metadata_category != "ALL":
        #values = ["ALL", "FAQ", "Travel", "Wage", "Attendance"],
        category_filter_value = metadata_category.lower()
        conditions.append(MetadataCondition("equals", "category", category_filter_value))

    if len(conditions) == 0:
        pass
    if len(conditions) == 1:
        condition = conditions[0]
        filter = {        
            condition.operator: {
                "key": condition.key,
                "value": condition.value
            }
        }
    else:
        filter_conditions = []
        for condition in conditions:
            filter_conditions.append({
                condition.operator: {
                    "key": condition.key,
                    "value": condition.value
                }
            })
        
        filter = {
            'andAll': filter_conditions
            #'orAll': filter_conditions
        }

    #json_str = json.dumps(filter, indent=3)
    #print(json_str)

    return filter

with st.sidebar:
    st.markdown(":blue[Settings]")
    opt_kb_id = st.selectbox(label="Knowledge Base", options=opt_bedrock_kb_list, index = 0, key="kb_id", format_func=knowledge_base_format_func)
    opt_kb_doc_count = st.slider(label="Document Count", min_value=1, max_value=24, value=10, step=1, key="kb_doc_count")
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="temperature", help=cmn_constants.opt_help_temperature)
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p", help=cmn_constants.opt_help_top_p)
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k", help=cmn_constants.opt_help_top_k)
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    opt_query_transformation = st.selectbox(label="Query Transformation", options=["NONE", "QUERY_DECOMPOSITION"], index = 0, key="query_transformation")
    #opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

st.markdown("💬 Chatbot - Knowledge Base 2-2-6")
#st.markdown("Vector Search then LLM Query")



if "messages" not in st.session_state:
    st.session_state["messages"] = [
        #{"role": "user", "content": "Hello there."},
        #{"role": "assistant", "content": "How can I help you?"}
    ]

if "invocation_metrics" not in st.session_state:
    st.session_state["invocation_metrics"] = []

if "menu_kb_rg_session_id" not in st.session_state:
    st.session_state["menu_kb_rg_session_id"] = None



idx = 0
for msg in st.session_state.messages:
    if idx % 2 != 0:
        chat_message = st.chat_message(msg["role"])
        chat_message.write(msg["content"])
        #chat_message.markdown(f""":blue[{st.session_state.invocation_metrics[idx]}]""")

        # This is to display the reference chunks
        #st.markdown(f"menu_kb_reference_chunk_list: {st.session_state['menu_kb_reference_chunk_list']}")
        if "menu_kb_reference_chunk_list" in st.session_state and st.session_state["menu_kb_reference_chunk_list"] != None:
            show_references = chat_message.checkbox("Show References", value=False, key=f"show_references_{idx}")
            if show_references:
                reference_chunk_idx = 1
                for reference_chunk in st.session_state["menu_kb_reference_chunk_list"]:
                    with chat_message.expander(f"""[{reference_chunk_idx}] :blue[{reference_chunk}]"""):
                        #st.markdown(f""":gray[{reference_chunk_text_list[idx-1]}]""")
                        st.markdown(f""":gray[temp]""")
                    reference_chunk_idx += 1
    else:
        st.chat_message(msg["role"]).write(msg["content"])
    idx += 1


#########################################
document_category = "ALL"
if "hr-titan" in opt_kb_id:
    document_category = st.selectbox(":blue[**Category (Human Resources)**]", ("FAQ", "Attendance", "Trips", "Wage", "Employment"))
elif "legal-titan" in opt_kb_id:
    document_category = st.selectbox(":blue[**Category (Legal)**]", ("FAQ", "Contact"))
elif "concur-titan" in opt_kb_id: #GA
    document_category = st.selectbox(":blue[**Category (General Affairs)**]", ("FAQ", "Concur"))
#########################################

#st.chat_message("system").write(f"Category: {document_category}")

if user_prompt := st.chat_input():

    message_history = st.session_state.messages.copy()
    message_history.append({"role": "user", "content": user_prompt})
    #st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state["menu_kb_reference_chunk_list"] = None
    st.chat_message("user").write(user_prompt)

    #print(f"messages={st.session_state.messages}")
    reference_chunk_list = []
    reference_chunk_text_list = []
    reference_chunk_list_text = "" #"  \n\n  \n\n  Sources:  \n\n  "
    context_info = ""

    try:


        knowledge_base_id = opt_kb_id.split(" ", 1)[0]
        kb_retrieve_document_count = opt_kb_doc_count

        application_options = dict (
            metadata_category = document_category,
        )

        # Get the current timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt = f"""\n\nHuman: {user_prompt[0:900]}
        Assistant:
        """

        # Create the custom prompt with the current time and user query
        custom_prompt = f"""
        Human: You are a helpful assistant. The current time is {current_time}. Please answer the question with the provided context while following instructions provided:

        Question: $query$

        Instructions:
        - Only answer if you know the answer with certainty and is evident from the provided context.
        - Do not reformat, or convert any numeric values. Inserting commas is allowed for readability.
        - Present source data in tabular form as markdown. 

        Context:
        $search_results$

        $output_format_instructions$"""

        retrieval_configuration={
            'vectorSearchConfiguration': {
                'numberOfResults': kb_retrieve_document_count
            }
        }
        vector_search_configuration = retrieval_configuration['vectorSearchConfiguration']
        filters = medatata_create_filter_condition(application_options)
        if filters != None:
            print(json.dumps(filters, indent=2))
            vector_search_configuration['filter'] = filters

        session_id = st.session_state.menu_kb_rg_session_id

        params = {
            "input" : {
                'text': prompt.strip(),
            },
            "retrieveAndGenerateConfiguration" : {
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': opt_model_id,
                    'retrievalConfiguration': retrieval_configuration,
                    'generationConfiguration': {
                        #'additionalModelRequestFields': {
                        #    'string': {...}|[...]|123|123.4|'string'|True|None
                        #},
                        #'guardrailConfiguration': {
                        #    'guardrailId': 'string',
                        #    'guardrailVersion': 'string'
                        #},
                        'inferenceConfig': {
                            'textInferenceConfig': {
                                'maxTokens': opt_max_tokens,
                                #'stopSequences': [
                                #    'string',
                                #],
                                'temperature': opt_temperature,
                                'topP': opt_top_p
                            }
                        },
                        'promptTemplate': {
                            'textPromptTemplate': custom_prompt
                        }
                    },
                },  
            },
            
        }

        if session_id != "" and session_id is not None:
            params["sessionId"] = session_id #session_id=84219eab-2060-4a8f-a481-3356d66b8586
            #st.session_state.menu_kb_rg_session_id

        if opt_query_transformation == "QUERY_DECOMPOSITION":
            params["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["orchestrationConfiguration"] = {
                "queryTransformationConfiguration": {
                    "type": "QUERY_DECOMPOSITION"
                }
            }

        st.json(params, expanded=False)
        
        response = bedrock_agent_runtime.retrieve_and_generate(**params)

        with st.chat_message("assistant"):
            result_text = response['output']['text']
            st.write(result_text)

            # Process and display citations
            if 'citations' in response and len(response['citations']) > 0:
                st.write("Sources:")
                for idx, citation in enumerate(response['citations'], 1):
                    if 'retrievedReferences' in citation:
                        for ref_idx, ref in enumerate(citation['retrievedReferences'], 1):
                            with st.expander(f"Source {idx} (Reference {ref_idx})"):
                                if 'content' in ref and 'text' in ref['content']:
                                    st.write(f"**Content:** {ref['content']['text'][:200]}...")  # Display first 200 characters
                                if 'location' in ref:
                                    location = ref['location']
                                    if 's3Location' in location:
                                        st.write(f"**S3 Location:** {location['s3Location']['uri']}")
                                    elif 'webLocation' in location:
                                        st.write(f"**Web Location:** {location['webLocation']['url']}")
                                    # Add other location types as needed
                                if 'metadata' in ref:
                                    st.write("**Metadata:**")
                                    for key, value in ref['metadata'].items():
                                        st.write(f"- {key}: {value}")  

        ####
        
        st.session_state["menu_kb_rg_session_id"] = response['sessionId']

        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.messages.append({"role": "assistant", "content": result_text})
        #st.session_state.invocation_metrics.append("") # No Metrics for User Query
        #st.session_state.invocation_metrics.append(invocation_metrics) # Metric for AI Response

            
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s\n%s", message, traceback.format_exc())
        st.chat_message("system").write(f"An error occurred: {message}")
    except Exception as e:
        message = str(e)
        logger.error("An unexpected error occurred: %s\n%s", message, traceback.format_exc())
        st.chat_message("system").write(f"An unexpected error occurred: {message}")




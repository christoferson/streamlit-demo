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

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
bedrock_agent_runtime_oregon = boto3.client('bedrock-agent-runtime', region_name="us-west-2") # Oregon


opt_model_id_list = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

opt_bedrock_kb_list = app_bedrock_lib.list_knowledge_bases_with_options(["hr-titan", "legal-titan", "concur-titan"])
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

# NEW: Standalone rerank function using bedrock_agent_runtime.rerank()
def rerank_documents(query_text, documents, reranker_model, top_k):
    """
    Rerank documents using Bedrock's standalone rerank API

    Args:
        query_text (str): The query text to rerank against
        documents (list): List of document texts to rerank
        reranker_model (str): The reranker model ID
        top_k (int): Number of top results to return

    Returns:
        list: Reranked documents with scores
    """
    try:
        # Prepare sources for reranking
        sources = []
        for doc_text in documents:
            sources.append({
                'type': 'INLINE',
                'inlineDocumentSource': {
                    'type': 'TEXT',
                    'textDocument': {
                        'text': doc_text
                    }
                }
            })

        # Prepare queries
        queries = [{
            'type': 'TEXT',
            'textQuery': {
                'text': query_text
            }
        }]

        # Configure reranking
        reranking_configuration = {
            'type': 'BEDROCK_RERANKING_MODEL',
            'bedrockRerankingConfiguration': {
                'modelConfiguration': {
                    'modelArn': f'arn:aws:bedrock:us-west-2::foundation-model/{reranker_model}'
                },
                'numberOfResults': min(top_k, len(documents))
            }
        }

        # Call rerank API
        response = bedrock_agent_runtime_oregon.rerank(
            queries=queries,
            sources=sources,
            rerankingConfiguration=reranking_configuration
        )

        # Process results - FIXED: Use the correct response structure
        reranked_results = []

        # The response contains results with index and relevanceScore
        if 'results' in response:
            for result in response['results']:
                doc_index = result['index']
                relevance_score = result['relevanceScore']

                # Make sure the index is valid
                if 0 <= doc_index < len(documents):
                    reranked_results.append({
                        'text': documents[doc_index],  # Get original document by index
                        'score': relevance_score,
                        'index': doc_index
                    })
                else:
                    logger.warning(f"Invalid document index: {doc_index}")
        else:
            logger.warning("No 'results' found in rerank response")

        # Results are already sorted by relevance score (highest first) from the API
        return reranked_results

    except Exception as e:
        logger.error(f"Error in reranking: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def rerank_documents_filter(query_text, documents, reranker_model, top_k, relevance_threshold=0.6):
    """
    Rerank documents using Bedrock's standalone rerank API and filter by relevance threshold

    Args:
        query_text (str): The query text to rerank against
        documents (list): List of document texts to rerank
        reranker_model (str): The reranker model ID
        top_k (int): Number of top results to return before filtering
        relevance_threshold (float): Minimum relevance score to include (default: 0.6)

    Returns:
        list: Filtered reranked documents with scores above threshold
    """
    try:
        # Get reranked results using existing method
        reranked_results = rerank_documents(query_text, documents, reranker_model, top_k)

        if not reranked_results:
            return []

        # Filter by relevance threshold
        filtered_results = [
            result for result in reranked_results 
            if result['score'] >= relevance_threshold
        ]

        logger.info(f"Reranking: {len(reranked_results)} total → {len(filtered_results)} above threshold {relevance_threshold}")

        return filtered_results

    except Exception as e:
        logger.error(f"Error in rerank_documents_filter: {str(e)}")
        logger.error(traceback.format_exc())
        return []

with st.sidebar:
    st.markdown(":blue[Settings]")
    opt_kb_id = st.selectbox(label="Knowledge Base", options=opt_bedrock_kb_list, index = 0, key="kb_id", format_func=knowledge_base_format_func)
    opt_kb_doc_count = st.slider(label="Document Count", min_value=1, max_value=24, value=10, step=1, key="kb_doc_count")
    opt_model_id = st.selectbox(label="Model ID", options=opt_model_id_list, index = 0, key="model_id")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="temperature", help=cmn_constants.opt_help_temperature)
    opt_top_p = st.slider(label="Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p", help=cmn_constants.opt_help_top_p)
    opt_top_k = st.slider(label="Top K", min_value=0, max_value=500, value=250, step=1, key="top_k", help=cmn_constants.opt_help_top_k)
    opt_max_tokens = st.slider(label="Max Tokens", min_value=0, max_value=4096, value=2048, step=1, key="max_tokens")
    #opt_system_msg = st.text_area(label="System Message", value="", key="system_msg")

    # Add reranker options
    st.markdown("---")
    st.markdown(":blue[Reranking Settings]")
    #opt_enable_reranker = st.checkbox(label="Enable Reranker", value=False, key="enable_reranker")
    opt_reranking_method = st.selectbox(
        label="Reranking Method", 
        options=["None", "Built-in (with retrieve)", "Standalone (separate API)"], 
        index=0, 
        key="reranking_method",
        help="Choose how to apply reranking: None, built into retrieve call, or separate rerank API call"
    )
    ##cohere.rerank-v3-5:0, amazon.rerank-v1:0
    if opt_reranking_method != "None":
        opt_reranker_model_list = [
            "cohere.rerank-v3-5:0",
            "amazon.rerank-v1:0",
            
        ]
        opt_reranker_model = st.selectbox(label="Reranker Model", options=opt_reranker_model_list, index=0, key="reranker_model")
        opt_reranker_top_k = st.slider(label="Reranker Top K", min_value=1, max_value=opt_kb_doc_count, value=min(5, opt_kb_doc_count), step=1, key="reranker_top_k", 
            help="Number of documents to return after reranking")
        opt_relevance_threshold = st.slider(
            label="Relevance Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6, 
            step=0.05, 
            key="relevance_threshold",
            help="Filter out results below this relevance score"
        )

st.title("💬 Chatbot - Knowledge Base (bedrock_agent_runtime.retrieve)")
st.markdown("Vector Search then LLM Query")



if "messages" not in st.session_state:
    st.session_state["messages"] = [
        #{"role": "user", "content": "Hello there."},
        #{"role": "assistant", "content": "How can I help you?"}
    ]

if "invocation_metrics" not in st.session_state:
    st.session_state["invocation_metrics"] = []

idx = 0
for msg in st.session_state.messages:
    if idx % 2 != 0:
        chat_message = st.chat_message(msg["role"])
        chat_message.write(msg["content"])
        chat_message.markdown(f""":blue[{st.session_state.invocation_metrics[idx]}]""")

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
            #retrieve_search_type = settings["RetrieveSearchType"],
            #metadata_year = settings["MetadataYear"],
            metadata_category = document_category,

            #option_terse = settings["Terse"],
            #option_strict = settings["Strict"],
            #option_source_table_markdown_display = settings["SourceTableMarkdown"],
        )

        prompt = f"""\n\nHuman: {user_prompt[0:900]}
        Assistant:
        """

        retrieval_configuration={
            'vectorSearchConfiguration': {
                'numberOfResults': kb_retrieve_document_count
            }
        }

        # Add reranking configuration if enabled
        if opt_reranking_method == "Built-in (with retrieve)":
            retrieval_configuration['vectorSearchConfiguration']['rerankingConfiguration'] = {
                'type': 'BEDROCK_RERANKING_MODEL',
                'bedrockRerankingConfiguration': {
                    'modelConfiguration': {
                        'modelArn': f'arn:aws:bedrock:us-west-2::foundation-model/{opt_reranker_model}'
                    },
                    'numberOfRerankedResults': opt_reranker_top_k  # Note: it's 'numberOfRerankedResults', not 'numberOfResults'
                }
            }


        vector_search_configuration = retrieval_configuration['vectorSearchConfiguration']
        filters = medatata_create_filter_condition(application_options)
        if filters != None:
            print(json.dumps(filters, indent=2))
            vector_search_configuration['filter'] = filters

        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId = knowledge_base_id,
            retrievalQuery={
                'text': prompt,
            },
            retrievalConfiguration=retrieval_configuration
        )

        # Add this line immediately after the retrieve call:
        initial_results = response['retrievalResults']


        # Apply standalone reranking if selected
        if opt_reranking_method == "Standalone (separate API)":
            st.info("🔄 Applying standalone reranking...")

            # Extract document texts for reranking - FIX: Use correct path
            document_texts = [result['content']['text'] for result in initial_results]

            # Perform standalone reranking
            # Perform standalone reranking with filtering
            reranked_results = rerank_documents_filter(
                query_text=user_prompt,
                documents=document_texts,
                reranker_model=opt_reranker_model,
                top_k=opt_reranker_top_k,
                relevance_threshold=opt_relevance_threshold
            )

            if reranked_results:
                # Map reranked results back to original results
                final_results = []
                for reranked in reranked_results:
                    original_result = initial_results[reranked['index']]
                    final_results.append({
                        'uri': original_result['location']['s3Location']['uri'],
                        'text': reranked['text'],
                        'score': reranked['score'],  # Use reranked score
                        'original_score': original_result['score'],
                        'reranked': True
                    })
            else:
                # Fallback to original results if reranking fails
                final_results = []
                for result in initial_results:
                    final_results.append({
                        'uri': result['location']['s3Location']['uri'],
                        'text': result['content']['text'],  # FIX: Use correct path
                        'score': result['score'],
                        'reranked': False
                    })
                st.warning("⚠️ Reranking failed, using original results")
        else:
            # Use original results (either no reranking or built-in reranking)
            final_results = []
            for result in initial_results:
                final_results.append({
                    'uri': result['location']['s3Location']['uri'],  # FIX: Use correct path
                    'text': result['content']['text'],  # FIX: Use correct path
                    'score': result['score'],
                    'reranked': opt_reranking_method == "Built-in (with retrieve)"
                })
            

        
        # Process final results for display and context
        for i, result in enumerate(final_results):
            uri = result['uri']
            text = result['text']
            score = result['score']
            is_reranked = result.get('reranked', False)

            excerpt = text[0:75]
            rerank_indicator = " (reranked)" if is_reranked else ""

            # For standalone reranking, show both scores
            if opt_reranking_method == "Standalone (separate API)" and 'original_score' in result:
                score_display = f"{score:.4f} (orig: {result['original_score']:.4f})"
            else:
                score_display = f"{score:.4f}"

            print(f"{i} RetrievalResult{rerank_indicator}: {score_display} {uri} {excerpt}")

            context_info += f"{text}\n"
            uri_name = uri.split('/')[-1]
            reference_chunk_list.append(f"{score_display}{rerank_indicator} {uri_name}")
            reference_chunk_text_list.append(text)
            reference_chunk_list_text += f"[{i}] {score_display}{rerank_indicator} {uri_name} \n\n  "



    except Exception as e:
        logging.error(traceback.format_exc())
        st.chat_message("system").write(e)

    #####
    
    kb_system_message = f"""Use the following context information to answer the user question.
    <context>{context_info}</context>
    Human: {user_prompt}
    """

    kb_system_message = f"""Please answer the question with the provided context while following instructions provided.
    <context>{context_info}
    </context>
    <instructions>
    - Do not reformat, or convert any numeric values. Inserting commas is allowed for readability. 
    - If the unit of measure of the question does not match that of the source document, convert the input to the same unit of measure as the source first before deciding. 
    - When encountering geographic locations in user questions, first establish the administrative division specified in the source document. If the input location differs from this specified division (e.g., the source document uses prefectures or provinces while the user mentions cities), determine the corresponding administrative division (e.g., prefecture or province) where the mentioned city is situated. Use this determined administrative division to answer the question in accordance with the criteria outlined in the source document.  
    - When the source information is a table data, display the table data as markdown. Otherwise, do not reformat the source data.
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
        invocation_metrics = ""
        with st.chat_message("assistant"):
            result_area = st.empty()
            sources_area = st.empty()
            result_container = st.container(border=True)
            stream = response["body"]
            for event in stream:
                
                if event["chunk"]:

                    chunk = json.loads(event["chunk"]["bytes"])

                    if chunk['type'] == 'message_start':
                        opts = f"| temperature={opt_temperature} top_p={opt_top_p} top_k={opt_top_k} max_tokens={opt_max_tokens}"
                        #result_container.write(opts)

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
                        #result_container.write(stats)
                        invocation_metrics = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                        #result_container.markdown(f""":blue[{invocation_metrics}]""")
                        #result_area.markdown(f"{invocation_metrics} {result_text} ")
                        result_text_final = f"""{result_text}   \n\n\n
                        {reference_chunk_list_text}
                        """
                        #result_text += f"{reference_chunk_list_text}"
                        #result_area.write(f"{result_text_final}")
                        #result_container.markdown()
                        result_area.write(f"{result_text} \n\n  ")
                        #result_container.markdown(f"""{reference_chunk_list_text}""")
                        #with st.expander("Sources"):
                            #st.write(f"{reference_chunk_list_text}")
                        #    for idx, reference_chunk in reference_chunk_list:
                        #        st.write(f"{reference_chunk}")
                        sources_area.markdown("  \n\n")
                        idx = 1
                        for reference_chunk in reference_chunk_list:
                            with st.expander(f"""[{idx}] :green[{reference_chunk}]"""):
                                st.markdown(f""":gray[{reference_chunk_text_list[idx-1]}]""")
                            idx += 1
                        st.session_state["menu_kb_reference_chunk_list"] = reference_chunk_list

                elif "internalServerException" in event:
                    exception = event["internalServerException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                elif "modelStreamErrorException" in event:
                    exception = event["modelStreamErrorException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                elif "modelTimeoutException" in event:
                    exception = event["modelTimeoutException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                elif "throttlingException" in event:
                    exception = event["throttlingException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                elif "validationException" in event:
                    exception = event["validationException"]
                    result_text += f"\n\{exception}"
                    result_area.write(result_text)
                else:
                    result_text += f"\n\nUnknown Token"
                    result_area.write(result_text)
        

        ####
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.messages.append({"role": "assistant", "content": result_text})
        st.session_state.invocation_metrics.append("") # No Metrics for User Query
        st.session_state.invocation_metrics.append(invocation_metrics) # Metric for AI Response

        
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " + format(message))
        st.chat_message("system").write(message)




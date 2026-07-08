import streamlit as st
import boto3
import cmn_settings
import logging
import json
from botocore.exceptions import ClientError

AWS_REGION = cmn_settings.AWS_REGION

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Bedrock Knowledge Bases",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

st.markdown("##### :blue[Bedrock Managed Knowledge Bases]")


# ============================================================================
# AWS Clients
# ============================================================================

bedrock_agent = boto3.client('bedrock-agent', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

try:
    AWS_ACCOUNT_ID = boto3.client('sts', region_name=AWS_REGION).get_caller_identity()['Account']
except Exception:
    AWS_ACCOUNT_ID = ""
    logger.warning("Could not retrieve AWS account ID — us.* inference profile ARNs may fail")


# ============================================================================
# API Methods
# ============================================================================

def list_knowledge_bases(client, max_results=50):
    try:
        response = client.list_knowledge_bases(maxResults=max_results)
        logger.info(f"list_knowledge_bases response keys: {list(response.keys())}")
        summaries = response.get('knowledgeBaseSummaries', [])
        logger.info(f"Found {len(summaries)} knowledge base(s)")
        kb_list = []
        for kb in summaries:
            kb_id = kb.get('knowledgeBaseId', '')
            kb_name = kb.get('name', 'Unknown')
            kb_status = kb.get('status', 'Unknown')
            kb_list.append({
                'id': kb_id,
                'name': kb_name,
                'status': kb_status,
                'description': kb.get('description', ''),
                'updatedAt': kb.get('updatedAt', ''),
                'display': f"{kb_name} ({kb_id})",
            })
        return kb_list
    except (ClientError, Exception) as e:
        logger.error(f"Error listing knowledge bases: {e}")
        return []


def get_knowledge_base(client, kb_id):
    try:
        response = client.get_knowledge_base(knowledgeBaseId=kb_id)
        logger.info(f"get_knowledge_base response keys: {list(response.keys())}")
        return response.get('knowledgeBase', {})
    except (ClientError, Exception) as e:
        logger.error(f"Error getting knowledge base: {e}")
        return None


def list_data_sources(client, kb_id):
    try:
        response = client.list_data_sources(knowledgeBaseId=kb_id)
        logger.info(f"list_data_sources: {len(response.get('dataSourceSummaries', []))} source(s)")
        return response.get('dataSourceSummaries', [])
    except (ClientError, Exception) as e:
        logger.error(f"Error listing data sources: {e}")
        return None


def list_knowledge_base_documents(client, kb_id, data_source_id, max_results=50, next_token=None):
    try:
        params = {
            'knowledgeBaseId': kb_id,
            'dataSourceId': data_source_id,
            'maxResults': max_results,
        }
        if next_token:
            params['nextToken'] = next_token
        response = client.list_knowledge_base_documents(**params)
        logger.info(f"list_knowledge_base_documents: {len(response.get('documentDetails', []))} doc(s)")
        return {
            'documents': response.get('documentDetails', []),
            'nextToken': response.get('nextToken'),
        }
    except (ClientError, Exception) as e:
        logger.error(f"Error listing KB documents: {e}")
        return None


# ============================================================================
# Render Helpers
# ============================================================================

def render_kb_details_dialog(detail):
    kb_id = detail.get('knowledgeBaseId', 'N/A')
    name = detail.get('name', 'N/A')
    status = detail.get('status', 'N/A')
    status_color = "green" if status == "ACTIVE" else "orange" if status in ("CREATING", "UPDATING") else "red"

    st.subheader("Knowledge Base Details")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Basic Information:**")
        st.markdown(f":blue[Name: {name}]")
        st.markdown(f":blue[ID: `{kb_id}`]")
        st.markdown(f":{status_color}[Status: {status}]")
        arn = detail.get('knowledgeBaseArn', 'N/A')
        st.markdown(f":blue[ARN: `{arn}`]")

    with col2:
        st.markdown("**Timestamps:**")
        created = detail.get('createdAt', 'N/A')
        updated = detail.get('updatedAt', 'N/A')
        if hasattr(created, 'strftime'):
            created = created.strftime('%Y-%m-%d %H:%M:%S')
        if hasattr(updated, 'strftime'):
            updated = updated.strftime('%Y-%m-%d %H:%M:%S')
        st.markdown(f":blue[Created: {created}]")
        st.markdown(f":blue[Updated: {updated}]")
        role_arn = detail.get('roleArn', '')
        if role_arn:
            st.markdown(f":blue[Role ARN: `{role_arn}`]")

    if detail.get('description'):
        st.markdown("**Description:**")
        st.info(detail['description'])

    if detail.get('failureReasons'):
        st.markdown("**Failure Reasons:**")
        for reason in detail['failureReasons']:
            st.error(reason)

    # Knowledge base configuration
    kb_config = detail.get('knowledgeBaseConfiguration', {})
    if kb_config:
        st.markdown("**Knowledge Base Configuration:**")
        with st.container(border=True):
            kb_type = kb_config.get('type', 'N/A')
            st.markdown(f":blue[Type: **{kb_type}**]")

            if kb_type == 'VECTOR':
                vec_cfg = kb_config.get('vectorKnowledgeBaseConfiguration', {})
                if vec_cfg:
                    st.markdown(f":blue[Embedding Model: `{vec_cfg.get('embeddingModelArn', 'N/A')}`]")

            elif kb_type == 'MANAGED':
                mgd_cfg = kb_config.get('managedKnowledgeBaseConfiguration', {})
                if mgd_cfg:
                    st.markdown(f":blue[Embedding Model: `{mgd_cfg.get('embeddingModelArn', 'N/A')}`]")
                    if mgd_cfg.get('storageConfiguration'):
                        st.markdown(f":blue[Storage: {mgd_cfg['storageConfiguration']}]")

            elif kb_type == 'KENDRA':
                kendra_cfg = kb_config.get('kendraKnowledgeBaseConfiguration', {})
                if kendra_cfg:
                    st.markdown(f":blue[Kendra Index ARN: `{kendra_cfg.get('kendraIndexArn', 'N/A')}`]")

            elif kb_type == 'SQL':
                sql_cfg = kb_config.get('sqlKnowledgeBaseConfiguration', {})
                if sql_cfg:
                    st.markdown(f":blue[SQL Type: {sql_cfg.get('type', 'N/A')}]")

    # Storage configuration
    storage_config = detail.get('storageConfiguration', {})
    if storage_config:
        st.markdown("**Storage Configuration:**")
        with st.container(border=True):
            storage_type = storage_config.get('type', 'N/A')
            st.markdown(f":blue[Type: **{storage_type}**]")
            type_key_map = {
                'OPENSEARCH_SERVERLESS': 'opensearchServerlessConfiguration',
                'OPENSEARCH_MANAGED_CLUSTER': 'opensearchManagedClusterConfiguration',
                'PINECONE': 'pineconeConfiguration',
                'REDIS_ENTERPRISE_CLOUD': 'redisEnterpriseCloudConfiguration',
                'RDS': 'rdsConfiguration',
                'MONGO_DB_ATLAS': 'mongoDbAtlasConfiguration',
                'NEPTUNE_ANALYTICS': 'neptuneAnalyticsConfiguration',
                'S3_VECTORS': 's3VectorsConfiguration',
            }
            cfg_key = type_key_map.get(storage_type)
            if cfg_key and storage_config.get(cfg_key):
                cfg = storage_config[cfg_key]
                collection_or_index = (
                    cfg.get('collectionArn') or cfg.get('clusterArn') or
                    cfg.get('endpoint') or cfg.get('resourceArn') or
                    cfg.get('graphArn') or cfg.get('indexArn') or ''
                )
                if collection_or_index:
                    st.markdown(f":blue[Endpoint/ARN: `{collection_or_index}`]")
                vector_index = cfg.get('vectorIndexName') or cfg.get('indexName', '')
                if vector_index:
                    st.markdown(f":blue[Index: {vector_index}]")

    with st.expander("📄 Full Response", expanded=False):
        st.json(detail)


def render_kb_documents_dialog(kb_id, kb_name):
    st.subheader("Documents")
    st.markdown(f":violet[KB: **{kb_name}**]")

    with st.spinner("Loading data sources..."):
        data_sources = list_data_sources(bedrock_agent, kb_id)

    if data_sources is None:
        st.error("Failed to load data sources.")
        return
    if not data_sources:
        st.info("No data sources found for this knowledge base.")
        return

    ds_options = {
        f"{ds.get('name', ds.get('dataSourceId', ''))} ({ds.get('dataSourceId', '')})": ds.get('dataSourceId', '')
        for ds in data_sources
    }
    selected_ds_display = st.selectbox("Data Source", options=list(ds_options.keys()))
    selected_ds_id = ds_options[selected_ds_display]

    if st.button("🔄 Load Documents", type="primary", use_container_width=True):
        with st.spinner("Loading documents..."):
            result = list_knowledge_base_documents(bedrock_agent, kb_id, selected_ds_id)
        if result is None:
            st.error("Failed to load documents. Check logs.")
            return
        st.session_state["kb_docs_result"] = result

    result = st.session_state.get("kb_docs_result")
    if result is not None:
        docs = result['documents']
        st.markdown(f":blue[**{len(docs)} document(s)**]")
        for doc in docs:
            status = doc.get('status', 'N/A')
            status_color = (
                "green" if status == "INDEXED"
                else "orange" if status in ("PENDING", "STARTING", "IN_PROGRESS", "PARTIALLY_INDEXED", "METADATA_PARTIALLY_INDEXED")
                else "red"
            )
            identifier = doc.get('identifier', {})
            doc_label = (
                identifier.get('s3', {}).get('uri')
                or identifier.get('custom', {}).get('id')
                or 'N/A'
            )
            updated_at = doc.get('updatedAt', '')
            if hasattr(updated_at, 'strftime'):
                updated_at = updated_at.strftime('%Y-%m-%d %H:%M:%S')

            meta_parts = [f"<span style='color:var(--text-color);opacity:0.5'>{updated_at}</span>" if updated_at else ""]
            if doc.get('statusReason'):
                meta_parts.append(f"<span style='color:orange'>{doc['statusReason']}</span>")
            meta_str = " &nbsp;·&nbsp; ".join(p for p in meta_parts if p)
            st.markdown(
                f"<div style='font-size:0.85rem;padding:2px 0'>"
                f"<b>{doc_label}</b> &nbsp; "
                f"<span style='color:{'green' if status_color=='green' else 'orange' if status_color=='orange' else 'red'}'>{status}</span>"
                f"{'&nbsp;&nbsp;<small>' + meta_str + '</small>' if meta_str else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )

        if result.get('nextToken'):
            if st.button("Load More", type="secondary"):
                with st.spinner("Loading more..."):
                    more = list_knowledge_base_documents(
                        bedrock_agent, kb_id, selected_ds_id,
                        next_token=result['nextToken']
                    )
                if more:
                    st.session_state["kb_docs_result"] = {
                        'documents': docs + more['documents'],
                        'nextToken': more.get('nextToken'),
                    }
                    st.rerun()


# ============================================================================
# Cached wrappers
# ============================================================================

@st.cache_data(ttl=300)
def list_knowledge_bases_cached():
    return list_knowledge_bases(bedrock_agent)


# ============================================================================
# Session State
# ============================================================================

# retrieve_and_generate requires an inference-profile ARN (with account ID) for these models —
# on-demand foundation-model ARNs are rejected: "Invocation ... with on-demand throughput isn't supported."
# Note: Sonnet 5 / Opus 4.8 additionally require custom prompt templates for RAG — dropped for now.
RAG_MODEL_CONFIGS = {
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "model_arn": f"arn:aws:bedrock:us-east-1:{AWS_ACCOUNT_ID}:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "temperature_supported": True, "temperature_default": 0.1,
    },
}
RAG_MODEL_IDS = list(RAG_MODEL_CONFIGS.keys())

RAG_INFERENCE_REGIONS = [
    "us-east-1",
    "us-west-2",
    "eu-west-1",
    "ap-northeast-1",
]

if "selected_kb_id" not in st.session_state:
    st.session_state.selected_kb_id = None
if "selected_kb_type" not in st.session_state:
    st.session_state.selected_kb_type = None
if "selected_kb_ds_count" not in st.session_state:
    st.session_state.selected_kb_ds_count = None

RAG_MODES = ["Retrieve & Generate", "Retrieve", "Agentic Retrieve"]

# Which modes each KB type supports (verified against the API):
#   VECTOR  → retrieve + retrieve_and_generate; agentic_retrieve is not supported
#   MANAGED → retrieve + agentic_retrieve_stream; retrieve_and_generate is not supported
#   KENDRA / SQL → retrieve + retrieve_and_generate
def supported_modes_for_kb_type(kb_type):
    if kb_type == "MANAGED":
        return ["Retrieve", "Agentic Retrieve"]
    if kb_type in ("VECTOR", "KENDRA", "SQL"):
        return ["Retrieve & Generate", "Retrieve"]
    return list(RAG_MODES)  # unknown type — allow all, let the API decide


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    kb_col, refresh_col = st.columns([4, 1])
    kb_col.markdown("**Knowledge Bases**")
    if refresh_col.button("↻", help="Refresh knowledge base list", key="refresh_kbs"):
        list_knowledge_bases_cached.clear()
        st.rerun()

    with st.spinner("Loading knowledge bases..."):
        available_kbs = list_knowledge_bases_cached()

    if available_kbs:
        kb_options = {kb['display']: kb['id'] for kb in available_kbs}

        if st.session_state.selected_kb_id is None:
            st.session_state.selected_kb_id = list(kb_options.values())[0]

        selected_index = 0
        for idx, (display, kb_id) in enumerate(kb_options.items()):
            if kb_id == st.session_state.selected_kb_id:
                selected_index = idx
                break

        selected_display = st.selectbox(
            "Select Knowledge Base",
            options=list(kb_options.keys()),
            index=selected_index,
            key="kb_selectbox"
        )

        current_kb_id = kb_options[selected_display]
        selected_kb = next(kb for kb in available_kbs if kb['display'] == selected_display)

        with st.container(border=True):
            st.markdown(f":blue[**Name:** {selected_kb['name']}]")
            st.markdown(f":blue[**ID:** `{selected_kb['id']}`]")
            status = selected_kb['status']
            status_color = "green" if status == "ACTIVE" else "orange" if status in ("CREATING", "UPDATING") else "red"
            st.markdown(f":{status_color}[**Status:** {status}]")
            if selected_kb['description']:
                st.markdown(f":blue[{selected_kb['description']}]")
            if selected_kb['updatedAt']:
                ts = selected_kb['updatedAt']
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts)[:19]
                st.markdown(f":blue[Updated: {ts_str}]")

            is_selected = st.session_state.selected_kb_id == current_kb_id

            if is_selected:
                st.success("✓ Active")
                caption_bits = []
                if st.session_state.selected_kb_type:
                    caption_bits.append(f"Type: {st.session_state.selected_kb_type}")
                if st.session_state.selected_kb_ds_count is not None:
                    caption_bits.append(f"Data sources: {st.session_state.selected_kb_ds_count}")
                if caption_bits:
                    st.caption(" · ".join(caption_bits))
            else:
                if st.button("Select This KB", use_container_width=True, type="primary"):
                    with st.spinner("Loading knowledge base configuration..."):
                        detail = get_knowledge_base(bedrock_agent, current_kb_id)
                        data_sources = list_data_sources(bedrock_agent, current_kb_id)
                    kb_type = (detail or {}).get("knowledgeBaseConfiguration", {}).get("type")
                    st.session_state.selected_kb_id = current_kb_id
                    st.session_state.selected_kb_type = kb_type
                    st.session_state.selected_kb_ds_count = len(data_sources) if data_sources is not None else 0
                    logger.info(f"KB selected: {current_kb_id} type={kb_type} data_sources={st.session_state.selected_kb_ds_count}")
                    st.rerun()

            if st.button("📋 Show Details", use_container_width=True, type="secondary", key=f"details_{current_kb_id}"):
                with st.spinner("Loading knowledge base details..."):
                    detail = get_knowledge_base(bedrock_agent, current_kb_id)
                    logger.info(f"KB details fetched: {detail}")
                if detail:
                    @st.dialog(f"Knowledge Base: {selected_kb['name']}", width="large")
                    def show_kb_details_dialog():
                        render_kb_details_dialog(detail)
                    show_kb_details_dialog()
                else:
                    st.error("Failed to load knowledge base details.")

            if st.button("📄 Documents", use_container_width=True, type="secondary", key=f"docs_{current_kb_id}"):
                @st.dialog(f"Documents: {selected_kb['name']}", width="large")
                def show_kb_documents_dialog():
                    render_kb_documents_dialog(current_kb_id, selected_kb['name'])
                show_kb_documents_dialog()

        st.markdown("##### Inference Parameters")
        with st.container(border=True):
            rag_model = st.selectbox("Model", options=RAG_MODEL_IDS, index=0, key="rag_model_id")
            model_cfg = RAG_MODEL_CONFIGS[rag_model]

            rag_region = st.selectbox("Inference Region", options=RAG_INFERENCE_REGIONS,
                                      index=RAG_INFERENCE_REGIONS.index(AWS_REGION) if AWS_REGION in RAG_INFERENCE_REGIONS else 0,
                                      key="rag_region")

            if model_cfg.get("temperature_supported", False):
                override_temp = st.checkbox("Override Temperature", value=False, key="rag_override_temperature")
                if override_temp:
                    temp_default = model_cfg.get("temperature_default", 0.1)
                    st.slider("Temperature", min_value=0.0, max_value=1.0, value=temp_default, step=0.05, key="rag_temperature")
                else:
                    st.session_state["rag_temperature"] = None
            else:
                st.session_state["rag_temperature"] = None
                st.markdown(":orange[Temperature not supported for this model]")

    else:
        st.warning("No knowledge bases found in your account")


# ============================================================================
# Main Area — Retrieve and Generate
# ============================================================================

def agentic_retrieve_stream(runtime_client, kb_id, messages, model_arn, generate_response=True):
    try:
        response = runtime_client.agentic_retrieve_stream(
            agenticRetrieveConfiguration={
                'foundationModelType': 'CUSTOM',
                'foundationModelConfiguration': {
                    'type': 'BEDROCK_FOUNDATION_MODEL',
                    'bedrockFoundationModelConfiguration': {
                        'modelConfiguration': {'modelArn': model_arn}
                    }
                },
            },
            messages=messages,
            retrievers=[{'configuration': {'knowledgeBase': {'knowledgeBaseId': kb_id}}}],
            generateResponse=generate_response,
        )
        return response.get('stream')
    except (ClientError, Exception) as e:
        logger.error(f"Error in agentic_retrieve_stream: {e}")
        return None


def retrieve_only(runtime_client, kb_id, query, kb_type=None, top_k=10):
    try:
        # MANAGED KBs require managedSearchConfiguration; others use vectorSearchConfiguration.
        if kb_type == "MANAGED":
            search_cfg = {"managedSearchConfiguration": {"numberOfResults": top_k}}
        else:
            search_cfg = {"vectorSearchConfiguration": {"numberOfResults": top_k}}
        response = runtime_client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration=search_cfg,
        )
        logger.info(f"retrieve response keys: {list(response.keys())}")
        return response.get("retrievalResults", [])
    except (ClientError, Exception) as e:
        logger.error(f"Error in retrieve: {e}")
        return None


def retrieve_and_generate(runtime_client, kb_id, model_arn, query, session_id=None, temperature=None):
    kb_cfg = {"knowledgeBaseId": kb_id, "modelArn": model_arn}
    if temperature is not None:
        kb_cfg["generationConfiguration"] = {
            "inferenceConfig": {"textInferenceConfig": {"temperature": temperature}}
        }
    params = {
        "input": {"text": query},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": kb_cfg,
        },
    }
    if session_id:
        params["sessionId"] = session_id
    try:
        logger.info(f"retrieve_and_generate request:\n{json.dumps(params, indent=2, default=str)}")
        response = runtime_client.retrieve_and_generate(**params)
        logger.info(f"retrieve_and_generate response keys: {list(response.keys())}")
        return response
    except (ClientError, Exception) as e:
        logger.error(f"retrieve_and_generate FAILED\nrequest:\n{json.dumps(params, indent=2, default=str)}\nerror: {e}")
        return None

if not st.session_state.selected_kb_id:
    st.info("Select a knowledge base from the sidebar to start querying.")
else:
    rag_model_id = st.session_state.get("rag_model_id", RAG_MODEL_IDS[0])
    if rag_model_id not in RAG_MODEL_CONFIGS:
        rag_model_id = RAG_MODEL_IDS[0]
        st.session_state["rag_model_id"] = rag_model_id
    rag_region = st.session_state.get("rag_region", AWS_REGION)
    rag_temperature = st.session_state.get("rag_temperature")
    model_arn = RAG_MODEL_CONFIGS[rag_model_id]["model_arn"]
    logger.info(f"RAG model_id={rag_model_id} model_arn={model_arn}")

    # Resolve KB type / data-source count if not fetched yet (e.g. auto-selected on first load)
    if st.session_state.selected_kb_type is None or st.session_state.selected_kb_ds_count is None:
        with st.spinner("Loading knowledge base configuration..."):
            _detail = get_knowledge_base(bedrock_agent, st.session_state.selected_kb_id)
            _data_sources = list_data_sources(bedrock_agent, st.session_state.selected_kb_id)
        st.session_state.selected_kb_type = (_detail or {}).get("knowledgeBaseConfiguration", {}).get("type")
        st.session_state.selected_kb_ds_count = len(_data_sources) if _data_sources is not None else 0

    kb_type = st.session_state.selected_kb_type
    ds_count = st.session_state.selected_kb_ds_count
    st.markdown(
        f":violet[**KB:** {st.session_state.selected_kb_id}] &nbsp; "
        f":green[**Type:** {kb_type or 'unknown'}] &nbsp; "
        f":blue[**Model:** {rag_model_id}] &nbsp; :blue[**Region:** {rag_region}]"
    )

    if not ds_count:
        st.error(
            "This knowledge base has **no data sources** configured, so it contains no "
            "documents to query. Add and sync a data source, then try again."
        )
    else:
        allowed_modes = supported_modes_for_kb_type(kb_type)
        rag_mode = st.pills("Mode", allowed_modes, default=allowed_modes[0], key="rag_mode", label_visibility="collapsed")
        if rag_mode is None:
            rag_mode = allowed_modes[0]

        runtime_client = boto3.client('bedrock-agent-runtime', region_name=rag_region)

        prompt = st.chat_input("Ask a question...")

    if ds_count and prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        if rag_mode == "Retrieve & Generate":
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating..."):
                    response = retrieve_and_generate(
                        runtime_client,
                        st.session_state.selected_kb_id,
                        model_arn,
                        prompt,
                        temperature=rag_temperature,
                    )
                if response:
                    answer = response.get("output", {}).get("text", "No response received.")
                    citations = response.get("citations", [])
                    st.markdown(answer)
                    refs = [ref for c in citations for ref in c.get("retrievedReferences", [])]
                    if refs:
                        with st.expander(f"📎 {len(refs)} citation(s)", expanded=False):
                            for i, ref in enumerate(refs, 1):
                                content = ref.get("content", {}).get("text", "")
                                uri = ref.get("location", {}).get("s3Location", {}).get("uri", "")
                                snippet = content[:200] + ("..." if len(content) > 200 else "")
                                st.markdown(f":blue[**{i}.** {uri}]")
                                st.markdown(f"<small>{snippet}</small>", unsafe_allow_html=True)
                else:
                    st.error("Failed to get a response. Check logs for details.")

        elif rag_mode == "Retrieve":
            with st.spinner("Retrieving..."):
                results = retrieve_only(runtime_client, st.session_state.selected_kb_id, prompt, kb_type=kb_type)
            if results is None:
                st.error("Retrieval failed. Check logs for details.")
            elif not results:
                st.info("No results found.")
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**{len(results)} result(s) retrieved:**")
                    for i, result in enumerate(results, 1):
                        content = result.get("content", {}).get("text", "")
                        score = result.get("score", 0)
                        uri = result.get("location", {}).get("s3Location", {}).get("uri", "")
                        score_color = "green" if score >= 0.8 else "orange" if score >= 0.5 else "red"
                        with st.expander(f":{score_color}[{i}. Score: {score:.3f}] &nbsp; {uri}", expanded=i == 1):
                            st.markdown(content)

        else:  # Agentic Retrieve
            agentic_messages = [{"role": "user", "content": {"text": prompt}}]

            with st.chat_message("assistant"):
                trace_area = st.empty()
                answer_area = st.empty()
                results_area = st.container()

                stream = agentic_retrieve_stream(
                    runtime_client,
                    st.session_state.selected_kb_id,
                    agentic_messages,
                    model_arn,
                    generate_response=True,
                )
                if stream is None:
                    st.error("Agentic retrieval failed. Check logs.")
                else:
                    full_answer = ""
                    all_results = []
                    try:
                        for event in stream:
                            if "responseEvent" in event:
                                full_answer += event["responseEvent"].get("text", "")
                                answer_area.markdown(full_answer + "▌")
                            elif "traceEvent" in event:
                                attrs = event["traceEvent"].get("attributes", {})
                                step = attrs.get("step", "")
                                status = attrs.get("status", "")
                                msg = attrs.get("message", "")
                                if step and status:
                                    trace_area.markdown(f":blue[*{step} — {status}* {msg}]")
                            elif "result" in event:
                                all_results = event["result"].get("results", [])
                                if not full_answer:
                                    gen = event["result"].get("generatedResponse", {})
                                    full_answer = gen.get("answer", "")
                    except Exception as e:
                        logger.error(f"Error reading agentic stream: {e}")
                        st.error(f"Stream error: {e}")

                    trace_area.empty()
                    if full_answer:
                        answer_area.markdown(full_answer)
                    if all_results:
                        with results_area:
                            with st.expander(f"📎 {len(all_results)} retrieved chunk(s)", expanded=False):
                                for i, result in enumerate(all_results, 1):
                                    content = result.get("content", {}).get("text", "")
                                    md = result.get("metadata", {})
                                    # Agentic/managed results carry the source in metadata, not location.
                                    source = (
                                        md.get("_document_title")
                                        or md.get("_source_uri")
                                        or result.get("location", {}).get("s3Location", {}).get("uri", "")
                                        or "source"
                                    )
                                    snippet = content[:200] + ("..." if len(content) > 200 else "")
                                    st.markdown(f":blue[**{i}.** {source}]")
                                    st.markdown(f"<small>{snippet}</small>", unsafe_allow_html=True)

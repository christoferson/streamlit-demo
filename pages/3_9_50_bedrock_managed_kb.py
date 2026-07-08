import streamlit as st
import boto3
import cmn_settings
import logging
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

if "selected_kb_id" not in st.session_state:
    st.session_state.selected_kb_id = None


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
            else:
                if st.button("Select This KB", use_container_width=True, type="primary"):
                    st.session_state.selected_kb_id = current_kb_id
                    logger.info(f"KB selected: {current_kb_id}")
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

    else:
        st.warning("No knowledge bases found in your account")

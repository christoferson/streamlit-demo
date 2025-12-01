import streamlit as st
import boto3
import json
from typing import List, Dict, Any
from datetime import datetime
import os
import cmn_settings

AWS_REGION = "us-east-1"
AWS_PROFILE = os.environ.get("AWS_PROFILE", "default")

KNOWLEDGE_BASE_ID = cmn_settings.CMN_KB_KNOWLEDGE_BASE_ID
DATA_SOURCE_ID = cmn_settings.CMN_KB_DATA_SOURCE_ID
DOCUMENT_BUCKET_NAME = cmn_settings.CMN_KB_DOCUMENT_BUCKET_NAME
VECTOR_BUCKET_NAME = cmn_settings.CMN_KB_VECTOR_BUCKET_NAME
VECTOR_INDEX_NAME = cmn_settings.CMN_KB_VECTOR_INDEX_NAME
KB_METADATA_TABLE = cmn_settings.CMN_KB_KB_METADATA_TABLE
DOC_METADATA_TABLE = cmn_settings.CMN_KB_DOC_METADATA_TABLE

# Embedding Model Configuration
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSION = 1024

# ============================================================================
# AWS CLIENTS
# ============================================================================

@st.cache_resource
def get_aws_clients():
    """Initialize and cache AWS clients with profile"""
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)

    bedrock_agent_client = session.client("bedrock-agent-runtime")
    bedrock_runtime_client = session.client("bedrock-runtime")
    s3_client = session.client("s3")
    dynamodb = session.resource("dynamodb")

    return bedrock_agent_client, bedrock_runtime_client, s3_client, dynamodb

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_logical_kb(dynamodb, user_id: str, kb_name: str, description: str = "") -> dict:
    """Create a logical Knowledge Base (metadata only)"""
    import uuid

    kb_id = f"kb-{uuid.uuid4().hex[:8]}"
    s3_prefix = f"{user_id}/{kb_id}/"

    kb_metadata = {
        'kb_id': kb_id,
        'kb_name': kb_name,
        'description': description,
        'user_id': user_id,
        's3_prefix': s3_prefix,
        'created_at': datetime.utcnow().isoformat(),
        'status': 'active',
        'document_count': 0
    }

    table = dynamodb.Table(KB_METADATA_TABLE)
    table.put_item(Item=kb_metadata)

    return kb_metadata

def list_user_kbs(dynamodb, user_id: str) -> List[dict]:
    """List all Knowledge Bases for a user"""
    table = dynamodb.Table(KB_METADATA_TABLE)

    response = table.query(
        IndexName='UserIdIndex',
        KeyConditionExpression='user_id = :uid',
        ExpressionAttributeValues={':uid': user_id}
    )

    return response.get('Items', [])

def upload_document_to_kb(s3_client, user_id: str, kb_id: str, file, metadata: dict = None) -> str:
    """Upload document to S3 with proper prefix structure"""
    s3_key = f"{user_id}/{kb_id}/{file.name}"

    extra_args = {
        'Metadata': {
            'user_id': user_id,
            'kb_id': kb_id,
            'uploaded_by': 'streamlit-app'
        }
    }

    if metadata:
        extra_args['Metadata'].update(metadata)

    s3_client.upload_fileobj(
        file,
        DOCUMENT_BUCKET_NAME,
        s3_key,
        ExtraArgs=extra_args
    )

    return s3_key

def query_knowledge_base(
    bedrock_agent_client,
    query: str,
    user_id: str,
    kb_id: str,
    top_k: int = 5
) -> dict:
    """Query Knowledge Base with metadata filtering"""

    response = bedrock_agent_client.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={'text': query},
        retrievalConfiguration={
            'vectorSearchConfiguration': {
                'numberOfResults': top_k,
                'filter': {
                    'andAll': [
                        {
                            'equals': {
                                'key': 'user_id',
                                'value': user_id
                            }
                        },
                        {
                            'equals': {
                                'key': 'kb_id',
                                'value': kb_id
                            }
                        }
                    ]
                }
            }
        }
    )

    return response

def retrieve_and_generate(
    bedrock_agent_client,
    query: str,
    user_id: str,
    kb_id: str,
    model_arn: str = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
) -> dict:
    """Retrieve and Generate answer using RAG"""

    response = bedrock_agent_client.retrieve_and_generate(
        input={'text': query},
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                'modelArn': model_arn,
                'retrievalConfiguration': {
                    'vectorSearchConfiguration': {
                        'numberOfResults': 5,
                        'filter': {
                            'andAll': [
                                {
                                    'equals': {
                                        'key': 'user_id',
                                        'value': user_id
                                    }
                                },
                                {
                                    'equals': {
                                        'key': 'kb_id',
                                        'value': kb_id
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    )

    return response

def list_kb_documents(dynamodb, kb_id: str) -> List[dict]:
    """List all documents in a Knowledge Base"""
    table = dynamodb.Table(DOC_METADATA_TABLE)

    response = table.query(
        IndexName='KbIdIndex',
        KeyConditionExpression='kb_id = :kid',
        ExpressionAttributeValues={':kid': kb_id}
    )

    return response.get('Items', [])

def get_ingestion_jobs(bedrock_agent_client, max_results: int = 10) -> List[dict]:
    """Get recent ingestion jobs"""
    try:
        response = bedrock_agent_client.list_ingestion_jobs(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            dataSourceId=DATA_SOURCE_ID,
            maxResults=max_results
        )
        return response.get('ingestionJobSummaries', [])
    except Exception as e:
        st.error(f"Error listing ingestion jobs: {e}")
        return []

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Multi-Tenant Knowledge Base",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Multi-Tenant Knowledge Base System")
    st.markdown("**Pattern 1: Metadata-Based Isolation**")

    # Initialize clients
    try:
        bedrock_agent_client, bedrock_runtime_client, s3_client, dynamodb = get_aws_clients()
        st.sidebar.success("✅ Connected to AWS")
    except Exception as e:
        st.error(f"❌ Failed to connect to AWS: {e}")
        st.stop()

    # ========================================================================
    # SIDEBAR - User Selection
    # ========================================================================
    with st.sidebar:
        st.header("👤 User Context")

        # User ID input
        user_id = st.text_input(
            "User ID",
            value="user-alice",
            help="Enter your user ID (e.g., user-alice, user-bob)"
        )

        if not user_id:
            st.warning("Please enter a User ID")
            st.stop()

        st.markdown("---")

        # Configuration Info
        with st.expander("⚙️ Configuration", expanded=False):
            st.code(f"""
Region: {AWS_REGION}
Profile: {AWS_PROFILE}
KB ID: {KNOWLEDGE_BASE_ID}
Data Source: {DATA_SOURCE_ID}
Bucket: {DOCUMENT_BUCKET_NAME}
Model: {EMBEDDING_MODEL_ID}
Dimension: {EMBEDDING_DIMENSION}
            """)

    # ========================================================================
    # MAIN TABS
    # ========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 My Knowledge Bases",
        "➕ Create KB",
        "📤 Upload Documents",
        "🔍 Query KB",
        "📊 System Status"
    ])

    # ========================================================================
    # TAB 1: MY KNOWLEDGE BASES
    # ========================================================================
    with tab1:
        st.header("📚 My Knowledge Bases")

        if st.button("🔄 Refresh List"):
            st.rerun()

        try:
            user_kbs = list_user_kbs(dynamodb, user_id)

            if user_kbs:
                st.success(f"Found {len(user_kbs)} Knowledge Base(s)")

                for kb in user_kbs:
                    with st.expander(f"📁 {kb['kb_name']} ({kb['kb_id']})", expanded=True):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("KB ID", kb['kb_id'])
                        with col2:
                            st.metric("Status", kb.get('status', 'unknown'))
                        with col3:
                            st.metric("Documents", kb.get('document_count', 0))

                        st.markdown(f"**Description:** {kb.get('description', 'N/A')}")
                        st.markdown(f"**S3 Prefix:** `{kb['s3_prefix']}`")
                        st.markdown(f"**Created:** {kb['created_at']}")

                        # List documents in this KB
                        st.markdown("---")
                        st.markdown("**📄 Documents:**")

                        docs = list_kb_documents(dynamodb, kb['kb_id'])
                        if docs:
                            for doc in docs:
                                st.text(f"• {doc['filename']} ({doc.get('ingestion_status', 'unknown')})")
                        else:
                            st.info("No documents yet")
            else:
                st.info(f"No Knowledge Bases found for user: {user_id}")
                st.markdown("👉 Go to **Create KB** tab to create your first Knowledge Base")

        except Exception as e:
            st.error(f"Error loading Knowledge Bases: {e}")

    # ========================================================================
    # TAB 2: CREATE KB
    # ========================================================================
    with tab2:
        st.header("➕ Create New Knowledge Base")

        st.info("💡 This creates a **logical** Knowledge Base (metadata only). No AWS resources are created.")

        with st.form("create_kb_form"):
            kb_name = st.text_input(
                "Knowledge Base Name",
                placeholder="e.g., Technical Documentation"
            )

            kb_description = st.text_area(
                "Description",
                placeholder="e.g., Collection of AWS service documentation"
            )

            submitted = st.form_submit_button("Create Knowledge Base", type="primary")

            if submitted:
                if not kb_name:
                    st.error("Please enter a Knowledge Base name")
                else:
                    try:
                        with st.spinner("Creating Knowledge Base..."):
                            kb_metadata = create_logical_kb(
                                dynamodb,
                                user_id,
                                kb_name,
                                kb_description
                            )

                        st.success("✅ Knowledge Base created successfully!")

                        st.json(kb_metadata)

                        st.info(f"📁 Upload documents to: `s3://{DOCUMENT_BUCKET_NAME}/{kb_metadata['s3_prefix']}`")

                    except Exception as e:
                        st.error(f"Error creating Knowledge Base: {e}")

    # ========================================================================
    # TAB 3: UPLOAD DOCUMENTS
    # ========================================================================
    with tab3:
        st.header("📤 Upload Documents")

        # Select KB
        try:
            user_kbs = list_user_kbs(dynamodb, user_id)

            if not user_kbs:
                st.warning("No Knowledge Bases found. Please create one first.")
            else:
                kb_options = {kb['kb_name']: kb['kb_id'] for kb in user_kbs}

                selected_kb_name = st.selectbox(
                    "Select Knowledge Base",
                    options=list(kb_options.keys())
                )

                selected_kb_id = kb_options[selected_kb_name]

                st.info(f"📁 Selected KB: **{selected_kb_name}** (`{selected_kb_id}`)")

                # File uploader
                uploaded_files = st.file_uploader(
                    "Choose files to upload",
                    accept_multiple_files=True,
                    type=['txt', 'pdf', 'docx', 'md', 'json']
                )

                if uploaded_files:
                    st.markdown(f"**{len(uploaded_files)} file(s) selected**")

                    for file in uploaded_files:
                        st.text(f"• {file.name} ({file.size} bytes)")

                    if st.button("📤 Upload Files", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        uploaded_count = 0

                        for i, file in enumerate(uploaded_files):
                            try:
                                status_text.text(f"Uploading {file.name}...")

                                s3_key = upload_document_to_kb(
                                    s3_client,
                                    user_id,
                                    selected_kb_id,
                                    file
                                )

                                uploaded_count += 1
                                progress_bar.progress((i + 1) / len(uploaded_files))

                            except Exception as e:
                                st.error(f"Error uploading {file.name}: {e}")

                        status_text.empty()
                        progress_bar.empty()

                        if uploaded_count > 0:
                            st.success(f"✅ Successfully uploaded {uploaded_count} file(s)!")
                            st.info("⏳ Ingestion will start automatically via EventBridge → Lambda")
                            st.markdown("Check the **System Status** tab to monitor ingestion progress")

        except Exception as e:
            st.error(f"Error: {e}")

    # ========================================================================
    # TAB 4: QUERY KB
    # ========================================================================
    with tab4:
        st.header("🔍 Query Knowledge Base")

        try:
            user_kbs = list_user_kbs(dynamodb, user_id)

            if not user_kbs:
                st.warning("No Knowledge Bases found. Please create one first.")
            else:
                kb_options = {kb['kb_name']: kb['kb_id'] for kb in user_kbs}

                selected_kb_name = st.selectbox(
                    "Select Knowledge Base",
                    options=list(kb_options.keys()),
                    key="query_kb_select"
                )

                selected_kb_id = kb_options[selected_kb_name]

                st.info(f"📁 Querying: **{selected_kb_name}** (`{selected_kb_id}`)")

                # Query input
                query_text = st.text_input(
                    "Enter your question",
                    placeholder="e.g., What is Amazon S3?"
                )

                col1, col2 = st.columns([1, 3])

                with col1:
                    query_mode = st.radio(
                        "Query Mode",
                        ["Retrieve Only", "Retrieve & Generate (RAG)"]
                    )

                with col2:
                    if query_mode == "Retrieve Only":
                        top_k = st.slider("Number of results", 1, 10, 5)

                if st.button("🔍 Search", type="primary"):
                    if not query_text:
                        st.warning("Please enter a question")
                    else:
                        try:
                            with st.spinner("Searching..."):
                                if query_mode == "Retrieve Only":
                                    # Retrieve only
                                    response = query_knowledge_base(
                                        bedrock_agent_client,
                                        query_text,
                                        user_id,
                                        selected_kb_id,
                                        top_k
                                    )

                                    results = response.get('retrievalResults', [])

                                    if results:
                                        st.success(f"✅ Found {len(results)} result(s)")

                                        for i, result in enumerate(results, 1):
                                            with st.expander(f"📄 Result {i} (Score: {result.get('score', 0):.4f})", expanded=(i==1)):
                                                st.markdown("**Content:**")
                                                st.info(result.get('content', {}).get('text', 'N/A'))

                                                st.markdown("**Metadata:**")
                                                metadata = result.get('metadata', {})
                                                st.json(metadata)

                                                st.markdown("**Location:**")
                                                location = result.get('location', {})
                                                if 's3Location' in location:
                                                    st.code(location['s3Location'].get('uri', 'N/A'))
                                    else:
                                        st.warning("No results found")

                                else:
                                    # Retrieve and Generate
                                    response = retrieve_and_generate(
                                        bedrock_agent_client,
                                        query_text,
                                        user_id,
                                        selected_kb_id
                                    )

                                    # Display answer
                                    st.markdown("### 💬 Answer")
                                    answer = response.get('output', {}).get('text', 'No answer generated')
                                    st.success(answer)

                                    # Display citations
                                    st.markdown("---")
                                    st.markdown("### 📚 Citations")

                                    citations = response.get('citations', [])
                                    if citations:
                                        for i, citation in enumerate(citations, 1):
                                            with st.expander(f"Citation {i}"):
                                                refs = citation.get('retrievedReferences', [])
                                                for ref in refs:
                                                    st.markdown("**Content:**")
                                                    st.info(ref.get('content', {}).get('text', 'N/A'))

                                                    st.markdown("**Source:**")
                                                    location = ref.get('location', {})
                                                    if 's3Location' in location:
                                                        st.code(location['s3Location'].get('uri', 'N/A'))
                                    else:
                                        st.info("No citations available")

                        except Exception as e:
                            st.error(f"Error querying Knowledge Base: {e}")
                            import traceback
                            st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"Error: {e}")

    # ========================================================================
    # TAB 5: SYSTEM STATUS
    # ========================================================================
    with tab5:
        st.header("📊 System Status")

        if st.button("🔄 Refresh Status"):
            st.rerun()

        # Ingestion Jobs
        st.subheader("📥 Recent Ingestion Jobs")

        try:
            # Note: bedrock-agent-runtime doesn't have list_ingestion_jobs
            # We need bedrock-agent client for this
            session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
            bedrock_agent = session.client("bedrock-agent")

            jobs = bedrock_agent.list_ingestion_jobs(
                knowledgeBaseId=KNOWLEDGE_BASE_ID,
                dataSourceId=DATA_SOURCE_ID,
                maxResults=10
            )

            job_summaries = jobs.get('ingestionJobSummaries', [])

            if job_summaries:
                for job in job_summaries:
                    status = job.get('status', 'UNKNOWN')

                    status_emoji = {
                        'COMPLETE': '✅',
                        'IN_PROGRESS': '⏳',
                        'FAILED': '❌',
                        'STARTING': '🔄'
                    }.get(status, '❓')

                    with st.expander(f"{status_emoji} Job {job.get('ingestionJobId', 'N/A')} - {status}"):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Status", status)
                        with col2:
                            st.metric("Started", job.get('startedAt', 'N/A'))
                        with col3:
                            stats = job.get('statistics', {})
                            st.metric("Documents", stats.get('numberOfDocumentsScanned', 0))

                        st.json(job)
            else:
                st.info("No ingestion jobs found")

        except Exception as e:
            st.error(f"Error loading ingestion jobs: {e}")

        st.markdown("---")

        # System Configuration
        st.subheader("⚙️ System Configuration")

        config_data = {
            "AWS Region": AWS_REGION,
            "AWS Profile": AWS_PROFILE,
            "Knowledge Base ID": KNOWLEDGE_BASE_ID,
            "Data Source ID": DATA_SOURCE_ID,
            "Document Bucket": DOCUMENT_BUCKET_NAME,
            "KB Metadata Table": KB_METADATA_TABLE,
            "Doc Metadata Table": DOC_METADATA_TABLE,
            "Embedding Model": EMBEDDING_MODEL_ID,
            "Embedding Dimension": EMBEDDING_DIMENSION
        }

        st.json(config_data)

if __name__ == "__main__":
    main()
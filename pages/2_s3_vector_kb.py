import streamlit as st
import boto3
import json
from typing import List, Dict, Any
from datetime import datetime
import os
import cmn_settings
import json

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
MODEL_ARN = cmn_settings.CMN_KB_MODEL_ARN

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
    bedrock_agent = session.client("bedrock-agent")

    return bedrock_agent_client, bedrock_runtime_client, s3_client, dynamodb, bedrock_agent

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

def upload_document_with_metadata_json(s3_client, user_id: str, kb_id: str, file, metadata: dict = None):
    """Upload document with accompanying metadata.json"""

    # Upload the document
    s3_key = f"{user_id}/{kb_id}/{file.name}"
    s3_client.upload_fileobj(
        file,
        DOCUMENT_BUCKET_NAME,
        s3_key
    )

    # Create metadata.json
    metadata_content = {
        "metadataAttributes": {
            "user_id": user_id,
            "kb_id": kb_id,
            "filename": file.name,
            "uploaded_at": datetime.utcnow().isoformat()
        }
    }

    # Upload metadata.json with same prefix
    metadata_key = f"{user_id}/{kb_id}/{file.name}.metadata.json"
    s3_client.put_object(
        Bucket=DOCUMENT_BUCKET_NAME,
        Key=metadata_key,
        Body=json.dumps(metadata_content),
        ContentType='application/json'
    )

    return s3_key

def upload_document_to_kb(s3_client, user_id: str, kb_id: str, file, metadata: dict = None) -> str:
    """Upload document to S3 with metadata.json"""

    # Read file content
    file_content = file.read()
    file.seek(0)  # Reset file pointer

    # Upload the document
    s3_key = f"{user_id}/{kb_id}/{file.name}"
    s3_client.put_object(
        Bucket=DOCUMENT_BUCKET_NAME,
        Key=s3_key,
        Body=file_content
    )

    # Create and upload metadata.json
    metadata_json = {
        "metadataAttributes": {
            "user_id": user_id,
            "kb_id": kb_id,
            "filename": file.name,
            "uploaded_at": datetime.utcnow().isoformat()
        }
    }

    if metadata:
        metadata_json["metadataAttributes"].update(metadata)

    # Upload metadata.json (same name with .metadata.json suffix)
    metadata_key = f"{s3_key}.metadata.json"
    s3_client.put_object(
        Bucket=DOCUMENT_BUCKET_NAME,
        Key=metadata_key,
        Body=json.dumps(metadata_json),
        ContentType='application/json'
    )

    st.info(f"📄 Uploaded document: {s3_key}")
    st.info(f"📋 Uploaded metadata: {metadata_key}")

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
    model_arn: str = MODEL_ARN
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

def get_ingestion_jobs(bedrock_agent, max_results: int = 10) -> List[dict]:
    """Get recent ingestion jobs"""
    try:
        response = bedrock_agent.list_ingestion_jobs(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            dataSourceId=DATA_SOURCE_ID,
            maxResults=max_results
        )
        return response.get('ingestionJobSummaries', [])
    except Exception as e:
        st.error(f"Error listing ingestion jobs: {e}")
        return []

def check_s3_documents(s3_client, user_id: str, kb_id: str) -> List[dict]:
    """Check what documents exist in S3 for this user/kb"""
    try:
        prefix = f"{user_id}/{kb_id}/"
        response = s3_client.list_objects_v2(
            Bucket=DOCUMENT_BUCKET_NAME,
            Prefix=prefix
        )

        objects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                # Get object metadata
                head = s3_client.head_object(
                    Bucket=DOCUMENT_BUCKET_NAME,
                    Key=obj['Key']
                )
                objects.append({
                    'Key': obj['Key'],
                    'Size': obj['Size'],
                    'LastModified': obj['LastModified'],
                    'Metadata': head.get('Metadata', {})
                })
        return objects
    except Exception as e:
        st.error(f"Error checking S3: {e}")
        return []

def test_retrieve_no_filter(bedrock_agent_client, query: str, top_k: int = 5) -> dict:
    """Test retrieve without any filters to see if there's any data"""
    try:
        response = bedrock_agent_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': top_k
                }
            }
        )
        return response
    except Exception as e:
        return {'error': str(e)}

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
        bedrock_agent_client, bedrock_runtime_client, s3_client, dynamodb, bedrock_agent = get_aws_clients()
        st.sidebar.success("✅ Connected to AWS")
    except Exception as e:
        st.error(f"❌ Failed to connect to AWS: {e}")
        st.stop()

    # ========================================================================
    # SIDEBAR - User Selection & Configuration
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

        # ===== NEW: Configuration Display =====
        with st.expander("📋 System Configuration", expanded=False):
            st.markdown("### Core Settings")
            st.code(f"""
Region: {AWS_REGION}
Profile: {AWS_PROFILE}
KB ID: {KNOWLEDGE_BASE_ID}
Data Source: {DATA_SOURCE_ID}
Full DS ID: {KNOWLEDGE_BASE_ID}|{DATA_SOURCE_ID}
            """)

            st.markdown("### Storage")
            st.code(f"""
Document Bucket: {DOCUMENT_BUCKET_NAME}
Vector Bucket: {VECTOR_BUCKET_NAME}
Vector Index: {VECTOR_INDEX_NAME}
            """)

            st.markdown("### DynamoDB Tables")
            st.code(f"""
KB Metadata: {KB_METADATA_TABLE}
Doc Metadata: {DOC_METADATA_TABLE}
            """)

            st.markdown("### Embedding")
            st.code(f"""
Model: {EMBEDDING_MODEL_ID}
Dimension: {EMBEDDING_DIMENSION}
            """)

        # ===== NEW: Connection Test =====
        with st.expander("🔍 Connection Test", expanded=False):
            if st.button("Test Connections"):
                with st.spinner("Testing..."):
                    results = {}

                    # Test Knowledge Base
                    try:
                        kb_info = bedrock_agent.get_knowledge_base(
                            knowledgeBaseId=KNOWLEDGE_BASE_ID
                        )
                        results["Knowledge Base"] = f"✅ {kb_info['knowledgeBase']['status']}"
                    except Exception as e:
                        results["Knowledge Base"] = f"❌ {str(e)[:50]}"

                    # Test Data Source
                    try:
                        ds_info = bedrock_agent.get_data_source(
                            knowledgeBaseId=KNOWLEDGE_BASE_ID,
                            dataSourceId=DATA_SOURCE_ID
                        )
                        results["Data Source"] = f"✅ {ds_info['dataSource']['status']}"
                    except Exception as e:
                        results["Data Source"] = f"❌ {str(e)[:50]}"

                    # Test S3 Bucket
                    try:
                        s3_client.head_bucket(Bucket=DOCUMENT_BUCKET_NAME)
                        results["S3 Bucket"] = "✅ Accessible"
                    except Exception as e:
                        results["S3 Bucket"] = f"❌ {str(e)[:50]}"

                    # Test DynamoDB Tables
                    try:
                        kb_table = dynamodb.Table(KB_METADATA_TABLE)
                        kb_table.table_status
                        results["KB Table"] = "✅ Accessible"
                    except Exception as e:
                        results["KB Table"] = f"❌ {str(e)[:50]}"

                    try:
                        doc_table = dynamodb.Table(DOC_METADATA_TABLE)
                        doc_table.table_status
                        results["Doc Table"] = "✅ Accessible"
                    except Exception as e:
                        results["Doc Table"] = f"❌ {str(e)[:50]}"

                    # Display results
                    for service, status in results.items():
                        st.text(f"{service}: {status}")

    # ========================================================================
    # MAIN TABS
    # ========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📚 My Knowledge Bases",
        "➕ Create KB",
        "📤 Upload Documents",
        "🔍 Query KB",
        "📊 System Status",
        "🐛 Debug Search"  # NEW DEBUG TAB
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

                    # Show debug info
                    show_debug = st.checkbox("Show debug info", value=True)  # Changed to True by default

                if st.button("🔍 Search", type="primary"):
                    if not query_text:
                        st.warning("Please enter a question")
                    else:
                        try:
                            with st.spinner("Searching..."):
                                if query_mode == "Retrieve Only":
                                    # Show debug info if requested
                                    if show_debug:
                                        with st.expander("🔍 Debug: Request Details", expanded=True):
                                            debug_info = {
                                                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                                                "query": query_text,
                                                "user_id": user_id,
                                                "kb_id": selected_kb_id,
                                                "filter": {
                                                    'andAll': [
                                                        {'equals': {'key': 'user_id', 'value': user_id}},
                                                        {'equals': {'key': 'kb_id', 'value': selected_kb_id}}
                                                    ]
                                                },
                                                "numberOfResults": top_k
                                            }
                                            st.json(debug_info)

                                    # Retrieve only
                                    response = query_knowledge_base(
                                        bedrock_agent_client,
                                        query_text,
                                        user_id,
                                        selected_kb_id,
                                        top_k
                                    )

                                    # DEBUG: Show full response
                                    if show_debug:
                                        with st.expander("🔍 Debug: Full API Response", expanded=True):
                                            st.json(response)

                                    results = response.get('retrievalResults', [])

                                    # DEBUG: Show what we got
                                    st.write(f"**Debug:** Response keys: {list(response.keys())}")
                                    st.write(f"**Debug:** Number of results: {len(results)}")
                                    st.write(f"**Debug:** Results type: {type(results)}")

                                    if results:
                                        st.success(f"✅ Found {len(results)} result(s)")

                                        for i, result in enumerate(results, 1):
                                            with st.expander(f"📄 Result {i} (Score: {result.get('score', 0):.4f})", expanded=(i==1)):
                                                st.markdown("**Content:**")
                                                content_text = result.get('content', {}).get('text', 'N/A')
                                                st.info(content_text)

                                                st.markdown("**Metadata:**")
                                                metadata = result.get('metadata', {})
                                                st.json(metadata)

                                                st.markdown("**Location:**")
                                                location = result.get('location', {})
                                                if 's3Location' in location:
                                                    st.code(location['s3Location'].get('uri', 'N/A'))

                                                # Show full result in debug mode
                                                if show_debug:
                                                    with st.expander("Full Result Object"):
                                                        st.json(result)
                                    else:
                                        st.warning("⚠️ No results found")

                                        # Show helpful debugging info
                                        st.markdown("### 🔍 Troubleshooting")
                                        st.markdown("No results were returned. This could mean:")
                                        st.markdown("""
                                        1. **Documents not ingested yet** - Check System Status tab
                                        2. **Metadata mismatch** - Documents may not have correct user_id/kb_id
                                        3. **Query doesn't match content** - Try a different query
                                        4. **Ingestion failed** - Check ingestion job status
                                        """)

                                        # Check S3 documents
                                        with st.expander("📁 Check S3 Documents", expanded=True):
                                            s3_docs = check_s3_documents(s3_client, user_id, selected_kb_id)
                                            if s3_docs:
                                                st.success(f"Found {len(s3_docs)} document(s) in S3:")
                                                for doc in s3_docs:
                                                    st.text(f"• {doc['Key']}")
                                                    st.json(doc['Metadata'])
                                            else:
                                                st.warning(f"No documents found in S3 at: {user_id}/{selected_kb_id}/")

                                        # Check ingestion status
                                        with st.expander("📥 Check Recent Ingestion Jobs", expanded=True):
                                            try:
                                                jobs = bedrock_agent.list_ingestion_jobs(
                                                    knowledgeBaseId=KNOWLEDGE_BASE_ID,
                                                    dataSourceId=DATA_SOURCE_ID,
                                                    maxResults=5
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

                                                        st.text(f"{status_emoji} {job.get('ingestionJobId', 'N/A')}: {status}")

                                                        if status == 'COMPLETE':
                                                            stats = job.get('statistics', {})
                                                            st.text(f"   Scanned: {stats.get('numberOfDocumentsScanned', 0)}, "
                                                                f"Modified: {stats.get('numberOfModifiedDocuments', 0)}, "
                                                                f"Deleted: {stats.get('numberOfDocumentsDeleted', 0)}")
                                                else:
                                                    st.warning("No ingestion jobs found")
                                            except Exception as e:
                                                st.error(f"Error checking ingestion: {e}")

                                        # Suggest going to debug tab
                                        st.info("💡 Go to the **Debug Search** tab for more detailed diagnostics")

                                else:
                                    # Retrieve and Generate
                                    response = retrieve_and_generate(
                                        bedrock_agent_client,
                                        query_text,
                                        user_id,
                                        selected_kb_id
                                    )

                                    # DEBUG: Show full response
                                    if show_debug:
                                        with st.expander("🔍 Debug: Full API Response", expanded=False):
                                            st.json(response)

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

                                                    st.markdown("**Metadata:**")
                                                    st.json(ref.get('metadata', {}))
                                    else:
                                        st.info("No citations available")

                        except Exception as e:
                            st.error(f"❌ Error querying Knowledge Base: {e}")
                            import traceback
                            with st.expander("Full Error Traceback"):
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
                            started = job.get('startedAt', 'N/A')
                            if started != 'N/A':
                                started = started.strftime('%Y-%m-%d %H:%M:%S')
                            st.metric("Started", started)
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
            "Full Data Source ID": f"{KNOWLEDGE_BASE_ID}|{DATA_SOURCE_ID}",
            "Document Bucket": DOCUMENT_BUCKET_NAME,
            "Vector Bucket": VECTOR_BUCKET_NAME,
            "Vector Index": VECTOR_INDEX_NAME,
            "KB Metadata Table": KB_METADATA_TABLE,
            "Doc Metadata Table": DOC_METADATA_TABLE,
            "Embedding Model": EMBEDDING_MODEL_ID,
            "Embedding Dimension": EMBEDDING_DIMENSION
        }

        st.json(config_data)

    # ========================================================================
    # TAB 6: DEBUG SEARCH (NEW)
    # ========================================================================
    with tab6:
        st.header("🐛 Debug Search Issues")

        st.markdown("""
        This tab helps diagnose why searches return no results.
        """)

        # Select KB for debugging
        try:
            user_kbs = list_user_kbs(dynamodb, user_id)

            if not user_kbs:
                st.warning("No Knowledge Bases found. Create one first.")
            else:
                kb_options = {kb['kb_name']: kb['kb_id'] for kb in user_kbs}

                debug_kb_name = st.selectbox(
                    "Select Knowledge Base to Debug",
                    options=list(kb_options.keys()),
                    key="debug_kb_select"
                )

                debug_kb_id = kb_options[debug_kb_name]

                st.info(f"Debugging KB: **{debug_kb_name}** (`{debug_kb_id}`)")

                # Step 1: Check S3 Documents
                st.subheader("1️⃣ Check S3 Documents")
                if st.button("Check S3", key="debug_check_s3"):
                    with st.spinner("Checking S3..."):
                        s3_docs = check_s3_documents(s3_client, user_id, debug_kb_id)
                        if s3_docs:
                            st.success(f"✅ Found {len(s3_docs)} document(s) in S3")
                            for doc in s3_docs:
                                with st.expander(f"📄 {doc['Key']}"):
                                    st.json(doc)
                        else:
                            st.error(f"❌ No documents found in S3 at: {user_id}/{debug_kb_id}/")
                            st.markdown("**Action:** Upload documents in the Upload tab")

                # Step 2: Check Ingestion Status
                st.subheader("2️⃣ Check Ingestion Status")
                if st.button("Check Ingestion", key="debug_check_ingestion"):
                    with st.spinner("Checking ingestion jobs..."):
                        try:
                            jobs = bedrock_agent.list_ingestion_jobs(
                                knowledgeBaseId=KNOWLEDGE_BASE_ID,
                                dataSourceId=DATA_SOURCE_ID,
                                maxResults=5
                            )

                            job_summaries = jobs.get('ingestionJobSummaries', [])
                            if job_summaries:
                                st.success(f"✅ Found {len(job_summaries)} ingestion job(s)")
                                for job in job_summaries:
                                    status = job.get('status', 'UNKNOWN')
                                    status_emoji = {
                                        'COMPLETE': '✅',
                                        'IN_PROGRESS': '⏳',
                                        'FAILED': '❌',
                                        'STARTING': '🔄'
                                    }.get(status, '❓')

                                    with st.expander(f"{status_emoji} {job.get('ingestionJobId', 'N/A')} - {status}"):
                                        st.json(job)

                                        if status == 'FAILED':
                                            st.error("This job failed. Check CloudWatch logs for details.")
                                        elif status == 'IN_PROGRESS':
                                            st.info("Job is still running. Wait for completion.")
                                        elif status == 'COMPLETE':
                                            stats = job.get('statistics', {})
                                            if stats.get('numberOfDocumentsScanned', 0) == 0:
                                                st.warning("Job completed but no documents were scanned!")
                            else:
                                st.warning("❌ No ingestion jobs found")
                                st.markdown("**Action:** Upload a document to trigger ingestion")
                        except Exception as e:
                            st.error(f"Error: {e}")

                # Step 3: Test Retrieve WITHOUT Filter
                st.subheader("3️⃣ Test Retrieve (No Filter)")
                st.markdown("Test if ANY data exists in the knowledge base (ignoring metadata filters)")

                test_query = st.text_input("Test Query", value="Amazon S3", key="debug_test_query")

                if st.button("Test Retrieve (No Filter)", key="debug_test_retrieve"):
                    with st.spinner("Testing..."):
                        response = test_retrieve_no_filter(bedrock_agent_client, test_query, 10)

                        # Show full response
                        with st.expander("Full API Response", expanded=True):
                            st.json(response)

                        if 'error' in response:
                            st.error(f"❌ Error: {response['error']}")
                        else:
                            results = response.get('retrievalResults', [])
                            st.write(f"**Response keys:** {list(response.keys())}")
                            st.write(f"**Number of results:** {len(results)}")

                            if results:
                                st.success(f"✅ Found {len(results)} result(s) WITHOUT filter")
                                st.markdown("**This means:**")
                                st.markdown("- ✅ Knowledge base has data")
                                st.markdown("- ❌ But metadata filters are blocking results")
                                st.markdown("- 🔍 Check that documents have correct user_id and kb_id metadata")

                                for i, result in enumerate(results, 1):
                                    with st.expander(f"Result {i} - Score: {result.get('score', 0):.4f}"):
                                        st.markdown("**Content Preview:**")
                                        content = result.get('content', {}).get('text', 'N/A')
                                        st.info(content[:500] + "..." if len(content) > 500 else content)

                                        st.markdown("**Metadata:**")
                                        metadata = result.get('metadata', {})
                                        st.json(metadata)

                                        st.markdown("**Location:**")
                                        location = result.get('location', {})
                                        st.json(location)
                            else:
                                st.error("❌ No results found even WITHOUT filter")
                                st.markdown("**This means:**")
                                st.markdown("- ❌ Knowledge base is empty OR")
                                st.markdown("- ❌ Ingestion hasn't completed successfully")
                                st.markdown("- 🔍 Check ingestion job status above")

                # Step 4: Test Retrieve WITH Filter
                st.subheader("4️⃣ Test Retrieve (With Filter)")
                st.markdown("Test with metadata filters (normal search)")

                if st.button("Test Retrieve (With Filter)", key="debug_test_retrieve_filter"):
                    with st.spinner("Testing..."):
                        try:
                            response = query_knowledge_base(
                                bedrock_agent_client,
                                test_query,
                                user_id,
                                debug_kb_id,
                                10
                            )

                            # Show full response
                            with st.expander("Full API Response", expanded=True):
                                st.json(response)

                            results = response.get('retrievalResults', [])
                            st.write(f"**Response keys:** {list(response.keys())}")
                            st.write(f"**Number of results:** {len(results)}")

                            if results:
                                st.success(f"✅ Found {len(results)} result(s) WITH filter")
                                st.markdown("**Search is working correctly!**")

                                for i, result in enumerate(results, 1):
                                    with st.expander(f"Result {i} - Score: {result.get('score', 0):.4f}"):
                                        st.markdown("**Content Preview:**")
                                        content = result.get('content', {}).get('text', 'N/A')
                                        st.info(content[:500] + "..." if len(content) > 500 else content)

                                        st.markdown("**Metadata:**")
                                        metadata = result.get('metadata', {})
                                        st.json(metadata)

                                        # Check if metadata matches what we're filtering for
                                        if metadata:
                                            meta_user_id = metadata.get('user_id')
                                            meta_kb_id = metadata.get('kb_id')

                                            if meta_user_id == user_id and meta_kb_id == debug_kb_id:
                                                st.success(f"✅ Metadata matches: user_id={meta_user_id}, kb_id={meta_kb_id}")
                                            else:
                                                st.error(f"❌ Metadata mismatch!")
                                                st.error(f"Expected: user_id={user_id}, kb_id={debug_kb_id}")
                                                st.error(f"Got: user_id={meta_user_id}, kb_id={meta_kb_id}")

                                        st.markdown("**Location:**")
                                        location = result.get('location', {})
                                        st.json(location)
                            else:
                                st.error("❌ No results found WITH filter")
                                st.markdown("**Possible causes:**")
                                st.markdown("1. Documents don't have correct metadata (user_id, kb_id)")
                                st.markdown("2. Metadata values don't match exactly")
                                st.markdown(f"3. Expected: user_id=`{user_id}`, kb_id=`{debug_kb_id}`")

                                # Show what we're filtering for
                                with st.expander("Filter Details"):
                                    st.json({
                                        'andAll': [
                                            {'equals': {'key': 'user_id', 'value': user_id}},
                                            {'equals': {'key': 'kb_id', 'value': debug_kb_id}}
                                        ]
                                    })

                                st.markdown("---")
                                st.markdown("**Next Steps:**")
                                st.markdown("1. Run 'Test Retrieve (No Filter)' above to see if ANY data exists")
                                st.markdown("2. If data exists without filter, the problem is metadata")
                                st.markdown("3. Check the metadata on documents that DO return (without filter)")
                                st.markdown("4. Ensure ingestion job completed successfully")
                        except Exception as e:
                            st.error(f"Error: {e}")
                            import traceback
                            st.code(traceback.format_exc())

                # Step 5: Check DynamoDB Metadata
                st.subheader("5️⃣ Check DynamoDB Metadata")
                if st.button("Check DynamoDB", key="debug_check_dynamodb"):
                    with st.spinner("Checking DynamoDB..."):
                        # Check KB metadata
                        try:
                            kb_table = dynamodb.Table(KB_METADATA_TABLE)
                            kb_response = kb_table.get_item(Key={'kb_id': debug_kb_id})

                            if 'Item' in kb_response:
                                st.success("✅ KB metadata found")
                                with st.expander("KB Metadata"):
                                    st.json(kb_response['Item'])
                            else:
                                st.warning("❌ KB metadata not found")
                        except Exception as e:
                            st.error(f"Error checking KB metadata: {e}")

                        # Check document metadata
                        try:
                            docs = list_kb_documents(dynamodb, debug_kb_id)
                            if docs:
                                st.success(f"✅ Found {len(docs)} document(s) in metadata table")
                                for doc in docs:
                                    with st.expander(f"📄 {doc.get('filename', 'Unknown')}"):
                                        st.json(doc)
                            else:
                                st.warning("❌ No document metadata found")
                        except Exception as e:
                            st.error(f"Error checking document metadata: {e}")

                # Summary
                st.markdown("---")
                st.subheader("📋 Diagnostic Summary")
                st.markdown("""
                **Run all checks above in order:**

                1. ✅ S3 has documents → Documents uploaded
                2. ✅ Ingestion job COMPLETE → Documents processed
                3. ✅ Retrieve without filter works → Data in knowledge base
                4. ✅ Retrieve with filter works → Metadata correct
                5. ✅ DynamoDB has metadata → Tracking working

                **If any step fails, that's where the problem is!**
                """)

        except Exception as e:
            st.error(f"Error in debug tab: {e}")

if __name__ == "__main__":
    main()
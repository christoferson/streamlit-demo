import streamlit as st
import boto3
import json
from typing import List, Dict, Any
import time
from datetime import datetime
import cmn_settings

# ============================================================================
# CONFIGURATION
# ============================================================================

AWS_REGION = cmn_settings.VECTOR_REGION_ID

# Your Vector Bucket and Index Names
VECTOR_BUCKET_NAME = cmn_settings.VECTOR_BUCKET_NAME
VECTOR_INDEX_NAME = cmn_settings.VECTOR_INDEX_NAME

# ============================================================================
# EMBEDDING MODELS CONFIGURATION
# ============================================================================

EMBEDDING_MODELS = {
    "Amazon Nova Multimodal": {
        "model_id": "amazon.nova-2-multimodal-embeddings-v1:0",
        "dimensions": [256, 384, 1024, 3072],
        "default_dimension": 1024,
        "description": "Latest multimodal model supporting text, images, video, and audio",
        "supports_embedding_purpose": True,
        "embedding_purposes": {
            "index": "GENERIC_INDEX",
            "query": {
                "document": "DOCUMENT_RETRIEVAL",
                "text": "TEXT_RETRIEVAL",
                "image": "IMAGE_RETRIEVAL",
                "video": "VIDEO_RETRIEVAL",
                "audio": "AUDIO_RETRIEVAL",
                "generic": "GENERIC_RETRIEVAL"
            }
        }
    },
    "Amazon Titan Text V2": {
        "model_id": "amazon.titan-embed-text-v2:0",
        "dimensions": [256, 512, 1024],
        "default_dimension": 1024,
        "description": "Latest Titan model with configurable dimensions",
        "supports_embedding_purpose": False
    },
    "Amazon Titan Text V1": {
        "model_id": "amazon.titan-embed-text-v1",
        "dimensions": [1536],
        "default_dimension": 1536,
        "description": "First generation Titan text embeddings",
        "supports_embedding_purpose": False
    },
    "Amazon Titan Multimodal V1": {
        "model_id": "amazon.titan-embed-image-v1",
        "dimensions": [1024, 384],
        "default_dimension": 1024,
        "description": "Multimodal embeddings (1024 for images, 384 for text)",
        "supports_embedding_purpose": False
    },
    "Cohere Embed English V3": {
        "model_id": "cohere.embed-english-v3",
        "dimensions": [1024],
        "default_dimension": 1024,
        "description": "Optimized for English text",
        "supports_embedding_purpose": False
    },
    "Cohere Embed Multilingual V3": {
        "model_id": "cohere.embed-multilingual-v3",
        "dimensions": [1024],
        "default_dimension": 1024,
        "description": "Supports 100+ languages",
        "supports_embedding_purpose": False
    }
}

# ============================================================================
# METADATA CONFIGURATION
# ============================================================================

NON_FILTERABLE_KEYS = [
    "document_title",
    "source_url",
    "created_timestamp"
]

# ============================================================================
# AWS CLIENTS
# ============================================================================

@st.cache_resource
def get_aws_clients():
    """Initialize and cache AWS clients"""
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    s3vectors_client = boto3.client("s3vectors", region_name=AWS_REGION)
    return bedrock_client, s3vectors_client

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_name_from_id(model_id: str) -> str:
    """Get simplified model name from model ID for tracking"""
    for name, config in EMBEDDING_MODELS.items():
        if config["model_id"] == model_id:
            return name
    return model_id

def generate_embedding(
    text: str, 
    bedrock_client, 
    model_id: str, 
    dimension: int = None,
    embedding_purpose: str = None,
    is_query: bool = False
) -> List[float]:
    """
    Generate embedding vector using Amazon Bedrock

    Args:
        text: Input text to embed
        bedrock_client: Boto3 Bedrock client
        model_id: Bedrock model ID
        dimension: Optional dimension
        embedding_purpose: For Nova models (GENERIC_INDEX, DOCUMENT_RETRIEVAL, etc.)
        is_query: Whether this is a query (vs document indexing)

    Returns:
        List of floats representing the embedding vector
    """
    try:
        # Nova Multimodal Embeddings
        if "nova" in model_id.lower():
            # Determine embedding purpose
            if embedding_purpose is None:
                embedding_purpose = "GENERIC_INDEX" if not is_query else "DOCUMENT_RETRIEVAL"

            body = {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": embedding_purpose,
                    "embeddingDimension": dimension or 1024,
                    "text": {
                        "truncationMode": "END",
                        "value": text
                    }
                }
            }

            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json"
            )

            response_body = json.loads(response["body"].read())
            return response_body["embeddings"][0]["embedding"]

        # Titan V2
        elif "titan-embed-text-v2" in model_id:
            body = {"inputText": text}
            if dimension:
                body["dimensions"] = dimension

        # Cohere models
        elif "cohere" in model_id:
            body = {
                "texts": [text],
                "input_type": "search_document" if not is_query else "search_query"
            }

        # Titan V1 and other models
        else:
            body = {"inputText": text}

        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )
        response_body = json.loads(response["body"].read())

        # Extract embedding based on model response format
        if "cohere" in model_id:
            return response_body["embeddings"][0]
        else:
            return response_body["embedding"]

    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def add_documents_to_vector_store(
    texts: List[str], 
    bedrock_client, 
    s3vectors_client,
    model_id: str,
    dimension: int = None,
    metadata_list: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Add multiple documents to the vector store"""
    vectors_to_put = []
    timestamp = datetime.now().isoformat()

    for i, text in enumerate(texts):
        # For Nova: Always use GENERIC_INDEX for indexing
        embedding = generate_embedding(
            text, 
            bedrock_client, 
            model_id, 
            dimension,
            embedding_purpose="GENERIC_INDEX" if "nova" in model_id.lower() else None,
            is_query=False
        )

        if embedding is None:
            continue

        # Base metadata
        metadata = {
            "text": text,
            "document_index": str(i),
            "created_timestamp": timestamp,
            "embedding_model": model_id,
            "embedding_model_name": get_model_name_from_id(model_id)
        }

        # Add custom metadata from user
        if metadata_list and i < len(metadata_list):
            metadata.update(metadata_list[i])

        vectors_to_put.append({
            "key": f"doc_{int(time.time())}_{i}",
            "data": {"float32": embedding},
            "metadata": metadata
        })

    try:
        result = s3vectors_client.put_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=VECTOR_INDEX_NAME,
            vectors=vectors_to_put
        )
        return result
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return None

def query_vector_store(
    question: str,
    bedrock_client,
    s3vectors_client,
    model_id: str,
    dimension: int = None,
    embedding_purpose: str = None,
    top_k: int = 5,
    metadata_filter: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """Query the vector store for similar documents"""

    # For Nova: Use specified purpose, default to DOCUMENT_RETRIEVAL
    if "nova" in model_id.lower() and embedding_purpose is None:
        embedding_purpose = "DOCUMENT_RETRIEVAL"

    embedding = generate_embedding(
        question, 
        bedrock_client, 
        model_id, 
        dimension,
        embedding_purpose=embedding_purpose,
        is_query=True
    )

    if embedding is None:
        return []

    try:
        query_params = {
            "vectorBucketName": VECTOR_BUCKET_NAME,
            "indexName": VECTOR_INDEX_NAME,
            "queryVector": {"float32": embedding},
            "topK": top_k,
            "returnMetadata": True,
            "returnDistance": True,
        }

        # Only add filter if it has values
        if metadata_filter and len(metadata_filter) > 0:
            query_params["filter"] = metadata_filter

        response = s3vectors_client.query_vectors(**query_params)

        if 'vectors' not in response or len(response['vectors']) == 0:
            return []

        results = []
        for vector in response["vectors"]:
            result = {
                "key": vector.get('key', 'unknown'),
                "distance": vector.get('distance', 0),
                "metadata": vector.get('metadata', {}),
                "text": vector.get('metadata', {}).get('text', '')
            }
            results.append(result)

        return results

    except Exception as e:
        st.error(f"Error querying vectors: {str(e)}")
        return []

def list_indexes(s3vectors_client, prefix: str = None) -> List[Dict[str, Any]]:
    """List all indexes in the vector bucket"""
    try:
        indexes = []
        next_token = None

        while True:
            params = {
                "vectorBucketName": VECTOR_BUCKET_NAME,
                "maxResults": 100
            }

            if prefix:
                params["prefix"] = prefix

            if next_token:
                params["nextToken"] = next_token

            response = s3vectors_client.list_indexes(**params)

            if 'indexes' in response:
                indexes.extend(response['indexes'])

            next_token = response.get('nextToken')
            if not next_token:
                break

        return indexes
    except Exception as e:
        st.error(f"Error listing indexes: {str(e)}")
        return []

def list_all_vectors(s3vectors_client, max_results: int = None) -> List[Dict[str, Any]]:
    """List all vectors in the index with their metadata"""
    try:
        vectors = []
        next_token = None

        while True:
            params = {
                "vectorBucketName": VECTOR_BUCKET_NAME,
                "indexName": VECTOR_INDEX_NAME,
                "maxResults": 100,
                "returnMetadata": True
            }

            if next_token:
                params["nextToken"] = next_token

            response = s3vectors_client.list_vectors(**params)

            if 'vectors' in response:
                vectors.extend(response['vectors'])

            if max_results and len(vectors) >= max_results:
                vectors = vectors[:max_results]
                break

            next_token = response.get('nextToken')
            if not next_token:
                break

        return vectors
    except Exception as e:
        st.error(f"Error listing vectors: {str(e)}")
        return []

def get_vector_keys(s3vectors_client) -> List[str]:
    """Get all vector keys (for deletion)"""
    try:
        vector_keys = []
        next_token = None

        while True:
            params = {
                "vectorBucketName": VECTOR_BUCKET_NAME,
                "indexName": VECTOR_INDEX_NAME,
                "maxResults": 100
            }

            if next_token:
                params["nextToken"] = next_token

            response = s3vectors_client.list_vectors(**params)

            if 'vectors' in response:
                vector_keys.extend([v['key'] for v in response['vectors']])

            next_token = response.get('nextToken')
            if not next_token:
                break

        return vector_keys
    except Exception as e:
        st.error(f"Error getting vector keys: {str(e)}")
        return []

def delete_vectors(s3vectors_client, vector_keys: List[str]) -> Dict[str, Any]:
    """Delete vectors by their keys"""
    try:
        result = s3vectors_client.delete_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=VECTOR_INDEX_NAME,
            keys=vector_keys
        )
        return result
    except Exception as e:
        st.error(f"Error deleting vectors: {str(e)}")
        return None

def reset_index(s3vectors_client) -> tuple[bool, int]:
    """Delete all vectors from the index"""
    try:
        vector_keys = get_vector_keys(s3vectors_client)

        if not vector_keys:
            return True, 0

        batch_size = 100
        total_deleted = 0

        for i in range(0, len(vector_keys), batch_size):
            batch = vector_keys[i:i + batch_size]
            result = delete_vectors(s3vectors_client, batch)
            if result:
                total_deleted += len(batch)

        return True, total_deleted
    except Exception as e:
        st.error(f"Error resetting index: {str(e)}")
        return False, 0

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="S3 Vector Store Manager",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 S3 Vector Store Manager")
    st.markdown("Manage and query your S3 Vector Bucket with Amazon Bedrock embeddings")

    bedrock_client, s3vectors_client = get_aws_clients()

    # ========================================================================
    # EMBEDDING MODEL SELECTOR (SIDEBAR)
    # ========================================================================
    with st.sidebar:
        st.header("🤖 Embedding Model")

        # Model selection
        selected_model_name = st.selectbox(
            "Select Embedding Model",
            options=list(EMBEDDING_MODELS.keys()),
            index=0,  # Default to Nova
            help="Choose the embedding model to use for generating vectors"
        )

        model_config = EMBEDDING_MODELS[selected_model_name]
        model_id = model_config["model_id"]

        # Show model info
        st.info(f"**Model:** {selected_model_name}\n\n{model_config['description']}")

        # Dimension selection
        dimension = None
        if len(model_config["dimensions"]) > 1:
            dimension = st.selectbox(
                "Select Dimension",
                options=model_config["dimensions"],
                index=model_config["dimensions"].index(model_config["default_dimension"]),
                help="Higher dimensions = better accuracy but more storage"
            )
        else:
            dimension = model_config["dimensions"][0]
            st.text(f"Dimension: {dimension}")

        # Nova-specific: Embedding Purpose for queries
        embedding_purpose_query = None
        if model_config.get("supports_embedding_purpose", False):
            st.markdown("---")
            st.markdown("**🎯 Query Purpose (Nova only)**")

            purpose_options = {
                "Document Retrieval": "DOCUMENT_RETRIEVAL",
                "Text Retrieval": "TEXT_RETRIEVAL",
                "Generic Retrieval": "GENERIC_RETRIEVAL",
                "Image Retrieval": "IMAGE_RETRIEVAL",
                "Video Retrieval": "VIDEO_RETRIEVAL",
                "Audio Retrieval": "AUDIO_RETRIEVAL"
            }

            selected_purpose = st.selectbox(
                "Query Type",
                options=list(purpose_options.keys()),
                index=0,  # Default to Document Retrieval
                help="Optimize query embeddings for specific retrieval tasks"
            )

            embedding_purpose_query = purpose_options[selected_purpose]

            st.caption(f"📝 Index: GENERIC_INDEX\n🔍 Query: {embedding_purpose_query}")

        st.markdown("---")
        st.markdown(f"**Model ID:**")
        st.code(model_id, language=None)
        st.markdown(f"**Dimension:** {dimension}")

        if embedding_purpose_query:
            st.markdown(f"**Query Purpose:** {embedding_purpose_query}")

    with st.expander("⚙️ Configuration & Metadata Info", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Configuration:**")
            config_text = f"""
Embedding Model: {selected_model_name}
Model ID: {model_id}
Dimension: {dimension}
AWS Region: {AWS_REGION}
Vector Store: Connected ✓
            """
            if embedding_purpose_query:
                config_text += f"\nQuery Purpose: {embedding_purpose_query}"
            st.code(config_text)

        with col2:
            st.markdown("**❌ Non-Filterable Keys:**")
            st.caption("Cannot be used in query filters")
            for key in NON_FILTERABLE_KEYS:
                st.text(f"• {key}")

            st.markdown("**✅ Filterable Keys:**")
            st.caption("Can be used in query filters")
            st.text("• document_type")
            st.text("• category")
            st.text("• aws_service")
            st.text("• embedding_model")
            st.text("• embedding_model_name")
            st.text("• (any other metadata)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Add Documents", 
        "🔎 Query Documents", 
        "📋 Manage Bucket",
        "🗑️ Manage Index"
    ])

    # ========================================================================
    # TAB 1: ADD DOCUMENTS
    # ========================================================================
    with tab1:
        st.header("Add Documents to Vector Store")

        info_text = f"🤖 Using: **{selected_model_name}** (Dimension: {dimension})"
        if model_config.get("supports_embedding_purpose", False):
            info_text += "\n\n📝 Documents will be indexed with **GENERIC_INDEX**"
        st.info(info_text)

        input_method = st.radio(
            "Choose input method:",
            ["Single Document", "Multiple Documents (Bulk)", "Sample AWS Services Data"]
        )

        if input_method == "Single Document":
            st.subheader("Add Single Document")

            col1, col2 = st.columns([3, 1])

            with col1:
                document_text = st.text_area(
                    "Document Text",
                    height=200,
                    placeholder="Enter your document text here..."
                )

            with col2:
                st.markdown("**✅ Filterable Metadata**")
                st.caption("Can be used in queries")
                doc_type = st.text_input("Document Type", placeholder="e.g., article")
                category = st.text_input("Category", placeholder="e.g., technical")
                service = st.text_input("AWS Service", placeholder="e.g., S3")

                st.markdown("**❌ Non-Filterable Metadata**")
                st.caption("Display only, cannot filter")
                doc_title = st.text_input("Document Title", placeholder="e.g., S3 Guide")
                source_url = st.text_input("Source URL", placeholder="e.g., https://...")

            if st.button("➕ Add Document", type="primary"):
                if document_text.strip():
                    with st.spinner("Adding document to vector store..."):
                        metadata = {}

                        # ✅ FILTERABLE metadata
                        if doc_type:
                            metadata["document_type"] = doc_type
                        if category:
                            metadata["category"] = category
                        if service:
                            metadata["aws_service"] = service

                        # ❌ NON-FILTERABLE metadata
                        if doc_title:
                            metadata["document_title"] = doc_title
                        if source_url:
                            metadata["source_url"] = source_url

                        result = add_documents_to_vector_store(
                            [document_text],
                            bedrock_client,
                            s3vectors_client,
                            model_id,
                            dimension,
                            [metadata] if metadata else None
                        )

                        if result:
                            st.success("✅ Document added successfully!")
                            with st.expander("View API Response"):
                                st.json(result)
                else:
                    st.warning("⚠️ Please enter document text")

        elif input_method == "Multiple Documents (Bulk)":
            st.subheader("Add Multiple Documents")
            st.info("💡 Enter one document per line")

            bulk_text = st.text_area(
                "Documents (one per line)",
                height=300,
                placeholder="Document 1\nDocument 2\nDocument 3\n..."
            )

            st.markdown("**✅ Filterable Metadata (applies to all)**")
            col1, col2 = st.columns(2)
            with col1:
                bulk_doc_type = st.text_input("Document Type", placeholder="e.g., article")
            with col2:
                bulk_category = st.text_input("Category", placeholder="e.g., technical")

            if st.button("➕ Add All Documents", type="primary"):
                if bulk_text.strip():
                    documents = [doc.strip() for doc in bulk_text.split('\n') if doc.strip()]

                    if documents:
                        with st.spinner(f"Adding {len(documents)} documents to vector store..."):
                            metadata_list = []
                            for i in range(len(documents)):
                                metadata = {}
                                if bulk_doc_type:
                                    metadata["document_type"] = bulk_doc_type
                                if bulk_category:
                                    metadata["category"] = bulk_category
                                metadata_list.append(metadata)

                            result = add_documents_to_vector_store(
                                documents,
                                bedrock_client,
                                s3vectors_client,
                                model_id,
                                dimension,
                                metadata_list
                            )

                            if result:
                                st.success(f"✅ Successfully added {len(documents)} documents!")
                                with st.expander("View API Response"):
                                    st.json(result)
                    else:
                        st.warning("⚠️ No valid documents found")
                else:
                    st.warning("⚠️ Please enter documents")

        else:  # Sample AWS Services Data
            st.subheader("Add Sample AWS Services Data")
            st.info("📚 This will add sample AWS service descriptions")

            sample_data = [
                {
                    "text": "Amazon S3 is an object storage service offering industry-leading scalability, data availability, security, and performance.",
                    "service": "S3",
                    "category": "Storage"
                },
                {
                    "text": "Amazon EC2 provides secure and resizable compute capacity in the cloud, allowing you to launch virtual servers as needed.",
                    "service": "EC2",
                    "category": "Compute"
                },
                {
                    "text": "AWS Lambda lets you run code without provisioning or managing servers, paying only for the compute time you consume.",
                    "service": "Lambda",
                    "category": "Compute"
                },
                {
                    "text": "Amazon RDS makes it easy to set up, operate, and scale a relational database in the cloud with automated backups.",
                    "service": "RDS",
                    "category": "Database"
                },
                {
                    "text": "Amazon DynamoDB is a fully managed NoSQL database service that provides fast and predictable performance with seamless scalability.",
                    "service": "DynamoDB",
                    "category": "Database"
                },
                {
                    "text": "Amazon VPC lets you provision a logically isolated section of the AWS Cloud where you can launch AWS resources.",
                    "service": "VPC",
                    "category": "Networking"
                },
                {
                    "text": "Amazon CloudFront is a fast content delivery network service that securely delivers data, videos, and applications globally.",
                    "service": "CloudFront",
                    "category": "Networking"
                },
                {
                    "text": "AWS IAM enables you to manage access to AWS services and resources securely with fine-grained permissions.",
                    "service": "IAM",
                    "category": "Security"
                },
                {
                    "text": "Amazon SQS is a fully managed message queuing service that enables you to decouple and scale microservices.",
                    "service": "SQS",
                    "category": "Application Integration"
                },
                {
                    "text": "Amazon SNS is a fully managed messaging service for both application-to-application and application-to-person communication.",
                    "service": "SNS",
                    "category": "Application Integration"
                },
            ]

            with st.expander("📄 Preview Sample Documents", expanded=False):
                for i, item in enumerate(sample_data, 1):
                    st.markdown(f"**{i}. {item['service']}** ({item['category']})")
                    st.text(item['text'])
                    st.divider()

            st.markdown(f"**Total documents:** {len(sample_data)}")

            if st.button("➕ Add Sample Data", type="primary"):
                with st.spinner(f"Adding {len(sample_data)} sample documents..."):
                    texts = [item['text'] for item in sample_data]
                    metadata_list = [
                        {
                            # ✅ FILTERABLE
                            "document_type": "service_description",
                            "category": item['category'],
                            "aws_service": item['service'],
                            # ❌ NON-FILTERABLE
                            "document_title": f"{item['service']} Description"
                        }
                        for item in sample_data
                    ]

                    result = add_documents_to_vector_store(
                        texts,
                        bedrock_client,
                        s3vectors_client,
                        model_id,
                        dimension,
                        metadata_list
                    )

                    if result:
                        st.success(f"✅ Successfully added {len(sample_data)} sample documents!")
                        with st.expander("View API Response"):
                            st.json(result)

    # ========================================================================
    # TAB 2: QUERY DOCUMENTS
    # ========================================================================
    with tab2:
        st.header("Query Vector Store")

        info_text = f"🤖 Using: **{selected_model_name}** (Dimension: {dimension})"
        if embedding_purpose_query:
            info_text += f"\n\n🔍 Query Purpose: **{embedding_purpose_query}**"
        st.info(info_text)

        col1, col2 = st.columns([3, 1])

        with col1:
            query_text = st.text_input(
                "Enter your query",
                placeholder="e.g., What is a serverless compute service?"
            )

        with col2:
            top_k = st.number_input(
                "Number of results",
                min_value=1,
                max_value=20,
                value=5
            )

        with st.expander("🔧 Metadata Filters (Optional)", expanded=False):
            st.markdown("**✅ Filter by Filterable Metadata**")
            st.caption("Only keys not in NonFilterableMetadataKeys can be used")

            col1, col2, col3 = st.columns(3)

            with col1:
                filter_doc_type = st.text_input(
                    "Document Type", 
                    key="filter_doc_type",
                    placeholder="e.g., service_description"
                )
            with col2:
                filter_category = st.text_input(
                    "Category", 
                    key="filter_category",
                    placeholder="e.g., Compute"
                )
            with col3:
                filter_service = st.text_input(
                    "AWS Service", 
                    key="filter_service",
                    placeholder="e.g., Lambda"
                )

            st.warning("⚠️ Cannot filter on: document_title, source_url, created_timestamp")

        if st.button("🔎 Search", type="primary"):
            if query_text.strip():
                with st.spinner("Searching vector store..."):
                    # Build metadata filter (only filterable keys)
                    metadata_filter = {}
                    if filter_doc_type:
                        metadata_filter["document_type"] = filter_doc_type
                    if filter_category:
                        metadata_filter["category"] = filter_category
                    if filter_service:
                        metadata_filter["aws_service"] = filter_service

                    # Pass None if no filters
                    filter_to_use = metadata_filter if metadata_filter else None

                    results = query_vector_store(
                        query_text,
                        bedrock_client,
                        s3vectors_client,
                        model_id,
                        dimension,
                        embedding_purpose=embedding_purpose_query if embedding_purpose_query else None,
                        top_k=top_k,
                        metadata_filter=filter_to_use
                    )

                    if results:
                        st.success(f"✅ Found {len(results)} results")

                        for i, result in enumerate(results, 1):
                            with st.container():
                                st.markdown(f"### 📄 Result {i}")

                                col1, col2, col3 = st.columns([2, 1, 1])

                                with col1:
                                    st.metric("Document", f"Result #{i}")
                                with col2:
                                    st.metric("Distance", f"{result['distance']:.4f}")
                                with col3:
                                    similarity_score = max(0, 1 - result['distance'])
                                    st.metric("Similarity", f"{similarity_score:.2%}")

                                if result['text']:
                                    st.markdown("**Content:**")
                                    st.info(result['text'])

                                if result['metadata']:
                                    with st.expander("📋 View Metadata"):
                                        # Separate filterable and non-filterable
                                        filterable = {}
                                        non_filterable = {}

                                        for key, value in result['metadata'].items():
                                            if key in NON_FILTERABLE_KEYS:
                                                non_filterable[key] = value
                                            else:
                                                filterable[key] = value

                                        if filterable:
                                            st.markdown("**✅ Filterable Metadata:**")
                                            st.json(filterable)

                                        if non_filterable:
                                            st.markdown("**❌ Non-Filterable Metadata:**")
                                            st.json(non_filterable)

                                st.divider()
                    else:
                        st.warning("⚠️ No results found")
            else:
                st.warning("⚠️ Please enter a query")

        st.markdown("---")
        st.markdown("### 💡 Sample Queries")

        sample_queries = [
            "What is a serverless compute service?",
            "Tell me about object storage",
            "How do I manage database services?",
            "What services help with messaging?"
        ]

        cols = st.columns(len(sample_queries))
        for i, sample_query in enumerate(sample_queries):
            with cols[i]:
                if st.button(f"🔍 Query {i+1}", key=f"sample_{i}"):
                    st.session_state.sample_query = sample_query
                    st.rerun()

        if 'sample_query' in st.session_state:
            st.info(f"Selected query: {st.session_state.sample_query}")
            del st.session_state.sample_query

    # ========================================================================
    # TAB 3: MANAGE BUCKET
    # ========================================================================
    with tab3:
        st.header("📋 Manage Bucket")
        st.markdown("View all vector indexes in the current vector bucket")

        # Filter by prefix (optional)
        prefix_filter = st.text_input(
            "Filter by prefix (optional)",
            placeholder="e.g., my-index",
            help="Only show indexes that start with this prefix"
        )

        if st.button("🔄 Load Indexes", type="primary"):
            with st.spinner("Loading indexes..."):
                indexes = list_indexes(
                    s3vectors_client, 
                    prefix=prefix_filter if prefix_filter.strip() else None
                )
                st.session_state.indexes = indexes
                st.rerun()

        # Display indexes
        if 'indexes' in st.session_state:
            indexes = st.session_state.indexes

            if indexes:
                st.success(f"✅ Found {len(indexes)} index(es)")

                # Summary information - no columns, no clipping
                st.markdown("### 📊 Summary")
                st.metric("Total Indexes", len(indexes))

                st.markdown("**Vector Bucket:**")
                st.code(VECTOR_BUCKET_NAME, language=None)

                st.markdown("**Current Index:**")
                st.code(VECTOR_INDEX_NAME, language=None)

                st.markdown("---")

                # Display each index
                st.markdown("### 📑 Indexes")

                for i, index in enumerate(indexes, 1):
                    is_current = index.get('indexName') == VECTOR_INDEX_NAME

                    with st.expander(
                        f"{'🟢' if is_current else '⚪'} Index {i}: {index.get('indexName', 'unknown')}" + 
                        (" (Current)" if is_current else ""),
                        expanded=is_current
                    ):
                        st.markdown("**Index Name:**")
                        st.code(index.get('indexName', 'N/A'), language=None)

                        st.markdown("**Vector Bucket Name:**")
                        st.code(index.get('vectorBucketName', 'N/A'), language=None)

                        st.markdown("**Amazon Resource Name (ARN):**")
                        st.code(index.get('indexArn', 'N/A'), language=None)

                        creation_time = index.get('creationTime')
                        if creation_time:
                            st.markdown("**Creation Time:**")
                            st.text(str(creation_time))

                        # Show full index details
                        with st.expander("View Full Index Details"):
                            st.json(index)
            else:
                st.info("ℹ️ No indexes found in this vector bucket")
        else:
            st.info("👆 Click 'Load Indexes' to view all indexes in the vector bucket")

    # ========================================================================
    # TAB 4: MANAGE INDEX
    # ========================================================================
    with tab4:
        st.header("Manage Vector Index")

        st.markdown("### 📊 Index Statistics")

        # Row 1: Refresh Count button
        if st.button("🔄 Refresh Count", type="secondary"):
            with st.spinner("Counting vectors..."):
                vector_keys = get_vector_keys(s3vectors_client)
                st.session_state.vector_count = len(vector_keys)
                st.rerun()

        # Display count
        if 'vector_count' in st.session_state:
            st.metric("Total Vectors in Index", st.session_state.vector_count)
        else:
            st.info("Click 'Refresh Count' to see total vectors")

        st.markdown("---")

        # Row 2: Preview Index
        st.markdown("### 👁️ Preview Index")

        preview_count = st.number_input(
            "Number of vectors to preview",
            min_value=1,
            max_value=100,
            value=10,
            help="Select how many vectors to preview"
        )

        if st.button("👁️ Load Preview", type="secondary"):
            with st.spinner(f"Loading {preview_count} vectors..."):
                vectors = list_all_vectors(s3vectors_client, max_results=preview_count)
                st.session_state.preview_vectors = vectors
                # Also update count if not already set
                if 'vector_count' not in st.session_state:
                    st.session_state.vector_count = len(get_vector_keys(s3vectors_client))
                st.rerun()

        # Display preview
        if 'preview_vectors' in st.session_state and st.session_state.preview_vectors:
            st.markdown(f"**Showing {len(st.session_state.preview_vectors)} vectors:**")

            for i, vector in enumerate(st.session_state.preview_vectors, 1):
                with st.expander(f"📄 Vector {i}: {vector.get('key', 'unknown')}", expanded=False):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown("**Vector Key:**")
                        st.code(vector.get('key', 'unknown'))

                    with col2:
                        metadata = vector.get('metadata', {})

                        # Get text content
                        text_content = metadata.get('text', '')
                        if text_content:
                            st.markdown("**Content:**")
                            st.info(text_content[:200] + "..." if len(text_content) > 200 else text_content)

                    # Show all metadata
                    if metadata:
                        st.markdown("**All Metadata:**")

                        # Separate filterable and non-filterable
                        filterable = {}
                        non_filterable = {}

                        for key, value in metadata.items():
                            if key in NON_FILTERABLE_KEYS:
                                non_filterable[key] = value
                            else:
                                filterable[key] = value

                        col1, col2 = st.columns(2)

                        with col1:
                            if filterable:
                                st.markdown("**✅ Filterable:**")
                                st.json(filterable)

                        with col2:
                            if non_filterable:
                                st.markdown("**❌ Non-Filterable:**")
                                st.json(non_filterable)

        st.markdown("---")
        st.markdown("### 🗑️ Reset Index")

        st.warning("""
        ⚠️ **Warning: This action cannot be undone!**

        This will delete ALL vectors from the index. The index structure will remain,
        but all document embeddings and metadata will be permanently removed.
        """)

        # Confirmation checkbox
        confirm_reset = st.checkbox("I understand this will delete all vectors permanently")

        if st.button(
            "🗑️ Reset Index",
            type="primary",
            disabled=not confirm_reset,
            help="Delete all vectors from the index"
        ):
            with st.spinner("Deleting all vectors..."):
                success, count = reset_index(s3vectors_client)

                if success:
                    if count > 0:
                        st.success(f"✅ Successfully deleted {count} vectors!")
                        st.balloons()
                        # Clear session state
                        st.session_state.vector_count = 0
                        if 'preview_vectors' in st.session_state:
                            del st.session_state.preview_vectors
                        st.rerun()
                    else:
                        st.info("ℹ️ Index was already empty")
                else:
                    st.error("❌ Failed to reset index")

if __name__ == "__main__":
    main()
import os
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

AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"

# Your Vector Bucket and Index Names
VECTOR_BUCKET_NAME = cmn_settings.VECTOR_BUCKET_NAME
VECTOR_INDEX_NAME = cmn_settings.VECTOR_INDEX_NAME

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

def generate_embedding(text: str, bedrock_client) -> List[float]:
    """Generate embedding vector using Amazon Bedrock"""
    try:
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({"inputText": text})
        )
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def add_documents_to_vector_store(
    texts: List[str], 
    bedrock_client, 
    s3vectors_client,
    metadata_list: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Add multiple documents to the vector store"""
    vectors_to_put = []
    timestamp = datetime.now().isoformat()

    for i, text in enumerate(texts):
        embedding = generate_embedding(text, bedrock_client)
        if embedding is None:
            continue

        metadata = {
            "text": text,
            "added_timestamp": timestamp,
            "document_index": str(i)
        }

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
    top_k: int = 5,
    metadata_filter: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """Query the vector store for similar documents"""
    embedding = generate_embedding(question, bedrock_client)
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

        if metadata_filter:
            query_params["metadataFilter"] = metadata_filter

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

    with st.expander("⚙️ Configuration", expanded=False):
        st.code(f"""
Vector Bucket: {VECTOR_BUCKET_NAME}
Vector Index: {VECTOR_INDEX_NAME}
Embedding Model: {BEDROCK_MODEL_ID}
AWS Region: {AWS_REGION}
        """)

    tab1, tab2 = st.tabs(["📝 Add Documents", "🔎 Query Documents"])

    # ========================================================================
    # TAB 1: ADD DOCUMENTS
    # ========================================================================
    with tab1:
        st.header("Add Documents to Vector Store")

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
                st.markdown("**Optional Metadata**")
                doc_type = st.text_input("Document Type", placeholder="e.g., article")
                category = st.text_input("Category", placeholder="e.g., technical")
                service = st.text_input("AWS Service", placeholder="e.g., S3")

            if st.button("➕ Add Document", type="primary"):
                if document_text.strip():
                    with st.spinner("Adding document to vector store..."):
                        metadata = {}
                        if doc_type:
                            metadata["document_type"] = doc_type
                        if category:
                            metadata["category"] = category
                        if service:
                            metadata["aws_service"] = service

                        result = add_documents_to_vector_store(
                            [document_text],
                            bedrock_client,
                            s3vectors_client,
                            [metadata] if metadata else None
                        )

                        if result:
                            st.success("✅ Document added successfully!")
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

            col1, col2 = st.columns(2)
            with col1:
                bulk_doc_type = st.text_input("Document Type (all)", placeholder="e.g., article")
            with col2:
                bulk_category = st.text_input("Category (all)", placeholder="e.g., technical")

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
                                metadata_list
                            )

                            if result:
                                st.success(f"✅ Successfully added {len(documents)} documents!")
                                st.json(result)
                    else:
                        st.warning("⚠️ No valid documents found")
                else:
                    st.warning("⚠️ Please enter documents")

        else:  # Sample AWS Services Data
            st.subheader("Add Sample AWS Services Data")
            st.info("📚 This will add sample AWS service descriptions")

            sample_texts = [
                "Amazon S3 is an object storage service offering industry-leading scalability, data availability, security, and performance.",
                "Amazon EC2 provides secure and resizable compute capacity in the cloud, allowing you to launch virtual servers as needed.",
                "AWS Lambda lets you run code without provisioning or managing servers, paying only for the compute time you consume.",
                "Amazon RDS makes it easy to set up, operate, and scale a relational database in the cloud with automated backups.",
                "Amazon DynamoDB is a fully managed NoSQL database service that provides fast and predictable performance with seamless scalability.",
                "Amazon VPC lets you provision a logically isolated section of the AWS Cloud where you can launch AWS resources.",
                "Amazon CloudFront is a fast content delivery network service that securely delivers data, videos, and applications globally.",
                "AWS IAM enables you to manage access to AWS services and resources securely with fine-grained permissions.",
                "Amazon SQS is a fully managed message queuing service that enables you to decouple and scale microservices.",
                "Amazon SNS is a fully managed messaging service for both application-to-application and application-to-person communication.",
                "Amazon ECS is a fully managed container orchestration service that makes it easy to deploy and scale containerized applications.",
                "AWS CloudFormation provides a common language for describing and provisioning infrastructure resources in your cloud environment.",
                "Amazon Bedrock is a fully managed service that offers foundation models from leading AI companies through a single API.",
                "Amazon SageMaker helps data scientists and developers prepare, build, train, and deploy machine learning models quickly.",
                "AWS Glue is a serverless data integration service that makes it easy to discover, prepare, and combine data for analytics."
            ]

            with st.expander("📄 Preview Sample Documents", expanded=False):
                for i, text in enumerate(sample_texts, 1):
                    st.text(f"{i}. {text}")

            st.markdown(f"**Total documents:** {len(sample_texts)}")

            if st.button("➕ Add Sample Data", type="primary"):
                with st.spinner(f"Adding {len(sample_texts)} sample documents..."):
                    metadata_list = [
                        {
                            "document_type": "service_description",
                            "category": "aws_services",
                            "aws_service": text.split()[1]  # Extract service name
                        }
                        for text in sample_texts
                    ]

                    result = add_documents_to_vector_store(
                        sample_texts,
                        bedrock_client,
                        s3vectors_client,
                        metadata_list
                    )

                    if result:
                        st.success(f"✅ Successfully added {len(sample_texts)} sample documents!")
                        st.json(result)

    # ========================================================================
    # TAB 2: QUERY DOCUMENTS
    # ========================================================================
    with tab2:
        st.header("Query Vector Store")

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

        with st.expander("🔧 Advanced Filters (Optional)", expanded=False):
            st.markdown("**Filter by metadata:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                filter_doc_type = st.text_input("Document Type", key="filter_doc_type")
            with col2:
                filter_category = st.text_input("Category", key="filter_category")
            with col3:
                filter_service = st.text_input("AWS Service", key="filter_service")

        if st.button("🔎 Search", type="primary"):
            if query_text.strip():
                with st.spinner("Searching vector store..."):
                    metadata_filter = {}
                    if filter_doc_type:
                        metadata_filter["document_type"] = filter_doc_type
                    if filter_category:
                        metadata_filter["category"] = filter_category
                    if filter_service:
                        metadata_filter["aws_service"] = filter_service

                    results = query_vector_store(
                        query_text,
                        bedrock_client,
                        s3vectors_client,
                        top_k=top_k,
                        metadata_filter=metadata_filter if metadata_filter else None
                    )

                    if results:
                        st.success(f"✅ Found {len(results)} results")

                        for i, result in enumerate(results, 1):
                            with st.container():
                                st.markdown(f"### 📄 Result {i}")

                                col1, col2, col3 = st.columns([2, 1, 1])

                                with col1:
                                    st.metric("Document Key", result['key'])
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
                                        st.json(result['metadata'])

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

if __name__ == "__main__":
    main()
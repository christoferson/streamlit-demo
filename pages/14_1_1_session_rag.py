import boto3
import streamlit as st
import cmn_auth
import cmn_settings
import logging
import json
import os
import uuid
from pathlib import Path
from botocore.exceptions import ClientError
from typing import List, Dict
import faiss
import numpy as np
import pickle

AWS_REGION = cmn_settings.AWS_REGION
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

# Directory to store session data
SESSION_DATA_DIR = Path("session_data")
SESSION_DATA_DIR.mkdir(exist_ok=True)

# Model configurations
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
CLAUDE_MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"

opt_model_id_list = [

    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "global.anthropic.claude-opus-4-6-v1",
]


class SessionRAG:
    """Handles session-based RAG with FAISS vector store"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = SESSION_DATA_DIR / session_id
        self.session_dir.mkdir(exist_ok=True)

        self.index_path = self.session_dir / "faiss.index"
        self.metadata_path = self.session_dir / "metadata.pkl"
        self.dimension = 1024  # Titan embedding v2 dimension

        # Initialize or load FAISS index
        if self.index_path.exists():
            self.load_index()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Amazon Titan"""
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": self.dimension,
                "normalize": True
            })

            response = bedrock_runtime.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response['body'].read())
            embedding = np.array(response_body['embedding'], dtype=np.float32)
            return embedding

        except ClientError as err:
            logger.error(f"Error getting embedding: {err}")
            raise

    def add_document(self, text: str, metadata: Dict):
        """Add document to FAISS index"""
        embedding = self.get_embedding(text)
        embedding = embedding.reshape(1, -1)

        self.index.add(embedding)
        self.metadata.append({
            "text": text,
            "metadata": metadata
        })

        self.save_index()

    def add_documents_batch(self, texts: List[str], metadatas: List[Dict]):
        """Add multiple documents at once"""
        for text, metadata in zip(texts, metadatas):
            self.add_document(text, metadata)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []

        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({
                    **self.metadata[idx],
                    "distance": float(distance)
                })

        return results

    def save_index(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        """Load FAISS index and metadata"""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def get_stats(self) -> Dict:
        """Get statistics about the index"""
        return {
            "total_documents": self.index.ntotal,
            "dimension": self.dimension,
            "session_id": self.session_id
        }


def extract_text_from_file(uploaded_file) -> str:
    """Extract text content from uploaded file"""
    file_type = uploaded_file.type

    if file_type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif file_type == "text/markdown":
        return uploaded_file.getvalue().decode("utf-8")
    elif file_type == "text/csv":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        # For other types, return as string (could be enhanced with proper parsers)
        try:
            return uploaded_file.getvalue().decode("utf-8")
        except:
            return f"[Binary file: {uploaded_file.name}]"


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def generate_rag_response(rag: SessionRAG, query: str, model_id: str, temperature: float, max_tokens: int, top_k_retrieval: int) -> tuple:
    """Generate response using RAG"""

    # Retrieve relevant documents
    results = rag.search(query, k=top_k_retrieval)

    if not results:
        return "No documents found in the knowledge base. Please upload documents first.", []

    # Prepare context from retrieved documents
    context_parts = []
    for idx, result in enumerate(results, 1):
        context_parts.append(f"[Document {idx}]\n{result['text']}\n")

    context = "\n".join(context_parts)

    # Prepare prompt
    prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

    # Call Claude via Bedrock
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]

    inference_config = {
        #"temperature": temperature,
        "maxTokens": max_tokens,
        #"topP": 1.0,
    }

    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig=inference_config
    )

    answer = response['output']['message']['content'][0]['text']

    return answer, results


# Page configuration
st.set_page_config(
    page_title="Session RAG",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if hasattr(st, 'logo'):
    st.logo(icon_image="images/logo.png", image="images/logo_text.png")

st.title("Session-Based RAG with FAISS")
st.write("Create sessions, upload documents, and query using RAG with Amazon Bedrock")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    # Session management
    st.subheader("Session Management")

    # List existing sessions
    existing_sessions = [d.name for d in SESSION_DATA_DIR.iterdir() if d.is_dir()]

    session_action = st.radio(
        "Action",
        ["New Session", "Load Existing"],
        key="session_action"
    )

    if session_action == "New Session":
        if st.button("Create New Session"):
            new_session_id = str(uuid.uuid4())[:8]
            st.session_state["current_session_id"] = new_session_id
            st.session_state["rag"] = SessionRAG(new_session_id)
            st.success(f"Created session: {new_session_id}")
            st.rerun()
    else:
        if existing_sessions:
            selected_session = st.selectbox(
                "Select Session",
                existing_sessions,
                key="selected_session"
            )
            if st.button("Load Session"):
                st.session_state["current_session_id"] = selected_session
                st.session_state["rag"] = SessionRAG(selected_session)
                st.success(f"Loaded session: {selected_session}")
                st.rerun()
        else:
            st.info("No existing sessions found")

    st.divider()

    # Model configuration
    st.subheader("Model Settings")
    opt_model_id = st.selectbox(
        "Model ID",
        options=opt_model_id_list,
        index=0,
        key="model_id"
    )
    opt_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        key="temperature"
    )
    opt_max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=4096,
        value=2048,
        step=256,
        key="max_tokens"
    )
    opt_top_k_retrieval = st.slider(
        "Top K Documents",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="top_k_retrieval"
    )

# Main content
if "current_session_id" not in st.session_state:
    st.info("Please create a new session or load an existing one from the sidebar")
    st.stop()

# Display current session info
rag = st.session_state["rag"]
stats = rag.get_stats()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Session ID", stats["session_id"])
with col2:
    st.metric("Total Documents", stats["total_documents"])
with col3:
    st.metric("Embedding Dimension", stats["dimension"])

# Document upload section
st.header("Document Management")

uploaded_files = st.file_uploader(
    "Upload documents to add to the knowledge base",
    type=["txt", "md", "csv", "pdf", "docx", "json"],
    accept_multiple_files=True,
    key="file_uploader"
)

col1, col2 = st.columns([2, 1])
with col1:
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000)
with col2:
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200)

if uploaded_files and st.button("Process and Add Documents"):
    with st.spinner("Processing documents..."):
        progress_bar = st.progress(0)

        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Extract text
                text = extract_text_from_file(uploaded_file)

                # Chunk text
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)

                # Add chunks to RAG
                for chunk_idx, chunk in enumerate(chunks):
                    metadata = {
                        "filename": uploaded_file.name,
                        "chunk_id": chunk_idx,
                        "total_chunks": len(chunks)
                    }
                    rag.add_document(chunk, metadata)

                progress_bar.progress((idx + 1) / len(uploaded_files))

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        st.success(f"Successfully processed {len(uploaded_files)} documents!")
        st.rerun()

# Query section
st.header("Query Knowledge Base")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for idx, source in enumerate(message["sources"], 1):
                    st.caption(f"**Source {idx}** (Distance: {source['distance']:.4f})")
                    st.caption(f"File: {source['metadata']['filename']}")
                    st.text(source['text'][:200] + "..." if len(source['text']) > 200 else source['text'])
                    st.divider()

# Chat input
if prompt := st.chat_input(
    "Ask a question about your documents" if stats["total_documents"] > 0 else "Upload documents first",
    disabled=(stats["total_documents"] == 0)
):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            try:
                answer, sources = generate_rag_response(
                    rag,
                    prompt,
                    opt_model_id,
                    opt_temperature,
                    opt_max_tokens,
                    opt_top_k_retrieval
                )

                st.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("View Sources"):
                        for idx, source in enumerate(sources, 1):
                            st.caption(f"**Source {idx}** (Distance: {source['distance']:.4f})")
                            st.caption(f"File: {source['metadata']['filename']}")
                            st.text(source['text'][:500] + "..." if len(source['text']) > 500 else source['text'])
                            st.divider()

                # Save to history
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)

import os
import boto3
import json
import base64
import uuid
from typing import List, Dict, Optional, Tuple
from PIL import Image
import io
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import streamlit as st
from dataclasses import dataclass


@dataclass
class Config:
    # AWS Configuration
    AWS_PROFILE: str = os.getenv('AWS_PROFILE', 'default')
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-east-1')

    # OpenSearch Serverless Configuration
    OPENSEARCH_ENDPOINT: str = os.getenv('OPENSEARCH_ENDPOINT', 'your-collection-endpoint.us-east-1.aoss.amazonaws.com')
    COLLECTION_NAME: str = os.getenv('COLLECTION_NAME', 'multimodal-search')
    INDEX_NAME: str = os.getenv('INDEX_NAME', 'products')

    # Bedrock Configuration
    TITAN_IMAGE_MODEL_ID: str = 'amazon.titan-embed-image-v1'
    TITAN_TEXT_MODEL_ID: str = 'amazon.titan-embed-text-v1'

    # Embedding Configuration
    EMBEDDING_DIMENSION: int = 1024

    # App Configuration
    MAX_FILE_SIZE_MB: int = 10
    SUPPORTED_IMAGE_FORMATS: list = None
    SEARCH_RESULTS_LIMIT: int = 20

    def __post_init__(self):
        if self.SUPPORTED_IMAGE_FORMATS is None:
            self.SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp']

# Global config instance
config = Config()

class MultimodalSearchService:
    def __init__(self):
        self.session = None
        self.bedrock_client = None
        self.opensearch_client = None
        self.connected = False

    def connect_aws(self) -> bool:
        """Connect to AWS services"""
        try:
            self.session = self._get_aws_session()
            self.bedrock_client = self._get_bedrock_client()
            self.opensearch_client = self._get_opensearch_client()
            self.connected = True
            return True
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
            self.connected = False
            return False

    def _get_aws_session(self) -> boto3.Session:
        """Get AWS session with specified profile"""
        if config.AWS_PROFILE and config.AWS_PROFILE != 'default':
            session = boto3.Session(
                profile_name=config.AWS_PROFILE,
                region_name=config.AWS_REGION
            )
        else:
            session = boto3.Session(region_name=config.AWS_REGION)

        # Test credentials
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        st.success(f"✅ AWS Profile: {config.AWS_PROFILE}")
        st.info(f"Account: {identity.get('Account', 'Unknown')}")

        return session

    def _get_bedrock_client(self):
        """Initialize Bedrock client"""
        return self.session.client('bedrock-runtime', region_name=config.AWS_REGION)

    def _get_opensearch_client(self):
        """Initialize OpenSearch Serverless client"""
        credentials = self.session.get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            config.AWS_REGION,
            'aoss',
            session_token=credentials.token
        )

        client = OpenSearch(
            hosts=[{'host': config.OPENSEARCH_ENDPOINT, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60
        )

        # Test connection
        client.info()
        st.success("✅ OpenSearch Connected")
        return client

    def check_index_exists(self) -> bool:
        """Check if index exists"""
        if not self.connected:
            return False
        try:
            return self.opensearch_client.indices.exists(index=config.INDEX_NAME)
        except Exception as e:
            st.error(f"❌ Error checking index: {str(e)}")
            return False

    def create_index(self) -> bool:
        """Create index with proper mappings"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        index_body = {
            "settings": {
                "index.knn": True,
                "index.knn.space_type": "cosinesimil",
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "product_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "description": {"type": "text"},
                    "image_path": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "image_embedding": {
                        "type": "knn_vector",
                        "dimension": config.EMBEDDING_DIMENSION,
                        "space_type": "cosinesimil"
                    },
                    "text_embedding": {
                        "type": "knn_vector",
                        "dimension": config.EMBEDDING_DIMENSION,
                        "space_type": "cosinesimil"
                    }
                }
            }
        }

        try:
            if self.opensearch_client.indices.exists(index=config.INDEX_NAME):
                st.warning(f"⚠️ Index '{config.INDEX_NAME}' already exists")
                return True

            self.opensearch_client.indices.create(
                index=config.INDEX_NAME,
                body=index_body
            )
            st.success(f"✅ Created index: {config.INDEX_NAME}")
            return True
        except Exception as e:
            st.error(f"❌ Index creation failed: {str(e)}")
            return False

    def delete_index(self) -> bool:
        """Delete the index"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        try:
            if not self.opensearch_client.indices.exists(index=config.INDEX_NAME):
                st.warning(f"⚠️ Index '{config.INDEX_NAME}' does not exist")
                return True

            self.opensearch_client.indices.delete(index=config.INDEX_NAME)
            st.success(f"✅ Deleted index: {config.INDEX_NAME}")
            return True
        except Exception as e:
            st.error(f"❌ Index deletion failed: {str(e)}")
            return False

    def get_connection_status(self) -> Dict:
        """Get connection and index status"""
        status = {
            'aws_connected': self.connected,
            'index_exists': False,
            'can_operate': False
        }

        if self.connected:
            status['index_exists'] = self.check_index_exists()
            status['can_operate'] = status['index_exists']

        return status

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    def get_image_embedding(self, image: Image.Image) -> List[float]:
        """Get embedding for an image using Titan"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return None

        try:
            image_base64 = self._image_to_base64(image)

            body = json.dumps({
                "inputImage": image_base64,
                "embeddingConfig": {
                    "outputEmbeddingLength": config.EMBEDDING_DIMENSION
                }
            })

            response = self.bedrock_client.invoke_model(
                modelId=config.TITAN_IMAGE_MODEL_ID,
                body=body,
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

        except Exception as e:
            st.error(f"❌ Image embedding failed: {str(e)}")
            return None

    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Titan"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return None

        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": config.EMBEDDING_DIMENSION
            })

            response = self.bedrock_client.invoke_model(
                modelId=config.TITAN_TEXT_MODEL_ID,
                body=body,
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

        except Exception as e:
            st.error(f"❌ Text embedding failed: {str(e)}")
            return None

    def register_product(self, image: Image.Image, title: str, description: str) -> bool:
        """Register a new product with image and text"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        if not self.check_index_exists():
            st.error("❌ Index does not exist. Please create index first.")
            return False

        try:
            product_id = str(uuid.uuid4())

            # Get embeddings
            with st.spinner("🔄 Generating image embedding..."):
                image_embedding = self.get_image_embedding(image)

            with st.spinner("🔄 Generating text embedding..."):
                text_embedding = self.get_text_embedding(f"{title}. {description}")

            if not image_embedding or not text_embedding:
                return False

            # Prepare document
            doc = {
                "product_id": product_id,
                "title": title,
                "description": description,
                "image_embedding": image_embedding,
                "text_embedding": text_embedding,
                "created_at": "now"
            }

            # Index document
            with st.spinner("💾 Saving to OpenSearch..."):
                response = self.opensearch_client.index(
                    index=config.INDEX_NAME,
                    id=product_id,
                    body=doc,
                    refresh=True
                )

            if response['result'] in ['created', 'updated']:
                st.success(f"✅ Product registered successfully! ID: {product_id}")
                return True
            else:
                st.error("❌ Failed to register product")
                return False

        except Exception as e:
            st.error(f"❌ Registration failed: {str(e)}")
            return False

    def search_by_image(self, image: Image.Image, limit: int = 10) -> List[Dict]:
        """Search for similar products using an image"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return []

        if not self.check_index_exists():
            st.error("❌ Index does not exist. Please create index first.")
            return []

        try:
            with st.spinner("🔍 Searching by image..."):
                image_embedding = self.get_image_embedding(image)

                if not image_embedding:
                    return []

                search_body = {
                    "size": limit,
                    "_source": ["product_id", "title", "description"],
                    "query": {
                        "knn": {
                            "image_embedding": {
                                "vector": image_embedding,
                                "k": limit
                            }
                        }
                    }
                }

                response = self.opensearch_client.search(
                    index=config.INDEX_NAME,
                    body=search_body
                )

                results = []
                for hit in response['hits']['hits']:
                    results.append({
                        'product_id': hit['_source']['product_id'],
                        'title': hit['_source']['title'],
                        'description': hit['_source']['description'],
                        'score': hit['_score']
                    })

                return results

        except Exception as e:
            st.error(f"❌ Image search failed: {str(e)}")
            return []

    def search_by_text(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for similar products using text"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return []

        if not self.check_index_exists():
            st.error("❌ Index does not exist. Please create index first.")
            return []

        try:
            with st.spinner("🔍 Searching by text..."):
                text_embedding = self.get_text_embedding(query)

                if not text_embedding:
                    return []

                search_body = {
                    "size": limit,
                    "_source": ["product_id", "title", "description"],
                    "query": {
                        "knn": {
                            "text_embedding": {
                                "vector": text_embedding,
                                "k": limit
                            }
                        }
                    }
                }

                response = self.opensearch_client.search(
                    index=config.INDEX_NAME,
                    body=search_body
                )

                results = []
                for hit in response['hits']['hits']:
                    results.append({
                        'product_id': hit['_source']['product_id'],
                        'title': hit['_source']['title'],
                        'description': hit['_source']['description'],
                        'score': hit['_score']
                    })

                return results

        except Exception as e:
            st.error(f"❌ Text search failed: {str(e)}")
            return []

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        if not self.connected:
            return {'error': 'Not connected to AWS services'}

        if not self.check_index_exists():
            return {'error': 'Index does not exist'}

        try:
            response = self.opensearch_client.count(index=config.INDEX_NAME)
            return {
                'total_products': response['count'],
                'index_name': config.INDEX_NAME,
                'collection_name': config.COLLECTION_NAME
            }
        except Exception as e:
            return {'error': str(e)}
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
from dotenv import load_dotenv
from opensearchpy import AWSV4SignerAuth
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('opensearch_debug.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


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
    TITAN_MULTIMODAL_MODEL_ID: str = 'amazon.titan-embed-image-v1'


    # Embedding Configuration
    EMBEDDING_DIMENSION: int = 1024

    # App Configuration
    MAX_FILE_SIZE_MB: int = 10
    SUPPORTED_IMAGE_FORMATS: list = None
    SEARCH_RESULTS_LIMIT: int = 20

    def __post_init__(self):
        if self.SUPPORTED_IMAGE_FORMATS is None:
            self.SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp']

        # Log configuration
        logger.info("=== Configuration Loaded ===")
        logger.info(f"AWS_PROFILE: {self.AWS_PROFILE}")
        logger.info(f"AWS_REGION: {self.AWS_REGION}")
        logger.info(f"OPENSEARCH_ENDPOINT: {self.OPENSEARCH_ENDPOINT}")
        logger.info(f"COLLECTION_NAME: {self.COLLECTION_NAME}")
        logger.info(f"INDEX_NAME: {self.INDEX_NAME}")


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
            logger.info("Getting Bedrock client...")
            self.bedrock_client = self._get_bedrock_client()
            self.opensearch_client = self._get_opensearch_client()
            self.connected = True
            return True
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)} {config.OPENSEARCH_ENDPOINT}")
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
        logger.info("Creating OpenSearch Serverless client...")

        try:
            credentials = self.session.get_credentials()
            if not credentials:
                raise Exception("No AWS credentials available")

            logger.info("✅ AWS credentials obtained")
            logger.info(f"Access Key ID: {credentials.access_key[:10]}...")

            # Create auth
            logger.info("Creating AWSV4SignerAuth...")
            awsauth = AWSV4SignerAuth(credentials, config.AWS_REGION, 'aoss')
            logger.info("✅ AWSV4SignerAuth created")

            # Clean endpoint
            endpoint = config.OPENSEARCH_ENDPOINT.replace('https://', '').replace('http://', '')
            logger.info(f"Using endpoint: {endpoint}")

            # Create client
            logger.info("Creating OpenSearch client...")
            client = OpenSearch(
                hosts=[{'host': endpoint, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=60
            )

            logger.info("✅ OpenSearch client created successfully")
            st.success("✅ OpenSearch Client Ready")

            # Don't test connection here - it will be tested when we actually use it
            return client

        except Exception as e:
            logger.error(f"Failed to create OpenSearch client: {str(e)}")
            logger.exception("OpenSearch client creation exception:")
            raise e

    # def _get_opensearch_client(self):
        
    #     """Initialize OpenSearch Serverless client"""
    #     logger.info("Creating OpenSearch Serverless client...")

    #     try:
    #         # Get credentials
    #         logger.info("Getting AWS credentials for OpenSearch...")
    #         credentials = self.session.get_credentials()

    #         if not credentials:
    #             raise Exception("No AWS credentials available")

    #         logger.info("✅ AWS credentials obtained")
    #         logger.info(f"Access Key ID: {credentials.access_key[:10]}...")

    #         # Create auth
    #         logger.info("Creating AWSV4SignerAuth...")
    #         awsauth = AWSV4SignerAuth(credentials, config.AWS_REGION, 'aoss')
    #         logger.info("✅ AWSV4SignerAuth created")

    #         # Clean endpoint
    #         endpoint = config.OPENSEARCH_ENDPOINT.replace('https://', '').replace('http://', '')
    #         logger.info(f"Using endpoint: {endpoint}")

    #         # Create client
    #         logger.info("Creating OpenSearch client...")
    #         client = OpenSearch(
    #             hosts=[{'host': endpoint, 'port': 443}],
    #             http_auth=awsauth,
    #             use_ssl=True,
    #             verify_certs=True,
    #             connection_class=RequestsHttpConnection,
    #             timeout=60
    #         )
    #         logger.info("✅ OpenSearch client created")

    #         # Test connection with a different method since info() doesn't work on Serverless
    #         logger.info("Testing OpenSearch connection with cat.indices...")
    #         try:
    #             # Use cat.indices instead of info() for OpenSearch Serverless
    #             indices_response = client.cat.indices(format='json')
    #             logger.info("✅ OpenSearch connection test successful")
    #             logger.info(f"Available indices: {indices_response}")
    #             st.success("✅ OpenSearch Connected")
    #             return client

    #         except Exception as conn_error:
    #             # If cat.indices fails, try a simple search on a non-existent index
    #             # This should return a proper error (not 404) if connection works
    #             logger.info("cat.indices failed, trying alternative connection test...")
    #             try:
    #                 # This will fail but with a different error if connection works
    #                 client.search(index='_test_connection_index_that_does_not_exist', body={'query': {'match_all': {}}})
    #             except Exception as search_error:
    #                 error_str = str(search_error).lower()
    #                 if 'no such index' in error_str or 'index_not_found' in error_str or '404' in error_str:
    #                     # This is expected - means connection works but index doesn't exist
    #                     logger.info("✅ OpenSearch connection confirmed (index not found is expected)")
    #                     st.success("✅ OpenSearch Connected")
    #                     return client
    #                 else:
    #                     # Different error - connection issue
    #                     logger.error(f"Connection test failed: {search_error}")
    #                     raise search_error

    #             logger.error(f"OpenSearch connection test failed: {str(conn_error)}")
    #             raise conn_error

    #     except Exception as e:
    #         logger.error(f"Failed to create OpenSearch client: {str(e)}")
    #         logger.exception("OpenSearch client creation exception:")
    #         raise e

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
        """Create index with proper mappings for OpenSearch Serverless"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        # OpenSearch Serverless index configuration
        index_body = {
            "settings": {
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
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "text_embedding": {
                        "type": "knn_vector",
                        "dimension": config.EMBEDDING_DIMENSION,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
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


    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Titan Multimodal Embeddings"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return None

        try:
            # Use multimodal model for text
            body = json.dumps({
                "inputText": text,
                "embeddingConfig": {
                    "outputEmbeddingLength": config.EMBEDDING_DIMENSION
                }
            })

            response = self.bedrock_client.invoke_model(
                modelId=config.TITAN_MULTIMODAL_MODEL_ID,  # Same model as image
                body=body,
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

        except Exception as e:
            st.error(f"❌ Text embedding failed: {str(e)}")
            return None

    def get_image_embedding(self, image: Image.Image) -> List[float]:
        """Get embedding for an image using Titan Multimodal Embeddings"""
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
                modelId=config.TITAN_MULTIMODAL_MODEL_ID,  # Same model as text
                body=body,
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

        except Exception as e:
            st.error(f"❌ Image embedding failed: {str(e)}")
            return None

    from datetime import datetime, timezone

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

            # Get embeddings using the multimodal model
            with st.spinner("🔄 Generating image embedding..."):
                image_embedding = self.get_image_embedding(image)

            with st.spinner("🔄 Generating text embedding..."):
                text_embedding = self.get_text_embedding(f"{title}. {description}")

            if not image_embedding or not text_embedding:
                return False

            # Prepare document with proper timestamp
            doc = {
                "product_id": product_id,
                "title": title,
                "description": description,
                "image_embedding": image_embedding,
                "text_embedding": text_embedding,
                "created_at": datetime.now(timezone.utc).isoformat()  # Changed from "now"
            }

            # Index document - Vector Search Collection will auto-generate _id
            with st.spinner("💾 Saving to OpenSearch..."):
                response = self.opensearch_client.index(
                    index=config.INDEX_NAME,
                    body=doc
                    # No refresh parameter - not supported in Vector Search Collection
                    # No id parameter - not supported in Vector Search Collection
                )

            if response['result'] in ['created', 'updated']:
                # Get the auto-generated document ID from the response
                document_id = response['_id']
                st.success(f"✅ Product registered successfully!")
                st.info(f"Product ID: {product_id}")
                st.info(f"Document ID: {document_id}")
                return True
            else:
                st.error("❌ Failed to register product")
                return False

        except Exception as e:
            st.error(f"❌ Registration failed: {str(e)}")
            logger.error(f"Registration error: {str(e)}")
            return False

    def update_product(self, product_id: str, image: Image.Image = None, title: str = None, description: str = None) -> bool:
        """Update an existing product"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        try:
            # まず、product_idでドキュメントを検索
            search_body = {
                "query": {
                    "term": {
                        "product_id": product_id
                    }
                }
            }

            response = self.opensearch_client.search(
                index=config.INDEX_NAME,
                body=search_body
            )

            if response['hits']['total']['value'] == 0:
                st.error(f"❌ Product with ID {product_id} not found")
                return False

            # 内部的な_idを取得
            internal_doc_id = response['hits']['hits'][0]['_id']
            existing_doc = response['hits']['hits'][0]['_source']

            # 更新するフィールドを準備
            update_doc = {}

            if title is not None:
                update_doc['title'] = title
            if description is not None:
                update_doc['description'] = description

            # 新しい埋め込みを生成（必要な場合）
            if image is not None:
                with st.spinner("🔄 Generating new image embedding..."):
                    image_embedding = self.get_image_embedding(image)
                    if image_embedding:
                        update_doc['image_embedding'] = image_embedding

            if title is not None or description is not None:
                # テキストが更新された場合、テキスト埋め込みも更新
                new_text = f"{title or existing_doc.get('title', '')}. {description or existing_doc.get('description', '')}"
                with st.spinner("🔄 Generating new text embedding..."):
                    text_embedding = self.get_text_embedding(new_text)
                    if text_embedding:
                        update_doc['text_embedding'] = text_embedding

            if not update_doc:
                st.warning("⚠️ No fields to update")
                return False

            # ドキュメントを更新
            with st.spinner("💾 Updating document..."):
                update_response = self.opensearch_client.update(
                    index=config.INDEX_NAME,
                    id=internal_doc_id,
                    body={"doc": update_doc},
                    refresh=True
                )

            if update_response['result'] == 'updated':
                st.success(f"✅ Product {product_id} updated successfully!")
                return True
            else:
                st.error("❌ Failed to update product")
                return False

        except Exception as e:
            st.error(f"❌ Update failed: {str(e)}")
            return False

    def delete_product(self, product_id: str) -> bool:
        """Delete a product by its product_id"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        try:
            # product_idでドキュメントを検索
            search_body = {
                "query": {
                    "term": {
                        "product_id": product_id
                    }
                }
            }

            response = self.opensearch_client.search(
                index=config.INDEX_NAME,
                body=search_body
            )

            if response['hits']['total']['value'] == 0:
                st.error(f"❌ Product with ID {product_id} not found")
                return False

            # 内部的な_idを取得
            internal_doc_id = response['hits']['hits'][0]['_id']

            # ドキュメントを削除
            with st.spinner("🗑️ Deleting product..."):
                delete_response = self.opensearch_client.delete(
                    index=config.INDEX_NAME,
                    id=internal_doc_id,
                    refresh=True
                )

            if delete_response['result'] == 'deleted':
                st.success(f"✅ Product {product_id} deleted successfully!")
                return True
            else:
                st.error("❌ Failed to delete product")
                return False

        except Exception as e:
            st.error(f"❌ Delete failed: {str(e)}")
            return False

    def get_product(self, product_id: str) -> Dict:
        """Get a product by its product_id"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return None

        try:
            search_body = {
                "query": {
                    "term": {
                        "product_id": product_id
                    }
                }
            }

            response = self.opensearch_client.search(
                index=config.INDEX_NAME,
                body=search_body
            )

            if response['hits']['total']['value'] == 0:
                st.error(f"❌ Product with ID {product_id} not found")
                return None

            hit = response['hits']['hits'][0]
            return {
                'internal_id': hit['_id'],
                'product_id': hit['_source']['product_id'],
                'title': hit['_source']['title'],
                'description': hit['_source']['description'],
                'created_at': hit['_source']['created_at']
            }

        except Exception as e:
            st.error(f"❌ Get product failed: {str(e)}")
            return None

    def list_all_products(self, limit: int = 100) -> List[Dict]:
        """List all products"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return []

        try:
            search_body = {
                "size": limit,
                "_source": ["product_id", "title", "description", "created_at"],
                "query": {
                    "match_all": {}
                },
                "sort": [
                    {"created_at": {"order": "desc"}}
                ]
            }

            response = self.opensearch_client.search(
                index=config.INDEX_NAME,
                body=search_body
            )

            products = []
            for hit in response['hits']['hits']:
                products.append({
                    'internal_id': hit['_id'],
                    'product_id': hit['_source']['product_id'],
                    'title': hit['_source']['title'],
                    'description': hit['_source']['description'],
                    'created_at': hit['_source']['created_at']
                })

            return products

        except Exception as e:
            st.error(f"❌ List products failed: {str(e)}")
            return []

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
            logger.error(f"Image search error: {str(e)}")
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

                # Updated search body for OpenSearch Serverless
                search_body = {
                    "size": limit,
                    "_source": ["product_id", "title", "description"],
                    "query": {
                        "knn": {
                            "text_embedding": {
                                "vector": text_embedding,
                                "k": limit,
                                "method_parameters": {
                                    "ef": 100
                                }
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
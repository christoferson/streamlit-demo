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
import numpy as np


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
    #TITAN_IMAGE_MODEL_ID: str = 'amazon.titan-embed-image-v1'
    #TITAN_TEXT_MODEL_ID: str = 'amazon.titan-embed-text-v1'
    TITAN_MULTIMODAL_MODEL_ID: str = 'amazon.titan-embed-image-v1'


    # Embedding Configuration
    EMBEDDING_DIMENSION: int = 1024

    # App Configuration
    MAX_FILE_SIZE_MB: int = 10
    SUPPORTED_IMAGE_FORMATS: list = None
    SEARCH_RESULTS_LIMIT: int = 20

    # Image URL Template
    IMAGE_URL_TEMPLATE: str = os.getenv('IMAGE_URL_TEMPLATE', 'https://xxx/{code}_bar')


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
        """Create index with proper mappings for OpenSearch Serverless using Faiss engine"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        # OpenSearch Serverless index configuration with Faiss
        index_body = {
            "settings": {
                "index.knn": True,  # Enable KNN
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
                            "space_type": "l2",  # Changed from cosinesimil
                            "engine": "faiss"    # Changed from nmslib
                        }
                    },
                    "text_embedding": {
                        "type": "knn_vector",
                        "dimension": config.EMBEDDING_DIMENSION,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",  # Changed from cosinesimil
                            "engine": "faiss"    # Changed from nmslib
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
            st.success(f"✅ Created index with Faiss engine: {config.INDEX_NAME}")
            return True
        except Exception as e:
            st.error(f"❌ Index creation failed: {str(e)}")
            logger.error(f"Index creation error: {str(e)}")
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

    def check_title_exists(self, title: str) -> Dict:
        """Check if a product with the given title already exists"""
        if not self.connected:
            return {'exists': False, 'error': 'Not connected'}

        if not self.check_index_exists():
            return {'exists': False, 'error': 'Index does not exist'}

        try:
            search_body = {
                "query": {
                    "term": {
                        "title.keyword": title
                    }
                }
            }

            response = self.opensearch_client.search(
                index=config.INDEX_NAME,
                body=search_body
            )

            if response['hits']['total']['value'] > 0:
                existing_product = response['hits']['hits'][0]['_source']
                return {
                    'exists': True,
                    'product_id': existing_product['product_id'],
                    'created_at': existing_product.get('created_at', 'Unknown')
                }
            else:
                return {'exists': False}

        except Exception as e:
            logger.error(f"Check title exists error: {str(e)}")
            return {'exists': False, 'error': str(e)}

    def register_product(self, image: Image.Image, title: str, description: str) -> bool:
        """Register a new product with image and text"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        if not self.check_index_exists():
            st.error("❌ Index does not exist. Please create index first.")
            return False

        # Check if title already exists
        title_check = self.check_title_exists(title)

        if title_check.get('exists', False):
            st.error(f"❌ Product with title '{title}' already exists!")
            st.info(f"Existing Product ID: {title_check['product_id']}")
            st.info(f"Created: {title_check['created_at']}")
            st.info("💡 Please use a different title or update the existing product instead.")
            return False

        if 'error' in title_check:
            st.warning(f"⚠️ Could not check for duplicate titles: {title_check['error']}")
            # Continue with registration despite the warning

        try:
            product_id = str(uuid.uuid4())

            # Get embeddings using the multimodal model
            with st.spinner("🔄 Generating image embedding..."):
                image_embedding = self.get_image_embedding(image)

            with st.spinner("🔄 Generating text embedding..."):
                #text_embedding = self.get_text_embedding(f"{title}. {description}")
                text_embedding = self.get_text_embedding(f"{title}")

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

    def update_product(self, product_id: str, image: Image.Image = None, title: str = None, description: str = None, trade_code: str = None) -> bool:
        """Update an existing product"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        try:
            # Search for document by product_id
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

            # Get internal _id
            internal_doc_id = response['hits']['hits'][0]['_id']
            existing_doc = response['hits']['hits'][0]['_source']

            # Prepare update fields
            update_doc = {}
            update_summary = []

            if title is not None:
                update_doc['title'] = title
                update_summary.append(f"title to '{title}'")

            if description is not None:
                update_doc['description'] = description
                update_summary.append("description")

            if trade_code is not None:
                update_doc['trade_code'] = trade_code.strip()
                update_summary.append(f"trade code to '{trade_code.strip()}'")

            # Generate new embeddings if needed
            if image is not None:
                with st.spinner("🔄 Generating new image embedding..."):
                    image_embedding = self.get_image_embedding(image)
                    if image_embedding:
                        update_doc['image_embedding'] = image_embedding
                        update_summary.append("image and image embedding")

            if title is not None or description is not None:
                # If text is updated, update text embedding too
                new_title = title if title is not None else existing_doc.get('title', '')
                new_description = description if description is not None else existing_doc.get('description', '')
                new_text = f"{new_title}. {new_description}"

                with st.spinner("🔄 Generating new text embedding..."):
                    text_embedding = self.get_text_embedding(new_text)
                    if text_embedding:
                        update_doc['text_embedding'] = text_embedding
                        if "text embedding" not in " ".join(update_summary):
                            update_summary.append("text embedding")

            if not update_doc:
                st.warning("⚠️ No fields to update")
                return False

            # Update document - no refresh parameter for Vector Search Collection
            with st.spinner("💾 Updating document..."):
                update_response = self.opensearch_client.update(
                    index=config.INDEX_NAME,
                    id=internal_doc_id,
                    body={"doc": update_doc}
                )

            if update_response['result'] == 'updated':
                st.success(f"✅ Product {product_id} updated successfully!")
                st.info(f"Updated: {', '.join(update_summary)}")
                return True
            else:
                st.error("❌ Failed to update product")
                return False

        except Exception as e:
            st.error(f"❌ Update failed: {str(e)}")
            logger.error(f"Update error: {str(e)}")
            return False

    def delete_product(self, product_id: str) -> bool:
        """Delete a product by its product_id"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return False

        try:
            # Search for document by product_id
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

            # Get internal _id
            internal_doc_id = response['hits']['hits'][0]['_id']

            # Delete document - no refresh parameter
            with st.spinner("🗑️ Deleting product..."):
                delete_response = self.opensearch_client.delete(
                    index=config.INDEX_NAME,
                    id=internal_doc_id
                    # Remove refresh=True - not supported in Vector Search Collection
                )

            if delete_response['result'] == 'deleted':
                st.success(f"✅ Product {product_id} deleted successfully!")
                return True
            else:
                st.error("❌ Failed to delete product")
                return False

        except Exception as e:
            st.error(f"❌ Delete failed: {str(e)}")
            logger.error(f"Delete error: {str(e)}")
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
                'trade_code': hit['_source'].get('trade_code', ''),  # Add trade_code
                'created_at': hit['_source']['created_at']
            }

        except Exception as e:
            st.error(f"❌ Get product failed: {str(e)}")
            logger.error(f"Get product error: {str(e)}")
            return None

    def list_all_products(self, limit: int = 100) -> List[Dict]:
        """List all products with better error handling and logging"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return []

        if not self.check_index_exists():
            st.error("❌ Index does not exist. Please create index first.")
            return []

        try:
            search_body = {
                "size": limit,
                "_source": ["product_id", "title", "description", "trade_code", "created_at"],
                "query": {
                    "match_all": {}
                },
                "sort": [
                    {"created_at": {"order": "desc"}}
                ]
            }

            logger.info(f"Listing all products with limit: {limit}")

            response = self.opensearch_client.search(
                index=config.INDEX_NAME,
                body=search_body
            )

            logger.info(f"List products response: {response['hits']['total']['value']} total hits")

            products = []
            for hit in response['hits']['hits']:
                trade_code = hit['_source'].get('trade_code', '')
                image_url = self.generate_image_url_from_trade_code(trade_code) if trade_code else ""

                products.append({
                    'internal_id': hit['_id'],
                    'product_id': hit['_source']['product_id'],
                    'title': hit['_source']['title'],
                    'description': hit['_source']['description'],
                    'trade_code': trade_code,
                    'image_url': image_url,
                    'created_at': hit['_source'].get('created_at', 'Unknown')
                })

            return products

        except Exception as e:
            st.error(f"❌ List products failed: {str(e)}")
            logger.error(f"List products error: {str(e)}")
            logger.exception("List products exception:")
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

                # Try multiple search strategies
                strategies = [
                    # Strategy 1: Standard KNN
                    {
                        "size": limit,
                        "_source": ["product_id", "title", "description"],
                        "query": {
                            "knn": {
                                "image_embedding": {
                                    "vector": image_embedding,
                                    "k": limit * 3  # Try larger k
                                }
                            }
                        }
                    },
                    # Strategy 2: Script score with manual similarity
                    {
                        "size": limit,
                        "_source": ["product_id", "title", "description"],
                        "query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'image_embedding') + 1.0",
                                    "params": {"query_vector": image_embedding}
                                }
                            }
                        }
                    }
                ]

                for i, search_body in enumerate(strategies):
                    try:
                        logger.info(f"Trying image search strategy {i+1}")

                        response = self.opensearch_client.search(
                            index=config.INDEX_NAME,
                            body=search_body
                        )

                        logger.info(f"Strategy {i+1} response: {response['hits']['total']['value']} hits")

                        if response['hits']['total']['value'] > 0:
                            results = []
                            for hit in response['hits']['hits']:
                                results.append({
                                    'product_id': hit['_source']['product_id'],
                                    'title': hit['_source']['title'],
                                    'description': hit['_source']['description'],
                                    'score': hit['_score']
                                })

                            if i > 0:
                                st.info(f"🔄 Used alternative search method (strategy {i+1})")

                            return results

                    except Exception as strategy_error:
                        logger.warning(f"Strategy {i+1} failed: {strategy_error}")
                        continue

                # If all strategies fail
                st.info("🔄 Vector search similarity threshold too strict - no similar images found")
                return []

        except Exception as e:
            st.error(f"❌ Image search failed: {str(e)}")
            logger.error(f"Image search error: {str(e)}")
            return []

    def search_by_text(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for similar products using text with fallback strategies"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return []

        if not self.check_index_exists():
            st.error("❌ Index does not exist. Please create index first.")
            return []

        try:
            with st.spinner("🔍 Searching by text..."):
                # Strategy 1: Try vector search first
                text_embedding = self.get_text_embedding(query)

                if not text_embedding:
                    st.error("❌ Failed to generate text embedding")
                    return []

                logger.info(f"Generated embedding for '{query}': length={len(text_embedding)}")

                # Try vector search with larger k and no minimum score
                vector_search_body = {
                    "size": limit * 2,  # Get more results to increase chances
                    "_source": ["product_id", "title", "description"],
                    "query": {
                        "knn": {
                            "text_embedding": {
                                "vector": text_embedding,
                                "k": limit * 2  # Increase k value
                            }
                        }
                    }
                }

                logger.info(f"Trying vector search with k={limit * 2}")

                response = self.opensearch_client.search(
                    index=config.INDEX_NAME,
                    body=vector_search_body
                )

                logger.info(f"Vector search found {response['hits']['total']['value']} hits")

                # If vector search finds results, use them
                if response['hits']['total']['value'] > 0:
                    results = []
                    for hit in response['hits']['hits'][:limit]:  # Limit final results
                        results.append({
                            'product_id': hit['_source']['product_id'],
                            'title': hit['_source']['title'],
                            'description': hit['_source']['description'],
                            'score': hit['_score']
                        })
                    logger.info(f"Returning {len(results)} vector search results")
                    return results

                # Strategy 2: Fallback to traditional text search if vector search fails
                logger.info("Vector search returned 0 results, trying traditional text search")

                fallback_search_body = {
                    "size": limit,
                    "_source": ["product_id", "title", "description"],
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "title": {
                                            "query": query,
                                            "boost": 2.0
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "description": {
                                            "query": query,
                                            "boost": 1.0
                                        }
                                    }
                                },
                                {
                                    "wildcard": {
                                        "title": {
                                            "value": f"*{query.upper()}*",
                                            "boost": 1.5
                                        }
                                    }
                                },
                                {
                                    "wildcard": {
                                        "description": {
                                            "value": f"*{query.upper()}*",
                                            "boost": 0.5
                                        }
                                    }
                                }
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    "sort": [
                        {"_score": {"order": "desc"}}
                    ]
                }

                fallback_response = self.opensearch_client.search(
                    index=config.INDEX_NAME,
                    body=fallback_search_body
                )

                logger.info(f"Fallback text search found {fallback_response['hits']['total']['value']} hits")

                results = []
                for hit in fallback_response['hits']['hits']:
                    results.append({
                        'product_id': hit['_source']['product_id'],
                        'title': hit['_source']['title'],
                        'description': hit['_source']['description'],
                        'score': hit['_score']
                    })

                if results:
                    st.info("🔄 Used traditional text search (vector search found no similar results)")

                return results

        except Exception as e:
            st.error(f"❌ Text search failed: {str(e)}")
            logger.error(f"Text search error: {str(e)}")
            logger.exception("Full text search exception:")
            return []

    def search_by_title(self, title_query: str, limit: int = 10, search_mode: str = "partial") -> List[Dict]:
        """Search for products by title using different matching strategies"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return []

        if not self.check_index_exists():
            st.error("❌ Index does not exist. Please create index first.")
            return []

        try:
            with st.spinner(f"🔍 Searching by title ({search_mode} match)..."):

                # Build query based on search mode
                if search_mode == "exact":
                    # Exact match - try both approaches
                    query = {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "title": {
                                            "query": title_query,
                                            "operator": "and"
                                        }
                                    }
                                },
                                {
                                    "term": {
                                        "title.keyword": title_query
                                    }
                                }
                            ]
                        }
                    }
                elif search_mode == "fuzzy":
                    # Fuzzy match with multiple approaches
                    query = {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "title": {
                                            "query": title_query,
                                            "fuzziness": "AUTO"
                                        }
                                    }
                                },
                                {
                                    "wildcard": {
                                        "title": {
                                            "value": f"*{title_query.lower()}*",
                                            "case_insensitive": True
                                        }
                                    }
                                }
                            ]
                        }
                    }
                else:  # partial match (default)
                    # Multiple partial matching strategies
                    query = {
                        "bool": {
                            "should": [
                                # Standard match with partial terms
                                {
                                    "match": {
                                        "title": {
                                            "query": title_query,
                                            "minimum_should_match": "1"
                                        }
                                    }
                                },
                                # Wildcard search (case insensitive)
                                {
                                    "wildcard": {
                                        "title": {
                                            "value": f"*{title_query.upper()}*"
                                        }
                                    }
                                },
                                {
                                    "wildcard": {
                                        "title": {
                                            "value": f"*{title_query.lower()}*"
                                        }
                                    }
                                },
                                # Match phrase prefix
                                {
                                    "match_phrase_prefix": {
                                        "title": {
                                            "query": title_query
                                        }
                                    }
                                },
                                # Query string for more flexible matching
                                {
                                    "query_string": {
                                        "query": f"*{title_query}*",
                                        "fields": ["title"],
                                        "default_operator": "OR"
                                    }
                                }
                            ]
                        }
                    }

                search_body = {
                    "size": limit,
                    "_source": ["product_id", "title", "description", "created_at"],
                    "query": query,
                    "sort": [
                        {"_score": {"order": "desc"}},
                        {"created_at": {"order": "desc"}}
                    ]
                }

                logger.info(f"Title search query ({search_mode}): {json.dumps(search_body, indent=2)}")

                response = self.opensearch_client.search(
                    index=config.INDEX_NAME,
                    body=search_body
                )

                logger.info(f"Title search response: {response['hits']['total']['value']} hits")

                results = []
                for hit in response['hits']['hits']:
                    results.append({
                        'product_id': hit['_source']['product_id'],
                        'title': hit['_source']['title'],
                        'description': hit['_source']['description'],
                        'created_at': hit['_source'].get('created_at', 'Unknown'),
                        'score': hit['_score']
                    })

                return results

        except Exception as e:
            st.error(f"❌ Title search failed: {str(e)}")
            logger.error(f"Title search error: {str(e)}")
            logger.exception("Title search exception:")
            return []


    def list_all_products(self, limit: int = 100) -> List[Dict]:
        """List all products with better error handling and logging"""
        if not self.connected:
            st.error("❌ Not connected to AWS services")
            return []

        if not self.check_index_exists():
            st.error("❌ Index does not exist. Please create index first.")
            return []

        try:
            search_body = {
                "size": limit,
                "_source": ["product_id", "title", "description", "trade_code", "created_at"],  # Added trade_code here
                "query": {
                    "match_all": {}
                },
                "sort": [
                    {"created_at": {"order": "desc"}}
                ]
            }

            logger.info(f"Listing all products with limit: {limit}")

            response = self.opensearch_client.search(
                index=config.INDEX_NAME,
                body=search_body
            )

            logger.info(f"List products response: {response['hits']['total']['value']} total hits")

            products = []
            for hit in response['hits']['hits']:
                trade_code = hit['_source'].get('trade_code', '')
                image_url = self.generate_image_url_from_trade_code(trade_code) if trade_code else ""

                products.append({
                    'internal_id': hit['_id'],
                    'product_id': hit['_source']['product_id'],
                    'title': hit['_source']['title'],
                    'description': hit['_source']['description'],
                    'trade_code': trade_code,  # Added trade_code
                    'image_url': image_url,    # Added image_url
                    'created_at': hit['_source'].get('created_at', 'Unknown')
                })

            return products

        except Exception as e:
            st.error(f"❌ List products failed: {str(e)}")
            logger.error(f"List products error: {str(e)}")
            logger.exception("List products exception:")
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

    def debug_vector_search(self, query: str) -> Dict:
        """Debug vector search to understand what's happening"""
        if not self.connected:
            return {'error': 'Not connected'}

        try:
            # Step 1: Generate embedding
            text_embedding = self.get_text_embedding(query)

            if not text_embedding:
                return {'error': 'Failed to generate embedding'}

            # Step 2: Check if documents have embeddings
            check_embeddings_body = {
                "size": 1,
                "_source": ["product_id", "title", "text_embedding"],
                "query": {"match_all": {}}
            }

            check_response = self.opensearch_client.search(
                index=config.INDEX_NAME,
                body=check_embeddings_body
            )

            has_embeddings = False
            embedding_length = 0

            if check_response['hits']['total']['value'] > 0:
                doc = check_response['hits']['hits'][0]['_source']
                if 'text_embedding' in doc and doc['text_embedding']:
                    has_embeddings = True
                    embedding_length = len(doc['text_embedding'])

            # Step 3: Try vector search
            vector_search_body = {
                "size": 5,
                "_source": ["product_id", "title", "description"],
                "query": {
                    "knn": {
                        "text_embedding": {
                            "vector": text_embedding,
                            "k": 5
                        }
                    }
                }
            }

            try:
                vector_response = self.opensearch_client.search(
                    index=config.INDEX_NAME,
                    body=vector_search_body
                )
                vector_search_success = True
                vector_hits = vector_response['hits']['total']['value']
            except Exception as ve:
                vector_search_success = False
                vector_hits = 0
                vector_error = str(ve)

            return {
                'query': query,
                'embedding_generated': True,
                'embedding_length': len(text_embedding),
                'expected_dimension': config.EMBEDDING_DIMENSION,
                'documents_have_embeddings': has_embeddings,
                'stored_embedding_length': embedding_length,
                'vector_search_success': vector_search_success,
                'vector_hits': vector_hits,
                'vector_error': vector_error if not vector_search_success else None,
                'sample_embedding': text_embedding[:5] if text_embedding else None
            }

        except Exception as e:
            return {'error': str(e)}

    def debug_image_search(self, image: Image.Image) -> Dict:
        """Debug image search to understand what's happening"""
        if not self.connected:
            return {'error': 'Not connected'}

        try:
            # Step 1: Generate embedding for search image
            search_embedding = self.get_image_embedding(image)

            if not search_embedding:
                return {'error': 'Failed to generate search embedding'}

            # Step 2: Get stored image embedding from database
            get_doc_body = {
                "size": 1,
                "_source": ["product_id", "title", "image_embedding"],
                "query": {"match_all": {}}
            }

            doc_response = self.opensearch_client.search(
                index=config.INDEX_NAME,
                body=get_doc_body
            )

            if doc_response['hits']['total']['value'] == 0:
                return {'error': 'No documents found'}

            stored_doc = doc_response['hits']['hits'][0]['_source']
            stored_embedding = stored_doc.get('image_embedding', [])

            if not stored_embedding:
                return {'error': 'No image embedding found in stored document'}

            # Step 3: Calculate manual similarity (cosine similarity)
            import numpy as np

            search_vec = np.array(search_embedding)
            stored_vec = np.array(stored_embedding)

            # Cosine similarity
            dot_product = np.dot(search_vec, stored_vec)
            norm_search = np.linalg.norm(search_vec)
            norm_stored = np.linalg.norm(stored_vec)
            cosine_similarity = dot_product / (norm_search * norm_stored)

            # Step 4: Try vector search
            vector_search_body = {
                "size": 5,
                "_source": ["product_id", "title"],
                "query": {
                    "knn": {
                        "image_embedding": {
                            "vector": search_embedding,
                            "k": 5
                        }
                    }
                }
            }

            try:
                vector_response = self.opensearch_client.search(
                    index=config.INDEX_NAME,
                    body=vector_search_body
                )
                vector_hits = vector_response['hits']['total']['value']
                vector_scores = [hit['_score'] for hit in vector_response['hits']['hits']]
            except Exception as ve:
                vector_hits = 0
                vector_scores = []
                vector_error = str(ve)

            return {
                'search_embedding_length': len(search_embedding),
                'stored_embedding_length': len(stored_embedding),
                'cosine_similarity': float(cosine_similarity),
                'search_embedding_sample': search_embedding[:3],
                'stored_embedding_sample': stored_embedding[:3],
                'vector_search_hits': vector_hits,
                'vector_search_scores': vector_scores,
                'stored_product_title': stored_doc.get('title', 'Unknown')
            }

        except Exception as e:
            return {'error': str(e)}

    def generate_image_url_from_trade_code(self, trade_code: str) -> str:
        """Generate image URL from trade code using template"""
        if not trade_code:
            return ""

        try:
            # Convert format: 1203A750.020 -> 1203A750_020
            if '.' in trade_code:
                formatted_code = trade_code.replace('.', '_')
            else:
                formatted_code = trade_code

            # Use template from config
            url = config.IMAGE_URL_TEMPLATE.format(trade_code=formatted_code)
            return url

        except Exception as e:
            logger.error(f"Error generating image URL from trade code '{trade_code}': {str(e)}")
            return ""
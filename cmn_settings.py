import os
from os.path import join, dirname
from dotenv import load_dotenv

load_dotenv(verbose=True)

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

AWS_REGION = os.environ.get("AWS_REGION")
AWS_BEDROCK_GUARDRAIL_IDENTIFIER = os.environ.get("AWS_BEDROCK_GUARDRAIL_IDENTIFIER")
AWS_BEDROCK_GUARDRAIL_VERSION = os.environ.get("AWS_BEDROCK_GUARDRAIL_VERSION")
APP_LOG_GROUP_METRICS_INVOCATIONS = os.environ.get("APP_LOG_GROUP_METRICS_INVOCATIONS")

CONVERSE_CH_SECRET_KEY = os.environ.get("CONVERSE_CH_SECRET_KEY")
CONVERSE_CH_ALGORITHM = os.environ.get("CONVERSE_CH_ALGORITHM")
CONVERSE_CH_SALT = os.environ.get("CONVERSE_CH_SALT")

VECTOR_REGION_ID = os.getenv("VECTOR_REGION_ID", "us-east-1")
VECTOR_EMBEDDING_MODEL_ID = os.getenv("VECTOR_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0") 
VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME", "s3-vector-myvectorbucket-xxx")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "s3-vector-myvectorindex-xxxx")


# Knowledge Base Configuration
CMN_KB_KNOWLEDGE_BASE_ID = os.getenv("CMN_KB_KNOWLEDGE_BASE_ID", "YOUR_KB_ID_HERE")
CMN_KB_DATA_SOURCE_ID = os.getenv("CMN_KB_DATA_SOURCE_ID", "YOUR_DATA_SOURCE_ID_HERE")
CMN_KB_DOCUMENT_BUCKET_NAME = os.getenv("CMN_KB_DOCUMENT_BUCKET_NAME", "your-document-bucket-name")
CMN_KB_VECTOR_BUCKET_NAME = os.getenv("CMN_KB_VECTOR_BUCKET_NAME", "your-vector-bucket-name")
CMN_KB_VECTOR_INDEX_NAME = os.getenv("CMN_KB_VECTOR_INDEX_NAME", "your-vector-index-name")
CMN_KB_KB_METADATA_TABLE = os.getenv("CMN_KB_KB_METADATA_TABLE", "your-kb-metadata-table")
CMN_KB_DOC_METADATA_TABLE = os.getenv("CMN_KB_DOC_METADATA_TABLE", "your-doc-metadata-table")
CMN_KB_MODEL_ARN = os.getenv("CMN_KB_MODEL_ARN", "arn:aws:bedrock:xxx::foundation-model/xxx")
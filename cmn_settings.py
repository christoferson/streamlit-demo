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
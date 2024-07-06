import os
import boto3
from typing import List
import cmn_settings

AWS_REGION = cmn_settings.AWS_REGION

bedrock_agent = boto3.client('bedrock-agent', region_name=AWS_REGION)

def list_knowledge_bases() -> List[str]:

    response = bedrock_agent.list_knowledge_bases(maxResults=20) #response = bedrock_agent.list_knowledge_bases(maxResults = 5)

    kb_id_list = []
    for i, knowledgeBaseSummary in enumerate(response['knowledgeBaseSummaries']):
        kb_id = knowledgeBaseSummary['knowledgeBaseId']
        name = knowledgeBaseSummary['name']
        description = knowledgeBaseSummary['description']
        status = knowledgeBaseSummary['status']
        updatedAt = knowledgeBaseSummary['updatedAt']
        #print(f"{i} RetrievalResult: {kb_id} {name} {description} {status} {updatedAt}")
        kb_id_list.append(f"{kb_id} {name}")
    
    if not kb_id_list:
        kb_id_list = ["EMPTY EMPTY"]
    
    return kb_id_list

def list_knowledge_bases_with_options(kb_filter_list:List[str]) -> List[str]:

    response = bedrock_agent.list_knowledge_bases(maxResults=20) #response = bedrock_agent.list_knowledge_bases(maxResults = 5)

    kb_id_list = []
    for i, knowledgeBaseSummary in enumerate(response['knowledgeBaseSummaries']):
        kb_id = knowledgeBaseSummary['knowledgeBaseId']
        kb_name = knowledgeBaseSummary['name']
        description = knowledgeBaseSummary['description']
        status = knowledgeBaseSummary['status']
        updatedAt = knowledgeBaseSummary['updatedAt']
        #print(f"{i} RetrievalResult: {kb_id} {name} {description} {status} {updatedAt}")
        res = [kb_filter for kb_filter in kb_filter_list if(kb_filter in kb_name)]
        if res:
            kb_id_list.append(f"{kb_id} {kb_name}")
    
    if not kb_id_list:
        kb_id_list = ["EMPTY EMPTY"]
    
    return kb_id_list



def list_knowledge_bases_with_options_v2(kb_filter_list:List[str]) -> List[str]:

    response = bedrock_agent.list_knowledge_bases(maxResults=20)

    kb_id_list = []
    for i, knowledgeBaseSummary in enumerate(response['knowledgeBaseSummaries']):
        kb_id = knowledgeBaseSummary['knowledgeBaseId']
        kb_name = knowledgeBaseSummary['name']
        description = knowledgeBaseSummary['description']
        status = knowledgeBaseSummary['status']
        updatedAt = knowledgeBaseSummary['updatedAt']
        #print(f"{i} RetrievalResult: {kb_id} {name} {description} {status} {updatedAt}")
        res = [kb_filter for kb_filter in kb_filter_list if(kb_filter in kb_name)]
        if res:
            kb_id_list.append(f"{kb_id} {kb_name}")
    
    return kb_id_list

def bedrock_list_models(bedrock):
    response = bedrock.list_foundation_models(byOutputModality="TEXT")

    for item in response["modelSummaries"]:
        print(item['modelId'])
import boto3
import cmn_settings
from datetime import datetime, timedelta, timezone
import json

AWS_REGION = cmn_settings.AWS_REGION

cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)

# def list_available_models(namespace='AWS/Bedrock', metric_name='Invocations'):
#     paginator = cloudwatch.get_paginator('list_metrics')
#     models = set()
#     for page in paginator.paginate(Namespace=namespace, MetricName=metric_name):
#         for metric in page['Metrics']:
#             for dimension in metric['Dimensions']:
#                 if dimension['Name'] == 'ModelId':
#                     models.add(dimension['Value'])
#     return list(models)


def cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='Invocations', 
                          start_time:datetime=datetime.now() - timedelta(days=7),
                          end_time:datetime=datetime.now(), aggregate_stat="Sum"):
    print(f"**********************cloudwatch_get_metric {metric_namespace} {metric_name}")
    metric_data = {
        'Values': [],
        'Timestamps': [],
        'NextToken': None,
        #'Labels': [],
    }
        
    get_metric_data_response = _call(metric_namespace, metric_name, start_time, end_time, next_token="", aggregate_stat=aggregate_stat)
    #print("Response structure:")
    #print(json.dumps(get_metric_data_response, indent=2, default=str))
    metric_data_list = get_metric_data_response['MetricDataResults']
    metric_data_values = metric_data_list[0]['Values']
    metric_data_timestamps = metric_data_list[0]['Timestamps']
    #metric_data_labels = metric_data_list[0].get('Labels', [])
    metric_data['Values'].extend(metric_data_values)
    metric_data['Timestamps'].extend(metric_data_timestamps)
    #metric_data['Labels'].extend(metric_data_labels)
    next_token = ""
    if 'NextToken' in get_metric_data_response:
        next_token = get_metric_data_response['NextToken']
        while next_token != "":
            print(f"**********************cloudwatch_get_metric {metric_name} Next={next_token}")
            get_metric_data_response = _call(metric_namespace, metric_name, start_time, end_time, next_token=next_token, aggregate_stat=aggregate_stat)
            metric_data_list = get_metric_data_response['MetricDataResults']
            metric_data_values = metric_data_list[0]['Values']
            metric_data_timestamps = metric_data_list[0]['Timestamps']
            #metric_data_labels = metric_data_list[0].get('Labels', [])
            metric_data['Values'].extend(metric_data_values)
            metric_data['Timestamps'].extend(metric_data_timestamps)
            #metric_data['Labels'].extend(metric_data_labels)
            next_token = ""
            if 'NextToken' in get_metric_data_response:
                next_token = get_metric_data_response['NextToken']

    return metric_data


def _call(metric_namespace, metric_name, start_time, end_time, next_token = "", aggregate_stat="Sum"):
    metric_data_query_list = [
            {
                'Id': 'claude_3_sonnet_invocations',
                'MetricStat': {
                    'Metric': {
                        'Namespace': metric_namespace,
                        'MetricName': metric_name,
                        'Dimensions': [
                            # {
                            #     'Name': 'ModelId',
                            #     'Value': 'anthropic.claude-3-sonnet-20240229-v1:0'
                            # },
                            # {
                            #     'Name': 'ModelId',
                            #     'Value': 'anthropic.claude-3-5-sonnet-20240620-v1:0'
                            # },
                            # {
                            #     'Name': 'ModelId',
                            #     'Value': 'stability.stable-diffusion-xl-v1'
                            # },
                        ]
                    },
                    'Period': 86400, #Seconds
                    'Stat': aggregate_stat,
                    #'Unit': 'Count/Second' #''Seconds'|'Microseconds'|'Milliseconds'|'Bytes'|'Kilobytes'|'Megabytes'|'Gigabytes'|'Terabytes'|'Bits'|'Kilobits'|'Megabits'|'Gigabits'|'Terabits'|'Percent'|'Count'|'Bytes/Second'|'Kilobytes/Second'|'Megabytes/Second'|'Gigabytes/Second'|'Terabytes/Second'|'Bits/Second'|'Kilobits/Second'|'Megabits/Second'|'Gigabits/Second'|'Terabits/Second'|'Count/Second'|'None'
                },
                #'Expression': 'string',
                #'Label': 'string',
                #'ReturnData': True|False,
                #'Period': 123,
                #'AccountId': 'string'
            },
        ]
    
    
    if next_token != "":
        get_metric_data_response = cloudwatch.get_metric_data(
            MetricDataQueries=metric_data_query_list,
            StartTime=start_time, #datetime(2024, 7, 2),
            EndTime=end_time, #datetime(2024, 7, 11),
            NextToken=next_token,
            ScanBy= 'TimestampAscending', #'TimestampDescending'|'TimestampAscending',
            #MaxDatapoints=123,
            #LabelOptions={
            #    'Timezone': 'string'
            #}
        )
    else:
        get_metric_data_response = cloudwatch.get_metric_data(
            MetricDataQueries=metric_data_query_list,
            StartTime=start_time, #datetime(2024, 7, 2),
            EndTime=end_time, #datetime(2024, 7, 11),
            ScanBy= 'TimestampAscending', #'TimestampDescending'|'TimestampAscending',
            #MaxDatapoints=123,
            #LabelOptions={
            #    'Timezone': 'string'
            #}
        )
    return get_metric_data_response



######



def cloudwatch_put_metric(metric_namespace='App/Chat', metric_name='UserInvocation', metric_value=1,
                          dimensions = []):
    
    now = datetime.now(timezone.utc)
    metric_data = [
            {
                'MetricName': metric_name,
                'Dimensions': dimensions,
                'Timestamp': now,
                'Value': metric_value,
                #'StorageResolution': 1,
                'Unit': 'Count',
            },
        ]
    
    response = cloudwatch.put_metric_data(Namespace=metric_namespace, MetricData=metric_data)
    print(response)
    



###

def cloudwatch_get_metric_with_dimensions(
            metric_namespace='AWS/Bedrock', 
            metric_name='Invocations', 
            metric_dimensions = [],
            start_time:datetime=datetime.now() - timedelta(days=7),
            end_time:datetime=datetime.now()):
    print(f"**********************cloudwatch_get_metric_with_dimensions {metric_namespace} {metric_name} {metric_dimensions}")
    metric_data = {
        'Values': [],
        'Timestamps': [],
        'NextToken': None
    }
        
    get_metric_data_response = _cloudwatch_get_metric_with_dimensions(metric_namespace, metric_name, metric_dimensions, 
                                                                      start_time, end_time, next_token="")
    metric_data_list = get_metric_data_response['MetricDataResults']
    metric_data_values = metric_data_list[0]['Values']
    metric_data_timestamps = metric_data_list[0]['Timestamps']
    metric_data['Values'].extend(metric_data_values)
    metric_data['Timestamps'].extend(metric_data_timestamps)
    next_token = ""
    if 'NextToken' in get_metric_data_response:
        next_token = get_metric_data_response['NextToken']
        while next_token != "":
            print(f"**********************cloudwatch_get_metric {metric_name} Next={next_token}")
            get_metric_data_response = _cloudwatch_get_metric_with_dimensions(metric_namespace, metric_name, metric_dimensions, 
                                                                      start_time, end_time, next_token=next_token)
            metric_data_list = get_metric_data_response['MetricDataResults']
            metric_data_values = metric_data_list[0]['Values']
            metric_data_timestamps = metric_data_list[0]['Timestamps']
            metric_data['Values'].extend(metric_data_values)
            metric_data['Timestamps'].extend(metric_data_timestamps)
            next_token = ""
            if 'NextToken' in get_metric_data_response:
                next_token = get_metric_data_response['NextToken']

    return metric_data


def _cloudwatch_get_metric_with_dimensions(metric_namespace, metric_name, metric_dimensions, start_time, end_time, next_token = ""):
    metric_data_query_list = [
            {
                'Id': 'claude_3_sonnet_invocations',
                'MetricStat': {
                    'Metric': {
                        'Namespace': metric_namespace,
                        'MetricName': metric_name,
                        'Dimensions': metric_dimensions,
                    },
                    'Period': 86400, #Seconds
                    'Stat': 'Sum',
                    #'Unit': 'Count/Second' #''Seconds'|'Microseconds'|'Milliseconds'|'Bytes'|'Kilobytes'|'Megabytes'|'Gigabytes'|'Terabytes'|'Bits'|'Kilobits'|'Megabits'|'Gigabits'|'Terabits'|'Percent'|'Count'|'Bytes/Second'|'Kilobytes/Second'|'Megabytes/Second'|'Gigabytes/Second'|'Terabytes/Second'|'Bits/Second'|'Kilobits/Second'|'Megabits/Second'|'Gigabits/Second'|'Terabits/Second'|'Count/Second'|'None'
                },
                #'Expression': 'string',
                #'Label': 'string',
                #'ReturnData': True|False,
                #'Period': 123,
                #'AccountId': 'string'
            },
        ]
    if next_token != "":
        get_metric_data_response = cloudwatch.get_metric_data(
            MetricDataQueries=metric_data_query_list,
            StartTime=start_time, #datetime(2024, 7, 2),
            EndTime=end_time, #datetime(2024, 7, 11),
            NextToken=next_token,
            ScanBy= 'TimestampAscending', #'TimestampDescending'|'TimestampAscending',
            #MaxDatapoints=123,
            #LabelOptions={
            #    'Timezone': 'string'
            #}
        )
    else:
        get_metric_data_response = cloudwatch.get_metric_data(
            MetricDataQueries=metric_data_query_list,
            StartTime=start_time, #datetime(2024, 7, 2),
            EndTime=end_time, #datetime(2024, 7, 11),
            ScanBy= 'TimestampAscending', #'TimestampDescending'|'TimestampAscending',
            #MaxDatapoints=123,
            #LabelOptions={
            #    'Timezone': 'string'
            #}
        )

    #get_metric_data_response_str = json.dumps(get_metric_data_response, indent=2)
    print(get_metric_data_response)
    return get_metric_data_response



#####



###

def cloudwatch_get_metric_expression(
            metric_namespace='AWS/Bedrock', 
            metric_name='Invocations',
            start_time:datetime=datetime.now() - timedelta(days=7),
            end_time:datetime=datetime.now()):
    print(f"**********************cloudwatch_get_metric_expression {metric_namespace} {metric_name}")
    metric_data = {
        'Values': [],
        'Timestamps': [],
        'NextToken': None
    }
        
    get_metric_data_response = _cloudwatch_get_metric_expression(metric_namespace, metric_name, 
                                                                      start_time, end_time, next_token="")
    metric_data_list = get_metric_data_response['MetricDataResults']
    if len(metric_data_list) > 0:
        metric_data_values = metric_data_list[0]['Values']
        metric_data_timestamps = metric_data_list[0]['Timestamps']
        metric_data['Values'].extend(metric_data_values)
        metric_data['Timestamps'].extend(metric_data_timestamps)
        next_token = ""
        if 'NextToken' in get_metric_data_response:
            next_token = get_metric_data_response['NextToken']
            while next_token != "":
                print(f"**********************cloudwatch_get_metric_expression {metric_name} Next={next_token}")
                get_metric_data_response = _cloudwatch_get_metric_expression(metric_namespace, metric_name,
                                                                        start_time, end_time, next_token=next_token)
                metric_data_list = get_metric_data_response['MetricDataResults']
                metric_data_values = metric_data_list[0]['Values']
                metric_data_timestamps = metric_data_list[0]['Timestamps']
                metric_data['Values'].extend(metric_data_values)
                metric_data['Timestamps'].extend(metric_data_timestamps)
                next_token = ""
                if 'NextToken' in get_metric_data_response:
                    next_token = get_metric_data_response['NextToken']

    return metric_data


def _cloudwatch_get_metric_expression(
        metric_namespace, metric_name, start_time, end_time, next_token = ""):
    expr = '{{App/Chat, UserInvocation}} MetricName="UserInvocation"'
    expr = '{App/Chat, UserInvocation} MetricName="UserInvocation" Name="User" Value="Fred"'
    expr = '{App/Chat, UserInvocation} MetricName="UserInvocation"'
    metric_data_query_list = [
            {
                'Id': 'claude_3_sonnet_invocations',
                
                'Expression': f"SEARCH('{expr}', 'Sum', 3600)",
                'Label': 'string',
                'ReturnData': True,
                'Period': 86400,
                #'AccountId': 'string'
            },
        ]
    
    metric_data_query_list_json = json.dumps(metric_data_query_list, indent=2)
    print(metric_data_query_list_json)

    if next_token != "":
        get_metric_data_response = cloudwatch.get_metric_data(
            MetricDataQueries=metric_data_query_list,
            StartTime=start_time, #datetime(2024, 7, 2),
            EndTime=end_time, #datetime(2024, 7, 11),
            NextToken=next_token,
            ScanBy= 'TimestampAscending', #'TimestampDescending'|'TimestampAscending',
            #MaxDatapoints=123,
            #LabelOptions={
            #    'Timezone': 'string'
            #}
        )
    else:
        get_metric_data_response = cloudwatch.get_metric_data(
            MetricDataQueries=metric_data_query_list,
            StartTime=start_time, #datetime(2024, 7, 2),
            EndTime=end_time, #datetime(2024, 7, 11),
            ScanBy= 'TimestampAscending', #'TimestampDescending'|'TimestampAscending',
            #MaxDatapoints=123,
            #LabelOptions={
            #    'Timezone': 'string'
            #}
        )

    get_metric_data_response_json = json.dumps(get_metric_data_response, indent=2)
    print(get_metric_data_response_json)

    return get_metric_data_response



#-------------------
import boto3
import cmn_settings
from datetime import datetime, timedelta

AWS_REGION = cmn_settings.AWS_REGION

cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)

def cloudwatch_get_metric(metric_name='Invocations', 
                          start_time:datetime=datetime.now() - timedelta(days=7),
                          end_time:datetime=datetime.now()):
    print(f"**********************cloudwatch_get_metric {metric_name}")
    metric_data = {

    }
        
    get_metric_data_response = _call(metric_name, start_time, end_time)

    #print(get_metric_data_response)
    metric_data_list = get_metric_data_response['MetricDataResults']
    #for metric in metric_data_list:
    #    st.markdown(metric['Timestamps'])

    metric_data_values = metric_data_list[0]['Values']
    metric_data_timestamps = metric_data_list[0]['Timestamps']
    next_token = None
    if 'NextToken' in get_metric_data_response:
        next_token = get_metric_data_response['NextToken']

    metric_data = {
        'Values': metric_data_values,
        'Timestamps': metric_data_timestamps,
        'NextToken': next_token
    }

    return metric_data


def _call(metric_name, start_time, end_time):
    get_metric_data_response = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'claude_3_sonnet_invocations',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/Bedrock',
                        'MetricName': metric_name,
                        'Dimensions': [
                            #{
                            #    'Name': 'ModelId',
                            #    'Value': 'anthropic.claude-3-sonnet-20240229-v1:0'
                            #},
                        ]
                    },
                    'Period': 60,
                    'Stat': 'Sum',
                    #'Unit': 'Count/Second' #''Seconds'|'Microseconds'|'Milliseconds'|'Bytes'|'Kilobytes'|'Megabytes'|'Gigabytes'|'Terabytes'|'Bits'|'Kilobits'|'Megabits'|'Gigabits'|'Terabits'|'Percent'|'Count'|'Bytes/Second'|'Kilobytes/Second'|'Megabytes/Second'|'Gigabytes/Second'|'Terabytes/Second'|'Bits/Second'|'Kilobits/Second'|'Megabits/Second'|'Gigabits/Second'|'Terabits/Second'|'Count/Second'|'None'
                },
                #'Expression': 'string',
                #'Label': 'string',
                #'ReturnData': True|False,
                #'Period': 123,
                #'AccountId': 'string'
            },
        ],
        StartTime=start_time, #datetime(2024, 7, 2),
        EndTime=end_time, #datetime(2024, 7, 11),
        #NextToken='string',
        ScanBy= 'TimestampAscending', #'TimestampDescending'|'TimestampAscending',
        #MaxDatapoints=123,
        #LabelOptions={
        #    'Timezone': 'string'
        #}
    )

    return get_metric_data_response

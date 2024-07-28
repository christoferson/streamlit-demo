import boto3
import cmn_settings
import streamlit as st
import numpy as np
import pandas as pd
import cmn.cloudwatch_metrics_lib

from datetime import datetime

AWS_REGION = cmn_settings.AWS_REGION

client = boto3.client("cloudwatch", region_name=AWS_REGION)

def list_metrics():
    list_metrics_response = client.list_metrics(Namespace='AWS/Bedrock', MetricName='Invocations')
    for metric in list_metrics_response['Metrics']:
        st.markdown(metric)

@st.cache_data
def bedrock_cloudwatch_get_metric(metric_name):
    metric_data = cmn.cloudwatch_metrics_lib.cloudwatch_get_metric(metric_name)
    return metric_data


get_metric_data_response = client.get_metric_data(
    MetricDataQueries=[
        {
            'Id': 'claude_3_sonnet_invocations',
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/Bedrock',
                    'MetricName': 'Invocations',
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
    StartTime=datetime(2024, 7, 2),
    EndTime=datetime(2024, 7, 11),
    #NextToken='string',
    ScanBy= 'TimestampAscending', #'TimestampDescending'|'TimestampAscending',
    #MaxDatapoints=123,
    #LabelOptions={
    #    'Timezone': 'string'
    #}
)

#print(get_metric_data_response)
metric_data_list = get_metric_data_response['MetricDataResults']
#for metric in metric_data_list:
#    st.markdown(metric['Timestamps'])

metric_data_values = metric_data_list[0]['Values']
metric_data_timestamps = metric_data_list[0]['Timestamps']

#st.markdown(np.random.randn(20, 3))

#chart_data = pd.DataFrame(, columns=["Count"])

#st.line_chart(
#   chart_data, x="Count", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional
#)

df = pd.DataFrame({
    'Timestamp': metric_data_timestamps,
    'Value': metric_data_values
})

# Set the Streamlit app title
st.title("CloudWatch Metric Data")

# Display the DataFrame in Streamlit
st.write("DataFrame:", df)

# Plot the DataFrame as a line chart
#st.line_chart(df.set_index('Timestamp'))
st.markdown("Invocations")
st.line_chart(df, x='Timestamp', y='Value', color=["#FF0000"], x_label='Time', y_label='Count')

st.divider()

metric_data_input_token_count = bedrock_cloudwatch_get_metric(metric_name='InputTokenCount')
metric_data_input_token_count_df = pd.DataFrame({
    'Timestamp': metric_data_input_token_count['Timestamps'],
    'Value': metric_data_input_token_count['Values'],
})
st.line_chart(metric_data_input_token_count_df, x='Timestamp', y='Value', color=["#FF0000"], x_label='Time', y_label='Count')
import boto3
import cmn_settings
import streamlit as st
import numpy as np
import pandas as pd
import cmn.cloudwatch_metrics_lib

from datetime import datetime, timedelta

AWS_REGION = cmn_settings.AWS_REGION

client = boto3.client("cloudwatch", region_name=AWS_REGION)

def list_metrics():
    list_metrics_response = client.list_metrics(Namespace='AWS/Bedrock', MetricName='Invocations')
    for metric in list_metrics_response['Metrics']:
        st.markdown(metric)

@st.cache_data(show_spinner='Loading Metrics')
def bedrock_cloudwatch_get_metric(metric_name, start_time, end_time):
    metric_data = cmn.cloudwatch_metrics_lib.cloudwatch_get_metric(metric_name, start_time, end_time)
    print(f"NextToken: {metric_data['NextToken']}")
    return metric_data

start_time = datetime.now() - timedelta(weeks=12) #datetime(2024, 7, 2) #datetime.now() - timedelta(days=7),
end_time = datetime.now() #datetime(2024, 7, 11) #datetime.now() - timedelta(days=1),

st.markdown("Invocations")
metric_data_invocation_count = bedrock_cloudwatch_get_metric(metric_name='Invocations', start_time=start_time, end_time=end_time)
metric_data_invocation_count_df = pd.DataFrame({
    'Timestamp': metric_data_invocation_count['Timestamps'],
    'Value': metric_data_invocation_count['Values'],
})
st.dataframe(metric_data_invocation_count_df)
metric_data_invocation_count_df['Date'] = metric_data_invocation_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
st.line_chart(metric_data_invocation_count_df, x='Timestamp', y='Value', color=["#FF0000"], x_label='Time', y_label='Count')
st.bar_chart(metric_data_invocation_count_df, x='Timestamp', y='Value', color=["#FF0000"])
st.divider()
st.dataframe(metric_data_invocation_count_df)
st.line_chart(metric_data_invocation_count_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Count')
st.bar_chart(metric_data_invocation_count_df, x='Date', y='Value', color=["#FF0000"])


st.markdown("InputTokenCount")
metric_data_input_token_count = bedrock_cloudwatch_get_metric(metric_name='InputTokenCount', start_time=start_time, end_time=end_time)
metric_data_input_token_count_df = pd.DataFrame({
    'Timestamp': metric_data_input_token_count['Timestamps'],
    'InputTokenCount': metric_data_input_token_count['Values'],
})
st.line_chart(metric_data_input_token_count_df, x='Timestamp', y='InputTokenCount', color=["#FF0000"], x_label='Time', y_label='Count')

st.markdown("OutputTokenCount")
metric_data_output_token_count = bedrock_cloudwatch_get_metric(metric_name='OutputTokenCount', start_time=start_time, end_time=end_time)
metric_data_output_token_count_df = pd.DataFrame({
    'Timestamp': metric_data_output_token_count['Timestamps'],
    'OutputTokenCount': metric_data_output_token_count['Values'],
})
st.line_chart(metric_data_output_token_count_df, x='Timestamp', y='OutputTokenCount', color=["#FF0000"], x_label='Time', y_label='Count')

st.markdown("TokenCount")
metric_data_token_count_df = pd.merge(metric_data_input_token_count_df, metric_data_output_token_count_df, on='Timestamp', how='outer')
#metric_data_token_count_df.set_index('Timestamp', inplace=True)
st.line_chart(metric_data_token_count_df, x='Timestamp', y=['InputTokenCount', 'OutputTokenCount'], color=["#FF0000", "#FF00FF"])

st.divider()

chart_data = pd.DataFrame(np.random.randn(200, 3), columns=["a", "b", "c"])

st.vega_lite_chart(
   chart_data,
   {
       "mark": {"type": "circle", "tooltip": True},
       "encoding": {
           "x": {"field": "a", "type": "quantitative"},
           "y": {"field": "b", "type": "quantitative"},
           "size": {"field": "c", "type": "quantitative"},
           "color": {"field": "c", "type": "quantitative"},
       },
   },
)

st.divider()

# Group by week and sum the InputTokenCount
#metric_data_input_token_count_df['Week'] = metric_data_input_token_count_df['Timestamp'].dt.strftime('%Y-%U')
#metric_data_input_token_count_weekly_df = metric_data_input_token_count_df.groupby('Week', as_index=False)['InputTokenCount'].sum()
#st.dataframe(metric_data_input_token_count_weekly_df)
# Plot the weekly aggregated data
#st.line_chart(metric_data_input_token_count_weekly_df, x='Week', y='InputTokenCount', color=["#FF0000"], x_label='Week', y_label='Total Count')


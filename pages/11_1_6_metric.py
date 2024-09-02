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
def bedrock_cloudwatch_get_metric(metric_namespace, metric_name, start_time, end_time):
    metric_data = cmn.cloudwatch_metrics_lib.cloudwatch_get_metric(metric_namespace, metric_name, start_time, end_time)
    #print(f"NextToken: {metric_data['NextToken']}")
    return metric_data

@st.cache_data(show_spinner='Loading Metrics')
def bedrock_cloudwatch_get_metric_with_dimensions(metric_namespace, metric_name, metric_dimensions, start_time, end_time):
    metric_data = cmn.cloudwatch_metrics_lib.cloudwatch_get_metric_with_dimensions(metric_namespace, metric_name, metric_dimensions, start_time, end_time)
    #print(f"NextToken: {metric_data['NextToken']}")
    return metric_data

@st.cache_data(show_spinner='Loading Metrics')
def bedrock_cloudwatch_get_metric_expressoin(metric_namespace, metric_name, start_time, end_time):
    metric_data = cmn.cloudwatch_metrics_lib.cloudwatch_get_metric_expression(metric_namespace, metric_name, start_time, end_time)
    #print(f"NextToken: {metric_data['NextToken']}")
    return metric_data


start_time = datetime.now() - timedelta(weeks=15) #datetime(2024, 7, 2) #datetime.now() - timedelta(days=7),
end_time = datetime.now() #datetime(2024, 7, 11) #datetime.now() - timedelta(days=1),

st.markdown(f"##### :green[{start_time} to {end_time}]")



# Group by week and sum the InputTokenCount
# Plot the weekly aggregated data
#st.line_chart(metric_data_input_token_count_weekly_df, x='Week', y='InputTokenCount', color=["#FF0000"], x_label='Week', y_label='Total Count')



tab1, tab2, tab3, tab4, tab5 = st.tabs(["Invocation", "InputToken", "OutputToken", "OutputImage", "UserInvocation"])

with tab1:
    st.markdown("##### :blue[Invocations by Date]")

    metric_data_invocation_count = bedrock_cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='Invocations', start_time=start_time, end_time=end_time)
    metric_data_invocation_count_df = pd.DataFrame({
        'Timestamp': metric_data_invocation_count['Timestamps'],
        'Value': metric_data_invocation_count['Values'],
    })

    metric_data_invocation_count_df['Date'] = metric_data_invocation_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
    # Group by date and sum the values
    metric_data_invocation_count_by_date_df = metric_data_invocation_count_df.groupby('Date').agg({'Value': 'sum'}).reset_index()


    st.dataframe(metric_data_invocation_count_by_date_df, use_container_width=True)
    st.line_chart(metric_data_invocation_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Count')
    st.bar_chart(metric_data_invocation_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Count')

    st.markdown("##### :blue[Invocations by Week]")
    metric_data_invocation_count_df['Week'] = metric_data_invocation_count_df['Timestamp'].dt.strftime('%Y-%W')
    metric_data_invocation_count_weekly_df = metric_data_invocation_count_df.groupby('Week', as_index=False)['Value'].sum()
    st.dataframe(metric_data_invocation_count_weekly_df, use_container_width=True)
    st.line_chart(metric_data_invocation_count_weekly_df, x='Week', y='Value', color=["#FF0000"], x_label='Week', y_label='Count')
    st.bar_chart(metric_data_invocation_count_weekly_df, x='Week', y='Value', color=["#FF0000"], x_label='Week', y_label='Count')


with tab2:
    st.markdown("##### :blue[Input Token by Date]")
    metric_data_input_token_count = bedrock_cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='InputTokenCount', start_time=start_time, end_time=end_time)
    metric_data_input_token_count_df = pd.DataFrame({
        'Timestamp': metric_data_input_token_count['Timestamps'],
        'InputTokenCount': metric_data_input_token_count['Values'],
    })
    metric_data_input_token_count_df['Date'] = metric_data_input_token_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
    # Group by date and sum the values
    metric_data_input_token_count_by_date_df = metric_data_input_token_count_df.groupby('Date').agg({'InputTokenCount': 'sum'}).reset_index()

    st.dataframe(metric_data_input_token_count_by_date_df, use_container_width=True)
    st.line_chart(metric_data_input_token_count_by_date_df, x='Date', y='InputTokenCount', color=["#FF0000"], x_label='Date', y_label='InputTokenCount')
    st.bar_chart(metric_data_input_token_count_by_date_df, x='Date', y='InputTokenCount', color=["#FF0000"], x_label='Date', y_label='InputTokenCount')


    st.markdown("##### :blue[Input Token by Week]")
    metric_data_input_token_count_df['Week'] = metric_data_input_token_count_df['Timestamp'].dt.strftime('%Y-%W')
    metric_data_input_token_count_weekly_df = metric_data_input_token_count_df.groupby('Week', as_index=False)['InputTokenCount'].sum()
    st.dataframe(metric_data_input_token_count_weekly_df, use_container_width=True)
    st.line_chart(metric_data_input_token_count_weekly_df, x='Week', y='InputTokenCount', color=["#FF0000"], x_label='Week', y_label='InputTokenCount')
    st.bar_chart(metric_data_input_token_count_weekly_df, x='Week', y='InputTokenCount', color=["#FF0000"], x_label='Week', y_label='InputTokenCount')

with tab3:
    st.markdown("##### :blue[Output Token by Date]")
    metric_data_output_token_count = bedrock_cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='OutputTokenCount', start_time=start_time, end_time=end_time)
    metric_data_output_token_count_df = pd.DataFrame({
        'Timestamp': metric_data_output_token_count['Timestamps'],
        'OutputTokenCount': metric_data_output_token_count['Values'],
    })

    metric_data_output_token_count_df['Date'] = metric_data_output_token_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
    metric_data_output_token_count_by_date_df = metric_data_output_token_count_df.groupby('Date').agg({'OutputTokenCount': 'sum'}).reset_index()

    st.dataframe(metric_data_output_token_count_by_date_df, use_container_width=True)
    st.line_chart(metric_data_output_token_count_by_date_df, x='Date', y='OutputTokenCount', color=["#FF0000"], x_label='Date', y_label='OutputTokenCount')
    st.bar_chart(metric_data_output_token_count_by_date_df, x='Date', y='OutputTokenCount', color=["#FF0000"], x_label='Date', y_label='OutputTokenCount')

    st.markdown("##### :blue[Output Token by Week]")
    metric_data_output_token_count_df['Week'] = metric_data_output_token_count_df['Timestamp'].dt.strftime('%Y-%W')
    metric_data_output_token_count_weekly_df = metric_data_output_token_count_df.groupby('Week', as_index=False)['OutputTokenCount'].sum()
    st.dataframe(metric_data_output_token_count_weekly_df, use_container_width=True)
    st.line_chart(metric_data_output_token_count_weekly_df, x='Week', y='OutputTokenCount', color=["#FF0000"], x_label='Week', y_label='OutputTokenCount')
    st.bar_chart(metric_data_output_token_count_weekly_df, x='Week', y='OutputTokenCount', color=["#FF0000"], x_label='Week', y_label='OutputTokenCount')


with tab4:
    st.markdown("##### :blue[Output Image]")

    metric_data_output_image_count = bedrock_cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='OutputImageCount', start_time=start_time, end_time=end_time)
    metric_data_output_image_count_df = pd.DataFrame({
        'Timestamp': metric_data_output_image_count['Timestamps'],
        'Value': metric_data_output_image_count['Values'],
    })
    st.dataframe(metric_data_output_image_count_df, use_container_width=True)
    metric_data_output_image_count_df['Date'] = metric_data_output_image_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
    # Group by date and sum the values
    metric_data_output_image_count_by_date_df = metric_data_output_image_count_df.groupby('Date').agg({'Value': 'sum'}).reset_index()
    st.dataframe(metric_data_output_image_count_by_date_df, use_container_width=True)
    st.line_chart(metric_data_output_image_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')
    st.bar_chart(metric_data_output_image_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')

with tab5:
   
    #metric_data_app_user_invocation_count = bedrock_cloudwatch_get_metric(metric_namespace='App/Chat', metric_name='UserInvocation', start_time=start_time, end_time=end_time)

    metric_data_app_user_invocation_count = bedrock_cloudwatch_get_metric_with_dimensions(
        metric_namespace='App/Chat', 
            metric_name='UserInvocation', metric_dimensions=[{
                'Name': 'User',
                'Value': 'Fred',
            }], start_time=start_time, end_time=end_time)
    #metric_data_app_user_invocation_count = bedrock_cloudwatch_get_metric_expressoin(metric_namespace='App/Chat', metric_name='UserInvocation', start_time=start_time, end_time=end_time)
    
    metric_data_app_user_invocation_count_df = pd.DataFrame({
        'Timestamp': metric_data_app_user_invocation_count['Timestamps'],
        'Value': metric_data_app_user_invocation_count['Values'],
    })
    st.dataframe(metric_data_app_user_invocation_count_df, use_container_width=True)
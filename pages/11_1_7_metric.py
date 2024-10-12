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
def bedrock_cloudwatch_get_metric(metric_namespace, metric_name, start_time, end_time, aggregate_stat="Sum"):
    metric_data = cmn.cloudwatch_metrics_lib.cloudwatch_get_metric(metric_namespace, metric_name, start_time, end_time, aggregate_stat)
    #print(f"NextToken: {metric_data['NextToken']}")
    return metric_data

@st.cache_data(show_spinner='Loading Metrics with Dimensions')
def bedrock_cloudwatch_get_metric_with_dimensions(metric_namespace, metric_name, metric_dimensions, start_time, end_time):
    metric_data = cmn.cloudwatch_metrics_lib.cloudwatch_get_metric_with_dimensions(metric_namespace, metric_name, metric_dimensions, start_time, end_time)
    #print(f"NextToken: {metric_data['NextToken']}")
    return metric_data
    return metric_data

start_time = datetime.now() - timedelta(weeks=15) #datetime(2024, 7, 2) #datetime.now() - timedelta(days=7),
end_time = datetime.now() #datetime(2024, 7, 11) #datetime.now() - timedelta(days=1),

st.markdown(f"##### :green[{start_time} to {end_time}]")

# List available metrics
#available_metrics = client.list_metrics(Namespace='AWS/Bedrock')
#print("Available metrics:")
#for metric in available_metrics['Metrics']:
#    print(f"- {metric['MetricName']}")


opt_model_id_list = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    #"anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "us.anthropic.claude-3-opus-20240229-v1:0",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "cohere.command-r-v1:0", # The model returned the following errors: Malformed input request: #: extraneous key [top_k] is not permitted, please reformat your input and try again.
    "cohere.command-r-plus-v1:0",
    "meta.llama2-13b-chat-v1", # Llama 2 Chat 13B
    "meta.llama2-70b-chat-v1", # Llama 2 Chat 70B
    "meta.llama3-8b-instruct-v1:0", # Llama 3 8b Instruct
    "meta.llama3-70b-instruct-v1:0",  # Llama 3 70b Instruct
    "us.meta.llama3-2-11b-instruct-v1:0", # Vision
    "us.meta.llama3-2-90b-instruct-v1:0", # Vision
    #"mistral.mistral-7b-instruct-v0:2", # Mistral 7B Instruct Does not support system message
    #"mistral.mixtral-8x7b-instruct-v0:1", # Mixtral 8X7B Instruct Does not support system message
    "mistral.mistral-small-2402-v1:0", # Mistral Small
    "mistral.mistral-large-2402-v1:0", # Mistral Large
]


# Group by week and sum the InputTokenCount
# Plot the weekly aggregated data
#st.line_chart(metric_data_input_token_count_weekly_df, x='Week', y='InputTokenCount', color=["#FF0000"], x_label='Week', y_label='Total Count')



tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Invocation", "InputToken", "OutputToken", "OutputImage", "Latency", "Errors", "UserInvocation"])

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
    st.markdown("##### :blue[Latency]")

    metric_data_invocation_latency = bedrock_cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='InvocationLatency', start_time=start_time, end_time=end_time, aggregate_stat="Average")
    #metric_data_output_image_count = bedrock_cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='InvocationClientErrors', start_time=start_time, end_time=end_time)
    
    #print(metric_data_output_image_count)
    metric_data_invocation_latency_df = pd.DataFrame({
        'Timestamp': metric_data_invocation_latency['Timestamps'],
        #'Value': metric_data_invocation_latency['Values'],
        'Value': [round(value / 1000, 2) for value in metric_data_invocation_latency['Values']],  # Convert to seconds and round to 2 decimal places 
    })
    #st.dataframe(metric_data_invocation_latency_df, use_container_width=True)

    metric_data_invocation_latency_df['Date'] = metric_data_invocation_latency_df['Timestamp'].dt.strftime('%Y-%m-%d')
    # Group by date and sum the values
    metric_data_invocation_latency_by_date_df = metric_data_invocation_latency_df.groupby('Date').agg({'Value': 'mean'}).reset_index()
    st.dataframe(metric_data_invocation_latency_by_date_df, use_container_width=True)
    st.line_chart(metric_data_invocation_latency_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')
    st.bar_chart(metric_data_invocation_latency_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')



with tab6:
    st.markdown("##### :blue[Errors]")

    st.markdown("##### :green[Errors - Client]")

    metric_data_error_client_count = bedrock_cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='InvocationClientErrors', start_time=start_time, end_time=end_time)
    metric_data_error_client_count_df = pd.DataFrame({
        'Timestamp': metric_data_error_client_count['Timestamps'],
        'Value': metric_data_error_client_count['Values'],
    })
    #st.dataframe(metric_data_error_client_count_df, use_container_width=True)
    if not metric_data_error_client_count_df.empty:
        metric_data_error_client_count_df['Date'] = metric_data_error_client_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
        # Group by date and sum the values
        metric_data_error_client_count_by_date_df = metric_data_error_client_count_df.groupby('Date').agg({'Value': 'sum'}).reset_index()
        #st.dataframe(metric_data_error_client_count_by_date_df, use_container_width=True)
        #st.line_chart(metric_data_error_client_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')
        st.bar_chart(metric_data_error_client_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')



    st.markdown("##### :green[Errors - Server]")

    metric_data_error_server_count = bedrock_cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='InvocationServerErrors', start_time=start_time, end_time=end_time)
    metric_data_error_server_count_df = pd.DataFrame({
        'Timestamp': metric_data_error_server_count['Timestamps'],
        'Value': metric_data_error_server_count['Values'],
    })
    st.dataframe(metric_data_error_server_count_df, use_container_width=True)
    if not metric_data_error_server_count_df.empty:
        metric_data_error_server_count_df['Date'] = metric_data_error_server_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
        # Group by date and sum the values
        metric_data_error_server_count_by_date_df = metric_data_error_server_count_df.groupby('Date').agg({'Value': 'sum'}).reset_index()
        st.dataframe(metric_data_error_server_count_by_date_df, use_container_width=True)
        #st.line_chart(metric_data_error_server_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')
        st.bar_chart(metric_data_error_server_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')
    else:
        st.info("No data available for Errors - Server")


    st.markdown("##### :green[Errors - Throttle]")

    metric_data_error_throttle_count = bedrock_cloudwatch_get_metric(metric_namespace='AWS/Bedrock', metric_name='InvocationThrottles', start_time=start_time, end_time=end_time)
    metric_data_error_throttle_count_df = pd.DataFrame({
        'Timestamp': metric_data_error_throttle_count['Timestamps'],
        'Value': metric_data_error_throttle_count['Values'],
    })
    st.dataframe(metric_data_error_throttle_count_df, use_container_width=True)
    if not metric_data_error_throttle_count_df.empty:
        metric_data_error_throttle_count_df['Date'] = metric_data_error_throttle_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
        # Group by date and sum the values
        metric_data_error_throttle_count_by_date_df = metric_data_error_throttle_count_df.groupby('Date').agg({'Value': 'sum'}).reset_index()
        #st.dataframe(metric_data_error_throttle_count_by_date_df, use_container_width=True)
        #st.line_chart(metric_data_error_throttle_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')
        #st.bar_chart(metric_data_error_throttle_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Value')
    else:
        st.info("No data available for Errors - Throttle")

with tab7:
   
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


m_model_id = st.selectbox("Select Model ID", opt_model_id_list)

mtab1, mtab2, mtab3 = st.tabs(["Invocation (M)", "InputToken (M)", "Reserved"])

with mtab1:

    st.markdown("##### :blue[Invocations (M) by Date]")

    metric_data_invocation_count = bedrock_cloudwatch_get_metric_with_dimensions(
        metric_namespace='AWS/Bedrock', 
        metric_name='Invocations', 
        metric_dimensions=[{ 'Name': 'ModelId', 'Value': m_model_id }],
        start_time=start_time, end_time=end_time)

    metric_data_invocation_count_df = pd.DataFrame({
        'Timestamp': metric_data_invocation_count['Timestamps'],
        'Value': metric_data_invocation_count['Values'],
    })

    metric_data_invocation_count_df['Date'] = metric_data_invocation_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
    metric_data_invocation_count_by_date_df = metric_data_invocation_count_df.groupby('Date').agg({'Value': 'sum'}).reset_index()

    st.dataframe(metric_data_invocation_count_by_date_df, use_container_width=True)
    st.line_chart(metric_data_invocation_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Count')
    st.bar_chart(metric_data_invocation_count_by_date_df, x='Date', y='Value', color=["#FF0000"], x_label='Date', y_label='Count')


with mtab2:

    st.markdown("##### :blue[Input Token by Date (M)]")

    metric_data_input_token_count = bedrock_cloudwatch_get_metric_with_dimensions(
        metric_namespace='AWS/Bedrock', 
        metric_name='InputTokenCount',
        metric_dimensions=[{ 'Name': 'ModelId', 'Value': m_model_id }],
        start_time=start_time, end_time=end_time)

    metric_data_input_token_count_df = pd.DataFrame({
        'Timestamp': metric_data_input_token_count['Timestamps'],
        'InputTokenCount': metric_data_input_token_count['Values'],
    })
    metric_data_input_token_count_df['Date'] = metric_data_input_token_count_df['Timestamp'].dt.strftime('%Y-%m-%d')
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

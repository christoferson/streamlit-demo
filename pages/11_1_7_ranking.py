import boto3
import cmn_settings
import streamlit as st
import numpy as np
import pandas as pd
import cmn.cloudwatch_metrics_lib

from datetime import datetime, timedelta

AWS_REGION = cmn_settings.AWS_REGION
APP_LOG_GROUP_METRICS_INVOCATIONS = cmn_settings.APP_LOG_GROUP_METRICS_INVOCATIONS

client = boto3.client("cloudwatch", region_name=AWS_REGION)
#cloudwatch_logs = boto3.client('logs')


@st.cache_data(show_spinner='Loading Metrics')
def bedrock_cloudwatch_insights(start_time:datetime, end_time:datetime):
    #metric_data = cmn.cloudwatch_metrics_lib.cloudwatch_get_metric(metric_namespace, metric_name, start_time, end_time, aggregate_stat)
    
    query_string = """
        fields @timestamp, @message
        | sort @timestamp desc
        | limit 5
        """
    metric_data = cmn.cloudwatch_metrics_lib.query_cloudwatch_logs_insights(
        APP_LOG_GROUP_METRICS_INVOCATIONS,
        query_string,
        start_time=start_time,
        end_time=end_time,
        max_attempts=60,  # Maximum number of retry attempts
        timeout_seconds=300,  # 5 minutes timeout
        poll_interval=1  # Time between polling attempts in seconds
    )
    
    return metric_data

# Add this at the very top of your script
week_options = list(range(8, 25))  # Creates a list from 8 to 24
num_weeks = st.selectbox(
    "Select number of weeks to show",
    options=week_options,
    index=week_options.index(15),  # Default to 15 weeks
    key="num_weeks_selector"
)

# Update the start_time calculation
start_time = datetime.now() - timedelta(weeks=num_weeks)
end_time = datetime.now()

st.markdown(f"##### :green[{start_time} to {end_time}]")

metric_data = bedrock_cloudwatch_insights(start_time=start_time, end_time=end_time)

st.json(metric_data)
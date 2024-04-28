import streamlit as st
import boto3
import settings
import json

AWS_REGION = settings.AWS_REGION

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    opt_temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.1, value=0.1, step=0.1, key="temperature")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)

st.title("💬 Chatbot 3")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        #{"role": "user", "content": "Hello there."},
        #{"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    #if not openai_api_key:
    #    st.info("Please add your OpenAI API key to continue.")
    #    st.stop()

    #client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    #user_message =  {"role": "user", "content": f"{prompt}"}
    #messages = [st.session_state.messages]
    print(f"messages={st.session_state.messages}")

    request = {
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": opt_temperature,
        "top_p": 0.5,
        "top_k": 200,
        "max_tokens": 2048,
        "system": "You are a helpful assistant.",
        "messages": st.session_state.messages
    }
    json.dumps(request, indent=3)
    bedrock_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    response = bedrock_runtime.invoke_model_with_response_stream(modelId = bedrock_model_id, body = json.dumps(request))

    #with st.chat_message("assistant", avatar=setAvatar("assistant")):
    with st.chat_message("assistant"):
        result_area = st.empty()
        result_text = ""
        stream = response["body"]
        for event in stream:
                chunk = json.loads(event["chunk"]["bytes"])

                if chunk['type'] == 'message_start':
                    #print(f"Input Tokens: {chunk['message']['usage']['input_tokens']}")
                    pass

                elif chunk['type'] == 'message_delta':
                    #print(f"\nStop reason: {chunk['delta']['stop_reason']}")
                    #print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
                    #print(f"Output tokens: {chunk['usage']['output_tokens']}")
                    pass

                elif chunk['type'] == 'content_block_delta':
                    if chunk['delta']['type'] == 'text_delta':
                        text = chunk['delta']['text']
                        #await msg.stream_token(f"{text}")
                        result_text += f"{text}"
                        result_area.write(result_text)

                elif chunk['type'] == 'message_stop':
                    invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                    input_token_count = invocation_metrics["inputTokenCount"]
                    output_token_count = invocation_metrics["outputTokenCount"]
                    latency = invocation_metrics["invocationLatency"]
                    lag = invocation_metrics["firstByteLatency"]
                    stats = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                    #await msg.stream_token(f"\n\n{stats}")
                    result_text += f"\n\n{stats}"
                    result_area.write(result_text)

        st.session_state.messages.append({"role": "assistant", "content": result_text})
        
    #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    #msg = "response.choices[0].message.content"
    
    #st.chat_message("assistant").write(msg)
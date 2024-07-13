# streamlit-demo
Streamlit Demo

##### Installation
pip install --upgrade boto3
pip install --upgrade streamlit
pip install streamlit==1.35.0
pip install --upgrade python-dotenv
pip install --upgrade PyJWT

##### Start Application
streamlit run app.py --server.headless=true
run options: https://docs.streamlit.io/develop/api-reference/configuration/config.toml

##### Local Url
http://localhost:8501/

https://docs.streamlit.io/library/api-reference


##### System prompts can include:

Task instructions and objectives
Personality traits, roles, and tone guidelines
Contextual information for the user input
Creativity constraints and style guidance
External knowledge, data, or reference material
Rules, guidelines, and guardrails
Output verification standards and requirements

e.g. All your output must be pirate speech

# Contents

##### Bedrock Converse Streaming API
[Converse API](pages/3_5_1_converse_demo.py)


##### Bedrock Models
- Claude Sonnet 3.5 "anthropic.claude-3-5-sonnet-20240620-v1:0"
- https://aws.amazon.com/jp/blogs/aws/anthropics-claude-3-5-sonnet-model-now-available-in-amazon-bedrock-the-most-intelligent-claude-model-yet/

https://github.com/build-on-aws/python-fm-playground
https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html

##### Text to Image

- (Stability.ai Diffusion 1.0 text to image)[https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-text-image.html]

TODO
Converse + Tool
Agents


##### Bedrock Cloudwatch Metrics
https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-cw.html

Invocations
InvocationLatency
InvocationClientErrors
InvocationServerErrors
InvocationThrottles
InputTokenCount
LegacyModelInvocations
OutputTokenCount
OutputImageCount

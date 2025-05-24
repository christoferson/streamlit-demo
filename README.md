# streamlit-demo
Streamlit Demo

##### Installation
pip install --upgrade boto3
pip install --upgrade streamlit
pip install streamlit==1.35.0
pip install --upgrade python-dotenv
pip install --upgrade PyJWT
pip install --upgrade streamlit-webrtc

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

Claude Opus 4, Claude Sonnet 4
anthropic.claude-opus-4-20250514-v1:0
anthropic.claude-sonnet-4-20250514-v1:0
- https://aws.amazon.com/jp/blogs/aws/claude-opus-4-anthropics-most-powerful-model-for-coding-is-now-in-amazon-bedrock/

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


##### Cross Region Inference Profile #####

https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
https://docs.aws.amazon.com/bedrock/latest/userguide/throughput.html



##### Errors & Troubleshooting

##### Error: The provided model doesn't support on-demand throughput.
If you are using a cross region model, e.g. llama, make sure you are using the cross region endpoint.

When using LLama, using the following model id will result to an error:
meta.llama3-2-11b-instruct-v1:0
Instead, you should use the following:
us.meta.llama3-2-11b-instruct-v1:0
us.meta.llama3-2-90b-instruct-v1:0
Note, make sure to replace 'us' with the correct region. Refer to the followng for the list of model id.
https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference-support.html
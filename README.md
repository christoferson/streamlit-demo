# Overview
Streamlit Demo | Code examples using Amazon Bedrock and other AWS AI services on Streamlit

A portfolio of 24 working demos: chat with any Bedrock model, tool-using agents, RAG, web search,
image generation and editing, and text to speech. The home page explains each one; the full menu,
with notes per page, is in [MENU.md](MENU.md).

##### Installation
Python 3.12, in a virtual environment:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

boto3 1.43.127 or later is required (the web search page calls `invoke_web_search`).
Set `AWS_REGION` (e.g. `us-east-1`) and AWS credentials with Bedrock access, in `.env` or the environment.

##### Start Application
```bash
.venv/Scripts/python -m streamlit run app.py --server.headless true
```

run options: https://docs.streamlit.io/develop/api-reference/configuration/config.toml

##### Local Url
http://localhost:8501/app

##### Tests
```bash
.venv/Scripts/python -m unittest discover -s tests   # no AWS calls
.venv/Scripts/python tools/check_models.py --ping    # checks catalog models answer (a few cents)
```


##### System prompts can include:

- Task instructions and objectives
- Personality traits, roles, and tone guidelines
- Contextual information for the user input
- Creativity constraints and style guidance
- External knowledge, data, or reference material
- Rules, guidelines, and guardrails
- Output verification standards and requirements


# Contents

| Section | Demos |
|---|---|
| Chat & agents | [Chat with any model](demos/3_5_13_converse_demo.py), [Tool-using assistant](demos/3_7_5_converse_tool_v3.py), [Tool use, step by step](demos/3_7_4_converse_tool_demo.py), [OpenAI-compatible API](demos/3_8_01_bedrock_mantle.py), [AgentCore harness chat](demos/3_9_01_bedrock_harness_chat.py), [Chat with native requests](demos/3_4_chat_demo.py) |
| Search & knowledge (RAG) | [Ask about a file](demos/6_2_4_filekb_demo.py), [Search the web](demos/3_9_02_agentcore_web_search.py) (AgentCore Web Search API), [Chat with your documents](demos/14_1_1_session_rag.py), [Managed knowledge bases](demos/3_9_50_bedrock_managed_kb.py), [S3 Vectors store](demos/2_s3_vector.py), [Multi-tenant knowledge base](demos/2_s3_vector_kb_3.py) |
| Images & vision | [Generate images](demos/9_1_8_image_generator.py), [Edit images](demos/9_5_7_image_stability_edit.py), [Sketch & style control](demos/9_5_7_image_stability_control.py), [Upscale images](demos/9_5_7_image_stability_upscale.py), [Ask about an image](demos/8_1_6_image_query.py), [Compare image colors](demos/8_5_7_image_color_compare.py), [Multimodal product search](demos/99_01_multimodal_search.py) |
| Voice & speech | [Text to speech](demos/3_6_50_polly_speech.py) (Amazon Polly generative voices) |
| Prompts & tools | [Token counter](demos/1_count_token.py), [Prompt manager](demos/2_prompt_template.py), [Structured output](demos/13_1_1_resformat.py), [Table row selection](demos/12_1_1_dataset.py) |

Design notes: [DESIGN.md](DESIGN.md), [design/](design/) (converse architecture, model catalog, per-page facts).


##### Bedrock Models
Current models and their IDs are in `cmn/bedrock_models.py` (see [design/model-catalog.md](design/model-catalog.md)).
The notes below are historical.

- Claude Sonnet 3.5 "anthropic.claude-3-5-sonnet-20240620-v1:0"
- https://aws.amazon.com/jp/blogs/aws/anthropics-claude-3-5-sonnet-model-now-available-in-amazon-bedrock-the-most-intelligent-claude-model-yet/

- https://github.com/build-on-aws/python-fm-playground
- https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html

- Claude Opus 4, Claude Sonnet 4
- anthropic.claude-opus-4-20250514-v1:0, anthropic.claude-sonnet-4-20250514-v1:0
- https://aws.amazon.com/jp/blogs/aws/claude-opus-4-anthropics-most-powerful-model-for-coding-is-now-in-amazon-bedrock/

- Claude Sonnet 4.5 "anthropic.claude-sonnet-4-5-20250929-v1:0"
- Latest version with improved performance and capabilities
- **Note:** When configuring model parameters, only temperature OR top_p can be specified, but not both parameters simultaneously.

##### Text to Image

- (Stability.ai Diffusion 1.0 text to image)[https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-text-image.html]


##### Bedrock Cloudwatch Metrics
https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-cw.html

- Invocations
- InvocationLatency
- InvocationClientErrors
- InvocationServerErrors
- InvocationThrottles
- InputTokenCount
- LegacyModelInvocations
- OutputTokenCount
- OutputImageCount


##### Cross Region Inference Profile #####

- https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/throughput.html



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


TODO:
SQL / relational and natural language
meeting minutes
coding buddy

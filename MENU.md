# App menu

What visitors see: the sidebar menu and the home page. Both are generated from one list,
`MENU_SECTIONS` in `cmn/view/menu.py`, so edit that and this file together.

Per-page facts (UI, APIs, models, settings, quirks, differences between similar pages) are in
`design/demo-pages.md`. Write menu text from there instead of re-reading the pages.

## How it works

- `app.py` is the router. It builds `st.navigation()` from `MENU_SECTIONS`, so **only pages listed
  there appear in the sidebar**. Files in `demos/` that aren't listed aren't reachable.
- The folder is `demos/`, not `pages/`, on purpose. While a `pages/` folder exists, Streamlit runs
  a fresh server's first request with its old auto-navigation, so a visitor who opens a demo URL
  directly gets a sidebar of raw file names (all pages, broken ones too). Don't create `pages/`.
- `home.py` is the landing page: intro, counts, "Start here" cards (pages with `featured=True`),
  every demo by section, and a short "How it's built".
- Page URLs are Streamlit's default: the file name minus its first number
  (`demos/3_7_5_converse_tool_v3.py` -> `/app/7_5_converse_tool_v3`), the same as the old
  `pages/` auto-navigation, so old links still work.

Run it: `py -3.12 -m streamlit run app.py --server.headless true`, then open http://localhost:8501/app.

### Adding a page to the menu

1. Add a `MenuPage(...)` to the right section in `cmn/view/menu.py`.
   - `title`: what it does, in plain words ("Edit images"), not the file name.
   - `icon`: a Material icon, `":material/<name>:"` (no emoji).
   - `summary`: one or two sentences for someone who has never seen the app.
   - `details`: two to four "what you can do" bullets.
   - `services`: AWS services / APIs, shown as blue badges.
   - `try_this`: something a visitor can do in a minute (green). Base it on what the page really
     supports, e.g. an example prompt it ships with.
   - `needs`: resources it expects in the AWS account (orange). Leave empty if none.
   - `differs`: how it differs from a similar-looking demo (violet).
   - `featured=True` puts it on the home page's "Start here" cards. Keep that to about six pages
     you have checked end to end.
2. Add it to the table below.

### Home page colors

| Look | Meaning |
|---|---|
| :blue-badge[...] | AWS service or API the demo uses |
| :green[**Try this**] | A one-minute test |
| :orange[**Needs setup**] | Resources the demo expects in the AWS account |
| :violet[**How it differs**] | Difference from similar-looking demos |

## Menu

Featured pages are marked ★.

### Chat & agents

| Menu title | File | Notes |
|---|---|---|
| ★ Chat with any model | `3_5_13_converse_demo.py` | Model picker comes from the catalog. English guide text is still Lorem ipsum; Export menu does nothing |
| ★ Tool-using assistant | `3_7_5_converse_tool_v3.py` | Slow first load (imports plotting, PDF, PPTX) |
| Tool use, step by step | `3_7_4_converse_tool_demo.py` | Prototype of the tool loop |
| ★ OpenAI-compatible API | `3_8_01_bedrock_mantle.py` | Bedrock Mantle, us-east-1 |
| AgentCore harness chat | `3_9_01_bedrock_harness_chat.py` | Needs an AgentCore harness in the account |
| Chat with native requests | `3_4_chat_demo.py` | InvokeModel. Polly code is present but never called |

### Search & knowledge (RAG)

| Menu title | File | Notes |
|---|---|---|
| ★ Ask about a file | `6_2_4_filekb_demo.py` | No setup; RetrieveAndGenerate on one uploaded file |
| Chat with your documents | `14_1_1_session_rag.py` | Local FAISS, sessions saved under `session_data/`. PDF/DOCX are read as plain text, and the temperature slider isn't sent |
| Managed knowledge bases | `3_9_50_bedrock_managed_kb.py` | The account's knowledge base has no data sources yet; model ARN is fixed to us-east-1 |
| S3 Vectors store | `2_s3_vector.py` | Needs an S3 Vectors bucket. Sample Query buttons only show the text |
| Multi-tenant knowledge base | `2_s3_vector_kb_3.py` | Needs a KB, S3 buckets, DynamoDB tables and an ingestion Lambda |

### Images & vision

| Menu title | File | Notes |
|---|---|---|
| ★ Generate images | `9_1_8_image_generator.py` | Stability models, called in us-west-2 |
| ★ Edit images | `9_5_7_image_stability_edit.py` | |
| Sketch & style control | `9_5_7_image_stability_control.py` | |
| Upscale images | `9_5_7_image_stability_upscale.py` | |
| Ask about an image | `8_1_6_image_query.py` | |
| Compare image colors | `8_5_7_image_color_compare.py` | |
| Multimodal product search | `99_01_multimodal_search.py` | Needs an OpenSearch endpoint |

### Voice & speech

| Menu title | File | Notes |
|---|---|---|
| Text to speech | `3_6_50_polly_speech.py` | Amazon Polly generative engine (Oct 2026 upgrade). SynthesizeSpeech only; this boto3 has no StartSpeechSynthesisStream |

### Prompts & tools

| Menu title | File | Notes |
|---|---|---|
| Token counter | `1_count_token.py` | CountTokens is free |
| Prompt manager | `2_prompt_template.py` | |
| Structured output | `13_1_1_resformat.py` | |
| Table row selection | `12_1_1_dataset.py` | No AWS calls |

## Not in the menu

These files are still in `demos/` but are left out until fixed. Status from a load test and a
Bedrock model listing on 2026-11-01.

| File | Why it's hidden |
|---|---|
| `3_6_01_speak_demo.py` | `No module named 'sounddevice'`; Nova Sonic v1 is no longer listed |
| `5_2_6_kb_demo.py`, `5_2_7_kb_demo.py` | Crash on load: `KeyError: 'description'`; Claude 3 models (end of life) |
| `5_3_1_kb_rg_demo.py`, `5_3_2_kb_rg_demo.py` | Same as above |
| `6_2_6_docquery_demo.py` | Defaults to Claude 3 Sonnet (end of life) |
| `7_1_utilities.py`, `7_3_copyeditor.py` | `No module named 'pyperclip'`; Claude 3 models |
| `7_4_grammar.py` | Crash on load: `st.session_state` has no `analysis_complete` |
| `8_2_1_multi_image_query.py` | Defaults to Claude 3.5 Sonnet (end of life) |
| `9_2_1_image_generator_si.py` | SDXL / SD3 Large no longer available |
| `9_3_5_image_variation.py`, `9_3_6_image_variation.py` | SDXL no longer available |
| `9_5_6_image_*_amz.py` (5 pages) | Titan Image Generator v2 no longer available |
| `9_5_6_image_variation_amz_nova.py` | Nova Canvas no longer available |
| `10_1_5_agent.py` | Crash on load: `KeyError: 'latestAgentVersion'` |
| `11_1_7_metric.py` | Crash on load: `.dt accessor` on non-datetime column |
| `11_1_7_ranking.py` | CloudWatch Logs `AccessDeniedException` |

To bring one back: fix it, check it loads (AppTest) and answers, then add it to `MENU_SECTIONS`.

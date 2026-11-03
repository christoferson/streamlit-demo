# Demo pages reference

Facts about each page in the menu, read from the code on 2026-11-01, so you don't have to re-read
22 long files. The home page text in `cmn/view/menu.py` is written from this. When you change a page,
update its entry here. Menu structure and hidden pages are in `MENU.md`.

Settings come from `cmn_settings.py` (`.env` / environment) unless noted. "Catalog" means
`FOUNDATION_MODELS` in `cmn/bedrock_models.py`.

## Known quirks (visible to visitors)

| Page | Quirk |
|---|---|
| `3_5_13` | English guide text is Lorem ipsum (Japanese is real); Export menu is a no-op; audio input is accepted but ignored |
| `3_4` | Polly and guardrail code is never called |
| `2_s3_vector` | Sample Query buttons only show the text; they don't run it or fill the input |
| `3_9_50` | Model ARN is always built for us-east-1, whatever "Inference Region" is set to |
| `14_1_1` | Temperature slider isn't sent to the model; PDF/DOCX are decoded as UTF-8 text, not parsed |
| `8_1_6` | The image goes only with the current turn; saved history keeps text only |
| `1_count_token` | Default model `us.anthropic.claude-sonnet-4-20250514-v1:0` is end of life |
| `2_prompt_template` | Model dropdown lists old models (Claude 3, Llama 3); the prompt's own model is used by default |

## Chat & agents

### `3_5_13_converse_demo.py` — Chat with any model
- **UI:** title "Chat with any model". Wide chat plus a sticky Info Panel. Material-icon avatars (`AVATARS`: person / smart_toy / error). Sidebar: model picker (whole catalog), Temperature / TopP / TopK only if the model supports them, Max Tokens, system message. Attachments through `chat_input` (placeholder "Ask <model name>"): txt, csv, pdf, md if the model supports documents; png, jpg, jpeg, gif, webp if it supports vision. Bar above the input: left = current model, temperature (if supported), max tokens; right = message count out of 160, clear, download, Export. Delete one exchange; download / upload a signed conversation JSON. Guide toggle EN / 日本語. Info Panel: Last Response (tokens in / out / total, latency) and Quick Info (provider, features).
- **APIs:** `converse_stream` via `build_converse_request` / `process_stream`. CloudWatch Logs `put_log_events` to `/app/openai/metrics/invocations` (one stream per day; failures only logged). Default model `global.anthropic.claude-sonnet-5`.
- **Settings:** `CONVERSE_CH_SECRET_KEY`, `CONVERSE_CH_SALT`, `CONVERSE_CH_ALGORITHM` (HMAC signing in `cmn_security`), `AWS_REGION`.
- **Try:** attach a PDF or image, ask "Summarize this", download the conversation and load it back.

### `3_7_5_converse_tool_v3.py` — Tool-using assistant
- **UI:** chat. Sidebar: Model ID, Temperature (only if the model accepts it), Max Tokens, system prompt generated from the tool registry, Show Invocation Metrics, Tools list. Examples toggle with sample questions. Tool calls show inline (args and result JSON), with rich renderings below.
- **APIs:** `converse_stream` through `ConversationManager` (unlimited tool loop). Fixed model list: Sonnet 4.5 (default), Sonnet 4, `global.openai.gpt-6.1-terra`, Cohere Command R / R+, Llama 3 8B / 70B, Mistral Large 2402, Kimi K2.5.
- **Tools (14):** calculator, acronym, url_content, wikipedia, aws_docs, datetime, sales_data, product (NL-to-SQL on DuckDB), chart, sales_kpi, sales_forecast, sales_anomaly, pptx, pdf. Renderers: Plotly charts, product tables, KPI / anomaly / forecast cards, PPTX / PDF download links. Metrics sum per turn: tokens, latency, LLM calls, tool order.
- **Settings:** `AWS_REGION`. Sales / product data are local.
- **Note:** first load takes 20–40 s (imports plotting, PDF, PPTX libraries).
- **Try:** "Compare 2024 vs 2023 sales performance. What caused any underperformance?" or "Show me all products under $100".

### `3_7_4_converse_tool_demo.py` — Tool use, step by step
- **UI:** wide chat plus a right Info Panel. Sidebar: model picker, sliders, system message, "Available Tools" checkboxes (unticked tools aren't offered). One uploader: PNG, JPEG, TXT, CSV (with preview), PDF, MD. Info Panel: summed metrics with per-call breakdown, Tool Use Log (input / result JSON, newest first).
- **APIs:** `converse_stream` with `toolConfig`. Models: catalog entries with streaming tool use, excluding opus / fable / nova-pro / 70b / large.
- **Tools (5):** calculator, acronym, url_content, wikipedia, datetime. Capped at 5 rounds (`MAX_TOOL_ROUNDS`). Sends `reasoningContent` with signatures back during the loop. Records which tools were enabled per message.
- **Try:** "What is today's date, and what is 17% of 2,340?"

### `3_8_01_bedrock_mantle.py` — OpenAI-compatible API
- **UI:** chat. Sidebar: Model ID, Web Search (Off / client-side DuckDuckGo / Bedrock built-in server-side), "Allow live web access" in built-in mode, Max Tokens (4096), system message, metrics toggle. Image and document attachments. Examples toggle. History limit 200.
- **APIs:** Bedrock Mantle. `anthropic.*` models: Anthropic SDK `AnthropicBedrockMantle`, `messages.create(stream=True)`. `openai.*` models: OpenAI SDK against `https://bedrock-mantle.{region}.api.aws/openai/v1`, Responses API streaming; bearer token from `aws_bedrock_token_generator`, refreshed every 55 min. Models: `anthropic.claude-opus-4-7`, `openai.gpt-6.1-terra`, `gpt-5.5`, `gpt-5.6-sol/terra/luna` (default `gpt-5.6-luna`).
- **Features:** tools `web_search`, `url_content_loader`. Built-in search shows a trace (searched / read / found in page) and deduplicated Sources. Built-in search with a Claude model falls back to client-side with a warning. System prompt adds today's date and citation rules.
- **Settings:** `AWS_REGION`. IAM `bedrock-websearch:InvokeSearch`; live web also `bedrock-websearch:ExternalWebAccess`. GPT-6.1 Terra on Mantle: us-east-1 only.
- **Try:** `openai.gpt-5.6-luna`, "Bedrock built-in", "What are today's top AWS news headlines?"

### `3_9_01_bedrock_harness_chat.py` — AgentCore harness chat
- **UI:** pick a harness (refresh, Show Details), model, optional temperature / max tokens, system prompt override. Session Info (session, actor, memory IDs). Buttons: View Memory Events, View Long-Term Memory (list or search by namespace), View Sessions (preview, resume), Run Command (shell in the harness microVM; presets like `python --version`, `pip list`, `df -h`), New Session. Streaming replies with "Calling <tool>" status.
- **APIs:** `bedrock-agentcore-control` list_harnesses, get_harness, get_memory; `bedrock-agentcore` invoke_harness (stream), invoke_agent_runtime_command, list_sessions, list_events, list_memory_records, query_memory; `bedrock-agent-runtime` end_session. Models: Sonnet 4.6 (default), Sonnet 5, Opus 4.8.
- **Settings:** an AgentCore harness in `AWS_REGION`; memory configured on it for memory features. Actor ID fixed to `default-user`.
- **Try:** select a harness, Run Command → "OS info", chat, then View Sessions and resume.

### `3_4_chat_demo.py` — Chat with native requests
- **UI:** simple text chat. Sidebar: model picker, sliders, system message; last call's params, tokens, latency and first-byte lag.
- **APIs:** `invoke_model_with_response_stream` with a hand-built body per provider (Anthropic Messages, Nova, Llama 3/4 prompt tokens, Mistral `[INST]`); second client for us-west-2 models. CloudWatch `put_metric_data` (namespace `App/Chat`, metric `UserInvocation`, dummy user). Models: catalog Anthropic / Amazon / Meta / Mistral, excluding large tiers.
- **Try:** same question on Claude, Nova and Llama; compare timings.

**Differences:** `3_5_13` is the general chat on Converse (all models, attachments, save/load, metrics, no tools). `3_4` is the low-level InvokeModel version with per-provider bodies, text only. `3_8_01` uses the Mantle OpenAI-compatible endpoint (Responses API / Anthropic SDK) and adds web search with citations. `3_7_4` is the readable 5-tool prototype (checkboxes, 5-round cap, JSON results); `3_7_5` is the layered production version (ConversationManager, ToolRegistry, RendererRegistry, 14 tools, charts and downloads, unlimited loop).

## Search & knowledge (RAG)

### `6_2_4_filekb_demo.py` — Ask about a file
- **UI:** attach one file with `chat_input` (pdf, md, txt, docx, html, csv, xls, xlsx, png, jpg, jpeg, gif, webp; about 20k tokens max). Active-file panel (thumbnail for images); file kept for follow-ups. Citations JSON expander plus inline numbered references.
- **APIs:** `retrieve_and_generate` with `EXTERNAL_SOURCES` / `BYTE_CONTENT`; session ID reused for follow-ups, reset when the file changes. Images go to `converse` (EXTERNAL_SOURCES rejects them). Model: Sonnet 4.5.
- **Settings:** `AWS_REGION` only.
- **Try:** attach a PDF, "What are the key points?", then a follow-up.

### `14_1_1_session_rag.py` — Chat with your documents
- **UI:** create an 8-character session or load one. Model, temperature, max tokens (256–4096), Top K (1–10). Upload txt, md, csv, pdf, docx, json; chunk size / overlap (1000 / 200); Process. Answers with "View Sources" (L2 distance, filename).
- **APIs:** `invoke_model` with `amazon.titan-embed-text-v2:0` (1024-dim, normalized); `converse` for answers. Models: Sonnet 4.5 (default), Sonnet 4, Sonnet 3.7 / 3.5 v2 (end of life), Opus 4.6.
- **Storage:** FAISS `IndexFlatL2` + metadata saved under `session_data/<id>/` (`faiss.index`, `metadata.pkl`); git-ignored.
- **Settings:** `AWS_REGION`; packages `faiss`, `numpy`.
- **Try:** new session, upload a .txt / .md, Process, ask, open View Sources.

### `3_9_50_bedrock_managed_kb.py` — Managed knowledge bases
- **UI:** pick a KB; Select, Show Details (config, storage, embedding model), Documents dialog (data source, per-document ingestion status, paging). Model, inference region, temperature, max chunks (1–100), metadata filter (up to 3 equals conditions, AND). Mode pills filtered by KB type; Markdown / Plain toggle.
- **Modes:** VECTOR, KENDRA, SQL → Retrieve & Generate, Retrieve. MANAGED → Retrieve, Agentic Retrieve (streams trace steps). Score colors: green ≥ 0.8, orange ≥ 0.5.
- **APIs:** `bedrock-agent` list_knowledge_bases, get_knowledge_base, list_data_sources, list_knowledge_base_documents; `bedrock-agent-runtime` retrieve, retrieve_and_generate, agentic_retrieve_stream; `sts` get_caller_identity for the profile ARN. Model `us.anthropic.claude-sonnet-4-5-20250929-v1:0` via inference-profile ARN.
- **Settings:** an existing KB with a synced data source in `AWS_REGION`.

### `2_s3_vector.py` — S3 Vectors store
- **UI:** sidebar embedding model, dimension, Nova query purpose. Tabs: Add Documents (single with filterable metadata `document_type`, `category`, `aws_service` and display-only `title`, `URL`; bulk; 10 sample AWS docs), Query (top-k, filters, distance / similarity), Manage Bucket (list indexes), Manage Index (count, preview, reset with confirm).
- **APIs:** `invoke_model` embeddings: `amazon.nova-2-multimodal-embeddings-v1:0` (default; 256 / 384 / 1024 / 3072; GENERIC_INDEX to index, *_RETRIEVAL to query), Titan Text v2 / v1, Titan Multimodal v1, Cohere embed v3. `s3vectors` put_vectors, query_vectors, list_indexes, list_vectors, delete_vectors. No LLM answer.
- **Settings:** `VECTOR_BUCKET_NAME`, `VECTOR_INDEX_NAME`, `VECTOR_REGION_ID` (default us-east-1). Index dimension must match the model.
- **Try:** Add Sample Data, query "What is a serverless compute service?" with Category = Compute.

### `2_s3_vector_kb_3.py` — Multi-tenant knowledge base
- **UI:** User ID (default `user-alice`), LLM picker, config view, Test Connections. Tabs: My KBs, Create KB (metadata in DynamoDB), Upload (to `s3://bucket/user/kb/` with `.metadata.json` sidecar: user_id, kb_id), Query KB (Retrieve Only / Retrieve & Generate, debug), System Status (ingestion jobs), Manage Documents (download, delete, Sync All), Debug Search.
- **APIs:** `retrieve` / `retrieve_and_generate` with a fixed `andAll` filter on user_id and kb_id; `bedrock-agent` start_ingestion_job, list_ingestion_jobs, get_knowledge_base, get_data_source; `s3` put / delete / head_bucket; `dynamodb` put_item and GSI queries. Region hardcoded us-east-1, `AWS_PROFILE`. Models: Nova Pro, Llama 4 Maverick, Sonnet 4, Sonnet 4.5.
- **Settings:** `CMN_KB_KNOWLEDGE_BASE_ID`, `CMN_KB_DATA_SOURCE_ID`, `CMN_KB_DOCUMENT_BUCKET_NAME`, `CMN_KB_VECTOR_BUCKET_NAME`, `CMN_KB_VECTOR_INDEX_NAME`, `CMN_KB_KB_METADATA_TABLE` (GSI `UserIdIndex`), `CMN_KB_DOC_METADATA_TABLE` (GSI `KbIdIndex`). Auto-ingestion relies on an external EventBridge → Lambda pipeline.
- **Try:** as user-alice create a KB, upload a .txt, wait for ingestion, Retrieve Only; switch to user-bob.

**Differences:** `6_2_4` one file, no setup. `14_1_1` hand-built RAG (Titan + local FAISS, sessions on disk). `3_9_50` console for existing managed KBs (all KB types, three modes, filters). `2_s3_vector` raw vector store management, retrieval only. `2_s3_vector_kb_3` multi-tenant SaaS pattern over one KB, heaviest setup.

## Images & vision

### `9_1_8_image_generator.py` — Generate images
- **UI:** Basic and Notes tabs. Model, aspect ratio (1:1, 16:9, 21:9, 2:3), output format (JPEG / PNG), seed (-1 random), negative prompt (pre-filled). One image per prompt in the chat history.
- **APIs:** `invoke_model` on a us-west-2 client. Models: `stability.stable-image-ultra-v1:1` (default), `stable-image-core-v1:1`, `sd3-5-large-v1:0`. Old Titan / Nova Canvas / SDXL branches remain in code with empty model lists (end of life).
- **Try:** "a lighthouse on a cliff at sunset, cinematic", 16:9, Ultra then Core.

### `9_5_7_image_stability_edit.py` — Edit images
- **UI:** six tools: Remove Background, Inpaint, Outpaint, Search and Recolor, Search and Replace, Erase. Per-tool controls (prompt, select / search prompt, negative prompt, style preset, creativity, outpaint px per side, grow_mask, seed, format). Upload image (and mask / alpha for Inpaint and Erase) with size and ratio checks. Result (checkerboard for transparency), metrics, download, side-by-side.
- **APIs:** `invoke_model` in `AWS_REGION` (hardcoded us-east-1 in the page). Models: `us.stability.stable-image-remove-background-v1:0`, `-inpaint-v1:0`, `us.stability.stable-outpaint-v1:0`, `-search-recolor-v1:0`, `-search-replace-v1:0`, `-erase-object-v1:0`.
- **Try:** product photo → Remove Background → download PNG.

### `9_5_7_image_stability_control.py` — Sketch & style control
- **UI:** Control Sketch, Control Structure, Style Guide, Style Transfer (content + style image). Prompt (defaults), negative prompt, control_strength, fidelity, composition_fidelity, style_strength, change_strength, aspect ratio (Style Guide), style preset, seed, format; 16 MB payload estimate. Download; help tabs.
- **APIs:** `invoke_model`: `us.stability.stable-image-control-sketch-v1:0`, `-control-structure-v1:0`, `-style-guide-v1:0`, `us.stability.stable-style-transfer-v1:0`.
- **Try:** Control Sketch with a line drawing and the default prompt.

### `9_5_7_image_stability_upscale.py` — Upscale images
- **UI:** Creative (20–40x to 4K; prompt, creativity, style), Conservative (keeps detail), Fast (4x, image only). Output-size estimate against 16 MB. Download; help and troubleshooting tabs.
- **APIs:** `invoke_model`: `us.stability.stable-creative-upscale-v1:0`, `-conservative-upscale-v1:0`, `-fast-upscale-v1:0`.
- **Try:** small JPEG (≤ 1536 px) → Fast Upscale.

### `8_1_6_image_query.py` — Ask about an image
- **UI:** right column image uploader (PNG / JPEG) with preview, "Analyze with Rekognition" (raw response + dominant colors), Clear History. Chat in a fixed-height container; long image-analysis system prompt (description, VQA, OCR, objects, scene, aesthetics). Token / latency line per answer.
- **APIs:** `converse_stream` with an image block; `rekognition` detect_labels (`IMAGE_PROPERTIES`, 5 colors). Model: Sonnet 4.5.
- **Try:** a sign or receipt photo, "Extract all visible text", then Analyze with Rekognition.

### `8_5_7_image_color_compare.py` — Compare image colors
- **UI:** two uploaders; per image "Get Image Properties (Rekognition)" (brightness, sharpness, contrast, up to 7 dominant colors with RGB / hex / CSS name / %) and "(Pillow)" (local metrics + KMeans colors over an edge-based mask). "Compare" (Claude rates similarity 1–10) and "Compare with Meta" (adds the measurements, quality checks, answers in English and Japanese). Streams.
- **APIs:** `invoke_model_with_response_stream` with an Anthropic body; `rekognition` detect_labels (cached). Models: Opus 5 (default), Sonnet 5, Fable 5, Opus 4.8 / 4.7 / 4.6 / 4.5, Sonnet 4.6 / 4.5, Haiku 4.5; sampling params left out where rejected.
- **Settings:** packages `opencv`, `scikit-learn`.

### `99_01_multimodal_search.py` (+ `cmn/multimodal_search.py`) — Multimodal product search
- **UI:** sidebar connection / index status, Connect, Refresh, Create Index, Delete Index (double-click confirm), config, debug, product count, result limit (1–50). Tabs: Register Product (no duplicate titles), Search by Image, Search by Text (kNN, falls back to keyword + wildcard), Search by Title (partial / exact / fuzzy), List All Products (filter, delete, update), Update Product.
- **APIs:** `invoke_model` with `amazon.titan-embed-image-v1` for image and text (1024-dim; text embedding uses the title only). OpenSearch Serverless (`opensearch-py`, SigV4 service `aoss`), knn_vector fields `image_embedding`, `text_embedding` (HNSW, faiss, L2).
- **Settings (read in `cmn/multimodal_search.py`, not `cmn_settings`):** `OPENSEARCH_ENDPOINT`, `COLLECTION_NAME` (default `multimodal-search`), `INDEX_NAME` (default `products`), `AWS_PROFILE`, `AWS_REGION`, optional `IMAGE_URL_TEMPLATE`. 10 MB per file; jpg, jpeg, png, webp.

**Differences:** `9_1_8` text-to-image (us-west-2). `9_5_7` pages are image-to-image through `us.stability.*` profiles: edit changes your image, control makes a new image guided by yours, upscale only raises resolution. `8_1_6` multi-turn chat about one image (Converse); `8_5_7` one-shot color / quality comparison of two images with measurements in the prompt (InvokeModel).

## Voice & speech

### `3_6_50_polly_speech.py` — Text to speech (added 2026-11-04)
- **UI:** sidebar Engine (Generative default / Long-form / Neural / Standard), "Only voices new in the October 2026 upgrade" toggle, Language, Voice (new voices listed first and marked "(new)"), "Compare with" (same voice on another engine it supports), audio format (MP3 / OGG). Main: "What's new" expander, sample pills (Money & numbers, Dates & addresses, Codes & phone numbers, Surprise, Uncertainty, Hindi → Kiara, Finnish → Suvi), text area (3000 characters max), SSML toggle, Speak. Results: one card per engine with player, characters billed, latency, size and download; kept in session state.
- **APIs:** `polly` `describe_voices(Engine=...)` (paged, cached 1 h) and `synthesize_speech`. Bilingual voices (Arjun, Kiara, Kajal: en-IN + hi-IN) get `LanguageCode` when speaking their second language. No Bedrock call.
- **Facts (us-east-1, 2026-11-04):** 55 generative voices, 24 languages. All 12 new voice names present: Kenneth, Nate (en-US), Russell, Curtis (en-AU), Kiara, Arjun (en-IN + hi-IN), Tina (de-DE), Carlos (es-US), Leo, Carolina (pt-BR), Hyunduk (ko-KR), Suvi (fi-FI). Russell also supports standard; Suvi also neural. Typical latency: generative 1.3–2.7 s, neural about 0.5 s.
- **Not used:** `StartSpeechSynthesisStream` (bidirectional streaming) is not in boto3 1.43.103. Speech marks are not supported on generative voices.
- **Settings:** `AWS_REGION` (generative is in 11 regions); IAM `polly:DescribeVoices`, `polly:SynthesizeSpeech`.
- **Try:** "Surprise" with Kenneth; then Ruth with "Compare with" → Neural.

## Prompts & tools

### `1_count_token.py` — Token counter
- **UI:** region (12) and model pickers. Tabs: InvokeModel (message, max tokens, temperature) and Converse (system prompt + editable message list). Shows Input Tokens and the request body.
- **APIs:** `count_tokens` with `input.invokeModel` or `input.converse` (free).

### `2_prompt_template.py` — Prompt manager
- **UI:** Invoke Prompt tab: Prompt ARN / ID + version (DRAFT), Load; prompt info, model, overrides (temperature, topP, max tokens), templates, auto-detected `{{variable}}` inputs plus custom variables, Invoke; reply, usage, full response. List Prompts tab. Documentation tab (setup, code, IAM policy).
- **APIs:** `bedrock-agent` get_prompt, list_prompts; `converse`.
- **Settings:** at least one prompt in Bedrock Prompt Management.

### `13_1_1_resformat.py` — Structured output
- **UI:** shared telemetry inputs (speed 85 / limit 65, hard braking 3, rapid acceleration 2, sharp turns 1, rating, duration, time of day). Tabs: Streaming (Prompt), Tool Use (Schema Enforced; editable schema), Prompt Engineering (AWS Blog), Tool Use (AWS Blog), Guide. Each returns risk level, score / 100, insight, factors, recommendations, JSON, latency, tokens, estimated cost.
- **APIs (us-east-1 hardcoded):** tab 1 `converse_stream` asking for JSON (strips fences, first-token latency); tab 2 `converse` with forced `toolChoice` `generate_risk_assessment`; tab 3 `invoke_model` with schema in the prompt (`converse` for non-Claude); tab 4 `invoke_model` native tools with `tool_choice` `analyze_driver_risk` (falls back to `converse`). Models: Nova Pro, Gemma 3 27B, Qwen3 Next 80B, Haiku 4.5 (default), Sonnet 4.5.

### `12_1_1_dataset.py` — Table row selection
- **UI:** random table of 50 animals; `SelectboxColumn` status, `ProgressColumn` health; selecting a row shows details; regenerate button. No AWS calls.

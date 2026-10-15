# Streamlit Demo — Design

High-level design of the repository: what the packages are, how a page talks to Bedrock, and where
the design is heading. Topic deep-dives live in `design/`:

- `design/converse-design.md` — Converse tool loop (v3.7.4 vs v3.7.5), registries, stream processing, message formats
- `design/model-catalog.md` — `FOUNDATION_MODELS`: adding models, parameter sets, ID formats, end-of-life handling

*Last reviewed: 2026-10-15*

---

## 1. System overview

A Streamlit multi-page app of independent demos on AWS Bedrock (chat, tool use, knowledge bases,
agents, image generation, metrics). Each page is self-contained; shared code lives in `cmn/` and a
few top-level `cmn_*.py` modules.

```
Browser ──► Streamlit server (app.py + pages/*.py, auto-discovered)
                │
                ├── Presentation   pages/*.py, cmn/tools/renderer/*, cmn/view/processor/*   (imports streamlit)
                ├── Business logic cmn/bedrock/converse/*, cmn/tools/tool/*, cmn/bedrock_models.py (no streamlit)
                └── AWS access     boto3 clients (bedrock-runtime, bedrock-agent[-runtime], bedrock,
                                   polly, rekognition, cloudwatch, logs, s3vectors, opensearch)
                                        │
                                        ▼
                                   AWS Bedrock & friends
```

**Runtime facts**

| Concern | How it works today |
|---|---|
| Entry point | `app.py` — landing page with a settings demo and a "list models" button. Not a router. |
| Navigation | Streamlit `pages/` auto-discovery. Sorted by the leading number; the rest of the version prefix shows in the sidebar label (e.g. "5 13 converse demo"). `st.navigation` is not used. |
| Config | `cmn_settings.py` loads `.env` (cwd, then repo root) into module constants at import. `.streamlit/config.toml` sets `baseUrlPath="app"`, upload limit, theme. |
| Auth | `cmn_auth.check_password()` (hmac against `st.secrets["passwords"]`). Imported by ~30 pages but **called nowhere** — the app is effectively unauthenticated. ALB/OIDC header handling in `app.py` is a stub. |
| Interpreter | Python 3.12 + Streamlit 1.59 (see CLAUDE.md). Imports assume the working directory is the repo root. |
| Deployment | `Dockerfile` at repo root. |

---

## 2. Repository layout

```
app.py                     Landing page
cmn_settings.py            Environment config (.env → constants)            ← canonical settings module
settings.py                Legacy subset of cmn_settings (AWS_REGION only)  ← deprecated, see §7
cmn_auth.py                Password gate (st.secrets)
cmn_security.py            HMAC signing of saved conversations (ConversationSecurity)
cmn_constants.py           Shared help text + CSS snippets
cmn_st_audio.py            Polly / Comprehend helpers (only used by archived pages)
app_bedrock_lib.py         List knowledge bases / agents / models
pages/                     Live demo pages (N_M_K_<topic>.py)
pages-archive/<topic>/     Superseded page versions (git-ignored, not loaded)
cmn/                       Shared package (see §3)
design/                    Topic design docs
tests/                     unittest suites (no AWS calls)
tools/                     CLI utilities (check_models.py, tool_ch_generate_secrets.py)
images/                    Logos used by st.logo() in most pages
```

### Page naming

`<topic>_<sub>_<version>_<name>.py`, e.g. `3_5_13_converse_demo.py` = topic 3 (chat), sub-topic 5
(converse), version 13. When a newer version lands, the old one moves to
`pages-archive/<name>/` (e.g. `pages-archive/converse_demo/3_5_01 … 3_5_12`). Only the latest
version of each topic should stay in `pages/`.

| Topic | Pages |
|---|---|
| 1–2 | Token counting, prompt management, S3 Vectors |
| 3 | Chat: chat demo, converse, speech (Nova Sonic), converse tools, Mantle, AgentCore harness, managed KB |
| 5–6 | Knowledge bases: retrieve, retrieve-and-generate, file/document query |
| 7 | Text utilities: translate, copy-edit, grammar |
| 8 | Image understanding: query, multi-image, colour compare |
| 9 | Image generation: Stability, Titan / Nova Canvas |
| 10–14, 99 | Agents, CloudWatch metrics, dataset, response formats, session RAG, multimodal search |

---

## 3. Packages (`cmn/`)

### 3.1 `cmn/bedrock_models.py` — model catalog
`FOUNDATION_MODELS: FoundationModelCatalog` of frozen `FoundationModel(model_id, name, provider,
inference_parameters, features, region, listed)`. API: `find(id)`, `listed_models()`,
`display_name`, `is_feature_supported(Feature.X)`, `inference_parameters.top_k.is_supported`.
No streamlit, no boto3. Details: `design/model-catalog.md`.

### 3.2 `cmn/bedrock/` — AWS access + conversation core

| Module | Responsibility |
|---|---|
| `client_manager.py` | `BedrockClientFactory` — boto3 clients with a shared botocore `Config` (adaptive retries, timeouts, pool size). |
| `converse/request_builder.py` | `InferenceSettings` + `FoundationModel` → `converse_stream()` kwargs; sends only the parameters the model supports. |
| `converse/stream_processor.py` | `process_stream(stream, on_text_delta) → StreamResult` — parses text, tool-use blocks, stop reason, metrics, stream exceptions. |
| `converse/conversation_manager.py` | `ConversationManager.run(history, callbacks…)` — the multi-tool loop: call model → run requested tools → append `toolResult` → repeat until no tool use. |
| `converse/conversation_history.py` | `trim_conversation_history`, `validate_conversation_structure`, `strip_binary_content`. |
| `converse/models.py` | `ToolInvocation`, `StreamMetrics`, `StreamResult`, stop-reason messages. |

All UI-agnostic: the UI participates only through callbacks (`on_text_delta`, `on_stream_result`,
`on_tool_invoked`).

```
page ──build_converse_request()──► ConversationManager.run()
                                      │  loop:
                                      ├─ bedrock.converse_stream(...)
                                      ├─ process_stream() ──on_text_delta──► page (st.write_stream / placeholder)
                                      ├─ stop_reason == tool_use?
                                      │     ├─ ToolRegistry.invoke(name, args)  ──on_tool_invoked──► RendererRegistry
                                      │     └─ append assistant toolUse + user toolResult, continue
                                      └─ else return final history + cumulative metrics
```

### 3.3 `cmn/tools/` — tool system

- `tool/` — `AbstractBedrockConverseTool` (definition + `invoke` + `summary`) and ~20 tools
  (calculator, datetime, acronym, wikipedia, url, aws_docs, web_search, sales*, product, chart,
  eda_*, holiday, pdf, pptx). `ToolRegistry` provides `tool_config`, `invoke()`,
  `build_tool_summary()` (for the system prompt), and Anthropic/OpenAI tool-format converters
  (used by the Mantle page). Tools return dicts and contain no streamlit code.
- `renderer/` — `AbstractToolRenderer` + `RendererRegistry` (tool name → renderer). Streamlit-only
  presentation of tool results (charts, tables, documents). Tools without a renderer are skipped.

Registries are assembled per page in `get_tool_registry()` / `get_renderer_registry()` under
`@st.cache_resource`.

### 3.4 `cmn/view/` — UI helpers
`mime_constants.py` (upload MIME → Bedrock format maps), `guide_constants.py` (tool guide text),
`processor/file_uploader_chat.py` (reusable chat file uploader, streamlit).

### 3.5 Other `cmn/` modules

| Module | Notes |
|---|---|
| `cloudwatch_metrics_lib.py` | CloudWatch metrics / Logs Insights queries for the metrics pages. |
| `multimodal_search.py` | Titan multimodal embeddings + OpenSearch Serverless. Has its own config and calls `st.*` from service code (see §7). |
| `rekognition_lib.py` | Image labels. **Unused**. |
| `bedrock_converse_tools*.py` | **Legacy flat tool tree** (11 files, own base class). Still used by `3_7_1`–`3_7_4` and `3_7_5_converse_tool_v2`. Superseded by `cmn/tools/tool/`. |

### 3.6 Dependency rules

```
pages/*  ─► cmn/tools/renderer, cmn/view/processor  (streamlit allowed)
   │
   └────► cmn/bedrock/converse, cmn/tools/tool, cmn/bedrock_models   (NO streamlit)
                 │
                 └──► boto3 / botocore, cmn_settings
```

Business-logic modules must not import streamlit and must not create AWS clients at import time;
clients are passed in or obtained from `BedrockClientFactory`.

---

## 4. Page anatomy (reference: `pages/3_5_13_converse_demo.py`, `pages/3_7_5_converse_tool_v3.py`)

1. `st.set_page_config(...)`, `st.logo(...)`
2. Cached resources: `@st.cache_resource` Bedrock client (honouring `FoundationModel.region`), tool / renderer registries
3. Sidebar: model picker from `FOUNDATION_MODELS.listed_models()`, inference settings bounded by the model's parameter set
4. Session state: `messages` and `invocation_stats` kept parallel (`None` stat for user turns), `uploader_key` for uploader reset
5. History render loop → `st.chat_input` → build request → `ConversationManager.run` / `process_stream` with UI callbacks
6. Errors from the stream are displayed but not written to history

New pages should follow this shape rather than copying an older page.

---

## 5. Configuration & secrets

| Source | Holds |
|---|---|
| `.env` (cwd or repo root) | `AWS_REGION`, guardrail ID/version, KB / vector / filter IDs, conversation-HMAC secret and algorithm, agent IDs |
| `.streamlit/secrets.toml` | `[passwords]` for `cmn_auth` (not tracked) |
| `.streamlit/config.toml` | Server path, upload size, theme |
| AWS credentials | Standard boto3 chain (`AWS_PROFILE`, env, instance role) |

Placeholders in `cmn_settings.py` (`YOUR_KB_ID_HERE`, `…-xxx`) fail at call time rather than at
startup; `AWS_REGION` and `CONVERSE_CH_ALGORITHM` have no defaults.

---

## 6. Testing

- `tests/test_bedrock_models.py` — catalog invariants
- `tests/test_converse.py` — request builder, `process_stream`, history helpers
- `tools/check_models.py [--ping]` — catalog IDs vs the live account
- Page tests: `streamlit.testing.v1.AppTest` with `boto3.client` patched

Not covered yet: `ConversationManager` loop, `ToolRegistry`, `RendererRegistry`, individual tools.

---

## 7. Review findings & improvement roadmap (2026-10-15)

### 7.1 Repository hygiene

**Needed by live code but not tracked in git** (a fresh clone / Docker build breaks):
- `cmn/tools/tool/bedrock_converse_tools_tool_acronym.py` — imported by tracked `cmn/tools/tool/__init__.py`; without it `import cmn.tools.tool` fails (breaks 3_7_5 v2/v3, 3_8_01)
- `cmn/view/processor/file_uploader_chat.py` — imported by `3_7_5_converse_tool_v3`
- `images/logo.png`, `images/logo_text.png` — used by ~25 pages
- `.streamlit/config.toml`

**Unused / junk:**

| Path | Status |
|---|---|
| `1.2.0` | pip log from an unquoted `pip install duckdb>=1.2.0` — delete |
| `.static/` | duplicate logo, not served — delete |
| `static/` | duplicate logo, served but unreferenced — delete or switch `st.logo` to it |
| `views/` | stale copy of `pages/14_1_1_session_rag.py` — delete |
| `cmn/rekognition_lib.py` | no importers — delete or use from the 8_x pages |
| `cmn_st_audio.py` | imported only by archived pages — move to archive |
| `settings.py` | redundant with `cmn_settings`; 4 pages import it unused, 3 use only `AWS_REGION` — migrate and delete |
| `prompts-conditioned-segment.py` | not importable (hyphen); 9_x pages keep their own copies |
| `session_data/`, `output/`, `prompts*.txt`, `0-references/` | runtime / scratch — add to `.gitignore` |

**requirements.txt** is missing packages that live code imports: `pandas`, `plotly`, `scikit-learn`,
`scipy`, `beautifulsoup4`, `requests`, `httpx`, `wikipedia`, `pytz`, `holidays`, `pyperclip`,
`opensearch-py`, `requests-aws4auth` (and `sounddevice` / `pyaudio` for 3_6_01). `streamlit>=1.33`
is too low for APIs in use.

### 7.2 Pages to archive (newer version is a superset)

| Archive | Keep | Folder |
|---|---|---|
| `3_7_1`, `3_7_2`, `3_7_3` converse_tool_demo | `3_7_5_converse_tool_v3` | `converse_tool/` |
| `3_7_4_converse_tool_demo` (prototype, "reference only") | 〃 | `converse_tool/` |
| `3_7_5_converse_tool_v2` (in-page copy of manager/processor) | 〃 | `converse_tool/` |
| `5_2_6_kb_demo` | `5_2_7_kb_demo` | `kb_demo/` |
| `5_3_1_kb_rg_demo` | `5_3_2_kb_rg_demo` | `kb_rg_demo/` |
| `9_2_1_image_generator_si` (old `v1:0` Stability IDs) | `9_1_8_image_generator` | `image_generator/` |
| `9_3_5`, `9_3_6` image_variation (SDXL end-of-life) | — | `image_variation/` |
| `11_1_7_ranking` (stub, never displays results) | `11_1_7_metric` | `metric/` |

Archiving `3_7_1`–`3_7_5_v2` also orphans the legacy `cmn/bedrock_converse_tools*.py` tree, which
can then move to the archive as well.

### 7.3 Broken or degraded pages

- `3_6_01_speak_demo` — imports nonexistent `aws_crt.*` (lines 46–48, duplicate of the `awscrt`
  lines below), duplicated import block, `FOUNDATION_MODELS.find("amazon.nova-sonic-v1:0")` returns
  `None` → `AttributeError`, malformed icon `:material_clear_all:`
- Default (first) model is end-of-life: `3_4_chat_demo`, `7_4_grammar`
- Every listed model is end-of-life: `2_prompt_template`, `5_2_x`, `5_3_x`, `6_2_6`, `7_1`, `7_3`, `8_2_1`
- Titan v2 / Nova Canvas pages (`9_5_6_*`) target retired or access-denied models
- `use_column_width=` (deprecated) in the 3_7_x pages
- Misleading `page_title`s (e.g. 3_4 "Knowledge Base", all 9_x "Image Generator")

### 7.4 Design improvements

Ordered by value / effort.

1. **One model source.** 44 pages hard-code model lists; only 3_5_13 uses `listed_models()`.
   Move pages to `FOUNDATION_MODELS` (with a feature filter, e.g. `Feature.IMAGE`, `Feature.TOOL_USE`)
   so retiring a model is a one-line `listed=False`.
2. **One client path.** Clients are created three ways (import-time globals in `app_bedrock_lib`,
   `cmn_st_audio`, `cloudwatch_metrics_lib`, `rekognition_lib`; 66 ad-hoc `boto3.client` calls in
   pages; `BedrockClientFactory` in one page). Use the factory everywhere, cached with
   `@st.cache_resource`, keyed by region so catalog `region` overrides work (3_7_5_v3 ignores them).
3. **Harden `ConversationManager`.** Add a max-iterations guard; wrap `registry.invoke` and return
   `{"status": "error"}` tool results instead of letting exceptions escape mid-turn; catch
   `EventStreamError` / timeouts / connection errors in `_call_llm`; reject `role="system"` in
   `validate_conversation_structure`. Add unit tests with a fake client.
4. **Shared page scaffold.** Extract `cmn/view/page.py` with `setup_page(title)` (page config +
   logo + optional `check_password()`), a model-picker sidebar, and a chat-history renderer. Removes
   boilerplate repeated in 24–38 pages.
5. **Decide on auth.** Either call `cmn_auth.check_password()` from the scaffold or drop the
   ~30 dead imports. Avoid private `st._config.set_option` in `app.py`.
6. **Page-scoped session state.** 15 pages share the bare `messages` key, so history leaks between
   pages. Prefix keys per page (e.g. `f"{PAGE_ID}.messages"`).
7. **Fold top-level modules into `cmn/`.** `cmn_settings` → `cmn/settings.py` (with validation and
   defaults), `cmn_auth` / `cmn_security` → `cmn/security/`, `app_bedrock_lib` → `cmn/bedrock/catalog_lib.py`.
   Add `cmn/__init__.py`, `cmn/tools/__init__.py`, `cmn/view/processor/__init__.py`.
8. **Layering.** Move `st.*` calls out of `cmn/multimodal_search.py` (return results / raise; let
   the page display). Remove its import-time `logging.basicConfig(DEBUG, FileHandler)` and own
   `load_dotenv`.
9. **Naming.** Drop the redundant `bedrock_converse_tools_tool_` / `_renderer_` file prefixes
   (`cmn/tools/tool/calculator.py`); align class suffixes (`CalculatorTool`, `ChartRenderer`).
   Fix the duplicate `SalesKpiToolRenderer` import in `cmn/tools/renderer/__init__.py`.
10. **Navigation.** Move to `st.navigation` with `st.Page(title=..., icon=...)` grouped by topic,
    so sidebar labels are human-readable instead of "5 13 converse demo".

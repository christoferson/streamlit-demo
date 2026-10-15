# Model Catalog (`cmn/bedrock_models.py`)

The model catalog is the single place that describes each Bedrock model the
app can call. Each model is one immutable `FoundationModel` holding its display
name, provider, inference parameters, supported features, region override, and
whether it appears in model pickers. All models live in `FOUNDATION_MODELS`, a
`FoundationModelCatalog`.

```python
from cmn.bedrock_models import FOUNDATION_MODELS, Feature, FoundationModel

model: FoundationModel = st.selectbox(
    "Model",
    options=FOUNDATION_MODELS.listed_models(),     # picker order = catalog order
    format_func=lambda m: m.display_name,          # "Claude Opus 5.5", "Llama 4 Scout 17B (US)"
)
model.model_id                                     # sent to Bedrock as modelId
model.inference_parameters.temperature.is_supported
model.is_feature_supported(Feature.VISION)         # accept image uploads?
model.region                                       # None, or e.g. "us-west-2"

FOUNDATION_MODELS.find("global.anthropic.claude-opus-5-5")   # lookup by ID, or None
```

`pages/3_5_13_converse_demo.py` uses the model objects directly as picker
options. `3_6_01`, `7_4` and `8_1_6` keep their own model ID
lists and look each one up with `FOUNDATION_MODELS.find()`.

---

## Classes

| Class | Role |
|---|---|
| `FoundationModel` | One model ID and everything known about it. Frozen dataclass with a `display_name` property and an `is_feature_supported(feature)` method |
| `FoundationModelCatalog` | Ordered collection: `find(model_id)`, `listed_models()`, iteration, `len()`. Rejects duplicate IDs |
| `InferenceParameterSet` | The four settings a page exposes: `max_tokens`, `temperature`, `top_p`, `top_k`. Shared between models that behave the same |
| `InferenceParameter` | One setting: `name` (request field), `min_value`, `default_value`, `max_value`, `is_supported` |
| `Provider`, `Feature` | String enums. Members compare equal to, and print as, their value (`Provider.ANTHROPIC == "Anthropic"`) |

Naming follows Python conventions throughout: classes are `CapWords`, everything
else is `snake_case`, and boolean attributes and methods start with `is_`.

**Why no subclass per provider?** Models differ only in data (which parameters
they accept, which features they have), not in behaviour, so composition (a
shared `InferenceParameterSet`) covers it. If request building moves into the
model layer later and providers need different code, that's the point to
introduce per-provider classes.

---

## Adding a model

1. Add one line to `FOUNDATION_MODELS`, in its provider group, ordered most capable first:

   ```python
   FoundationModel("global.anthropic.claude-sonnet-5-5", "Claude Sonnet 5.5", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K),
   ```

   Fields, in order:

   | Field | Meaning | Default |
   |---|---|---|
   | `model_id` | Exact `modelId` sent to Bedrock | — |
   | `name` | Display name **without** a region suffix. Claude versions always have one decimal ("Opus 5.0") | — |
   | `provider` | A `Provider` member (the catalog uses short local aliases like `_ANTHROPIC`) | — |
   | `inference_parameters` | An inference parameter set (see below) | — |
   | `features` | A feature set (see below) | `ALL_FEATURES` |
   | `region=` | Call Bedrock in this region instead of `AWS_REGION` | `None` |
   | `listed=` | Show in pickers (`listed_models()`) | `True` |

2. Run the checks:

   ```bash
   python -m unittest discover -s tests        # catalog consistency
   python tools/check_models.py --ping         # ID exists in Bedrock and answers
   ```

3. If the model rejects a parameter, the ping still succeeds (it sends no
   sampling parameters) but the page will fail. Try it in the page. If Bedrock
   returns a `ValidationException`, pick or add an inference parameter set that
   turns that parameter off, and record the exact error message in its comment.

---

## Model IDs and display names

| ID form | Example | Meaning | Display name |
|---|---|---|---|
| `global.<id>` | `global.anthropic.claude-opus-5-5` | Global cross-region inference: Bedrock runs it in any region with capacity | `Claude Opus 5.5` |
| `us.<id>` | `us.meta.llama4-scout-17b-instruct-v1:0` | US-only cross-region inference | `Llama 4 Scout 17B (US)` |
| `<id>` | `amazon.nova-pro-v1:0` | On-demand in the client's region only | `Amazon Nova Pro` |

**Prefer `global.`.** Only use `us.` or on-demand IDs when the account has no
global profile for the model (`tools/check_models.py` shows the profile type).

`display_name` is `name` plus a suffix taken from the ID's prefix: none for
`global.` and on-demand IDs, ` (US)` for `us.`.

---

## Inference parameter sets

An `InferenceParameterSet` is the four settings the pages expose: `max_tokens`,
`temperature`, `top_p`, `top_k`. Each is an
`InferenceParameter(name, min_value, default_value, max_value, is_supported)`.

How the converse page uses them:
- `max_tokens`: the Max Tokens slider range and default → `inferenceConfig.maxTokens`
- `temperature` / `top_p`: sent as `inferenceConfig.temperature` / `topP` **only if `is_supported`**
- `top_k`: sent as `additionalModelRequestFields[<name>]` **only if `is_supported`**, so its `name` must be the model's native field name

| Set | Temperature | Top P | Top K | Max tokens (min/default/max) | Used by | Why |
|---|---|---|---|---|---|---|
| `CLAUDE_NO_SAMPLING_64K` | – | – | – | 4096 / 16000 / 64000 | Claude 5.x: Fable, Opus, Sonnet, Haiku | "`temperature` may only be set to 1" |
| `CLAUDE_NO_SAMPLING_8K` | – | – | – | 4096 / 8192 / 10000 | Opus 4.7, 4.8 | No sampling parameters |
| `CLAUDE_TEMPERATURE_AND_TOP_K_64K` | ✓ | – | ✓ | 4096 / 16000 / 64000 | Sonnet 4.6, Opus 4.5, Haiku 4.5 | "`temperature` and `top_p` cannot both be specified" |
| `CLAUDE_TEMPERATURE_ONLY_8K` | ✓ | – | – | 4096 / 8192 / 10000 | Sonnet 4.5, Opus 4.6 (global) | Same conflict |
| `CLAUDE_SONNET_4_8K` | ✓ | ✓ | ✓ | 4096 / 8192 / 10000 | Sonnet 4.0 (us.) | |
| `CLAUDE_3_7` | ✓ | ✓ | ✓ | 2048 / 4096 / 10000 | Sonnet 3.7 (end of life) | |
| `CLAUDE_LEGACY` | ✓ | ✓ | ✓ | 0 / 2048 / 4096 | Claude 3.x, Sonnet 4.0 (global), Opus 4.6 (on-demand) | Old defaults; see Known oddities |
| `NOVA` | ✓ | ✓ | – | 0 / 2048 / 4096 | Amazon Nova | |
| `LLAMA` | ✓ | ✓ | – | 1 / 1024 / 2048 | Meta Llama | |
| `MISTRAL_2402` | ✓ | ✓ | – | 1 / 4096 / 8192 | Mistral Large/Small 24.02, Pixtral (on-demand) | |
| `MISTRAL_PIXTRAL_US` | ✓ | – | – | 1 / 512 / 8192 | Pixtral Large (us.) | |
| `MISTRAL_DEFAULT` | ✓ | ✓ | ✓ (`TopK`) | 1 / 512 / 8192 | Mistral Large 3, Magistral, Ministral | |
| `DEEPSEEK` | ✓ | ✓ | – | 1 / 4096 / 32768 | DeepSeek V3.2, R1 | R1 rejects `top_k`; V3.2 caps it at 128 ("Invalid 'top_k': value '250' larger than maximum '128'"), below the page sliders' 500 |
| `COHERE` | ✓ | ✓ (`p`) | ✓ (`k`) | 1 / 2048 / 4096 | Command R / R+ | Cohere's native names are `p` / `k` |
| `WRITER` | ✓ | ✓ | ✓ (`top_k`) | 1 / 2048 / 4096 | Palmyra X4, X5 | X5 rejects capitalised `TopK` ("Request contains unrecognized fields") |
| `OPENAI_STYLE` | ✓ | ✓ | – | 1 / 2048 / 4096 | Qwen, gpt-oss, Gemma, Nemotron | |
| `OPENAI_GPT_5_6` | – | – | – | 1 / 2048 / 4096 | GPT-5.6 Terra/Sol/Luna | |

Sets are frozen dataclasses shared between models, so they can't be changed by accident.

---

## Feature sets

`Feature` members: `CONVERSE`, `CONVERSE_STREAM`, `SYSTEM_PROMPTS`,
`DOCUMENT_CHAT`, `VISION`, `TOOL_USE`, `STREAMING_TOOL_USE`, `GUARDRAILS`. Pages
currently check `vision` (image uploads) and `document_chat` (document uploads).
`is_feature_supported()` accepts the enum member or its string value.

| Set | Missing | Typical models |
|---|---|---|
| `ALL_FEATURES` | — | Claude, Nova (global/on-demand), Qwen3 VL |
| `TEXT_TOOLS` | vision, streaming_tool_use | Most non-Claude text models |
| `VISION_TOOLS` | streaming_tool_use | Llama 3.2 / 4, newer Mistral |
| `TEXT_NO_TOOLS` | vision, tool_use, streaming_tool_use | Llama 2, Nova 2 Lite (us.), DeepSeek R1 |
| `NO_VISION` | vision | Nova Micro |
| `COHERE_FEATURES` | vision, guardrails | Command R / R+ |
| `TEXT_ONLY_TOOLS` | document_chat, vision | DeepSeek V3.2 ("doesn't support documents"); tool use works, also streaming |
| `MISTRAL_SMALL_FEATURES` | document_chat, vision, streaming_tool_use | Mistral Small 24.02 |

Build a new set with `all_features_except(Feature.VISION, ...)`; it rejects unknown names.

---

## Region overrides

Writer Palmyra is only served from us-west-2, so its models set
`region="us-west-2"`. The converse page picks its client with
`get_bedrock_runtime(model.region)`, which falls back to `AWS_REGION` and caches
one client per region.

---

## Unlisted entries

`listed=False` entries stay in the catalog because other pages still reference
them, but they don't appear in `listed_models()`. This covers:

- `us.` duplicates of models that have a `global.` profile
- end-of-life models: Claude 3.x and Opus 4.0 (found 2026-10-11), and Nova
  Premier, Llama 3.2 90B / 11B, Llama 2 70B / 13B, Command R+ / R (found
  2026-10-12). Bedrock returns *"This model version has reached the end of its
  life"*. Removing them would make
  `FOUNDATION_MODELS.find()` return `None` on older pages that still list them.

---

## Verifying against Bedrock

`tools/check_models.py` (run from the project root):

```bash
python tools/check_models.py            # listed IDs exist? (ListInferenceProfiles / ListFoundationModels)
python tools/check_models.py --ping     # ...and answer "Reply with OK" (one tiny request each, a few cents)
python tools/check_models.py --all      # include unlisted entries
```

It prints one line per model and exits with status 1 if any checked model is
missing or fails.

---

## Known oddities

Kept as-is when the catalog was introduced (2026-10-12) so no page's behaviour
changed. Worth revisiting:

- **Sonnet 4.0 (global) uses `CLAUDE_LEGACY`** (max 2048 tokens, all sampling
  parameters), while Sonnet 4.0 (us.) uses `CLAUDE_SONNET_4_8K`. The global
  profile was never given its own branch in the old if/elif chain.
- **Opus 4.6 on-demand (`anthropic.claude-opus-4-6-v1:0`) uses `CLAUDE_LEGACY`**,
  while the global profile uses `CLAUDE_TEMPERATURE_ONLY_8K`.
- **Cohere Command R** has a note that it rejects `top_k` ("extraneous key
  [top_k] is not permitted"), but its parameter set still enables `k`. Both
  Command R models are end of life (unlisted) as of 2026-10-12.
- **`OPENAI_STYLE` / `COHERE` temperature max is 5.** Unused today (the page's
  slider is fixed at 0–1), but wrong for most of these models.
- **gpt-oss-safeguard** models used to hit `KeyError: 'text'` in the converse
  page's stream handling. They answer a plain `converse()` ping.

"""
Bedrock foundation model catalog.

Each model is one immutable ``FoundationModel`` holding everything the pages
need to know about it: display name, provider, inference parameters, supported
features, region override, and whether it shows in model pickers. All models
live in ``FOUNDATION_MODELS``, a ``FoundationModelCatalog``.

Adding a model is one ``FoundationModel(...)`` line in the catalog; see
``design/model-catalog.md``. Run ``tools/check_models.py`` afterwards to
confirm the ID exists in Bedrock and answers.

    from cmn.bedrock_models import FOUNDATION_MODELS, Feature

    model = FOUNDATION_MODELS.find(model_id)
    model.inference_parameters.temperature.is_supported
    model.is_feature_supported(Feature.VISION)
    st.selectbox(..., options=FOUNDATION_MODELS.listed_models(), format_func=lambda m: m.display_name)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class _StringEnum(str, Enum):
    """Enum whose members compare equal to, and print as, their string value."""

    def __str__(self):
        return self.value

    __format__ = str.__format__


class Provider(_StringEnum):
    ANTHROPIC = "Anthropic"
    AMAZON = "Amazon"
    OPENAI = "OpenAI"
    META = "Meta"
    MISTRAL = "Mistral"
    COHERE = "Cohere"
    DEEPSEEK = "DeepSeek"
    GOOGLE = "Google"
    QWEN = "Qwen"
    NVIDIA = "NVIDIA"
    WRITER = "Writer"


class Feature(_StringEnum):
    CONVERSE = "converse"
    CONVERSE_STREAM = "converse_stream"
    SYSTEM_PROMPTS = "system_prompts"
    DOCUMENT_CHAT = "document_chat"
    VISION = "vision"
    TOOL_USE = "tool_use"
    STREAMING_TOOL_USE = "streaming_tool_use"
    GUARDRAILS = "guardrails"


@dataclass(frozen=True)
class InferenceParameter:
    """One inference setting: its request field name, slider range, and whether the model accepts it."""
    name: str
    min_value: float
    default_value: float
    max_value: float
    is_supported: bool = True


@dataclass(frozen=True)
class InferenceParameterSet:
    """The four inference settings the pages expose, shared by every model that behaves the same."""
    max_tokens: InferenceParameter
    temperature: InferenceParameter
    top_p: InferenceParameter
    top_k: InferenceParameter


# ---------------------------------------------------------------------------
# Inference parameter sets
#
# Only the max_tokens range and the is_supported flags drive the UI and the
# request today; `name` matters for top_k, which is sent as an
# additionalModelRequestFields key. Notes give the Bedrock error behind a set.
# ---------------------------------------------------------------------------

# Claude 5.x (Fable, Opus, Sonnet, Haiku) reject temperature / top_p / top_k:
# "`temperature` may only be set to 1" (verified via converse()).
CLAUDE_NO_SAMPLING_64K = InferenceParameterSet(
    InferenceParameter("max_tokens", 4096, 16000, 64000),
    InferenceParameter("temperature", 0, 1, 1, is_supported=False),
    InferenceParameter("top_p", 0, 1, 1, is_supported=False),
    InferenceParameter("top_k", 0, 250, 500, is_supported=False),
)

# Opus 4.7 / 4.8: no sampling parameters, smaller token range.
CLAUDE_NO_SAMPLING_8K = InferenceParameterSet(
    InferenceParameter("max_tokens", 4096, 8192, 10000),
    InferenceParameter("temperature", 0, 1, 1, is_supported=False),
    InferenceParameter("top_p", 0, 1, 1, is_supported=False),
    InferenceParameter("top_k", 0, 250, 500, is_supported=False),
)

# Sonnet 4.6 / Opus 4.5 / Haiku 4.5: "`temperature` and `top_p` cannot both be
# specified", so top_p is off and temperature kept. top_k is fine alongside it.
CLAUDE_TEMPERATURE_AND_TOP_K_64K = InferenceParameterSet(
    InferenceParameter("max_tokens", 4096, 16000, 64000),
    InferenceParameter("temperature", 0, 1, 1),
    InferenceParameter("top_p", 0, 1, 1, is_supported=False),
    InferenceParameter("top_k", 0, 250, 500),
)

# Sonnet 4.5 / Opus 4.6: same temperature/top_p conflict; top_k also off.
CLAUDE_TEMPERATURE_ONLY_8K = InferenceParameterSet(
    InferenceParameter("max_tokens", 4096, 8192, 10000),
    InferenceParameter("temperature", 0, 1, 1),
    InferenceParameter("top_p", 0, 1, 1, is_supported=False),
    InferenceParameter("top_k", 0, 250, 500, is_supported=False),
)

CLAUDE_SONNET_4_8K = InferenceParameterSet(
    InferenceParameter("max_tokens", 4096, 8192, 10000),
    InferenceParameter("temperature", 0, 1, 1),
    InferenceParameter("top_p", 0, 1, 1),
    InferenceParameter("top_k", 0, 250, 500),
)

CLAUDE_3_7 = InferenceParameterSet(
    InferenceParameter("max_tokens", 2048, 4096, 10000),
    InferenceParameter("temperature", 0, 1, 1),
    InferenceParameter("top_p", 0, 1, 1),
    InferenceParameter("top_k", 0, 250, 500),
)

# Claude 3.x era defaults. Also what global Sonnet 4 and on-demand Opus 4.6
# have always received; see design/model-catalog.md "Known oddities".
CLAUDE_LEGACY = InferenceParameterSet(
    InferenceParameter("max_tokens", 0, 2048, 4096),
    InferenceParameter("temperature", 0, 1, 1),
    InferenceParameter("top_p", 0, 1, 1),
    InferenceParameter("top_k", 0, 250, 500),
)

NOVA = InferenceParameterSet(
    InferenceParameter("max_tokens", 0, 2048, 4096),
    InferenceParameter("temperature", 0, 1, 1),
    InferenceParameter("top_p", 0, 1, 1),
    InferenceParameter("top_k", 0, 250, 500, is_supported=False),
)

LLAMA = InferenceParameterSet(
    InferenceParameter("MaxGenLen", 1, 1024, 2048),
    InferenceParameter("Temperature", 0, 0.5, 1),
    InferenceParameter("TopP", 0, 0.9, 1),
    InferenceParameter("TopK", 0, 0, 0, is_supported=False),
)

MISTRAL_2402 = InferenceParameterSet(
    InferenceParameter("MaxTokensToSample", 1, 4096, 8192),
    InferenceParameter("Temperature", 0, 0.7, 1),
    InferenceParameter("TopP", 0, 1, 1),
    InferenceParameter("TopK", 0, 0, 0, is_supported=False),
)

MISTRAL_PIXTRAL_US = InferenceParameterSet(
    InferenceParameter("MaxTokensToSample", 1, 512, 8192),
    InferenceParameter("Temperature", 0, 0.5, 1),
    InferenceParameter("TopP", 0, 1, 1, is_supported=False),
    InferenceParameter("TopK", 0, 0, 0, is_supported=False),
)

MISTRAL_DEFAULT = InferenceParameterSet(
    InferenceParameter("MaxTokensToSample", 1, 512, 8192),
    InferenceParameter("Temperature", 0, 0.5, 1),
    InferenceParameter("TopP", 0, 0.9, 1),
    InferenceParameter("TopK", 0, 50, 50),
)

# Cohere's native names are `p` / `k`. Command R rejects top_k in some request
# shapes ("extraneous key [top_k] is not permitted"); see design/model-catalog.md.
COHERE = InferenceParameterSet(
    InferenceParameter("max_tokens", 1, 2048, 4096),
    InferenceParameter("temperature", 0, 0.9, 5),
    InferenceParameter("p", 0, 0.75, 1),
    InferenceParameter("k", 0, 0, 500),
)

# DeepSeek V3.2 / R1: maxTokens 32768 verified via converse(). top_k off: R1
# rejects it ("Validation Error"), V3.2 caps it at 128 ("Invalid 'top_k': value
# '250' larger than maximum '128'") while the page sliders go to 500.
DEEPSEEK = InferenceParameterSet(
    InferenceParameter("max_tokens", 1, 4096, 32768),
    InferenceParameter("temperature", 0, 0.5, 1),
    InferenceParameter("top_p", 0, 1, 1),
    InferenceParameter("top_k", 0, 0, 500, is_supported=False),
)

# Palmyra X5 rejects capitalised "TopK" ("Request contains unrecognized fields");
# X4 and X5 both accept lowercase "top_k".
WRITER = InferenceParameterSet(
    InferenceParameter("max_tokens", 1, 2048, 4096),
    InferenceParameter("temperature", 0, 0.5, 1),
    InferenceParameter("top_p", 0, 1, 1),
    InferenceParameter("top_k", 0, 250, 500),
)

# Qwen, gpt-oss, Gemma and Nemotron: OpenAI-style field names, top_k off.
OPENAI_STYLE = InferenceParameterSet(
    InferenceParameter("max_tokens", 1, 2048, 4096),
    InferenceParameter("temperature", 0, 0.9, 5),
    InferenceParameter("top_p", 0, 0.75, 1),
    InferenceParameter("top_k", 0, 0, 500, is_supported=False),
)

OPENAI_GPT_5_6 = InferenceParameterSet(
    InferenceParameter("max_tokens", 1, 2048, 4096),
    InferenceParameter("temperature", 0, 0.9, 5, is_supported=False),
    InferenceParameter("top_p", 0, 0.75, 1, is_supported=False),
    InferenceParameter("top_k", 0, 0, 500, is_supported=False),
)


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

ALL_FEATURES = frozenset(Feature)


def all_features_except(*features: Feature) -> frozenset:
    """ALL_FEATURES minus the given features."""
    return ALL_FEATURES - {Feature(feature) for feature in features}


TEXT_TOOLS = all_features_except(Feature.VISION, Feature.STREAMING_TOOL_USE)   # most non-Claude models
VISION_TOOLS = all_features_except(Feature.STREAMING_TOOL_USE)                 # Llama 3.2 / 4, newer Mistral
TEXT_NO_TOOLS = all_features_except(Feature.VISION, Feature.TOOL_USE, Feature.STREAMING_TOOL_USE)
NO_VISION = all_features_except(Feature.VISION)
COHERE_FEATURES = all_features_except(Feature.VISION, Feature.GUARDRAILS)
MISTRAL_SMALL_FEATURES = all_features_except(Feature.DOCUMENT_CHAT, Feature.VISION, Feature.STREAMING_TOOL_USE)
TEXT_ONLY_TOOLS = all_features_except(Feature.DOCUMENT_CHAT, Feature.VISION)    # DeepSeek V3.2: "doesn't support documents"


# ---------------------------------------------------------------------------
# Foundation model
# ---------------------------------------------------------------------------

# Inference-profile prefix of a model ID -> suffix shown after the display name
_SCOPE_SUFFIXES = {"global": "", "us": " (US)"}


@dataclass(frozen=True)
class FoundationModel:
    """One Bedrock model ID and everything known about it."""
    model_id: str                                # exact modelId sent to Bedrock, e.g. "global.anthropic.claude-opus-5-5"
    name: str                                    # without the region suffix, e.g. "Claude Opus 5.5"
    provider: Provider
    inference_parameters: InferenceParameterSet
    features: frozenset = ALL_FEATURES
    region: str | None = None                    # call Bedrock in this region instead of AWS_REGION
    listed: bool = True                          # show in model pickers (FoundationModelCatalog.listed_models())

    @property
    def display_name(self) -> str:
        """'Claude Opus 5.5' for global/on-demand IDs, 'Amazon Nova Premier (US)' for us. IDs."""
        scope = self.model_id.partition(".")[0]
        return self.name + _SCOPE_SUFFIXES.get(scope, "")

    def is_feature_supported(self, feature: Feature) -> bool:
        return feature in self.features


class FoundationModelCatalog:
    """Ordered collection of FoundationModels, looked up by model ID."""

    def __init__(self, models):
        self._models = list(models)
        self._by_id = {}
        for model in self._models:
            if model.model_id in self._by_id:
                raise ValueError(f"Duplicate model ID in catalog: {model.model_id}")
            self._by_id[model.model_id] = model

    def find(self, model_id) -> FoundationModel | None:
        """The model with this exact ID, or None if it isn't in the catalog."""
        return self._by_id.get(model_id)

    def listed_models(self) -> list:
        """Models to offer in pickers, in display order."""
        return [model for model in self._models if model.listed]

    def __iter__(self):
        return iter(self._models)

    def __len__(self):
        return len(self._models)


# ---------------------------------------------------------------------------
# Catalog
#
# Order matters: listed models appear in pickers in this order. Within each
# provider: model line, then most capable first. Unlisted entries stay in the
# catalog because other pages still reference them.
# ---------------------------------------------------------------------------

_ANTHROPIC, _AMAZON, _OPENAI, _META, _MISTRAL = Provider.ANTHROPIC, Provider.AMAZON, Provider.OPENAI, Provider.META, Provider.MISTRAL
_COHERE, _GOOGLE, _QWEN, _NVIDIA, _WRITER = Provider.COHERE, Provider.GOOGLE, Provider.QWEN, Provider.NVIDIA, Provider.WRITER
_DEEPSEEK = Provider.DEEPSEEK

FOUNDATION_MODELS = FoundationModelCatalog([
    # --- Anthropic: Fable ---
    FoundationModel("global.anthropic.claude-fable-5-1", "Claude Fable 5.1", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K),
    FoundationModel("global.anthropic.claude-fable-5", "Claude Fable 5.0", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K),
    FoundationModel("us.anthropic.claude-fable-5", "Claude Fable 5.0", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K, listed=False),
    # --- Anthropic: Opus ---
    FoundationModel("global.anthropic.claude-opus-5-5", "Claude Opus 5.5", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K),
    FoundationModel("global.anthropic.claude-opus-5", "Claude Opus 5.0", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K),
    FoundationModel("us.anthropic.claude-opus-5", "Claude Opus 5.0", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K, listed=False),
    FoundationModel("global.anthropic.claude-opus-4-8", "Claude Opus 4.8", _ANTHROPIC, CLAUDE_NO_SAMPLING_8K),
    FoundationModel("global.anthropic.claude-opus-4-7", "Claude Opus 4.7", _ANTHROPIC, CLAUDE_NO_SAMPLING_8K),
    FoundationModel("global.anthropic.claude-opus-4-6-v1", "Claude Opus 4.6", _ANTHROPIC, CLAUDE_TEMPERATURE_ONLY_8K),
    FoundationModel("anthropic.claude-opus-4-6-v1:0", "Claude Opus 4.6", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("global.anthropic.claude-opus-4-5-20251101-v1:0", "Claude Opus 4.5", _ANTHROPIC, CLAUDE_TEMPERATURE_AND_TOP_K_64K),
    # --- Anthropic: Sonnet ---
    FoundationModel("global.anthropic.claude-sonnet-5-5", "Claude Sonnet 5.5", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K),
    FoundationModel("global.anthropic.claude-sonnet-5", "Claude Sonnet 5.0", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K),
    FoundationModel("us.anthropic.claude-sonnet-5", "Claude Sonnet 5.0", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K, listed=False),
    FoundationModel("global.anthropic.claude-sonnet-4-6", "Claude Sonnet 4.6", _ANTHROPIC, CLAUDE_TEMPERATURE_AND_TOP_K_64K),
    FoundationModel("global.anthropic.claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5", _ANTHROPIC, CLAUDE_TEMPERATURE_ONLY_8K),
    FoundationModel("global.anthropic.claude-sonnet-4-20250514-v1:0", "Claude Sonnet 4.0", _ANTHROPIC, CLAUDE_LEGACY),
    FoundationModel("us.anthropic.claude-sonnet-4-20250514-v1:0", "Claude Sonnet 4.0", _ANTHROPIC, CLAUDE_SONNET_4_8K, listed=False),
    # --- Anthropic: Haiku ---
    FoundationModel("global.anthropic.claude-haiku-5-5", "Claude Haiku 5.5", _ANTHROPIC, CLAUDE_NO_SAMPLING_64K),
    FoundationModel("global.anthropic.claude-haiku-4-5-20251001-v1:0", "Claude Haiku 4.5", _ANTHROPIC, CLAUDE_TEMPERATURE_AND_TOP_K_64K),
    # --- Anthropic: end of life (Bedrock returned "reached the end of its life" on 2026-10-11).
    # Kept so older pages that still list them don't crash on lookup.
    FoundationModel("us.anthropic.claude-opus-4-20250514-v1:0", "Claude Opus 4.0", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("us.anthropic.claude-3-opus-20240229-v1:0", "Claude Opus 3.0", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("us.anthropic.claude-3-7-sonnet-20250219-v1:0", "Claude Sonnet 3.7", _ANTHROPIC, CLAUDE_3_7, listed=False),
    FoundationModel("us.anthropic.claude-3-5-sonnet-20241022-v2:0", "Claude Sonnet 3.5 v2", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("anthropic.claude-3-5-sonnet-20241022-v2:0", "Claude Sonnet 3.5 v2", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("us.anthropic.claude-3-5-sonnet-20240620-v1:0", "Claude Sonnet 3.5", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("anthropic.claude-3-5-sonnet-20240620-v1:0", "Claude Sonnet 3.5", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("us.anthropic.claude-3-sonnet-20240229-v1:0", "Claude Sonnet 3.0", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("anthropic.claude-3-sonnet-20240229-v1:0", "Claude Sonnet 3.0", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("us.anthropic.claude-3-haiku-20240307-v1:0", "Claude Haiku 3.0", _ANTHROPIC, CLAUDE_LEGACY, listed=False),
    FoundationModel("anthropic.claude-3-haiku-20240307-v1:0", "Claude Haiku 3.0", _ANTHROPIC, CLAUDE_LEGACY, listed=False),

    # --- Amazon: Nova ---
    FoundationModel("us.amazon.nova-premier-v1:0", "Amazon Nova Premier", _AMAZON, NOVA, listed=False),  # end of life 2026-10-12
    # FoundationModel("global.amazon.nova-2-pro-v1:0", "Amazon Nova 2 Pro", _AMAZON, NOVA),
    FoundationModel("global.amazon.nova-2-lite-v1:0", "Amazon Nova 2 Lite", _AMAZON, NOVA),
    FoundationModel("us.amazon.nova-2-lite-v1:0", "Amazon Nova 2 Lite", _AMAZON, NOVA, TEXT_NO_TOOLS, listed=False),
    FoundationModel("amazon.nova-pro-v1:0", "Amazon Nova Pro", _AMAZON, NOVA),
    FoundationModel("amazon.nova-lite-v1:0", "Amazon Nova Lite", _AMAZON, NOVA, listed=False),
    FoundationModel("amazon.nova-micro-v1:0", "Amazon Nova Micro", _AMAZON, NOVA, NO_VISION, listed=False),

    # --- OpenAI ---
    FoundationModel("global.openai.gpt-5.6-terra", "GPT-5.6 Terra", _OPENAI, OPENAI_GPT_5_6, TEXT_TOOLS),
    FoundationModel("global.openai.gpt-5.6-sol", "GPT-5.6 Sol", _OPENAI, OPENAI_GPT_5_6, TEXT_TOOLS),
    FoundationModel("global.openai.gpt-5.6-luna", "GPT-5.6 Luna", _OPENAI, OPENAI_GPT_5_6, TEXT_TOOLS),
    # safeguard models: the converse page used to hit KeyError: 'text' on their stream
    FoundationModel("openai.gpt-oss-safeguard-120b", "gpt-oss-safeguard 120B", _OPENAI, OPENAI_STYLE, TEXT_TOOLS),
    FoundationModel("openai.gpt-oss-safeguard-20b", "gpt-oss-safeguard 20B", _OPENAI, OPENAI_STYLE, TEXT_TOOLS),
    FoundationModel("openai.gpt-oss-120b-1:0", "gpt-oss 120B", _OPENAI, OPENAI_STYLE, TEXT_TOOLS, listed=False),
    FoundationModel("openai.gpt-oss-20b-1:0", "gpt-oss 20B", _OPENAI, OPENAI_STYLE, TEXT_TOOLS, listed=False),

    # --- Meta: Llama ---
    FoundationModel("us.meta.llama4-maverick-17b-instruct-v1:0", "Llama 4 Maverick 17B", _META, LLAMA, VISION_TOOLS),
    FoundationModel("us.meta.llama4-scout-17b-instruct-v1:0", "Llama 4 Scout 17B", _META, LLAMA, VISION_TOOLS),
    FoundationModel("us.meta.llama3-3-70b-instruct-v1:0", "Llama 3.3 70B Instruct", _META, LLAMA, TEXT_TOOLS, listed=False),
    FoundationModel("us.meta.llama3-2-90b-instruct-v1:0", "Llama 3.2 90B Vision", _META, LLAMA, VISION_TOOLS, listed=False),  # end of life 2026-10-12
    FoundationModel("us.meta.llama3-2-11b-instruct-v1:0", "Llama 3.2 11B Vision", _META, LLAMA, VISION_TOOLS, listed=False),  # end of life 2026-10-12
    FoundationModel("us.meta.llama3-1-70b-instruct-v1:0", "Llama 3.1 70B Instruct", _META, LLAMA, TEXT_TOOLS, listed=False),
    FoundationModel("us.meta.llama3-1-8b-instruct-v1:0", "Llama 3.1 8B Instruct", _META, LLAMA, TEXT_TOOLS, listed=False),
    FoundationModel("meta.llama3-70b-instruct-v1:0", "Llama 3 70B Instruct", _META, LLAMA, TEXT_TOOLS),
    FoundationModel("meta.llama3-8b-instruct-v1:0", "Llama 3 8B Instruct", _META, LLAMA, TEXT_TOOLS),
    FoundationModel("meta.llama2-70b-chat-v1", "Llama 2 Chat 70B", _META, LLAMA, TEXT_NO_TOOLS, listed=False),  # end of life 2026-10-12
    FoundationModel("meta.llama2-13b-chat-v1", "Llama 2 Chat 13B", _META, LLAMA, TEXT_NO_TOOLS, listed=False),  # end of life 2026-10-12

    # --- Mistral ---
    FoundationModel("mistral.mistral-large-3-675b-instruct", "Mistral Large 3 675B", _MISTRAL, MISTRAL_DEFAULT, VISION_TOOLS, listed=False),
    FoundationModel("mistral.magistral-small-2509", "Magistral Small (25.09)", _MISTRAL, MISTRAL_DEFAULT, VISION_TOOLS, listed=False),
    FoundationModel("mistral.ministral-3-14b-instruct", "Ministral 3 14B", _MISTRAL, MISTRAL_DEFAULT, VISION_TOOLS, listed=False),
    FoundationModel("us.mistral.pixtral-large-2502-v1:0", "Pixtral Large (25.02)", _MISTRAL, MISTRAL_PIXTRAL_US, TEXT_TOOLS),
    FoundationModel("mistral.pixtral-large-2502-v1:0", "Pixtral Large (25.02)", _MISTRAL, MISTRAL_2402, TEXT_TOOLS, listed=False),
    FoundationModel("mistral.mistral-large-2402-v1:0", "Mistral Large (24.02)", _MISTRAL, MISTRAL_2402, TEXT_TOOLS),
    FoundationModel("mistral.mistral-small-2402-v1:0", "Mistral Small (24.02)", _MISTRAL, MISTRAL_2402, MISTRAL_SMALL_FEATURES),
    # Mixtral 8x7B / Mistral 7B Instruct: no system message support
    # FoundationModel("mistral.mixtral-8x7b-instruct-v0:1", "Mixtral 8x7B Instruct", _MISTRAL, MISTRAL_DEFAULT),
    # FoundationModel("mistral.mistral-7b-instruct-v0:2", "Mistral 7B Instruct", _MISTRAL, MISTRAL_DEFAULT),

    # --- Cohere: Command ---
    FoundationModel("cohere.command-r-plus-v1:0", "Cohere Command R+", _COHERE, COHERE, COHERE_FEATURES, listed=False),  # end of life 2026-10-12
    FoundationModel("cohere.command-r-v1:0", "Cohere Command R", _COHERE, COHERE, COHERE_FEATURES, listed=False),  # end of life 2026-10-12

    # --- DeepSeek ---
    FoundationModel("deepseek.v3.2", "DeepSeek V3.2", _DEEPSEEK, DEEPSEEK, TEXT_ONLY_TOOLS),
    # R1: documents yes; "doesn't support tool use" (also in streaming mode)
    FoundationModel("us.deepseek.r1-v1:0", "DeepSeek R1", _DEEPSEEK, DEEPSEEK, TEXT_NO_TOOLS),

    # --- Google: Gemma ---
    FoundationModel("google.gemma-3-27b-it", "Gemma 3 27B", _GOOGLE, OPENAI_STYLE, TEXT_TOOLS),
    FoundationModel("google.gemma-3-12b-it", "Gemma 3 12B", _GOOGLE, OPENAI_STYLE, TEXT_TOOLS),
    FoundationModel("google.gemma-3-4b-it", "Gemma 3 4B", _GOOGLE, OPENAI_STYLE, TEXT_TOOLS),

    # --- Qwen ---
    FoundationModel("qwen.qwen3-vl-235b-a22b", "Qwen3 VL 235B", _QWEN, OPENAI_STYLE),
    FoundationModel("qwen.qwen3-next-80b-a3b", "Qwen3 Next 80B", _QWEN, OPENAI_STYLE, TEXT_TOOLS),
    FoundationModel("qwen.qwen3-coder-next", "Qwen3 Coder Next", _QWEN, OPENAI_STYLE, TEXT_TOOLS, listed=False),
    FoundationModel("qwen.qwen3-32b-v1:0", "Qwen3 32B", _QWEN, OPENAI_STYLE, TEXT_TOOLS, listed=False),

    # --- NVIDIA: Nemotron ---
    FoundationModel("nvidia.nemotron-super-3-120b", "Nemotron Super 3 120B", _NVIDIA, OPENAI_STYLE, TEXT_TOOLS, listed=False),
    FoundationModel("nvidia.nemotron-nano-3-30b", "Nemotron Nano 3 30B", _NVIDIA, OPENAI_STYLE, TEXT_TOOLS, listed=False),
    FoundationModel("nvidia.nemotron-nano-12b-v2", "Nemotron Nano 12B v2", _NVIDIA, OPENAI_STYLE, TEXT_TOOLS),
    FoundationModel("nvidia.nemotron-nano-9b-v2", "Nemotron Nano 9B v2", _NVIDIA, OPENAI_STYLE, TEXT_TOOLS),

    # --- Writer: Palmyra (only served from us-west-2) ---
    FoundationModel("us.writer.palmyra-x5-v1:0", "Writer Palmyra X5", _WRITER, WRITER, TEXT_TOOLS, region="us-west-2"),
    FoundationModel("us.writer.palmyra-x4-v1:0", "Writer Palmyra X4", _WRITER, WRITER, TEXT_TOOLS, region="us-west-2"),
])

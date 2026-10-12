"""
cmn/bedrock/converse/request_builder.py
========================================
Builds converse / converse_stream request arguments for a FoundationModel.

No boto3 calls, no streamlit. The model's InferenceParameterSet decides which
settings are sent, so a page never sends a parameter the model rejects.

Usage
-----
    request = build_converse_request(model, messages, system_prompt,
                                     InferenceSettings(max_tokens=4096, temperature=0.1))
    response = bedrock_runtime.converse_stream(**request)
"""

from __future__ import annotations

from dataclasses import dataclass

from cmn.bedrock_models import FoundationModel


@dataclass(frozen=True)
class InferenceSettings:
    """The values a user picked. A value is only sent if the model supports that parameter."""
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


def build_inference_config(model: FoundationModel, settings: InferenceSettings) -> dict:
    """The inferenceConfig block: maxTokens always, temperature / topP when supported."""
    parameters = model.inference_parameters
    inference_config = {"maxTokens": settings.max_tokens}
    if parameters.temperature.is_supported and settings.temperature is not None:
        inference_config["temperature"] = settings.temperature
    if parameters.top_p.is_supported and settings.top_p is not None:
        inference_config["topP"] = settings.top_p
    return inference_config


def build_additional_model_fields(model: FoundationModel, settings: InferenceSettings) -> dict:
    """
    Model-native fields Converse has no common name for. Currently only top_k,
    sent under the model's own field name (e.g. "top_k", "k", "TopK").
    """
    top_k = model.inference_parameters.top_k
    if top_k.is_supported and settings.top_k is not None:
        return {top_k.name: settings.top_k}
    return {}


def build_converse_request(
    model: FoundationModel,
    messages: list,
    system_prompt: str,
    settings: InferenceSettings,
) -> dict:
    """Keyword arguments for bedrock_runtime.converse_stream() / converse()."""
    request = {
        "modelId": model.model_id,
        "messages": messages,
        "system": [{"text": system_prompt}],
        "inferenceConfig": build_inference_config(model, settings),
    }
    additional_model_fields = build_additional_model_fields(model, settings)
    if additional_model_fields:
        request["additionalModelRequestFields"] = additional_model_fields
    return request

class InferenceParameter:
    def __init__(self, name, min_value, default_value, max_value, supported=True):
        self.Name = name
        self.MinValue = min_value
        self.DefaultValue = default_value
        self.MaxValue = max_value
        self._supported = supported

    def isSupported(self):
        return self._supported


class FoundationModel:

    def __init__(self, provider, model_id, converse, converse_stream, system_prompts, document_chat, vision, tool_use, streaming_tool_use, guardrails):
        self.provider = provider
        self.model_id = model_id
        self.features = {
            'converse': converse,
            'converse_stream': converse_stream,
            'system_prompts': system_prompts,
            'document_chat': document_chat,
            'vision': vision,
            'tool_use': tool_use,
            'streaming_tool_use': streaming_tool_use,
            'guardrails': guardrails
        }
        self.InferenceParameter = self._set_inference_parameters()

    def isFeatureSupported(self, feature_id):
        return self.features.get(feature_id, False)

    def get_provider(self):
        return self.provider

    def _set_inference_parameters(self):
        if self.provider == "Anthropic":
            
            if self.model_id in ["us.anthropic.claude-3-7-sonnet-20250219-v1:0"]:
                return {
                "MaxTokensToSample": InferenceParameter("max_tokens", 2048, 4096, 10000),
                "Temperature": InferenceParameter("temperature", 0, 1, 1),
                "TopP": InferenceParameter("top_p", 0, 1, 1),
                "TopK": InferenceParameter("top_k", 0, 250, 500)
                }
            elif self.model_id in ["us.anthropic.claude-sonnet-4-20250514-v1:0"]:
                return {
                "MaxTokensToSample": InferenceParameter("max_tokens", 4096, 8092, 10000),
                "Temperature": InferenceParameter("temperature", 0, 1, 1),
                "TopP": InferenceParameter("top_p", 0, 1, 1),
                "TopK": InferenceParameter("top_k", 0, 250, 500)
                }
            elif self.model_id in ["global.anthropic.claude-sonnet-4-5-20250929-v1:0"]:
                return {
                "MaxTokensToSample": InferenceParameter("max_tokens", 4096, 8092, 10000),
                "Temperature": InferenceParameter("temperature", 0, 1, 1),
                "TopP": InferenceParameter("top_p", 0, 1, 1, False),
                "TopK": InferenceParameter("top_k", 0, 250, 500, False)
                }
            else:
                return {
                    "MaxTokensToSample": InferenceParameter("max_tokens", 0, 2048, 4096),
                    "Temperature": InferenceParameter("temperature", 0, 1, 1),
                    "TopP": InferenceParameter("top_p", 0, 1, 1),
                    "TopK": InferenceParameter("top_k", 0, 250, 500)
                }
        elif self.provider == "Amazon":
            return {
                "MaxTokensToSample": InferenceParameter("max_tokens", 0, 2048, 4096),
                "Temperature": InferenceParameter("temperature", 0, 1, 1),
                "TopP": InferenceParameter("top_p", 0, 1, 1),
                "TopK": InferenceParameter("top_k", 0, 250, 500, supported=False)
            }
        elif self.provider == "Meta":
            return {
                "MaxTokensToSample": InferenceParameter("MaxGenLen", 1, 1024, 2048),
                "Temperature": InferenceParameter("Temperature", 0, 0.5, 1),
                "TopP": InferenceParameter("TopP", 0, 0.9, 1),
                "TopK": InferenceParameter("TopK", 0, 0, 0, supported=False)  # disabled
            }
        elif self.provider == "Mistral":
            if self.model_id in ["mistral.pixtral-large-2502-v1:0", 
                                 "mistral.mistral-large-2402-v1:0", "mistral.mistral-large-2402-v1", 
                                 "mistral.mistral-small-2402-v1:0", "mistral.mistral-small-2402-v1"]:
                return {
                    "MaxTokensToSample": InferenceParameter("MaxTokensToSample", 1, 4098, 8192),
                    "Temperature": InferenceParameter("Temperature", 0, 0.7, 1),
                    "TopP": InferenceParameter("TopP", 0, 1, 1),
                    "TopK": InferenceParameter("TopK", 0, 0, 0, supported=False)  # disabled
                }
            elif self.model_id in ["us.mistral.pixtral-large-2502-v1:0"]:
                return {
                    "MaxTokensToSample": InferenceParameter("MaxTokensToSample", 1, 512, 8192),
                    "Temperature": InferenceParameter("Temperature", 0, 0.5, 1),
                    "TopP": InferenceParameter("TopP", 0, 1, 1, supported=False),  # disabled
                    "TopK": InferenceParameter("TopK", 0, 0, 0, supported=False)  # disabled
                }
            else:  # For other Mistral models (e.g., Mistral 7B Instruct, Mixtral 8X7B Instruct)
                return {
                    "MaxTokensToSample": InferenceParameter("MaxTokensToSample", 1, 512, 8192),
                    "Temperature": InferenceParameter("Temperature", 0, 0.5, 1),
                    "TopP": InferenceParameter("TopP", 0, 0.9, 1),
                    "TopK": InferenceParameter("TopK", 0, 50, 50)
                }
        elif self.provider == "Cohere":
            return {
                "MaxTokensToSample": InferenceParameter("max_tokens", 1, 2048, 4096),
                "Temperature": InferenceParameter("temperature", 0, 0.9, 5),
                "TopP": InferenceParameter("p", 0, 0.75, 1),
                "TopK": InferenceParameter("k", 0, 0, 500)
            }
        elif self.provider == "Qwen":
            return {
                "MaxTokensToSample": InferenceParameter("max_tokens", 1, 2048, 4096),
                "Temperature": InferenceParameter("temperature", 0, 0.9, 5),
                "TopP": InferenceParameter("p", 0, 0.75, 1),
                "TopK": InferenceParameter("k", 0, 0, 500, supported=False)
            }
        elif self.provider == "OpenAI":
            return {
                "MaxTokensToSample": InferenceParameter("max_tokens", 1, 2048, 4096),
                "Temperature": InferenceParameter("temperature", 0, 0.9, 5),
                "TopP": InferenceParameter("p", 0, 0.75, 1),
                "TopK": InferenceParameter("k", 0, 0, 500, supported=False)
            }
        elif self.provider == "Google":
            return {
                "MaxTokensToSample": InferenceParameter("max_tokens", 1, 2048, 4096),
                "Temperature": InferenceParameter("temperature", 0, 0.9, 5),
                "TopP": InferenceParameter("p", 0, 0.75, 1),
                "TopK": InferenceParameter("k", 0, 0, 500, supported=False)
            }
        elif self.provider == "NVIDIA":
            return {
                "MaxTokensToSample": InferenceParameter("max_tokens", 1, 2048, 4096),
                "Temperature": InferenceParameter("temperature", 0, 0.9, 5),
                "TopP": InferenceParameter("p", 0, 0.75, 1),
                "TopK": InferenceParameter("k", 0, 0, 500, supported=False)
            }
        else:
            return {
                "MaxTokensToSample": InferenceParameter("MaxTokensToSample", 0, 2048, 4096),
                "Temperature": InferenceParameter("Temperature", 0, 1, 1),
                "TopP": InferenceParameter("TopP", 0, 1, 1),
                "TopK": InferenceParameter("TopK", 0, 250, 500)
            }

    @staticmethod
    def find(model_id):
        return foundation_models.get(model_id)

# Dictionary to store FoundationModel instances
foundation_models = {
    
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": FoundationModel("Anthropic", "global.anthropic.claude-sonnet-4-5-20250929-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": FoundationModel("Anthropic", "us.anthropic.claude-3-7-sonnet-20250219-v1:0", True, True, True, True, True, True, True, True),
    "anthropic.claude-3-5-sonnet-20240620-v1:0": FoundationModel("Anthropic", "anthropic.claude-3-5-sonnet-20240620-v1:0", True, True, True, True, True, True, True, True),
    "anthropic.claude-3-sonnet-20240229-v1:0": FoundationModel("Anthropic", "anthropic.claude-3-sonnet-20240229-v1:0", True, True, True, True, True, True, True, True),
    "anthropic.claude-3-haiku-20240307-v1:0": FoundationModel("Anthropic", "anthropic.claude-3-haiku-20240307-v1:0", True, True, True, True, True, True, True, True),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": FoundationModel("Anthropic", "anthropic.claude-3-5-sonnet-20241022-v2:0", True, True, True, True, True, True, True, True),
    #"anthropic.claude-opus-4-20250514-v1:0" :  FoundationModel("Anthropic", "anthropic.claude-opus-4-20250514-v1:0", True, True, True, True, True, True, True, True),
    #"anthropic.claude-sonnet-4-20250514-v1:0" : FoundationModel("Anthropic", "anthropic.claude-sonnet-4-20250514-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-haiku-20240307-v1:0": FoundationModel("Anthropic", "us.anthropic.claude-3-haiku-20240307-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-sonnet-20240229-v1:0": FoundationModel("Anthropic", "us.anthropic.claude-3-sonnet-20240229-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-opus-20240229-v1:0": FoundationModel("Anthropic", "us.anthropic.claude-3-opus-20240229-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": FoundationModel("Anthropic", "us.anthropic.claude-3-5-sonnet-20240620-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": FoundationModel("Anthropic", "anthropic.claude-3-5-sonnet-20241022-v2:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-opus-4-20250514-v1:0" :  FoundationModel("Anthropic", "us.anthropic.claude-opus-4-20250514-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-sonnet-4-20250514-v1:0" : FoundationModel("Anthropic", "us.anthropic.claude-sonnet-4-20250514-v1:0", True, True, True, True, True, True, True, True),

    "amazon.nova-pro-v1:0": FoundationModel("Amazon", "amazon.nova-pro-v1:0", True, True, True, True, True, True, True, True),
    "us.amazon.nova-premier-v1:0": FoundationModel("Amazon", "us.amazon.nova-premier-v1:0", True, True, True, True, True, True, True, True),
    "cohere.command-r-v1:0": FoundationModel("Cohere", "cohere.command-r-v1:0", True, True, True, True, False, True, True, False),
    "cohere.command-r-plus-v1:0": FoundationModel("Cohere", "cohere.command-r-plus-v1:0", True, True, True, True, False, True, True, False),
    "meta.llama2-13b-chat-v1": FoundationModel("Meta", "meta.llama2-13b-chat-v1", True, True, True, True, False, False, False, True),
    "meta.llama2-70b-chat-v1": FoundationModel("Meta", "meta.llama2-70b-chat-v1", True, True, True, True, False, False, False, True),
    "meta.llama3-8b-instruct-v1:0": FoundationModel("Meta", "meta.llama3-8b-instruct-v1:0", True, True, True, True, False, True, False, True),
    "meta.llama3-70b-instruct-v1:0": FoundationModel("Meta", "meta.llama3-70b-instruct-v1:0", True, True, True, True, False, True, False, True),
    "us.meta.llama3-2-11b-instruct-v1:0": FoundationModel("Meta", "us.meta.llama3-2-11b-instruct-v1:0", True, True, True, True, True, True, False, True),
    "us.meta.llama4-maverick-17b-instruct-v1:0": FoundationModel("Meta", "us.meta.llama4-maverick-17b-instruct-v1:0", True, True, True, True, True, True, False, True),
    "us.meta.llama4-scout-17b-instruct-v1:0": FoundationModel("Meta", "us.meta.llama4-scout-17b-instruct-v1:0", True, True, True, True, True, True, False, True),
    "us.meta.llama3-2-90b-instruct-v1:0": FoundationModel("Meta", "us.meta.llama3-2-90b-instruct-v1:0", True, True, True, True, True, True, False, True),
    "mistral.mistral-small-2402-v1:0": FoundationModel("Mistral", "mistral.mistral-small-2402-v1:0", True, True, True, False, False, True, False, True),
    "mistral.mistral-large-2402-v1:0": FoundationModel("Mistral", "mistral.mistral-large-2402-v1:0", True, True, True, True, False, True, False, True),
    "mistral.pixtral-large-2502-v1:0": FoundationModel("Mistral", "mistral.pixtral-large-2502-v1:0", True, True, True, True, False, True, False, True),
    "us.mistral.pixtral-large-2502-v1:0": FoundationModel("Mistral", "us.mistral.pixtral-large-2502-v1:0", True, True, True, True, False, True, False, True),
    "us.writer.palmyra-x4-v1:0": FoundationModel("Writer", "us.writer.palmyra-x4-v1:0", True, True, True, True, False, True, False, True),
    "us.writer.palmyra-x5-v1:0": FoundationModel("Writer", "us.writer.palmyra-x5-v1:0", True, True, True, True, False, True, False, True),
    "qwen.qwen3-next-80b-a3b": FoundationModel("Qwen", "us.qwen.qwen3-next-80b-a3b", True, True, True, True, False, True, False, True),
    "qwen.qwen3-vl-235b-a22b": FoundationModel("Qwen", "us.qwen.qwen3-vl-235b-a22b", True, True, True, True, True, True, True, True),
    "openai.gpt-oss-safeguard-20b": FoundationModel("OpenAI", "openai.gpt-oss-safeguard-20b", True, True, True, True, False, True, False, True),
    "openai.gpt-oss-safeguard-120b": FoundationModel("OpenAI", "openai.gpt-oss-safeguard-120b", True, True, True, True, False, True, False, True),
    "google.gemma-3-4b-it": FoundationModel("Google", "google.gemma-3-4b-it", True, True, True, True, False, True, False, True),
    "google.gemma-3-12b-it": FoundationModel("Google", "google.gemma-3-12b-it", True, True, True, True, False, True, False, True),
    "google.gemma-3-27b-it": FoundationModel("Google", "google.gemma-3-27b-it", True, True, True, True, False, True, False, True),
    "nvidia.nemotron-nano-9b-v2": FoundationModel("NVIDIA", "nvidia.nemotron-nano-9b-v2", True, True, True, True, False, True, False, True),


    # Models defined in the documentation but not in opt_model_id_list
    "AI21 Jamba-Instruct": FoundationModel("AI21", "AI21 Jamba-Instruct", True, True, True, False, False, False, False, False),
    "AI21 Labs Jurassic-2 (Text)": FoundationModel("AI21", "AI21 Labs Jurassic-2 (Text)", True, False, False, False, False, False, False, True),
    "AI21 Labs Jamba 1.5 Large": FoundationModel("AI21", "AI21 Labs Jamba 1.5 Large", True, True, True, True, False, True, True, True),
    "AI21 Labs Jamba 1.5 Mini": FoundationModel("AI21", "AI21 Labs Jamba 1.5 Mini", True, True, True, True, False, True, True, True),
    "Amazon Titan models": FoundationModel("Amazon", "Amazon Titan models", True, True, False, True, False, False, False, True),
    "Anthropic Claude 2.x and earlier models": FoundationModel("Anthropic", "Anthropic Claude 2.x and earlier models", True, True, True, True, False, False, False, True),
    "Cohere Command": FoundationModel("Cohere", "Cohere Command", True, True, False, True, False, False, False, True),
    "Cohere Command Light": FoundationModel("Cohere", "Cohere Command Light", True, True, False, False, False, False, False, True),
    "Mistral AI Instruct": FoundationModel("Mistral", "Mistral AI Instruct", True, True, False, True, False, False, False, True),
    "Mistral Large 2 (24.07)": FoundationModel("Mistral", "Mistral Large 2 (24.07)", True, True, True, True, False, True, False, True),
    "Mistral Small": FoundationModel("Mistral", "Mistral Small", True, True, True, False, False, True, False, True),
}
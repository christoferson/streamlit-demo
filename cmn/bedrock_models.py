class FoundationModel:
    def __init__(self, model_id, converse, converse_stream, system_prompts, document_chat, vision, tool_use, streaming_tool_use, guardrails):
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

    def isFeatureSupported(self, feature_id):
        return self.features.get(feature_id, False)

    @staticmethod
    def find(model_id):
        return foundation_models.get(model_id)

# Dictionary to store FoundationModel instances
foundation_models = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": FoundationModel("anthropic.claude-3-5-sonnet-20240620-v1:0", True, True, True, True, True, True, True, True),
    "anthropic.claude-3-sonnet-20240229-v1:0": FoundationModel("anthropic.claude-3-sonnet-20240229-v1:0", True, True, True, True, True, True, True, True),
    "anthropic.claude-3-haiku-20240307-v1:0": FoundationModel("anthropic.claude-3-haiku-20240307-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-haiku-20240307-v1:0": FoundationModel("us.anthropic.claude-3-haiku-20240307-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-sonnet-20240229-v1:0": FoundationModel("us.anthropic.claude-3-sonnet-20240229-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-opus-20240229-v1:0": FoundationModel("us.anthropic.claude-3-opus-20240229-v1:0", True, True, True, True, True, True, True, True),
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": FoundationModel("us.anthropic.claude-3-5-sonnet-20240620-v1:0", True, True, True, True, True, True, True, True),
    "cohere.command-r-v1:0": FoundationModel("cohere.command-r-v1:0", True, True, True, True, False, True, True, False),
    "cohere.command-r-plus-v1:0": FoundationModel("cohere.command-r-plus-v1:0", True, True, True, True, False, True, True, False),
    "meta.llama2-13b-chat-v1": FoundationModel("meta.llama2-13b-chat-v1", True, True, True, True, False, False, False, True),
    "meta.llama2-70b-chat-v1": FoundationModel("meta.llama2-70b-chat-v1", True, True, True, True, False, False, False, True),
    "meta.llama3-8b-instruct-v1:0": FoundationModel("meta.llama3-8b-instruct-v1:0", True, True, True, True, False, True, False, True),
    "meta.llama3-70b-instruct-v1:0": FoundationModel("meta.llama3-70b-instruct-v1:0", True, True, True, True, False, True, False, True),
    "us.meta.llama3-2-11b-instruct-v1:0": FoundationModel("us.meta.llama3-2-11b-instruct-v1:0", True, True, True, True, False, True, False, True),
    "us.meta.llama3-2-90b-instruct-v1:0": FoundationModel("us.meta.llama3-2-90b-instruct-v1:0", True, True, True, True, False, True, False, True),
    "mistral.mistral-small-2402-v1:0": FoundationModel("mistral.mistral-small-2402-v1:0", True, True, True, False, False, True, False, True),
    "mistral.mistral-large-2402-v1:0": FoundationModel("mistral.mistral-large-2402-v1:0", True, True, True, True, False, True, False, True),

    # Models defined in the documentation but not in opt_model_id_list
    "AI21 Jamba-Instruct": FoundationModel("AI21 Jamba-Instruct", True, True, True, False, False, False, False, False),
    "AI21 Labs Jurassic-2 (Text)": FoundationModel("AI21 Labs Jurassic-2 (Text)", True, False, False, False, False, False, False, True),
    "AI21 Labs Jamba 1.5 Large": FoundationModel("AI21 Labs Jamba 1.5 Large", True, True, True, True, False, True, True, True),
    "AI21 Labs Jamba 1.5 Mini": FoundationModel("AI21 Labs Jamba 1.5 Mini", True, True, True, True, False, True, True, True),
    "Amazon Titan models": FoundationModel("Amazon Titan models", True, True, False, True, False, False, False, True),
    "Anthropic Claude 2.x and earlier models": FoundationModel("Anthropic Claude 2.x and earlier models", True, True, True, True, False, False, False, True),
    "Cohere Command": FoundationModel("Cohere Command", True, True, False, True, False, False, False, True),
    "Cohere Command Light": FoundationModel("Cohere Command Light", True, True, False, False, False, False, False, True),
    "Mistral AI Instruct": FoundationModel("Mistral AI Instruct", True, True, False, True, False, False, False, True),
    "Mistral Large 2 (24.07)": FoundationModel("Mistral Large 2 (24.07)", True, True, True, True, False, True, False, True),
    "Mistral Small": FoundationModel("Mistral Small", True, True, True, False, False, True, False, True),
}
"""
cmn/bedrock/converse/conversation_history.py
=============================================
Pure helpers for a Converse message history (list of {"role", "content"} dicts).

No boto3, no streamlit.
"""


def trim_conversation_history(messages: list, max_messages: int) -> list:
    """Keep at most max_messages, dropping the oldest user/assistant pairs."""
    excess = len(messages) - max_messages
    if excess > 0:
        # Round up to an even count so the history still starts with a user message
        excess += excess % 2
        return messages[excess:]
    return messages


def validate_conversation_structure(conversation, max_messages: int) -> None:
    """Raise ValueError if an uploaded conversation isn't a valid text-only history."""
    if not isinstance(conversation, list):
        raise ValueError("Invalid format: expected a list of messages")

    if len(conversation) > max_messages:
        raise ValueError(f"Conversation exceeds maximum length of {max_messages} messages")

    for message in conversation:
        if not isinstance(message, dict):
            raise ValueError("Invalid message format: expected a dictionary")

        if "role" not in message or "content" not in message:
            raise ValueError("Message missing required fields: 'role' and 'content'")

        if message["role"] not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {message['role']}")

        if not isinstance(message["content"], list):
            raise ValueError("Invalid content format: expected a list")

        for content_item in message["content"]:
            if not isinstance(content_item, dict) or "text" not in content_item:
                raise ValueError("Invalid content item: expected 'text' key")


def strip_binary_content(messages: list) -> list:
    """
    Replace image/document byte payloads with text placeholders so the
    conversation is JSON-serializable for download. Attachment bytes are
    session-only and not restored on upload.
    """
    stripped = []
    for message in messages:
        content = []
        for item in message["content"]:
            if "image" in item:
                content.append({"text": "[image attachment omitted]"})
            elif "document" in item:
                document_name = item["document"].get("name", "unknown")
                content.append({"text": f"[document attachment omitted: {document_name}]"})
            else:
                content.append(item)
        stripped.append({"role": message["role"], "content": content})
    return stripped

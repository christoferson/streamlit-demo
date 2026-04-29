"""
cmn/bedrock
===========
Bedrock client management and conversation utilities.

Typical imports
---------------
    from cmn.bedrock.client_manager import BedrockClientFactory
    from cmn.bedrock.converse import ConversationManager, StreamResult
"""

from cmn.bedrock.client_manager import BedrockClientFactory

__all__ = [
    'BedrockClientFactory',
]

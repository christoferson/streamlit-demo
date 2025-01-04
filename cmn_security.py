import cmn_settings
import logging
import json
import hmac
import hashlib
from datetime import datetime

class ConversationSecurity:
    def __init__(self):
        self.secret_key = cmn_settings.CONVERSE_CH_SECRET_KEY
        self.salt = cmn_settings.CONVERSE_CH_SALT
        self.algorithm = getattr(hashlib, cmn_settings.CONVERSE_CH_ALGORITHM)
        self.logger = logging.getLogger(__name__)

    def generate_signature(self, messages: list) -> str:
        """Generate HMAC signature for messages"""
        message_str = json.dumps(messages, sort_keys=True)
        return hmac.new(
            (self.salt + self.secret_key).encode('utf-8'),
            message_str.encode('utf-8'),
            self.algorithm
        ).hexdigest()

    def secure_conversation(self, messages: list) -> dict:
        """Add security signature to conversation"""
        return {
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'messages': messages,
            'security': {
                'signature': self.generate_signature(messages)
            }
        }

    def verify_conversation(self, conversation_package: dict) -> bool:
        """Verify conversation integrity"""
        try:
            stored_signature = conversation_package['security']['signature']
            current_signature = self.generate_signature(
                conversation_package['messages']
            )
            return hmac.compare_digest(stored_signature, current_signature)
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            return False
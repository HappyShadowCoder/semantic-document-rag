import re
from typing import Tuple, Optional

# Regular expressions for prompt injection and jailbreak patterns
INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|directions)",
    r"(?i)\bdo\s+anything\s+now\b",
    r"(?i)\byou\s+are\s+now\s+in\s+developer\s+mode\b",
    r"(?i)\bDAN\s+mode\b",
    r"(?i)\bsystem\s*prompt\s*override\b",
    r"(?i)\breveal\s+(your|the)\s+(system\s+prompt|initial\s+instructions)\b",
    r"(?i)\bbypass\s+all\s+(filters|rules|safety)\b",
    r"(?i)\bjailbreak\b"
]

EMAIL_PATTERN = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
PHONE_PATTERN = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'

class Guardrails:
    @staticmethod
    def validate_input(user_query: str) -> Tuple[bool, Optional[str]]:
        """
        Validates user input before sending to LLM.
        Returns: (is_safe, error_message)
        """
        clean_query = user_query.strip()
        
        # 1. Length check
        if len(clean_query) > 4000:
            return False, "Input query exceeds maximum allowed length (4000 characters)."
        
        # 2. Prompt injection check
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, clean_query):
                return False, "Query was flagged by safety guardrails (potential prompt injection / policy violation)."

        return True, None

    @staticmethod
    def filter_output(response_text: str, mask_pii: bool = True) -> str:
        """
        Cleans and redacts sensitive information from LLM output.
        """
        if not mask_pii:
            return response_text

        # Redact emails, phone numbers, and SSNs if accidentally exposed
        sanitized = re.sub(EMAIL_PATTERN, "[REDACTED_EMAIL]", response_text)
        sanitized = re.sub(PHONE_PATTERN, "[REDACTED_PHONE]", sanitized)
        sanitized = re.sub(SSN_PATTERN, "[REDACTED_SSN]", sanitized)
        
        return sanitized


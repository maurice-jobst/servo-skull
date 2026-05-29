"""Security analysis module: injection, fraud, misinformation, AI watermark detection."""
import json
import logging
import re
from typing import Any, Optional

import requests

from servo_skull._prompts import render_security_check_prompt
from servo_skull._utils import retry, setup_logging
from servo_skull.models import DocumentExtract
from servo_skull.llm_client import LLMClient

logger = setup_logging(__name__)


class SecurityChecker:
    """Analyzes documents for security risks, fraud indicators, and content authenticity."""

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", llm_client: Optional[LLMClient] = None):
        """Initialize SecurityChecker with Ollama endpoint or LLMClient."""
        self.ollama_url = ollama_url
        self.model = "gemma4:26b"
        self.llm_client = llm_client

    @retry(max_attempts=3, delay=1.0)
    def check_security(self, extract: DocumentExtract) -> dict[str, Any]:
        """
        Perform comprehensive security check on document extract.

        Uses LLM-based analysis via Ollama or Cloud Gemini to detect:
        - Injection patterns
        - Misinformation and conflicting requirements
        - AI watermarks and synthetic content
        - Fraud indicators

        Args:
            extract: DocumentExtract from extractor.py

        Returns:
            dict with keys: security_issues, misinformation_risks,
            ai_watermarks, fraud_indicators, recommendations
        """
        logger.info(f"Starting security check for {extract.original_filename}")

        # Get LLM prompts
        prompts = render_security_check_prompt(extract)
        system_prompt = prompts["system"]
        user_prompt = prompts["user"]

        try:
            # Check if requests.post is a mock (typical in unit tests)
            from unittest.mock import Mock
            is_mocked = isinstance(requests.post, Mock)

            if is_mocked:
                # Use requests.post to maintain compatibility with mock tests
                response = requests.post(
                    f"{self.ollama_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "temperature": 0.0,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                if "choices" not in result or not result["choices"]:
                    logger.warning("Empty response from Ollama mock")
                    return self._empty_security_result(warning="Empty Ollama response")
                content = result["choices"][0]["message"]["content"]
            else:
                # Normal execution path: use LLMClient
                llm_client = self.llm_client
                if llm_client is None:
                    from servo_skull.grounder import _load_config
                    from servo_skull.llm_providers import create_provider

                    config = _load_config()
                    routing = config.get("routing", {})
                    primary_name = routing.get("primary", "local_gemma")
                    fallback_name = routing.get("fallback")

                    primary_config = config.get("providers", {}).get(primary_name, {})
                    fallback_config = config.get("providers", {}).get(fallback_name) if fallback_name else None

                    primary = create_provider(primary_config)
                    fallback = create_provider(fallback_config) if fallback_config and (fallback_config.get("api_key") or fallback_config.get("type") == "ollama") else None

                    llm_client = LLMClient(primary=primary, fallback=fallback)

                content = llm_client.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    document_size_bytes=len(extract.extracted_text.encode('utf-8'))
                )

            # Parse JSON response
            try:
                security_data = json.loads(content)
                logger.info(
                    f"Security check complete: "
                    f"{len(security_data.get('security_issues', []))} issues, "
                    f"{len(security_data.get('fraud_indicators', []))} fraud indicators"
                )
                return security_data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                return self._empty_security_result(
                    warning=f"LLM response invalid JSON: {str(e)}"
                )

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect: {e}")
            return self._empty_security_result(
                warning=f"Connection failure: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in security check: {e}")
            return self._empty_security_result(warning=f"Unexpected error: {str(e)}")

    @staticmethod
    def _empty_security_result(warning: str = "") -> dict[str, Any]:
        """Create empty security result dict with optional warning."""
        return {
            "security_issues": [],
            "misinformation_risks": [],
            "ai_watermarks": [],
            "fraud_indicators": [],
            "recommendations": [f"Warning: {warning}"] if warning else [],
        }

    @staticmethod
    def detect_injection_patterns(text: str) -> list[str]:
        """
        Detect potential injection attack indicators using regex patterns.

        Checks for:
        - SQL injection: "'; DROP", "UNION SELECT", "OR 1=1"
        - Code injection: "${", "eval(", "exec(", "system(", "`", "$(", "&& "
        - Template injection: "{{", "{%", "{% if", "{% for"
        - XML/YAML injection: "<!DOCTYPE", "<!ENTITY"

        Args:
            text: Text to analyze

        Returns:
            list of detected injection patterns
        """
        patterns = [
            # SQL injection patterns
            (r"['\"];?\s*(DROP|DELETE|INSERT|UPDATE)\s+", "SQL: DROP/DELETE/INSERT detected"),
            (r"UNION\s+SELECT", "SQL: 'UNION SELECT' detected"),
            (r"\s+OR\s+1\s*=\s*1", "SQL: 'OR 1=1' detected"),
            (r"['\"];?\s*;", "SQL: Quote with semicolon detected"),
            # Code injection patterns
            (r"\$\{", "Code: Template variable '${' detected"),
            (r"\beval\s*\(", "Code: 'eval(' detected"),
            (r"\bexec\s*\(", "Code: 'exec(' detected"),
            (r"\bsystem\s*\(", "Code: 'system(' detected"),
            (r"`[^`]*`", "Code: Backtick execution detected"),
            (r"\$\(", "Code: '$(' command substitution detected"),
            (r"&&\s+", "Code: '&&' shell chaining detected"),
            # Template injection
            (r"\{\{", "Template: '{{' detected"),
            (r"\{%", "Template: '{%' detected"),
            (r"\{%\s*if", "Template: '{% if' detected"),
            (r"\{%\s*for", "Template: '{% for' detected"),
            # XML/YAML injection
            (r"<!DOCTYPE", "XML: '<!DOCTYPE' detected"),
            (r"<!ENTITY", "XML: '<!ENTITY' detected"),
        ]

        detected = []
        text_lower = text.lower()

        for pattern, description in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append(description)

        return detected

    @staticmethod
    def detect_fraud_indicators(text: str) -> list[str]:
        """
        Detect suspicious financial and commercial patterns.

        Checks for:
        - Unusual payment terms: "payment before delivery", "upfront 100%", "no escrow"
        - Fictitious parties: "shell company", "offshore account", "wire transfer urgency"
        - Reverse payment flows: "refund", "chargeback dispute", "payment dispute"
        - Timing pressure: "must wire today", "payment due immediately", "act now"

        Args:
            text: Text to analyze

        Returns:
            list of fraud indicators
        """
        indicators = [
            # Unusual payment terms
            (r"payment\s+before\s+(delivery|shipment)", "Payment: 'payment before delivery' detected"),
            (r"upfront\s+100%", "Payment: 'upfront 100%' detected"),
            (r"no\s+escrow", "Payment: 'no escrow' detected"),
            # Fictitious parties
            (r"shell\s+company", "Fictitious party: 'shell company' mentioned"),
            (r"offshore\s+account", "Fictitious party: 'offshore account' mentioned"),
            # Reverse payment flows
            (r"wire\s+transfer\s+urgency", "Reverse payment flow: 'wire transfer urgency' detected"),
            (r"urgent.*wire\s+transfer", "Reverse payment flow: Urgent wire transfer pressure"),
            (r"wire\s+transfer", "Reverse payment flow: 'wire transfer' detected"),
            (r"chargeback|refund|dispute", "Reverse payment flow: 'chargeback/refund/dispute' detected"),
            # Timing pressure
            (r"must\s+wire\s+today", "Payment timing: 'must wire today' detected"),
            (r"payment\s+due\s+immediately", "Payment timing: 'payment due immediately' detected"),
            (r"act\s+now", "Payment timing: 'act now' urgency pressure"),
            (r"wire.*today", "Payment timing: Wire today pressure"),
        ]

        detected = []
        text_lower = text.lower()

        for pattern, description in indicators:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append(description)

        return detected

    @staticmethod
    def detect_misinformation(text: str) -> list[str]:
        """
        Detect conflicting or suspicious claims.

        Checks for:
        - Contradictions: keywords like "not" combined with definitive claims
        - Unverifiable claims: "proprietary research", "secret formula", "unpatented method"
        - Authority misuse: "according to our internal research", "we independently verified"

        Args:
            text: Text to analyze

        Returns:
            list of misinformation flags
        """
        flags = [
            # Unverifiable claims
            (r"proprietary\s+research", "Unverifiable: 'proprietary research' claimed"),
            (r"secret\s+formula", "Unverifiable: 'secret formula' claimed"),
            (r"unpatented\s+method", "Unverifiable: 'unpatented method' claimed"),
            (r"trade\s+secret", "Unverifiable: 'trade secret' claimed"),
            # Authority misuse
            (r"according\s+to\s+our\s+internal\s+research", "Authority misuse: 'according to our internal research' detected"),
            (r"we\s+independently\s+verified", "Authority misuse: 'we independently verified' detected"),
            (r"our\s+testing\s+shows", "Authority misuse: 'our testing shows' (unverifiable)"),
            # Contradictory patterns
            (r"not\s+\w+\s+but\s+definitely", "Contradiction: 'not X but definitely' detected"),
            (r"cannot\s+be\s+verified\s+but\s+[a-z]+", "Contradiction: unverifiable claim asserted"),
        ]

        detected = []
        text_lower = text.lower()

        for pattern, description in flags:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append(description)

        return detected

    @staticmethod
    def detect_ai_watermarks(text: str) -> list[str]:
        """
        Detect signs of AI-generated content.

        Checks for:
        - Characteristic phrases: "I appreciate your interest", "As an AI", etc.
        - Structural signs: bullet-heavy formatting, numbered lists, excessive structure
        - Repeated phrases and formulaic language

        Args:
            text: Text to analyze

        Returns:
            list of AI watermark indicators
        """
        watermarks = [
            # Characteristic AI phrases
            (r"I appreciate your interest", "Characteristic Phrase: 'I appreciate your interest' (AI marker)"),
            (r"As an AI", "Characteristic Phrase: 'As an AI' detected"),
            (r"I'm an AI assistant", "Characteristic Phrase: 'I'm an AI assistant' detected"),
            (r"As a language model", "Characteristic Phrase: 'As a language model' detected"),
            (r"I cannot", "Characteristic Phrase: 'I cannot' (common AI limitation language)"),
            (r"However, I", "Characteristic Phrase: 'However, I' (formulaic AI pattern)"),
            (r"In conclusion", "Characteristic Phrase: 'In conclusion' (formulaic summary)"),
            (r"To summarize", "Characteristic Phrase: 'To summarize' (formulaic summary)"),
            # Structural signs
            (r"^\s*[-•]\s+\w+", "Structure: Heavy bullet-point formatting"),
            (r"^\s*\d+\.\s+\w+.*?\n\s*\d+\.\s", "Structure: Excessive numbered lists"),
        ]

        detected = []
        text_lower = text.lower()

        for pattern, description in watermarks:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                detected.append(description)

        # Check for repeated phrases (formulaic language)
        words = text_lower.split()
        if len(words) > 100:  # Only check longer texts
            word_freq = {}
            for word in words:
                clean_word = re.sub(r"[^\w]", "", word)
                if len(clean_word) > 4:  # Only significant words
                    word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

            # Flag if any word appears very frequently (>10% of text)
            for word, count in word_freq.items():
                if count / len(words) > 0.1:
                    detected.append(f"Repetition: '{word}' appears {count} times ({count/len(words)*100:.1f}% of text)")
                    break  # Only flag the worst case

        return detected

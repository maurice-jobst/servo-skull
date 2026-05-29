"""PII pseudonymization vault using GLiNER NER.

Provides a session-scoped, in-memory bidirectional mapping between original
named entities and type-preserving pseudonym tokens (e.g. ``<PERSON_1>``).
The vault is never serialised to disk; call :meth:`PIIVault.clear` when the
ingestion run completes to wipe all entity data from memory.

Usage::

    vault = PIIVault()
    vault.scan(text)                 # detect entities, populate maps
    safe_prompt = vault.pseudonymize(text)
    cloud_response = llm.chat(safe_prompt)
    final_output = vault.rehydrate(cloud_response)
    vault.clear()
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# GLiNER entity labels → token prefix mapping
_LABEL_PREFIX: dict[str, str] = {
    "person": "PERSON",
    "organization": "ORG",
    "org": "ORG",
    "email": "EMAIL",
    "phone": "PHONE",
    "phone number": "PHONE",
    "location": "LOC",
    "address": "LOC",
    "date": "DATE",
    "time": "DATE",
    "ip address": "IP",
    "url": "URL",
    "credit card number": "PAN",
    "national id": "ID",
    "id number": "ID",
}

# Entity labels to request from GLiNER
_GLINER_LABELS: list[str] = [
    "person",
    "organization",
    "email",
    "phone number",
    "location",
    "address",
    "date",
    "ip address",
    "url",
    "credit card number",
    "national id",
]


def _token_prefix(label: str) -> str:
    """Return the canonical token prefix for a GLiNER entity label."""
    return _LABEL_PREFIX.get(label.lower(), label.upper().replace(" ", "_"))


class PIIVault:
    """Session-scoped PII pseudonymization vault.

    Detects named entities using GLiNER and maintains a bidirectional map
    between original entity strings and type-preserving pseudonym tokens.
    All state is in-memory; the vault enforces determinism within a session
    (the same entity string always resolves to the same token).

    Attributes:
        entity_map: Maps original entity text → pseudonym token.
        reverse_map: Maps pseudonym token → original entity text.
        contains_pii: True after :meth:`scan` if any entities were found.
        entity_types_found: Sorted list of GLiNER label types detected.
    """

    def __init__(self, gliner_model: str = "urchade/gliner_medium-v2.1") -> None:
        """Initialise the vault.

        Args:
            gliner_model: HuggingFace model ID for GLiNER. The model is loaded
                lazily on first call to :meth:`scan`.
        """
        self._gliner_model_id = gliner_model
        self._gliner: object | None = None  # lazy-loaded
        self._counters: dict[str, int] = {}  # prefix → counter

        self.entity_map: dict[str, str] = {}   # original → token
        self.reverse_map: dict[str, str] = {}  # token → original
        self.contains_pii: bool = False
        self.entity_types_found: list[str] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_gliner(self) -> None:
        """Lazily load the GLiNER model (one-time, cached on instance)."""
        if self._gliner is not None:
            return
        try:
            from gliner import GLiNER  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "gliner is required for PII detection. "
                "Install it with: uv pip install 'gliner>=0.2.0'"
            ) from exc
        logger.info("Loading GLiNER model '%s' (one-time)…", self._gliner_model_id)
        self._gliner = GLiNER.from_pretrained(self._gliner_model_id)
        logger.info("GLiNER model loaded.")

    def _next_token(self, prefix: str) -> str:
        """Generate the next sequential token for a given prefix."""
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"<{prefix}_{n}>"

    def _register(self, text: str, label: str) -> str:
        """Register an entity and return its canonical token.

        If the entity was already registered this session the existing token
        is returned, ensuring determinism.
        """
        if text in self.entity_map:
            return self.entity_map[text]
        prefix = _token_prefix(label)
        token = self._next_token(prefix)
        self.entity_map[text] = token
        self.reverse_map[token] = text
        return token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, text: str, threshold: float = 0.5) -> "PIIClassification":
        """Detect PII entities in *text* and populate the vault maps.

        This method is idempotent with respect to the running entity map:
        new entities extend the map; previously seen entities are unchanged.

        Args:
            text: Raw document text to scan.
            threshold: GLiNER confidence threshold (0–1). Entities below this
                score are ignored.

        Returns:
            A :class:`~servo_skull.models.PIIClassification` describing what
            was found.
        """
        from servo_skull.models import PIIClassification  # avoid circular import

        if not text or not text.strip():
            return PIIClassification(
                document_id="",
                contains_pii=False,
                entity_types_found=[],
                entity_count=0,
                forced_local=False,
            )

        self._ensure_gliner()

        # GLiNER accepts text + list of labels; chunk large texts to stay within
        # the model's max sequence length (~512 tokens ≈ ~2000 chars).
        chunk_size = 2000
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        label_set: set[str] = set()
        for chunk in chunks:
            entities = self._gliner.predict_entities(  # type: ignore[union-attr]
                chunk, _GLINER_LABELS, threshold=threshold
            )
            for ent in entities:
                original = ent["text"]
                label = ent["label"]
                self._register(original, label)
                label_set.add(label)

        new_types = sorted(label_set)
        for t in new_types:
            if t not in self.entity_types_found:
                self.entity_types_found.append(t)
        self.entity_types_found.sort()

        found_count = len(self.entity_map)
        self.contains_pii = found_count > 0

        if self.contains_pii:
            logger.info(
                "PIIVault: scanned %d chars, found %d entities (%s).",
                len(text),
                found_count,
                ", ".join(self.entity_types_found),
            )
        else:
            logger.debug("PIIVault: no PII entities detected.")

        return PIIClassification(
            document_id="",
            contains_pii=self.contains_pii,
            entity_types_found=list(self.entity_types_found),
            entity_count=found_count,
            forced_local=False,
        )

    def pseudonymize(self, text: str) -> str:
        """Replace all known entity strings in *text* with their tokens.

        Substitution is longest-match-first to avoid partial replacements
        (e.g. replacing ``"John"`` inside ``"John Smith"``).

        Args:
            text: Text to pseudonymize.

        Returns:
            Text with entity strings replaced by ``<TYPE_N>`` tokens.
        """
        if not self.entity_map or not text:
            return text

        # Sort by length descending — longest match first
        sorted_entities = sorted(self.entity_map.keys(), key=len, reverse=True)
        result = text
        for original in sorted_entities:
            token = self.entity_map[original]
            result = result.replace(original, token)
        return result

    def rehydrate(self, text: str) -> str:
        """Restore pseudonym tokens in *text* back to their original values.

        Args:
            text: LLM response text potentially containing ``<TYPE_N>`` tokens.

        Returns:
            Text with all known tokens replaced by their original entity strings.
        """
        if not self.reverse_map or not text:
            return text

        result = text
        for token, original in self.reverse_map.items():
            result = result.replace(token, original)
        return result

    def clear(self) -> None:
        """Wipe all entity mappings from memory.

        Call this after the ingestion run to ensure entity data does not
        persist between documents.
        """
        self.entity_map.clear()
        self.reverse_map.clear()
        self._counters.clear()
        self.entity_types_found.clear()
        self.contains_pii = False
        logger.debug("PIIVault: session cleared.")

    def __len__(self) -> int:
        """Return the number of entities currently registered."""
        return len(self.entity_map)

    def __repr__(self) -> str:
        return (
            f"PIIVault(entities={len(self)}, "
            f"types={self.entity_types_found}, "
            f"model='{self._gliner_model_id}')"
        )

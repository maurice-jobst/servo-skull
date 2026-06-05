"""Tests for PIIVault — pseudonymization and rehydration round-trip."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vault_with_mock_gliner(entities: list[dict]) -> "PIIVault":
    """Return a PIIVault whose GLiNER model is replaced with a mock."""
    from servo_skull.pii_vault import PIIVault

    vault = PIIVault.__new__(PIIVault)
    vault._gliner_model_id = "mock-model"
    vault._counters = {}
    vault.entity_map = {}
    vault.reverse_map = {}
    vault.contains_pii = False
    vault.entity_types_found = []

    # Mock the GLiNER model
    mock_gliner = MagicMock()
    mock_gliner.predict_entities.return_value = entities
    vault._gliner = mock_gliner
    return vault


# ---------------------------------------------------------------------------
# PIIVault unit tests
# ---------------------------------------------------------------------------

class TestPIIVaultPseudonymize:
    def test_pseudonymize_single_person(self):
        vault = _make_vault_with_mock_gliner([
            {"text": "John Smith", "label": "person"},
        ])
        vault.scan("Hello John Smith, welcome.")
        result = vault.pseudonymize("Hello John Smith, welcome.")
        assert "John Smith" not in result
        assert "<PERSON_1>" in result

    def test_pseudonymize_deterministic(self):
        """Same entity must always get the same token within a session."""
        vault = _make_vault_with_mock_gliner([
            {"text": "Jane Doe", "label": "person"},
        ])
        vault.scan("Jane Doe spoke.")
        t1 = vault.pseudonymize("Jane Doe")
        t2 = vault.pseudonymize("Jane Doe again")
        assert t1 == "<PERSON_1>"
        assert "<PERSON_1>" in t2

    def test_pseudonymize_multiple_types(self):
        vault = _make_vault_with_mock_gliner([
            {"text": "Alice", "label": "person"},
            {"text": "ACME Corp", "label": "organization"},
            {"text": "alice@example.com", "label": "email"},
        ])
        vault.scan("Alice at ACME Corp: alice@example.com")
        result = vault.pseudonymize("Alice at ACME Corp: alice@example.com")
        assert "Alice" not in result
        assert "ACME Corp" not in result
        assert "alice@example.com" not in result
        assert "<PERSON_1>" in result
        assert "<ORG_1>" in result
        assert "<EMAIL_1>" in result

    def test_pseudonymize_longest_match_first(self):
        """'John Smith' must be replaced as a unit, not 'John' then 'Smith'."""
        vault = _make_vault_with_mock_gliner([
            {"text": "John Smith", "label": "person"},
            {"text": "John", "label": "person"},
        ])
        vault.scan("John Smith is here")
        result = vault.pseudonymize("John Smith is here")
        # Should not produce "<PERSON_N> Smith"
        assert "Smith" not in result

    def test_pseudonymize_empty_text(self):
        vault = _make_vault_with_mock_gliner([])
        vault.scan("")
        assert vault.pseudonymize("") == ""
        assert vault.pseudonymize("no entities here") == "no entities here"


class TestPIIVaultRehydrate:
    def test_rehydrate_round_trip(self):
        vault = _make_vault_with_mock_gliner([
            {"text": "Bob", "label": "person"},
        ])
        vault.scan("Bob is the owner.")
        pseudo = vault.pseudonymize("Bob is the owner.")
        restored = vault.rehydrate(pseudo)
        assert restored == "Bob is the owner."

    def test_rehydrate_partial_response(self):
        """LLM may reference tokens in a different sentence structure."""
        vault = _make_vault_with_mock_gliner([
            {"text": "Globex Corporation", "label": "organization"},
        ])
        vault.scan("Globex Corporation submitted the bid.")
        vault.pseudonymize("Globex Corporation submitted the bid.")
        # Simulate LLM response referencing the token
        llm_response = "The organisation <ORG_1> has submitted a compliant bid."
        restored = vault.rehydrate(llm_response)
        assert "Globex Corporation" in restored
        assert "<ORG_1>" not in restored

    def test_rehydrate_no_tokens_unchanged(self):
        vault = _make_vault_with_mock_gliner([])
        vault.scan("no pii")
        result = vault.rehydrate("LLM says something without tokens.")
        assert result == "LLM says something without tokens."


class TestPIIVaultClear:
    def test_clear_wipes_maps(self):
        vault = _make_vault_with_mock_gliner([
            {"text": "Eve", "label": "person"},
        ])
        vault.scan("Eve is here.")
        assert len(vault) == 1
        vault.clear()
        assert len(vault) == 0
        assert not vault.contains_pii
        assert vault.entity_types_found == []

    def test_clear_isolates_sessions(self):
        """After clear(), the same entity should get a fresh token (counter reset)."""
        vault = _make_vault_with_mock_gliner([
            {"text": "Eve", "label": "person"},
        ])
        vault.scan("Eve")
        token_first = vault.entity_map["Eve"]
        vault.clear()

        # Re-register same entity after clear
        vault._gliner.predict_entities.return_value = [
            {"text": "Eve", "label": "person"}
        ]
        vault.scan("Eve again")
        token_second = vault.entity_map["Eve"]
        # Both should be PERSON_1 since counter resets — deterministic
        assert token_first == token_second == "<PERSON_1>"


class TestPIIVaultScan:
    def test_scan_returns_pii_classification(self):
        from servo_skull.models import PIIClassification
        vault = _make_vault_with_mock_gliner([
            {"text": "Charlie", "label": "person"},
        ])
        result = vault.scan("Charlie is here.")
        assert isinstance(result, PIIClassification)
        assert result.contains_pii is True
        assert result.entity_count == 1
        assert "person" in result.entity_types_found

    def test_scan_empty_text_no_pii(self):
        from servo_skull.models import PIIClassification
        vault = _make_vault_with_mock_gliner([])
        result = vault.scan("")
        assert isinstance(result, PIIClassification)
        assert result.contains_pii is False
        assert result.entity_count == 0

    def test_scan_no_gliner_raises_import_error(self):
        from servo_skull.pii_vault import PIIVault
        vault = PIIVault.__new__(PIIVault)
        vault._gliner_model_id = "mock"
        vault._gliner = None
        vault._counters = {}
        vault.entity_map = {}
        vault.reverse_map = {}
        vault.contains_pii = False
        vault.entity_types_found = []

        with patch.dict("sys.modules", {"gliner": None}):
            with pytest.raises(ImportError, match="gliner"):
                vault._ensure_gliner()

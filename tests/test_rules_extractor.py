"""Test rules extraction logic."""
import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from servo_skull.models import DocumentExtract, Requirement
from servo_skull.rules_extractor import extract_rules, write_rule_markdown


def test_extract_rules_success(sample_document_extract):
    """Test successful extraction of rules using mock LLMClient."""
    mock_llm_client = MagicMock()
    mock_llm_client.chat.return_value = json.dumps([
        {
            "rule_id": "Rule_5_1",
            "title": "Button Contrast",
            "description": "Buttons must have 4.5:1 contrast ratio.",
            "original_clause": "All UI buttons must maintain 4.5:1 contrast ratio.",
            "source_reference": "Section 5.1"
        }
    ])

    rules = extract_rules(sample_document_extract, llm_client=mock_llm_client, verbose=True)
    assert len(rules) == 1
    assert isinstance(rules[0], Requirement)
    assert rules[0].rule_id == "Rule_5_1"
    assert rules[0].title == "Button Contrast"
    assert rules[0].description == "Buttons must have 4.5:1 contrast ratio."
    assert rules[0].original_clause == "All UI buttons must maintain 4.5:1 contrast ratio."
    assert rules[0].source_reference == "Section 5.1"
    assert rules[0].status == "draft"


def test_extract_rules_invalid_json(sample_document_extract):
    """Test robust error handling when LLM returns invalid JSON."""
    mock_llm_client = MagicMock()
    mock_llm_client.chat.return_value = "invalid json response"

    rules = extract_rules(sample_document_extract, llm_client=mock_llm_client, verbose=True)
    assert rules == []


def test_write_rule_markdown(tmp_path):
    """Test writing a requirement object to a markdown rule file."""
    req = Requirement(
        rule_id="Rule_5_1",
        title="Button Contrast",
        description="Buttons must have 4.5:1 contrast ratio.",
        original_clause="All UI buttons must maintain 4.5:1 contrast ratio.",
        source_reference="Section 5.1",
        status="draft"
    )

    output_file = write_rule_markdown(req, tmp_path)
    assert output_file.exists()
    assert output_file.name == "Rule_5_1.md"

    content = output_file.read_text(encoding="utf-8")
    assert "type: rule" in content
    assert 'rule_id: Rule_5_1' in content
    assert 'title: "Button Contrast"' in content
    assert "All UI buttons must maintain 4.5:1 contrast ratio." in content
    assert 'page where type = "user-story" and references = "{{page}}"' in content

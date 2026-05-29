"""Test security checker for injection, fraud, misinformation, and AI watermark detection."""
import json
from unittest.mock import MagicMock, patch

import pytest

from servo_skull.models import DocumentExtract
from servo_skull.security_checker import SecurityChecker


@pytest.fixture
def security_checker():
    """Create a SecurityChecker instance for testing."""
    return SecurityChecker()


# ============================================================================
# Tests for check_security() - LLM-based analysis
# ============================================================================


def test_check_security_returns_dict(sample_document_extract):
    """Verify check_security returns a dict."""
    checker = SecurityChecker()

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "security_issues": [],
                            "misinformation_risks": [],
                            "ai_watermarks": [],
                            "fraud_indicators": [],
                            "recommendations": [],
                        })
                    }
                }
            ]
        }

        result = checker.check_security(sample_document_extract)
        assert isinstance(result, dict)


def test_check_security_includes_all_fields(sample_document_extract):
    """Verify check_security result includes required keys."""
    checker = SecurityChecker()

    with patch("requests.post") as mock_post:
        mock_response = {
            "security_issues": [
                {
                    "type": "injection",
                    "severity": "high",
                    "description": "Potential SQL injection detected",
                }
            ],
            "misinformation_risks": [],
            "ai_watermarks": [],
            "fraud_indicators": [],
            "recommendations": ["Verify document authenticity"],
        }
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_response)}}]
        }

        result = checker.check_security(sample_document_extract)

        assert "security_issues" in result
        assert "misinformation_risks" in result
        assert "ai_watermarks" in result
        assert "fraud_indicators" in result
        assert "recommendations" in result


def test_check_security_handles_ollama_failure(sample_document_extract):
    """Verify check_security returns empty result when Ollama fails."""
    checker = SecurityChecker()

    with patch("requests.post") as mock_post:
        mock_post.side_effect = Exception("Connection refused")

        result = checker.check_security(sample_document_extract)

        assert isinstance(result, dict)
        assert result["security_issues"] == []
        assert result["misinformation_risks"] == []
        assert result["ai_watermarks"] == []
        assert result["fraud_indicators"] == []
        # Verify the result has the expected structure
        assert "recommendations" in result


def test_check_security_handles_invalid_json(sample_document_extract):
    """Verify check_security handles invalid JSON from LLM."""
    checker = SecurityChecker()

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "not valid json {{"}}]
        }

        result = checker.check_security(sample_document_extract)

        assert isinstance(result, dict)
        assert result["security_issues"] == []
        assert "recommendations" in result  # Warning is in recommendations


def test_check_security_includes_document_context(sample_document_extract):
    """Verify check_security processes document filename."""
    checker = SecurityChecker()

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "security_issues": [],
                            "misinformation_risks": [],
                            "ai_watermarks": [],
                            "fraud_indicators": [],
                            "recommendations": [],
                        })
                    }
                }
            ]
        }

        result = checker.check_security(sample_document_extract)

        # Verify the call included the document in the prompt
        call_args = mock_post.call_args
        assert call_args is not None
        payload = call_args.kwargs["json"]
        assert "messages" in payload
        user_message = [m for m in payload["messages"] if m["role"] == "user"][0]
        assert sample_document_extract.original_filename in user_message["content"]


# ============================================================================
# Tests for detect_injection_patterns()
# ============================================================================


def test_detect_injection_patterns_sql(security_checker):
    """Test SQL injection detection."""
    text = "Find records where id = 1 UNION SELECT * FROM users"
    result = security_checker.detect_injection_patterns(text)
    assert len(result) > 0
    assert any("SQL" in r for r in result)


def test_detect_injection_patterns_sql_drop(security_checker):
    """Test SQL DROP injection detection."""
    text = "Query with '; DROP TABLE users; -- comment"
    result = security_checker.detect_injection_patterns(text)
    assert len(result) > 0
    assert any("SQL" in r for r in result)


def test_detect_injection_patterns_code_eval(security_checker):
    """Test code injection with eval()."""
    text = "Execute this dangerous code: eval(user_input)"
    result = security_checker.detect_injection_patterns(text)
    assert len(result) > 0
    assert any("Code" in r for r in result)


def test_detect_injection_patterns_code_exec(security_checker):
    """Test code injection with exec()."""
    text = "Run: exec('import os; os.system(cmd)')"
    result = security_checker.detect_injection_patterns(text)
    assert len(result) > 0
    assert any("Code" in r for r in result)


def test_detect_injection_patterns_template(security_checker):
    """Test template injection detection."""
    text = "Use this template: {{ variable }} {% if condition %}"
    result = security_checker.detect_injection_patterns(text)
    assert len(result) > 0
    assert any("Template" in r for r in result)


def test_detect_injection_patterns_command(security_checker):
    """Test command injection detection."""
    text = "Execute: $(rm -rf /)"
    result = security_checker.detect_injection_patterns(text)
    assert len(result) > 0
    assert any("Code" in r for r in result)


def test_detect_injection_patterns_empty(security_checker):
    """Test clean text returns no injection patterns."""
    text = "This is a normal, clean document with no malicious code."
    result = security_checker.detect_injection_patterns(text)
    assert result == []


def test_detect_injection_patterns_case_insensitive(security_checker):
    """Test injection detection is case-insensitive."""
    text = "Get data where id = 1 union select * from users"
    result = security_checker.detect_injection_patterns(text)
    assert len(result) > 0


# ============================================================================
# Tests for detect_fraud_indicators()
# ============================================================================


def test_detect_fraud_indicators_payment_timing(security_checker):
    """Test fraud detection for urgent payment timing."""
    text = "Payment must wire today or contract is voided."
    result = security_checker.detect_fraud_indicators(text)
    assert len(result) > 0
    assert any("Payment timing" in r for r in result)


def test_detect_fraud_indicators_upfront(security_checker):
    """Test fraud detection for upfront payment demands."""
    text = "Please send upfront 100% of contract value before delivery."
    result = security_checker.detect_fraud_indicators(text)
    assert len(result) > 0
    assert any("payment" in r.lower() for r in result)


def test_detect_fraud_indicators_shell_company(security_checker):
    """Test fraud detection for shell company references."""
    text = "This contract involves a shell company registered offshore."
    result = security_checker.detect_fraud_indicators(text)
    assert len(result) > 0
    assert any("Fictitious" in r for r in result)


def test_detect_fraud_indicators_wire_transfer(security_checker):
    """Test fraud detection for wire transfer urgency."""
    text = "Wire transfer only. Same-day payment required."
    result = security_checker.detect_fraud_indicators(text)
    assert len(result) > 0


def test_detect_fraud_indicators_chargeback(security_checker):
    """Test fraud detection for chargeback threats."""
    text = "We can initiate a chargeback dispute if you don't cooperate."
    result = security_checker.detect_fraud_indicators(text)
    assert len(result) > 0
    assert any("Reverse" in r for r in result)


def test_detect_fraud_indicators_empty(security_checker):
    """Test clean text returns no fraud indicators."""
    text = "Standard terms: Payment 30 days after invoice with proper documentation."
    result = security_checker.detect_fraud_indicators(text)
    assert result == []


def test_detect_fraud_indicators_case_insensitive(security_checker):
    """Test fraud detection is case-insensitive."""
    text = "Payment MUST WIRE TODAY to proceed"
    result = security_checker.detect_fraud_indicators(text)
    assert len(result) > 0


# ============================================================================
# Tests for detect_misinformation()
# ============================================================================


def test_detect_misinformation_proprietary(security_checker):
    """Test misinformation detection for proprietary claims."""
    text = "Our proprietary research shows this is the best solution."
    result = security_checker.detect_misinformation(text)
    assert len(result) > 0
    assert any("Unverifiable" in r for r in result)


def test_detect_misinformation_secret(security_checker):
    """Test misinformation detection for secret formula."""
    text = "The secret formula cannot be disclosed under NDA."
    result = security_checker.detect_misinformation(text)
    assert len(result) > 0
    assert any("Unverifiable" in r for r in result)


def test_detect_misinformation_authority_misuse(security_checker):
    """Test misinformation detection for authority misuse."""
    text = "According to our internal research, this is certified compliant."
    result = security_checker.detect_misinformation(text)
    assert len(result) > 0
    assert any("Authority" in r for r in result)


def test_detect_misinformation_independent_claim(security_checker):
    """Test misinformation detection for false independence claims."""
    text = "We independently verified that our solution meets all requirements."
    result = security_checker.detect_misinformation(text)
    assert len(result) > 0
    assert any("Authority" in r for r in result)


def test_detect_misinformation_empty(security_checker):
    """Test clean text returns no misinformation flags."""
    text = "This product has been tested and meets ISO 27001 requirements."
    result = security_checker.detect_misinformation(text)
    assert result == []


def test_detect_misinformation_case_insensitive(security_checker):
    """Test misinformation detection is case-insensitive."""
    text = "Our PROPRIETARY RESEARCH indicates compliance."
    result = security_checker.detect_misinformation(text)
    assert len(result) > 0


# ============================================================================
# Tests for detect_ai_watermarks()
# ============================================================================


def test_detect_ai_watermarks_phrase_ai_assistant(security_checker):
    """Test AI watermark detection for 'As an AI' phrase."""
    text = "As an AI assistant, I appreciate your interest in our system."
    result = security_checker.detect_ai_watermarks(text)
    assert len(result) > 0
    assert any("Phrase" in r for r in result)


def test_detect_ai_watermarks_phrase_cannot(security_checker):
    """Test AI watermark detection for 'I cannot' phrase."""
    text = "I cannot provide this information due to limitations."
    result = security_checker.detect_ai_watermarks(text)
    assert len(result) > 0
    assert any("Characteristic" in r for r in result)


def test_detect_ai_watermarks_formulaic(security_checker):
    """Test AI watermark detection for formulaic language."""
    text = "Furthermore, it is important to note that in conclusion, this represents best practice."
    result = security_checker.detect_ai_watermarks(text)
    assert len(result) > 0


def test_detect_ai_watermarks_excessive_bullets(security_checker):
    """Test AI watermark detection for excessive bullet-point structure."""
    text = """
    - Point one
    - Point two
    - Point three
    - Point four
    - Point five
    - Point six
    - Point seven
    - Point eight
    - Point nine
    - Point ten
    """
    result = security_checker.detect_ai_watermarks(text)
    # Should detect structural pattern
    assert any("Structural" in r or "bullet" in r.lower() for r in result)


def test_detect_ai_watermarks_numbered_list(security_checker):
    """Test AI watermark detection for excessive numbered lists."""
    text = """
    1. First requirement
    2. Second requirement
    3. Third requirement
    4. Fourth requirement
    5. Fifth requirement
    6. Sixth requirement
    7. Seventh requirement
    8. Eighth requirement
    9. Ninth requirement
    10. Tenth requirement
    """
    result = security_checker.detect_ai_watermarks(text)
    assert any("Structural" in r or "numbered" in r.lower() for r in result)


def test_detect_ai_watermarks_empty(security_checker):
    """Test clean text returns no AI watermark indicators."""
    text = "This product was built over 5 years with 20 engineers and represents genuine innovation."
    result = security_checker.detect_ai_watermarks(text)
    assert result == []


def test_detect_ai_watermarks_case_insensitive(security_checker):
    """Test AI watermark detection is case-insensitive."""
    text = "AS AN AI ASSISTANT, I CANNOT HELP WITH THIS REQUEST."
    result = security_checker.detect_ai_watermarks(text)
    assert len(result) > 0


# ============================================================================
# Integration tests
# ============================================================================


def test_security_checker_multiple_issues(sample_document_extract):
    """Test security checker detecting multiple issue types."""
    # Modify sample to include various issues
    extract = DocumentExtract(
        document_id="multi-issue",
        original_filename="suspicious.pdf",
        document_type="pdf",
        extracted_text="""
        This contract requires upfront 100% wire transfer immediately.
        Our proprietary research shows SQL injection with ' UNION SELECT *.
        As an AI, I cannot provide further details.
        """,
        metadata={},
        confidence=0.9,
        extraction_tool="test",
    )

    checker = SecurityChecker()

    # Check individual pattern detection
    injections = checker.detect_injection_patterns(extract.extracted_text)
    fraud = checker.detect_fraud_indicators(extract.extracted_text)
    misinformation = checker.detect_misinformation(extract.extracted_text)
    ai_marks = checker.detect_ai_watermarks(extract.extracted_text)

    assert len(injections) > 0
    assert len(fraud) > 0
    assert len(misinformation) > 0
    assert len(ai_marks) > 0


def test_security_checker_clean_document(sample_document_extract):
    """Test security checker on completely clean document."""
    checker = SecurityChecker()

    injections = checker.detect_injection_patterns(sample_document_extract.extracted_text)
    fraud = checker.detect_fraud_indicators(sample_document_extract.extracted_text)
    misinformation = checker.detect_misinformation(sample_document_extract.extracted_text)
    ai_marks = checker.detect_ai_watermarks(sample_document_extract.extracted_text)

    assert injections == []
    assert fraud == []
    assert misinformation == []
    assert ai_marks == []


def test_detect_all_patterns_are_strings(security_checker):
    """Verify all detection methods return list of strings."""
    text = "Some text with ' SQL injection"

    injections = security_checker.detect_injection_patterns(text)
    fraud = security_checker.detect_fraud_indicators(text)
    misinformation = security_checker.detect_misinformation(text)
    ai_marks = security_checker.detect_ai_watermarks(text)

    for item in injections + fraud + misinformation + ai_marks:
        assert isinstance(item, str)

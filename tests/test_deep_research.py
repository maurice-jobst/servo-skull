# pipelines/servo-skull/tests/test_deep_research.py
import pytest
from unittest.mock import MagicMock, patch


def test_deep_research_client_init_requires_api_key():
    from servo_skull.deep_research import DeepResearchClient
    with pytest.raises(ValueError, match="api_key"):
        DeepResearchClient(api_key="")


def test_deep_research_client_builds_report(monkeypatch):
    """Smoke test with mocked genai response."""
    from servo_skull.deep_research import DeepResearchClient
    from servo_skull.models import DeepResearchReport

    mock_response = MagicMock()
    mock_response.text = "# Research Report\n\nGDPR applies..."
    mock_response.candidates = []

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("servo_skull.deep_research.genai.Client", return_value=mock_client):
        client = DeepResearchClient(api_key="test-key")
        report = client.research(
            query="data retention policy review",
            domain="compliance-review",
        )

    assert isinstance(report, DeepResearchReport)
    assert "GDPR" in report.report_text
    assert report.domain == "compliance-review"
    assert report.model == "gemini-2.5-flash"


def test_deep_research_client_max_model():
    """Verify max flag selects the max model."""
    from servo_skull.deep_research import DeepResearchClient
    client = DeepResearchClient(api_key="test-key", use_max=True)
    assert client.model == "gemini-2.5-pro"

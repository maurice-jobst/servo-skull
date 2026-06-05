"""LLM prompt templates for Servo-Skull gap analysis and security checks."""
import logging
from servo_skull._utils import setup_logging
from servo_skull.models import DocumentExtract

logger = setup_logging(__name__)


def render_gap_analysis_prompt(extract: DocumentExtract, codex: str) -> dict[str, str]:
    """
    Render a gap analysis prompt for LLM consumption.

    Performs 4-dimensional gap analysis against a domain codex:
    1. Stakeholder Ecosystem (Legal, Sales, Engineering, Finance, Operations, Partners)
    2. Delivery Methodology Stack (Governance, Execution, Operations layers)
    3. Affected Components (Data, Service, Client, Integration layers)
    4. Compliance Stack (data protection, security, accessibility standards)

    Args:
        extract: DocumentExtract with original_filename and extracted_text
        codex: Domain codex text (the grounding framework for gap analysis)

    Returns:
        dict with "system" and "user" keys, both strings
    """
    system_prompt = f"""You are an expert gap analyst for Servo-Skull, a pre-project document scoping engine.

Your task is to perform a 4-dimensional gap analysis on the provided document.

The analysis framework has four intersecting variable dimensions:

**Dimension 1: Stakeholder Ecosystem**
- Legal & GRC: Contract liability, SLA penalties, data residency, regulatory timelines
- Sales & Business Development: Win themes, commercial viability, pricing, localization
- Engineering/Architecture: API compatibility, infrastructure constraints, technical debt, system dependencies
- Finance & Controlling: Budget constraints, billing triggers, depreciation, ROI modeling
- Operations & Support: SLAs, maintenance intervals, service desk escalation, disaster recovery
- External Partners: Integration checkpoints with third-party providers and compliance authorities

**Dimension 2: Delivery Methodology Stack**
- Governance Layer: PRINCE2, PMBOK (stage-gates, business case, risk registers)
- Execution Layer: Agile/Scrum (epics, user stories, sprint mechanics)
- Operations Layer: Lightweight CSM or ITIL 4 (service desk, SLAs, monitoring)

**Dimension 3: Affected Components**
- Data Layer: Persistence, schemas, migrations, data lifecycle, retention
- Service Layer: APIs, business logic, orchestration, settlement, reconciliation
- Client/Edge Layer: User interfaces, devices, firmware/OTA updates, offline behavior
- Integration Layer: Third-party connectors, message buses, event streams, webhooks

**Dimension 4: Compliance Stack**
- GDPR / data-protection: Lawful basis, data residency, subject rights, retention limits
- SOC 2 / ISO 27001: Security controls, audit trails, access management
- WCAG 2.2 / accessibility: Conformance level, assistive-tech compatibility
- Domain-specific regulation: Any sector standards named in the document or codex

Analyze the document against these dimensions and the domain codex provided. Identify gaps where:
- Information is explicitly missing or ambiguous
- Requirements conflict with the framework
- Assumptions are unstated or questionable

For each gap found, cite the extracted text where the gap appears.

Write all descriptions and gap explanations in a clear, professional, direct style aimed at developers and product managers. Target a readability difficulty level corresponding to a Flesch Reading Ease score of 30 to 50 (professional/college level) by keeping explanations focused and avoiding conversational filler.

Ensure complete compliance with our security and PII isolation guidelines:
1. PII Redaction: Replace all personal names, named developers, and specific workforce members in descriptions and contexts with generalized role tokens (e.g., `<CTO>`, `<CPO>`, `<PM>`, `<Dev>`).
2. Public vs. Private: Clearly distinguish between Public attributes (architecture layouts, technical interfaces, compliance statuses) and Private calibrations (named bandwidth concerns, personal alignment assessments, or velocity blockages). Never serialize named private calibrations; represent them using role-based placeholders.

Rate hallucination risk on a scale 0.0-1.0, where:
- 0.0 = fully grounded in the extracted text (no speculation)
- 1.0 = highly speculative (not mentioned in text)

Output MUST be valid JSON."""


    user_prompt = f"""Document Analysis Request
===========================

**Filename:** {extract.original_filename}

**Document Type:** {extract.document_type}

**Confidence Score:** {extract.confidence}

**Extracted Text:**
```
{extract.extracted_text}
```

**Domain Codex (grounding rules):**
```
{codex}
```

---

Perform gap analysis on the extracted text. Return valid JSON with this structure:

{{
  "gaps": {{
    "stakeholder": [
      {{
        "gap": "Description of missing or ambiguous stakeholder requirement",
        "severity": "high|medium|low",
        "context": "Relevant text snippet or explanation",
        "dimension": "Legal|Sales|Engineering|Finance|Operations|Partners"
      }}
    ],
    "methodology": [
      {{
        "gap": "Description of missing or ambiguous methodology requirement",
        "severity": "high|medium|low",
        "context": "Relevant text snippet or explanation",
        "layer": "Governance|Execution|Operations"
      }}
    ],
    "components": [
      {{
        "gap": "Description of missing or ambiguous component requirement",
        "severity": "high|medium|low",
        "context": "Relevant text snippet or explanation",
        "component": "Data|Service|Client|Integration"
      }}
    ],
    "compliance": [
      {{
        "gap": "Description of missing or ambiguous compliance requirement",
        "severity": "high|medium|low",
        "context": "Relevant text snippet or explanation",
        "standard": "GDPR|SOC2|WCAG|ISO27001|Domain"
      }}
    ]
  }},
  "risk_flags": [
    {{
      "risk": "Description of risk",
      "domain": "commercial|legal|technical|operational",
      "severity": "high|medium|low",
      "recommendation": "Suggested mitigation"
    }}
  ],
  "security_flags": [],
  "hallucination_score": 0.15,
  "grounding_notes": "Summary of grounding assessment and key observations"
}}

Ensure all gaps are grounded in the extracted text. Rate hallucination carefully."""

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


def render_security_check_prompt(extract: DocumentExtract) -> dict[str, str]:
    """
    Render a security check prompt for LLM consumption.

    Analyzes extracted document text for security, compliance, and fraud indicators:
    - Injection patterns (SQL, code, template injection)
    - Misinformation or conflicting requirements
    - AI watermarks or synthetic content indicators
    - Fraud patterns (unusual payment terms, fictitious parties)

    Args:
        extract: DocumentExtract with original_filename and extracted_text

    Returns:
        dict with "system" and "user" keys, both strings
    """
    system_prompt = """You are a security and compliance analyst for Servo-Skull, a pre-project document scoping engine.

Your task is to analyze the provided document text for security risks, compliance violations, and fraud indicators.

**Check for:**

1. **Injection Patterns:**
   - SQL injection indicators (unusual quote nesting, SQL keywords, schema references)
   - Code injection (executable patterns, script tags, command syntax)
   - Template injection (template expressions, variable interpolation attempts)
   - LDAP/command injection syntax

2. **Misinformation & Conflicting Requirements:**
   - Contradictory statements about the same requirement
   - Claims that contradict known standards or regulations
   - Anachronistic or historically inaccurate information
   - Unsupported assertions about product capabilities

3. **AI Watermarks & Synthetic Content:**
   - Patterns common in LLM-generated text (overly formal, repetitive phrases)
   - Inconsistent writing style across sections
   - Implausible technical specifications
   - Hallucinated compliance certifications or standards

4. **Fraud Patterns:**
   - Unusual payment terms (immediate payment, wire transfers, cryptocurrency)
   - Fictitious parties or shell companies
   - Misrepresented authority or signing power
   - Fake compliance attestations or certifications
   - Unusual urgency or pressure tactics
   - Impossible delivery timelines for scope

Output MUST be valid JSON."""

    user_prompt = f"""Security & Compliance Check Request
====================================

**Filename:** {extract.original_filename}

**Document Type:** {extract.document_type}

**Document Text:**
```
{extract.extracted_text}
```

---

Analyze the document for security, misinformation, AI synthesis, and fraud indicators.

Return valid JSON with this structure:

{{
  "security_issues": [
    {{
      "type": "injection|command|template",
      "severity": "critical|high|medium|low",
      "location": "Text excerpt or section where found",
      "description": "Detailed description of the issue",
      "recommendation": "Mitigation or escalation step"
    }}
  ],
  "misinformation_risks": [
    {{
      "claim": "The suspicious claim found in text",
      "context": "Where it appears",
      "contradiction": "Why it conflicts with known standards/facts",
      "severity": "high|medium|low"
    }}
  ],
  "ai_watermarks": [
    {{
      "pattern": "Description of LLM-like pattern",
      "severity": "high|medium|low",
      "location": "Text excerpt"
    }}
  ],
  "fraud_indicators": [
    {{
      "indicator": "Unusual payment term, fake party, impossible timeline, etc.",
      "severity": "critical|high|medium|low",
      "context": "Relevant text or explanation",
      "recommendation": "Escalation or verification step"
    }}
  ],
  "recommendations": [
    "High-level action items to validate document authenticity and compliance"
  ]
}}

Be thorough but avoid over-flagging. Flag only genuine security or fraud indicators."""

    return {
        "system": system_prompt,
        "user": user_prompt,
    }

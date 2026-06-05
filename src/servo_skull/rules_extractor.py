"""Automated rule/requirement extractor from document extracts."""
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import click

from servo_skull._utils import setup_logging
from servo_skull.grounder import _load_config
from servo_skull.llm_client import LLMClient
from servo_skull.llm_providers import create_provider
from servo_skull.models import DocumentExtract, Requirement

logger = setup_logging(__name__)


def extract_rules(
    extract: DocumentExtract,
    llm_client: Optional[LLMClient] = None,
    verbose: bool = False
) -> list[Requirement]:
    """
    Extract structured requirements/rules from a document extract using LLM.

    Args:
        extract: DocumentExtract containing raw text.
        llm_client: Configured LLMClient instance.
        verbose: Enable verbose logging.

    Returns:
        List of Requirement models.
    """
    if verbose:
        logger.info(f"Starting rule extraction for document {extract.document_id}")

    # Set up LLM client if none provided
    if llm_client is None:
        config = _load_config()
        routing = config.get("routing", {})
        primary_name = routing.get("primary", "local_gemma")
        fallback_name = routing.get("fallback", "cloud_openai")

        providers = config.get("providers", {})
        primary_config = providers.get(primary_name, {})
        fallback_config = providers.get(fallback_name)

        primary = create_provider(primary_config)

        # Instantiate fallback if present and has api_key
        fallback = None
        if fallback_config:
            from servo_skull.llm_providers import expand_env_vars
            expanded_fallback = expand_env_vars(fallback_config)
            prov_type = expanded_fallback.get("type", "").lower()
            api_key = expanded_fallback.get("api_key", "")

            if prov_type in ("openai", "anthropic", "gemini"):
                if api_key and not api_key.startswith("$"):
                    fallback = create_provider(fallback_config)
            else:
                fallback = create_provider(fallback_config)

        llm_client = LLMClient(primary=primary, fallback=fallback)

    system_prompt = """You are a precise systems analyst specializing in requirements extraction and regulatory compliance.

Your task is to analyze the provided document text and extract all explicit and implicit requirements, rules, standards, and constraints.

Extract:
- Named regulatory and compliance rules (e.g. "Rule 5.1", "Section 2.17").
- Technical system constraints (e.g. memory requirements, timing limits, data formats).
- Operational SLAs and functional requirements.

For each rule, identify:
1. rule_id: A slugified unique ID based on the rule title/reference (e.g., Rule_5_1 or Req_2_17). Only use alphanumeric characters, underscores, and dashes.
2. title: A short descriptive title.
3. description: Detailed explanation of what the rule requires. Write the description in a clear, highly professional, and direct style aimed at developers and product managers. Target a Flesch Reading Ease score of 30 to 50 (college graduate/technical level) by keeping sentences focused, avoiding wordy filler, and ensuring clarity.
4. original_clause: The exact text snippet or clause from the document that mentions the rule.
5. source_reference: The section number, clause, or page number where it is located.

Ensure complete compliance with our security and PII isolation guidelines:
- PII Redaction: Replace all personal names, named developers, and specific workforce members in titles, descriptions, and source references with generalized role tokens (e.g., `<CTO>`, `<CPO>`, `<PM>`, `<Dev>`).
- Public vs. Private: Clearly separate technical system/architectural attributes (public) from stakeholder-specific personal bandwidth calibrations or velocity constraints (private). Never serialize named private calibrations; represent them using role-based placeholders.

Your output MUST be a valid JSON array of objects representing these rules."""

    user_prompt = f"""Extract all requirements and rules from this document.

**Filename:** {extract.original_filename}
**Document Type:** {extract.document_type}

**Document Text:**
```
{extract.extracted_text}
```

---

Return a valid JSON list of requirements with this structure:
[
  {{
    "rule_id": "Rule_5_1",
    "title": "Button Contrast Ratio",
    "description": "All physical UI buttons must maintain a minimum contrast ratio of 4.5:1 against backgrounds.",
    "original_clause": "All physical and digital UI buttons must maintain a minimum contrast ratio of 4.5:1 against their backgrounds.",
    "source_reference": "Section 5.1"
  }}
]
"""

    try:
        response = llm_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            document_size_bytes=len(extract.extracted_text.encode("utf-8"))
        )

        # Parse output JSON
        try:
            # Strip markdown block formatting if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response.split("```json", 1)[1].split("```", 1)[0].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.split("```", 1)[1].split("```", 1)[0].strip()

            raw_rules = json.loads(cleaned_response)
            if not isinstance(raw_rules, list):
                if isinstance(raw_rules, dict) and "rules" in raw_rules:
                    raw_rules = raw_rules["rules"]
                elif isinstance(raw_rules, dict) and "requirements" in raw_rules:
                    raw_rules = raw_rules["requirements"]
                else:
                    raise ValueError("LLM response did not contain a list of rules")

            requirements = []
            for r in raw_rules:
                # Clean up rule_id to ensure it's a valid wiki filename
                rule_id = r.get("rule_id", "").replace(" ", "_")
                rule_id = re.sub(r"[^a-zA-Z0-9_\-]", "", rule_id)
                if not rule_id:
                    rule_id = f"Req_{len(requirements) + 1}"

                requirements.append(
                    Requirement(
                        rule_id=rule_id,
                        title=r.get("title", "Unnamed Requirement"),
                        description=r.get("description", ""),
                        original_clause=r.get("original_clause", ""),
                        source_reference=r.get("source_reference", ""),
                        status=r.get("status", "draft")
                    )
                )

            if verbose:
                logger.info(f"Successfully extracted {len(requirements)} rules.")
            return requirements

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM requirements output: {e}. Raw response: {response}")
            return []

    except Exception as e:
        logger.error(f"LLM rules extraction call failed: {e}")
        return []


def write_rule_markdown(requirement: Requirement, output_dir: Path) -> Path:
    """
    Format a Requirement into a markdown rule file.

    Args:
        requirement: The Requirement model to write.
        output_dir: The directory to write the file to.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rule_file = output_dir / f"{requirement.rule_id}.md"

    from servo_skull._utils import calculate_flesch_reading_ease
    flesch_score = calculate_flesch_reading_ease(requirement.description)

    md_content = f"""---
type: rule
rule_id: {requirement.rule_id}
title: "{requirement.title}"
status: {requirement.status}
source_reference: "{requirement.source_reference}"
flesch_reading_ease: {flesch_score}
---
# Rule: {requirement.title}

## 📋 Description
{requirement.description}

## 🔍 Original Clause
> "{requirement.original_clause}"

## 📖 Coverage
```query
page where type = "user-story" and references = "{{{{page}}}}"
```
"""
    rule_file.write_text(md_content, encoding="utf-8")
    return rule_file

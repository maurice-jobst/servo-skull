"""LLM Grounder for Servo-Skull gap analysis with hallucination detection."""
import json
import logging
from typing import Any, Optional
import httpx
import tomllib
from pathlib import Path

from servo_skull._prompts import render_gap_analysis_prompt, render_security_check_prompt
from servo_skull._utils import retry, setup_logging
from servo_skull.llm_client import LLMClient
from servo_skull.llm_providers import create_provider
from servo_skull.models import DocumentExtract, GapAnalysis

logger = setup_logging(__name__)


def _load_config() -> dict[str, Any]:
    """Load configuration from providers.toml."""
    paths_to_try = [
        Path(__file__).parent.parent.parent.parent.parent / "config" / "providers.toml",
        Path(__file__).parent.parent.parent.parent / "config" / "providers.toml",
        Path(__file__).parent.parent.parent / "config" / "providers.toml",
    ]
    
    config_path = None
    for p in paths_to_try:
        if p.exists():
            config_path = p
            break
            
    if config_path is None:
        logger.warning("Config file providers.toml not found, using defaults")
        return {
            "providers": {
                "local_gemma": {
                    "type": "ollama",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "model": "gemma4:26b",
                    "temperature": 0.0,
                    "timeout": 300.0,
                },
                "cloud_openai": {
                    "type": "openai",
                    "api_key": "",
                    "model": "gpt-4",
                    "timeout": 300.0,
                }
            },
        }

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def analyze_gaps(
    extract: DocumentExtract,
    codex: str,
    llm_client: Optional[LLMClient] = None
) -> GapAnalysis:
    """
    Perform 4-dimensional gap analysis on extracted document.

    Uses LLM to analyze gaps and scores hallucination by comparing LLM claims
    against the extracted text.

    Args:
        extract: DocumentExtract from extractor
        codex: domain codex framework rules text
        llm_client: LLMClient instance with provider chain. If None, creates default.

    Returns:
        GapAnalysis with gaps, risk_flags, security_flags, hallucination_score
    """
    logger.info(f"Starting gap analysis for document {extract.document_id}")

    # Create LLMClient if not provided
    if llm_client is None:
        config = _load_config()
        routing = config.get("routing", {})
        primary_name = routing.get("primary", "local_gemma")
        fallback_name = routing.get("fallback", "cloud_openai")

        providers = config.get("providers", {})
        primary_config = providers.get(primary_name, {})
        fallback_config = providers.get(fallback_name)

        primary = create_provider(primary_config)

        # Instantiate fallback if config is present and if it requires an API key, it has it
        fallback = None
        if fallback_config:
            from servo_skull.llm_providers import expand_env_vars
            expanded_fallback = expand_env_vars(fallback_config)
            prov_type = expanded_fallback.get("type", "").lower()
            api_key = expanded_fallback.get("api_key", "")

            # If it's a provider requiring api_key (openai, anthropic, gemini), make sure it has one
            if prov_type in ("openai", "anthropic", "gemini"):
                if api_key and not api_key.startswith("$"):
                    fallback = create_provider(fallback_config)
            else:
                fallback = create_provider(fallback_config)

        llm_client = LLMClient(primary=primary, fallback=fallback)

    try:
        # Generate prompts
        prompts = render_gap_analysis_prompt(extract, codex)
        system_prompt = prompts["system"]
        user_prompt = prompts["user"]

        # Call LLM via unified LLMClient (handles fallback automatically)
        logger.info("Calling LLM for gap analysis")
        response = llm_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            document_size_bytes=len(extract.extracted_text.encode('utf-8'))
        )

        # Parse response
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response.split("```json", 1)[1].split("```", 1)[0].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.split("```", 1)[1].split("```", 1)[0].strip()
            
        try:
            llm_output = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return GapAnalysis(
                document_id=extract.document_id,
                gaps={},
                risk_flags=[],
                security_flags=[],
                hallucination_score=1.0,
                grounding_notes="LLM response was not valid JSON",
                llm_model="unknown",
            )

        # Score hallucination
        hallucination_score = hallucination_score_func(extract.extracted_text, llm_output)

        # Build grounding notes
        grounding_notes = _build_grounding_notes(llm_output, hallucination_score)

        # Clean security flags to ensure they are a list of strings
        raw_security_flags = llm_output.get("security_flags", [])
        cleaned_security_flags = []
        for flag in raw_security_flags:
            if isinstance(flag, dict):
                if "risk" in flag:
                    cleaned_security_flags.append(str(flag["risk"]))
                elif "flag" in flag:
                    cleaned_security_flags.append(str(flag["flag"]))
                else:
                    cleaned_security_flags.append(json.dumps(flag))
            elif isinstance(flag, str):
                cleaned_security_flags.append(flag)
            else:
                cleaned_security_flags.append(str(flag))

        # Create GapAnalysis model
        gap_analysis = GapAnalysis(
            document_id=extract.document_id,
            gaps=llm_output.get("gaps", {}),
            risk_flags=llm_output.get("risk_flags", []),
            security_flags=cleaned_security_flags,
            hallucination_score=hallucination_score,
            grounding_notes=grounding_notes,
            llm_model="llm_client",
        )

        logger.info(f"Gap analysis complete. Hallucination score: {hallucination_score:.2f}")
        return gap_analysis

    except httpx.HTTPError as e:
        logger.error(f"LLM request failed: {e}")
        return GapAnalysis(
            document_id=extract.document_id,
            gaps={},
            risk_flags=[],
            security_flags=[],
            hallucination_score=0.0,
            grounding_notes=f"LLM request failed: {type(e).__name__}",
            llm_model="unknown",
        )


def hallucination_score_func(extract_text: str, llm_claims: dict[str, Any]) -> float:
    """
    Score hallucination by comparing LLM claims against extracted text.

    Calculates what fraction of LLM claims are not grounded in the extracted text.

    Args:
        extract_text: Original extracted document text
        llm_claims: LLM analysis dict with gaps, risk_flags, security_flags

    Returns:
        float 0.0-1.0 where 0.0=fully grounded, 1.0=fully speculative
    """
    if not llm_claims:
        return 0.0

    extract_lower = extract_text.lower()
    total_claims = 0
    ungrounded_claims = 0

    # Check gaps
    gaps = llm_claims.get("gaps", {})
    for dimension, gap_list in gaps.items():
        if isinstance(gap_list, list):
            for gap_item in gap_list:
                if isinstance(gap_item, dict):
                    total_claims += 1
                    gap_text = gap_item.get("gap", "").lower()
                    context = gap_item.get("context", "").lower()

                    # Check if gap keywords or context appear in extracted text
                    if not _is_grounded(gap_text, context, extract_lower):
                        ungrounded_claims += 1

    # Check risk_flags
    risk_flags = llm_claims.get("risk_flags", [])
    if isinstance(risk_flags, list):
        for flag_item in risk_flags:
            if isinstance(flag_item, dict):
                total_claims += 1
                risk_text = flag_item.get("risk", "").lower()
                context = flag_item.get("recommendation", "").lower()

                if not _is_grounded(risk_text, context, extract_lower):
                    ungrounded_claims += 1

    # Check security_flags
    security_flags = llm_claims.get("security_flags", [])
    if isinstance(security_flags, list):
        for flag in security_flags:
            if isinstance(flag, str):
                total_claims += 1
                if not _is_grounded(flag.lower(), "", extract_lower):
                    ungrounded_claims += 1

    # Calculate score: 0.0 = fully grounded, 1.0 = fully speculative
    if total_claims == 0:
        return 0.0

    score = ungrounded_claims / total_claims
    return min(1.0, max(0.0, score))


def _is_grounded(claim_text: str, context_text: str, extract_lower: str) -> bool:
    """
    Check if a claim is grounded in extracted text.

    Uses case-insensitive substring matching on key terms.

    Args:
        claim_text: The claim to check
        context_text: Context or explanation
        extract_lower: Extracted text

    Returns:
        True if grounded, False if speculative
    """
    claim_text = claim_text.lower()
    context_text = context_text.lower()
    extract_lower = extract_lower.lower()

    if not claim_text:
        return True

    stop_words = {"for", "all", "the", "and", "our", "not", "but", "with", "from", "this", "that", "security"}

    # Split claim into keywords (3+ chars, not stop words)
    keywords = [w for w in claim_text.split() if len(w) >= 3 and w not in stop_words]
    if not keywords:
        return True

    # Check if any keyword appears in extract
    for keyword in keywords[:3]:  # Check first 3 keywords
        if keyword in extract_lower:
            return True

    # If context available, check context keywords
    if context_text:
        context_keywords = [w for w in context_text.split() if len(w) >= 3 and w not in stop_words]
        for keyword in context_keywords[:2]:
            if keyword in extract_lower:
                return True

    return False


def _build_grounding_notes(llm_output: dict[str, Any], hallucination_score: float) -> str:
    """Build grounding notes summary from LLM output and hallucination score."""
    gaps_count = 0
    for dimension, gap_list in llm_output.get("gaps", {}).items():
        if isinstance(gap_list, list):
            gaps_count += len(gap_list)

    risk_flags_count = len(llm_output.get("risk_flags", []))
    security_flags_count = len(llm_output.get("security_flags", []))

    score_desc = "fully grounded"
    if hallucination_score >= 0.8:
        score_desc = "highly speculative"
    elif hallucination_score >= 0.5:
        score_desc = "partially grounded"
    elif hallucination_score >= 0.2:
        score_desc = "mostly grounded"

    return f"Found {gaps_count} gaps, {risk_flags_count} risk flags, {security_flags_count} security flags. Analysis is {score_desc} (hallucination score: {hallucination_score:.2f})."

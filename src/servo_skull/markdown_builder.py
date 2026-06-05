"""RAG markdown builder with Binaric Cant footer."""
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from servo_skull._utils import setup_logging
from servo_skull.models import DocumentExtract, GapAnalysis, RichMarkdown

logger = setup_logging(__name__)


class MarkdownBuilder:
    """Synthesizes DocumentExtract + GapAnalysis + security analysis into RAG-optimized Markdown."""

    # Keywords to extract for "See Also" links
    SEE_ALSO_KEYWORDS = [
        "api", "specification", "framework", "compliance", "standard",
        "regulation", "directive", "requirement", "architecture",
        "interface", "protocol", "data", "schema", "technical",
        "implementation", "guideline", "procedure", "policy"
    ]

    def build_markdown(
        self,
        extract: DocumentExtract,
        analysis: GapAnalysis,
        security_dict: dict[str, Any],
    ) -> RichMarkdown:
        """
        Synthesize DocumentExtract + GapAnalysis + security analysis into RAG-optimized Markdown.

        Args:
            extract: DocumentExtract from extractor.py
            analysis: GapAnalysis from grounder.py
            security_dict: dict from security_checker.py

        Returns:
            RichMarkdown model with frontmatter, content, see_also, scriptorum_refs, binaric_cant_footer
        """
        logger.info(f"Building markdown for document {extract.document_id}")

        # Build content sections first, so we can run readability scoring on it
        content = self._build_content(extract, analysis, security_dict)

        # Calculate readability
        from servo_skull._utils import calculate_flesch_reading_ease
        flesch_score = calculate_flesch_reading_ease(content)

        # Build YAML frontmatter
        frontmatter = self._build_frontmatter(extract, analysis, security_dict, flesch_score)

        # Extract "See Also" links
        see_also = self._extract_see_also(extract.extracted_text)

        # Extract SCRIPTORUM references
        scriptorum_refs = self._extract_scriptorum_refs(analysis)

        # Build Binaric Cant footer
        binaric_cant_footer = self._build_binaric_cant_footer(
            extract, analysis, security_dict, see_also, scriptorum_refs
        )

        # Generate output path
        output_path = self._generate_output_path(extract)

        # Generate timestamp
        generated_timestamp = datetime.now(timezone.utc).isoformat()

        # Create RichMarkdown model
        rich_markdown = RichMarkdown(
            document_id=extract.document_id,
            frontmatter=frontmatter,
            content=content,
            see_also=see_also,
            scriptorum_refs=scriptorum_refs,
            binaric_cant_footer=binaric_cant_footer,
            output_path=output_path,
            generated_timestamp=generated_timestamp,
        )

        logger.info(f"Markdown built for document {extract.document_id}")
        return rich_markdown

    def _build_frontmatter(
        self,
        extract: DocumentExtract,
        analysis: GapAnalysis,
        security_dict: dict[str, Any],
        flesch_score: float = 0.0,
    ) -> dict[str, Any]:
        """Build YAML frontmatter metadata."""
        security_issues_count = len(security_dict.get("security_issues", []))

        frontmatter = {
            "doc_id": extract.document_id,
            "filename": extract.original_filename,
            "document_type": extract.document_type,
            "extraction_tool": extract.extraction_tool,
            "confidence": extract.confidence,
            "analysis_timestamp": analysis.analysis_timestamp,
            "hallucination_score": analysis.hallucination_score,
            "security_issues_count": security_issues_count,
            "flesch_reading_ease": flesch_score,
        }

        return frontmatter

    def _build_content(
        self,
        extract: DocumentExtract,
        analysis: GapAnalysis,
        security_dict: dict[str, Any],
    ) -> str:
        """Build main Markdown content with all sections."""
        sections = []

        # Section 1: Document Overview
        sections.append(self._build_section_overview(extract))

        # Section 2: Extracted Content
        sections.append(self._build_section_extracted_content(extract))

        # Section 3: Gap Analysis
        sections.append(self._build_section_gap_analysis(analysis))

        # Section 4: Risk Assessment
        sections.append(self._build_section_risk_assessment(analysis, security_dict))

        # Section 5: Grounding & Hallucination
        sections.append(self._build_section_grounding(analysis))

        # Section 6: Recommendations
        sections.append(self._build_section_recommendations(security_dict))

        # Combine all sections
        content = "\n\n".join(sections)
        return content

    def _build_section_overview(self, extract: DocumentExtract) -> str:
        """Build Document Overview section."""
        lines = [
            "# Document Overview",
            "",
            f"**Filename:** {extract.original_filename}",
            f"**Document Type:** {extract.document_type}",
            f"**Extraction Method:** {extract.extraction_tool}",
            f"**Confidence:** {extract.confidence:.1%}",
            f"**Document ID:** `{extract.document_id}`",
        ]

        if extract.metadata:
            lines.append("")
            lines.append("**Metadata:**")
            for key, value in extract.metadata.items():
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def _build_section_extracted_content(self, extract: DocumentExtract) -> str:
        """Build Extracted Content section (first 500 chars)."""
        text_sample = extract.extracted_text
        truncated = False

        if len(text_sample) > 500:
            text_sample = text_sample[:500]
            truncated = True

        # Escape special Markdown characters
        text_sample = self._escape_markdown(text_sample)

        lines = [
            "# Extracted Content",
            "",
            "```",
            text_sample,
            "```",
        ]

        if truncated:
            lines.append("")
            lines.append("*(Content truncated. Full text available in extraction artifact.)*")

        return "\n".join(lines)

    def _build_section_gap_analysis(self, analysis: GapAnalysis) -> str:
        """Build Gap Analysis section for all 4 dimensions."""
        lines = ["# Gap Analysis", ""]

        # Four dimensions: stakeholder, methodology, subsystems, compliance
        dimensions = ["stakeholder", "methodology", "subsystems", "compliance"]

        for dimension in dimensions:
            lines.append(f"## {dimension.capitalize()}")

            gaps = analysis.gaps.get(dimension, [])
            if gaps:
                for gap_item in gaps:
                    gap_text = gap_item.get("gap", "Unknown gap")
                    severity = gap_item.get("severity", "medium")
                    context = gap_item.get("context", "")

                    lines.append(f"- **{gap_text}** (severity: {severity})")
                    if context:
                        lines.append(f"  - Context: {context}")
            else:
                lines.append("- No gaps identified")

            lines.append("")

        return "\n".join(lines)

    def _build_section_risk_assessment(
        self,
        analysis: GapAnalysis,
        security_dict: dict[str, Any],
    ) -> str:
        """Build Risk Assessment section."""
        lines = ["# Risk Assessment", ""]

        # Risk flags
        if analysis.risk_flags:
            lines.append("## Risk Flags")
            for flag_item in analysis.risk_flags:
                risk_text = flag_item.get("risk", "Unknown risk")
                severity = flag_item.get("severity", "medium")
                lines.append(f"- **{risk_text}** (severity: {severity})")
            lines.append("")

        # Security flags
        if analysis.security_flags:
            lines.append("## Security Flags")
            for flag in analysis.security_flags:
                lines.append(f"- {flag}")
            lines.append("")

        # Security issues from security_checker
        security_issues = security_dict.get("security_issues", [])
        if security_issues:
            lines.append("## Security Issues")
            for issue in security_issues:
                if isinstance(issue, dict):
                    issue_text = issue.get("issue", str(issue))
                else:
                    issue_text = str(issue)
                lines.append(f"- {issue_text}")
            lines.append("")

        # Misinformation risks
        misinformation = security_dict.get("misinformation_risks", [])
        if misinformation:
            lines.append("## Misinformation Risks")
            for risk in misinformation:
                if isinstance(risk, dict):
                    risk_text = risk.get("risk", str(risk))
                else:
                    risk_text = str(risk)
                lines.append(f"- {risk_text}")
            lines.append("")

        # AI watermarks
        ai_watermarks = security_dict.get("ai_watermarks", [])
        if ai_watermarks:
            lines.append("## AI Watermarks")
            for watermark in ai_watermarks:
                if isinstance(watermark, dict):
                    watermark_text = watermark.get("watermark", str(watermark))
                else:
                    watermark_text = str(watermark)
                lines.append(f"- {watermark_text}")
            lines.append("")

        # Fraud indicators
        fraud_indicators = security_dict.get("fraud_indicators", [])
        if fraud_indicators:
            lines.append("## Fraud Indicators")
            for indicator in fraud_indicators:
                if isinstance(indicator, dict):
                    indicator_text = indicator.get("indicator", str(indicator))
                else:
                    indicator_text = str(indicator)
                lines.append(f"- {indicator_text}")

        return "\n".join(lines)

    def _build_section_grounding(self, analysis: GapAnalysis) -> str:
        """Build Grounding & Hallucination section."""
        lines = [
            "# Grounding & Hallucination",
            "",
        ]

        # Hallucination score with explanation
        score = analysis.hallucination_score
        if score >= 0.8:
            score_desc = "highly speculative"
        elif score >= 0.5:
            score_desc = "partially grounded"
        elif score >= 0.2:
            score_desc = "mostly grounded"
        else:
            score_desc = "fully grounded"

        lines.append(f"**Hallucination Score:** {score:.2f}/1.0 ({score_desc})")
        lines.append("")
        lines.append("This score indicates the fraction of LLM claims not grounded in the original extracted text. "
                     "A score of 0.0 means fully grounded; 1.0 means fully speculative.")
        lines.append("")

        # Grounding notes
        if analysis.grounding_notes:
            lines.append("**Grounding Notes:**")
            lines.append("")
            lines.append(analysis.grounding_notes)

        return "\n".join(lines)

    def _build_section_recommendations(self, security_dict: dict[str, Any]) -> str:
        """Build Recommendations section."""
        lines = ["# Recommendations", ""]

        recommendations = security_dict.get("recommendations", [])
        if recommendations:
            for rec in recommendations:
                if isinstance(rec, dict):
                    rec_text = rec.get("recommendation", str(rec))
                else:
                    rec_text = str(rec)
                lines.append(f"- {rec_text}")
        else:
            lines.append("No specific recommendations at this time.")

        return "\n".join(lines)

    def _extract_see_also(self, extracted_text: str) -> list[tuple[str, str]]:
        """
        Extract "See Also" links from extracted text.

        Uses keyword matching to identify related documents/concepts.
        Limits to 5 most-relevant links.

        Returns:
            List of (title, context) tuples
        """
        see_also = []

        # Find sentences containing keywords
        sentences = re.split(r'[.!?]+', extracted_text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            for keyword in self.SEE_ALSO_KEYWORDS:
                if keyword in sentence_lower:
                    # Extract potential title from this sentence
                    # Look for capitalized phrases or quoted text
                    title_match = re.search(r'(?:[""]([^""]*?)[""]|([A-Z][^.!?]*?))(?:\s+(?:is|are|was|were|in|from|for))?', sentence)
                    if title_match:
                        title = title_match.group(1) or title_match.group(2)
                        title = title.strip()

                        if title and len(title) > 3 and len(title) < 100:
                            # Use keyword as context
                            context = f"Related to {keyword}"
                            see_also.append((title, context))

                            if len(see_also) >= 5:
                                break

            if len(see_also) >= 5:
                break

        # Deduplicate by title
        seen_titles = set()
        deduped = []
        for title, context in see_also:
            if title not in seen_titles:
                seen_titles.add(title)
                deduped.append((title, context))

        return deduped[:5]  # Limit to 5

    def _extract_scriptorum_refs(self, analysis: GapAnalysis) -> list[str]:
        """
        Extract SCRIPTORUM references from gap analysis.

        Source documents referenced by gaps/risks.

        Returns:
            List of document IDs or filenames
        """
        refs = []

        # Extract from gaps
        for dimension, gap_list in analysis.gaps.items():
            if isinstance(gap_list, list):
                for gap_item in gap_list:
                    if isinstance(gap_item, dict):
                        context = gap_item.get("context", "")
                        if context:
                            # Look for document references in context
                            # e.g., "tender_specification.pdf", "compliance_2025.md"
                            doc_matches = re.findall(r'\b[\w\-]+\.(?:pdf|md|docx|txt)\b', context, re.IGNORECASE)
                            refs.extend(doc_matches)

        # Extract from risk flags
        for flag_item in analysis.risk_flags:
            if isinstance(flag_item, dict):
                recommendation = flag_item.get("recommendation", "")
                if recommendation:
                    doc_matches = re.findall(r'\b[\w\-]+\.(?:pdf|md|docx|txt)\b', recommendation, re.IGNORECASE)
                    refs.extend(doc_matches)

        # Deduplicate
        refs = list(set(refs))

        return refs

    def _build_binaric_cant_footer(
        self,
        extract: DocumentExtract,
        analysis: GapAnalysis,
        security_dict: dict[str, Any],
        see_also: list[tuple[str, str]],
        scriptorum_refs: list[str],
    ) -> dict[str, Any]:
        """
        Build Binaric Cant footer (compressed JSON summary for downstream agents).

        Returns:
            Dict with: doc_id, doc_type, gaps, risks, hallucination_score, security_issues, extraction_confidence, scriptorum_refs
        """
        # Count gaps per dimension
        gaps_count = 0
        gap_dims = []
        for dimension, gap_list in analysis.gaps.items():
            if isinstance(gap_list, list) and len(gap_list) > 0:
                gaps_count += len(gap_list)
                gap_dims.append(dimension)

        # Count risks and collect severity levels
        risks_count = len(analysis.risk_flags)
        risk_severities = []
        for flag_item in analysis.risk_flags:
            if isinstance(flag_item, dict):
                severity = flag_item.get("severity", "medium")
                if severity not in risk_severities:
                    risk_severities.append(severity)

        # Count security issues
        security_issues_count = len(security_dict.get("security_issues", []))

        footer = {
            "doc_id": extract.document_id,
            "doc_type": extract.document_type,
            "gaps": {
                "count": gaps_count,
                "dims": gap_dims,
            },
            "risks": {
                "count": risks_count,
                "severity": risk_severities,
            },
            "hallucination_score": analysis.hallucination_score,
            "security_issues": security_issues_count,
            "extraction_confidence": extract.confidence,
            "scriptorum_refs": scriptorum_refs,
        }

        return footer

    def _generate_output_path(self, extract: DocumentExtract) -> str:
        """Generate output file path for the markdown."""
        # Format: <date>-<doc-type>-<filename-slug>.md
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filename_slug = re.sub(r'[^\w\-]', '_', extract.original_filename.replace('.', '_'))
        output_filename = f"{timestamp}-{extract.document_type}-{filename_slug}.md"

        return f"artifacts/{output_filename}"

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape special Markdown characters in text."""
        # Escape backslashes first
        text = text.replace("\\", "\\\\")
        # Escape backticks to prevent code interpretation
        text = text.replace("`", "\\`")
        # Note: We don't escape *, _, etc. in code blocks since they're already escaped by triple backticks
        return text

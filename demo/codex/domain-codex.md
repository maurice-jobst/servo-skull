# Domain Codex — Web Platform Procurement

This codex is the grounding framework for gap analysis. Every requirements
document is scored against these rules. A gap exists wherever a document is
silent on, ambiguous about, or in conflict with a rule below.

## Stakeholder Ecosystem

- **Legal & GRC** — Every contract must state SLA penalties, liability caps,
  and data-residency commitments. A bare availability target without remedy
  terms is a gap.
- **Sales & Business Development** — Commercial viability and pricing model
  must be explicit. "Fixed budget" without a figure or billing trigger is a gap.
- **Engineering / Architecture** — Integration points must name a protocol or
  standard (e.g. OIDC, SAML, REST, webhook). "Integrate with X" alone is a gap.
- **Finance & Controlling** — Billing triggers and milestone payments must be
  defined where phased billing is mentioned.
- **Operations & Support** — Disaster recovery, backup cadence, and support
  escalation must be specified for any production system.

## Delivery Methodology Stack

- **Governance** — A stage-gate or business-case checkpoint is expected
  (PRINCE2 / PMBOK). Silence on governance is a gap.
- **Execution** — The delivery cadence (e.g. Agile/Scrum sprints) should be
  named where a timeline is committed.
- **Operations** — A service-management model (ITIL 4 / lightweight CSM) is
  expected for anything with an availability target.

## Affected Components

- **Data Layer** — Any export or warehouse integration must specify the record
  schema, format, and retention period.
- **Service Layer** — APIs and business logic must state authentication and
  authorization expectations.
- **Client / Edge Layer** — Any public-facing UI must state accessibility
  conformance (WCAG 2.2) and supported browsers.
- **Integration Layer** — Third-party connectors must name the protocol,
  payload format, and failure-handling behavior.

## Compliance Stack

- **GDPR / data protection** — Storing customer-submitted data requires a stated
  lawful basis, retention limit, and subject-rights handling. Silence is a gap.
- **SOC 2 / ISO 27001** — Security controls, audit trails, and access management
  must be addressed for any system handling personal data.
- **WCAG 2.2 / accessibility** — A public web form must declare a conformance
  level (A / AA / AAA).
- **Domain-specific** — Any sector standard named in the document must be
  cross-checked for completeness.

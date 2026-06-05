# Request for Proposal — Customer Feedback Portal

**Issuing organization:** Globex Corporation
**Reference:** RFP-2026-CFP-014
**Document type:** Functional & non-functional requirements specification

---

## 1. Background

Globex Corporation operates a network of regional service centers and wishes to
procure a web-based Customer Feedback Portal. The portal will let end customers
submit feedback, track the status of their submissions, and receive resolution
notices. Internal staff will triage, route, and respond to submissions.

We expect the platform to be delivered within two quarters and to integrate with
our existing identity provider.

## 2. Functional Requirements

- **FR-1** Customers shall submit feedback through a public web form.
- **FR-2** Each submission shall receive a unique tracking reference.
- **FR-3** Customers shall view the status of their submissions after authenticating.
- **FR-4** Staff shall triage incoming submissions into categories.
- **FR-5** The system shall notify customers when a submission status changes.
- **FR-6** Staff shall generate monthly summary reports of submission volumes.

## 3. Non-Functional Requirements

- **NFR-1** The portal shall support up to 5,000 concurrent users.
- **NFR-2** Page responses shall complete within 2 seconds under nominal load.
- **NFR-3** The portal shall be available 99.5% of the time, measured monthly.

## 4. Integration

- The portal shall authenticate users via the corporate identity provider.
- The portal shall export resolved-submission records to the data warehouse.

## 5. Constraints

- The solution shall be hosted in the vendor's managed cloud environment.
- The project budget is fixed; phased billing is acceptable.

## 6. Out of Scope

- Native mobile applications are not required in this phase.

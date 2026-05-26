# Refiner regression report
_Generated 2026-05-26 05:18:24 UTC_

## Summary
- Total wrong cases: **86**
- Matched: **79** (91.9%)
- Unmatched: **7**

## Per-Status fix rates

### `Approve with edits` (n=51)
- Name changed: 20 / 51 (39%)
- Description changed: 31 / 51 (61%)
- Keywords changed: 51 / 51 (100%)
- Now flagged dup/oos/mis: 18 / 51 (35%)
- Newly flagged (was clean in A): 15 / 51 (29%)
- Flag breakdown in B: is_duplicate=8, is_out_of_scope=8, is_misinterpretation=4

### `Reject` (n=26)
- Name changed: 12 / 26 (46%)
- Description changed: 16 / 26 (62%)
- Keywords changed: 26 / 26 (100%)
- Now flagged dup/oos/mis: 17 / 26 (65%)
- Newly flagged (was clean in A): 16 / 26 (62%)
- Flag breakdown in B: is_duplicate=3, is_out_of_scope=13, is_misinterpretation=2

### `See note above` (n=1)
- Name changed: 1 / 1 (100%)
- Description changed: 1 / 1 (100%)
- Keywords changed: 1 / 1 (100%)
- Now flagged dup/oos/mis: 0 / 1 (0%)
- Newly flagged (was clean in A): 0 / 1 (0%)
- Flag breakdown in B: is_duplicate=0, is_out_of_scope=0, is_misinterpretation=0

### `__none__` (n=1)
- Name changed: 0 / 1 (0%)
- Description changed: 1 / 1 (100%)
- Keywords changed: 1 / 1 (100%)
- Now flagged dup/oos/mis: 0 / 1 (0%)
- Newly flagged (was clean in A): 0 / 1 (0%)
- Flag breakdown in B: is_duplicate=0, is_out_of_scope=0, is_misinterpretation=0

## Approve-with-edits quality
- Cases: **51**  (alignment method: `llm`)
- Name edits with a quoted gold suggestion: **16**
- Name-edit alignment: **4/16** (25%)
- **Falsely flagged** dup/oos/mis (these are APPROVED skills): **18/51** (dup=8, oos=8, mis=4)

  Misaligned name edits (output vs reviewer suggestion):
  - `Ethical Compliance Guidance` → out=`Ethical Compliance Guidance` | want=`Compliance Guidance`
  - `Asset Management Implementation` → out=`Asset Management Implementation` | want=`Asset Protection Management`
  - `Feedback Response Management` → out=`Documentation Submission` | want=`Stakeholder Feedback Elicitation`
  - `Digital Technology Utilization` → out=`Digital Technology Utilisation` | want=`New Technology Investigation an Plan`
  - `Documentation Preparation` → out=`Evaluation Documentation` | want=`Evaluation Findings Documentation`
  - `Information Synthesis Writing` → out=`Information Synthesis Writing` | want=`Information Synthesis`
  - `Log Analysis` → out=`Log Analysis` | want=`Threat Assessment`
  - `Template Provisioning` → out=`Provisioning Services Configuration` | want=`Service and Template Provisioning`
  - `System Vulnerability Identification` → out=`System Vulnerability Identification` | want=`ISP Security Testing`
  - `Framework Utilization` → out=`Framework Utilisation` | want=`IDE and Frameworks Utilisation`
  - `Process Implementation` → out=`Process Implementation` | want=`Client Support Implementation`
  - `Technical Report Writing` → out=`Technical Report Writing` | want=`Test Result Reporting`

## Unmatched cases (7)
These rows in the wrong-cases file have no `(unit_code, org_name)` match in the latest refiner output — usually due to extractor drift.

- `ICTCYS407` | `Reliability Consistency Analysis` | status: See note above re Threat Data Analysis
- `ICTCYS610` | `Plan Alignment Evaluation` | status: Approve with edits
- `ICTICT527` | `Process Suitability Assessment` | status: Approve with edits
- `ICTNWK546` | `Document Submission` | status: Approve with edits
- `ICTSAS527` | `Documentation Submission Management` | status: Approve with edits
- `ICTTEN417` | `Network Functionality Validation` | status: Approve with edits
- `ICTTEN419` | `Worksite Coordination` | status: Approve with edits

# Refiner regression report
_Generated 2026-05-26 07:22:35 UTC_

## Summary
- Total wrong cases: **86**
- Matched: **79** (91.9%)
- Unmatched: **7**

## Per-Status fix rates

### `Approve with edits` (n=51)
- Name changed: 25 / 51 (49%)
- Description changed: 30 / 51 (59%)
- Keywords changed: 51 / 51 (100%)
- Now flagged dup/oos/mis: 16 / 51 (31%)
- Newly flagged (was clean in A): 13 / 51 (25%)
- Flag breakdown in B: is_duplicate=6, is_out_of_scope=7, is_misinterpretation=3

### `Reject` (n=26)
- Name changed: 15 / 26 (58%)
- Description changed: 15 / 26 (58%)
- Keywords changed: 26 / 26 (100%)
- Now flagged dup/oos/mis: 14 / 26 (54%)
- Newly flagged (was clean in A): 13 / 26 (50%)
- Flag breakdown in B: is_duplicate=5, is_out_of_scope=8, is_misinterpretation=2

### `See note above` (n=1)
- Name changed: 1 / 1 (100%)
- Description changed: 0 / 1 (0%)
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
- Name-edit alignment: **5/16** (31%)
- **Falsely flagged** dup/oos/mis (these are APPROVED skills): **16/51** (dup=6, oos=7, mis=3)

  Misaligned name edits (output vs reviewer suggestion):
  - `Ethical Compliance Guidance` → out=`Ethical Responsibility Explanation` | want=`Compliance Guidance`
  - `Asset Management Implementation` → out=`Asset Management Implementation` | want=`Asset Protection Management`
  - `Feedback Response Management` → out=`Feedback Handling` | want=`Stakeholder Feedback Elicitation`
  - `Digital Technology Utilization` → out=`Digital Technology Utilisation` | want=`New Technology Investigation an Plan`
  - `Documentation Preparation` → out=`Evaluation Documentation` | want=`Evaluation Findings Documentation`
  - `Information Synthesis Writing` → out=`Information Synthesis Writing` | want=`Information Synthesis`
  - `Log Analysis` → out=`Logging Technique Analysis` | want=`Threat Assessment`
  - `System Vulnerability Identification` → out=`System Vulnerability Identification` | want=`ISP Security Testing`
  - `Forensic Evidence Collection` → out=`Forensic Evidence Procedure` | want=`Forensic Evidence Procedure Development`
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

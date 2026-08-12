# Grounding-complete CG-Bench subset

This directory is a mechanically derived, case-level quarantine of the
256-case CG-Bench navigation artifact. Two cases contained one truncated
Qwen-VL JSON response each. Because the source videos are not present on this
machine, those observations were not reconstructed from partial text.

The quarantine uses only `descriptor_status != grounded_qwen_vl_read`. It does
not inspect answer text, answer keys, clue intervals, relevance labels, or
navigation outcomes. Whole cases are removed so no partial reasoning
trajectory is presented as complete.

Result:

- 254 retained cases across train/validation/test;
- 662/662 grounded transitions;
- 662/662 Qwen3-VL-Embedding-2B rows;
- contiguous embedding row indices and a regenerated checksum;
- zero validator errors;
- no training performed.

This closes the descriptor/embedding integrity gate only. `formal_eligible` and
`training_ready` remain false until a video-disjoint frozen L1/L1.5 set retains
enough multi-hop clues and delayed-reasoning cases.

`quarantine_report.json` records the two excluded case IDs and the exclusion
contract. `validation_report.json` is the independent artifact validation.
The original 256-case files in the parent directory remain unchanged.

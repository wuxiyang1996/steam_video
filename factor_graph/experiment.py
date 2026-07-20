"""Run the locked four-arm GTSAM correction pilot experiment."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

from .pilot import run_correction_pilot


VARIANTS = ("correction", "frozen", "no_loop_closure", "shuffled_evidence")


def run_experiment() -> dict[str, Any]:
    runs = {variant: run_correction_pilot(variant) for variant in VARIANTS}
    summaries: dict[str, dict[str, Any]] = {}
    for variant, run in runs.items():
        trace = run["public_categorical_trace"]
        before = trace["belief_before"]
        after = trace["belief_after"]
        summaries[variant] = {
            "conflict_before": before["conflict_a_c"],
            "conflict_after": after["conflict_a_c"],
            "identity_b_c_before": before["identity_b_c"],
            "identity_b_c_after": after["identity_b_c"],
            "next_step": trace["replanned_next_step"],
        }
    gates = {
        "observed_measurement_updates_conflict": (
            summaries["correction"]["conflict_after"] == "accepted"
        ),
        "loop_closure_changes_identity_belief": (
            summaries["correction"]["identity_b_c_after"]
            != summaries["correction"]["identity_b_c_before"]
        ),
        "frozen_posterior_does_not_correct": (
            summaries["frozen"]["identity_b_c_after"]
            == summaries["frozen"]["identity_b_c_before"]
        ),
        "shuffled_measurement_does_not_fake_target_correction": (
            summaries["shuffled_evidence"]["conflict_after"]
            != summaries["correction"]["conflict_after"]
        ),
    }
    return {
        "schema_version": "gtsam_correction_experiment/v0.1",
        "status": "pilot_not_benchmark",
        "runtime": {
            "gtsam": version("gtsam"),
            "python": platform.python_version(),
        },
        "variants": list(VARIANTS),
        "summaries": summaries,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_experiment()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

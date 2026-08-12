from steam_video_new.implicit_world_model.full_graph_iwm.matched_action_diagnostic import (
    analyze_matched_initial_actions,
)


def _transition(action, outcome, progress="unchanged", answerability="not_ready"):
    return {
        "action": action,
        "observation": {"outcome": outcome},
        "belief_delta": {
            "progress": progress,
            "answerability_after": answerability,
        },
    }


def _run(arm, selected_target, predictions, *, false_elimination=False):
    selected = {
        "kind": "start_at",
        "source_id": None,
        "target_id": selected_target,
        "edge_id": None,
        "relation": None,
    }
    paths = []
    for target, hypothesis_outcomes in predictions.items():
        action = {**selected, "target_id": target}
        paths.append(
            {
                "conditioned_outcomes": [
                    {
                        "trajectory_id": f"trajectory:{index}",
                        "hypothesis": f"hypothesis:{index}",
                        "transitions": [_transition(action, outcome)],
                    }
                    for index, outcome in enumerate(hypothesis_outcomes)
                ]
            }
        )
    return {
        "case_id": "case:one",
        "arm": arm,
        "steps": [
            {
                "observation_id": selected_target,
                "decision": {
                    "selected_action": selected,
                    "imagined_paths": paths,
                    "preference_audit": {
                        "evidence_scheduler_used": arm == "world_model_guided"
                    },
                },
            }
        ],
        "realized_labels_evaluator_only": [
            {
                "observation_id": selected_target,
                "observation_outcome": "inconclusive",
                "belief_delta": {
                    "progress": "unchanged",
                    "answerability_after": "not_ready",
                },
            }
        ],
        "metrics": {
            "false_correct_hypothesis_elimination": false_elimination,
        },
    }


def test_matched_action_diagnostic_uses_one_action_unit_and_excludes_conflicts() -> (
    None
):
    wm = _run(
        "world_model_guided",
        "node:a",
        {
            "node:a": ["inconclusive", "inconclusive"],
            "node:b": ["support", "counterevidence"],
        },
        false_elimination=True,
    )
    shuffled = _run(
        "shuffled_world_model_prediction",
        "node:b",
        {
            "node:a": ["support", "counterevidence"],
            "node:b": ["inconclusive", "inconclusive"],
        },
    )

    report = analyze_matched_initial_actions([{"runs": [wm, shuffled]}])

    assert report["truth_action_count"] == 2
    assert report["arms"]["world_model_guided"]["matched_executed_action_count"] == 2
    assert (
        report["arms"]["world_model_guided"]["hypothesis_invariant_action_count"] == 1
    )
    assert (
        report["arms"]["world_model_guided"]["calibration"]["observation_outcome"][
            "row_count"
        ]
        == 1
    )
    assert (
        report["scheduler_reliance"]["world_model_guided"]["scheduler_reliance_rate"]
        == 1.0
    )
    assert (
        report["correction_safety"][
            "false_elimination_with_all_inconclusive_reads_count"
        ]
        == 1
    )

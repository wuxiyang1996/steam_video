from steam_video_new.implicit_world_model.full_graph_iwm.zero_shot_diagnostic import (
    analyze_zero_shot_artifacts,
)


def test_zero_shot_diagnostic_separates_pre_read_and_post_read_failures() -> None:
    before = {
        "trajectory_id": "trajectory:correct",
        "hypothesis": "answer",
        "belief": {"missing_roles": ["event"], "answerability": "not_ready"},
    }
    after = {
        "trajectory_id": "trajectory:correct",
        "hypothesis": "answer",
        "status": "supported",
        "belief": {"missing_roles": [], "answerability": "ready"},
    }
    artifact = {
        "errors": [],
        "method_failures": [],
        "action_divergence": [
            {
                "candidate_arm": "no_world_model",
                "reference_action_ids": ["read:a"],
                "candidate_action_ids": ["read:b"],
            }
        ],
        "runs": [
            {
                "case_id": "case:one",
                "arm": "world_model_guided",
                "steps": [
                    {
                        "observation_id": "node:a",
                        "pool_before": {"trajectories": [before]},
                        "assessments": [
                            {
                                "trajectory_id": "trajectory:correct",
                                "effect": "complete",
                            }
                        ],
                        "pool_after": {"trajectories": [after]},
                        "decision": {
                            "imagined_paths": [
                                {
                                    "conditioned_outcomes": [
                                        {
                                            "transitions": [
                                                {
                                                    "action": {"reads_evidence": True},
                                                    "observation": {
                                                        "outcome": "inconclusive"
                                                    },
                                                    "belief_delta": {
                                                        "progress": "advanced",
                                                        "answerability_after": "ready",
                                                        "contradiction_change": "unchanged",
                                                        "resolved_roles": ["event"],
                                                        "opened_roles": [],
                                                    },
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        },
                    }
                ],
                "realized_labels_evaluator_only": [
                    {
                        "observation_id": "node:a",
                        "observation_outcome": "support",
                    }
                ],
                "empirical_audit": {
                    "joint_chain_coverage": {"outcome_retention_rate": 1.0},
                    "post_read_belief_divergence": {"divergence_rate": 1.0},
                    "transition_calibration": {"rows": []},
                },
                "metrics": {
                    "answer_correct": True,
                    "clue_recall": 1.0,
                    "answer_abstained": False,
                },
            }
        ],
    }
    hidden = {"cases": [{"case_id": "case:one", "answer_text": "answer"}]}

    report = analyze_zero_shot_artifacts([artifact], hidden)

    assert report["runtime_integrity"]["observed_first_action_divergence_count"] == 1
    assert (
        report["pre_read_transition"][
            "obvious_semantic_contract_violation_count"
        ]
        == 1
    )
    assert (
        report["post_read_oracle_observation"][
            "correct_hypothesis_supported_or_complete_on_support_rate"
        ]
        == 1.0
    )

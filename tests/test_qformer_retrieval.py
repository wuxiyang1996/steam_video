import numpy as np

from streaming_3bench.baseline.qformer_retrieval import ThreeBenchRetriever
from streaming_3bench.baseline.qformer_rag_stats import holm_adjust, paired_report


def _row(index, start, end):
    return {
        "node_id": f"node-{index}",
        "source_video_id": "video",
        "clip_id": f"clip-{index}",
        "video_path": "/video.mp4",
        "time_span": {"start_s": start, "end_s": end},
    }


def test_uniform_arm_uses_same_visible_pool_and_clips_future_end():
    retriever = ThreeBenchRetriever.__new__(ThreeBenchRetriever)
    retriever.by_path = {
        "/video.mp4": [_row(0, 0, 30), _row(1, 30, 60), _row(2, 60, 90)]
    }
    selected = retriever.retrieve(
        video_path="/video.mp4",
        question_text="unused",
        visible_until_s=45,
        arm="uniform",
        top_k=3,
    )
    assert [row["clip"]["clip_id"] for row in selected] == ["clip-0", "clip-1"]
    assert selected[-1]["clip"]["end_s"] == 45
    assert all(row["clip"]["start_s"] < 45 for row in selected)


def test_paired_statistics_are_deterministic_and_holm_monotone():
    report = paired_report(
        [True, True, False, True], [False, True, False, False], seed=7, samples=1000
    )
    assert report["delta"] == 0.5
    assert report["left_only_correct"] == 2
    adjusted = holm_adjust([("a", 0.01), ("b", 0.04), ("c", 0.2)])
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_visual_scoring_copies_read_only_mmap_values():
    visual = np.asarray([1.0, 0.0], dtype=np.float32)
    visual.setflags(write=False)

    class Store:
        def node(self, _node_id):
            return {"validity": np.asarray([True, False, True, True]), "features": {"visual": visual}}

    retriever = ThreeBenchRetriever.__new__(ThreeBenchRetriever)
    retriever.features = Store()
    scores = retriever._visual_scores([{"node_id": "node"}], np.asarray([1.0, 0.0]))
    assert scores.tolist() == [1.0]
    assert visual.tolist() == [1.0, 0.0]

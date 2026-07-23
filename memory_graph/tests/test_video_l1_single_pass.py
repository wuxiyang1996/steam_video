from pathlib import Path
from threading import Barrier, Lock

import memory_graph.video_l1 as video_l1
from memory_graph.video_l1 import QwenVideoL1Extractor, VideoL1Config


class _OneWindowProvider:
    provider_name = "test-one-window"

    def windows(self, video_path: str | Path, *, duration_s: float):
        del video_path
        return [
            {
                "start_s": 0.0,
                "end_s": duration_s,
                "purpose": "test",
                "boundary_reason": "stream_end",
            }
        ]


class _TwoWindowProvider:
    provider_name = "test-two-windows"

    def windows(self, video_path: str | Path, *, duration_s: float):
        del video_path, duration_s
        return [
            {"start_s": 0.0, "end_s": 2.0, "purpose": "test"},
            {"start_s": 2.0, "end_s": 4.0, "purpose": "test"},
        ]


class _GroundedClient:
    model = "test-vlm"

    def __init__(self) -> None:
        self.calls = 0

    def perceive(self, prompt, *, image_urls, system):
        del prompt, image_urls, system
        self.calls += 1
        return {
            "events": [
                {
                    "predicate": "opens box",
                    "coarse_start_s": 0.0,
                    "coarse_end_s": 2.0,
                    "grounding_status": "observed",
                    "action_kind": "action",
                    "evidence_frames": [0, 1],
                    "participants": [
                        {
                            "role": "agent",
                            "entity_type": "person",
                            "surface": "woman",
                            "visual_signature": "red shirt",
                            "evidence_frames": [0, 1],
                        }
                    ],
                    "states": [
                        {
                            "participant_index": 0,
                            "attribute": "pose",
                            "value": "standing",
                            "polarity": "positive",
                            "evidence_frames": [0],
                        }
                    ],
                }
            ]
        }


def test_grounded_single_pass_uses_one_model_call_per_window(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    monkeypatch.setattr(video_l1, "_video_duration_s", lambda _: 2.0)
    monkeypatch.setattr(
        video_l1,
        "_sample_frame_data_uris",
        lambda *args, **kwargs: (
            ["data:image/jpeg;base64,AA==", "data:image/jpeg;base64,AA=="],
            [
                {"frame_index": 0, "time_s": 0.0},
                {"frame_index": 1, "time_s": 2.0},
            ],
        ),
    )
    client = _GroundedClient()
    extractor = QwenVideoL1Extractor(
        client=client,
        config=VideoL1Config(localization_mode="grounded_single_pass"),
        window_provider=_OneWindowProvider(),
    )

    result = extractor.extract(video_path=video, video_id="video-a")

    assert client.calls == 1
    assert result.protocol_version == "video-only-l1/v0.2-grounded-single-pass"
    assert result.coarse_window_count == 1
    assert result.model_call_count == 1
    assert result.fine_localized_count == 1
    assert result.nodes[0].text == "opens box"
    assert result.nodes[0].metadata["localization"]["evidence_frames"] == [0, 1]


def test_grounded_single_pass_concurrently_scans_windows_with_isolated_clients(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    monkeypatch.setattr(video_l1, "_video_duration_s", lambda _: 4.0)
    monkeypatch.setattr(
        video_l1,
        "_sample_frame_data_uris",
        lambda *args, **kwargs: (
            ["data:image/jpeg;base64,AA==", "data:image/jpeg;base64,AA=="],
            [
                {"frame_index": 0, "time_s": 0.0},
                {"frame_index": 1, "time_s": 2.0},
            ],
        ),
    )
    barrier = Barrier(2, timeout=2.0)
    lock = Lock()
    client_ids: list[int] = []

    class ConcurrentClient:
        model = "test-vlm"

        def perceive(self, prompt, *, image_urls, system):
            del prompt, image_urls, system
            with lock:
                client_ids.append(id(self))
            barrier.wait()
            return {"events": []}

    extractor = QwenVideoL1Extractor(
        client=ConcurrentClient(),
        client_factory=ConcurrentClient,
        config=VideoL1Config(
            localization_mode="grounded_single_pass",
            request_concurrency=2,
        ),
        window_provider=_TwoWindowProvider(),
    )

    result = extractor.extract(video_path=video, video_id="video-a")

    assert result.coarse_window_count == 2
    assert result.model_call_count == 2
    assert len(client_ids) == 2
    assert len(set(client_ids)) == 2


def test_grounded_single_pass_rejects_event_without_frame_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    monkeypatch.setattr(video_l1, "_video_duration_s", lambda _: 2.0)
    monkeypatch.setattr(
        video_l1,
        "_sample_frame_data_uris",
        lambda *args, **kwargs: (
            ["data:image/jpeg;base64,AA=="],
            [{"frame_index": 0, "time_s": 0.0}],
        ),
    )
    client = _GroundedClient()
    original = client.perceive

    def without_evidence(*args, **kwargs):
        payload = original(*args, **kwargs)
        payload["events"][0]["evidence_frames"] = []
        return payload

    client.perceive = without_evidence  # type: ignore[method-assign]
    extractor = QwenVideoL1Extractor(
        client=client,
        config=VideoL1Config(localization_mode="grounded_single_pass"),
        window_provider=_OneWindowProvider(),
    )

    result = extractor.extract(video_path=video, video_id="video-a")

    assert result.nodes == ()
    assert result.rejected[0]["stage"] == "single_pass"


def test_grounded_single_pass_accepts_explicit_visible_endpoints(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    monkeypatch.setattr(video_l1, "_video_duration_s", lambda _: 2.0)
    monkeypatch.setattr(
        video_l1,
        "_sample_frame_data_uris",
        lambda *args, **kwargs: (
            ["data:image/jpeg;base64,AA==", "data:image/jpeg;base64,AA=="],
            [
                {"frame_index": 0, "time_s": 0.0},
                {"frame_index": 1, "time_s": 2.0},
            ],
        ),
    )
    client = _GroundedClient()
    original = client.perceive

    def with_endpoints(*args, **kwargs):
        payload = original(*args, **kwargs)
        event = payload["events"][0]
        event["evidence_frames"] = []
        event["visible_start_frame"] = 0
        event["visible_end_frame"] = 1
        return payload

    client.perceive = with_endpoints  # type: ignore[method-assign]
    extractor = QwenVideoL1Extractor(
        client=client,
        config=VideoL1Config(localization_mode="grounded_single_pass"),
        window_provider=_OneWindowProvider(),
    )

    result = extractor.extract(video_path=video, video_id="video-a")

    assert len(result.nodes) == 1
    assert result.nodes[0].metadata["localization"]["evidence_frames"] == [0, 1]

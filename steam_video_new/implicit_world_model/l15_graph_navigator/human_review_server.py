"""Serve the local, outcome-blind transition review application."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .transition_review import apply_transition_review, validate_transition_review_packet
from .visual_review import validate_visual_review_bundle


_STATIC_ROOT = Path(__file__).resolve().parent / "human_review"


def serve_transition_review(
    packet_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    visual_index_path: Path | None = None,
    visual_key_path: Path | None = None,
) -> None:
    packet = json.loads(packet_path.expanduser().resolve().read_text(encoding="utf-8"))
    errors = validate_transition_review_packet(packet)
    if errors or packet.get("annotation_status") != "unreviewed":
        raise ValueError(
            "human review server requires a valid unreviewed packet: "
            + "; ".join(errors[:8])
        )
    if (visual_index_path is None) != (visual_key_path is None):
        raise ValueError("--visual-index and --visual-key must be supplied together")
    visual_index: dict[str, Any] | None = None
    media_bindings: dict[str, dict[str, Any]] = {}
    if visual_index_path is not None and visual_key_path is not None:
        visual_index = json.loads(
            visual_index_path.expanduser().resolve().read_text(encoding="utf-8")
        )
        visual_key = json.loads(
            visual_key_path.expanduser().resolve().read_text(encoding="utf-8")
        )
        visual_errors = validate_visual_review_bundle(packet, visual_index, visual_key)
        if visual_errors:
            raise ValueError("invalid visual review bundle: " + "; ".join(visual_errors[:8]))
        media_bindings = {
            str(row["asset_id"]): row for row in visual_key.get("assets") or []
        }
        for binding in media_bindings.values():
            path = Path(str(binding["video_path"])).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"bound visual media is missing: {binding['asset_id']}")
    handler = _handler(packet, visual_index=visual_index, media_bindings=media_bindings)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Human review UI: http://{host}:{port}")
    print(f"Packet: {packet_path.expanduser().resolve()}")
    print(f"Visual evidence: {'enabled' if visual_index is not None else 'disabled'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _handler(
    packet: dict[str, Any],
    *,
    visual_index: dict[str, Any] | None = None,
    media_bindings: dict[str, dict[str, Any]] | None = None,
):
    bindings = media_bindings or {}

    class ReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/packet":
                self._json(HTTPStatus.OK, packet)
                return
            if path == "/api/visual":
                self._json(
                    HTTPStatus.OK,
                    visual_index
                    if visual_index is not None
                    else {
                        "schema_version": "steam-visual-transition-review/v0.1",
                        "packet_id": packet["packet_id"],
                        "enabled": False,
                        "items": [],
                    },
                )
                return
            if path.startswith("/api/media/"):
                asset_id = path.removeprefix("/api/media/")
                binding = bindings.get(asset_id)
                if binding is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._media(Path(str(binding["video_path"])).resolve())
                return
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            target = (_STATIC_ROOT / relative).resolve()
            if _STATIC_ROOT.resolve() not in target.parents and target != _STATIC_ROOT.resolve():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            payload = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/validate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                if length <= 0 or length > 4_000_000:
                    raise ValueError("invalid request size")
                review = json.loads(self.rfile.read(length).decode("utf-8"))
                if review.get("labels_source") != "independent_human":
                    raise ValueError("website export must use independent_human labels")
                locked = apply_transition_review(packet, review)
                self._json(
                    HTTPStatus.OK,
                    {
                        "valid": True,
                        "annotation_status": locked["annotation_status"],
                        "locked_sha256": locked["locked_sha256"],
                    },
                )
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"valid": False, "error": str(exc)})

        def _json(self, status: HTTPStatus, value: Any) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _media(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = path.stat().st_size
            try:
                byte_range = _parse_range_header(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            if byte_range is None:
                start, end = 0, size - 1
                status = HTTPStatus.OK
            else:
                start, end = byte_range
                status = HTTPStatus.PARTIAL_CONTENT
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "private, no-store")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            remaining = end - start + 1
            with path.open("rb") as handle:
                handle.seek(start)
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ReviewHandler


def _parse_range_header(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if size <= 0 or not value.startswith("bytes=") or "," in value:
        raise ValueError("unsupported byte range")
    spec = value[6:].strip()
    if "-" not in spec:
        raise ValueError("malformed byte range")
    start_text, end_text = spec.split("-", 1)
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError("invalid suffix range")
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError as exc:
        raise ValueError("malformed byte range") from exc
    if start < 0 or start >= size or end < start:
        raise ValueError("unsatisfiable byte range")
    return start, min(end, size - 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--visual-index", type=Path)
    parser.add_argument("--visual-key", type=Path)
    args = parser.parse_args(argv)
    serve_transition_review(
        args.packet,
        host=args.host,
        port=args.port,
        visual_index_path=args.visual_index,
        visual_key_path=args.visual_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

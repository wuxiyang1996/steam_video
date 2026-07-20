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


_STATIC_ROOT = Path(__file__).resolve().parent / "human_review"


def serve_transition_review(
    packet_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    packet = json.loads(packet_path.expanduser().resolve().read_text(encoding="utf-8"))
    errors = validate_transition_review_packet(packet)
    if errors or packet.get("annotation_status") != "unreviewed":
        raise ValueError(
            "human review server requires a valid unreviewed packet: "
            + "; ".join(errors[:8])
        )
    handler = _handler(packet)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Human review UI: http://{host}:{port}")
    print(f"Packet: {packet_path.expanduser().resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _handler(packet: dict[str, Any]):
    class ReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/packet":
                self._json(HTTPStatus.OK, packet)
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

        def log_message(self, format: str, *args: object) -> None:
            return

    return ReviewHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    serve_transition_review(args.packet, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

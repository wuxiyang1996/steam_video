"""Compatibility entrypoint for `dataset_clip_wrapper.verification.run_repair_protocol`."""

from .verification.run_repair_protocol import *  # noqa: F401,F403
from .verification.run_repair_protocol import main


if __name__ == "__main__":
    raise SystemExit(main())

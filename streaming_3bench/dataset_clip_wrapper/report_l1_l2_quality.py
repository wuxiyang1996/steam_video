"""Compatibility entrypoint for `dataset_clip_wrapper.verification.report_l1_l2_quality`."""

from .verification.report_l1_l2_quality import *  # noqa: F401,F403
from .verification.report_l1_l2_quality import main


if __name__ == "__main__":
    raise SystemExit(main())

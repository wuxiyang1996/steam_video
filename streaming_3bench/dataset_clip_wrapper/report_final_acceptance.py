"""Compatibility entrypoint for `dataset_clip_wrapper.verification.report_final_acceptance`."""

from .verification.report_final_acceptance import *  # noqa: F401,F403
from .verification.report_final_acceptance import main


if __name__ == "__main__":
    raise SystemExit(main())

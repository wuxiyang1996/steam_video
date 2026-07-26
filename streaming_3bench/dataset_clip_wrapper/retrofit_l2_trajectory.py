"""Compatibility entrypoint for `dataset_clip_wrapper.verification.retrofit_l2_trajectory`."""

from .verification.retrofit_l2_trajectory import *  # noqa: F401,F403
from .verification.retrofit_l2_trajectory import main


if __name__ == "__main__":
    raise SystemExit(main())

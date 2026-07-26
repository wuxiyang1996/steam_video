"""Compatibility entrypoint for `dataset_clip_wrapper.runners.run_staged_llm_pipeline`."""

from .runners.run_staged_llm_pipeline import *  # noqa: F401,F403
from .runners.run_staged_llm_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())

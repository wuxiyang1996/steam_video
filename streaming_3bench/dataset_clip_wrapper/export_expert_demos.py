"""Compatibility entrypoint for `dataset_clip_wrapper.expert_demos.export_expert_demos`."""

from .expert_demos.export_expert_demos import *  # noqa: F401,F403
from .expert_demos.export_expert_demos import main


if __name__ == "__main__":
    raise SystemExit(main())

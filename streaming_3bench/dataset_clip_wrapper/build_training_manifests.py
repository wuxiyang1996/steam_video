"""Compatibility entrypoint for `dataset_clip_wrapper.manifests.build_training_manifests`."""

from .manifests.build_training_manifests import *  # noqa: F401,F403
from .manifests.build_training_manifests import main


if __name__ == "__main__":
    raise SystemExit(main())

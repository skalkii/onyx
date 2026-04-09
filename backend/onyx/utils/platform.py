import logging
import os

logger = logging.getLogger(__name__)

_ONYX_DOCKER_ENV_STR = "ONYX_RUNNING_IN_DOCKER"
_DANSWER_DOCKER_ENV_STR = "DANSWER_RUNNING_IN_DOCKER"


def _resolve_container_flag() -> bool:
    onyx_val = os.getenv(_ONYX_DOCKER_ENV_STR)
    if onyx_val is not None:
        return onyx_val.lower() == "true"

    danswer_val = os.getenv(_DANSWER_DOCKER_ENV_STR)
    if danswer_val is not None:
        logger.warning(
            "%s is deprecated and will be ignored in a future release. "
            "Use %s instead.",
            _DANSWER_DOCKER_ENV_STR,
            _ONYX_DOCKER_ENV_STR,
        )
        return danswer_val.lower() == "true"

    return False


_IS_RUNNING_IN_CONTAINER: bool = _resolve_container_flag()


def is_running_in_container() -> bool:
    return _IS_RUNNING_IN_CONTAINER

from __future__ import annotations

import os
from typing import Any, Optional, Union


def env_var_is_set(env_var: str) -> bool:
    """Check if an environment variable is set.

    Args:
        env_var (str): The name of the environment variable.

    Returns:
        bool: True if the environment variable is set, False otherwise.
    """
    return env_var in os.environ and os.environ[env_var] not in (
        "",
        "0",
        "false",
        "False",
    )


_get_from_env_default_sentinel = object()


def get_from_dict_or_env(
    data: dict[str, Any],
    key: Union[str, list[str]],
    env_key: str,
    default: Optional[Union[str, object]] = _get_from_env_default_sentinel,
) -> str:
    """Get a value from a dictionary or an environment variable.

    Args:
        data: The dictionary to look up the key in.
        key: The key to look up in the dictionary. This can be a list of keys to try
            in order.
        env_key: The environment variable to look up if the key is not
            in the dictionary.
         default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to object() as a sentinel. https://peps.python.org/pep-0661/
    """
    if isinstance(key, (list, tuple)):
        for k in key:
            if k in data and data[k]:
                return data[k]

    if isinstance(key, str):
        if key in data and data[key]:
            return data[key]

    if isinstance(key, (list, tuple)):
        key_for_err = key[0]
    else:
        key_for_err = key
    if default is _get_from_env_default_sentinel:
        return get_from_env(key_for_err, env_key)
    else:
        return get_from_env(key_for_err, env_key, default=default)


def get_from_env(
    key: str,
    env_key: str,
    default: Optional[Union[str, object]] = _get_from_env_default_sentinel,
) -> str:
    """Get a value from a dictionary or an environment variable.
    Args:
        key: The key to look up in the dictionary.
        env_key: The environment variable to look up if the key is not
            in the dictionary.
        default: The default value to return if the key is not in the dictionary
            or the environment. Defaults to object() as a sentinel. https://peps.python.org/pep-0661/

    Returns:
        str: The value of the key.

    Raises:
        ValueError: If the key is not in the dictionary and no default value is
            provided or if the environment variable is not set.
    """
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not _get_from_env_default_sentinel:
        return str(default)
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )

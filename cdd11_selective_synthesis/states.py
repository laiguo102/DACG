"""Canonical CDD-11 state definitions.

Tuple order is only used for canonical names. Rendering always uses the
explicit operator order in :mod:`renderer`.
"""

from __future__ import annotations

from collections.abc import Iterable

DEGRADATIONS = ("low", "haze", "rain", "snow")
RENDER_ORDER = ("low", "rain", "snow", "haze")

VALID_STATE_TUPLES: tuple[tuple[str, ...], ...] = (
    (),
    ("low",),
    ("haze",),
    ("rain",),
    ("snow",),
    ("low", "haze"),
    ("low", "rain"),
    ("low", "snow"),
    ("haze", "rain"),
    ("haze", "snow"),
    ("low", "haze", "rain"),
    ("low", "haze", "snow"),
)

STATE_NAMES = {
    state: ("clean" if not state else "_".join(state))
    for state in VALID_STATE_TUPLES
}
NAME_TO_STATE = {name: state for state, name in STATE_NAMES.items()}
_ORDER = {name: index for index, name in enumerate(DEGRADATIONS)}


def canonical_state(degradations: Iterable[str]) -> tuple[str, ...]:
    values = tuple(degradations)
    unknown = sorted(set(values) - set(DEGRADATIONS))
    if unknown:
        raise ValueError(f"Unknown degradation(s): {unknown}")
    if len(set(values)) != len(values):
        raise ValueError("A state cannot contain a duplicate degradation")
    if "rain" in values and "snow" in values:
        raise ValueError("Rain and snow cannot be enabled in the same state")
    state = tuple(sorted(values, key=_ORDER.__getitem__))
    if state not in STATE_NAMES:
        raise ValueError(f"Unsupported CDD-11 state: {state}")
    return state


def state_name(state: Iterable[str]) -> str:
    return STATE_NAMES[canonical_state(state)]


def state_from_name(name: str) -> tuple[str, ...]:
    try:
        return NAME_TO_STATE[name]
    except KeyError as error:
        raise ValueError(f"Unknown CDD-11 state name: {name}") from error

"""Optional T27 analysis without changing V1.6 predictions."""

import numpy as np

try:
    from .pass_travel_t27 import detect_same_team_travel
except ImportError:
    from pass_travel_t27 import detect_same_team_travel


def analyze_t27(features, labels, graded=None, **kwargs):
    """Detect conservative same-team travel candidates."""
    n = int(features["n_frames"])
    labels = np.asarray(labels)

    if len(labels) != n:
        raise ValueError("labels length mismatch")

    if graded is None:
        raise ValueError(
            "graded mask is required; supply the actual match mask "
            "or an explicit synthetic mask for testing"
        )

    graded = np.asarray(graded, dtype=bool)

    if len(graded) != n:
        raise ValueError("graded length mismatch")

    dead_mask = labels == "D"

    candidates = detect_same_team_travel(
        features,
        graded=graded,
        dead_mask=dead_mask,
        **kwargs
    )

    accepted = []

    for candidate in candidates:
        expected = "H" if candidate.team == 0 else "A"

        segment = labels[
            candidate.start_frame:candidate.end_frame + 1
        ]

        if np.all(segment == expected):
            accepted.append(candidate)

    return accepted

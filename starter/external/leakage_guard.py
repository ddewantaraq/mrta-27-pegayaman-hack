"""Report shipped models whose training matches overlap an evaluation track.

Training on Track B/C/D matches is legal (they are other games) but makes
those tracks in-sample. New candidates may not add overlap beyond
LEGACY_ALLOWED, the Sample_Game_1 fit that v1.10-v2.3 models inherited.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
STARTER = os.path.dirname(HERE)
SKILLCORNER_SPLIT = os.path.join(HERE, 'skillcorner_split.json')

TRACK_B = ('Sample_Game_1', 'Sample_Game_2')
TRACK_C = ('J03WMX', 'J03WPY', 'J03WQQ', 'J03WR9')
LEGACY_ALLOWED = frozenset({'Sample_Game_1'})
LEGACY_MAX_VERSION = (2, 5)

_MATCH_RE = re.compile(r'(J03W[A-Z0-9]{2}|Sample_Game_\d|SC_\d+)')


def track_d():
    if not os.path.isfile(SKILLCORNER_SPLIT):
        return ()
    with open(SKILLCORNER_SPLIT) as handle:
        return tuple(json.load(handle)['track_d'])


def tracks():
    return {'B': TRACK_B, 'C': TRACK_C, 'D': track_d()}


def _metadata_path(path):
    root, ext = os.path.splitext(path)
    return root + '.json' if ext == '.npz' else path


def _ids(values):
    out = set()
    for value in values:
        out.update(_MATCH_RE.findall(str(value)))
    return out


def training_matches(path):
    """Match ids a model JSON (or the JSON beside an .npz) declares it trained on."""
    meta = _metadata_path(path)
    if not os.path.isfile(meta):
        raise FileNotFoundError('no training metadata for {}'.format(path))
    with open(meta) as handle:
        data = json.load(handle)
    found = set()
    for key in ('training_matches', 'training_games'):
        found |= _ids(data.get(key, ()))
    training = data.get('training')
    if isinstance(training, dict):
        found |= _ids(training.get('datasets', ()))
    if not found:
        raise ValueError('{} declares no training matches'.format(meta))
    return found


def model_paths(kwargs):
    """Every model file run_labeller loads for these kwargs, implicit ones included."""
    from residual_ownership_model import DEFAULT_RESIDUAL_MODEL_PATH
    from boundary_ensemble_model import DEFAULT_BOUNDARY_MODEL_PATH
    paths = []
    for key in ('boosted_dead_model', 'learned_dead_model', 'set_ownership_model',
                'residual_ownership_model', 'boundary_ensemble_model',
                'error_corrector_model', 'event_first_model'):
        if kwargs.get(key):
            paths.append(kwargs[key])
    if (kwargs.get('boundary_ensemble_model') or kwargs.get('error_corrector_model')
            or kwargs.get('event_first_model')):
        paths += [DEFAULT_RESIDUAL_MODEL_PATH, DEFAULT_BOUNDARY_MODEL_PATH]
    return sorted(set(os.path.abspath(p) for p in paths))


def overlap(kwargs):
    """{track: {match: [model basenames]}} for every in-sample evaluation match."""
    out = {}
    for path in model_paths(kwargs):
        trained = training_matches(path)
        for track, matches in tracks().items():
            for match in matches:
                if match in trained:
                    out.setdefault(track, {}).setdefault(match, []).append(
                        os.path.basename(path))
    return out


def violations(kwargs, version=None):
    """Overlaps a candidate is not allowed to have."""
    legacy = version is not None and tuple(version) <= LEGACY_MAX_VERSION
    bad = {}
    for track, matches in overlap(kwargs).items():
        for match, models in matches.items():
            if legacy or match in LEGACY_ALLOWED:
                continue
            bad.setdefault(track, {})[match] = models
    return bad


def report(kwargs, label, version=None):
    lines = []
    found = overlap(kwargs)
    for track in sorted(tracks()):
        for match in tracks()[track]:
            models = found.get(track, {}).get(match)
            if models:
                lines.append('  {:12s} Track {} {:14s} IN-SAMPLE ({})'.format(
                    label, track, match, ', '.join(sorted(set(models)))))
    if not lines:
        lines.append('  {:12s} clean on all evaluation tracks'.format(label))
    return lines, violations(kwargs, version)


def in_sample_matches(kwargs):
    return {m for matches in overlap(kwargs).values() for m in matches}

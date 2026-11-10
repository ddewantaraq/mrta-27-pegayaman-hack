"""Approved model pointer (LATEST) and auto submission naming.

LATEST names the approved-best CSV under evaluation/results/.
Auto predict bumps from that version (not max bare archive sm_v*).
Prefer bare sm_v*.csv; add _compliant only when bare already exists.
"""

from __future__ import print_function

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
LATEST_NAME = "LATEST"
FALLBACK_LATEST = "sm_v1_7_compliant.csv"
MAX_MINOR = 10

_RE_MAJOR_MINOR_COMPLIANT = re.compile(
    r"^sm_v(\d+)_(\d+)_compliant\.csv$", re.IGNORECASE)
_RE_MAJOR_ONLY_COMPLIANT = re.compile(
    r"^sm_v(\d+)_compliant\.csv$", re.IGNORECASE)
_RE_MAJOR_MINOR = re.compile(r"^sm_v(\d+)_(\d+)\.csv$", re.IGNORECASE)
_RE_MAJOR_ONLY = re.compile(r"^sm_v(\d+)\.csv$", re.IGNORECASE)


def parse_sm_version(filename):
    """Return (major, minor) or None. Accepts bare and *_compliant.csv."""
    name = os.path.basename(filename)
    m = _RE_MAJOR_MINOR_COMPLIANT.match(name)
    if m:
        major, minor = int(m.group(1)), int(m.group(2))
        if 0 <= minor <= MAX_MINOR:
            return major, minor
        return None
    m = _RE_MAJOR_ONLY_COMPLIANT.match(name)
    if m:
        return int(m.group(1)), 0
    m = _RE_MAJOR_MINOR.match(name)
    if m:
        major, minor = int(m.group(1)), int(m.group(2))
        if 0 <= minor <= MAX_MINOR:
            return major, minor
        return None
    m = _RE_MAJOR_ONLY.match(name)
    if m:
        return int(m.group(1)), 0
    return None


def format_sm_filename(major, minor, compliant=False):
    """1.0 -> sm_v1.csv; 1.2 -> sm_v1_2.csv; compliant adds _compliant before .csv."""
    if minor == 0:
        base = "sm_v{}".format(major)
    else:
        base = "sm_v{}_{}".format(major, minor)
    if compliant:
        return "{}_compliant.csv".format(base)
    return "{}.csv".format(base)


def format_version_label(major, minor):
    """Display id: sm_v1 / sm_v1_7."""
    if minor == 0:
        return "sm_v{}".format(major)
    return "sm_v{}_{}".format(major, minor)


def next_sm_version(major, minor):
    """Bump minor until 10, then major+1 with minor 0."""
    if minor < MAX_MINOR:
        return major, minor + 1
    return major + 1, 0


def latest_file_path(results_dir=None):
    """Path to the LATEST pointer file."""
    results_dir = results_dir or RESULTS_DIR
    return os.path.join(results_dir, LATEST_NAME)


def read_latest(results_dir=None):
    """Absolute path to the approved-best CSV named in LATEST (or fallback)."""
    results_dir = results_dir or RESULTS_DIR
    pointer = latest_file_path(results_dir)
    name = FALLBACK_LATEST
    if os.path.isfile(pointer):
        with open(pointer, "r") as handle:
            line = handle.readline().strip()
        if line:
            name = os.path.basename(line)
    path = os.path.join(results_dir, name)
    return path


def latest_version(results_dir=None):
    """(major, minor) from LATEST artifact name, or (1, 7) fallback parse."""
    path = read_latest(results_dir)
    parsed = parse_sm_version(path)
    if parsed is None:
        return 1, 7
    return parsed


def next_out_path(results_dir=None):
    """Next submission path after LATEST version.

    Prefer bare sm_v*.csv if missing; else *_compliant.csv; else bump version.
    Floor is LATEST only — not max over archive bare files.
    """
    results_dir = results_dir or RESULTS_DIR
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    major, minor = next_sm_version(*latest_version(results_dir))
    while True:
        bare = format_sm_filename(major, minor, compliant=False)
        compliant = format_sm_filename(major, minor, compliant=True)
        bare_path = os.path.join(results_dir, bare)
        compliant_path = os.path.join(results_dir, compliant)
        if not os.path.isfile(bare_path):
            return bare_path, major, minor
        if not os.path.isfile(compliant_path):
            return compliant_path, major, minor
        major, minor = next_sm_version(major, minor)

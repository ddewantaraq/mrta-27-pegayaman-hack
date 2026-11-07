"""Download Metrica Sports open sample games 1 and 2 into data_external/metrica/.

Public data: https://github.com/metrica-sports/sample-data
Acknowledge Metrica Sports if used publicly. Not BAR-ATH; no competition GT.

    .venv/bin/python starter/external/fetch_metrica.py
"""

from __future__ import print_function

import argparse
import os
import sys

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve  # noqa: F401  # py2 fallback unused

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(ROOT, "data_external", "metrica")

BASE = ("https://raw.githubusercontent.com/metrica-sports/sample-data/"
        "master/data")

GAMES = {
    "Sample_Game_1": {
        "events": "Sample_Game_1_RawEventsData.csv",
        "home": "Sample_Game_1_RawTrackingData_Home_Team.csv",
        "away": "Sample_Game_1_RawTrackingData_Away_Team.csv",
    },
    "Sample_Game_2": {
        "events": "Sample_Game_2_RawEventsData.csv",
        "home": "Sample_Game_2_RawTrackingData_Home_Team.csv",
        "away": "Sample_Game_2_RawTrackingData_Away_Team.csv",
    },
}


def game_paths(game_name, root=DEFAULT_OUT):
    """Absolute paths for one sample game's CSVs."""
    files = GAMES[game_name]
    folder = os.path.join(root, game_name)
    return {
        "events": os.path.join(folder, files["events"]),
        "home": os.path.join(folder, files["home"]),
        "away": os.path.join(folder, files["away"]),
        "folder": folder,
    }


def fetch_game(game_name, root=DEFAULT_OUT, force=False):
    """Download one game's three CSVs if missing (or always if force)."""
    if game_name not in GAMES:
        raise KeyError("unknown game {!r}; have {}".format(
            game_name, sorted(GAMES)))
    paths = game_paths(game_name, root)
    if not os.path.isdir(paths["folder"]):
        os.makedirs(paths["folder"])
    files = GAMES[game_name]
    for key, filename in files.items():
        dest = paths[key]
        if os.path.isfile(dest) and not force:
            print("have {}".format(dest))
            continue
        url = "{}/{}/{}".format(BASE, game_name, filename)
        print("fetch {} -> {}".format(url, dest))
        urlretrieve(url, dest)
    return paths


def fetch_all(root=DEFAULT_OUT, force=False):
    """Download Sample_Game_1 and Sample_Game_2."""
    out = {}
    for name in sorted(GAMES):
        out[name] = fetch_game(name, root=root, force=force)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="root folder (default data_external/metrica)")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if files exist")
    args = parser.parse_args()
    fetch_all(root=args.out, force=args.force)
    print("done -> {}".format(args.out))


if __name__ == "__main__":
    main()

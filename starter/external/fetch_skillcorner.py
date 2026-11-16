"""Download SkillCorner Open Data (A-League 2024/25) into data_external/skillcorner/.

Public data, MIT licence, no registration:
https://github.com/SkillCorner/opendata (joint SkillCorner / PySport release).
Pinned to one commit; every file is checked against skillcorner_manifest.json
(hashes only, no data). Not BAR-ATH; no competition GT.

    .venv/bin/python starter/external/fetch_skillcorner.py            # all 20
    .venv/bin/python starter/external/fetch_skillcorner.py --matches 2017461
"""
import argparse
import hashlib
import json
import os
import shutil
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(ROOT, 'data_external', 'skillcorner')
MANIFEST = os.path.join(HERE, 'skillcorner_manifest.json')

REPO = 'SkillCorner/opendata'
COMMIT = '4340d274572876239c154c90bc507a9b3250a656'
RAW = 'https://raw.githubusercontent.com/{}/{}/data'.format(REPO, COMMIT)
MEDIA = 'https://media.githubusercontent.com/media/{}/{}/data'.format(REPO, COMMIT)

MATCH_IDS = (
    '1874553', '1886347', '1899585', '1925299', '1927964', '1953632', '1959846',
    '1986691', '1996435', '1996436', '2006229', '2006363', '2007448', '2007721',
    '2010085', '2011166', '2013725', '2015213', '2016236', '2017461',
)
SUFFIXES = ('_match.json', '_tracking_extrapolated.jsonl',
            '_dynamic_events.csv', '_phases_of_play.csv')
LFS_HEADER = b'version https://git-lfs.github.com/spec/v1'


def match_paths(match_id, root=DEFAULT_OUT):
    folder = os.path.join(root, str(match_id))
    return {s: os.path.join(folder, str(match_id) + s) for s in SUFFIXES}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _download(url, dest):
    tmp = dest + '.part'
    with urllib.request.urlopen(url) as response, open(tmp, 'wb') as out:
        shutil.copyfileobj(response, out)
    os.replace(tmp, dest)


def fetch_file(match_id, suffix, dest):
    rel = 'matches/{0}/{0}{1}'.format(match_id, suffix)
    _download('{}/{}'.format(RAW, rel), dest)
    with open(dest, 'rb') as handle:
        if handle.read(len(LFS_HEADER)) == LFS_HEADER:
            _download('{}/{}'.format(MEDIA, rel), dest)


def load_manifest():
    if not os.path.isfile(MANIFEST):
        return dict(repo=REPO, commit=COMMIT, licence='MIT', files={})
    with open(MANIFEST) as handle:
        return json.load(handle)


def fetch(matches=MATCH_IDS, root=DEFAULT_OUT, write_manifest=False, force=False):
    manifest = load_manifest()
    if manifest['commit'] != COMMIT:
        raise ValueError('manifest pinned to a different commit')
    for match_id in matches:
        for suffix, dest in match_paths(match_id, root).items():
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            key = os.path.basename(dest)
            if force or not os.path.isfile(dest):
                print('fetch', key, flush=True)
                fetch_file(match_id, suffix, dest)
            digest = sha256(dest)
            want = manifest['files'].get(key)
            if want is None:
                if not write_manifest:
                    raise ValueError('{} not in manifest; rerun with --write-manifest'.format(key))
                manifest['files'][key] = digest
            elif want != digest:
                raise ValueError('sha256 mismatch for {}'.format(key))
    if write_manifest:
        with open(MANIFEST, 'w') as handle:
            json.dump(manifest, handle, indent=1, sort_keys=True)
            handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--matches', nargs='+', default=list(MATCH_IDS))
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--write-manifest', action='store_true',
                        help='record hashes for files not yet in the manifest')
    args = parser.parse_args()
    fetch(args.matches, args.out, args.write_manifest, args.force)
    print('done ->', args.out)


if __name__ == '__main__':
    main()

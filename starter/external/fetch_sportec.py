"""Fetch CC BY 4.0 Sportec Open DFL matches, pinned figshare article v1.

Bassek et al. (2025), https://doi.org/10.1038/s41597-025-04505-y
Data: https://doi.org/10.6084/m9.figshare.28196177.v1
Files stay in ignored data_external/. Downloads are size and MD5 verified.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'data_external' / 'sportec'
API = 'https://api.figshare.com/v2/articles/28196177/versions/1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', nargs='+', default=['J03WOH'],
                        help='match IDs, or all (about 2.6 GB)')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    with urlopen(API, timeout=60) as response:
        manifest = json.load(response)
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for item in manifest['files']:
        name = item['name']
        if Path(name).name != name:
            raise ValueError('Unexpected filename')
        if 'all' not in args.matches and not any(m in name for m in args.matches):
            continue
        path = DEST / name
        if path.exists() and path.stat().st_size == item['size']:
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'md5').hexdigest() if hasattr(hashlib, 'file_digest') else None
                if digest is None:
                    h = hashlib.md5()
                    for block in iter(lambda: handle.read(1024*1024), b''):
                        h.update(block)
                    digest = h.hexdigest()
            if digest == item['computed_md5']:
                print('verified', name, flush=True)
                continue
        partial = path.with_suffix('.xml.partial')
        print('download', name, item['size'], flush=True)
        h = hashlib.md5()
        with urlopen(item['download_url'], timeout=120) as response, partial.open('wb') as handle:
            for block in iter(lambda: response.read(1024*1024), b''):
                handle.write(block)
                h.update(block)
        if partial.stat().st_size != item['size'] or h.hexdigest() != item['computed_md5']:
            raise ValueError('Download verification failed: ' + name)
        partial.replace(path)
        print('verified', name, flush=True)


if __name__ == '__main__':
    main()

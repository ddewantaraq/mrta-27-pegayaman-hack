"""Fetch unused Metrica Sample Game 3 for a reserved external test.

Source: https://github.com/metrica-sports/sample-data (acknowledge Metrica Sports).
This is not the challenge match. Keep annotations out of model development
until a candidate and its decision rule have been frozen.
"""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'data_external'/'metrica'/'Sample_Game_3'
BASE = 'https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in ('Sample_Game_3_metadata.xml', 'Sample_Game_3_tracking.txt', 'Sample_Game_3_events.json'):
        path = DEST/name
        if not path.exists():
            print('fetch', name, flush=True)
            partial = path.with_suffix(path.suffix+'.partial')
            with urlopen(BASE+name, timeout=120) as response, partial.open('wb') as handle:
                for chunk in iter(lambda: response.read(1024*1024), b''):
                    handle.write(chunk)
            partial.replace(path)
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                h.update(chunk)
        manifest.append(dict(name=name, bytes=path.stat().st_size, sha256=h.hexdigest(), url=BASE+name))
        print('have', name, path.stat().st_size, flush=True)
    (DEST/'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

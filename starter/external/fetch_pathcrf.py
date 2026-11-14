"""Inspect/download official PathCRF code and a public pretrained checkpoint.

No event/tracking datasets or annotation files are downloaded by this helper.
Everything stays under ignored data_external/pathcrf, with SHA256 provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'hyunsungkim-ds/pathcrf'
ROOT = Path(__file__).resolve().parents[2]/'data_external'/'pathcrf'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent':'possession-research'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    tree = json.loads(fetch('https://api.github.com/repos/'+REPO+'/git/trees/main?recursive=1'))
    commit = tree['sha']
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'tree_manifest.json').write_text(json.dumps(tree, indent=2))
    selected = []
    for item in tree['tree']:
        path = item['path']
        if item['type'] != 'blob':
            continue
        if (path.endswith('.py') or path in ('README.md','LICENSE','requirements.txt','tutorial.ipynb')
                or path.startswith('saved/140/')):
            print(path, item.get('size'), flush=True)
            selected.append(path)
    if not args.download:
        return
    manifest = {'repository': REPO, 'commit': commit, 'files': []}
    for relative in selected:
        target = (ROOT/relative).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise ValueError('Unsafe repository path')
        url = 'https://raw.githubusercontent.com/'+REPO+'/'+commit+'/'+relative
        content = fetch(url)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
        manifest['files'].append(dict(path=relative,bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),url=url))
        print('downloaded',relative,len(content),flush=True)
    (ROOT/'download_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()

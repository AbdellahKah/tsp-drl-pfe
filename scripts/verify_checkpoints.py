"""Compare the three selected actor checkpoint binaries with the archived CSV.

This checks the exact bytes and filenames, not experimental accuracy.
No checkpoint is unpickled or executed by this script.
"""

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'artifacts' / 'aggregation' / 'multiseed_selected_checkpoints.csv'


def verify():
    ok = True
    with MANIFEST.open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        seed = int(row['training_seed'])
        name = row['selected_checkpoint_file']
        expected = row['selected_checkpoint_sha256']
        path = ROOT / 'models' / f'seed_{seed}' / name
        if not path.is_file():
            print(f'MISSING {seed}: {path.relative_to(ROOT)}')
            ok = False
            continue
        sha = hashlib.sha256()
        with path.open('rb') as data:
            for chunk in iter(lambda: data.read(1024 * 1024), b''):
                sha.update(chunk)
        digest = sha.hexdigest()
        if digest != expected:
            print(f'MISMATCH {seed}: expected {expected}; observed {digest}')
            ok = False
        else:
            print(f'OK {seed}: {name} ({digest})')
    return ok and len(records) == 3


if __name__ == '__main__':
    raise SystemExit(0 if verify() else 1)

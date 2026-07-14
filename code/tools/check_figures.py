#!/usr/bin/env python3
"""
check_figures.py — find missing / stale / duplicated figures referenced by a LaTeX file.

A manuscript's .tex references figures by path. During a revision those files drift:
the path may point at something that no longer exists locally (it only compiles because a
copy lives in Overleaf), or a newer/better version was regenerated into a different folder
and never copied over what the paper actually compiles.

This script parses every \\includegraphics, reports which referenced files are MISSING, and
for each figure lists every copy of that basename in the repo with modification time and
md5 — so you can immediately see when the file the paper uses is not the newest one.

Usage:
    python check_figures.py path/to/main.tex [--root REPO_ROOT]

--root defaults to the current directory. \\includegraphics paths are resolved relative to
--root first, then relative to the .tex file's own directory.
"""
import argparse
import hashlib
import os
import re
import sys
from datetime import datetime

INCLUDE_RE = re.compile(r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}')
# common latex image extensions to try when the path has none
EXTS = ['', '.pdf', '.png', '.jpg', '.jpeg', '.eps']


def md5(path):
    try:
        return hashlib.md5(open(path, 'rb').read()).hexdigest()[:8]
    except OSError:
        return '????????'


def mtime(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
    except OSError:
        return '????-??-?? ??:??'


def resolve(path, roots):
    """Return the first existing file for `path` (trying extensions) under any root."""
    for root in roots:
        for ext in EXTS:
            cand = os.path.normpath(os.path.join(root, path + ext))
            if os.path.isfile(cand):
                return cand
    return None


def index_repo(root):
    """basename -> [full paths] for every file under root (skip .git)."""
    idx = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if os.sep + '.git' in dirpath:
            continue
        for fn in filenames:
            idx.setdefault(fn, []).append(os.path.join(dirpath, fn))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tex')
    ap.add_argument('--root', default='.')
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    texdir = os.path.dirname(os.path.abspath(args.tex))
    roots = [root, texdir]

    refs = []
    for lineno, raw in enumerate(open(args.tex, encoding='utf-8', errors='replace'), 1):
        # ignore anything after an unescaped % so commented-out includes are skipped
        code = re.split(r'(?<!\\)%', raw, 1)[0]
        for m in INCLUDE_RE.finditer(code):
            refs.append((lineno, m.group(1).strip()))

    if not refs:
        print('No \\includegraphics found.')
        return

    print(f'{len(refs)} figure reference(s) in {args.tex}\n')
    repo = index_repo(root)

    n_missing = 0
    n_stale = 0
    for line, path in refs:
        resolved = resolve(path, roots)
        base = os.path.basename(path)
        # gather all copies of this basename (try extensionless too)
        copies = []
        for name, paths in repo.items():
            stem = os.path.splitext(name)[0]
            if name == base or stem == os.path.splitext(base)[0]:
                copies.extend(paths)
        copies = sorted(set(copies), key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                        reverse=True)

        if resolved is None:
            n_missing += 1
            print(f'[MISSING] line {line}: {path}')
            if copies:
                print('    but copies of this name exist elsewhere:')
                for c in copies:
                    print(f'      {mtime(c)}  md5:{md5(c)}  {os.path.relpath(c, root)}')
            else:
                print('    and no file of this basename exists anywhere under root.')
            print()
            continue

        # resolved exists — is it the newest copy?
        newest = copies[0] if copies else resolved
        is_newest = os.path.abspath(newest) == os.path.abspath(resolved)
        tag = 'OK' if is_newest or len(copies) <= 1 else 'STALE?'
        if tag == 'STALE?':
            n_stale += 1
        print(f'[{tag}] line {line}: {path}')
        print(f'    used:   {mtime(resolved)}  md5:{md5(resolved)}  {os.path.relpath(resolved, root)}')
        for c in copies:
            if os.path.abspath(c) != os.path.abspath(resolved):
                newer = os.path.getmtime(c) > os.path.getmtime(resolved)
                mark = '  <-- NEWER' if newer else ''
                print(f'    other:  {mtime(c)}  md5:{md5(c)}  {os.path.relpath(c, root)}{mark}')
        print()

    print('-' * 60)
    print(f'{n_missing} missing, {n_stale} possibly-stale (a newer copy exists).')
    if n_stale:
        print('NOTE: newer mtime does not mean better — open the images and compare '
              'content against the caption/claims before swapping.')
    sys.exit(1 if n_missing else 0)


if __name__ == '__main__':
    main()

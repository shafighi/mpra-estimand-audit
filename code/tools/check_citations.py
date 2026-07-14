#!/usr/bin/env python3
"""
check_citations.py — find undefined \\cite keys (the usual "Citation undefined" warnings).

BibTeX citation keys are case-insensitive, so this matches case-insensitively to avoid
false alarms. Reports keys cited in the .tex but absent from the .bib (these are real
compile warnings and often the mysterious "N warnings" a revision suddenly has after new
references were added), and optionally keys defined but never cited.

Usage:
    python check_citations.py main.tex --bib references.bib [more.bib ...] [--unused]
"""
import argparse
import re
import sys

CITE_RE = re.compile(r'\\(?:cite|citep|citet|citeauthor|citeyear|citealt|citealp|Citep|Citet)'
                     r'\s*(?:\[[^\]]*\])?\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}')
ENTRY_RE = re.compile(r'@\s*\w+\s*\{\s*([^,\s]+)\s*,')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tex')
    ap.add_argument('--bib', nargs='+', required=True)
    ap.add_argument('--unused', action='store_true',
                    help='also list bib entries that are never cited')
    args = ap.parse_args()

    tex = open(args.tex, encoding='utf-8', errors='replace').read()
    cited = []
    for m in CITE_RE.finditer(tex):
        for key in m.group(1).split(','):
            key = key.strip()
            if key:
                cited.append(key)
    cited_set = {k.lower(): k for k in cited}  # preserve one original spelling

    defined = set()
    for b in args.bib:
        btext = open(b, encoding='utf-8', errors='replace').read()
        for m in ENTRY_RE.finditer(btext):
            defined.add(m.group(1).strip().lower())

    undefined = sorted({orig for low, orig in cited_set.items() if low not in defined})

    print(f'{len(cited_set)} distinct citation keys used, {len(defined)} bib entries found.\n')
    if undefined:
        print(f'UNDEFINED — cited but not in any .bib ({len(undefined)}):')
        for k in undefined:
            print(f'  \\cite{{{k}}}')
        print('\nThese produce "Citation undefined" warnings. Add the entries or fix the keys.')
    else:
        print('All cited keys are defined. No undefined-citation warnings expected.')

    if args.unused:
        used_low = set(cited_set)
        unused = sorted(defined - used_low)
        print(f'\nUnused bib entries ({len(unused)}) — harmless, but you can prune:')
        for k in unused:
            print(f'  {k}')

    sys.exit(1 if undefined else 0)


if __name__ == '__main__':
    main()

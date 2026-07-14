#!/usr/bin/env python3
"""
lint_tex.py — flag the specific LaTeX pitfalls that bite manuscript revisions.

These are not general style nits; each one caused a real, hard-to-diagnose problem in a
journal revision (blank pages, undefined font shapes, mysterious warnings, em-dashes an
editor asked to remove). Line numbers and a suggested fix are printed for each hit.

Usage:
    python lint_tex.py main.tex
"""
import re
import sys


def strip_comment(line):
    """Return the non-comment part of a line (respects \\%)."""
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == '\\' and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if c == '%':
            break
        out.append(c)
        i += 1
    return ''.join(out)


def main():
    if len(sys.argv) != 2:
        print('usage: python lint_tex.py main.tex')
        sys.exit(2)
    lines = open(sys.argv[1], encoding='utf-8', errors='replace').read().split('\n')

    findings = []  # (line_no, tag, message)

    # track tabular depth so we can tell "--- in prose" from "--- as a table N/A marker"
    depth = 0
    seen_pkgs = {}

    for n, raw in enumerate(lines, 1):
        code = strip_comment(raw)

        if re.search(r'\\begin\{(tabular|tabularx|array|longtable)', code):
            depth += 1
        if re.search(r'\\end\{(tabular|tabularx|array|longtable)', code):
            depth = max(0, depth - 1)

        # 1. em-dash in prose (--- ). Inside a tabular it's usually a deliberate "N/A".
        if '---' in code and depth == 0:
            findings.append((n, 'em-dash',
                             'Literal "---" (em-dash) in prose. If a journal asked to '
                             'remove em-dashes, replace with a comma or " -- ". '
                             '(Inside tables "---" as an N/A marker is usually fine.)'))

        # 2. starred float with [H] -> undefined / blank page
        if re.search(r'\\begin\{(table|figure)\*\}\s*\[[^\]]*H[^\]]*\]', code):
            findings.append((n, 'float-H',
                             '\\begin{table*|figure*}[H]: the float package "H" specifier is '
                             'NOT defined for double-column floats and can eject a blank page. '
                             'Use [t!] (and make sibling floats the same type so they stay in order).'))

        # 3. \textsc{...} nested inside \textbf{...} -> T1/lmr/bx/sc undefined
        #    cheap heuristic: same line contains \textbf{ ... \textsc{
        if re.search(r'\\textbf\{[^}]*\\textsc\{', code):
            findings.append((n, 'bold-smallcaps',
                             '\\textsc{} inside \\textbf{}: bold small-caps (T1/lmr/bx/sc) is '
                             'undefined in Latin Modern -> "Font shape undefined" warning. '
                             'Use plain text instead of \\textsc inside a bold title.'))

        # 4. malformed \url : {\url http...} instead of \url{...}
        if re.search(r'\{\s*\\url\s+\S', code):
            findings.append((n, 'url',
                             'Malformed URL "{\\url http...}". Use \\url{http...} with braces.'))

        # 5. duplicate \usepackage of the same package
        m = re.search(r'\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}', code)
        if m:
            for pkg in m.group(1).split(','):
                pkg = pkg.strip()
                if pkg in seen_pkgs:
                    findings.append((n, 'dup-package',
                                     f'Package "{pkg}" already loaded on line {seen_pkgs[pkg]}. '
                                     f'Remove the duplicate.'))
                else:
                    seen_pkgs[pkg] = n

    # 6. todonotes loaded but \todo/\TODO never used -> pointless marginparwidth warning
    full = '\n'.join(lines)
    if 'todonotes' in full and not re.search(r'\\todo\b|\\TODO\b|\\listoftodos', full):
        # find the line
        for n, raw in enumerate(lines, 1):
            if 'todonotes' in strip_comment(raw):
                findings.append((n, 'unused-todonotes',
                                 'todonotes is loaded but never used -> emits a marginparwidth '
                                 'warning. Remove the package (and its \\newcommand) or set '
                                 '\\setlength{\\marginparwidth}{2cm} before loading it.'))
                break

    findings.sort()
    if not findings:
        print('No known pitfalls found.')
        return
    print(f'{len(findings)} potential issue(s):\n')
    for n, tag, msg in findings:
        print(f'  line {n:>5}  [{tag}]  {msg}')
    sys.exit(1)


if __name__ == '__main__':
    main()

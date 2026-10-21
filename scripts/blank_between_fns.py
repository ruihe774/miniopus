#!/usr/bin/env python3
"""Ensure exactly one blank line between consecutive Rust functions.

Leading attributes/doc comments/comments belong to the function that follows them,
so the blank line goes above them. Usage: blank_between_fns.py [--check] FILE...
"""
import re
import sys


def mask(src):
    """Replace comments and string/char literal contents with spaces (newlines kept)."""
    out = list(src)
    i, n = 0, len(src)

    def blank(a, b):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth += 1
                    j += 2
                elif src.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c == "r" and re.match(r'r#*"', src[i:i + 40]) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            m = re.match(r'r(#*)"', src[i:])
            end = '"' + m.group(1)
            j = src.find(end, i + m.end())
            j = n if j < 0 else j + len(end)
            blank(i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
        elif c == "'":
            m = re.match(r"'(\\.[^']*|[^\\'])'", src[i:])
            if m:
                blank(i, i + m.end())
                i += m.end()
            else:
                i += 1  # lifetime
        else:
            i += 1
    return "".join(out)


FN = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:(?:const|unsafe|async|default)[ \t]+)*"
    r"(?:extern(?:[ \t]+\"[^\"]*\"|[ \t]+)?[ \t]*)?fn[ \t]+[A-Za-z_]",
    re.M,
)


def process(src):
    m = mask(src)
    lines = src.split("\n")
    starts = [0]
    for ln in m.split("\n"):
        starts.append(starts[-1] + len(ln) + 1)

    def line_of(off):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return lo

    fns = []  # (first_line incl. attrs/comments, end_line)
    for mt in FN.finditer(m):
        first = line_of(mt.start())
        # find body start or `;` at paren/bracket depth 0
        i, depth, end = mt.end(), 0, None
        while i < len(m):
            ch = m[i]
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif depth == 0 and ch == ";":
                end = i
                break
            elif depth == 0 and ch == "{":
                d = 0
                while i < len(m):
                    if m[i] == "{":
                        d += 1
                    elif m[i] == "}":
                        d -= 1
                        if d == 0:
                            break
                    i += 1
                end = i
                break
            i += 1
        if end is None:
            continue
        while first > 0 and re.match(r"\s*(#\[|#!\[|//|/\*|\*)", lines[first - 1]):
            first -= 1
        fns.append((first, line_of(end)))

    ends = {e for _, e in fns}
    insert = [f for f, _ in fns if f > 0 and lines[f - 1].strip() and (f - 1) in ends]
    for f in sorted(set(insert), reverse=True):
        lines.insert(f, "")
    return "\n".join(lines)


def main():
    check = "--check" in sys.argv
    bad = 0
    for p in (a for a in sys.argv[1:] if a != "--check"):
        src = open(p).read()
        new = process(src)
        if new != src:
            bad += 1
            if check:
                print(p)
            else:
                open(p, "w").write(new)
    sys.exit(1 if check and bad else 0)


main()

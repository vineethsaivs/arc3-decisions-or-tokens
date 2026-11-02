#!/usr/bin/env python3
"""Style checks for the v3 PDF body (FRAMING_v3 section 6), run from the repo root:

    python3 -I -B paper/v3/analysis/v3_style_check.py [--verbose]

Reads the section files that main.tex inputs before \\appendix. Prose is the text outside figure, table and equation
environments, outside boxes fenced by '% count:off' / '% count:on', with citations, references, labels and comments
removed. A number is a run of digits (with , . or a trailing %) that is not part of a name or an id (ARC-AGI-3,
Qwen3.8, V3-B5, k_3); a date counts once. Prints: sentences over 3 numbers, paragraphs over 5, words per section
(prose, captions, table cells, boxes), dashes anywhere in the tex tree, record codes and banned words in the body,
and the count of the required phrases. Exit 1 if any hard rule fails. Stdlib only.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.normpath(os.path.join(HERE, "..", "tex"))
MAX_SENT, MAX_PARA = 3, 5

CODES = re.compile(r"\b(l1a1\w*|y8\w*|r3p|r3kp|r3fbp|hm32\w*|p448\w*|s14\w*|lora[RH]\d|rq1020\S*|W[ABC]\b)")
BANNED = re.compile(r"\b(novel|significant(ly)?|replica(s)?)\b", re.I)


def body_files():
    main = open(os.path.join(TEX, "main.tex"), encoding="utf8").read()
    main = main[main.index("\\begin{document}"):main.index("\\appendix")]   # preamble inputs (placeholders) are not body
    return [os.path.join(TEX, m + ".tex") for m in re.findall(r"\\input\{(sections/[^}]+)\}", main)]


def strip_comments(t):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in t.splitlines())


def split_parts(raw):
    """Return (prose, captions, cells, boxes) text of one section file."""
    boxes = " ".join(re.findall(r"% count:off(.*?)% count:on", raw, re.S))
    raw = re.sub(r"% count:off.*?% count:on", "\n\n", raw, flags=re.S)
    t = strip_comments(raw)
    while "\\notebox{" in t:                      # brace-matched box bodies
        a = t.index("\\notebox{") + len("\\notebox{")
        depth, b = 1, a
        while depth:
            depth += {"{": 1, "}": -1}.get(t[b], 0)
            b += 1
        boxes += " " + t[a:b - 1]
        t = t[:a - len("\\notebox{")] + "\n\n" + t[b:]
    caps, cells = [], []
    for env in ("figure", "table"):
        for m in re.finditer(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (env, env), t, re.S):
            body = m.group(1)
            caps += re.findall(r"\\caption\{(.*?)\}\\label", body, re.S)
            tab = re.search(r"\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}", body, re.S)
            if tab:
                cells.append(tab.group(1))
        t = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), "\n\n", t, flags=re.S)
    t = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " EQUATION. ", t, flags=re.S)
    return t, " ".join(caps), " ".join(cells), boxes


def clean(t):
    t = re.sub(r"\\(section|subsection)\*?\{[^}]*\}(\\label\{[^}]*\})?", "\n\n", t)
    t = re.sub(r"\\paragraph\{([^}]*)\}", r"\n\n\1 ", t)
    t = re.sub(r"\\item\b", "\n\n", t)
    t = re.sub(r"\\begin\{\w+\}|\\end\{\w+\}|\\setlength\{[^}]*\}\{[^}]*\}", " ", t)
    t = re.sub(r"(Sec|Eq|Figure|Table|Appendix)s?\.?~\\(eq)?ref\{[^}]*\}", r"\1", t)
    t = re.sub(r"\\(eq)?ref\{[^}]*\}|\\label\{[^}]*\}", "", t)
    t = re.sub(r"\\cite[pt]?\*?\{[^}]*\}", "CITE", t)
    t = re.sub(r"\\(textbf|emph|texttt|textit|subid|pubonly)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\rankof\{([^}]*)\}\{([^}]*)\}", r"rank \1 of \2", t)
    t = t.replace("~", " ").replace("\\%", "%").replace("\\$", "$").replace("\\,", " ").replace("``", '"').replace("''", '"')
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("{", "").replace("}", "").replace("$", "")
    return t


def numbers(s):
    s = s.replace("RTX PRO 6000", "RTXPRO").replace("Milestone 2", "Milestone")
    s = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "DATEX9", s)
    s = re.sub(r"\b\d{2}:\d{2}Z", "TIMEX9", s)
    out = []
    s = re.sub(r"(^|\s)\(\d\)\s", " ", s)          # list markers (1), (2)
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?%?", s):
        a = m.start()
        prev = s[a - 1] if a else " "
        if prev.isalnum() or prev in "_.":
            continue
        if prev == "-" and a >= 2 and s[a - 2].isalpha():
            continue
        out.append(m.group(0).rstrip(",."))
    out += ["DATE"] * s.count("DATEX9") + ["TIME"] * s.count("TIMEX9")
    return out


def sentences(p):
    return [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", p.strip()) if x.strip()]


def words(t):
    return len(re.findall(r"[A-Za-z0-9][\w,.%'+-]*", clean(t)))


def main():
    verbose = "--verbose" in sys.argv
    fail = 0
    print("Section words (prose / captions / table cells / boxes) and number-density checks")
    tot = [0, 0, 0, 0]
    over_s, over_p, n_par = [], [], 0
    phrases = {"we do not know why": 0, "coincides with": 0}
    for f in body_files():
        raw = open(f, encoding="utf8").read()
        prose, caps, cells, boxes = split_parts(raw)
        prose_c = clean(prose)
        w = [words(prose), words(caps), words(cells), words(boxes)]
        tot = [a + b for a, b in zip(tot, w)]
        name = os.path.basename(f)
        print("  %-18s %5d %5d %5d %5d" % (name, *w))
        for k in phrases:
            phrases[k] += len(re.findall(k, (prose_c + clean(boxes)).lower()))
        for para in re.split(r"\n\s*\n", prose_c):
            para = " ".join(para.split())
            if not para or para in ("Abstract",):
                continue
            n_par += 1
            nums = numbers(para)
            if len(nums) > MAX_PARA:
                over_p.append((name, len(nums), para[:90]))
            for s in sentences(para):
                ns = numbers(s)
                if len(ns) > MAX_SENT:
                    over_s.append((name, len(ns), ns, s[:110]))
            if verbose:
                print("    [%d] %s" % (len(nums), para[:70]))
        for m in CODES.finditer(clean(prose) + " " + clean(caps) + " " + clean(cells)):
            print("  RECORD CODE in %s: %s" % (name, m.group(0)))
            fail = 1
        for m in BANNED.finditer(prose_c + " " + clean(caps)):
            print("  WORD in %s: %s" % (name, m.group(0)))
    print("  %-18s %5d %5d %5d %5d   (all: %d)" % ("total", *tot, sum(tot)))
    print("Prose paragraphs: %d; over %d numbers: %d; sentences over %d numbers: %d"
          % (n_par, MAX_PARA, len(over_p), MAX_SENT, len(over_s)))
    for x in over_p:
        print("  PARA %s (%d): %s" % x)
    for name, n, ns, s in over_s:
        print("  SENT %s (%d %s): %s" % (name, n, ns, s))
    if over_p or over_s:
        fail = 1
    print("Required phrases:", phrases)
    if phrases["we do not know why"] != 1:
        fail = 1
    dash = []
    for f in sorted(glob.glob(os.path.join(TEX, "**", "*.tex"), recursive=True)) + [os.path.join(TEX, "refs.bib")]:
        if "/v2ref/" in f:
            continue
        for n, line in enumerate(open(f, encoding="utf8"), 1):
            code = re.sub(r"\\url\{[^}]*\}|url = \{[^}]*\}", "", line)
            if "\u2013" in line or "\u2014" in line or re.search(r"(?<![-{])--(?!-)", strip_comments(code)):
                dash.append("%s:%d" % (os.path.relpath(f, TEX), n))
    print("Dashes (en, em or LaTeX --):", dash or "none")
    if dash:
        fail = 1
    return fail


if __name__ == "__main__":
    sys.exit(main())

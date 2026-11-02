#!/usr/bin/env python3
"""Appendix B: build paper/v2/tex/sections/claims_table.tex from section 5 of framing/FRAMING.md.

    python3 -I -B paper/v2/analysis/build_claims_appendix.py            # write the table
    python3 -I -B paper/v2/analysis/build_claims_appendix.py --check    # claim ids: cited without a row, rows nobody cites

Row grammar (five cells: ID, claim, source, tier and strength, hedge):
  ID      a printed id, a comma list ("B1, B2, I5") or a range ("N3-10 to N3-12"). Ids in square brackets are
          comment tags answered by the row and are not printed ("A4 [N-E6s0]"); a trailing * matches a prefix.
  source  the part before the first "Read", "Recomputed" or "Not checked" marker is printed, followed by that
          marker as the check (read, recomputed, not checked); the rest is verification detail and is not printed.
  tier    a tier starting with "retired" marks a row the paper no longer makes; it is not printed.
Text is LaTeX-escaped; nothing is retyped except one small, asserted edit:

  REDACT  G7 names the submission id and the rank. They are routed through \\submissionref{} and \\rankref{}
          (main.tex prints them in the public copy, nothing in the anonymised copy).

The "Not run" line under the tables is carried as a closing row. Stdlib only, no network, about 1 s.
"""
import fnmatch
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "framing", "FRAMING.md")
SECTIONS = os.path.join(HERE, "..", "tex", "sections")
DST = os.path.join(SECTIONS, "claims_table.tex")

# (id, raw text to find, replacement); applied in order to the claim cell before escaping
REDACT = [
    ("G7", " (submission 57127096, rank 9 of 4,253 on 2026-10-13)", "@@SUBREF@@@@RANKREF@@"),
]

SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
MARK = re.compile(r"\s(Read|Recomputed|Not checked)\b")
CHECK = {"Read": "read", "Recomputed": "recomputed", "Not checked": "not checked"}
IDTOK = re.compile(r"^(?:[A-G]\d{1,2}|H\d{1,2}|I\d|YN\d|N\d(?:[-.][A-Za-z0-9]+)*|N-E6\S*|S\d\d|NEW:\S+|audit_\w+-\w+|E5-conversion|C9-slope)$")


def esc(t, brk=False):
    """Escape plain text. brk=True adds line-break points after _ and / for file names."""
    math = []

    def keep(m):
        sub = "\\infty" if m.group(2) == "inf" else m.group(2)
        math.append("$%s_{%s}$" % (m.group(1), sub))
        return "\x00%d\x00" % (len(math) - 1)

    t = re.sub(r"\b([A-Za-z])_(\d+|[A-Za-z]{1,3})\b", keep, t)      # A_1, p_T, S_inf
    t = re.sub(r"\^(\d+)", lambda m: (math.append("\\textsuperscript{%s}" % m.group(1)) or "\x00%d\x00" % (len(math) - 1)), t)
    t = t.replace("<=", "\x01").replace(">=", "\x02")
    t = "".join(SPECIAL.get(c, c) for c in t)
    t = t.replace("<", "$<$").replace(">", "$>$").replace("\x01", "$\\leq$").replace("\x02", "$\\geq$")
    t = t.replace("_", "\\_\\allowbreak{}" if brk else "\\_")
    if brk:
        t = t.replace("/", "/\\allowbreak{}")
    t = re.sub(r'"([^"]*)"', r"``\1''", t)
    t = re.sub(r"(?<![\w'])'(?=\w)", "`", t)                         # opening single quote
    t = re.sub(r"\x00(\d+)\x00", lambda m: math[int(m.group(1))], t)
    return t


def cell(t, brk=True):
    """Escape a cell: `code` spans become \\texttt, the rest is plain text; sentinels become macros."""
    parts = t.split("`")
    out = [("\\texttt{%s}" % esc(p, True)) if i % 2 else esc(p, brk) for i, p in enumerate(parts)]
    s = "".join(out)
    return s.replace("@@SUBREF@@", "\\submissionref{}").replace("@@RANKREF@@", "\\rankref{}")


def parse_id(idc):
    """'B1, B2 [I5]' -> (printed text, ids, aliases). Ranges 'N3-10 to N3-12' expand."""
    m = re.match(r"^(.*?)(?:\s*\[(.*)\])?$", idc.strip())
    printed = m.group(1).strip()
    aliases = m.group(2).split() if m.group(2) else []
    ids = []
    for t in [x.strip() for x in printed.split(",")]:
        r = re.match(r"^(.*?)(\d+) to (.*?)(\d+)$", t)
        if r:
            assert r.group(1) == r.group(3), idc
            ids += [r.group(1) + str(k) for k in range(int(r.group(2)), int(r.group(4)) + 1)]
        else:
            ids.append(t)
    return printed, ids, aliases


def read_map():
    text = open(SRC, encoding="utf8").read()
    a = text.index("## 5. Claims-evidence map")
    b = text.index("## 6. Writing rules")
    sec = text[a:b]
    head = {}
    for key in ("Tier", "Strength", "Check"):
        m = re.search(r"^%s: (.*)$" % key, sec, re.M)
        head[key] = m.group(1).strip()
    nr = re.search(r"^Not run, and named as such in the paper: (.*)$", sec, re.M).group(1).strip().rstrip(".")
    groups, cur = [], None
    for line in sec.splitlines():
        if line.startswith("### "):
            cur = (line[4:].strip(), [])
            groups.append(cur)
        elif cur and line.startswith("| ") and not line.startswith("| ---") and not line.startswith("| ID "):
            cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
            assert len(cells) == 5, line[:70]
            cur[1].append(cells)
    return head, nr, groups


def cited_ids():
    """Every claim-id-like token in a % comment of sections/*.tex (not claims_table.tex)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(SECTIONS, "*.tex"))):
        if os.path.basename(f) == "claims_table.tex":
            continue
        for n, line in enumerate(open(f, encoding="utf8"), 1):
            m = re.search(r"(?<!\\)%(.*)$", line)
            if m:
                for tok in m.group(1).split():
                    tok = tok.strip(",;:()")
                    if IDTOK.match(tok):
                        out.setdefault(tok, []).append("%s:%d" % (os.path.basename(f), n))
    return out


def check(groups):
    cited = cited_ids()
    live, retired, pats, rows = set(), set(), [], []
    for _, rs in groups:
        for cid, claim, src, tier, hed in rs:
            printed, ids, aliases = parse_id(cid)
            names = ids + aliases
            (retired if tier.startswith("retired") else live).update(n for n in names if "*" not in n)
            pats += [n for n in names if "*" in n]
            rows.append((cid, tier.startswith("retired"), names))
    covered = lambda t: t in live or any(fnmatch.fnmatch(t, p) for p in pats)
    missing = sorted(t for t in cited if not covered(t) and t not in retired)
    only_retired = sorted(t for t in cited if t in retired and not covered(t))
    uncited = [cid for cid, ret, names in rows
               if not any(((n in cited) if "*" not in n else any(fnmatch.fnmatch(t, n) for t in cited)) for n in names)]
    print("claim ids cited in comments: %d; rows: %d (%d retired)" % (len(cited), len(rows), sum(r for _, r, _ in rows)))
    print("cited, no row:", missing or "none")
    print("cited, row is retired:", [(t, cited[t][:2]) for t in only_retired] or "none")
    print("rows no comment cites:", uncited or "none")
    return 1 if missing else 0


def main():
    head, nr, groups = read_map()
    if "--check" in sys.argv:
        return check(groups)
    kept = [x.strip() for x in nr.split(";")]

    redone = set()
    rows, n_print, n_ret = [], 0, 0
    for title, rs in groups:
        body = []
        for cid, claim, src, tier, hed in rs:
            if tier.startswith("retired"):
                n_ret += 1
                continue
            for rid, old, new in REDACT:
                if rid == parse_id(cid)[0]:
                    assert old in claim, (cid, old)
                    claim = claim.replace(old, new)
                    redone.add((rid, old))
            printed, _, _ = parse_id(cid)
            m = MARK.search(" " + src)
            assert m, ("no check marker in the source of " + cid)
            src_print = (" " + src)[:m.start()].strip().rstrip(".;,")
            chk = CHECK[m.group(1)]
            assert not MARK.search(claim + " " + hed + " " + tier), cid
            hed = hed.strip()
            parts = [cell(claim.strip()),
                     "\\textit{Source:} %s, %s." % (cell(src_print), chk),
                     "\\textit{Tier:} %s." % cell(tier.strip().rstrip("."))]
            if hed and hed != "none":
                parts.append("\\textit{Hedge:} %s" % cell(hed))
            body.append("%s & %s \\\\" % (cell(printed), " ".join(parts)))
            n_print += 1
        rows.append((title, body))
    assert redone == {(r, o) for r, o, _ in REDACT}, "a redaction found nothing"

    cap = "Claims and evidence map. Tier: %s Strength: %s Check: %s" % (head["Tier"], head["Strength"], head["Check"])
    cols = "@{}>{\\raggedright\\arraybackslash}p{0.09\\linewidth}>{\\raggedright\\arraybackslash}p{0.89\\linewidth}@{}"
    hd = "ID & Claim, source, tier and hedge \\\\"
    o = ["% Generated by paper/v2/analysis/build_claims_appendix.py from FRAMING.md section 5. Do not edit by hand.",
         "\\begingroup\\renewcommand{\\baselinestretch}{0.95}\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{longtable}{%s}" % cols,
         "\\caption{%s}\\label{tab:claims}\\\\" % cell(cap, False),
         "\\toprule", hd, "\\midrule", "\\endfirsthead",
         "\\multicolumn{2}{@{}l}{\\textit{Table~\\ref{tab:claims}, continued}}\\\\", "\\toprule", hd, "\\midrule", "\\endhead",
         "\\bottomrule", "\\endfoot", "\\bottomrule", "\\endlastfoot"]
    for k, (title, body) in enumerate(rows):
        if k:
            o.append("\\midrule")
        o.append("\\multicolumn{2}{@{}l}{\\textbf{%s}}\\\\*" % cell(title, False))     # no page break after a group title
        for b in body:
            o.append(b)
            o.append("\\addlinespace[0.5pt]")
    o.append("\\midrule")
    o.append("\\multicolumn{2}{@{}p{0.98\\linewidth}}{\\textit{Not run, and named as such in the paper: %s.}}\\\\" % cell("; ".join(kept), False))
    o += ["\\end{longtable}", "\\endgroup", ""]
    out = "\n".join(o)

    assert "\u2013" not in out and "\u2014" not in out and "--" not in out.replace("\\\\", ""), "dash in output"
    assert all(ord(c) < 128 for c in out), "non-ASCII in output"
    assert out.count("{") == out.count("}"), "unbalanced braces"
    open(DST, "w", encoding="utf8").write(out)
    print("wrote %s: %d rows in %d groups (%d retired rows skipped), %d bytes" % (os.path.relpath(DST), n_print, len(rows), n_ret, len(out)))


if __name__ == "__main__":
    sys.exit(main())

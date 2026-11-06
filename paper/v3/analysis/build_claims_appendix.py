#!/usr/bin/env python3
"""Appendix B of the v3 PDF: build paper/v3/tex/sections/claims_table.tex from three claim tables.

    python3 -I -B paper/v3/analysis/build_claims_appendix.py            # write the table
    python3 -I -B paper/v3/analysis/build_claims_appendix.py --check    # ids cited in v3 comments without a row

Sources, printed in this order (same five-cell grammar as v2; see paper/v2/analysis/build_claims_appendix.py):
  paper/v2/framing/FRAMING.md section 5        v2 rows (A1 ...); rows the dossier updates get an "Updated in v3" note
  paper/v3/evidence/dossier_1013_1021.md       V3 rows, one group per dossier topic
  paper/v3/evidence/claims_addendum.md         V3-W rows added while writing the v3 text
Reuses the v2 builder's escaping and id parsing. Identifying strings are routed through macros that main.tex defines
for the public and the anonymised copy: submission ids -> \\subid{}, the Kaggle handle -> \\ownerid{}, "rank N of M" ->
\\rankof{N}{M}. Stdlib only, no network.
"""
import fnmatch
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PAPER, "v2", "analysis"))
import build_claims_appendix as v2b  # noqa: E402  (esc, cell, parse_id, read_map, MARK, CHECK)

V2_MAP = os.path.join(PAPER, "v2", "framing", "FRAMING.md")
DOSSIER = os.path.join(PAPER, "v3", "evidence", "dossier_1013_1021.md")
ADDENDUM = os.path.join(PAPER, "v3", "evidence", "claims_addendum.md")
SECTIONS = os.path.join(PAPER, "v3", "tex", "sections")
DST = os.path.join(SECTIONS, "claims_table.tex")

# v2 rows that the dossier's section 9 updates: first printed id -> note
UPDATED = {
    "G7": "V3-B1, V3-B2", "G1": "V3-B1, V3-B7", "G3, G5": "V3-B5", "B11": "V3-B7",
    "D1, YN1": "V3-H5", "D2": "V3-H5", "YN7": "V3-H5", "E8": "V3-H2, V3-H4", "E1": "V3-T7",
    "E6, E11, E12": "V3-T7", "F3": "V3-L9, V3-L10, V3-T3", "N5, N6, H1, H6": "V3-X1, V3-X8, V3-X11",
    "YN6": "V3-X13; the interleaved test has since run (V3-W36: -2.5 levels, p 0.42)", "B1, B2, I5, I6": "V3-X3, V3-X4", "B4, I1 to I4, I8": "V3-X12",
    "H10": "not recounted; the AI-use line in Appendix K replaces it",
    "C2, N2, C9": "the extrapolated share is 85% to 91% (Table plateau); 94% is the eight-run value, not an extrapolation",
    "C4": "v3 quotes only the two-sided values, at the identical-run SDs 5.95 and 8.25 (V3-W12)",
}
ANON_HIDE = {"V3-B3"}
SKIP = {"V3-R3"}   # a planning model; the dossier says it does not belong among the paper's claims
IDTOK = re.compile(v2b.IDTOK.pattern[:-2] + r"|V3-[A-Z]\d{1,2})$")
SUBID = re.compile(r"\b(5[67]\d{6})\b")
RANK = re.compile(r"\brank (\d+) of ([\d,]+)")


def prep(t):
    """Markdown and identifying strings to sentinels that survive v2b.cell (which escapes braces)."""
    t = re.sub(r"\*\*(.+?)\*\*", "\x06\\1\x07", t)
    t = t.replace("--", "-\x05-")
    t = SUBID.sub("\x0e\\1\x0f", t)
    t = RANK.sub("\x10\\1\x11\\2\x12", t)
    return t.replace("vineethsaivs", "\x13")


def post(t):
    return (t.replace("\x06", "\\textbf{").replace("\x07", "}").replace("\x05", "{}")
             .replace("\x0e", "\\subid{").replace("\x0f", "}")
             .replace("\x10", "\\rankof{").replace("\x11", "}{").replace("\x12", "}")
             .replace("\x13", "\\ownerid{}"))


def cell(t, brk=True):
    return post(v2b.cell(prep(t), brk))


def read_md_groups(path, title_prefix):
    """Groups of five-cell rows whose id starts with V3-, titled by the enclosing '## ' heading."""
    groups, cur = [], None
    for line in open(path, encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("## "):
            cur = (title_prefix + re.sub(r"^\d+\.\s*", "", line[3:].strip()), [])
            groups.append(cur)
        elif cur is not None and line.startswith("| V3-"):
            cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
            assert len(cells) == 5, line[:70]
            cur[1].append(cells)
    return [g for g in groups if g[1]]


def all_groups():
    head, nr, groups = v2b.read_map()
    groups = [("v2 " + t, rs) for t, rs in groups]
    return head, nr, groups + read_md_groups(DOSSIER, "V3 ") + read_md_groups(ADDENDUM, "V3 ")


def cited_ids():
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
    live, retired, pats = set(), set(), []
    for _, rs in groups:
        for cid, claim, src, tier, hed in rs:
            _, ids, aliases = v2b.parse_id(cid)
            names = ids + aliases
            (retired if tier.startswith("retired") else live).update(n for n in names if "*" not in n)
            pats += [n for n in names if "*" in n]
    covered = lambda t: t in live or any(fnmatch.fnmatch(t, p) for p in pats)
    missing = sorted(t for t in cited if not covered(t))
    v3rows = sorted(n for n in live if n.startswith("V3-"))
    print("claim ids cited in v3 comments: %d; live ids with rows: %d (V3: %d)" % (len(cited), len(live), len(v3rows)))
    print("cited, no live row:", [(t, cited[t][:2]) for t in missing] or "none")
    print("V3 rows no v3 comment cites:", [n for n in v3rows if n not in cited] or "none")
    return 1 if missing else 0


def main():
    head, nr, groups = all_groups()
    if "--check" in sys.argv:
        return check(groups)
    kept = [x.strip() for x in nr.split(";")]
    rows, n_print, n_ret, seen = [], 0, 0, set()
    for title, rs in groups:
        body = []
        for cid, claim, src, tier, hed in rs:
            if tier.startswith("retired") or v2b.parse_id(cid)[0] in SKIP:
                n_ret += 1
                continue
            printed, _, _ = v2b.parse_id(cid)
            m = v2b.MARK.search(" " + src)
            if m:   # v2 grammar: the part before the capitalised marker is printed
                src_print, chk = (" " + src)[:m.start()].strip().rstrip(".;,"), v2b.CHECK[m.group(1)]
            else:   # a few dossier cells mark the check inline in lower case: print the whole cell
                m = re.search(r"\b(recomputed|read|not checked|derived)\b", src, re.I)
                assert m, "no check marker in the source of " + cid
                src_print, chk = src.strip().rstrip(".;,"), "source checked"
            claim_tex = cell(claim.strip())
            if printed in ANON_HIDE:   # our own rank by date identifies the entry
                claim_tex = "\\ifanon Our rank history is withheld in the anonymised copy.\\else %s\\fi" % claim_tex
            parts = [claim_tex,
                     "\\textit{Source:} %s, %s." % (cell(src_print), chk),
                     "\\textit{Tier:} %s." % cell(tier.strip().rstrip("."))]
            if hed.strip() and hed.strip() != "none":
                parts.append("\\textit{Hedge:} %s" % cell(hed.strip()))
            if printed in UPDATED:
                parts.append("\\textit{Updated in v3:} %s." % cell(UPDATED[printed]))
                seen.add(printed)
            body.append("%s & %s \\\\" % (cell(printed), " ".join(parts)))
            n_print += 1
        rows.append((title, body))
    assert seen == set(UPDATED), "an update note found no row: %s" % (set(UPDATED) - seen)

    cap = ("Claims and evidence map: v2 rows (ids without a prefix, through 2026-10-13) and V3 rows (2026-10-13 to "
           "2026-10-21). Tier: %s Strength: %s Check: %s" % (head["Tier"], head["Strength"], head["Check"]))
    cols = "@{}>{\\raggedright\\arraybackslash}p{0.09\\linewidth}>{\\raggedright\\arraybackslash}p{0.89\\linewidth}@{}"
    hd = "ID & Claim, source, tier and hedge \\\\"
    o = ["% Generated by paper/v3/analysis/build_claims_appendix.py. Do not edit by hand.",
         "\\begingroup\\renewcommand{\\baselinestretch}{0.95}\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{longtable}{%s}" % cols,
         "\\caption{%s}\\label{tab:claims}\\\\" % cell(cap, False),
         "\\toprule", hd, "\\midrule", "\\endfirsthead",
         "\\multicolumn{2}{@{}l}{\\textit{Table~\\ref{tab:claims}, continued}}\\\\", "\\toprule", hd, "\\midrule", "\\endhead",
         "\\bottomrule", "\\endfoot", "\\bottomrule", "\\endlastfoot"]
    for k, (title, body) in enumerate(rows):
        if k:
            o.append("\\midrule")
        o.append("\\multicolumn{2}{@{}l}{\\textbf{%s}}\\\\*" % cell(title, False))
        for b in body:
            o += [b, "\\addlinespace[0.5pt]"]
    o.append("\\midrule")
    o.append("\\multicolumn{2}{@{}p{0.98\\linewidth}}{\\textit{Not run in v2, and named as such there: %s. "
             "K0 headroom has since been measured (V3-T7), and the interleaved yield test has since run (V3-W36).}}\\\\" % cell("; ".join(kept), False))
    o += ["\\end{longtable}", "\\endgroup", ""]
    out = "\n".join(o)
    assert "\u2013" not in out and "\u2014" not in out and "--" not in out.replace("\\\\", ""), "dash in output"
    assert all(ord(c) < 128 for c in out), "non-ASCII in output"
    assert out.count("{") == out.count("}"), "unbalanced braces"
    assert not re.search(r"[\x00-\x08\x0e-\x1f]", out), "sentinel left in output"
    open(DST, "w", encoding="utf8").write(out)
    print("wrote %s: %d rows in %d groups (%d retired rows skipped), %d bytes"
          % (os.path.relpath(DST, PAPER), n_print, len(rows), n_ret, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

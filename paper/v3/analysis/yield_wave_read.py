"""Read the interleaved yield wave by its predeclared rules
(solution/kaggle/generated/gate-predeclared-2026-10-21-yield-interleaved.json): y8 (l1a1y8sv2r3rq) against y2
(l1a1y8sv2r3rqy2). Stdlib only; read-only on the runs.

    python3 -I paper/v3/analysis/yield_wave_read.py --y8 yw-y8-{c,e}-{1..4} --y2 yw-y2-{c,e}-{1..4} [--json OUT]
    python3 -I paper/v3/analysis/yield_wave_read.py --test-aa --y8 rq1020-{1..4} --y2 rq1020-{5..8}   # reader test
    python3 -I paper/v3/analysis/yield_wave_read.py --power | --self-test                            # the JSON's power; rules

Per run it fetches only small files: solution/gcp/vertex_fetch.sh NAME (benchmark.json, serve.log, kernel.log,
notebook.log, progress.txt, DONE; about 6 MB) into solution/kaggle/generated/kernel-outputs/vertex-NAME, and the 25
per-game transcripts (under 4 MB each, any file over 50 MB refused) into a temporary directory that is deleted after
parsing; the per-game summary is kept in paper/v3/analysis/out/yield_wave/NAME.json. The harness's per-request logs
(25 to 132 MB each) and the reqlog blobs stay in GCS. Matched-token levels use benchmark.json's per-action generated
tokens (franzen_readout.level_tokens), which sum to 98.0% of the per-game totals on rq1020-1.

No verdict while any run lacks DONE (no interim look at levels). --test-aa reads two groups of the same notebook to
test the reader: it skips the arm identity and manipulation checks and the 6-run minimum, and gives no verdict.
"""
import argparse
import json
import math
import random
import re
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "paper/v2/analysis"), str(REPO / "solution/gcp"), str(REPO / "solution/analysis")]
import franzen_readout as fr  # noqa: E402
import wave_read as wr  # noqa: E402  (run validity: games, READY, server death, OOM)
import yield_mech as ym  # noqa: E402  (transcript parser, invocation end status)
import yield_stats as ys  # noqa: E402  (t_sf, boot_ci, exact_perm)

KO = REPO / "solution/kaggle/generated/kernel-outputs"
CACHE = Path(__file__).resolve().parent / "out/yield_wave"
BUCKETS = {"gs://arc3-rep-uc1": "us-central1", "gs://arc3-rep-ue4": "us-east4"}
ARM = {"y8": ("l1a1y8sv2r3rq", "a8e2e846639b", "8192"), "y2": ("l1a1y8sv2r3rqy2", "e695102c96ba", "2048")}
SEED, ALPHA, MIN_VALID, BIG = 20261021, 0.05, 6, 50 * 1024 * 1024
MARKS = (100_000, 150_000, 200_000, 250_000, 300_000)
FLAGS = ("server death", "OOM")  # counted (intention to treat), reported, and dropped in the sensitivity read
TAIL_RATE = 11.2  # levels per million generated tokens, the generous tail rate (FRAMING.md D7, D9)


def gs(*args):
    return subprocess.run(["gcloud", "storage", *args], capture_output=True, text=True)


def read(p):
    return p.read_text(errors="replace") if p.is_file() else ""


def bucket_of(name):
    return next((b for b in BUCKETS if gs("ls", f"{b}/runs/{name}/DONE").returncode == 0), None)


def transcripts(name, bucket):
    """Per-game counters from the run's transcripts, cached once the run is DONE."""
    cache = CACHE / f"{name}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    src = f"{bucket}/runs/{name}/working/transcripts/"
    ls = gs("ls", "-l", src)
    uris = []
    for line in ls.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].endswith("_p0.txt"):
            if int(parts[0]) > BIG:
                sys.exit(f"{parts[2]} is {int(parts[0]) / 2**20:.0f} MB: refusing to copy it to the Mac")
            uris.append(parts[2])
    if not uris:
        return None
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        if gs("cp", *uris, tmp).returncode:
            sys.exit(f"copy of {src} failed")
        for p in sorted(Path(tmp).glob("*_p0.txt")):
            invs = ym.parse_transcript(p)
            ends = [ym.end_status(i["sections"]) for i in invs]
            out[p.name.split("-")[0]] = {
                "invocations": len(invs),
                "resume": sum(ends[k - 1] != "executed" for k in range(1, len(invs))),  # yield_mech.py definition
                "yield_ends": ends.count("yield"),
                "requests": sum(lab == "MODEL RESPONSE META" for i in invs for lab, _ in i["sections"]),
                "yield_tokens": sorted({m for i in invs for lab, t in i["sections"] if lab == "ANALYZER STATUS"
                                        for m in re.findall(r"^yield_tokens: (\S+)", t, re.M)})}
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def load(name, arm, o):
    d = KO / f"vertex-{name}"
    bucket = bucket_of(name) if not o.no_fetch else None
    if bucket and not ((d / "DONE").is_file() and (d / "benchmark.json").is_file()):
        subprocess.run([str(REPO / "solution/gcp/vertex_fetch.sh"), name, str(d)], check=False, stdout=subprocess.DEVNULL)
    r = {"name": name, "arm": arm, "region": BUCKETS.get(bucket, "?"), "done": "rc=" in read(d / "DONE")}
    if not r["done"]:
        return r
    rec = wr.run_record(d)
    r["flags"] = [w for w in rec["why"] if w in FLAGS]
    r["invalid"] = [w for w in rec["why"] if w not in FLAGS] + ([] if rec["rc"] == "0" else [f"rc={rec['rc']}"])
    arm_name, sha, _ = ARM["y8" if o.test_aa else arm]
    nbs = re.findall(r"notebook=(\S+)", read(d / "progress.txt"))
    if not nbs or not all(sha in x or f"candidates-franzen-{arm_name}/" in x for x in nbs):
        r["invalid"].append(f"notebook is not {arm_name} ({', '.join(nbs) or 'none in progress.txt'})")
    x = fr.readout(d)
    r.update(levels=x.get("levels"), tokens=x.get("generated_tokens"), tok_s=x.get("tok_s"),
             cache_hit=x.get("cache_hit_pct"), accept=x.get("accept_len_mean"), prefill_new=x.get("prefill_new_tokens"),
             pg={g.split("-")[0]: v for g, v in (x.get("per_game") or {}).items()})
    r["tx"] = transcripts(name, bucket) if bucket and not o.no_transcripts else (
        json.loads((CACHE / f"{name}.json").read_text()) if (CACHE / f"{name}.json").is_file() else None)
    return r


def welch(a, b):
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    d, se = st.fmean(a) - st.fmean(b), math.sqrt(va + vb)
    if se == 0:
        return d, float("nan"), float("nan"), float("nan")
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return d, d / se, df, min(1.0, 2 * ys.t_sf(abs(d / se), df))


def compare(label, a, b, scale=1.0, nd=2):
    """Welch t and a bootstrap 95% CI for mean(a) - mean(b); prints one line, returns the numbers."""
    a, b = [v / scale for v in a], [v / scale for v in b]
    d, t, df, p = welch(a, b)
    lo, hi = ys.boot_ci(a, b, n=20000, seed=SEED)
    print(f"  {label}: y8 {st.fmean(a):.{nd}f} (SD {st.stdev(a):.{nd}f}), y2 {st.fmean(b):.{nd}f} (SD {st.stdev(b):.{nd}f}); "
          f"Delta {d:+.{nd}f}, Welch t {t:.2f}, df {df:.1f}, p {p:.3g}; bootstrap 95% CI {lo:+.{nd}f} to {hi:+.{nd}f}")
    return {"y8": st.fmean(a), "y2": st.fmean(b), "sd_y8": st.stdev(a), "sd_y2": st.stdev(b),
            "delta": d, "t": t, "df": df, "p": p, "ci": [lo, hi]}


def levels_at(r, games, cut):
    return sum(sum(1 for e in r["pg"][g]["level_tokens"] if e <= cut[g]) for g in games)


def strata(r, games):
    """{(game, level): (events, exposure tokens)}, yield_pertoken.py method B."""
    out = {}
    for g in games:
        x, start = r["pg"][g], 0
        for lvl, e in enumerate(x["level_tokens"], 1):
            out[(g, lvl)] = (1, max(e - start, 1))
            start = e
        if x["levels"] < (x["of"] or 0):
            out[(g, x["levels"] + 1)] = (0, max(x["tokens"] - start, 0))
    return out


def mh(a, b):
    """Mantel-Haenszel rate ratio a over b; exposure averaged per run so arm size drops out (yield_pertoken.mh_rr)."""
    def pool(rs):
        ev, ex = {}, {}
        for s in rs:
            for k, (n, t) in s.items():
                ev[k], ex[k] = ev.get(k, 0) + n / len(rs), ex.get(k, 0) + t / len(rs)
        return ev, ex
    (da, ea), (db, eb) = pool(a), pool(b)
    num = den = 0.0
    for k in ea.keys() | eb.keys():
        ta, tb = ea.get(k, 0), eb.get(k, 0)
        if ta + tb > 0:
            num += da.get(k, 0) * tb / (ta + tb)
            den += db.get(k, 0) * ta / (ta + tb)
    return num / den if den else float("nan")


def decide(res, bad):
    """The predeclared decision rules (JSON decision_rules, amended 02:55Z; superseded_rules_0248Z kept) on the computed outcomes."""
    if bad:
        return f"NO READ: manipulation check failed (wrong yield_tokens in {bad})"
    if res["resume"]["y8"] >= 0.5 * res["resume"]["y2"]:
        return (f"NO READ: manipulation check failed (resume openers per game y8 {res['resume']['y8']:.2f} not below "
                f"half of y2 {res['resume']['y2']:.2f})")
    pr = res["primary"]
    if pr["p"] >= ALPHA:
        return ("C NULL: the yield's gain did not survive an interleaved test"
                + (" (the non-interleaved +12.5 is excluded)" if pr["ci"][1] < 12.5 else
                   " (effects up to the CI's upper bound are not excluded)")
                + ". Paper: FRAMING_v3 section 9's 'if not' branch; the family gain goes to serving tokens and day")
    if pr["delta"] < 0:
        return "D HARM: the 8,192 yield lowered levels interleaved. Paper: decisions moved only down; flag the pin to the owner"
    if pr["ci"][0] > res["token_explained"] or res["matched"]["ci"][0] > 0 or res["per_token_rr"]["ci"][0] > 1:
        return ("A DECISIONS: the gain exceeds what its tokens buy. Paper: restore 'Decisions, Not Tokens'; contribution 3 "
                "is a context lever tested interleaved")
    return ("B TOKENS: the gain fits the yield's extra tokens. Paper: 'tokens matter more than the budget curve says; "
            "decisions moved only one way (down, by imitation)'")


def decide_0248(res):
    """The superseded 02:48Z rules (JSON superseded_rules_0248Z), reported beside the binding verdict."""
    pr = res["primary"]
    if pr["p"] >= ALPHA or pr["delta"] < 0:
        return "same as the binding rules (C or D)"
    if res["matched"]["ci"][0] > 0 or res["per_token_rr"]["ci"][0] > 1:
        return "A DECISIONS"
    return "B TOKENS" if res["tokens_per_game"]["delta"] > 0 else "B OPEN"


def self_test():
    base = {"resume": {"y8": 8.3, "y2": 23.0}, "primary": {"p": 0.01, "delta": 12.0, "ci": [6.0, 18.0]},
            "token_explained": 1.6, "matched": {"ci": [-5.0, 20.0]}, "per_token_rr": {"ci": [0.95, 1.2]}}
    case = lambda **kw: decide({**base, **kw}, [])  # noqa: E731
    assert case().startswith("A "), case()
    assert case(token_explained=7.0).startswith("B "), "gain inside what the extra tokens buy"
    assert case(token_explained=7.0, per_token_rr={"ci": [1.01, 1.3]}).startswith("A ")
    assert case(primary={"p": 0.2, "delta": 3.0, "ci": [-4.0, 10.0]}).startswith("C ") and "excluded)" in case(
        primary={"p": 0.2, "delta": 3.0, "ci": [-4.0, 10.0]})
    assert case(primary={"p": 0.01, "delta": -10.0, "ci": [-16.0, -4.0]}).startswith("D ")
    assert case(resume={"y8": 15.0, "y2": 23.0}).startswith("NO READ")
    assert decide(base, ["yw-y2-c-1"]).startswith("NO READ")
    print("self-test ok: 7 cases")


def t_ppf(q, df):
    lo, hi = 0.0, 20.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 1 - ys.t_sf(mid, df, 4000) < q else (lo, mid)
    return (lo + hi) / 2


def power(n=8, sims=200_000):
    """Student t, 2n-2 df (for equal n the Welch statistic equals the pooled one), two-sided 5%, simulated noncentral t."""
    df = 2 * n - 2
    tc, t80 = t_ppf(0.975, df), t_ppf(0.80, df)
    rng = random.Random(SEED)
    print(f"{n} vs {n}, df {df}, t_0.975 {tc:.3f}, t_0.80 {t80:.3f}")
    for sd in (5.95, 8.25):
        se = sd * math.sqrt(2 / n)
        pw = {d: sum(abs((rng.gauss(0, 1) + d / se) / math.sqrt(rng.gammavariate(df / 2, 2) / df)) > tc
                     for _ in range(sims)) / sims for d in (1.0, 5.0, 9.0, 12.5)}
        print(f"SD {sd}: SE {se:.3f}, MDE80 {(tc + t80) * se:.2f} (normal {2.802 * se:.2f}); power "
              + ", ".join(f"{d:+g}: {p:.3f}" for d, p in pw.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--y8", nargs="*", default=[])
    ap.add_argument("--y2", nargs="*", default=[])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--no-fetch", action="store_true", help="read local copies only (no gcloud)")
    ap.add_argument("--no-transcripts", action="store_true", help="skip transcripts (no openers, no manipulation check)")
    ap.add_argument("--test-aa", action="store_true", help="reader test on two groups of the y8 notebook: no verdict")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="check the decision rules on synthetic outcomes")
    o = ap.parse_args()
    if o.power:
        return power()
    if o.self_test:
        return self_test()
    if not o.y8 or not o.y2:
        ap.error("--y8 and --y2 are required")
    runs = [load(n, "y8", o) for n in o.y8] + [load(n, "y2", o) for n in o.y2]
    if not all(r["done"] for r in runs):
        print("NO READ: not finished (no DONE): " + ", ".join(f"{r['name']} ({r['region']})" for r in runs if not r["done"]))
        print("The predeclared rule reads only when every launched run has DONE; no levels are shown before that.")
        return
    print("run | arm | region | levels | K tokens/game | resume openers/game | requests/game | tok/s | notes")
    for r in runs:
        tx = r["tx"] or {}
        print(f"{r['name']} | {r['arm']} | {r['region']} | {r['levels']} | {r['tokens'] / 25e3:.1f} | "
              + (f"{sum(g['resume'] for g in tx.values()) / 25:.2f} | {sum(g['requests'] for g in tx.values()) / 25:.1f}"
                 if tx else "n/a | n/a") + f" | {r['tok_s'] or 0:.1f} | "
              + "; ".join([f"INVALID: {w}" for w in r["invalid"]] + [f"flag: {w}" for w in r["flags"]]))
    v = {k: [r for r in runs if r["arm"] == k and not r["invalid"]] for k in ARM}
    if not o.test_aa and min(len(x) for x in v.values()) < MIN_VALID:
        print(f"\nNO READ: {len(v['y8'])} valid y8, {len(v['y2'])} valid y2; the rule needs {MIN_VALID} each "
              "(rerun each run that did not complete once, as NAME-r)")
        return
    games = sorted(set.intersection(*(set(r["pg"]) for r in v["y8"] + v["y2"])))
    res = {"runs": [{k: r.get(k) for k in ("name", "arm", "region", "levels", "tokens", "tok_s", "invalid", "flags")}
                    for r in runs], "games": len(games)}
    lv = {k: [r["levels"] for r in v[k]] for k in ARM}
    print(f"\nPRIMARY: levels per run on {len(games)} games ({len(v['y8'])} y8, {len(v['y2'])} y2 valid runs)")
    res["primary"] = compare("levels", lv["y8"], lv["y2"])
    _, p1, p2, splits = ys.exact_perm(lv["y8"], lv["y2"])
    print(f"  exact run-label permutation: two-sided p {p2:.3g} ({splits:,} splits)")
    res["primary"]["perm_p"] = p2
    clean = {k: [r for r in v[k] if not r["flags"]] for k in ARM}
    if any(r["flags"] for k in ARM for r in v[k]) and min(len(x) for x in clean.values()) > 1:
        res["sensitivity_unflagged"] = compare("sensitivity, without flagged runs",
                                               [r["levels"] for r in clean["y8"]], [r["levels"] for r in clean["y2"]])
    regs = [g for g in BUCKETS.values() if all(any(r["region"] == g for r in v[k]) for k in ARM)]
    if regs:
        dr = {g: st.fmean(r["levels"] for r in v["y8"] if r["region"] == g)
              - st.fmean(r["levels"] for r in v["y2"] if r["region"] == g) for g in regs}
        print("  by region (report only): " + ", ".join(f"{g} {d:+.2f}" for g, d in dr.items())
              + f"; region-stratified mean {st.fmean(dr.values()):+.2f}")
        res["by_region"] = dr

    d = {g: st.fmean(r["pg"][g]["levels"] for r in v["y8"]) - st.fmean(r["pg"][g]["levels"] for r in v["y2"])
         for g in games}
    up, down = sum(x > 0 for x in d.values()), sum(x < 0 for x in d.values())
    n = up + down
    sign_p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(up, down) + 1)) / 2 ** n) if n else 1.0
    print(f"\nPAIRED PER GAME: {up} up, {down} down, {len(games) - n} tied; sign test two-sided p {sign_p:.3g}")
    print("  " + ", ".join(f"{g} {x:+.2f}" for g, x in sorted(d.items(), key=lambda kv: -kv[1])))
    res["per_game"] = {"d": d, "up": up, "down": down, "sign_p": sign_p}

    print("\nSECONDARY")
    res["tokens_per_game"] = compare("generated K tokens per game", [r["tokens"] / 25 for r in v["y8"]],
                                     [r["tokens"] / 25 for r in v["y2"]], scale=1e3, nd=1)
    print(f"  ratio y8/y2 {res['tokens_per_game']['y8'] / res['tokens_per_game']['y2']:.3f}")
    has_tx = all(r["tx"] for k in ARM for r in v[k])
    if has_tx:
        per = lambda r, key: sum(g[key] for g in r["tx"].values()) / len(r["tx"])  # noqa: E731
        for key, label in (("resume", "resume openers per game"), ("yield_ends", "yield ends per game"),
                           ("requests", "requests per game")):
            res[key] = compare(label, [per(r, key) for r in v["y8"]], [per(r, key) for r in v["y2"]])
        print(f"  resume openers ratio y8/y2 {res['resume']['y8'] / res['resume']['y2']:.3f}")
        res["tokens_per_request"] = compare("generated tokens per request",
                                            [r["tokens"] / sum(g["requests"] for g in r["tx"].values()) for r in v["y8"]],
                                            [r["tokens"] / sum(g["requests"] for g in r["tx"].values()) for r in v["y2"]],
                                            nd=0)
    else:
        print("  resume openers: n/a (transcripts not read)")
    cut = {g: min(st.median(r["pg"][g]["tokens"] for r in v["y8"]), st.median(r["pg"][g]["tokens"] for r in v["y2"]))
           for g in games}
    print(f"  matched tokens: each game cut at the smaller arm median of its tokens ({sum(cut.values()) / 25e3:.1f}K per "
          "game on average)")
    res["matched"] = compare("levels at matched tokens per game", [levels_at(r, games, cut) for r in v["y8"]],
                             [levels_at(r, games, cut) for r in v["y2"]])
    res["marks"] = {}
    for m in MARKS:
        a, b = ([levels_at(r, games, dict.fromkeys(games, m)) for r in v[k]] for k in ARM)
        reach = [st.fmean(sum(r["pg"][g]["tokens"] >= m for g in games) for r in v[k]) for k in ARM]
        _, _, _, p = welch(a, b)
        print(f"  at {m // 1000}K tokens per game: y8 {st.fmean(a):.1f}, y2 {st.fmean(b):.1f}, Delta "
              f"{st.fmean(a) - st.fmean(b):+.1f}, Welch p {p:.3g}; games reaching y8 {reach[0]:.1f}, y2 {reach[1]:.1f}")
        res["marks"][m] = {"y8": st.fmean(a), "y2": st.fmean(b), "p": p, "reach": reach}
    sa, sb = ([strata(r, games) for r in v[k]] for k in ARM)
    rng = random.Random(SEED)
    boots = sorted(mh(rng.choices(sa, k=len(sa)), rng.choices(sb, k=len(sb))) for _ in range(2000))
    rr = mh(sa, sb)
    res["per_token_rr"] = {"rr": rr, "ci": [boots[50], boots[1949]]}
    print(f"  per-token clear rate, Mantel-Haenszel y8/y2 {rr:.3f}, run-bootstrap 95% CI {boots[50]:.3f} to {boots[1949]:.3f}")
    ser = {k: [st.fmean(r[f] for r in v[a] if r.get(f) is not None) if any(r.get(f) is not None for r in v[a])
               else float("nan") for a in ARM] for k, f in (("tok/s", "tok_s"), ("cache hit %", "cache_hit"),
                                                            ("accept", "accept"), ("new prefill M", "prefill_new"))}
    ser["new prefill M"] = [x / 1e6 for x in ser["new prefill M"]]
    print("  serving (report only), y8 vs y2: " + ", ".join(f"{k} {a:.2f} vs {b:.2f}" for k, (a, b) in ser.items()))
    res["serving"] = ser

    res["token_explained"] = max(0.0, res["tokens_per_game"]["delta"] * 25 / 1e3 * TAIL_RATE)
    print(f"  token-explained levels: {res['token_explained']:.2f} (extra tokens per run x {TAIL_RATE} levels per M)")
    print()
    if o.test_aa:
        verdict = "A/A TEST: two groups of the same notebook; no verdict"
    elif not has_tx:
        verdict = "NO VERDICT: the manipulation check needs the transcripts (rerun without --no-transcripts)"
    else:
        bad = sorted({r["name"] for k in ARM for r in v[k] for g in r["tx"].values() if g["yield_tokens"] != [ARM[k][2]]})
        verdict = decide(res, bad)
        if not verdict.startswith("NO READ"):
            res["verdict_superseded_0248Z"] = decide_0248(res)
            print("superseded 02:48Z rules: " + res["verdict_superseded_0248Z"])
    print("VERDICT: " + verdict)
    res["verdict"] = verdict
    if o.json:
        o.json.parent.mkdir(parents=True, exist_ok=True)
        o.json.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()

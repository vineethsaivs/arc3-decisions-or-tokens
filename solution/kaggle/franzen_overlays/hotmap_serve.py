"""Arm hotmap: serve the MTP drafter a draft vocabulary built from our own traffic instead of Pennyroyal's.

What. The drafter scores only the ids in --speculative-token-map: its head is those rows of the target lm_head
(sglang/srt/speculative/eagle_worker_v2.py:292-293, hot_vocab.py:279-293), so a sampled token outside the map always
rejects its draft position. Pennyroyal's 64K map was built from SGLang source code and misses 1.41% of what our model
generates (the three "Hmm" ids alone 0.66%); a map built from our transcripts misses under 0.3% on held-out games, and
a 32K one also halves the draft head's reads per draft step (docs/research/2026-10-12-next/d4-drafter.md section 5,
PLAN.md item 3). Lossless: the target verifies every token, so only acceptance and step time can change.

How. solution/kaggle/build_arc_hot_map.py writes the ids as a zlib + base64 bitmap over the vocabulary with its
sha256 (generated/arc-hot-map-2026-10-12.json); build_franzen_arms.hotmap() embeds one in cell 15 in place of
Pennyroyal's map. install() decodes and checks it (sha256, count, the five ids a reply needs drafted), writes the ids
as a list of ints with the venv's torch (the notebook process never imports torch) and returns the path for
--speculative-token-map, which SGLang reads with torch.load(weights_only=True) (spec_utils.py:653-680).
solution/sim/test_franzen_hotmap.py round-trips both maps through that loader.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path

MUST = (248044, 248046, 248058, 248059, 248069)  # <|endoftext|> <|im_end|> <tool_call> </tool_call> </think>
WORK = "/tmp/sgl-intel/hotmap"
SAVE = ("import json, sys, torch; ids = json.load(open(sys.argv[1])); torch.save(ids, sys.argv[2]); "
        "assert torch.load(sys.argv[2], weights_only=True) == ids")


def decode(bitmap_b64: str, sha256: str, n: int) -> list[int]:
    bits = zlib.decompress(base64.b64decode(bitmap_b64))
    if hashlib.sha256(bits).hexdigest() != sha256:
        raise RuntimeError(f"hotmap: the embedded bitmap does not match sha256 {sha256}")
    ids = [i for i in range(len(bits) * 8) if bits[i >> 3] >> (i & 7) & 1]
    if len(ids) != n or not set(MUST) <= set(ids):
        raise RuntimeError(f"hotmap: {len(ids)} ids (want {n}), or one of {MUST} is missing")
    return ids


def install(bitmap_b64: str, sha256: str, n: int, python, run, env, work=WORK) -> Path:
    ids = decode(bitmap_b64, sha256, n)
    src, out = Path(work, f"arc-traffic-{n}.json"), Path(work, f"arc-traffic-{n}.pt")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps(ids))
    run([python, "-c", SAVE, src, out], env=env)
    print(f"hotmap: ARC traffic map {n} ids (bitmap sha256 {sha256[:12]}) -> {out}", flush=True)
    return out

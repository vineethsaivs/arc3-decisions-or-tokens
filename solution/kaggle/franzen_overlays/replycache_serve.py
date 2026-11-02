"""Arm replycache, serving side: one serve.log line per finished request saying how deep it
left its reply in the radix cache. The fix itself is harness-side (franzen_overlays/replycache.py).

Runs inside the launcher cell (cell 15): apply(VENV, env, MODEL_DIR) after the install and
target validation, before the server starts. Off unless env["ARC3_REPLYCACHE_DIAG"] == "1"; off,
it leaves SGLang untouched. On, it also prints the git blob of MODEL_DIR/chat_template.jinja (the
file cell 15 passes as --chat-template; p448's view symlinks it), so a bench log records whether
the server renders history with the template the round trip was checked on.

The line, logged at the end of UnifiedRadixCache.cache_finished_req:

  replycache rid=R in=I out=O kv=K track=T cache_len=C protected=P donated=D

in, out    prompt and output tokens; they equal the usage log's prompt_tokens and
           completion_tokens, which is how a line joins to a harness request
kv         committed KV length
track      req.mamba_last_track_seqlen: the last 64-token boundary whose recurrent state decode
           tracked (None: no boundary crossed after the prompt)
cache_len  the page-aligned depth inserted for this request
protected  the depth the request already held from its prompt
donated    whether the tree kept this request's recurrent state (False: a node there had one,
           or nothing was inserted)

The reply was donated when cache_len > protected. Against the next request of the same game:
cached_tokens == cache_len means the match took the whole donation; cached_tokens at or below
protected means the re-sent text diverged inside the reply (or the state was evicted first).

Not shipped: ARC3_REPLYCACHE_DONATE. It was to exist only if the wheel shows a branch that skips
donating a finished request's final committed state. For this notebook it does not (Pennyroyal
v2.5.3 wheel, Franzen's build, sglang/srt/...):
- Cell 15 sets MAMBA_RADIX="extra_buffer" (franzen-p448s14-v1 serve.log: strategy extra_buffer,
  mamba_track_interval 64, page size 64, EAGLE 3 steps, overlap scheduling on). The suspected
  skip, managers/scheduler_components/batch_result_processor.py _mamba_lazy_spec_update (1224;
  mamba_lazy_is_insert = False at 1263, also 1106), runs only when
  runtime_context.mamba_extra_buffer_lazy_enabled() (1556), i.e. strategy "extra_buffer_lazy".
  In this mode release_kv_cache always gets is_insert=True (1153-1158).
- The non-lazy spec path records every crossing: _mamba_check_track_boundary (1295) reads
  num_correct_drafts_per_req_cpu, which _resolve_spec_v2_tokens fills on every spec v2 decode
  result (691), and _mamba_prefix_cache_update (1173) sets mamba_last_track_seqlen and _idx.
  With overlap on, the ping-pong buffer has 2 slots (mem_cache/memory_pool.py 1259) and the
  keep slot is mamba_last_track_idx (1570).
- At finish, mem_cache/unified_cache/components/mamba_component.py prepare_for_caching_req (560)
  inserts tokens up to cache_len = mamba_last_track_seqlen (568) with the keep slot's state
  (589), and cleanup_after_caching_req hands that slot to the tree (681). Keys are plain
  tokens: UnifiedTreeCore turns EAGLE bigram keys off when a MAMBA component exists
  (mem_cache/unified_cache/unified_tree_core.py 394).
- The live cache is UnifiedRadixCache: mem_cache/registry.py builds it for every default
  config, hierarchical cache included (187); nothing in the wheel constructs or subclasses
  MambaRadixCache (mem_cache/mamba_radix_cache.py). serve.log: "Tree cache initialized:
  source=default impl=UnifiedRadixCache ... hybrid_ssm=True". No run logged "mamba checkpoint
  skipped" (franzen-p448s14-v1, -rep-v1, dfranzen-m2-v3).
- The usage logs agree: in the two p448s14 runs, replies of 128+ tokens whose reasoning was
  under 32 tokens were fully reused 99 times in 100, the rest about 40%. A donation skip
  would hit the short ones too. What stays recomputed after a donated reply is the tail past
  its last tracked boundary (under 64 tokens); the 64-token page size sets that floor.
So the misses come from the harness re-sending different text, and this module ships only the
diagnostic. apply() raises if ARC3_REPLYCACHE_DONATE=1, so no arm can claim a donate fix.

One asserted text edit to the installed SGLang, only with the diagnostic on:
  srt/mem_cache/unified_radix_cache.py   cache_finished_req logs the line above.
nvfp4kv_serve.py's 0002 patch edits the same file (cache_unfinished_req); the two apply in either
order (solution/sim/test_franzen_replycache.py).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MARK = "arm replycache"
DIAG, DONATE = "ARC3_REPLYCACHE_DIAG", "ARC3_REPLYCACHE_DONATE"
LOG_OLD = ("        if is_insert and result is not None and result.last_device_node is not None:\n"
           "            req.last_node = result.last_device_node\n")
LOG_NEW = LOG_OLD + (
    f"        # {MARK} ({DIAG}=1): how deep this finished request left its reply in the tree\n"
    "        logger.info(\n"
    '            "replycache rid=%s in=%d out=%d kv=%d track=%s cache_len=%d protected=%d donated=%s",\n'
    "            req.rid, len(req.origin_input_ids), len(req.output_ids), kv_len_to_handle,\n"
    "            req.mamba_last_track_seqlen, page_aligned_len if is_insert else 0,\n"
    "            req.cache_protected_len, None if result is None else not result.mamba_exist,\n"
    "        )\n")
# git blob of solution/sim/fixtures/q38_chat_template.jinja, the template the round trip was checked on
TEMPLATE_BLOB = "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3"


def template_note(model_dir) -> str:
    path = Path(model_dir, "chat_template.jinja")
    if not path.is_file():
        return f"replycache: no {path.name} in MODEL_DIR, so the server renders history with another template"
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    same = "the one" if blob == TEMPLATE_BLOB else f"NOT {TEMPLATE_BLOB}, the one"
    return f"replycache: chat template git blob {blob}, {same} the round trip was checked on"


def patch_sglang(venv) -> Path:
    """Add the log line to the installed SGLang exactly once; a rerun of the cell is a no-op."""
    roots = sorted(Path(venv).glob("lib/python*/site-packages/sglang"))
    if len(roots) != 1:
        raise RuntimeError(f"replycache: expected one installed sglang under {venv}, found {roots}")
    path = roots[0] / "srt/mem_cache/unified_radix_cache.py"
    text = path.read_text()
    if LOG_NEW in text:
        return path
    if MARK in text or text.count(LOG_OLD) != 1:
        raise RuntimeError(f"replycache: {path.name} does not match the Pennyroyal v2.5.3 source; "
                           "this SGLang build is not the one the arm was written against")
    text = text.replace(LOG_OLD, LOG_NEW)
    compile(text, str(path), "exec")
    path.write_text(text)
    return path


def apply(venv, env, model_dir) -> None:
    if env.get(DONATE) == "1":
        raise RuntimeError(f"replycache: {DONATE}=1 has no fix behind it; the wheel donates every finished "
                           "request's last tracked state in extra_buffer mode (see franzen_overlays/replycache_serve.py)")
    if env.get(DIAG) != "1":
        print(f"replycache: {DIAG} is not 1, SGLang left unchanged", flush=True)
        return
    path = patch_sglang(venv)
    print(f"replycache: one 'replycache rid=...' line per finished request in serve.log; patched {path.name}",
          flush=True)
    print(template_note(model_dir), flush=True)

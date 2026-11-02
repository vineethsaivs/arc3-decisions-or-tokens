"""specstats: one `spec:` line per model reply in the transcript, the server's own MTP accept stats for that request.

Why (docs/research/2026-10-12-next/PLAN.md item 2). serve.log gives accept length only per decode batch, mixed across
games, so a serving gate cannot pair runs game by game and a drafter refit cannot be checked on held-out games. SGLang
already computes per-request stats in the tokenizer manager for every request (tokenizer_manager.py:2827-2900) and
returns them as sglext.spec_tokens_details when the request sets return_spec_tokens_details (protocol.py:889,
serving_chat.py:1883-1901). Three edits ask for them, keep them on _ChatCompletionResult and print them in the MODEL
RESPONSE META block, which goes to the transcript only: nothing the model sees changes, and the server does no new work.

The line: spec: accept_length=2.831 verify_ct=812 correct_drafts=1487 proposed_drafts=2436 hist=120/233/181/278
(accept_length = completion tokens / verify steps; hist = verify steps that accepted 0, 1, 2, ... drafts).
solution/analysis/specstats_counters.py reads it. Stdlib only; every edit is asserted to match once and nothing is
written unless tool_agent.py still compiles. solution/sim/test_franzen_specstats.py checks it offline.
"""
from pathlib import Path

_AGENT = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
_MARK = "overlay specstats"

_SPEC_LINE = '''def _spec_line(sglext: Any) -> str:
    """overlay specstats: the server's MTP stats for this request (sglext.spec_tokens_details), one line."""
    spec = sglext.get("spec_tokens_details") if isinstance(sglext, dict) else None
    if isinstance(spec, list):
        spec = spec[0] if spec else None
    if not isinstance(spec, dict) or not spec.get("spec_verify_ct"):
        return "\\nspec: (none)"
    hist = "/".join(str(int(x)) for x in spec.get("spec_correct_drafts_histogram") or [])
    return (
        f"\\nspec: accept_length={float(spec.get('spec_accept_length') or 0.0):.3f} "
        f"verify_ct={int(spec['spec_verify_ct'])} correct_drafts={int(spec.get('spec_num_correct_drafts') or 0)} "
        f"proposed_drafts={int(spec.get('spec_num_proposed_drafts') or 0)} hist={hist or '-'}"
    )


def _format_model_response_meta(
'''

_EDITS = [
    ('        if template_kwargs:\n            payload["chat_template_kwargs"] = template_kwargs\n',
     '        if template_kwargs:\n            payload["chat_template_kwargs"] = template_kwargs\n'
     f'        payload["return_spec_tokens_details"] = True  # {_MARK}: per-request MTP stats, transcript only\n'),
    ('    usage: dict[str, Any] | None = None\n    served_by: str = ""\n',
     f'    usage: dict[str, Any] | None = None\n    served_by: str = ""\n    sglext: dict[str, Any] | None = None  # {_MARK}\n'),
    ('            served_by=str(payload.get("provider", "") or ""),\n        )\n',
     f'            served_by=str(payload.get("provider", "") or ""),\n            sglext=payload.get("sglext"),  # {_MARK}\n'
     '        )\n'),
    ('def _format_model_response_meta(\n', _SPEC_LINE),
    ('                append_transcript(\n                    "MODEL RESPONSE META",\n                    response_meta,\n',
     '                append_transcript(\n                    "MODEL RESPONSE META",\n'
     f'                    response_meta + _spec_line(result.sglext),  # {_MARK}\n'),
]

_text = _AGENT.read_text()
if _MARK in _text:
    raise RuntimeError("specstats overlay: already applied to " + str(_AGENT))
for _old, _new in _EDITS:
    _count = _text.count(_old)
    if _count != 1:
        raise RuntimeError(f"specstats overlay: anchor matches {_count} times in {_AGENT.name}, expected 1: "
                           f"{_old.strip()[:70]!r}")
    _text = _text.replace(_old, _new)
compile(_text, str(_AGENT), "exec")
_AGENT.write_text(_text)
print("arm overlay specstats: one 'spec:' line of per-request MTP stats per reply in the transcript")

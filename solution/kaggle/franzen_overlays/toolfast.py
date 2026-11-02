"""toolfast: stop re-encoding and re-parsing the whole game history on every action.

Measured (docs/research/2026-10-11-improve/field-web.md section 3 and its verifier): after every executed action
the solver rewrites the whole history with json.dumps(indent=2), CPython's pure-Python encoder, and every
action() reply re-parses that file, int-converts every cell and re-renders every frame's ascii. At 500 entries
that is about 0.5 s per action on a Mac (Kaggle about 2x slower), all while the game holds its priority-gate
slot: in-batch gaps sum to 2,092 to 3,410 s per 25-game replica and running requests fall from about 13.4 to
12.2 to 12.5 of 14 late in the window.

Two host-side edits, nothing in the sandbox child:
(a) runtime_state.py: write_runtime_state encodes only entries appended since its last write (identity prefix)
    as compact JSON and remembers what a parse of the file would return; load_runtime_state returns that from
    memory while the file is the one this process last wrote ((size, mtime_ns) unchanged), else parses as before.
    An entry that does not survive a JSON round trip unchanged is remembered as its parsed copy, so load returns
    exactly what parsing the file returns.
(b) tool_agent.py: _ascii_frame_view_payload computes a Frame's ascii and shape once per Frame object (an
    identity memo whose entry is dropped when the frame is freed); the payload, grid included, is unchanged, so
    the bytes piped to the sandbox are byte-identical to the shipped code.

Run by build_franzen_arms.py after the other harness overlays, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib
only. Every edit is an exact replacement asserted to match once, and nothing is written unless both files still
compile. solution/sim/test_franzen_toolfast.py checks byte identity against the shipped functions offline.
"""
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"
_STATE = _inf / "agent" / "runtime_state.py"
_AGENT = _inf / "agent" / "tool_agent.py"
_MARK = "overlay toolfast"

_EDITS = {
    _STATE: [
        ("import json\n", "import json\nimport os  # overlay toolfast\n"),
        ("def load_runtime_state(path: Path) -> tuple[Frame | None, list[HistoryEntry]]:\n"
         "    if not path.exists():\n"
         "        return None, []\n"
         "    payload = json.loads(path.read_text(encoding=\"utf-8\"))\n",
         "# overlay toolfast: per state file, what its last write holds and what a parse of it returns\n"
         "_TOOLFAST_CACHE: dict[str, dict[str, Any]] = {}\n"
         "\n"
         "\n"
         "def _toolfast_key(path: Path) -> tuple[int, int] | None:\n"
         "    try:\n"
         "        stat = os.stat(path)\n"
         "    except OSError:\n"
         "        return None\n"
         "    return (stat.st_size, stat.st_mtime_ns)\n"
         "\n"
         "\n"
         "def load_runtime_state(path: Path) -> tuple[Frame | None, list[HistoryEntry]]:\n"
         "    if not path.exists():\n"
         "        return None, []\n"
         "    # overlay toolfast: the file this process last wrote comes back from memory, as a parse would return it\n"
         "    cached = _TOOLFAST_CACHE.get(str(path))\n"
         "    if cached is not None and cached[\"key\"] is not None and cached[\"key\"] == _toolfast_key(path):\n"
         "        return cached[\"cur\"], [entry for entry in cached[\"entries\"] if entry is not None]\n"
         "    payload = json.loads(path.read_text(encoding=\"utf-8\"))\n"),
        ("    path.parent.mkdir(parents=True, exist_ok=True)\n"
         "    payload = {\n"
         "        \"current_frame\": frame_to_payload(current_frame),\n"
         "        \"history\": [history_entry_to_payload(entry) for entry in history],\n"
         "    }\n"
         "    tmp_path = path.with_suffix(f\"{path.suffix}.tmp\")\n"
         "    tmp_path.write_text(json.dumps(payload, indent=2), encoding=\"utf-8\")\n"
         "    tmp_path.replace(path)\n",
         "    path.parent.mkdir(parents=True, exist_ok=True)\n"
         "    # overlay toolfast: encode only entries appended since the last write (compact JSON, same content)\n"
         "    cached = _TOOLFAST_CACHE.get(str(path)) or {\"raw\": [], \"frags\": [], \"entries\": []}\n"
         "    keep = 0\n"
         "    while keep < min(len(cached[\"raw\"]), len(history)) and cached[\"raw\"][keep] is history[keep]:\n"
         "        keep += 1\n"
         "    raw, frags, entries = cached[\"raw\"][:keep], cached[\"frags\"][:keep], cached[\"entries\"][:keep]\n"
         "    for entry in history[keep:]:\n"
         "        frag = json.dumps(history_entry_to_payload(entry), separators=(\",\", \":\"))\n"
         "        parsed = history_entry_from_payload(json.loads(frag))\n"
         "        raw.append(entry)\n"
         "        frags.append(frag)\n"
         "        entries.append(entry if parsed is not None and parsed == entry else parsed)\n"
         "    current_text = json.dumps(frame_to_payload(current_frame), separators=(\",\", \":\"))\n"
         "    tmp_path = path.with_suffix(f\"{path.suffix}.tmp\")\n"
         "    tmp_path.write_text('{\"current_frame\":' + current_text + ',\"history\":[' + \",\".join(frags) + \"]}\",\n"
         "                        encoding=\"utf-8\")\n"
         "    tmp_path.replace(path)\n"
         "    _TOOLFAST_CACHE[str(path)] = {\"raw\": raw, \"frags\": frags, \"entries\": entries,\n"
         "                                  \"cur\": frame_from_payload(json.loads(current_text)),\n"
         "                                  \"key\": _toolfast_key(path)}\n"),
    ],
    _AGENT: [
        ("import threading\nimport json\n", "import threading\nimport json\nimport weakref  # overlay toolfast\n"),
        ("def _ascii_frame_view_payload(frame: Frame | None) -> dict[str, Any] | None:\n"
         "    view = _to_ascii_frame_view(frame)\n"
         "    if view is None:\n"
         "        return None\n"
         "    return {\n"
         "        \"ascii\": view.ascii,\n"
         "        \"step\": view.step,\n"
         "        \"level\": view.level,\n"
         "        \"shape\": [int(view.shape[0]), int(view.shape[1])],\n"
         "        \"grid\": [list(row) for row in frame.grid],\n"
         "    }\n",
         "# overlay toolfast: id(frame) -> (weakref to the frame, ascii, shape); an entry leaves when its frame is freed\n"
         "_TOOLFAST_ASCII: dict[int, tuple[Any, str, list[int]]] = {}\n"
         "\n"
         "\n"
         "def _ascii_frame_view_payload(frame: Frame | None) -> dict[str, Any] | None:\n"
         "    if frame is None:\n"
         "        return None\n"
         "    key = id(frame)\n"
         "    hit = _TOOLFAST_ASCII.get(key)\n"
         "    if hit is None or hit[0]() is not frame:\n"
         "        view = _to_ascii_frame_view(frame)\n"
         "        hit = (weakref.ref(frame, lambda _ref, _key=key: _TOOLFAST_ASCII.pop(_key, None)),\n"
         "               view.ascii, [int(view.shape[0]), int(view.shape[1])])\n"
         "        _TOOLFAST_ASCII[key] = hit\n"
         "    return {\n"
         "        \"ascii\": hit[1],\n"
         "        \"step\": frame.step,\n"
         "        \"level\": frame.level,\n"
         "        \"shape\": list(hit[2]),\n"
         "        \"grid\": [list(row) for row in frame.grid],\n"
         "    }\n"),
    ],
}

_texts = {path: path.read_text() for path in _EDITS}
for _path, _text in _texts.items():
    if _MARK in _text:
        raise RuntimeError("toolfast overlay: already applied to " + str(_path))
for _path, _edits in _EDITS.items():
    for _old, _new in _edits:
        _count = _texts[_path].count(_old)
        if _count != 1:
            raise RuntimeError(f"toolfast overlay: anchor matches {_count} times in {_path.name}, expected 1: "
                               f"{_old.strip()[:70]!r}")
        _texts[_path] = _texts[_path].replace(_old, _new)
for _path, _text in _texts.items():
    compile(_text, str(_path), "exec")
for _path, _text in _texts.items():
    _path.write_text(_text)
print("arm overlay toolfast: incremental runtime-state write, cached load, per-frame ascii memo")

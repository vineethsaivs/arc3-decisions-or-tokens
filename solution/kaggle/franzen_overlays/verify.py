"""verify: one-call checks of the model's own rule and goal guess against this level's recorded history.

Every public harness that gains a lot with fixed weights makes a theory cheap to falsify against the log: Twin
(arXiv 2608.14490) replays the model's world model over every logged transition and treats every logged board as a
certified non-goal (93.3 with the harness, 61.1 without, same model). Ours asks for it in a prompt line and the model
does it on about 6% of cleared levels (53 replay blocks in 8,437 code blocks over 3 seg8 runs), while it writes a
simulator in 20 to 35% of level plays and checks a goal guess against history 0.3 times per run. A hand-written replay
costs 1.2K to 2K characters of code. Two sandbox helpers make it one call, at no action cost:

- `check_model(step, observe=None, start=None)`: replays the current level's recorded transitions, newest first
  (RESETs, deaths and level changes skipped), through the model's rule. With `observe`, it compares
  `step(observe(before), action) == observe(after)`, so the model's own state reader and simulator plug in (a lambda
  adapts another signature); without it, `step(frame, action)` returns a dict {(row, col): char} of only the cells it
  predicts (or a full ascii board, or None to abstain). It reports matches, skips and the first mismatches, and says
  when the same board and action led to different results on this level (the board alone cannot determine them).
- `check_goal(is_goal, start=None)`: runs a goal test on every recorded board of this level; the level ended on none
  of them, so a test that is true on one is wrong or incomplete (or that board needed one more action).

Safety, from the 11-01 skeptic read: `action` is disabled while the model's functions run (18 of 166 simulator-named
functions in one build call action() themselves), only Exception is caught (refusals and SnippetTimeout propagate),
each call stops after 8 s or, under tool2, 2 s before the snippet's own limit; output stays short. The system prompt
documents both with an adapter example, the opener's tool line names them, unstick2's replay bullet and stall check
point to them.

Run by build_franzen_arms.py after Franzen's harness patch, as exec(code, {'BUNDLE_DIR': BUNDLE_DIR}). Stdlib only.
Every edit is asserted to match once; nothing is written unless all three files still compile, and the sandbox child
source is compiled on its own as well.
"""
import textwrap
from pathlib import Path

_inf = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference"  # noqa: F821
_MARK = "overlay verify"

_CHILD = '''        # overlay verify: replay this level's log through the model's own rule; test a goal guess on its boards
        def _verify_budget():
            limit = 8.0
            t2 = globals().get("_T2")
            if t2 is not None and globals().get("_t2_used") is not None and t2.get("start"):
                limit = min(limit, max(0.5, t2["limit"] - globals()["_t2_used"]() - 2.0))
            return limit

        def _verify_level(start):
            hist = runtime_globals.get("history") or []
            starts = runtime_globals.get("level_starts") or [0]
            i0 = starts[-1] if start is None else max(0, int(start))
            return hist, i0

        class _verify_no_action:
            def __enter__(self):
                self.saved = runtime_globals.get("action")

                def _blocked(*args, **kwargs):
                    raise RuntimeError("action() is disabled inside check_model/check_goal; pass a pure function")

                runtime_globals["action"] = _blocked
                return self

            def __exit__(self, *exc):
                runtime_globals["action"] = self.saved
                return False

        def _verify_short(value, n=160):
            text = repr(value)
            return text if len(text) <= n else text[:n] + "..."

        def _verify_cells(pred, after):
            rows = (after.ascii or "").splitlines()
            grid = getattr(after, "_grid", None) or []
            bad, claimed = [], 0
            for key, want in pred.items():
                try:
                    r, c = int(key[0]), int(key[1])
                except Exception:
                    continue
                claimed += 1
                if isinstance(want, str):
                    got = rows[r][c] if 0 <= r < len(rows) and 0 <= c < len(rows[r]) else None
                else:
                    got = grid[r][c] if 0 <= r < len(grid) and 0 <= c < len(grid[r]) else None
                if got != want:
                    bad.append((r, c, want, got))
            return claimed, bad

        def check_model(step, observe=None, start=None, show=3):
            import time as _vt
            hist, i0 = _verify_level(start)
            skipped = {"RESET": 0, "death": 0, "level change": 0}
            todo = []
            for i in range(max(i0 + 1, 1), len(hist)):
                entry, before, after = hist[i], hist[i - 1].frame, hist[i].frame
                act = str(entry.action or "")
                if before is None or after is None or not act:
                    continue
                if act.strip().upper() == "RESET" or entry.result.get("automatic"):
                    skipped["RESET"] += 1
                elif before.level != after.level or entry.result.get("level_completed"):
                    skipped["level change"] += 1
                elif entry.result.get("game_over"):
                    skipped["death"] += 1
                else:
                    todo.append((i, before, act, after))
            seen, contradictions = {}, 0
            for i, before, act, after in todo:
                key = (before.ascii, act)
                if key in seen and seen[key] != after.ascii:
                    contradictions += 1
                seen.setdefault(key, after.ascii)
            limit, t0 = _verify_budget(), _vt.monotonic()
            checked = matched = unclaimed = 0
            mismatches, first_error, stopped = [], None, False
            with _verify_no_action():
                for i, before, act, after in reversed(todo):
                    if _vt.monotonic() - t0 > limit:
                        stopped = True
                        break
                    checked += 1
                    try:
                        if observe is not None:
                            pred, want = step(observe(before), act), observe(after)
                            if pred == want:
                                matched += 1
                            else:
                                if isinstance(pred, dict) and isinstance(want, dict):
                                    diff = [k for k in set(pred) | set(want) if pred.get(k) != want.get(k)][:6]
                                    detail = "; ".join(f"{k!r}: predicted {_verify_short(pred.get(k), 40)} got "
                                                       f"{_verify_short(want.get(k), 40)}" for k in diff)
                                else:
                                    detail = f"predicted {_verify_short(pred)} got {_verify_short(want)}"
                                mismatches.append((i, act, detail))
                            continue
                        pred = step(before, act)
                        if pred is None:
                            unclaimed += 1
                            checked -= 1
                            continue
                        if isinstance(pred, str):
                            p_rows, a_rows = pred.splitlines(), (after.ascii or "").splitlines()
                            bad = [(r, c, pc, ac) for r, (pr, ar) in enumerate(zip(p_rows, a_rows))
                                   for c, (pc, ac) in enumerate(zip(pr, ar)) if pc != ac]
                            if len(p_rows) != len(a_rows):
                                bad.append(("rows", len(p_rows), len(a_rows), None))
                        elif isinstance(pred, dict):
                            _, bad = _verify_cells(pred, after)
                        else:
                            bad = [("type", type(pred).__name__, "dict or ascii str", None)]
                        if bad:
                            detail = f"{len(bad)} cells differ: " + ", ".join(
                                f"({b[0]},{b[1]}) predicted {b[2]!r} got {b[3]!r}" for b in bad[:6])
                            mismatches.append((i, act, detail))
                        else:
                            matched += 1
                    except Exception as exc:
                        if first_error is None:
                            first_error = f"history[{i}] {act!r}: {type(exc).__name__}: {str(exc)[:100]}"
                        mismatches.append((i, act, f"{type(exc).__name__}: {str(exc)[:100]}"))
            skips = ", ".join(f"{v} {k}" for k, v in skipped.items() if v) or "none"
            head = (f"check_model: {matched} of {checked} checked transitions on this level match "
                    f"(skipped: {skips}; abstained: {unclaimed}; newest first"
                    + (f", stopped at the time limit with {len(todo) - checked - unclaimed} older ones unchecked)"
                       if stopped else ", all covered)"))
            lines = [head]
            if mismatches:
                lines.append("First mismatches (newest first): " + " | ".join(
                    f"history[{i}] {act!r}: {d}" for i, act, d in mismatches[:show]))
            if contradictions:
                lines.append(f"Note: on this level {contradictions} recorded (board, action) pair(s) led to different "
                             "results, so the board alone does not determine them (a hidden counter, timer, facing "
                             "or position off screen).")
            print(" ".join(lines) if len(" ".join(lines)) < 1400 else " ".join(lines)[:1400] + " ...")
            return {"checked": checked, "matched": matched, "skipped": skipped, "abstained": unclaimed,
                    "mismatches": [m[0] for m in mismatches], "contradictions": contradictions}

        def check_goal(is_goal, start=None):
            import time as _vt
            hist, i0 = _verify_level(start)
            cur = runtime_globals.get("current_frame")
            level = cur.level if cur is not None else None
            boards = [(j, hist[j].frame) for j in range(i0, len(hist))
                      if hist[j].frame is not None and (level is None or hist[j].frame.level == level)]
            if cur is not None and not boards:  # history[-1].frame is the current board when history exists
                boards.append((len(hist) - 1 if hist else 0, cur))
            limit, t0 = _verify_budget(), _vt.monotonic()
            hits, errors, done = [], [], 0
            with _verify_no_action():
                for j, frame in boards:
                    if _vt.monotonic() - t0 > limit:
                        break
                    done += 1
                    try:
                        if is_goal(frame):
                            hits.append(j)
                    except Exception as exc:
                        errors.append(f"history[{j}]: {type(exc).__name__}: {str(exc)[:100]}")
            if hits:
                shown = ", ".join(str(j) for j in hits[:8])
                msg = (f"check_goal: true on {len(hits)} of {done} recorded boards of this level (history {shown}). "
                       "The level did not end on any of them, so this goal test is wrong or incomplete, or those "
                       "boards still needed one more action.")
            else:
                msg = f"check_goal: false on all {done} recorded boards of this level, consistent so far."
            if errors:
                msg += f" {len(errors)} raised, first: {errors[0]}"
            if done < len(boards):
                msg += f" Stopped at the time limit with {len(boards) - done} boards unchecked."
            print(msg)
            return hits

'''

_SB = _inf / "agent" / "python_tool_sandbox.py"
_EDITS = {
    _SB: [
        ("        def _refresh_state(state_payload):\n", _CHILD + "        def _refresh_state(state_payload):\n"),
        ('            runtime_globals["frame_diff"] = frame_diff\n',
         '            runtime_globals["frame_diff"] = frame_diff\n'
         '            runtime_globals["check_model"] = check_model  # overlay verify\n'
         '            runtime_globals["check_goal"] = check_goal\n'),
    ],
    _inf / "agent" / "prompts.py": [
        ('    "- Examples: `frame_diff()` compares the latest action',
         '    "- `check_model(step, observe=None)` replays every recorded transition of the current level (RESETs, deaths '
         'and level changes skipped, newest first) through your rule and reports how many match and the first '
         'mismatches, at no action cost. With `observe`, it checks `step(observe(before_frame), action) == '
         'observe(after_frame)`: pass the function you already use to read your state from a frame, and your '
         'simulator. `action` is the recorded string such as \'UP\' or \'MOUSE(row=3, col=4)\', so wrap a simulator with '
         'another signature in a lambda, for example `check_model(lambda s, a: sim(s, KEYS[a]), observe=parse)`. '
         'Without `observe`, `step(frame, action)` may return a dict {(row, col): char} of only the cells it predicts, '
         'or None to skip a transition. A mismatch names the transition your rule gets wrong: fix that rule before '
         'planning with it. It also says when the same board and action led to different results (hidden state).\\n"\n'
         '    "- `check_goal(is_goal)` runs your goal test on every recorded board of this level. The level never ended '
         'on any of them, so a test that is true on one is wrong or incomplete, or that board still needed one more '
         'action.\\n"\n'
         '    "- Examples: `frame_diff()` compares the latest action'),
        ('    "- Before running a long plan from a simulator or rule you wrote, replay this level\'s "\n'
         '    "recorded `transitions` through it in code and fix any mismatch first.\\n"\n',
         '    "- Before running a long plan from a simulator or rule you wrote, run `check_model` on it "\n'
         '    "(it replays this level\'s recorded transitions) and fix any mismatch first; test a goal guess with "\n'
         '    "`check_goal` before planning toward it.\\n"\n'),
    ],
    _inf / "agent" / "tool_agent.py": [
        ('                "RESET) wastes actions and teaches nothing.")\n',
         '                "RESET) wastes actions and teaches nothing. If you have a simulator, run check_model on it "\n'
         '                "before a long search; test a goal guess with check_goal.")  # overlay verify\n'),
        ('`frame_diff(before, after)`, and `action(actions)`.",',
         '`frame_diff(before, after)`, `check_model(step, observe)`, `check_goal(is_goal)`, and `action(actions)`.",'),
    ],
}

_plan = []
for _path, _edits in _EDITS.items():
    _text = _path.read_text()
    if _MARK in _text or "def check_model(" in _text:
        raise RuntimeError("verify overlay: already applied to " + str(_path))
    for _old, _new in _edits:
        _count = _text.count(_old)
        if _count != 1:
            raise RuntimeError(f"verify overlay: anchor matches {_count} times in {_path.name}, expected 1: "
                               f"{_old.strip()[:70]!r}")
        _text = _text.replace(_old, _new)
    compile(_text, str(_path), "exec")
    if _path == _SB:  # the child runs from a string inside this file: compile that source on its own too
        _start = _text.index('_SANDBOX_BOOTSTRAP = textwrap.dedent(\n    r"""') + len('_SANDBOX_BOOTSTRAP = textwrap.dedent(\n    r"""')
        compile(textwrap.dedent(_text[_start:_text.index('"""', _start)]), "<sandbox child>", "exec")
    _plan.append((_path, _text))
for _path, _text in _plan:
    _path.write_text(_text)
print("arm overlay verify: check_model and check_goal replay this level's log through the model's own rule and goal test")

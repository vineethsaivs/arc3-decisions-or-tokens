"""franzen_overlays/d3draft_serve.py on a synthetic drafter: the 29 refit tensors land byte for byte, everything
else and the header stay identical, the view's link is repointed, a rerun is a no-op, and a wrong shape, a wrong
sha256 or a non-symlink refuse to splice.

    python3 solution/sim/test_franzen_d3draft.py      # stdlib only
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "solution/kaggle/franzen_overlays/d3draft_serve.py"


def write_st(path: Path, tensors: dict[str, tuple[list[int], bytes]]) -> None:
    hdr, off, blobs = {}, 0, []
    for name, (shape, blob) in tensors.items():
        hdr[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + len(blob)]}
        off += len(blob)
        blobs.append(blob)
    hdr["__metadata__"] = {"format": "pt"}
    text = json.dumps(hdr).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def read_tensor(path: Path, ns: dict, name: str) -> bytes:
    hdr, base = ns["header"](path)
    a, b = hdr[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        return f.read(b - a)


def main() -> None:
    ns: dict = {}
    exec(compile(OVERLAY.read_text(), str(OVERLAY), "exec"), ns)
    with tempfile.TemporaryDirectory(prefix="d3draft-") as tmp:
        tmp = Path(tmp)
        src_dir, view, work = tmp / "input", tmp / "view", tmp / "work"
        src_dir.mkdir()
        view.mkdir()
        mtp = {f"mtp.t{i:02d}.weight": ([4, i + 1], bytes([i]) * (8 * (i + 1))) for i in range(29)}
        others = {"lm_head.weight": ([4, 4], b"L" * 32), "model.language_model.embed_tokens.weight": ([4, 2], b"E" * 16)}
        original = src_dir / "mtp-dense.safetensors"
        write_st(original, {**others, **mtp})
        weight_map = {name: "mtp-dense.safetensors" for name in {**others, **mtp}}
        (src_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        for name in ("mtp-dense.safetensors", "model.safetensors.index.json"):
            (view / name).symlink_to(src_dir / name)
        refit = tmp / "refit.safetensors"
        write_st(refit, {name: (shape, bytes([200 + i % 50]) * len(blob)) for i, (name, (shape, blob)) in enumerate(mtp.items())})
        before = original.read_bytes()

        for bad, message in ((dict(expect_sha256="0" * 64), "sha256"),):
            try:
                ns["splice"](view, refit, work, **bad)
            except RuntimeError as exc:
                assert message in str(exc), exc
            else:
                raise AssertionError("a wrong sha256 spliced")

        out = ns["splice"](view, refit, work, expect_sha256=None)
        link = view / "mtp-dense.safetensors"
        assert link.is_symlink() and link.resolve() == out.resolve()
        assert original.read_bytes() == before, "the input file changed"
        assert ns["header"](out) == ns["header"](original), "header changed"
        for name in mtp:
            assert read_tensor(out, ns, name) == read_tensor(refit, ns, name), name
        for name in others:
            assert read_tensor(out, ns, name) == read_tensor(original, ns, name), name
        assert out.stat().st_size == original.stat().st_size
        print("ok   29 tensors spliced byte for byte; header, other tensors and the input file unchanged")

        mtime = out.stat().st_mtime_ns
        assert ns["splice"](view, refit, work, expect_sha256=None) == out and out.stat().st_mtime_ns == mtime
        print("ok   a rerun on a spliced view is a no-op")

        link.unlink()
        link.symlink_to(src_dir / "mtp-dense.safetensors")
        bad_refit = tmp / "bad.safetensors"
        bad = dict(mtp)
        first = next(iter(bad))
        bad[first] = ([2, 2], bad[first][1])
        write_st(bad_refit, bad)
        try:
            ns["splice"](view, bad_refit, tmp / "work2", expect_sha256=None)
        except RuntimeError as exc:
            assert "dtype, shape or size" in str(exc), exc
        else:
            raise AssertionError("a wrong shape spliced")
        os.remove(link)
        link.write_bytes(original.read_bytes())
        try:
            ns["splice"](view, refit, tmp / "work3", expect_sha256=None)
        except RuntimeError as exc:
            assert "not the draft view's symlink" in str(exc), exc
        else:
            raise AssertionError("a non-symlink was replaced")
        print("ok   a wrong shape, a wrong sha256 and a non-symlink refuse to splice")


if __name__ == "__main__":
    main()

"""Build Franzen's patched harness tree for the offline overlay tests, once per patch.

The Tufa source bundle is vendored at solution/sim/fixtures/franzen-bundle (a copy of Kaggle
dataset dfranzen/taaf-kaggle-source-bundle-copy v1) and the harness patch is cell 5 of the fork
notebook, so the tree no longer lives only in a scratch directory that a cleanup can delete
(lost 2026-10-02). Tests copy the tree and never edit it. FRANZEN_TREE overrides the location.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "solution/sim/fixtures/franzen-bundle"
NOTEBOOK = REPO / "solution/kaggle/generated/candidates-fork-franzen/submission.ipynb"


def harness_patch() -> str:
    cell = json.loads(NOTEBOOK.read_text())["cells"][5]
    source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
    head, body = source.split("\n", 1)
    assert head.startswith("%%writefile"), head
    return body if body.endswith("\n") else body + "\n"


def pristine_tree() -> Path:
    if os.environ.get("FRANZEN_TREE"):
        return Path(os.environ["FRANZEN_TREE"])
    patch = harness_patch()
    out = Path.home() / ".cache/arc3" / f"franzen-tree-{hashlib.sha256(patch.encode()).hexdigest()[:12]}"
    if (out / ".built").is_file():
        return out
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(BUNDLE, tmp)
    patch_file = tmp / "harness-changes.patch"
    patch_file.write_text(patch)
    subprocess.run(["git", "apply", "--include=ARC3-Inference/*", str(patch_file)],
                   cwd=tmp / "src", check=True)
    patch_file.unlink()
    (tmp / ".built").write_text("ok\n")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


if __name__ == "__main__":
    tree = pristine_tree()
    assert (tree / "src/ARC3-Inference/inference/agent/priority_scheduler.py").is_file()
    print(tree)


WHEEL_NAME = "sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl"
WHEEL_SHA256 = "d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225"


def pennyroyal_wheel() -> Path:
    """The 25 MB Pennyroyal SGLang wheel Franzen serves with, cached under ~/.cache/arc3."""
    out = Path.home() / ".cache/arc3/pennyroyal-v253" / WHEEL_NAME
    if out.is_file() and hashlib.sha256(out.read_bytes()).hexdigest() == WHEEL_SHA256:
        return out
    import urllib.parse
    import urllib.request
    token_file = REPO / "ARC-AGI-3-Kaggle-Starter/.kaggle/access_token"
    if not token_file.is_file():
        token_file = Path.home() / ".kaggle/access_token"
    url = ("https://www.kaggle.com/api/v1/datasets/download/dfranzen/pennyroyal-v253/"
           + urllib.parse.quote(f"wheels/{WHEEL_NAME}", safe=""))
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token_file.read_text().strip()}"})
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    if data[:2] != b"PK":  # some API paths wrap a single file in a zip of its own
        raise RuntimeError(f"unexpected download for {WHEEL_NAME}: {data[:80]!r}")
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if WHEEL_NAME in names and len(names) == 1:
            data = archive.read(WHEEL_NAME)
    if hashlib.sha256(data).hexdigest() != WHEEL_SHA256:
        raise RuntimeError("Pennyroyal wheel hash mismatch")
    out.write_bytes(data)
    return out

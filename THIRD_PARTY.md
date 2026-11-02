# Third-party material

This is not legal advice. Our own files are licensed as stated in `LICENSE` (paper and data CC BY 4.0; code under CC BY 4.0 or MIT-0, your choice). Nothing below is covered by that grant. The paper's table is from 2026-10-13; the rows for Lesilva,
REAP, Olympie and lovedheart were checked against files in this tree on 2026-11-02.

- Franzen notebook and patch (dfranzen, Milestone 2): his repository is Apache-2.0; the Kaggle notebook is public and our
  fork's first cell says "Apache 2.0 by default". Vendored here under its own terms. His two Kaggle datasets show licence
  "Unknown". Credit him by name wherever the fork is described.
- Tufa Duck harness (Tufa Labs, ARC3-Inference): no LICENSE file at the repository root or in ARC3-Inference; the
  pyproject classifier says MIT, which is a claim and not a grant. Not relicensable by us. Credit Tufa Labs.
- Qwen3.8-Flash-Next weights: Qwen Community License 1.0 (MIT-style with conditions: keep the notice; field-of-use
  limits for large services). Not an OSI-approved text. Prize eligibility is unanswered by the hosts; we claim nothing.
- Intel W4A16 AutoRound checkpoint: derivative of the Qwen weights, same licence (qwen-community-1.0).
- albucino drafter (MTP): qwen-community-1.0.
- ursuciprian D3 refit tensors: qwen-community-1.0 (training tools Apache-2.0). This kit does not contain the tensors.
- Lesilva NVFP4 KV patches (github.com/Lesilva/qwen38-flashnext-nvfp4-kv, revision 540722120a): Apache-2.0, "Copyright
  2026 Lesilva". solution/kaggle/franzen_overlays/nvfp4kv/ holds his two patches, LICENSE, NOTICE and PROVENANCE.json
  unchanged. They stay Apache-2.0 with that LICENSE and NOTICE, whichever option is chosen for our own lines. The NOTICE
  says they modify SGLang and Pennyroyal sources (Apache-2.0). Only the wrapper nvfp4kv_serve.py is ours. This arm is
  not part of the shipped 40.98 configuration.
- REAP keep-448 checkpoint (lee-chang-93/Qwen3.8-Flash-Next-NVFP4-REAP-k448; REAP method by Cerebras): checkpoint under the
  NVIDIA Open Model License plus the Qwen Community License 1.0; the REAP method repository is Apache-2.0. We ship no
  weights. solution/kaggle/franzen_overlays/p448_serve.py embeds a table of the 64 dropped expert ids per layer (48
  layers), recovered by matching the checkpoint's router rows to the Intel checkpoint's. Whether a list of ids falls under
  those licences is not settled (our reading: facts about which experts were kept). Credit lee-chang-93 and REAP.
- arc-hot-map-2026-10-12.json (solution/kaggle/generated): not derived from the REAP checkpoint. It is a bitmap of token
  ids that our own runs generated (seven runs, 41.6M tokens) over the Qwen tokenizer's vocabulary, with checksums; no
  weights and no text. Ours; the vocabulary it indexes is Qwen's.
- Olympie fork (Gabriel Olympie, patch 0004 low-M BF16 GEMM, github.com/gabrielolympie/sglang-flashnext-sm120): no licence
  file found. The vendored solution/sim/fixtures/pennyroyal-v253/environ.py names the patch and holds its flag only;
  the patch itself ships inside Franzen's wheel, which declares Apache-2.0. The fork's own terms are unknown. Not ours.
- lovedheart sglang fork (QSA pending-ring fix, commit 427e9380ad): Apache-2.0. qsaring_serve.py ports it as asserted edits
  and keeps that licence with its notice, whichever option is chosen for our own lines.
- shiiin9 D-prime priority formula: source notebook is public; its licence was not checked or read. The vendored copy is
  docs/research/2026-10-04-climb/evidence/kernels/shiiin9_form_priority_dprime.py (252 lines). Not ours to relicense.
- SGLang, the Pennyroyal build and the Ratsimbazafy fork: Apache-2.0. The vendored pennyroyal-v253 files keep their LICENSE.
- ARC-AGI repository (arcprize): MIT.
- No model weights, Kaggle datasets or the 288 MB pickles are in this kit.

## Not included in this repository (licence unread or unsettled; audit of 2026-11-02)

- The Tufa Duck bundle and Franzen's fork of the notebook (the vendored `franzen-bundle` and `candidates-fork-franzen`), shiiin9's
  D-prime file and the lee-chang-93 expert-id table (`p448_serve.py`). The overlay tests and some figure scripts that build
  against those files (`fig_glyph`, `fig_scroll`, `fig_clock`, `fig_gates`, most `test_franzen_*`) therefore do not run from this
  repository alone; get the upstream files from their authors and read their licences. `build_franzen_arms.py` reads the D-prime file
  from a path it does not find here.
- Credit: Daniel Franzen (notebook and harness patch, Apache-2.0 in his repository), Tufa Labs (Duck, RE-ARC-3), shiiin9 (D-prime),
  lee-chang-93 (REAP keep-448 checkpoint), J. Pezzulli (Pennyroyal), Intel, albucino, ursuciprian, lovedheart, Lesilva, Gabriel Olympie.

# mab776/Strata - our fork of Niko1221/Strata

The Strata engine and server that serve Qwen3.8-Flash-Next on our box (Portainer stack 67 `strata-coder` on :8319, and
stack 76 `strata-gpu1`, the GPU 1 side-load), and since 2026-10-27 DeepSeek-V4.1-Flash-1017 (ISTA IQ2_XXS) through the
tree's own DeepSeek engine `strata-dsv41` (stack 79 `deepseek-v41-1017` on :8330). The upstream README.md is upstream's; this file is ours.

## Rule: keep the diff tiny

Since 2026-10-26 the fork carries only what upstream cannot give us through a config key (Marc: "4-9 % slower is a lot
better than 8.5k lines to maintain"). Speed work goes upstream or into the config `env`, not into this commit. The
earlier prompt-path port (w4a4x2, in-place experts, fused kernels, a vendored llama.cpp MMQ copy: ~1,000 own lines +
7,400 copied) was dropped at 0.2.6; it lives on in tag `mab776-0.2.6` if it is ever wanted again (measured 10-26: +4 %
prefill at 8k, +9 % at 32k / 131k, decode equal).

## Branches

| Branch | What it is |
| --- | --- |
| `main` | An exact copy of upstream `main` (Niko1221/Strata). Never committed to; synced by fast-forward. |
| `mab776-main` | **The code that runs.** `main` + exactly one commit (ours). Rebuilt onto `main` on every sync. |

Every deploy of `mab776-main` gets a tag `mab776-<upstream version>[.N]`; the tags are the history (older per-version
branches were replaced by tags of the same tips).

Sync (in the live tree, remotes `upstream` = Niko1221/Strata and `mab776` = this fork):

    git fetch upstream --tags && git push mab776 upstream/main:refs/heads/main
    git checkout -B mab776-main <upstream tag> && git cherry-pick <the previous mab776-main commit>
    # check: `git range-diff <old commit>^! HEAD^!` shows only moved context lines; build; gate; then
    git push --force-with-lease mab776 mab776-main && git tag mab776-<version> && git push mab776 mab776-<version>

## What the one commit holds

- **Abliterated model id** (`serve/server.py`, ~22 lines): config `"abliterated_model_name"` = a second model id in
  `/v1/models` that forces `experimental_speed_projection` (the refusal control vector) on; replies echo the requested
  id. Served only when the engine has a control vector loaded.
- **Vision `flash_attn`** (`serve/server.py`, 2 lines): the vision section's `"flash_attn"` passes `--flash-attn` to
  `strata-vision` (1.8 GB instead of 10.9 GB on GPU 1).
- **`data/uncensor/`**: the NVFP4-GPTQ refusal vector from sergqwer/strata-nvfp4 (see its README), loaded by the config's
  `--control-vector-scaled ... --cvec-mode project --cvec-dir per-layer`.
- **DeepSeek prompt lookup** (`src/dsv41/ds4_main.cpp` +40, `ds4_model.cu` +12; env `STRATA_DS4_LOOKUP=N`, 0 = off, we run
  14): a 3-gram match in the conversation proposes up to N tokens, checked by the engine's own verify pass (exact sampling,
  so the output distribution is unchanged). The verify buffers are allocated when lookup is on even without the drafter.
  Measured 10-26 on the copy-heavy edits: 38 -> 66 tok/s (+73 %). Upstream has no prompt lookup for DeepSeek yet (0.2.7).
- This file, `.mab776-build.sh` (Qwen engine) and `.mab776-build-dsv41.sh` (DeepSeek engine).

Upstream opt-ins we turn on through the config `env` instead of code: `STRATA_GDN_CHUNKED=1` (chunked DeltaNet
recurrence) and `STRATA_PROMPT_ATTN_IMMA=1` (INT8 prompt attention).

## Build and run

`third_party/llama.cpp` is not in git (setup downloads the pinned commit; copy it from the previous tree), nor is `.venv`.
Build: `.mab776-build.sh` (header), which puts `strata` and `libstrata_kq.so` in `engine/`; `engine/strata-vision` is
copied from the previous tree. The configs (`strata-coder.json`, `strata-gpu1.json`) and BOTH
`*.shared-settings.json` files are gitignored and live in the tree - copy all four into every new tree (without the
settings files every request runs at xhigh effort). Bench kit and gate: Portainer config repo `strata-coder/bench/`
(`gate_battery.py`, `slim_ab_1026.sh`, `gate_027.sh`).

DeepSeek: `.mab776-build-dsv41.sh` puts `strata-dsv41` in `engine-dsv41/`; config `strata-ds1017.json` (gitignored, copy it
too). Its pack `models/deepseek-v41/ista/pack-niko-2.00-v` is built by `pack_gguf.py` from the untouched ISTA files + the
vision tower `mmproj-BF16.gguf` (0.2.7's pin) with `--dspark-noise-token 128799` (the default came out 0 = a broken
drafter). The pack's `tokenizer/chat_template.jinja` is ISTA's 10-25 template with ONE edit of ours: effort `medium` = 75
(= `high`); without it every client that sends `medium` gets a 400. ISTA's original sits beside it (`.ista-1025`).
Settings that won the 10-27 overnight sweep (config, no code): `DS4_SPLIT=0,40` (every layer on the RTX PRO 5000, the
drafter and the vision tower alone on the 5070 Ti), `DS4_RING_MB=1600`, `--dspark on`, `--prefill-chunk 8192`.
A weight pool for the prompt gemm (+9 % at 30k, ~40 lines) is NOT in the fork: stashed in the Portainer repo
`deepseek-v41/stash/` as an upstream idea.

# mab776/Strata - our fork of Niko1221/Strata

The Strata engine and server that serve Qwen3.8-Flash-Next on our box (Portainer stack 67 `strata-coder` on :8319, and
stack 76 `strata-gpu1`, the GPU 1 side-load). The upstream README.md is upstream's; this file is ours.

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
- This file and `.mab776-build.sh`.

Upstream opt-ins we turn on through the config `env` instead of code: `STRATA_GDN_CHUNKED=1` (chunked DeltaNet
recurrence) and `STRATA_PROMPT_ATTN_IMMA=1` (INT8 prompt attention).

## Build and run

`third_party/llama.cpp` is not in git (setup downloads the pinned commit; copy it from the previous tree), nor is `.venv`.
Build: `.mab776-build.sh` (header), which puts `strata` and `libstrata_kq.so` in `engine/`; `engine/strata-vision` is
copied from the previous tree. The configs (`strata-coder.json`, `strata-gpu1.json`) and BOTH
`*.shared-settings.json` files are gitignored and live in the tree - copy all four into every new tree (without the
settings files every request runs at xhigh effort). Bench kit and gate: Portainer config repo `strata-coder/bench/`
(`gate_battery.py`, `slim_ab_1026.sh`).

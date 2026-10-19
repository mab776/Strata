# mab776/Strata - our fork of Niko1221/Strata

The Strata engine and server that serve Qwen3.8-Flash-Next on our box (Portainer stack 67 `strata-coder` on :8319, and
stack 76 `strata-gpu1`, the GPU 1 side-load). The upstream README.md is upstream's; this file is ours.

## Branches

| Branch | What it is |
| --- | --- |
| `main` | An exact copy of upstream `main` (Niko1221/Strata). Never committed to; synced by fast-forward. |
| `mab776-main` | **The code that runs.** `main` + exactly one commit (ours). Rebuilt onto `main` on every sync. |
| `feature/*` | Work in progress (ports, experiments, merges). Folded into the one commit of `mab776-main` when it ships. |

Every deploy of `mab776-main` gets a tag `mab776-<upstream version>[.N]` (an internal release); the tags are the history.
Older per-version branches (`sparse-host-01xx`, `mab776-nvfp4-*`, `mab776-020`) were replaced by tags of the same tips
(`mab776-0.1.39` ... `mab776-0.2.0-beta`, `archive/*` for the untagged ones).

Sync, on the box (tree `models/strata-data/src-020`, remotes `upstream` and `mab776`, push URLs disabled on purpose):

    git fetch upstream && git push git@github.com:mab776/Strata.git upstream/main:refs/heads/main
    git checkout -B mab776-main upstream/main && git cherry-pick <the previous mab776-main commit>   # resolve, build, test
    git push --force-with-lease git@github.com:mab776/Strata.git mab776-main && git tag mab776-<version> && git push ... <tag>

## What the one commit holds

- **Abliterated model id** (`serve/server.py`): config `"abliterated_model_name"` = a second model id in `/v1/models` that
  forces `experimental_speed_projection` (the refusal control vector) on; replies echo the requested id. Served only when
  the engine has a control vector loaded.
- **Vision `flash_attn`** (`serve/server.py`, `Vision.reader_args`): the vision section's `"flash_attn"` passes
  `--flash-attn` to `strata-vision`.
- **`"lan_without_api_key": true`** (`serve/server.py`): 0.2.0 refuses a non-loopback bind without an inference API key.
  This box is LAN/VPN only and its clients (OpenHands, VS Code, the LLM router, the menu planner) send none, so the key
  keeps 0.1.x's behaviour. The Host-header check (`allowed_hosts`) still applies.
- **`data/uncensor/`**: the NVFP4-GPTQ refusal vector from sergqwer/strata-nvfp4 (see its README), loaded by the config's
  `--control-vector-scaled ... --cvec-mode project --cvec-dir per-layer`.
- This file.

## Build and run

`third_party/llama.cpp` is not in git (setup downloads the pinned commit; copy it from the previous tree). Build inside
the `strata-server` image (CUDA 13.0, sm_120): `cmake --build build --target strata strata_kq`, then copy `build/strata`
and `build/libstrata_kq.so` to `engine/`. The configs (`strata-coder.json`, `strata-gpu1.json`) are gitignored and live
in the tree; the bench kit and the measurements are in the Portainer config repo (`strata-coder/bench/`).

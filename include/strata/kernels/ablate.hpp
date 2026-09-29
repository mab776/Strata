// include/strata/kernels/ablate.hpp - inference-time DIRECTIONAL ABLATION (refusal removal), the engine's side of the
// llama.cpp `replayfix-ablate` patch (mab776, 2026-09-28) that serves apetersson's Qwen3.8-Flash-Next abliteration
// adapter without its runtime-LoRA cost.
//
// The adapter is, on every attn_output / ssm_out / ple_value / ffn_down_shexp / ffn_down_exps, the rank-1 edit
// W' = (I - s d d^T) W with ONE global unit direction d and s = 1.5 - so each edited OUTPUT becomes
//
//   y <- P y = y - s d (d^T y)
//
// which is what this module applies, after each layer's mixer output (attn_output / ssm_out), after each layer's
// COMBINED FFN output (linear, so exact on routed sum + gated shared expert) and on layer 1's PLE value.  The two
// weights the rank-1 form cannot express (layer 47's attn_output and ffn_down_shexp are rank-33 edits, 92-99 %
// orthogonal to d) come from the residual LoRA file as y += B (A x):
//
//   mixer weight with a LoRA:   y <- y + B (A x)                   (no projection - as llama.cpp skips it)
//   shexp weight with a LoRA:   S <- P^-1 (S + g B (A act))       (g = the shared gate, 1 when S is ungated)
//                               P^-1 = I + s/(1-s) d d^T, so the combined FFN projection that follows leaves
//                               exactly P(routed) + g (S + B A act) - llama.cpp's result.
//
// Off unless loaded (--ablate).  A loaded ablation is switched per request through a device flag (the `cvec=0|1`
// request key, shared with the control vector), so the captured graphs are the same either way; with the flag off
// every kernel returns before touching memory and the output is bit-identical to the engine without it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels {

struct AblateLora {
    int layer = -1;
    int kind = 0;                 ///< 0 = the mixer output weight (attn_output / ssm_out), 1 = ffn_down_shexp
    int rank = 0;
    int64_t n_in = 0;
    std::vector<float> a;         ///< rank x n_in, row-major (llama.cpp lora_a)
    std::vector<float> b;         ///< n_embd x rank, row-major (llama.cpp lora_b)
};

/// Upload: `d` n_embd (normalised here), strength s (!= 1 when a shexp LoRA is loaded: P^-1 needs it), the layers
/// the projection covers [0, n_layers), and the LoRAs.  Starts ON.
bool ablate_upload(const std::vector<float>& d, float s, int64_t n_embd, int64_t n_layers,
                   const std::vector<AblateLora>& loras, std::string& err);
bool ablate_loaded();
float ablate_strength();
int ablate_lora_count();
/// A layer split: the tables on the CURRENT device too.  No-op when not loaded or already there.
bool ablate_replicate(std::string& err);
/// The per-request switch on every device holding the tables (synchronizes them when it changes).
void ablate_set_enabled(bool on);

/// y (T rows of n_embd, row stride y_ld) <- P y.  For the FFN output and the PLE value.
void ablate_project(float* y, int64_t y_ld, int64_t layer, int64_t T, void* stream);
/// The mixer output of `layer`: its LoRA when it has one (x = the output projection's input, f32 or f16 rows of
/// n_in with stride x_ld; must be non-null then), else P y.
void ablate_mixer(float* y, int64_t y_ld, const float* x, int64_t x_ld, int64_t layer, int64_t T, void* stream);
void ablate_mixer_h(float* y, int64_t y_ld, const uint16_t* x16, int64_t x_ld, int64_t layer, int64_t T, void* stream);
/// The shared expert of `layer`, before the combine: S <- P^-1 (S + g B (A act)) when the layer has a shexp LoRA,
/// else nothing.  `g` = T gate scalars when S is already gated, nullptr when the gate comes later.
bool ablate_has_shexp_lora(int64_t layer);
void ablate_shared(float* S, int64_t s_ld, const float* act, int64_t act_ld, const float* g, int64_t layer, int64_t T,
                   void* stream);
void ablate_shared_h(float* S, int64_t s_ld, const uint16_t* act16, int64_t act_ld, const float* g, int64_t layer,
                     int64_t T, void* stream);

}  // namespace strata::kernels

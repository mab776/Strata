// src/kernels/cuda/ablate.cu - see include/strata/kernels/ablate.hpp.
#include "strata/kernels/ablate.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int MAXK = 16;       // n_embd up to 4096, held in registers between the dot and the update
constexpr int MAXR = 64;       // LoRA rank

struct LoraDev { int layer, kind, rank; int64_t n_in; const float* a; const float* bt; };   // bt: rank x n_embd

float g_s = 0.0f;
int64_t g_n = 0, g_layers = 0;
bool g_on_host = false;
std::vector<float> g_dir_host;
std::vector<AblateLora> g_loras_host;
std::vector<int> g_mixer_lora, g_shexp_lora;   // per layer: index into the LoRAs, -1 = none

constexpr int kDevices = 64;
struct DevTables { float* dir = nullptr; int* on = nullptr; float* lora = nullptr; std::vector<LoraDev> loras; };
DevTables g_dev[kDevices];

int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices) d = 0;
    return d;
}

bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr) return true;
    size_t floats = 0;
    for (const AblateLora& l : g_loras_host) floats += l.a.size() + l.b.size();
    const int flag = g_on_host ? 1 : 0;
    if (cudaMalloc(&t.dir, (size_t) g_n * sizeof(float)) != cudaSuccess || cudaMalloc(&t.on, sizeof(int)) != cudaSuccess ||
        (floats && cudaMalloc(&t.lora, floats * sizeof(float)) != cudaSuccess) ||
        cudaMemcpy(t.dir, g_dir_host.data(), (size_t) g_n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.on, &flag, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "ablate: device allocation failed";
        t = DevTables{};
        return false;
    }
    float* p = t.lora;
    for (const AblateLora& l : g_loras_host) {
        std::vector<float> bt((size_t) l.rank * g_n);   // transposed, so a thread's column reads are coalesced
        for (int64_t d = 0; d < g_n; ++d)
            for (int r = 0; r < l.rank; ++r) bt[(size_t) r * g_n + d] = l.b[(size_t) d * l.rank + r];
        if (cudaMemcpy(p, l.a.data(), l.a.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(p + l.a.size(), bt.data(), bt.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "ablate: LoRA upload failed";
            return false;
        }
        t.loras.push_back({l.layer, l.kind, l.rank, l.n_in, p, p + l.a.size()});
        p += l.a.size() + l.b.size();
    }
    return true;
}

const DevTables& here() {
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) throw std::runtime_error("ablate: the tables are not on this device (ablate_replicate)");
    return t;
}

__device__ __forceinline__ float block_sum(float v, float* part) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();   // `part` may still be read by a previous reduction
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float p = threadIdx.x < WARPS ? part[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
        if (threadIdx.x == 0) part[0] = p;
    }
    __syncthreads();
    return part[0];
}

// y <- y - coef d (d . y), one block per row
__global__ void project_kernel(float* __restrict__ y, int64_t y_ld, const float* __restrict__ dir, float coef,
                               const int* __restrict__ on, int n) {
    if (*on == 0) return;
    __shared__ float part[WARPS];
    float* r = y + (int64_t) blockIdx.x * y_ld;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) { x[k] = r[d]; dot = fmaf(x[k], dir[d], dot); }
    }
    dot = block_sum(dot, part) * coef;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) r[d] = fmaf(-dot, dir[d], x[k]);
    }
}

__device__ __forceinline__ float ld_x(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float ld_x(const uint16_t* p, int64_t i) { return __half2float(__ushort_as_half(p[i])); }

// y <- y + gate * B (A x), then (post_coef != 0) y <- y - post_coef d (d . y); one block per row
template <typename XT>
__global__ void lora_kernel(float* __restrict__ y, int64_t y_ld, const XT* __restrict__ x, int64_t x_ld, int n_in,
                            const float* __restrict__ A, const float* __restrict__ Bt, int rank,
                            const float* __restrict__ g, float post_coef, const float* __restrict__ dir,
                            const int* __restrict__ on, int n) {
    if (*on == 0) return;
    __shared__ float t_r[MAXR];
    __shared__ float part[WARPS];
    const int64_t row = blockIdx.x;
    const XT* xr = x + row * x_ld;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int r = warp; r < rank; r += WARPS) {   // t = A x, one warp per rank row
        const float* a = A + (int64_t) r * n_in;
        float acc = 0.0f;
        for (int i = lane; i < n_in; i += 32) acc = fmaf(a[i], ld_x(xr, i), acc);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) t_r[r] = acc;
    }
    __syncthreads();
    const float gate = g ? g[row] : 1.0f;
    float* yr = y + row * y_ld;
    float v[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float acc = 0.0f;
            for (int r = 0; r < rank; ++r) acc = fmaf(Bt[(int64_t) r * n + d], t_r[r], acc);
            v[k] = fmaf(gate, acc, yr[d]);
            dot = fmaf(v[k], dir[d], dot);
        }
    }
    if (post_coef != 0.0f) dot = block_sum(dot, part) * post_coef;   // uniform branch
    else dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) yr[d] = fmaf(-dot, dir[d], v[k]);
    }
}

void check_launch(const char* what) {
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error(std::string(what) + ": launch failed");
}

bool covers(int64_t layer) { return g_s != 0.0f && layer >= 0 && layer < g_layers; }

template <typename XT>
void lora_launch(float* y, int64_t y_ld, const XT* x, int64_t x_ld, int idx, const float* g, float post_coef, int64_t T,
                 void* stream) {
    const DevTables& t = here();
    const LoraDev& l = t.loras[(size_t) idx];
    lora_kernel<XT><<<(unsigned) T, THREADS, 0, (cudaStream_t) stream>>>(y, y_ld, x, x_ld, (int) l.n_in, l.a, l.bt,
                                                                        l.rank, g, post_coef, t.dir, t.on, (int) g_n);
    check_launch("ablate LoRA");
}

template <typename XT>
void mixer_impl(float* y, int64_t y_ld, const XT* x, int64_t x_ld, int64_t layer, int64_t T, void* stream) {
    if (!covers(layer) || T < 1) return;
    const int li = g_mixer_lora[(size_t) layer];
    if (li >= 0) {
        if (x == nullptr) throw std::runtime_error("ablate: layer " + std::to_string(layer) + "'s mixer LoRA needs its input");
        lora_launch(y, y_ld, x, x_ld, li, nullptr, 0.0f, T, stream);
    } else {
        ablate_project(y, y_ld, layer, T, stream);
    }
}

template <typename XT>
void shared_impl(float* S, int64_t s_ld, const XT* act, int64_t act_ld, const float* g, int64_t layer, int64_t T,
                 void* stream) {
    if (!covers(layer) || T < 1) return;
    const int li = g_shexp_lora[(size_t) layer];
    if (li < 0) return;
    lora_launch(S, s_ld, act, act_ld, li, g, g_s / (g_s - 1.0f), T, stream);   // then P^-1 = I - s/(s-1) d d^T
}

}  // namespace

bool ablate_upload(const std::vector<float>& d, float s, int64_t n_embd, int64_t n_layers,
                   const std::vector<AblateLora>& loras, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "ablate: unsupported n_embd"; return false; }
    if ((int64_t) d.size() != n_embd) { err = "ablate: the direction must have n_embd elements"; return false; }
    double nrm = 0.0;
    for (float v : d) nrm += (double) v * v;
    nrm = std::sqrt(nrm);
    if (!(nrm > 0.0)) { err = "ablate: the direction is zero"; return false; }
    g_mixer_lora.assign((size_t) n_layers, -1);
    g_shexp_lora.assign((size_t) n_layers, -1);
    for (size_t i = 0; i < loras.size(); ++i) {
        const AblateLora& l = loras[i];
        if (l.layer < 0 || l.layer >= n_layers || l.rank < 1 || l.rank > MAXR || l.n_in < 1 ||
            l.a.size() != (size_t) l.rank * l.n_in || l.b.size() != (size_t) n_embd * l.rank) {
            err = "ablate: LoRA for layer " + std::to_string(l.layer) + " has a bad shape (rank <= 64)";
            return false;
        }
        std::vector<int>& slot = l.kind == 0 ? g_mixer_lora : g_shexp_lora;
        if (slot[(size_t) l.layer] >= 0) { err = "ablate: two LoRAs for one weight of layer " + std::to_string(l.layer); return false; }
        slot[(size_t) l.layer] = (int) i;
        if (l.kind == 1 && std::fabs(s - 1.0f) < 1e-3f) {
            err = "ablate: a shared-expert LoRA needs a strength != 1 (the combined projection is inverted around it)";
            return false;
        }
    }
    int prev = 0;   // a new upload replaces the old tables on every device
    cudaGetDevice(&prev);
    for (int dv = 0; dv < kDevices; ++dv) {
        if (g_dev[dv].dir == nullptr) continue;
        cudaSetDevice(dv);
        cudaDeviceSynchronize();
        cudaFree(g_dev[dv].dir);
        cudaFree(g_dev[dv].on);
        if (g_dev[dv].lora) cudaFree(g_dev[dv].lora);
        g_dev[dv] = DevTables{};
    }
    cudaSetDevice(prev);
    g_dir_host.resize((size_t) n_embd);
    for (int64_t j = 0; j < n_embd; ++j) g_dir_host[(size_t) j] = (float) (d[(size_t) j] / nrm);
    g_loras_host = loras;
    g_n = n_embd;
    g_layers = n_layers;
    g_s = s;
    g_on_host = true;
    return upload_here(err);
}

bool ablate_loaded() { return g_n > 0 && g_s != 0.0f; }
float ablate_strength() { return g_s; }
int ablate_lora_count() { return (int) g_loras_host.size(); }

bool ablate_replicate(std::string& err) { return !ablate_loaded() || upload_here(err); }

void ablate_set_enabled(bool on) {
    if (!ablate_loaded() || on == g_on_host) return;
    int prev = 0;
    cudaGetDevice(&prev);
    const int v = on ? 1 : 0;
    for (int dv = 0; dv < kDevices; ++dv) {
        if (g_dev[dv].on == nullptr) continue;
        cudaSetDevice(dv);
        cudaDeviceSynchronize();   // nothing in flight may still read the flag
        cudaMemcpy(g_dev[dv].on, &v, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(prev);
    g_on_host = on;
}

void ablate_project(float* y, int64_t y_ld, int64_t layer, int64_t T, void* stream) {
    if (!covers(layer) || T < 1) return;
    const DevTables& t = here();
    project_kernel<<<(unsigned) T, THREADS, 0, (cudaStream_t) stream>>>(y, y_ld, t.dir, g_s, t.on, (int) g_n);
    check_launch("ablate_project");
}

void ablate_mixer(float* y, int64_t y_ld, const float* x, int64_t x_ld, int64_t layer, int64_t T, void* stream) {
    mixer_impl(y, y_ld, x, x_ld, layer, T, stream);
}
void ablate_mixer_h(float* y, int64_t y_ld, const uint16_t* x16, int64_t x_ld, int64_t layer, int64_t T, void* stream) {
    mixer_impl(y, y_ld, x16, x_ld, layer, T, stream);
}

bool ablate_has_shexp_lora(int64_t layer) {
    return covers(layer) && g_shexp_lora[(size_t) layer] >= 0;
}
void ablate_shared(float* S, int64_t s_ld, const float* act, int64_t act_ld, const float* g, int64_t layer, int64_t T,
                   void* stream) {
    shared_impl(S, s_ld, act, act_ld, g, layer, T, stream);
}
void ablate_shared_h(float* S, int64_t s_ld, const uint16_t* act16, int64_t act_ld, const float* g, int64_t layer,
                     int64_t T, void* stream) {
    shared_impl(S, s_ld, act16, act_ld, g, layer, T, stream);
}

}  // namespace strata::kernels

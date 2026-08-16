// Masked Adam step for gaussian parameters (Taming 3DGS, SIGGRAPH Asia 2024).
//
// Every parameter of a GaussianModel is laid out as [N, ...] with the gaussian
// index leading, so a per-gaussian visibility mask selects whole contiguous
// rows of M = numel/N floats. Threads that map to an invisible gaussian return
// before touching param/grad/exp_avg/exp_avg_sq, which is where the saving
// comes from: the DRAM traffic of an Adam step is proportional to the number of
// visible rows, not to N.
//
// The arithmetic mirrors torch.optim.Adam's single-tensor path exactly
// (including bias correction), so an all-true mask reproduces the dense step.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

__global__ void sparse_adam_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const bool* __restrict__ visible,
    const float beta1,
    const float beta2,
    const float eps,
    const float step_size,        // lr / bias_correction1
    const float bias_corr2_sqrt,  // sqrt(1 - beta2^t)
    const int64_t total,
    const int64_t M)
{
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride)
    {
        if (!visible[i / M]) continue;

        const float g = grad[i];
        const float m = beta1 * exp_avg[i] + (1.0f - beta1) * g;
        const float v = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
        exp_avg[i] = m;
        exp_avg_sq[i] = v;

        // Same association as torch's addcdiv_(exp_avg, denom, -step_size).
        const float denom = sqrtf(v) / bias_corr2_sqrt + eps;
        param[i] += (-step_size) * (m / denom);
    }
}

}  // namespace

void sparse_adam_step(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor visible,
    double beta1,
    double beta2,
    double eps,
    double step_size,
    double bias_corr2_sqrt)
{
    TORCH_CHECK(param.is_cuda() && param.is_contiguous(), "param must be contiguous CUDA");
    TORCH_CHECK(grad.is_cuda() && grad.is_contiguous(), "grad must be contiguous CUDA");
    TORCH_CHECK(exp_avg.is_contiguous() && exp_avg_sq.is_contiguous(), "moments must be contiguous");
    TORCH_CHECK(visible.is_cuda() && visible.is_contiguous(), "visible must be contiguous CUDA");
    TORCH_CHECK(param.scalar_type() == torch::kFloat32, "param must be float32");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32, "grad must be float32");
    TORCH_CHECK(visible.scalar_type() == torch::kBool, "visible must be bool");
    TORCH_CHECK(param.sizes() == grad.sizes(), "grad shape must match param");
    TORCH_CHECK(param.sizes() == exp_avg.sizes(), "exp_avg shape must match param");
    TORCH_CHECK(param.sizes() == exp_avg_sq.sizes(), "exp_avg_sq shape must match param");
    TORCH_CHECK(param.dim() >= 1, "param must have a leading gaussian dimension");

    const int64_t N = param.size(0);
    TORCH_CHECK(visible.numel() == N, "visible must have one entry per gaussian (",
                visible.numel(), " vs ", N, ")");

    const int64_t total = param.numel();
    if (total == 0) return;
    const int64_t M = total / N;

    const at::cuda::CUDAGuard guard(param.device());
    const int threads = 256;
    const int64_t want = (total + threads - 1) / threads;
    const int blocks = (int)std::min<int64_t>(want, 65535);

    sparse_adam_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        param.data_ptr<float>(),
        grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        visible.data_ptr<bool>(),
        (float)beta1, (float)beta2, (float)eps,
        (float)step_size, (float)bias_corr2_sqrt,
        total, M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_adam_step", &sparse_adam_step, "Masked Adam step over gaussian rows");
}

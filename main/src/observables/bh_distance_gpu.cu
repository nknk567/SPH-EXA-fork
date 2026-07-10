//
// Created by Noah Kubli on 10.07.2026.
//

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"

#include <array>
#include <stdexcept>

namespace bh_merger
{

struct SearchResult
{
    double pos[3];
    int    found;
};
__device__ SearchResult result0;
__device__ SearchResult result1;

template<typename T, typename Tid>
__global__ void computeBhPositionsGPUKernel(size_t begin, size_t end, const T* x, const T* y, const T* z, const Tid* id,
                                            Tid id_0, Tid id_1)

{
    unsigned i = begin + blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= end) return;

    if (id[i] == id_0)
    {
        result0.pos[0] = x[i];
        result0.pos[1] = y[i];
        result0.pos[2] = z[i];
        result0.found  = 1;
    }

    if (id[i] == id_1)
    {
        result1.pos[0] = x[i];
        result1.pos[1] = y[i];
        result1.pos[2] = z[i];
        result1.found  = 1;
    }
}

template<typename T, typename Tid>
std::array<double, 6> computeBhPositionsGPU(size_t begin, size_t end, T* x, T* y, T* z, Tid* id, Tid id_0, Tid id_1)
{
    std::array<double, 6> result{};

    if (begin == end) { return result; }
    unsigned numThreads = 256;
    unsigned numBlocks  = cstone::iceil(end - begin, numThreads);

    SearchResult result0_host{};
    SearchResult result1_host{};

    cudaMemcpyToSymbol(GPU_SYMBOL(result0), &result0_host, sizeof(SearchResult));
    cudaMemcpyToSymbol(GPU_SYMBOL(result1), &result1_host, sizeof(SearchResult));

    computeBhPositionsGPUKernel<<<numBlocks, numThreads>>>(begin, end, x, y, z, id, id_0, id_1);
    checkGpuErrors(cudaDeviceSynchronize());

    cudaMemcpyFromSymbol(&result0_host, GPU_SYMBOL(result0), sizeof(SearchResult));
    cudaMemcpyFromSymbol(&result1_host, GPU_SYMBOL(result1), sizeof(SearchResult));

    if (result0_host.found)
    {
        result[0] = result0_host.pos[0];
        result[1] = result0_host.pos[1];
        result[2] = result0_host.pos[2];
    }
    if (result1_host.found)
    {
        result[3] = result1_host.pos[0];
        result[4] = result1_host.pos[1];
        result[5] = result1_host.pos[2];
    }
    return result;
}

template std::array<double, 6> computeBhPositionsGPU(size_t, size_t, double*, double*, double*, uint64_t*, uint64_t,
                                                     uint64_t);

} // namespace bh_merger
//
// Created by Noah Kubli on 11.03.2024.
//

#include <thrust/functional.h>

#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/warpscan.cuh"

#include "central_force_gpu.hpp"
#include "central_potential.hpp"
#include "star_data.hpp"

namespace disk
{
using cstone::GpuConfig;
using cstone::LocalIndex;

static __device__ cstone::Vec4<double> force_device;
static __device__ float                t_star_device;

template<typename T>
__device__ void atomicAddVec4(cstone::Vec4<T>* x, const cstone::Vec4<T>& y)
{
    atomicAdd(&(*x)[0], y[0]);
    atomicAdd(&(*x)[1], y[1]);
    atomicAdd(&(*x)[2], y[2]);
    atomicAdd(&(*x)[3], y[3]);
}

template<size_t numThreads, typename Data>
__global__ void computeCentralForceGPUKernel(GroupView grp, const Data d, StarPotentialType potentialType)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;

    cstone::Vec4<double> force{};
    float                t_star{INFINITY};

    if (i >= last) { force = {0., 0., 0., 0.}; }
    else
    {
        if (potentialType == StarPotentialType::newtonian) { newtonianGravity(d, i, force, t_star); }
        else if (potentialType == StarPotentialType::einstein_precession) { einsteinPrecession(d, i, force, t_star); }
    }

    typedef cub::BlockReduce<cstone::Vec4<double>, numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage               temp_storage;

    cstone::Vec4<double> force_block = BlockReduce(temp_storage).Sum(force);
    __syncthreads();

    typedef cub::BlockReduce<float, numThreads>       BlockReduceTStar;
    __shared__ typename BlockReduceTStar::TempStorage temp_storage_t_star;
    BlockReduceTStar                                  reduce_t_star(temp_storage_t_star);

    float t_star_block = reduce_t_star.Reduce(t_star, thrust::minimum<>{});
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAddVec4(&force_device, force_block);
        cstone::atomicMinFloat(&t_star_device, t_star_block);
    }
}
template<size_t numThreads, typename Data>
__global__ void computeCentralForceGPUBdtKernel(GroupView grp, GroupView active_grp, float* groupDt, const Data d,
                                                StarPotentialType potentialType)
{
    LocalIndex laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    LocalIndex i       = grp.groupStart[warpIdx] + laneIdx;

    cstone::Vec4<double> force{};
    float                t_star{INFINITY};

    if (i >= grp.groupEnd[warpIdx]) { force = {0., 0., 0., 0.}; }
    else
    {
        bool inactive_group = false;
        if (grp.groupStart + warpIdx < active_grp.groupStart || grp.groupStart + warpIdx >= active_grp.groupEnd)
        {
            inactive_group = true;
        }

        if (potentialType == StarPotentialType::newtonian) { newtonianGravity(d, i, force, t_star, inactive_group); }
        else if (potentialType == StarPotentialType::einstein_precession)
        {
            einsteinPrecession(d, i, force, t_star, inactive_group);
        }

        if (inactive_group) { t_star = INFINITY; }
    }

    auto t_star_warp = cstone::warpMin(t_star);
    groupDt[warpIdx] = min(groupDt[warpIdx], t_star_warp);

    typedef cub::BlockReduce<cstone::Vec4<double>, numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage               temp_storage;

    cstone::Vec4<double> force_block = BlockReduce(temp_storage).Sum(force);
    __syncthreads();

    if (threadIdx.x == 0) { cstone::atomicMinFloat(&t_star_device, t_star_block); }
}

template<typename Treal, typename Thydro, typename Tmass>
void computeCentralForceGPU(const GroupView& grp, const GroupView& active_grp, const Treal* x, const Treal* y,
                            const Treal* z, Thydro* ax, Thydro* ay, Thydro* az, const Tmass* m, Treal g, StarData& star,
                            float* groupDt)
{
    //    cstone::LocalIndex numParticles = last - first;
    //    constexpr unsigned numThreads   = 256;
    //    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;

    cstone::Vec4<double> force_local{0., 0., 0., 0.};
    float                t_star_local{INFINITY};
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(force_device), &force_local, sizeof(force_local)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(t_star_device), &t_star_local, sizeof(t_star_local)));

    const double         inner_size2 = star.inner_size * star.inner_size;
    CentralPotentialData data{x, y, z, m, ax, ay, az, g, star.m, inner_size2, 1.0};
    data.star_position = star.position; // Initializing in aggregate list produces an error

    //    if (last > first)
    if (numBlocks > 0)
    {
        if (groupDt == nullptr)
        {
            computeCentralForceGPUKernel<numThreads><<<numBlocks, numThreads>>>(grp, data, star.potentialType);
        }
        else
        {
            computeCentralForceGPUKernel<numThreads>
                <<<numBlocks, numThreads>>>(grp, active_grp, groupDt, data, star.potentialType);
        }

        checkGpuErrors(cudaDeviceSynchronize());
        checkGpuErrors(cudaGetLastError());
    }
    checkGpuErrors(cudaMemcpyFromSymbol(&force_local, GPU_SYMBOL(force_device), sizeof(force_local)));
    checkGpuErrors(cudaMemcpyFromSymbol(&t_star_local, GPU_SYMBOL(t_star_device), sizeof(t_star_local)));

    star.force_local = force_local;
    star.t_star      = t_star_local;
}

#define COMPUTE_CENTRAL_FORCE_GPU(Treal, Thydro, Tmass)                                                                \
    template void computeCentralForceGPU(const GroupView&, const GroupView&, const Treal* x, const Treal* y,           \
                                         const Treal* z, Thydro* ax, Thydro* ay, Thydro* az, const Tmass* m, Treal g,  \
                                         StarData&, float*);

COMPUTE_CENTRAL_FORCE_GPU(double, double, double);
COMPUTE_CENTRAL_FORCE_GPU(double, float, double);
COMPUTE_CENTRAL_FORCE_GPU(double, float, float);

} // namespace disk

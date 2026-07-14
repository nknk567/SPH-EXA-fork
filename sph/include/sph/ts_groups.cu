/*! @file
 * @brief GPU functions to manage target particle groups and block time-steps
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/math.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class T>
__global__ void groupDivvKernel(float Krho, const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
                                const T* divv, float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numGroups)
    {
        float localMax = -INFINITY;

        LocalIndex segStart = grpStart[tid];
        LocalIndex segEnd   = grpEnd[tid];

        for (LocalIndex i = segStart; i < segEnd; ++i)
        {
            localMax = max(localMax, divv[i]);
        }

        groupDt[tid] = min(groupDt[tid], Krho / abs(localMax));
    }
}

template<class T>
void groupDivvTimestepGpu(float Krho, const GroupView& grp, const T* divv, float* groupDt)
{
    int numThreads = 256;
    int numBlocks  = cstone::iceil(grp.numGroups, numThreads);

    if (numBlocks == 0) { return; }
    groupDivvKernel<<<numBlocks, numThreads>>>(Krho, grp.groupStart, grp.groupEnd, grp.numGroups, divv, groupDt);
}

template void groupDivvTimestepGpu(float, const GroupView& grp, const float*, float*);
template void groupDivvTimestepGpu(float, const GroupView& grp, const double*, float*);

template<class T>
__global__ void groupAccKernel(float etaAcc, const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
                               const T* ax, const T* ay, const T* az, const T* h, float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numGroups)
    {
        float minH2_A2 = INFINITY;

        LocalIndex segStart = grpStart[tid];
        LocalIndex segEnd   = grpEnd[tid];

        for (LocalIndex i = segStart; i < segEnd; ++i)
        {
            minH2_A2 = min(minH2_A2, h[i] * h[i] / norm2(cstone::Vec3<T>{ax[i], ay[i], az[i]}));
        }

        groupDt[tid] = min(groupDt[tid], etaAcc * std::sqrt(std::sqrt(minH2_A2)));
    }
}

template<class T>
void groupAccTimestepGpu(float etaAcc, const GroupView& grp, const T* ax, const T* ay, const T* az, const T* h,
                         float* groupDt)
{
    int numThreads = 256;
    int numBlocks  = cstone::iceil(grp.numGroups, numThreads);

    if (numBlocks == 0) { return; }
    groupAccKernel<<<numBlocks, numThreads>>>(etaAcc, grp.groupStart, grp.groupEnd, grp.numGroups, ax, ay, az, h,
                                              groupDt);
}

template void groupAccTimestepGpu(float, const GroupView&, const double*, const double*, const double*, const double*,
                                  float*);
template void groupAccTimestepGpu(float, const GroupView&, const float*, const float*, const float*, const float*,
                                  float*);

/*! @brief limit the group time step such that particles move at most ~cAdv smoothing lengths per step
 *
 * Between full domain syncs, the octree, halo lists and neighbor structures are frozen; they remain valid
 * only while particle displacements stay small compared to h. Courant and acceleration criteria do not
 * bound bulk advection, so fast coherent (supersonic) flows can otherwise outrun the frozen structures
 * within a single substep.
 */
template<class Tv, class T>
__global__ void groupAdvKernel(float cAdv, const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
                               const Tv* vx, const Tv* vy, const Tv* vz, const T* h, float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numGroups)
    {
        float minH2_V2 = INFINITY;

        LocalIndex segStart = grpStart[tid];
        LocalIndex segEnd   = grpEnd[tid];

        for (LocalIndex i = segStart; i < segEnd; ++i)
        {
            float v2 = norm2(cstone::Vec3<float>{float(vx[i]), float(vy[i]), float(vz[i])});
            minH2_V2 = min(minH2_V2, float(h[i]) * float(h[i]) / v2);
        }

        groupDt[tid] = min(groupDt[tid], cAdv * std::sqrt(minH2_V2));
    }
}

template<class Tv, class T>
void groupAdvTimestepGpu(float cAdv, const GroupView& grp, const Tv* vx, const Tv* vy, const Tv* vz, const T* h,
                         float* groupDt)
{
    int numThreads = 256;
    int numBlocks  = cstone::iceil(grp.numGroups, numThreads);

    if (numBlocks == 0) { return; }
    groupAdvKernel<<<numBlocks, numThreads>>>(cAdv, grp.groupStart, grp.groupEnd, grp.numGroups, vx, vy, vz, h,
                                              groupDt);
}

template void groupAdvTimestepGpu(float, const GroupView&, const double*, const double*, const double*, const double*,
                                  float*);
template void groupAdvTimestepGpu(float, const GroupView&, const float*, const float*, const float*, const float*,
                                  float*);
template void groupAdvTimestepGpu(float, const GroupView&, const float*, const float*, const float*, const double*,
                                  float*);
template void groupAdvTimestepGpu(float, const GroupView&, const double*, const double*, const double*, const float*,
                                  float*);

__global__ void storeRungKernel(const GroupView grp, uint8_t rung, uint8_t* particleRungs)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    particleRungs[i] = rung;
}

void storeRungGpu(const GroupView& grp, uint8_t rung, uint8_t* particleRungs)
{
    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return; }
    storeRungKernel<<<numBlocks, numThreads>>>(grp, rung, particleRungs);
}

} // namespace sph

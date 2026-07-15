/*! @file
 * @brief GPU functions to manage target particle groups and block time-steps
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"
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

__device__ float advMaxVoverL_device;

/*! @brief advection time-step limit coupled to the tree structures frozen between full syncs
 *
 * Per particle, the drift budget is budgetPerH * h[i] (margin bought by the per-substep growth of
 * treeView.searchExtFactor, which scales the h-based search radius) plus cellBudget * leafEdge(i)
 * (slack from the extent of the particle's leaf cell box, a fixed budget per hierarchy).
 * The leaf cell of each particle is located by walking the layout array; group bodies are
 * index-contiguous, so after one binary search the leaf index only moves forward.
 */
template<class Tc, class Tv, class T>
__global__ void groupAdvTreeKernel(float budgetPerH, float cellBudget, const LocalIndex* layout,
                                   cstone::TreeNodeIndex numLeafNodes, const cstone::TreeNodeIndex* leafToInternal,
                                   const cstone::Vec3<Tc>* sizes, const LocalIndex* grpStart, const LocalIndex* grpEnd,
                                   LocalIndex numGroups, const Tv* vx, const Tv* vy, const Tv* vz, const T* h,
                                   float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numGroups) { return; }

    LocalIndex segStart = grpStart[tid];
    LocalIndex segEnd   = grpEnd[tid];

    //! binary search for the leaf cell of the first body: layout[leaf] <= segStart < layout[leaf + 1]
    cstone::TreeNodeIndex low = 0, high = numLeafNodes;
    while (high - low > 1)
    {
        cstone::TreeNodeIndex mid = (low + high) / 2;
        if (layout[mid] <= LocalIndex(segStart)) { low = mid; }
        else { high = mid; }
    }
    cstone::TreeNodeIndex leaf = low;

    float minDt     = INFINITY;
    float maxVoverL = 0.0f;
    for (LocalIndex i = segStart; i < segEnd; ++i)
    {
        while (leaf + 1 < numLeafNodes && layout[leaf + 1] <= i)
        {
            leaf++;
        }

        auto  halfSize = sizes[leafToInternal[leaf]];
        float leafEdge = 2.0f * float(min(halfSize[0], min(halfSize[1], halfSize[2])));

        float v      = std::sqrt(norm2(cstone::Vec3<float>{float(vx[i]), float(vy[i]), float(vz[i])}));
        float budget = budgetPerH * float(h[i]) + cellBudget * leafEdge;

        minDt = min(minDt, v > 0.0f ? budget / v : INFINITY);
        if (leafEdge > 0.0f) { maxVoverL = max(maxVoverL, v / leafEdge); }
    }

    groupDt[tid] = min(groupDt[tid], minDt);
    cstone::atomicMaxFloat(&advMaxVoverL_device, maxVoverL);
}

template<class Tc, class Tv, class T>
float groupAdvTreeTimestepGpu(float budgetPerH, float cellBudget, const LocalIndex* layout,
                              cstone::TreeNodeIndex numLeafNodes, const cstone::TreeNodeIndex* leafToInternal,
                              const cstone::Vec3<Tc>* sizes, const GroupView& grp, const Tv* vx, const Tv* vy,
                              const Tv* vz, const T* h, float* groupDt)
{
    int numThreads = 256;
    int numBlocks  = cstone::iceil(grp.numGroups, numThreads);

    float maxVoverL = 0.0f;
    if (numBlocks == 0) { return maxVoverL; }

    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(advMaxVoverL_device), &maxVoverL, sizeof(maxVoverL)));
    groupAdvTreeKernel<<<numBlocks, numThreads>>>(budgetPerH, cellBudget, layout, numLeafNodes, leafToInternal, sizes,
                                                  grp.groupStart, grp.groupEnd, grp.numGroups, vx, vy, vz, h, groupDt);
    checkGpuErrors(cudaMemcpyFromSymbol(&maxVoverL, GPU_SYMBOL(advMaxVoverL_device), sizeof(maxVoverL)));
    return maxVoverL;
}

#define GROUP_ADV_TREE_TIMESTEP_GPU(Tc, Tv, T)                                                                         \
    template float groupAdvTreeTimestepGpu(float, float, const LocalIndex*, cstone::TreeNodeIndex,                     \
                                           const cstone::TreeNodeIndex*, const cstone::Vec3<Tc>*, const GroupView&,    \
                                           const Tv*, const Tv*, const Tv*, const T*, float*);

GROUP_ADV_TREE_TIMESTEP_GPU(double, double, double);
GROUP_ADV_TREE_TIMESTEP_GPU(double, float, float);
GROUP_ADV_TREE_TIMESTEP_GPU(float, float, float);

__device__ unsigned long long advOutsideCount_device;

/*! @brief position- and direction-aware advection time-step limit, one warp per group, one lane per body
 *
 * Exact criterion: as long as a particle's current position stays inside its own (un-inflated) leaf-cell
 * box, any neighbor search that should find it is guaranteed to open that cell, independent of accumulated
 * drift. The per-particle budget is therefore the exit time from the leaf box along the velocity direction,
 * plus the sphere-inflation allowance 2h*(g-1)/|v|, which is renewed every substep consistent with
 * treeView.searchExtFactor compounding by g per substep. Particles already outside their box (counted in
 * the diagnostic) coast on the sphere allowance until the next full sync re-homes them.
 */
template<class Tc, class Tv, class T>
__global__ void groupAdvLeafKernel(float extGrowthMinusOne, const LocalIndex* layout,
                                   cstone::TreeNodeIndex numLeafNodes, const cstone::TreeNodeIndex* leafToInternal,
                                   const cstone::Vec3<Tc>* centers, const cstone::Vec3<Tc>* sizes, GroupView grp,
                                   const Tc* x, const Tc* y, const Tc* z, const Tv* vx, const Tv* vy, const Tv* vz,
                                   const T* h, float* groupDt)
{
    LocalIndex laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    //! no block-wide collectives below: whole warps may exit early
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex segStart = grp.groupStart[warpIdx];
    LocalIndex segEnd   = grp.groupEnd[warpIdx];

    //! lane 0 locates the leaf of the group's first body, all lanes then walk forward to their own body
    cstone::TreeNodeIndex leaf = 0;
    if (laneIdx == 0)
    {
        cstone::TreeNodeIndex low = 0, high = numLeafNodes;
        while (high - low > 1)
        {
            cstone::TreeNodeIndex mid = (low + high) / 2;
            if (layout[mid] <= segStart) { low = mid; }
            else { high = mid; }
        }
        leaf = low;
    }
    leaf = cstone::shflSync(leaf, 0);

    LocalIndex i       = segStart + laneIdx;
    float      dt      = INFINITY;
    float      vOverL  = 0.0f;
    bool       outside = false;

    if (i < segEnd)
    {
        while (leaf + 1 < numLeafNodes && layout[leaf + 1] <= i)
        {
            leaf++;
        }

        auto node     = leafToInternal[leaf];
        auto center   = centers[node];
        auto halfSize = sizes[node];

        double pos[3] = {double(x[i]), double(y[i]), double(z[i])};
        double vel[3] = {double(vx[i]), double(vy[i]), double(vz[i])};
        double v      = std::sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);

        //! time until the particle exits its own leaf box along its velocity, per axis
        double dtExit = INFINITY;
        for (int k = 0; k < 3; ++k)
        {
            if (vel[k] > 0.0) { dtExit = min(dtExit, max(0.0, double(center[k] + halfSize[k]) - pos[k]) / vel[k]); }
            else if (vel[k] < 0.0)
            {
                dtExit = min(dtExit, max(0.0, pos[k] - double(center[k] - halfSize[k])) / -vel[k]);
            }
        }
        //! empty-cell marker: no cell slack
        if (halfSize[0] == Tc(0) && halfSize[1] == Tc(0) && halfSize[2] == Tc(0)) { dtExit = 0.0; }

        double dtExt = v > 0.0 ? 2.0 * double(extGrowthMinusOne) * double(h[i]) / v : INFINITY;
        dt           = float(min(dtExit + dtExt, double(INFINITY)));

        float leafEdge = 2.0f * float(min(halfSize[0], min(halfSize[1], halfSize[2])));
        if (leafEdge > 0.0f) { vOverL = float(v) / leafEdge; }
        outside = (dtExit == 0.0);
    }

    float dtWarp     = cstone::warpMin(dt);
    float vOverLWarp = cstone::warpMax(vOverL);
    if (laneIdx == 0)
    {
        groupDt[warpIdx] = min(groupDt[warpIdx], dtWarp);
        cstone::atomicMaxFloat(&advMaxVoverL_device, vOverLWarp);
    }
    //! particles outside their leaf box are rare; one atomic per occurrence is acceptable
    if (outside) { atomicAdd(&advOutsideCount_device, 1ull); }
}

template<class Tc, class Tv, class T>
std::tuple<float, unsigned long long>
groupAdvLeafTimestepGpu(float extGrowthMinusOne, const LocalIndex* layout, cstone::TreeNodeIndex numLeafNodes,
                        const cstone::TreeNodeIndex* leafToInternal, const cstone::Vec3<Tc>* centers,
                        const cstone::Vec3<Tc>* sizes, const GroupView& grp, const Tc* x, const Tc* y, const Tc* z,
                        const Tv* vx, const Tv* vy, const Tv* vz, const T* h, float* groupDt)
{
    constexpr unsigned numThreads       = 256;
    unsigned           numWarpsPerBlock = numThreads / GpuConfig::warpSize;
    unsigned           numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;

    float              maxVoverL    = 0.0f;
    unsigned long long numOutside   = 0;
    if (numBlocks == 0) { return {maxVoverL, numOutside}; }

    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(advMaxVoverL_device), &maxVoverL, sizeof(maxVoverL)));
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(advOutsideCount_device), &numOutside, sizeof(numOutside)));

    groupAdvLeafKernel<<<numBlocks, numThreads>>>(extGrowthMinusOne, layout, numLeafNodes, leafToInternal, centers,
                                                  sizes, grp, x, y, z, vx, vy, vz, h, groupDt);

    checkGpuErrors(cudaMemcpyFromSymbol(&maxVoverL, GPU_SYMBOL(advMaxVoverL_device), sizeof(maxVoverL)));
    checkGpuErrors(cudaMemcpyFromSymbol(&numOutside, GPU_SYMBOL(advOutsideCount_device), sizeof(numOutside)));
    return {maxVoverL, numOutside};
}

#define GROUP_ADV_LEAF_TIMESTEP_GPU(Tc, Tv, T)                                                                         \
    template std::tuple<float, unsigned long long> groupAdvLeafTimestepGpu(                                            \
        float, const LocalIndex*, cstone::TreeNodeIndex, const cstone::TreeNodeIndex*, const cstone::Vec3<Tc>*,        \
        const cstone::Vec3<Tc>*, const GroupView&, const Tc*, const Tc*, const Tc*, const Tv*, const Tv*, const Tv*,   \
        const T*, float*);

GROUP_ADV_LEAF_TIMESTEP_GPU(double, double, double);
GROUP_ADV_LEAF_TIMESTEP_GPU(double, float, float);
GROUP_ADV_LEAF_TIMESTEP_GPU(float, float, float);

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

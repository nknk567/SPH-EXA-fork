/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Density i-loop GPU driver
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include "cstone/cuda/cuda_utils.cuh"

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/ve_kern.hpp"

namespace sph
{
namespace gpu
{

template<class Dataset>
void computeVe(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&)
{
    veIjLoop(d.neighborhood, d.K, rawPtr(d.xm), rawPtr(d.wh), rawPtr(d.kx));
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeVe(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                        const cstone::Box<SphTypes::CoordinateType>&);

template<class Tm, class T>
struct NominalBallmass
{
    T eta;
    __device__ T operator()(Tm mi) const { return eta * T(mi); }
};

template<class Dataset>
void fillNominalBallmass(const GroupView& grp, Dataset& d)
{
    using Th = typename Dataset::HydroType;
    using Tm = typename Dataset::Tmass;
    thrust::transform(thrust::device, rawPtr(d.m) + grp.firstBody, rawPtr(d.m) + grp.lastBody,
                      rawPtr(d.ballmass) + grp.firstBody, NominalBallmass<Tm, Th>{ballmassEta<Th>(d.ng0)});
    checkGpuErrors(cudaDeviceSynchronize());
}

template void fillNominalBallmass(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d);

template<class T>
struct RelativeHChange
{
    __device__ T operator()(const thrust::tuple<T, T>& hNew_h) const
    {
        return std::abs(thrust::get<0>(hNew_h) - thrust::get<1>(hNew_h)) / thrust::get<1>(hNew_h);
    }
};

template<class T>
struct Unconverged
{
    T tol;
    __device__ bool operator()(T relDh) const { return relDh >= tol; }
};

//! @brief detects hNew committed exactly at factor * hOld (bitwise match of the postamble's clamp)
template<class T>
struct CapHit
{
    T factor;
    __device__ bool operator()(const thrust::tuple<T, T>& hNew_h) const
    {
        return thrust::get<0>(hNew_h) == factor * thrust::get<1>(hNew_h);
    }
};

template<class Dataset, class Tv>
NRPassStats computeVeNR(const GroupView& grp, Dataset& d, const cstone::Box<typename Dataset::RealType>&, Tv* h0,
                        bool firstIteration, float tol, float hExtFactor, float bandMin, float bandMax)
{
    using Th = typename Dataset::HydroType;
    if (firstIteration)
    {
        cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, rawPtr(d.h), d.x.size(), h0);
    }
    veNRIjLoop(d.neighborhood, d.K, hExtFactor, tol, bandMin, bandMax, rawPtr(d.xm), rawPtr(d.m), h0,
               rawPtr(d.ballmass), rawPtr(d.wh), rawPtr(d.whd), rawPtr(d.kx), rawPtr(d.ay), rawPtr(d.ax));
    //! per-particle relative h change into the az scratch, consumed by computeVeNRTail
    auto begin = thrust::make_zip_iterator(rawPtr(d.ay) + grp.firstBody, rawPtr(d.h) + grp.firstBody);
    auto end   = thrust::make_zip_iterator(rawPtr(d.ay) + grp.lastBody, rawPtr(d.h) + grp.lastBody);
    thrust::transform(thrust::device, begin, end, rawPtr(d.az) + grp.firstBody, RelativeHChange<Th>{});
    size_t numUnconverged = thrust::count_if(thrust::device, rawPtr(d.az) + grp.firstBody,
                                             rawPtr(d.az) + grp.lastBody, Unconverged<Th>{Th(tol)});
    size_t capUp   = thrust::count_if(thrust::device, begin, end, CapHit<Th>{Th(1.1)});
    size_t capDown = thrust::count_if(thrust::device, begin, end, CapHit<Th>{Th(0.5)});
    //! post-pass zeros are exactly this pass's band-check resets, see NRPassStats::numReset
    size_t numReset = thrust::count(thrust::device, rawPtr(d.ballmass) + grp.firstBody,
                                    rawPtr(d.ballmass) + grp.lastBody, Th(0));
    // commit the updated smoothing lengths of locally owned particles
    cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, rawPtr(d.ay) + grp.firstBody,
                           grp.lastBody - grp.firstBody, rawPtr(d.h) + grp.firstBody);
    checkGpuErrors(cudaDeviceSynchronize());
    return {numUnconverged, capUp, capDown, numReset};
}

template NRPassStats computeVeNR(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                                 const cstone::Box<SphTypes::CoordinateType>&, SphTypes::HydroType*, bool, float,
                                 float, float, float);

template<class Dataset, class Tv>
std::pair<size_t, size_t> countHWallPinned(const GroupView& grp, Dataset& d, const Tv* h0, float hExtFactor)
{
    using Th   = typename Dataset::HydroType;
    auto begin = thrust::make_zip_iterator(rawPtr(d.h) + grp.firstBody, h0 + grp.firstBody);
    auto end   = thrust::make_zip_iterator(rawPtr(d.h) + grp.lastBody, h0 + grp.lastBody);
    size_t up   = thrust::count_if(thrust::device, begin, end, CapHit<Th>{Th(hExtFactor)});
    size_t down = thrust::count_if(thrust::device, begin, end, CapHit<Th>{Th(0.5)});
    return {up, down};
}

template std::pair<size_t, size_t> countHWallPinned(const GroupView&,
                                                    sphexa::ParticlesData<cstone::execution::Gpu>& d,
                                                    const SphTypes::HydroType*, float);

template<class Tv>
struct UnconvergedIndex
{
    const Tv* relDh;
    Tv        tol;
    __device__ bool operator()(cstone::LocalIndex i) const { return relDh[i] >= tol; }
};

/*! @brief fused NR tail: every thread iterates its own particle to convergence
 *
 * One launch for the entire tail phase: each thread keeps h in a register, re-traverses only
 * as long as ITS particle is unconverged (measured average over TDE tails: ~1.01 traversals
 * per particle), and commits h once. @p bins[k] counts the threads whose convergence iteration
 * was k (1-based); bins[maxIter + 1] counts those that exhausted the iteration budget. The
 * per-pass unconverged counts of the former lockstep implementation are recovered on the host
 * as suffix sums over bins — the trajectories are identical, since an update depends only on
 * the particle's own h and the frozen volume elements.
 */
template<class Tc, class T, class Tm, class KeyType>
__global__ __launch_bounds__(128) void veNRTailKernel(const cstone::LocalIndex* __restrict__ subset,
                                                      cstone::LocalIndex n, Tc K, T* __restrict__ ballmass,
                                                      T hExtFactor, T tol, T bandMin, T bandMax,
                                                      unsigned maxIter, const cstone::OctreeNsView<Tc, KeyType> tree,
                                                      const cstone::Box<Tc> box, const Tc* __restrict__ x,
                                                      const Tc* __restrict__ y, const Tc* __restrict__ z,
                                                      T* __restrict__ h, const T* __restrict__ xm,
                                                      const Tm* __restrict__ m, const T* __restrict__ h0,
                                                      const T* __restrict__ wh, const T* __restrict__ whd,
                                                      T* __restrict__ cnt, unsigned long long* __restrict__ bins)
{
    /* block-local histogram: most threads converge at the same iteration, so direct global
     * atomics on bins would serialize; aggregate per block first (dynamic shared memory),
     * then flush with at most one global atomic per bin per block. Layout: [1, maxIter] =
     * convergence-iteration counts, [maxIter + 1] = budget exhausted, [maxIter + 2] and
     * [maxIter + 3] = per-iteration up/down clamp events, [maxIter + 4] = band-check resets. */
    extern __shared__ unsigned long long sharedBins[];
    const unsigned                       numBins = maxIter + 5;
    for (unsigned k = threadIdx.x; k < numBins; k += blockDim.x)
    {
        sharedBins[k] = 0;
    }
    __syncthreads();

    cstone::LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        cstone::LocalIndex i = subset[tid];

        T                  hi    = h[i];
        unsigned           kConv = maxIter + 1;
        unsigned long long capU = 0, capD = 0;
        bool               fired = false;
        for (unsigned it = 1; it <= maxIter; ++it)
        {
            T hNew = veNRTraversalUpdate(i, hi, K, ballmass, hExtFactor, tol, bandMin, bandMax, tree, box, x, y, z,
                                         xm, m, h0, wh, whd, cnt);
            /* signals entering the tail were consumed by the update's re-seed branch, so a zero
             * after an update is a band-check reset fired by this update; the once-per-step gate
             * makes fired-detection through the array exact */
            fired = fired || ballmass[i] == T(0);
            capU += hNew == T(1.1) * hi;
            capD += hNew == T(0.5) * hi;
            T rel  = std::abs(hNew - hi) / hi;
            hi     = hNew;
            if (rel < tol)
            {
                kConv = it;
                break;
            }
        }
        h[i] = hi;
        atomicAdd(sharedBins + kConv, 1ull);
        if (capU) { atomicAdd(sharedBins + maxIter + 2, capU); }
        if (capD) { atomicAdd(sharedBins + maxIter + 3, capD); }
        if (fired) { atomicAdd(sharedBins + maxIter + 4, 1ull); }
    }

    __syncthreads();
    for (unsigned k = threadIdx.x; k < numBins; k += blockDim.x)
    {
        if (sharedBins[k] > 0) { atomicAdd(bins + k, sharedBins[k]); }
    }
}

template<class Dataset, class Tv>
unsigned computeVeNRTail(const GroupView& grp, Dataset& d, const cstone::Box<typename Dataset::RealType>& box,
                         const Tv* h0, unsigned maxPasses, std::vector<size_t>& unconvergedPerPass, float tol,
                         float hExtFactor, float bandMin, float bandMax, size_t& capUp, size_t& capDown,
                         size_t& numReset)
{
    using Th = typename Dataset::HydroType;
    if (maxPasses == 0) { return 0; }

    //! gather the particles left unconverged by the last computeVeNR pass (az scratch)
    thrust::device_vector<cstone::LocalIndex> subset(
        thrust::count_if(thrust::device, rawPtr(d.az) + grp.firstBody, rawPtr(d.az) + grp.lastBody,
                         Unconverged<Th>{Th(tol)}));
    if (subset.empty()) { return 0; }
    thrust::copy_if(thrust::device, thrust::counting_iterator<cstone::LocalIndex>(grp.firstBody),
                    thrust::counting_iterator<cstone::LocalIndex>(grp.lastBody), subset.begin(),
                    UnconvergedIndex<Th>{rawPtr(d.az), Th(tol)});

    thrust::device_vector<unsigned long long> devBins(maxPasses + 5, 0ull);
    cstone::LocalIndex                        n         = subset.size();
    unsigned                                  numBlocks = (n + 127) / 128;

    size_t sharedBytes = (maxPasses + 5) * sizeof(unsigned long long);
    veNRTailKernel<<<numBlocks, 128, sharedBytes>>>(
        thrust::raw_pointer_cast(subset.data()), n, d.K, rawPtr(d.ballmass), Th(hExtFactor), Th(tol), Th(bandMin),
        Th(bandMax), maxPasses, d.treeView, box, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h), rawPtr(d.xm),
        rawPtr(d.m), h0, rawPtr(d.wh), rawPtr(d.whd), rawPtr(d.ax), thrust::raw_pointer_cast(devBins.data()));
    checkGpuErrors(cudaDeviceSynchronize());
    thrust::host_vector<unsigned long long> bins = devBins;
    capUp += bins[maxPasses + 2];
    capDown += bins[maxPasses + 3];
    numReset += bins[maxPasses + 4];

    //! per-pass unconverged counts = suffix sums; passes performed = last needed iteration
    unsigned passes = maxPasses;
    while (passes > 1 && bins[passes] == 0 && bins[passes + 1] == 0)
    {
        --passes;
    }
    for (unsigned p = 1; p <= passes; ++p)
    {
        size_t stillUnconverged = 0;
        for (unsigned k = p + 1; k <= maxPasses + 1; ++k)
        {
            stillUnconverged += bins[k];
        }
        unconvergedPerPass.push_back(stillUnconverged);
    }
    return passes;
}

template unsigned computeVeNRTail(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                                  const cstone::Box<SphTypes::CoordinateType>&, const SphTypes::HydroType*, unsigned,
                                  std::vector<size_t>&, float, float, float, float, size_t&, size_t&, size_t&);

template<class Dataset, class Tv>
void computeVolstd(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&, Tv* volstd)
{
    volstdIjLoop(d.neighborhood, d.K, rawPtr(d.xm), rawPtr(d.kx), rawPtr(d.wh), volstd);
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeVolstd(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                            const cstone::Box<SphTypes::CoordinateType>&, SphTypes::HydroType*);

template<class Tv>
struct VolstdClamp
{
    //! @brief clamp the per-step change of the carried volume, see the ve-nr volstdGrow/ShrinkFactor parameters
    Tv growFactor, shrinkFactor;

    HOST_DEVICE_FUN Tv operator()(Tv volstdi, Tv xmOld) const
    {
        Tv lo = xmOld / shrinkFactor;
        Tv hi = xmOld * growFactor;
        return stl::min(stl::max(volstdi, lo), hi);
    }
};

template<class Dataset, class Tv>
void setVolumeElements(const GroupView& grp, Dataset& d, const Tv* volstd, float volstdGrowFactor,
                       float volstdShrinkFactor)
{
    thrust::transform(thrust::device, volstd + grp.firstBody, volstd + grp.lastBody, rawPtr(d.xm) + grp.firstBody,
                      rawPtr(d.xm) + grp.firstBody, VolstdClamp<Tv>{Tv(volstdGrowFactor), Tv(volstdShrinkFactor)});
    checkGpuErrors(cudaDeviceSynchronize());
}

template void setVolumeElements(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                                const SphTypes::HydroType*, float, float);

template<class T>
struct NonFinite
{
    HOST_DEVICE_FUN size_t operator()(T v) const { return !std::isfinite(v); }
};

template<class T>
size_t countNonFiniteGpu(const T* f, size_t first, size_t last)
{
    if (last <= first) { return 0; }
    return thrust::transform_reduce(thrust::device, f + first, f + last, NonFinite<T>{}, size_t(0),
                                    thrust::plus<size_t>{});
}

template size_t countNonFiniteGpu(const float*, size_t, size_t);
template size_t countNonFiniteGpu(const double*, size_t, size_t);

} // namespace gpu
} // namespace sph

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

template<class Dataset, class Tv>
size_t computeVeNR(const GroupView& grp, Dataset& d, const cstone::Box<typename Dataset::RealType>&, Tv* h0,
                   bool firstIteration, float tol, float hExtFactor)
{
    using Th = typename Dataset::HydroType;
    if (firstIteration)
    {
        cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, rawPtr(d.h), d.x.size(), h0);
    }
    veNRIjLoop(d.neighborhood, d.K, d.ng0, hExtFactor, rawPtr(d.xm), rawPtr(d.m), h0, rawPtr(d.wh), rawPtr(d.whd),
               rawPtr(d.kx), rawPtr(d.ay));
    //! per-particle relative h change into the az scratch, consumed by computeVeNRTail
    auto begin = thrust::make_zip_iterator(rawPtr(d.ay) + grp.firstBody, rawPtr(d.h) + grp.firstBody);
    auto end   = thrust::make_zip_iterator(rawPtr(d.ay) + grp.lastBody, rawPtr(d.h) + grp.lastBody);
    thrust::transform(thrust::device, begin, end, rawPtr(d.az) + grp.firstBody, RelativeHChange<Th>{});
    size_t numUnconverged = thrust::count_if(thrust::device, rawPtr(d.az) + grp.firstBody,
                                             rawPtr(d.az) + grp.lastBody, Unconverged<Th>{Th(tol)});
    // commit the updated smoothing lengths of locally owned particles
    cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, rawPtr(d.ay) + grp.firstBody,
                           grp.lastBody - grp.firstBody, rawPtr(d.h) + grp.firstBody);
    checkGpuErrors(cudaDeviceSynchronize());
    return numUnconverged;
}

template size_t computeVeNR(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                            const cstone::Box<SphTypes::CoordinateType>&, SphTypes::HydroType*, bool, float, float);

template<class Tv>
struct UnconvergedIndex
{
    const Tv* relDh;
    Tv        tol;
    __device__ bool operator()(cstone::LocalIndex i) const { return relDh[i] >= tol; }
};

template<class Tc, class T, class Tm, class KeyType>
__global__ __launch_bounds__(128) void veNRTailKernel(const cstone::LocalIndex* __restrict__ subset,
                                                      cstone::LocalIndex n, Tc K, T etaBallmass, T hExtFactor, T tol,
                                                      const cstone::OctreeNsView<Tc, KeyType> tree,
                                                      const cstone::Box<Tc> box, const Tc* __restrict__ x,
                                                      const Tc* __restrict__ y, const Tc* __restrict__ z,
                                                      T* __restrict__ h, const T* __restrict__ xm,
                                                      const Tm* __restrict__ m, const T* __restrict__ h0,
                                                      const T* __restrict__ wh, const T* __restrict__ whd,
                                                      unsigned long long* numUnconverged)
{
    cstone::LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    cstone::LocalIndex i = subset[tid];

    T hi   = h[i];
    T hNew = veNRTraversalUpdate(i, K, etaBallmass, hExtFactor, tree, box, x, y, z, h, xm, m, h0, wh, whd);
    h[i]   = hNew;
    if (std::abs(hNew - hi) / hi >= tol) { atomicAdd(numUnconverged, 1ull); }
}

template<class Dataset, class Tv>
unsigned computeVeNRTail(const GroupView& grp, Dataset& d, const cstone::Box<typename Dataset::RealType>& box,
                         const Tv* h0, unsigned maxPasses, std::vector<size_t>& unconvergedPerPass, float tol,
                         float hExtFactor)
{
    using Th = typename Dataset::HydroType;

    //! gather the particles left unconverged by the last computeVeNR pass (az scratch)
    thrust::device_vector<cstone::LocalIndex> subset(
        thrust::count_if(thrust::device, rawPtr(d.az) + grp.firstBody, rawPtr(d.az) + grp.lastBody,
                         Unconverged<Th>{Th(tol)}));
    if (subset.empty()) { return 0; }
    thrust::copy_if(thrust::device, thrust::counting_iterator<cstone::LocalIndex>(grp.firstBody),
                    thrust::counting_iterator<cstone::LocalIndex>(grp.lastBody), subset.begin(),
                    UnconvergedIndex<Th>{rawPtr(d.az), Th(tol)});

    thrust::device_vector<unsigned long long> devCount(1);
    cstone::LocalIndex                        n         = subset.size();
    unsigned                                  numBlocks = (n + 127) / 128;

    unsigned passes = 0;
    while (passes < maxPasses)
    {
        ++passes;
        devCount[0] = 0;
        veNRTailKernel<<<numBlocks, 128>>>(thrust::raw_pointer_cast(subset.data()), n, d.K, ballmassEta<Th>(d.ng0),
                                           Th(hExtFactor), Th(tol), d.treeView, box, rawPtr(d.x), rawPtr(d.y),
                                           rawPtr(d.z), rawPtr(d.h), rawPtr(d.xm), rawPtr(d.m), h0, rawPtr(d.wh),
                                           rawPtr(d.whd), thrust::raw_pointer_cast(devCount.data()));
        checkGpuErrors(cudaDeviceSynchronize());
        size_t numUnconverged = devCount[0];
        unconvergedPerPass.push_back(numUnconverged);
        if (numUnconverged == 0) { break; }
    }
    return passes;
}

template unsigned computeVeNRTail(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                                  const cstone::Box<SphTypes::CoordinateType>&, const SphTypes::HydroType*, unsigned,
                                  std::vector<size_t>&, float, float);

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

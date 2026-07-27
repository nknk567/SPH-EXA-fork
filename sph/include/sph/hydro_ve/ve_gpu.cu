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

#include <thrust/execution_policy.h>
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

template<class Dataset, class Tv>
typename Dataset::HydroType computeVeNR(const GroupView& grp, Dataset& d,
                                        const cstone::Box<typename Dataset::RealType>&, Tv* h0, bool firstIteration)
{
    using Th = typename Dataset::HydroType;
    if (firstIteration)
    {
        cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, rawPtr(d.h), d.x.size(), h0);
    }
    veNRIjLoop(d.neighborhood, d.K, d.ng0, rawPtr(d.xm), rawPtr(d.m), h0, rawPtr(d.wh), rawPtr(d.whd), rawPtr(d.kx),
               rawPtr(d.ay));
    auto begin = thrust::make_zip_iterator(rawPtr(d.ay) + grp.firstBody, rawPtr(d.h) + grp.firstBody);
    auto end   = thrust::make_zip_iterator(rawPtr(d.ay) + grp.lastBody, rawPtr(d.h) + grp.lastBody);
    Th   maxDh =
        thrust::transform_reduce(thrust::device, begin, end, RelativeHChange<Th>{}, Th(0), thrust::maximum<Th>{});
    // commit the updated smoothing lengths of locally owned particles
    cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, rawPtr(d.ay) + grp.firstBody,
                           grp.lastBody - grp.firstBody, rawPtr(d.h) + grp.firstBody);
    checkGpuErrors(cudaDeviceSynchronize());
    return maxDh;
}

template SphTypes::HydroType computeVeNR(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                                         const cstone::Box<SphTypes::CoordinateType>&, SphTypes::HydroType*, bool);

template<class Dataset, class Tv>
void computeVolstd(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&, Tv* volstd)
{
    volstdIjLoop(d.neighborhood, d.K, rawPtr(d.xm), rawPtr(d.kx), rawPtr(d.wh), volstd);
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeVolstd(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                            const cstone::Box<SphTypes::CoordinateType>&, SphTypes::HydroType*);

template<class Dataset, class Tv>
void setVolumeElements(const GroupView& grp, Dataset& d, const Tv* volstd)
{
    cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, volstd + grp.firstBody, grp.lastBody - grp.firstBody,
                           rawPtr(d.xm) + grp.firstBody);
    checkGpuErrors(cudaDeviceSynchronize());
}

template void setVolumeElements(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                                const SphTypes::HydroType*);

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

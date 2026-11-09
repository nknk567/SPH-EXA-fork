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

#include "cstone/cuda/cuda_utils.cuh"

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/ve_kern.hpp"

#include <thrust/count.h>
#include "sph/hydro_ve/ve.hpp"
#include "sph/hydro_ve/nr_statistics.hpp"

namespace sph
{
namespace gpu
{

template<class Dataset>
void computeVe(Dataset& d, const cstone::Box<typename Dataset::RealType>&)
{
    veIjLoop(d.neighborhood, d.K, rawPtr(d.xm), d.wh, rawPtr(d.kx));
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeVe(sphexa::ParticlesData<cstone::execution::Gpu>& d, const cstone::Box<SphTypes::CoordinateType>&);

template<class Tm, class T>
struct Ballmass
{
    T            eta;
    __device__ T operator()(Tm mi) const { return eta * T(mi); }
};

template<class Dataset>
void initializeBallmass(const GroupView& grp, Dataset& d)
{
    using Th = typename Dataset::HydroType;
    using Tm = typename Dataset::Tmass;
    thrust::transform(thrust::device, rawPtr(d.m) + grp.firstBody, rawPtr(d.m) + grp.lastBody,
                      rawPtr(d.ballmass) + grp.firstBody, Ballmass<Tm, Th>{ballmassEta<Th>(d.ng0)});
    checkGpuErrors(cudaDeviceSynchronize());
}

template void initializeBallmass(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d);

template<class Dataset>
void computeVeNR(Dataset& d, float hExtFactor, float hWallFactor, float hNRTol, float ngmin, float ngmax,
                 const typename Dataset::HydroType* h0, typename Dataset::HydroType* h_buf,
                 typename Dataset::HydroType* nc_buf)
{
    veNRIjLoop(d.neighborhood, d.K, d.xm.data(), d.m.data(), d.ballmass.data(), d.wh, d.kx.data(), hExtFactor,
               hWallFactor, hNRTol, ngmin, ngmax, h0, h_buf, nc_buf);
}

#define COMPUTE_VE_NR(Dataset)                                                                                         \
    template void computeVeNR(Dataset&, float, float, float, float, float, const typename Dataset::HydroType*,         \
                              typename Dataset::HydroType*, typename Dataset::HydroType*);

COMPUTE_VE_NR(sphexa::ParticlesData<cstone::execution::Gpu>);

template<class Th>
struct Unconverged
{
    Th              tol;
    __device__ bool operator()(const thrust::tuple<Th, Th>& h_new_h) const
    {
        const auto& h_new = thrust::get<0>(h_new_h);
        const auto& h     = thrust::get<1>(h_new_h);

        Th rel = std::abs(h_new - h) / h;
        return rel >= tol;
    }
};

template<typename Dataset>
sph::NRStatistics computeNRStat(const cstone::GroupView& grp, Dataset& d, const typename Dataset::HydroType* h_new,
                                const float hNRTol)
{
    auto begin           = thrust::make_zip_iterator(h_new + grp.firstBody, rawPtr(d.h) + grp.firstBody);
    auto end             = thrust::make_zip_iterator(h_new + grp.lastBody, rawPtr(d.h) + grp.lastBody);
    using Th             = typename Dataset::HydroType;
    size_t n_unconverged = thrust::count_if(thrust::device, begin, end, Unconverged<Th>{Th(hNRTol)});
    size_t n_reset =
        thrust::count(thrust::device, rawPtr(d.ballmass) + grp.firstBody, rawPtr(d.ballmass) + grp.lastBody, Th(0));
    return {n_unconverged, n_reset};
}

template sph::NRStatistics computeNRStat(const cstone::GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>&,
                                         const float*, const float);

} // namespace gpu
} // namespace sph

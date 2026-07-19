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

template<class Dataset, class Tv>
void computeVeNR(const GroupView& grp, Dataset& d, const cstone::Box<typename Dataset::RealType>&, Tv* h0,
                 bool firstIteration)
{
    if (firstIteration)
    {
        cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, rawPtr(d.h), d.x.size(), h0);
    }
    veNRIjLoop(d.neighborhood, d.K, rawPtr(d.xm), rawPtr(d.m), rawPtr(d.ballmass), h0, rawPtr(d.wh), rawPtr(d.whd),
               rawPtr(d.kx), rawPtr(d.ay));
    // commit the updated smoothing lengths of locally owned particles
    cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, rawPtr(d.ay) + grp.firstBody,
                           grp.lastBody - grp.firstBody, rawPtr(d.h) + grp.firstBody);
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeVeNR(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d,
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

template<class T, class Th, class Tm>
__global__ void ballmassFromDensityKernel(cstone::LocalIndex first, cstone::LocalIndex last, const T* kx, const T* xm,
                                          const Tm* m, const Th* h, T* ballmass)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i < last) { ballmass[i] = kx[i] * m[i] / xm[i] * h[i] * h[i] * h[i]; }
}

template<class Dataset>
void ballmassFromDensity(const GroupView& grp, Dataset& d)
{
    unsigned numThreads = 256;
    unsigned numBlocks  = cstone::iceil(grp.lastBody - grp.firstBody, numThreads);
    if (numBlocks == 0) { return; }
    ballmassFromDensityKernel<<<numBlocks, numThreads>>>(grp.firstBody, grp.lastBody, rawPtr(d.kx), rawPtr(d.xm),
                                                         rawPtr(d.m), rawPtr(d.h), rawPtr(d.ballmass));
    checkGpuErrors(cudaDeviceSynchronize());
}

template void ballmassFromDensity(const GroupView&, sphexa::ParticlesData<cstone::execution::Gpu>& d);

} // namespace gpu
} // namespace sph

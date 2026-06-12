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

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/iad_divv_curlv_kern.hpp"

namespace sph
{
namespace gpu
{

template<class T>
__global__ void gradHNewtonRaphsonKernel(size_t first, size_t last, T* h, const T* m, const T* xm, const T* kx,
                                         const T* gradh, T eta, T tolerance, int* unconverged)
{
    unsigned i = first + blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= last) return;

    T rho = kx[i] * m[i] / xm[i];

    T h2 = h[i] * h[i];
    T h3 = h2 * h[i];

    T eta3 = eta * eta * eta;

    T f = h3 * rho - eta3 * m[i];

    T fp = T(3) * h2 * rho * gradh[i];

    if (fabs(fp) < T(1e-20)) return;

    T dh = -f / fp;

    // optional limiter
    dh = max(T(-0.2) * h[i], min(T(0.2) * h[i], dh));

    h[i] += dh;

    if (fabs(dh) > tolerance * h[i]) atomicAdd(unconverged, 1);
}

template<class T>
bool gradHNewtonRaphsonIteration(T* h, const T* m, const T* xm, const T* kx, const T* gradh, size_t numParticles, T eta,
                                 T tolerance)
{
    thrust::device_vector<int> unconverged(1, 0);

    unsigned blockSize = 256;
    unsigned numBlocks = (numParticles + blockSize - 1) / blockSize;

    gradHNewtonRaphsonKernel<<<numBlocks, blockSize>>>(0, numParticles, h, m, xm, kx, gradh, eta, tolerance,
                                                       thrust::raw_pointer_cast(unconverged.data()));

    int numUnconverged;
    cudaMemcpy(&numUnconverged, thrust::raw_pointer_cast(unconverged.data()), sizeof(int), cudaMemcpyDeviceToHost);

    return numUnconverged == 0;
}

template<class Dataset>
bool computeGradHNewtonRaphsonIteration(const GroupView&, Dataset& d, typename Dataset::RealType eta,
                                        typename Dataset::RealType tolerance)
{
    bool converged = gradHNewtonRaphsonIterationCuda(rawPtr(d.h), rawPtr(d.m), rawPtr(d.xm), rawPtr(d.kx),
                                                     rawPtr(d.gradh), d.numParticles, eta, tolerance);

    checkGpuErrors(cudaDeviceSynchronize());

    return converged;
}
template bool computeGradHNewtonRaphsonIteration(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
                                                 SphTypes::RealType eta, SphTypes::RealType tolerance);
template<class Dataset>
void computeIadDivvCurlvGradh(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&)
{
    iadDivvCurlvGradhIjLoop(d.neighborhood, d.K, rawPtr(d.vx), rawPtr(d.vy), rawPtr(d.vz), rawPtr(d.m), rawPtr(d.xm),
                            rawPtr(d.kx), rawPtr(d.nc), rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22),
                            rawPtr(d.c23), rawPtr(d.c33), rawPtr(d.wh), rawPtr(d.whd), rawPtr(d.gradh), rawPtr(d.divv),
                            d.curlv.size() == d.x.size() ? rawPtr(d.curlv) : nullptr, rawPtr(d.dV11), rawPtr(d.dV12),
                            rawPtr(d.dV13), rawPtr(d.dV22), rawPtr(d.dV23), rawPtr(d.dV33), d.dV11.size() == d.x.size(),
                            d.condition_quality_target, rawPtr(d.iadRegularized));
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeIadDivvCurlvGradh(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
                                       const cstone::Box<SphTypes::CoordinateType>&);

} // namespace gpu
} // namespace sph

//
// Created by Noah Kubli on 12.11.2026.
//

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/tree/definitions.h"

#include "dissipation_gpu.hpp"

namespace disk
{

template<typename Thydro>
__global__ void integrateDissipationKernel(size_t first, size_t last, const Thydro* du_visc, Thydro* u_visc_tot,
                                           double dt)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }

    u_visc_tot[i] += Thydro(double(du_visc[i]) * dt);
}

template<typename Thydro>
void integrateDissipationGPU(size_t first, size_t last, const Thydro* du_visc, Thydro* u_visc_tot, double dt)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    if (numParticles == 0) { return; }

    integrateDissipationKernel<<<numBlocks, numThreads>>>(first, last, du_visc, u_visc_tot, dt);

    checkGpuErrors(cudaDeviceSynchronize());
    checkGpuErrors(cudaGetLastError());
}

#define INTEGRATE_DISSIPATION_GPU(Thydro)                                                                              \
    template void integrateDissipationGPU(size_t first, size_t last, const Thydro* du_visc, Thydro* u_visc_tot,        \
                                          double dt);

INTEGRATE_DISSIPATION_GPU(double);
INTEGRATE_DISSIPATION_GPU(float);

} // namespace disk

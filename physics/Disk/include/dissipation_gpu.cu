//
// Created by Noah Kubli on 12.11.2026.
//

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/tree/definitions.h"

#include "dissipation_gpu.hpp"
#include "star_data.hpp"

namespace disk
{

template<typename Treal, typename Thydro>
__global__ void integrateDissipationKernel(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                           const Thydro* du_visc, Treal* u_visc_tot, Treal* u_visc_restr, double dt,
                                           const cstone::Vec3<double> star_position, double r2)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }

    const double inc = double(du_visc[i]) * dt;
    u_visc_tot[i] += inc;

    const double dx    = x[i] - star_position[0];
    const double dy    = y[i] - star_position[1];
    const double dz    = z[i] - star_position[2];
    const double dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 > r2) { u_visc_restr[i] += inc; }
}

template<typename Treal, typename Thydro>
void integrateDissipationGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                             const Thydro* du_visc, Treal* u_visc_tot, Treal* u_visc_restr, double dt,
                             const StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    if (numParticles == 0) { return; }

    integrateDissipationKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, du_visc, u_visc_tot, u_visc_restr, dt,
                                                          star.position,
                                                          star.r_shock_restr * star.r_shock_restr);

    checkGpuErrors(cudaDeviceSynchronize());
    checkGpuErrors(cudaGetLastError());
}

#define INTEGRATE_DISSIPATION_GPU(Treal, Thydro)                                                                       \
    template void integrateDissipationGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,   \
                                          const Thydro* du_visc, Treal* u_visc_tot, Treal* u_visc_restr, double dt,   \
                                          const StarData& star);

INTEGRATE_DISSIPATION_GPU(double, double);
INTEGRATE_DISSIPATION_GPU(double, float);

} // namespace disk

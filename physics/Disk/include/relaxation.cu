
#include "cstone/cuda/cuda_utils.cuh"
#include "relaxation_gpu.hpp"

namespace disk
{

template<typename T, typename Th>
__global__ void moveToLocalMinimumKernel(size_t first, size_t last, T* x, T* y, T* z, const Th* h, const Th* ax,
                                         const Th* ay, const Th* az, Th* vx, Th* vy, Th* vz, const Th* dt, float minDt,
                                         const cstone::Box<T> box)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }

    double dx = 0.5 * dt[i] * dt[i] * ax[i];
    double dy = 0.5 * dt[i] * dt[i] * ay[i];
    double dz = 0.5 * dt[i] * dt[i] * az[i];

    const double d2 = dx * dx + dy * dy + dz * dz;
    const double h2 = h[i] * h[i];

    if (d2 > h2)
    {
        double factor = std::sqrt(h2 / d2);
        dx *= factor;
        dy *= factor;
        dz *= factor;
    }

    //    vx[i] = dx / minDt;
    //    vy[i] = dy / minDt;
    //    vz[i] = dz / minDt;
    vx[i] = 0.;
    vy[i] = 0.;
    vz[i] = 0.;
    x[i]  = x[i] + dx;
    y[i]  = y[i] + dy;
    z[i]  = z[i] + dz;
}

template<typename T, typename Th>
void moveToLocalMinimumGPU(size_t first, size_t last, T* x, T* y, T* z, const Th* h, const Th* ax, const Th* ay,
                           const Th* az, Th* vx, Th* vy, Th* vz, const Th* dt, float minDt, const cstone::Box<T>& box)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    moveToLocalMinimumKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, h, ax, ay, az, vx, vy, vz, dt, minDt,
                                                        box);

    checkGpuErrors(cudaDeviceSynchronize());
}

#define MOVE_TO_LOCAL_MINIMUM_GPU(T, Th)                                                                               \
    template void moveToLocalMinimumGPU(size_t, size_t, T*, T*, T*, const Th*, const Th*, const Th*, const Th*, Th*,   \
                                        Th*, Th*, const Th*, float, const cstone::Box<T>&);

MOVE_TO_LOCAL_MINIMUM_GPU(double, float);

} // namespace disk
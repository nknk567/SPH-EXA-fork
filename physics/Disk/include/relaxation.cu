
#include "cstone/cuda/cuda_utils.cuh"
#include "relaxation_gpu.hpp"

namespace disk
{

template<typename T, typename Th>
__global__ void moveToLocalMinimumKernel(size_t first, size_t last, T* x, T* y, T* z, const Th* ax, const Th* ay,
                                         const Th* az, Th* vx, Th* vy, Th* vz, const Th* dt, float minDt, const cstone::Box<T> box)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }

    double delta_x = 0.5 * dt[i] * dt[i] * ax[i];
    double delta_y = 0.5 * dt[i] * dt[i] * ay[i];
    double delta_z = 0.5 * dt[i] * dt[i] * az[i];

    vx[i] = delta_x / minDt;
    vy[i] = delta_y / minDt;
    vz[i] = delta_z / minDt;

    x[i] = x[i] + delta_x;
    y[i] = y[i] + delta_y;
    z[i] = z[i] + delta_z;
}

template<typename T, typename Th>
void moveToLocalMinimumGPU(size_t first, size_t last, T* x, T* y, T* z, const Th* ax, const Th* ay, const Th* az,
                            Th* vx,  Th* vy,  Th* vz, const Th* dt, const cstone::Box<T>& box)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    moveToLocalMinimumKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, ax, ay, az, dt, box);

    checkGpuErrors(cudaDeviceSynchronize());
}

#define MOVE_TO_LOCAL_MINIMUM_GPU(T, Th)                                                                               \
    template void moveToLocalMinimumGPU(size_t, size_t, T*, T*, T*, const Th*, const Th*, const Th*,  Th*,        \
                                         Th*,  Th*, const Th*, const cstone::Box<T>&);

MOVE_TO_LOCAL_MINIMUM_GPU(double, float);

} // namespace disk
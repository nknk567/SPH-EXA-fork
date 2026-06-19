
#include "relaxation_gpu.hpp"

namespace disk
{

template<typename T, typename Th>
__device__ void moveToLocalMinimumKernel(size_t first, size_t last, T* x, T* y, T* z, Th* ax, Th* ay, Th* az, Th* dt,
                                         const cstone::Box<T> box)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }
    x[i] = x[i] + 0.5 * dt * dt * ax[i];
    y[i] = y[i] + 0.5 * dt * dt * ay[i];
    z[i] = z[i] + 0.5 * dt * dt * az[i];
}

template<typename T, typename Th>
void moveToLocalMinimumGPU(size_t first, size_t last, T* x, T* y, T* z, const Th* ax, const Th* ay, const Th* az,
                           const Th* dt, const cstone::Box<T>& box)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    moveToLocalMinimumKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, ax, ay, az, dt, box);

    checkGpuErrors(cudaDeviceSynchronize());
}

#define MOVE_TO_LOCAL_MINIMUM_GPU(T, Th)                                                                           \
    template void moveToLocalMinimumGPU(size_t, size_t, T*, T*, T*, const Th*, const Th*, const Th*, const Th*,        \
                                        const cstone::Box<T>&);

MOVE_TO_LOCAL_MINIMUM_GPU(double, float);

} // namespace disk
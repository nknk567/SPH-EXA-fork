//
// Created by Noah Kubli on 19.03.2025.
//
#include "acceleration_timestep_gpu.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/tree/definitions.h"

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace sph
{

template<typename T, typename Th>
struct TimestepFactor
{
    __device__ T operator()(const thrust::tuple<T, T, T, Th>& xyzh)
    {
        const auto         h_i = thrust::get<3>(xyzh);
        const cstone::Vec3 A{thrust::get<0>(xyzh), thrust::get<1>(xyzh), thrust::get<2>(xyzh)};
        return h_i * h_i / norm2(A);
    }
};

template<typename T, typename Th>
T accelerationTimestepGPU(size_t first, size_t last, const T* x, const T* y, const T* z, const Th* h)
{
    auto begin = thrust::make_zip_iterator(x + first, y + first, z + first, h + first);
    auto end   = thrust::make_zip_iterator(x + last, y + last, z + last, h + last);
    return thrust::transform_reduce(thrust::device, begin, end, TimestepFactor<T, Th>{}, INFINITY,
                                    thrust::minimum<T>{});
}

#define ACCELERATION_TIMESTEP_GPU(T, Th)                                                                               \
    template T accelerationTimestepGPU(size_t first, size_t last, const T* x, const T* y, const T* z, const Th* h);

ACCELERATION_TIMESTEP_GPU(double, double);
ACCELERATION_TIMESTEP_GPU(double, float);

} // namespace sph
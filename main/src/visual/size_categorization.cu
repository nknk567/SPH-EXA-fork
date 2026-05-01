//
// Created by Noah Kubli on 01.05.2026.
//

#include "grid.hpp"

#include "cstone/primitives/primitives_gpu.h"
#include "cstone/cuda/device_vector.h"
#include "size_categorization_helper.hpp"

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace visual
{
template<typename Tc>
struct CategorizeFunctor
{
    const Grid g;

    template<typename Tuple>
    __device__ __host__ Tc operator()(const Tuple& t) const
    {
        const double x = thrust::get<0>(t);
        const double y = thrust::get<1>(t);
        const double z = thrust::get<2>(t);
        const float  h = thrust::get<3>(t);

        return categorize<Tc>(x, y, z, h, g);
    }
};

struct Counts
{
    size_t small;
    size_t large;
};

__host__ __device__ Counts operator+(const Counts& a, const Counts& b)
{
    return {a.small + b.small, a.large + b.large};
}

template<typename T, typename Th, typename Tc>
std::tuple<size_t, size_t> sizeCategorizationGPU(size_t startIndex, size_t endIndex, T* x, T* y, T* z, Th* h,
                                                 const Grid& g, Tc& tiles)
{
    const size_t n = endIndex - startIndex;
    auto         begin =
        thrust::make_zip_iterator(thrust::make_tuple(x + startIndex, y + startIndex, z + startIndex, h + startIndex));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(x + endIndex, y + endIndex, z + endIndex, h + endIndex));
    //    auto end = begin + n;

    using RenderCategoryType = typename Tc::value_type;
    thrust::transform(thrust::device, begin, end, tiles.begin(), CategorizeFunctor<RenderCategoryType>(g));

    const size_t n_small = thrust::count(thrust::device, tiles.begin(), tiles.end(), 0);
    const size_t n_large = thrust::count(thrust::device, tiles.begin(), tiles.end(), 1);
    //    auto result = thrust::transform_reduce(
    //        tiles.begin(), tiles.end(),
    //        [] __host__ __device__(int v)
    //        {
    //            Counts c{0, 0};
    //            if (v == 0) c.small = 1;
    //            if (v == 1) c.large = 1;
    //            return c;
    //        },
    //        Counts{0, 0}, thrust::plus<Counts>());
    //
    //    size_t n_small = result.small;
    //    size_t n_large = result.large;
    //    thrust::copy(tiles_gpu.begin(), tiles_gpu.end(), tiles.begin());

    return {n_small, n_large};
}

template std::tuple<size_t, size_t> sizeCategorizationGPU(size_t, size_t, double*, double*, double*, float*,
                                                          const Grid&, cstone::DeviceVector<size_t>&);
template std::tuple<size_t, size_t> sizeCategorizationGPU(size_t, size_t, double*, double*, double*, float*,
                                                          const Grid&, cstone::DeviceVector<uint8_t>&);

} // namespace visual

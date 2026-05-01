//
// Created by Noah Kubli on 02.05.2026.
//

#include "grid.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>

namespace visual
{
struct TileCountFunctor
{
    const Grid    g;
    const double* x;
    const double* y;
    const double* z;
    const float*  h;

    size_t* tile_counts;

    __device__ void operator()(size_t i) const
    {
        const double xi = x[i];
        const double yi = y[i];
        const double zi = z[i];
        const double hi = h[i];

        const auto [ix_min, ix_max, iy_min, iy_max] = particleTiles(xi, yi, zi, hi, g);

        for (size_t ix = ix_min; ix < ix_max; ix++)
        {
            for (size_t iy = iy_min; iy < iy_max; iy++)
            {
                const size_t tile_id = iy * g.n_tiles_x + ix;

                atomicAdd(&tile_counts[tile_id], 1);
            }
        }
    }
};

template<typename T, typename Th, typename Tm, typename Trho, typename Ta, typename Tw>
void computeTileCountGPU(size_t startIndex, size_t endIndex, const Grid& g, const Ta* a, const T* x, const T* y,
                         const T* z, const Th* h, const Tm* m, const Trho* rho, const Tw* w, size_t* tile_counts)
{
    thrust::for_each(thrust::counting_iterator<size_t>(startIndex), thrust::counting_iterator<size_t>(endIndex),
                     TileCountFunctor{g, x, y, z, h, tile_counts});
}

#define COMPUTE_TILE_COUNT(T, Th, Tm, Trho, Ta, Tw)                                                                    \
    template void computeTileCountGPU(size_t, size_t, const Grid&, const Ta*, const T*, const T*, const T*, const Th*, \
                                      const Tm*, const Trho*, const Tw*, size_t*);
COMPUTE_TILE_COUNT(double, float, float, float, float, float);

} // namespace visual
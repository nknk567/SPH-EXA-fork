//
// Created by Noah Kubli on 02.05.2026.
//

#include "grid.hpp"
#include "render_medium_gpu.hpp"
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
struct TileListFunctor
{
    Grid g;

    const double* x;
    const double* y;
    const double* z;
    const float*  h;

    const size_t* tile_offsets;
    size_t*       tile_fill;
    size_t*       tile_list;

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

                // atomic increment → gives unique slot
                const size_t local_id = atomicAdd(&tile_fill[tile_id], size_t(1));

                const size_t offset = tile_offsets[tile_id];

                tile_list[offset + local_id] = i;
            }
        }
    }
};

template<typename T, typename Th>
void computeTileListGPU(size_t startIndex, size_t endIndex, const T* x, const T* y, const T* z, const Th* h,
                        const Grid& g, const size_t* tile_offsets, size_t* tile_list)
{
    const size_t n_tiles = g.n_tiles;

    // tile_fill = per-tile write counters
    thrust::device_vector<size_t> tile_fill(n_tiles, 0);

    // launch
    thrust::for_each(
        thrust::counting_iterator<size_t>(startIndex), thrust::counting_iterator<size_t>(endIndex),
        TileListFunctor{g, x, y, z, h, tile_offsets, thrust::raw_pointer_cast(tile_fill.data()), tile_list});
}

#define COMPUTE_TILE_LIST_GPU(T, Th)                                                                                   \
    template void computeTileListGPU(size_t, size_t, const T*, const T*, const T*, const Th*, const Grid&,             \
                                     const size_t*, size_t*);
COMPUTE_TILE_LIST_GPU(double, float);

} // namespace visual
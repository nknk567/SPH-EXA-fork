//
// Created by Noah Kubli on 02.05.2026.
//

#include <span>

#include "grid.hpp"

#include "evaluate.hpp"
#include "render_medium_gpu.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/cuda/device_vector.h"

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

template<typename T, typename Ta, typename Th, typename Tm, typename Trho, typename Twh>
__global__ void renderTilesKernel(Grid g, const Ta* A, const T* __restrict__ x, const T* __restrict__ y,
                                  const T* __restrict__ z, const Th* __restrict__ h, const Tm* __restrict__ m,
                                  const Trho* rho, const Twh* __restrict__ wh, const size_t* __restrict__ tile_offsets,
                                  const size_t* __restrict__ tile_lists, double K, double* pixels)
{
    const int tile_id = blockIdx.x;
    const int tile_ix = tile_id % g.n_tiles_x;
    const int tile_iy = tile_id / g.n_tiles_x;

    const int tid      = threadIdx.x;
    const int local_ix = tid % g.tile_size;
    const int local_iy = tid / g.tile_size;

    const int pixel_ix = tile_ix * g.tile_size + local_ix;
    const int pixel_iy = tile_iy * g.tile_size + local_iy;

    if (pixel_ix >= g.pixel_width || pixel_iy >= g.pixel_height) return;

    const auto pixel_id = flattenPixel(pixel_ix, pixel_iy, g);

    //    auto contribution = evaluate(ix, iy, g, A, x, y, z, h, m, rho, wh) * K;

    //    const int pixel_id = pixel_iy * g.pixel_width + pixel_ix;

    const double x_pixel = g.pixel_x(pixel_ix);
    const double y_pixel = g.pixel_y(pixel_iy);

    const size_t particles_begin = tile_offsets[tile_id];
    const size_t particles_end   = tile_offsets[tile_id + 1];

    double contribution{};
    for (size_t k = particles_begin; k < particles_end; k++)
    {
        const size_t p_id = tile_lists[k];
        contribution +=
            evaluate(x_pixel, y_pixel, g, A[p_id], x[p_id], y[p_id], z[p_id], h[p_id], m[p_id], rho[p_id], wh);
    }

    pixels[pixel_id] += contribution * K;
}
template<typename T, typename Ta, typename Th, typename Tm, typename Trho, typename Twh>
void renderMediumGPU(size_t startIndex, size_t endIndex, const Ta* a, const T* x, const T* y, const T* z, const Th* h,
                     const Tm* m, const Trho* rho, const Grid& g, const Twh* wh, T K, double* pixels,
                     size_t* tile_offsets, size_t* tile_lists)
{
    const unsigned numBlocks  = g.n_tiles;
    const unsigned numThreads = g.tile_size * g.tile_size;
    //    thrust::device_vector<double> pixels_gpu(pixels.size());

    //    thrust::copy(pixels.begin(), pixels.end(), pixels_gpu.begin());
    renderTilesKernel<<<numBlocks, numThreads>>>(g, rho, x, y, z, h, m, rho, wh, tile_offsets, tile_lists, K, pixels);
    //    thrust::copy(pixels_gpu.begin(), pixels_gpu.end(), pixels.begin());
}

#define RENDER_MEDIUM_GPU(T, Ta, Th, Tm, Trho, Twh)                                                                    \
    template void renderMediumGPU(size_t, size_t, const Ta*, const T*, const T*, const T*, const Th*, const Tm*,       \
                                  const Trho*, const Grid&, const Twh*, T, double*, size_t*, size_t*);

RENDER_MEDIUM_GPU(double, float, float, float, float, float);
} // namespace visual

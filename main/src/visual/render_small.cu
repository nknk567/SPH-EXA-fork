//
// Created by Noah Kubli on 01.05.2026.
//

#include "grid.hpp"
#include "render_small_gpu.hpp"
#include "RenderFieldsSpan.hpp"

#include "cstone/primitives/primitives_gpu.h"
#include "cstone/cuda/device_vector.h"
#include "evaluate.hpp"
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
__device__ inline size_t discretize(const double x, const double xmin, const double delta_x, const double width)
{
    const double x_rel       = (x - xmin) / delta_x;
    const double x_rel_clamp = std::clamp(x_rel, 0., width - 1.0);
    const size_t ix          = static_cast<size_t>(x_rel_clamp);
    return ix;
}
template<typename Twh, typename Tk>
struct renderFunctor
{

    const Grid g;
    const Twh* wh;
    const Tk   K;

    template<typename Tuple>
    __device__ auto operator()(const Tuple& X) const
    {
        const auto x   = thrust::get<0>(X);
        const auto y   = thrust::get<1>(X);
        const auto z   = thrust::get<2>(X);
        const auto h   = thrust::get<3>(X);
        const auto m   = thrust::get<4>(X);
        const auto rho = thrust::get<5>(X);
        const auto A   = thrust::get<6>(X);

        const size_t ix = discretize(x, g.xmin, g.delta, g.pixel_width);
        const size_t iy = discretize(y, g.ymin, g.delta, g.pixel_height);

        auto pixel_index = flattenPixel(ix, iy, g);

        auto contribution = evaluate(ix, iy, g, A, x, y, z, h, m, rho, wh) * K;

        return thrust::make_tuple(pixel_index, contribution);
    }
};

template<typename RenderSpan, typename Twh, typename PixelsVecType, typename T, typename PixelIndexBuffer,
         typename PixelValueBuffer>
void renderSmallGPU(const RenderSpan& rs, const Grid& g, Twh* wh, T K, PixelsVecType& pixels,
                    PixelIndexBuffer& pixel_index_buffer1, PixelIndexBuffer& pixel_index_buffer2,
                    PixelValueBuffer& pixel_value_buffer1, PixelValueBuffer& pixel_value_buffer2)
{
    static_assert(std::is_integral_v<typename PixelIndexBuffer::value_type>);
    static_assert(std::is_same_v<typename PixelsVecType::value_type, typename PixelValueBuffer::value_type>);

    const size_t n_particles  = rs.size; // endIndex - startIndex;
    auto&        pixel_index  = pixel_index_buffer1;
    auto&        contribution = pixel_value_buffer1;

    //    thrust::device_vector<size_t> pixel_index(n_particles);
    //    thrust::device_vector<double> contribution(n_particles);

    auto begin =
        thrust::make_zip_iterator(thrust::make_tuple(rs.x, rs.y, rs.z, rs.h, rs.m, rs.rho, rs.render_quantity));

    auto end = begin + n_particles;
    pixel_index.resize(n_particles);
    contribution.resize(n_particles);
    //    thrust::transform(thrust::device, begin, end,
    //                      thrust::make_zip_iterator(thrust::make_tuple(pixel_index.begin(), contribution.begin())),
    //                      renderFunctor{g, wh, K});
    thrust::transform(thrust::device, begin, end,
                      thrust::make_zip_iterator(thrust::make_tuple(rawPtr(pixel_index), contribution.begin())),
                      renderFunctor{g, wh, K});

    thrust::sort_by_key(thrust::device, pixel_index.begin(), pixel_index.end(), contribution.begin());

    //    thrust::device_vector<size_t> out_keys(n_particles);
    //    thrust::device_vector<double> out_vals(n_particles);
    auto& out_keys = pixel_index_buffer2;
    auto& out_vals = pixel_value_buffer2;
    out_keys.resize(n_particles);
    out_vals.resize(n_particles);

    auto   new_end = thrust::reduce_by_key(thrust::device, pixel_index.begin(), pixel_index.end(), contribution.begin(),
                                           out_keys.begin(), out_vals.begin());
    size_t n_out   = new_end.first - out_keys.begin();

    cstone::scatterGpu(thrust::raw_pointer_cast(out_keys.data()), n_out, thrust::raw_pointer_cast(out_vals.data()),
                       thrust::raw_pointer_cast(pixels.data()));
}

template void renderSmallGPU(const RenderFieldsSpan<double, float, float, float, float>&, const Grid&, float*, double,
                             cstone::DeviceVector<double>&, cstone::DeviceVector<size_t>&,
                             cstone::DeviceVector<size_t>&, cstone::DeviceVector<double>&,
                             cstone::DeviceVector<double>&);

//#define RENDER_SMALL_GPU(T, Th, Tm, Ta, Trho, Twh)                                                                     \
//    template void renderSmallGPU(size_t, size_t, Ta*, T*, T*, T*, Th*, Tm*, Trho*, const Grid&, Twh*, T,               \
//                                 std::span<double>);
//
// RENDER_SMALL_GPU(double, float, float, float, float, float);

} // namespace visual

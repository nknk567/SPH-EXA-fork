//
// Created by Noah Kubli on 06.03.2026.
//

#pragma once

#include <vector>

#include "evaluate.hpp"
#include "grid.hpp"
#include "render_small_gpu.hpp"
#include "RenderFieldsSpan.hpp"

namespace visual
{

inline size_t discretize(const double x, const double xmin, const double delta_x, const double width)
{
    const double x_rel       = (x - xmin) / delta_x;
    const double x_rel_clamp = std::clamp(x_rel, 0., width - 1.0);
    const size_t ix          = static_cast<size_t>(x_rel_clamp);
    return ix;
}

template<typename Ta, typename T, typename Th, typename Tm, typename Trho, typename Tw>
void renderSmallImpl(size_t startIndex, size_t endIndex, const Ta* A, const T* x, const T* y, const T* z, const Th* h,
                     const Tm* m, const Trho* rho, const Tw* w, const auto K, const Grid& g, auto& pixels)
{
    std::vector<size_t> pixel_index(endIndex - startIndex);
    std::vector<double> contribution(endIndex - startIndex);
    for (size_t i = startIndex; i < endIndex; i++)
    {
        //        const size_t ix = discretize(d.x[i], g.xmin, g.delta(), g.pixel_width);
        //        const size_t iy = discretize(d.y[i], g.ymin, g.delta(), g.pixel_height);
        const size_t ix = discretize(x[i], g.xmin, g.delta, g.pixel_width);
        const size_t iy = discretize(y[i], g.ymin, g.delta, g.pixel_height);

        pixel_index[i - startIndex] = flattenPixel(ix, iy, g);

        //        const auto A = d.rho[i];

        contribution[i - startIndex] = evaluate(ix, iy, g, A[i], x[i], y[i], z[i], h[i], m[i], rho[i], w) * K;
    }

    for (size_t i = 0; i < pixel_index.size(); i++)
    {
        pixels[pixel_index[i]] += contribution[i];
    }
}

template<class Dataset>
void renderSmall(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g, auto& pixelsVec)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        auto fields = makeRenderFieldsSpan<"rho">(d, startIndex, endIndex);
        //        renderSmallGPU(startIndex, endIndex, fields, g, rawPtr(d.wh), d.K, pixelsVec);
        cstone::DeviceVector<size_t> buf1, buf2;
        cstone::DeviceVector<double> buf3, buf4;
        renderSmallGPU(fields, g, rawPtr(d.wh), d.K, pixelsVec, buf1, buf2, buf3, buf4);
    }
    else
    {
        renderSmallImpl(startIndex, endIndex, rawPtr(d.rho), rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h),
                        rawPtr(d.m), rawPtr(d.rho), d.wh.data(), d.K, g, pixelsVec);
    }

    //    const auto x   = toHost(d.x);
    //    const auto y   = toHost(d.y);
    //    const auto z   = toHost(d.z);
    //    const auto h   = toHost(d.h);
    //    const auto m   = toHost(d.m);
    //    const auto rho = toHost(d.rho);
    //    const auto wh  = toHost(d.wh);
    //
    //    renderSmallImpl(startIndex, endIndex, rawPtr(rho), rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(m),
    //                    rawPtr(rho), rawPtr(wh), d.K, g, pixels);
}

} // namespace visual
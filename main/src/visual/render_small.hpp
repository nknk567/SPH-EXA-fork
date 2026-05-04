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

void renderSmallImpl(const auto& fields_span, const Grid& g, const auto* w, const auto K, auto& pixels)
{
    std::vector<size_t> pixel_index(fields_span.size);
    std::vector<double> contribution(fields_span.size);
    for (size_t i = 0; i < fields_span.size; i++)
    {
        const size_t ix = discretize(fields_span.x[i], g.xmin, g.delta, g.pixel_width);
        const size_t iy = discretize(fields_span.y[i], g.ymin, g.delta, g.pixel_height);

        pixel_index[i] = flattenPixel(ix, iy, g);

        const auto& f = fields_span;
        contribution[i] =
            evaluate(ix, iy, g, f.render_quantity[i], f.x[i], f.y[i], f.z[i], f.h[i], f.m[i], f.rho[i], w) * K;
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
        auto                         fields = makeRenderFieldsSpan<"rho">(d, startIndex, endIndex);
        cstone::DeviceVector<size_t> buf1, buf2;
        cstone::DeviceVector<double> buf3, buf4;
        renderSmallGPU(fields, g, rawPtr(d.wh), d.K, pixelsVec, buf1, buf2, buf3, buf4);
    }
    else
    {
        auto fields = makeRenderFieldsSpan<"rho">(d, startIndex, endIndex);
        renderSmallImpl(fields, g, d.wh.data(), d.K, pixelsVec);
    }
}

} // namespace visual
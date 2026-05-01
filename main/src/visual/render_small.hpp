//
// Created by Noah Kubli on 06.03.2026.
//

#pragma once

#include <vector>

#include "evaluate.hpp"
#include "grid.hpp"
#include "render_small_gpu.hpp"

namespace visual
{

inline size_t discretize(const double x, const double xmin, const double delta_x, const double width)
{
    const double x_rel       = (x - xmin) / delta_x;
    const double x_rel_clamp = std::clamp(x_rel, 0., width - 1.0);
    const size_t ix          = static_cast<size_t>(x_rel_clamp);
    return ix;
}

template<typename Dataset>
void renderSmallImpl(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g, auto& pixels)
{
    std::vector<size_t> pixel_index(endIndex - startIndex);
    std::vector<double> contribution(endIndex - startIndex);
    for (size_t i = startIndex; i < endIndex; i++)
    {
        const size_t ix = discretize(d.x[i], g.xmin, g.delta(), g.pixel_width);
        const size_t iy = discretize(d.y[i], g.ymin, g.delta(), g.pixel_height);

        pixel_index[i - startIndex] = flattenPixel(ix, iy, g);

        const auto A = d.rho[i];
        contribution[i - startIndex] =
            evaluate(ix, iy, g, A, d.x[i], d.y[i], d.z[i], d.h[i], d.m[i], d.rho[i], d.wh.data()) * d.K;
    }

    for (size_t i = 0; i < pixel_index.size(); i++)
    {
        pixels[pixel_index[i]] += contribution[i];
    }
}

template<class Dataset>
void renderSmall(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g, std::vector<double>& pixels)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        renderSmallGPU(startIndex, endIndex, rawPtr(d.rho), rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h),
                       rawPtr(d.m), rawPtr(d.rho), g, rawPtr(d.wh), d.K, pixels);
    }
    else { renderSmallImpl(startIndex, endIndex, d, g, pixels); }
}

} // namespace visual
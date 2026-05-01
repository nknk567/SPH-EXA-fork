//
// Created by Noah Kubli on 06.03.2026.
//

#pragma once

#include <algorithm>
#include "grid.hpp"
#include "sph/table_lookup.hpp"

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

        //        const auto x_pixel = g.xmin + ix * g.delta();
        //        const auto y_pixel = g.ymin + iy * g.delta();

        const auto x_pixel = g.pixel_x(ix);
        const auto y_pixel = g.pixel_y(iy);

        using HType = typename decltype(d.h)::value_type;
        //        const auto h_lim = std::max(d.h[i], HType(g.delta() / 2.));
        const auto h_lim  = limit_h(d.h[i], g);
        const auto h_inv  = 1.0 / h_lim;
        const auto h3_inv = h_inv * h_inv * h_inv;

        const auto dx = x_pixel - d.x[i];
        const auto dy = y_pixel - d.y[i];
        const auto dz = g.z - d.z[i];

        const auto   dist   = std::sqrt(dx * dx + dy * dy + dz * dz);
        const HType  vloc   = dist * h_inv;
        const auto   w      = sph::lt::lookup(d.wh.data(), vloc);
        const double factor = d.m[i] / d.rho[i] * w;

        // Now set the quantity to rho.
        const double A_i             = d.rho[i];
        contribution[i - startIndex] = A_i * factor * h3_inv * d.K;
    }

    // w_j = m_j / (rho_j * h_j^3)
    // A = sigma(w_j * A_j * W(r/h); h = max(h, Delta / 2)
    //    std::vector<double> result(g.pixel_width * g.pixel_height, 0.);
    for (size_t i = 0; i < pixel_index.size(); i++)
    {
        pixels[pixel_index[i]] += contribution[i];
    }
}

template<class Dataset>
void renderSmall(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g, std::vector<double>& pixels)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) {}
    else { renderSmallImpl(startIndex, endIndex, d, g, pixels); }
}

} // namespace visual
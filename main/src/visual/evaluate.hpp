//
// Created by Noah Kubli on 01.05.2026.
//

#pragma once

#include <cmath>
#include "cstone/cuda/annotation.hpp"
#include "grid.hpp"
#include "sph/table_lookup.hpp"

namespace visual
{

HOST_DEVICE_FUN double evaluate(double x_pixel, double y_pixel, const Grid& g, auto A, auto x, auto y, auto z, auto h,
                                auto m, auto rho, const auto* wh)
{
    using HType = std::decay_t<decltype(h)>;
    //        const auto h_lim = std::max(d.h[i], HType(g.delta() / 2.));
    const auto h_lim  = limit_h(h, g);
    const auto h_inv  = 1.0 / h_lim;
    const auto h3_inv = h_inv * h_inv * h_inv;

    const auto dx = x_pixel - x;
    const auto dy = y_pixel - y;
    const auto dz = g.z - z;

    const auto   dist   = std::sqrt(dx * dx + dy * dy + dz * dz);
    const HType  vloc   = dist * h_inv;
    const auto   w      = sph::lt::lookup(wh, vloc);
    const double factor = m / rho * w;

    // Now set the quantity to rho.
    //    const double A_i = d.rho[i];
    return A * factor * h3_inv;
}

HOST_DEVICE_FUN double evaluate(size_t pixel_ix, size_t pixel_iy, const Grid& g, auto A, auto x, auto y, auto z, auto h,
                                auto m, auto rho, const auto* wh)
{
    const auto x_pixel = g.pixel_x(pixel_ix);
    const auto y_pixel = g.pixel_y(pixel_iy);

    using HType = std::decay_t<decltype(h)>;
    //        const auto h_lim = std::max(d.h[i], HType(g.delta() / 2.));
    const auto h_lim  = limit_h(h, g);
    const auto h_inv  = 1.0 / h_lim;
    const auto h3_inv = h_inv * h_inv * h_inv;

    const auto dx = x_pixel - x;
    const auto dy = y_pixel - y;
    const auto dz = g.z - z;

    const auto   dist   = std::sqrt(dx * dx + dy * dy + dz * dz);
    const HType  vloc   = dist * h_inv;
    const auto   w      = sph::lt::lookup(wh, vloc);
    const double factor = m / rho * w;

    // Now set the quantity to rho.
    //    const double A_i = d.rho[i];
    return A * factor * h3_inv;
}

} // namespace visual
//
// Created by Noah Kubli on 01.05.2026.
//

#pragma once

#include <algorithm>

#include "cstone/cuda/annotation.hpp"

namespace visual
{

HOST_DEVICE_FUN inline auto outside(double x, double y, double z, double search_radius, const Grid& g)
{
    const auto dz = z - g.z;
    if (std::abs(dz) > search_radius) return true;
    const auto r2 = search_radius * search_radius - dz * dz;

    auto       x_clamp = std::clamp(x, g.xmin, g.xmax);
    auto       y_clamp = std::clamp(y, g.ymin, g.ymax);
    const auto dx      = x - x_clamp;
    const auto dy      = y - y_clamp;
    return (dx * dx + dy * dy > r2);
};

template<typename T = size_t>
HOST_DEVICE_FUN size_t categorize(auto x, auto y, auto z, auto h, const Grid& g)
{
    const double search_radius = 2. * h;
    if (outside(x, y, z, search_radius, g)) { return T(-1); }
    else if (search_radius < g.h_small_max) { return T(0); }
    else { return T(1); }
}
} // namespace visual

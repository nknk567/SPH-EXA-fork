//
// Created by Noah Kubli on 04.03.2026.
//

#pragma once

#include "cstone/cuda/annotation.hpp"
#include <cmath>
#include <tuple>

namespace visual
{

struct Grid
{
    //        double xmin = -142.;
    //        double xmax = -140.;
    //        double ymin = -189.;
    const double xmin = -500;
    const double xmax = 500;
    const double ymin = -500;

    //    double ymax         = -187.;

    const double z            = 0.;
    const size_t pixel_width  = 128;
    const size_t pixel_height = 128;
    const size_t tile_size    = 16;

    const double ymax;
    const double delta;

    const double h_small_max;
    const double tile_width;
    const size_t n_tiles_x;
    const size_t n_tiles_y;
    const size_t n_tiles;

private:
    HOST_DEVICE_FUN constexpr double h_small_max_() const noexcept { return 0.5 * delta_(); };
    HOST_DEVICE_FUN constexpr double tile_width_() const noexcept { return tile_size * delta_(); }
    HOST_DEVICE_FUN constexpr size_t n_tiles_x_() const noexcept { return (pixel_width + tile_size - 1) / tile_size; }
    HOST_DEVICE_FUN constexpr size_t n_tiles_y_() const noexcept { return (pixel_height + tile_size - 1) / tile_size; }
    HOST_DEVICE_FUN constexpr size_t n_tiles_() const noexcept { return n_tiles_x_() * n_tiles_y_(); }

    HOST_DEVICE_FUN constexpr double ymax_() const noexcept { return ymin + pixel_height * delta_(); }
    HOST_DEVICE_FUN constexpr double delta_() const noexcept { return (xmax - xmin) / pixel_width; }

public:
    HOST_DEVICE_FUN constexpr double pixel_x(size_t i) const noexcept { return xmin + (i + 0.5) * delta; }
    HOST_DEVICE_FUN constexpr double pixel_y(size_t i) const noexcept { return ymin + (i + 0.5) * delta; }

    constexpr HOST_DEVICE_FUN Grid(double xmin = -500, double xmax = 500, double ymin = -500, double z = 0.,
                                   size_t pixel_width = 1024, size_t pixel_height = 1024, size_t tile_size = 16)
        : xmin(xmin)
        , xmax(xmax)
        , ymin(ymin)
        , z(z)
        , pixel_width(pixel_width)
        , pixel_height(pixel_height)
        , tile_size(tile_size)
        , h_small_max(h_small_max_())
        , tile_width(tile_width_())
        , n_tiles_x(n_tiles_x_())
        , n_tiles_y(n_tiles_y_())
        , n_tiles(n_tiles_())
        , ymax(ymax_())
        , delta(delta_())
    {
    }
    constexpr HOST_DEVICE_FUN Grid(const Grid& g)
        : xmin(g.xmin)
        , xmax(g.xmax)
        , ymin(g.ymin)
        , z(g.z)
        , pixel_width(g.pixel_width)
        , pixel_height(g.pixel_height)
        , tile_size(g.tile_size)
        , h_small_max(h_small_max_())
        , tile_width(tile_width_())
        , n_tiles_x(n_tiles_x_())
        , n_tiles_y(n_tiles_y_())
        , n_tiles(n_tiles_())
        , ymax(ymax_())
        , delta(delta_())
    {
    }
};

template<typename T>
HOST_DEVICE_FUN auto limit_h(T h, const Grid& g)
{
    return std::max(h, T(g.delta / 2.));
}

HOST_DEVICE_FUN inline auto particleTiles(double x, double y, double z, double h, const Grid& g)
{
    const double dz = z - g.z;
    // Projected search_radius; maybe cache in a first round
    //    const auto h_lim = std::max(h, double(g.delta() / 2.));
    const auto h_lim = limit_h(h, g);

    const auto r2 = 4. * h_lim * h_lim - dz * dz;
    if (r2 < 0.) return std::make_tuple(0zu, 0zu, 0zu, 0zu);
    const auto r = std::sqrt(r2);
    //    const auto r = std::sqrt(std::max(0.0, 4*h*h - dz*dz)) + g.delta();

    const double xmin = x - r;
    const double xmax = x + r;
    const double ymin = y - r;
    const double ymax = y + r;

    const double tx_min = std::floor((xmin - g.xmin) / g.tile_width);
    const double tx_max = std::ceil((xmax - g.xmin) / g.tile_width);
    const double ty_min = std::floor((ymin - g.ymin) / g.tile_width);
    const double ty_max = std::ceil((ymax - g.ymin) / g.tile_width);

    const size_t ix_min = static_cast<size_t>(std::clamp(tx_min, 0., double(g.n_tiles_x)));
    const size_t ix_max = static_cast<size_t>(std::clamp(tx_max, 0., double(g.n_tiles_x)));
    const size_t iy_min = static_cast<size_t>(std::clamp(ty_min, 0., double(g.n_tiles_y)));
    const size_t iy_max = static_cast<size_t>(std::clamp(ty_max, 0., double(g.n_tiles_y)));

    return std::make_tuple(ix_min, ix_max, iy_min, iy_max);
}
HOST_DEVICE_FUN inline size_t flattenPixel(const size_t ix, const size_t iy, const Grid& g)
{
    return iy * g.pixel_width + ix;
}

inline auto tilePixels(const Grid& g, size_t tile_id)
{
    const size_t tile_ix = tile_id % g.n_tiles_x;
    const size_t tile_iy = tile_id / g.n_tiles_x;

    const size_t ix_min = std::min(tile_ix * g.tile_size, g.pixel_width);
    const size_t ix_max = std::min((tile_ix + 1) * g.tile_size, g.pixel_width);
    const size_t iy_min = std::min(tile_iy * g.tile_size, g.pixel_height);
    const size_t iy_max = std::min((tile_iy + 1) * g.tile_size, g.pixel_height);

    return std::make_tuple(ix_min, ix_max, iy_min, iy_max);
    // Check if pixels are out of bounds
}

} // namespace visual
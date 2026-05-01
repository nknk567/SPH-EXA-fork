//
// Created by Noah Kubli on 04.03.2026.
//

#pragma once

namespace visual
{

struct Grid
{
    //    double xmin = -142.;
    //    double xmax = -140.;
    //    double ymin = -189.;
    double xmin = -900;
    double xmax = -700;
    double ymin = 300;

    //    double ymax         = -187.;

    double z            = 0.;
    size_t pixel_width  = 2048;
    size_t pixel_height = 2048;
    //    double delta_x      = (xmax - xmin) / pixel_width;
    //    double delta_y      = (ymax - ymin) / pixel_height;
    size_t tile_size = 16;

    //    double h_medium_max = delta() * tile_size;
    double h_small_max = 0.5 * delta();

    double tile_width() const { return tile_size * delta(); }
    size_t n_tiles_x() const { return (pixel_width + tile_size - 1) / tile_size; }
    size_t n_tiles_y() const { return (pixel_height + tile_size - 1) / tile_size; }
    size_t n_tiles() const { return n_tiles_x() * n_tiles_y(); }

    double ymax() const { return ymin + pixel_height * delta(); }
    double delta() const { return (xmax - xmin) / pixel_width; }
    double pixel_x(size_t i) const { return xmin + (i + 0.5) * delta(); }
    double pixel_y(size_t i) const { return ymin + (i + 0.5) * delta(); }
};

template<typename T>
auto limit_h(T h, const Grid& g)
{
    return std::max(h, T(g.delta() / 2.));
}

auto particleTiles(double x, double y, double z, double h, const Grid& g)
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

    const double tx_min = std::floor((xmin - g.xmin) / g.tile_width());
    const double tx_max = std::ceil((xmax - g.xmin) / g.tile_width());
    const double ty_min = std::floor((ymin - g.ymin) / g.tile_width());
    const double ty_max = std::ceil((ymax - g.ymin) / g.tile_width());

    const size_t ix_min = static_cast<size_t>(std::clamp(tx_min, 0., double(g.n_tiles_x())));
    const size_t ix_max = static_cast<size_t>(std::clamp(tx_max, 0., double(g.n_tiles_x())));
    const size_t iy_min = static_cast<size_t>(std::clamp(ty_min, 0., double(g.n_tiles_y())));
    const size_t iy_max = static_cast<size_t>(std::clamp(ty_max, 0., double(g.n_tiles_y())));

    return std::make_tuple(ix_min, ix_max, iy_min, iy_max);
}
inline size_t flattenPixel(const size_t ix, const size_t iy, const Grid& g) { return iy * g.pixel_width + ix; }

auto tilePixels(const Grid& g, size_t tile_id)
{
    const size_t tile_ix = tile_id % g.n_tiles_x();
    const size_t tile_iy = tile_id / g.n_tiles_x();

    const size_t ix_min = std::min(tile_ix * g.tile_size, g.pixel_width);
    const size_t ix_max = std::min((tile_ix + 1) * g.tile_size, g.pixel_width);
    const size_t iy_min = std::min(tile_iy * g.tile_size, g.pixel_height);
    const size_t iy_max = std::min((tile_iy + 1) * g.tile_size, g.pixel_height);

    return std::make_tuple(ix_min, ix_max, iy_min, iy_max);
    // Check if pixels are out of bounds
}

} // namespace visual
//
// Created by Noah Kubli on 30.04.2026.
//

#pragma once

namespace visual
{

// void renderMediumImpl(

struct TileRenderData
{
    // Number of overlapping particles per tile
    //    std::vector<size_t> tile_counts;
    // Last element contains size of tile list
    std::vector<size_t> tile_offsets;
    std::vector<size_t> tile_list;
};

template<class Dataset>
void computeTileCount(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g, std::span<size_t> tile_offsets)
{
    std::vector<size_t> tile_counts(tile_offsets.size() - 1);
    std::fill(tile_offsets.begin(), tile_offsets.end(), size_t(0));

    for (size_t i = startIndex; i < endIndex; i++)
    {
        const auto [ix_min, ix_max, iy_min, iy_max] = particleTiles(d.x[i], d.y[i], d.z[i], d.h[i], g);
        for (size_t iy = iy_min; iy < iy_max; iy++)
            for (size_t ix = ix_min; ix < ix_max; ix++)
            {
                const size_t tile_id = iy * g.n_tiles_x() + ix;
                tile_counts[tile_id]++;
            }
    }
    //    const size_t n_last_tile = tile_offsets.back();
    std::exclusive_scan(tile_counts.begin(), tile_counts.end(), tile_offsets.begin(), 0);
    tile_offsets.back() = std::accumulate(tile_counts.begin(), tile_counts.end(), size_t(0));
    // tile_offsets[tile_offsets.size() - 2] + tile_counts.back();
}

template<class Dataset>
void computeTileList(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g,
                     std::span<const size_t> tile_offsets, std::span<size_t> tile_list)
{
    std::fill(tile_list.begin(), tile_list.end(), size_t(0));

    std::vector<size_t> tile_fill(tile_offsets.size() - 1, size_t(0));
    for (size_t i = startIndex; i < endIndex; i++)
    {
        const auto [ix_min, ix_max, iy_min, iy_max] = particleTiles(d.x[i], d.y[i], d.z[i], d.h[i], g);
        for (size_t ix = ix_min; ix < ix_max; ix++)
            for (size_t iy = iy_min; iy < iy_max; iy++)
            {
                const size_t tile_id         = iy * g.n_tiles_x() + ix;
                const size_t local_id        = tile_fill[tile_id]++;
                const size_t offset          = tile_offsets[tile_id];
                tile_list[offset + local_id] = i;
            }
    }

    //     Check
    bool same = true;
    for (size_t i = 0; i < tile_fill.size(); i++)
    {
        if (tile_fill[i] != tile_offsets[i + 1] - tile_offsets[i]) same = false;
    }
    assert(same);
}

template<class Dataset>
double renderPixel(Dataset& d, const Grid& g, size_t pixel_id, std::span<const size_t> particle_list)
{
    size_t pixel_id_y = pixel_id / g.pixel_width;
    size_t pixel_id_x = pixel_id % g.pixel_width;
    //    const auto x_pixel    = g.xmin + pixel_id_x * g.delta();
    //    const auto y_pixel    = g.ymin + pixel_id_y * g.delta();

    const auto x_pixel = g.pixel_x(pixel_id_x);
    const auto y_pixel = g.pixel_y(pixel_id_y);

    double contribution{};

    for (const auto i : particle_list)
    {
        using HType = typename decltype(d.h)::value_type;
        //        const auto h_lim = std::max(d.h[i], HType(g.delta() / 2.));
        const auto h_lim = limit_h(d.h[i], g);
        const auto hInv  = 1.0 / h_lim;
        const auto h3Inv = hInv * hInv * hInv;

        const auto dx = x_pixel - d.x[i];
        const auto dy = y_pixel - d.y[i];
        const auto dz = g.z - d.z[i];

        const auto   dist   = std::sqrt(dx * dx + dy * dy + dz * dz);
        const HType  vloc   = dist * hInv;
        const auto   w      = sph::lt::lookup(d.wh.data(), vloc);
        const double factor = d.m[i] / d.rho[i] * w;
        // Now set the quantity to rho.
        const double A_i = d.rho[i];
        contribution += A_i * factor * h3Inv;
    }
    return contribution * d.K;
}

template<class Dataset>
void renderTile(Dataset& d, const Grid& g, size_t tile_id, std::span<const size_t> tile_list, std::span<double> pixels)
{
    const auto [ix_start, ix_end, iy_start, iy_end] = tilePixels(g, tile_id);
    for (size_t iy = iy_start; iy < iy_end; iy++)
        for (size_t ix = ix_start; ix < ix_end; ix++)
        {
            // Get pixel
            const size_t pixel_id = flattenPixel(ix, iy, g); // pixel_iy * g.pixel_width + ix;
            pixels[pixel_id] += renderPixel(d, g, pixel_id, tile_list);
        }
}

template<class Dataset>
void renderTileList(Dataset& d, const Grid& g, std::span<const size_t> tile_offsets, std::span<const size_t> tile_list,
                    std::span<double> pixels)
{
    for (size_t tile_id = 0; tile_id < tile_offsets.size() - 1; tile_id++)
    {
        const auto begin = tile_offsets[tile_id];
        const auto end   = tile_offsets[tile_id + 1];
        const auto list  = tile_list.subspan(begin, end - begin);
        renderTile(d, g, tile_id, list, pixels);
    }
}

template<class Dataset>
void renderMedium(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g, std::vector<double>& pixels)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) {}
    else
    {
        TileRenderData render_data;
        render_data.tile_offsets.resize(g.n_tiles() + 1);

        computeTileCount(startIndex, endIndex, d, g, render_data.tile_offsets);
        const size_t tile_list_size = render_data.tile_offsets.back();
        printf("tile list size: %zu\n", tile_list_size);

        render_data.tile_list.resize(tile_list_size);
        computeTileList(startIndex, endIndex, d, g, render_data.tile_offsets, render_data.tile_list);

        renderTileList(d, g, render_data.tile_offsets, render_data.tile_list, pixels);
    }

    /*
     *
     * compute tile overlap count
     * prefix sum
     * allocation
     * compute tile overlap
     * render
     */
}

// template<typename Dataset>
// void computeTileImpl(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g, auto& tileView)
//{
//     auto outside = [&g](double x, double y, double z, double search_radius)
//     {
//         const auto dz = z - g.z;
//         if (std::abs(dz) > search_radius) return true;
//         const auto r2 = search_radius * search_radius - dz * dz;
//
//         auto       x_clamp = std::clamp(x, g.xmin, g.xmax);
//         auto       y_clamp = std::clamp(y, g.ymin, g.ymax);
//         const auto dx      = x - x_clamp;
//         const auto dy      = y - y_clamp;
//         return (dx * dx + dy * dy > r2);
//     };
//     for (size_t i = startIndex; i < endIndex; i++)
//     {
//         else if (search_radius < g.h_small_max)
//         {
//             tileVec[i - startIndex] = 0;
//             n_small++;
//         }
//         //        else if (d.h[i] < g.h_medium_max) { tileVec[i - startIndex] = -3; }
//         else
//         {
//             // Determine sizes as multiple of a tile
//             //            const auto     h                      = d.h[i]; // This can be projected
//
//             const auto     relation               = search_radius / (g.tile_size * g.delta_x);
//             const uint64_t min_tile_size_multiple = static_cast<uint64_t>(std::ceil(relation));
//
//             tileVec[i - startIndex] = std::max(uint64_t(1), min_tile_size_multiple);
//             n_large++;
//         }
//     }
//     return std::make_tuple(n_small, n_large);
// }
} // namespace visual

//
// Created by Noah Kubli on 04.03.2026.
//

#pragma once

#include "grid.hpp"
#include <tuple>
#include <vector>

namespace visual
{

template<typename Dataset>
std::tuple<size_t, size_t> sizeCategorizationImpl(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g,
                                                  auto& tileVec)
{
    auto outside = [&g](double x, double y, double z, double search_radius)
    {
        const auto dz = z - g.z;
        if (std::abs(dz) > search_radius) return true;
        const auto r2 = search_radius * search_radius - dz * dz;

        auto       x_clamp = std::clamp(x, g.xmin, g.xmax);
        auto       y_clamp = std::clamp(y, g.ymin, g.ymax());
        const auto dx      = x - x_clamp;
        const auto dy      = y - y_clamp;
        return (dx * dx + dy * dy > r2);
    };
    tileVec.resize(endIndex - startIndex);
    size_t n_small{};
    size_t n_large{};
    for (size_t i = startIndex; i < endIndex; i++)
    {
        const double search_radius = 2. * d.h[i];
        if (outside(d.x[i], d.y[i], d.z[i], search_radius)) { tileVec[i - startIndex] = -1; }
        else if (search_radius < g.h_small_max)
        {
            tileVec[i - startIndex] = 0;
            n_small++;
        }
        //        else if (d.h[i] < g.h_medium_max) { tileVec[i - startIndex] = -3; }
        else
        {
            // Determine sizes as multiple of a tile
            //            const auto     h                      = d.h[i]; // This can be projected

            const auto     relation               = search_radius / (g.tile_size * g.delta());
            const uint64_t min_tile_size_multiple = static_cast<uint64_t>(std::ceil(relation));

            tileVec[i - startIndex] = std::max(uint64_t(1), min_tile_size_multiple);
            n_large++;
        }
    }
    return std::make_tuple(n_small, n_large);
}

// Particles outside: -2; small particles inside: -1; rest: 0
template<class Dataset>
std::tuple<size_t, size_t> sizeCategorization(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g,
                                              std::vector<size_t>& tileVec)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { return {0, 0}; }
    else { return sizeCategorizationImpl(startIndex, endIndex, d, g, tileVec); }
}

template<class... Arrays1, class... Arrays2>
void gather(std::span<const size_t> ordering, const size_t offset, std::tuple<Arrays1&...> arrays,
            std::tuple<Arrays2&...> scratchBuffers)
{
    auto reorderArray = [ordering, &scratchBuffers, offset](auto& array)
    {
        using VectorRef  = decltype(array);
        using VectorType = std::decay_t<VectorRef>;
        auto& swapSpace  = util::pickType<decltype(array)>(scratchBuffers);
        swapSpace.resize(array.size());
        cstone::gatherAcc<IsDeviceVector<VectorType>{}>(ordering, rawPtr(array) + offset, rawPtr(swapSpace) + offset);
        swap(swapSpace, array);
    };

    util::for_each_tuple(reorderArray, arrays);
}
template<typename ConservedFields, typename DependentFields, typename Dataset>
void sortByKeysImpl(size_t startIndex, size_t endIndex, Dataset& d, auto& keysVec)
{
    std::vector<size_t> order(keysVec.size());
    std::iota(order.begin(), order.end(), 0zu);

    cstone::sort_by_key(keysVec.begin(), keysVec.end(), order.begin());

    std::vector<double>   buf1;
    std::vector<float>    buf2;
    std::vector<unsigned> buf3;
    std::vector<uint64_t> buf4;

    auto buffers = std::tie(buf1, buf2, buf3, buf4);

    using DefaultFields = util::FieldList<"x", "y", "z", "h", "m", "keys">;

    using Fields = decltype(DefaultFields{} + ConservedFields{} + DependentFields{});
    gather(std::span(std::as_const(order)), startIndex, get<Fields>(d), buffers);
}

template<typename ConservedFields, typename DependentFields, class Dataset>
void sortByKey(size_t startIndex, size_t endIndex, Dataset& d, std::vector<size_t>& keys)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) {}
    else { sortByKeysImpl<ConservedFields, DependentFields>(startIndex, endIndex, d, keys); }
}

} // namespace visual
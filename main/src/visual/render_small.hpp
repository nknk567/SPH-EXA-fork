//
// Created by Noah Kubli on 06.03.2026.
//

#pragma once

#include "grid.hpp"

namespace visual
{

template<typename Dataset>
void renderSmallImpl(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g)
{
    std::vector<size_t> pixel_index(endIndex - startIndex);
    std::vector<double> contribution(endIndex - startIndex);
    for (size_t i = startIndex; i < endIndex; i++)
    {
        pixel_index[i] = position_to_pixel(d.x[i], d.y[i], d.z[i], g);

    }
}

template<class Dataset>
void renderSmall(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) {}
    else { renderSmallImpl(startIndex, endIndex, d, g); }
}

} // namespace visual
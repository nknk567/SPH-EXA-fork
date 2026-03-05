//
// Created by Noah Kubli on 04.03.2026.
//

#pragma once

#include "grid.hpp"

namespace visual
{

template<typename Dataset>
void sizeCategorizationImpl(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g)
{
    std::vector<size_t> tile(endIndex - startIndex);
    for (size_t i = startIndex; i < endIndex; i++)
    {
        if (d.h[i] < g.h_small_max) { tile[i - startIndex] = -1; }
        else if (d.h[i] > g.h_medium_max) { tile[i - startIndex] = -2; }
    }
}

template<class Dataset>
void sizeCategorization(size_t startIndex, size_t endIndex, Dataset& d, const Grid& g)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) {}
    else { sizeCategorizationImpl(startIndex, endIndex, d); }
}
} // namespace visual
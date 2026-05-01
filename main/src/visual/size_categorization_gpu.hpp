//
// Created by Noah Kubli on 01.05.2026.
//

#pragma once

#include "grid.hpp"
#include "cstone/cuda/device_vector.h"
#include <span>
#include <tuple>

namespace visual
{

template<typename T, typename Th, typename Tc>
std::tuple<size_t, size_t> sizeCategorizationGPU(size_t startIndex, size_t endIndex, T* x, T* y, T* z, Th* h,
                                                 const Grid& g, Tc& tiles);
}

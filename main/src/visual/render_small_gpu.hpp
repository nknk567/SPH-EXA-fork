//
// Created by Noah Kubli on 01.05.2026.
//

#pragma once

#include "grid.hpp"
#include <span>

namespace visual
{
template<typename T, typename Th, typename Tm, typename Ta, typename Trho, typename Twh>
extern void renderSmallGPU(size_t startIndex, size_t endIndex, Ta* a, T* x, T* y, T* z, Th* h, Tm* m, Trho* rho,
                           const Grid& g, Twh* wh, T K, std::span<double> pixels);
}
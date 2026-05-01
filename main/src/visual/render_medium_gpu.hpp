//
// Created by Noah Kubli on 02.05.2026.
//

#pragma once

#include <span>

#include "grid.hpp"

namespace visual
{
template<typename T, typename Th, typename Tm, typename Trho, typename Ta, typename Tw>
extern void computeTileCountGPU(size_t startIndex, size_t endIndex, const Grid& g, const Ta* a, const T* x, const T* y,
                                const T* z, const Th* h, const Tm* m, const Trho* rho, const Tw* w,
                                size_t* tile_counts);

template<typename T, typename Th>
void computeTileListGPU(size_t startIndex, size_t endIndex, const T* x, const T* y, const T* z, const Th* h,
                        const Grid& g, const size_t* tile_offsets, size_t* tile_list);

template<typename T, typename Ta, typename Th, typename Tm, typename Trho, typename Twh>
void renderMediumGPU(size_t startIndex, size_t endIndex, const Ta* a, const T* x, const T* y, const T* z, const Th* h,
                     const Tm* m, const Trho* rho, const Grid& g, const Twh* wh, T K, std::span<double> pixels,
                     size_t* tile_offsets, size_t* tile_lists);


}

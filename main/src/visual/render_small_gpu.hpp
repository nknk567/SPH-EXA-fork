//
// Created by Noah Kubli on 01.05.2026.
//

#pragma once

#include "grid.hpp"
#include <span>

namespace visual
{
template<typename RenderSpan, typename Twh, typename PixelsVecType, typename T, typename PixelIndexBuffer,
         typename PixelValueBuffer>
void renderSmallGPU(const RenderSpan& rs, const Grid& g, Twh* wh, T K, PixelsVecType& pixels,
                    PixelIndexBuffer& pixel_index_buffer, PixelValueBuffer& pixel_value_buffer);


}
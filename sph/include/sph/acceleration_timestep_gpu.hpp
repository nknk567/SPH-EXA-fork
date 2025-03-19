//
// Created by Noah Kubli on 19.03.2025.
//

#pragma once

namespace sph
{
template<typename T, typename Th>
T accelerationTimestepGPU(size_t first, size_t last, const T* x, const T* y, const T* z, const Th* h);
}
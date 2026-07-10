//
// Created by Noah Kubli on 10.07.2026.
//

#ifndef SPHEXA_BH_DISTANCE_GPU_HPP
#define SPHEXA_BH_DISTANCE_GPU_HPP

#include <array>

namespace bh_merger
{
template<typename T, typename Tid>
std::array<double, 6> computeBhPositionsGPU(size_t begin, size_t end, T* x, T* y, T* z, Tid* id, Tid id_0, Tid id_1);
}

#endif // SPHEXA_BH_DISTANCE_GPU_HPP

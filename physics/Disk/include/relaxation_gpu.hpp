//
// Created by Noah Kubli on 19.06.2026.
//

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"

namespace disk
{
template<typename T, typename Th>
void moveToLocalMinimumGPU(size_t first, size_t last, T* x, T* y, T* z, const Th* ax, const Th* ay, const Th* az,
                           Th* vx, Th* vy, Th* vz, const Th* dt, float minDt, const cstone::Box<T>& box);
}

//
// Created by Noah Kubli on 19.06.2026.
//

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"

namespace disk
{
template<typename T, typename Th>
extern void moveToLocalMinimumGPU(size_t first, size_t last, T* x, T* y, T* z, Th* ax, Th* ay, Th* az, Th* dt,
                                  const cstone::Box<T>& box);
}

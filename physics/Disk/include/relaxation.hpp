//
// Created by Noah Kubli on 19.06.2026.
//

#pragma once

#include "relaxation_gpu.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"

namespace disk
{

template<typename Tc, typename Dataset>
void moveToLocalMinimumImpl(const size_t first, const size_t last, Dataset& d, const cstone::Box<Tc>& box)
{
    auto update = [](const auto x, const auto dt, const auto ax) { return x + 0.5 * dt * dt * ax; };

#pragma omp parallel for schedule(static)
    for (size_t i = first; i < last; i++)
    {
        d.x[i] = update(d.x[i], d.dtCourant[i], d.ax[i]);
        d.y[i] = update(d.y[i], d.dtCourant[i], d.ay[i]);
        d.z[i] = update(d.z[i], d.dtCourant[i], d.az[i]);
    }
}

template<typename Tc, typename Dataset>
void moveToLocalMinimum(size_t first, size_t last, Dataset& d, const cstone::Box<Tc>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        moveToLocalMinimumGPU(first, last, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.ax), rawPtr(d.ay),
                              rawPtr(d.az), rawPtr(d.dtCourant), box);
    }
    else { moveToLocalMinimumImpl(first, last, d, box); }
}

} // namespace disk

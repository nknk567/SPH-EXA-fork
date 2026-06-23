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
    auto delta = [](const auto dt, const auto ax) { return 0.5 * dt * dt * ax; };

#pragma omp parallel for schedule(static)
    for (size_t i = first; i < last; i++)
    {
        auto dx = delta(d.dtCourant[i], d.ax[i]);
        auto dy = delta(d.dtCourant[i], d.ay[i]);
        auto dz = delta(d.dtCourant[i], d.az[i]);

        d.vx[i] = dx / d.minDt;
        d.vy[i] = dy / d.minDt;
        d.vz[i] = dz / d.minDt;

        const double d2 = dx * dx + dy * dy + dz * dz;
        const double h2 = d.h[i] * d.h[i];

        if (d2 > h2)
        {
            double factor = std::sqrt(h2 / d2);
            dx *= factor;
            dy *= factor;
            dz *= factor;
        }

        d.x[i] = d.x[i] + dx;
        d.y[i] = d.y[i] + dy;
        d.z[i] = d.z[i] + dz;
    }
}

template<typename Tc, typename Dataset>
void moveToLocalMinimum(size_t first, size_t last, Dataset& d, const cstone::Box<Tc>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        moveToLocalMinimumGPU(first, last, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h), rawPtr(d.ax),
                              rawPtr(d.ay), rawPtr(d.az), rawPtr(d.vx), rawPtr(d.vy), rawPtr(d.vz), rawPtr(d.dtCourant),
                              d.minDt, box);
    }
    else { moveToLocalMinimumImpl(first, last, d, box); }
}

} // namespace disk

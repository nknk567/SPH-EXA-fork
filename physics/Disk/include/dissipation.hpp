//
// Created by Noah Kubli on 12.11.2026.
//

#pragma once

#include "dissipation_gpu.hpp"
#include "get_ptr.hpp"

namespace disk
{

/*! @brief Accumulate this step's artificial-viscosity heating per particle:
 * u_visc_tot += du_visc_tot * dt.
 *
 * du_visc_tot is the AV part of du of the current force evaluation; dt is the time step that
 * advances u with that du (first-order accumulation of a non-negative rate). The increment
 * is formed in double and added to the float accumulator. The pericentre nozzle shock is
 * not separated here; the analysis does that from the dump cadence (baseline subtraction).
 */
template<typename Dataset>
void integrateDissipationImpl(size_t first, size_t last, Dataset& d, double dt)
{
    using Tacc = typename Dataset::HydroType;

#pragma omp parallel for schedule(static)
    for (size_t i = first; i < last; i++)
    {
        d.u_visc_tot[i] += Tacc(double(d.du_visc_tot[i]) * dt);
    }
}

template<typename Dataset>
void integrateDissipation(size_t first, size_t last, Dataset& d, double dt)
{
    if constexpr (d.useGpu)
    {
        integrateDissipationGPU(first, last, getPtr<"du_visc_tot">(d), getPtr<"u_visc_tot">(d), dt);
    }
    else { integrateDissipationImpl(first, last, d, dt); }
}

} // namespace disk

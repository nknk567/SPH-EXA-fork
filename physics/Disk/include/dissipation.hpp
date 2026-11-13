//
// Created by Noah Kubli on 12.11.2026.
//

#pragma once

#include "dissipation_gpu.hpp"
#include "get_ptr.hpp"

namespace disk
{

/*! @brief Accumulate this step's artificial-viscosity heating per particle.
 *
 * u_visc_tot   += du_visc * dt for every particle,
 * u_visc_restr += du_visc * dt only while the particle is farther than star.r_shock_restr
 *                 from the central object (excludes the pericentre nozzle shock).
 *
 * du_visc_tot is the AV part of du of the current force evaluation; dt is the time step that
 * advances u with that du (first-order accumulation of a non-negative rate). Call it before
 * computePositions so the distance gate uses the positions the forces were evaluated at.
 */
template<typename Dataset, typename StarData>
void integrateDissipationImpl(size_t first, size_t last, Dataset& d, double dt, const StarData& star)
{
    const double r2 = star.r_shock_restr * star.r_shock_restr;

#pragma omp parallel for schedule(static)
    for (size_t i = first; i < last; i++)
    {
        const double inc = double(d.du_visc_tot[i]) * dt;
        d.u_visc_tot[i] += inc;

        const double dx    = d.x[i] - star.position[0];
        const double dy    = d.y[i] - star.position[1];
        const double dz    = d.z[i] - star.position[2];
        const double dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 > r2) { d.u_visc_restr[i] += inc; }
    }
}

template<typename Dataset, typename StarData>
void integrateDissipation(size_t first, size_t last, Dataset& d, double dt, const StarData& star)
{
    if constexpr (d.useGpu)
    {
        integrateDissipationGPU(first, last, getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d),
                                getPtr<"du_visc_tot">(d), getPtr<"u_visc_tot">(d), getPtr<"u_visc_restr">(d), dt,
                                star);
    }
    else { integrateDissipationImpl(first, last, d, dt, star); }
}

} // namespace disk

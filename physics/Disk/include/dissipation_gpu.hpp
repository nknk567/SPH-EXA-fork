//
// Created by Noah Kubli on 12.11.2026.
//

#pragma once

#include "star_data.hpp"

namespace disk
{
template<typename Treal, typename Thydro>
void integrateDissipationGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                             const Thydro* du_visc, Treal* u_visc_tot, Treal* u_visc_restr, double dt,
                             const StarData& star);
}

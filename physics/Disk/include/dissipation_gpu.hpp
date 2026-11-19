//
// Created by Noah Kubli on 12.11.2026.
//

#pragma once

#include <cstddef>

namespace disk
{
template<typename Thydro>
void integrateDissipationGPU(size_t first, size_t last, const Thydro* du_visc, Thydro* u_visc_tot, double dt);
}

//
// Created by Noah Kubli on 11.03.2024.
//

#pragma once

#include "star_data.hpp"
#include "cstone/traversal/groups.hpp"

namespace disk
{
// template<typename Treal, typename Thydro, typename Tmass>
// void computeCentralForceGPU(const cstone::GroupView& grp, const Treal* x, const Treal* y, const Treal* z, Thydro* ax,
//                             Thydro* ay, Thydro* az, const Tmass* m, Treal g, StarData& star);
template<typename Treal, typename Thydro, typename Tmass>
void computeCentralForceGPU(const cstone::GroupView& grp, const cstone::GroupView& active_grp, const Treal* x, const Treal* y,
                            const Treal* z, Thydro* ax, Thydro* ay, Thydro* az, const Tmass* m, Treal g, StarData& star,
                            float* groupDt);

} // namespace disk

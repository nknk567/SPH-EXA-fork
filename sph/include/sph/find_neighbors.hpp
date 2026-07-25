#pragma once

#include "cstone/traversal/groups.hpp"

#include "sph/find_neighbors_gpu.hpp"
#include "sph/neighborhood.hpp"
#include "sph/particles_data.hpp"

namespace sph
{

inline void findNeighborsSfc(const cstone::GroupView& groups, sphexa::ParticlesData<cstone::execution::Cpu>& d,
                             const cstone::Box<SphTypes::CoordinateType>& box, bool subgroups = false)
{
    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    d.neighborhood.build(groups, d, box, subgroups);
}

using cstone::GroupView;
using cstone::LocalIndex;

template<class T, class KeyType>
bool updateSmoothingLengthCpu(size_t startIndex, size_t endIndex, unsigned ng0, const unsigned* nc, T* h, KeyType* keys,
                              bool adjustH)
{
    bool keysRemoved = false;
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        if (nc[i] <= 1)
        {
            keys[i]     = cstone::removeKey<KeyType>{};
            keysRemoved = true;
        }
        if (adjustH) { h[i] = updateH(ng0, nc[i], h[i]); }

#ifndef NDEBUG
        if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%lu) ngi %d h %f\n", i, nc[i], h[i]);
#endif
    }
    return keysRemoved;
}

/*! @brief flag unresolvable particles for removal and nudge h towards the target neighbor count
 *
 * @param adjustH  if false, only the removal flagging is performed. Used when h is instead
 *                 converged with Newton-Raphson iterations during the force computation.
 */
template<class Dataset>
bool updateSmoothingLength(const GroupView& grp, Dataset& d, bool adjustH = true)
{
    using namespace cstone;
    if constexpr (d.useGpu)
    {
        bool keysRemoved = updateSmoothingLengthGpu(grp, d.ng0, rawPtr(d.nc), rawPtr(d.h), rawPtr(d.keys), adjustH);
        return keysRemoved;
    }
    else
    {
        return updateSmoothingLengthCpu(grp.firstBody, grp.lastBody, d.ng0, rawPtr(d.nc), rawPtr(d.h), rawPtr(d.keys),
                                        adjustH);
    }
}

template<class T, class Dataset>
void updateSmoothingLengthIterative(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (d.useGpu) { updateSmoothingLengthIterativeGpu(groups, d, box); }
    else
    {
        const auto* x  = d.x.data();
        const auto* y  = d.y.data();
        const auto* z  = d.z.data();
        auto*       h  = d.h.data();
        auto*       nc = d.nc.data();
#pragma omp parallel for
        for (LocalIndex i = groups.firstBody; i < groups.lastBody; ++i)
        {
            updateHIterative(d.ng0, d.ngmax, box, d.treeView, i, x, y, z, h, nc);
        }
    }
}

//! @brief neighbor-count capacity guard for Newton-Raphson controlled smoothing lengths, see updateHIterativeNR
template<class T, class Dataset>
void updateSmoothingLengthIterativeNR(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (d.useGpu) { updateSmoothingLengthIterativeNRGpu(groups, d, box); }
    else
    {
        const auto* x  = d.x.data();
        const auto* y  = d.y.data();
        const auto* z  = d.z.data();
        auto*       h  = d.h.data();
        auto*       nc = d.nc.data();
#pragma omp parallel for
        for (LocalIndex i = groups.firstBody; i < groups.lastBody; ++i)
        {
            updateHIterativeNR(d.ng0, d.ngmax, box, d.treeView, i, x, y, z, h, nc);
        }
    }
}

} // namespace sph

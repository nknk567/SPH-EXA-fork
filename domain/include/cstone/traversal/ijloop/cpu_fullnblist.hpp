/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor search on CPU
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <memory>

#include "cstone/execution.hpp"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace cpu_full_nb_list_neighborhood_detail
{

template<class Tc, class KeyType, class ThP>
struct CpuFullNbListNeighborhood
{
    OctreeNsView<Tc, KeyType> tree;
    Box<Tc> box = {0, 0};
    LocalIndex firstBody, lastBody;
    std::unique_ptr<LocalIndex[]> neighborsCount, neighbors;
    const Tc *x, *y, *z;
    ThP h;
    unsigned ngmax;

    template<class... In, class... Out, class Interaction, class Postamble>
    void ijLoop(std::tuple<In*...> const& input,
                std::tuple<Out*...> const& output,
                Interaction&& interaction,
                Postamble&& postamble) const
    {
        const auto constInput = makeConst(input);
#pragma omp parallel for simd
        for (LocalIndex i = firstBody; i < lastBody; ++i)
            jLoop(constInput, output, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble), i);
    }

    Statistics stats() const
    {
        const LocalIndex numBodies = lastBody - firstBody;
        return {.numBodies = numBodies,
                .numBytes  = sizeof(LocalIndex) * numBodies + sizeof(LocalIndex) * numBodies * ngmax};
    }

    struct Subgroup
    {
        CpuFullNbListNeighborhood const& parent;
        GroupView groups;

        template<class... In, class... Out, class Interaction, class Postamble>
        void ijLoop(std::tuple<In*...> const& input,
                    std::tuple<Out*...> const& output,
                    Interaction&& interaction,
                    Postamble&& postamble) const
        {
            const auto constInput = makeConst(input);
#pragma omp parallel for
            for (LocalIndex g = 0; g < groups.numGroups; ++g)
#pragma omp simd
                for (LocalIndex i = groups.groupStart[g]; i < groups.groupEnd[g]; ++i)
                    parent.jLoop(constInput, output, std::forward<Interaction>(interaction),
                                 std::forward<Postamble>(postamble), i);
        }
    };

    Subgroup subgroup(GroupView const& groups) const { return {*this, groups}; }

protected:
    template<class Input, class Output, class Interaction, class Postamble>
    void
    jLoop(Input&& input, Output&& output, Interaction&& interaction, Postamble&& postamble, const LocalIndex i) const
    {
        const auto iData  = loadParticleData(x, y, z, h, std::forward<Input>(input), i);
        const bool usePbc = requiresPbcHandling(box, iData);

        const unsigned nbs = neighborsCount[i - firstBody];
        auto result        = interaction(iData, iData, Vec3<Tc>{0, 0, 0}, Tc(0));
        for (unsigned nb = 0; nb < nbs; ++nb)
        {
            const LocalIndex j = neighbors[(i - firstBody) * ngmax + nb];
            const auto jData   = loadParticleData(x, y, z, h, std::forward<Input>(input), j);

            const auto [ijPosDiff, distSq] = posDiffAndDistSq(usePbc, box, iData, jData);

            /* With symmetric interactions, pairs interact within 2 * max(h_i, h_j): each side
             * of the pair force carries the other side's kernel, which is nonzero out to the
             * other side's support radius (the own-kernel terms vanish there on their own).
             * Cutting at 2 * h_i only drops the reaction to the neighbor's contribution,
             * breaking momentum and energy conservation wherever h varies across a pair. */
            if (distSq < radiusSq(iData) || (tree.symmetric && distSq < radiusSq(jData)))
            {
                updateResult(result, interaction(iData, jData, ijPosDiff, distSq));
            }
        }

        storeParticleData(std::forward<Output>(output), i, postamble(iData, unwrapModifiers(result)));
    }
};
} // namespace cpu_full_nb_list_neighborhood_detail

struct CpuFullNbListNeighborhoodBuilder
{
    unsigned ngmax;

    template<class Tc, class KeyType, class ThP>
    cpu_full_nb_list_neighborhood_detail::CpuFullNbListNeighborhood<Tc, KeyType, ThP>
    build(execution::Cpu,
          OctreeNsView<Tc, KeyType> tree,
          const Box<Tc>& box,
          const LocalIndex totalBodies,
          const GroupView& groups,
          const Tc* const x,
          const Tc* const y,
          const Tc* const z,
          const ThP h) const
    {
        using namespace cpu_full_nb_list_neighborhood_detail;

        const LocalIndex numBodies = groups.lastBody - groups.firstBody;

        CpuFullNbListNeighborhood<Tc, KeyType, ThP> nbList{
            tree,
            box,
            groups.firstBody,
            groups.lastBody,
            std::make_unique_for_overwrite<LocalIndex[]>(numBodies),
            std::make_unique_for_overwrite<LocalIndex[]>(std::size_t(numBodies) * ngmax),
            x,
            y,
            z,
            h,
            ngmax};

        using Th            = std::remove_cvref_t<std::remove_pointer_t<ThP>>;
        ThP        hExt     = h;
        const bool extended = tree.searchExtFactor != 1;
        std::unique_ptr<Th[]> hExtData;
        if (extended)
        {
            if constexpr (std::is_pointer_v<ThP>)
            {
                hExtData = std::make_unique_for_overwrite<Th[]>(totalBodies);
#pragma omp parallel for
                for (LocalIndex i = 0; i < totalBodies; ++i)
                    hExtData[i] = h[i] * tree.searchExtFactor;
                hExt = hExtData.get();
            }
            else { hExt = h * tree.searchExtFactor; }
            tree.searchExtFactor = 1;
        }

        std::unique_ptr<std::uint8_t[]> fellBack;
        if (extended) { fellBack = std::make_unique<std::uint8_t[]>(numBodies); }

        unsigned maxNeighbors = 0;
#pragma omp parallel for reduction(max : maxNeighbors)
        for (LocalIndex i = 0; i < numBodies; ++i)
        {
            unsigned found =
                findNeighbors(i + groups.firstBody, x, y, z, hExt, tree, box, ngmax, &nbList.neighbors[i * ngmax]);
            maxNeighbors = std::max(maxNeighbors, found);
            if (extended && found > ngmax)
            {
                /* The neighborhood with the extended capture radius does not fit: fall back to the
                 * plain radius for this particle instead of storing an arbitrary truncation, which
                 * could drop neighbors inside the interaction radius while keeping extended-shell
                 * entries outside of it. */
                found = findNeighbors(i + groups.firstBody, x, y, z, h, tree, box, ngmax, &nbList.neighbors[i * ngmax]);
                fellBack[i] = 1;
            }
            nbList.neighborsCount[i] = std::min(found, ngmax);
        }

        unsigned maxAugmented = 0;
        if constexpr (std::is_pointer_v<ThP>)
        {
            if (tree.symmetric)
            {
                /* Transpose augmentation: the interaction loops process pairs within
                 * 2 * max(h_i, h_j) (see jLoop), but a pair with r beyond particle j's own
                 * search radius is only present in i's list. Append the reverse entry so that
                 * j receives the reaction to the neighbor-kernel term. Readers use the
                 * pre-augmentation counts, appends go through atomic cursors, so the pass is
                 * race-free; the appended entries are in nondeterministic order. */
                auto originalCount = std::make_unique_for_overwrite<LocalIndex[]>(numBodies);
                std::copy(nbList.neighborsCount.get(), nbList.neighborsCount.get() + numBodies, originalCount.get());

                //! search radius that particle j's own list build used
                auto ownSearchRadius = [&](LocalIndex gj, LocalIndex jl)
                { return Th(2) * ((fellBack && fellBack[jl]) ? h[gj] : hExt[gj]); };
                auto appendReverse = [&](LocalIndex source, LocalIndex gj)
                {
                    const LocalIndex jl = gj - groups.firstBody;
                    const Tc r2 = distanceSq<true>(x[source], y[source], z[source], x[gj], y[gj], z[gj], box);
                    const Th searchRadiusJ = ownSearchRadius(gj, jl);
                    if (r2 < searchRadiusJ * searchRadiusJ) { return; } // already in j's own list

                    unsigned pos;
#pragma omp atomic capture
                    pos = nbList.neighborsCount[jl]++;
                    if (pos < ngmax) { nbList.neighbors[std::size_t(jl) * ngmax + pos] = source; }
                };

#pragma omp parallel for
                for (LocalIndex i = 0; i < numBodies; ++i)
                {
                    const LocalIndex gi = i + groups.firstBody;
                    for (unsigned nb = 0; nb < originalCount[i]; ++nb)
                    {
                        const LocalIndex gj = nbList.neighbors[i * ngmax + nb];
                        if (gj < groups.firstBody || gj >= groups.lastBody) { continue; }
                        appendReverse(gi, gj);
                    }
                }

                /* Halo sources: a halo particle H with a large support may cover owned
                 * particles beyond their own search radii. Halos have no stored list, so search
                 * from H directly and transpose the finds. The reaction data of H (it is
                 * evaluated as a neighbor only) comes from the halo field exchanges; H's own
                 * force is computed by its owning rank the same way. No-op without halos. */
#pragma omp parallel
                {
                    auto scratch = std::make_unique_for_overwrite<LocalIndex[]>(ngmax);
#pragma omp for
                    for (LocalIndex gh = 0; gh < totalBodies; ++gh)
                    {
                        const bool isHalo = gh < groups.firstBody || gh >= groups.lastBody;
                        if (!isHalo) { continue; }
                        const unsigned found = findNeighbors(gh, x, y, z, hExt, tree, box, ngmax, scratch.get());
                        for (unsigned nb = 0; nb < std::min(found, ngmax); ++nb)
                        {
                            const LocalIndex gj = scratch[nb];
                            if (gj < groups.firstBody || gj >= groups.lastBody) { continue; }
                            appendReverse(gh, gj);
                        }
                    }
                }

#pragma omp parallel for reduction(max : maxAugmented)
                for (LocalIndex i = 0; i < numBodies; ++i)
                {
                    maxAugmented             = std::max(maxAugmented, unsigned(nbList.neighborsCount[i]));
                    nbList.neighborsCount[i] = std::min(nbList.neighborsCount[i], ngmax);
                }
            }
        }

        if (maxNeighbors > ngmax)
        {
            if (extended)
            {
                std::cerr << "WARNING: extended neighbor search found up to " << maxNeighbors
                          << " particles, exceeding ngmax = " << ngmax
                          << ". Fell back to the unextended search radius for the affected particles." << std::endl;
            }
            else
            {
                std::cerr << "WARNING: overflow in neighbor list. Missing neighbors! Try to increase ngmax. Current "
                             "ngmax is "
                          << ngmax << ", but found up to " << maxNeighbors << " neighbor particles." << std::endl;
            }
        }
        if (maxAugmented > ngmax)
        {
            std::cerr << "WARNING: neighbor-list symmetrization needed up to " << maxAugmented
                      << " entries, exceeding ngmax = " << ngmax
                      << ". Some pair reactions across strong h contrasts are dropped." << std::endl;
        }
        return nbList;
    }
};

} // namespace cstone::ijloop

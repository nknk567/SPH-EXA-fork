/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Tests for the CPU full neighbor list interaction loop, in particular the symmetric
 *        pair processing within 2 * max(h_i, h_j)
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "cstone/traversal/ijloop/cpu_fullnblist.hpp"
#include "coord_samples/random.hpp"

using namespace cstone;

namespace
{

//! @brief counts accepted pairs (including self) and checksums the neighbor x-coordinates
struct CountInteraction
{
    template<class ParticleData, class Tc, class T>
    constexpr auto operator()(const ParticleData& /*iData*/, const ParticleData& jData, Vec3<Tc> const& /*ijPosDiff*/,
                              T /*distSq*/) const
    {
        const auto& [j, jPos, hj, xj] = jData;
        return std::make_tuple(double(1), double(xj));
    }
};

struct IdentityPostamble
{
    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& /*iData*/, const Result& result) const
    {
        return result;
    }
};

} // namespace

/*! @brief compare the pair set processed by the ijLoop against an all-to-all reference
 *
 * With symmetric interactions enabled, a pair interacts within 2 * max(h_i, h_j) (the pair force
 * carries both particles' kernels); without, within the classical gather radius 2 * h_i.
 */
template<class KeyType>
static void symmetricNeighborLists()
{
    using T          = double;
    LocalIndex n     = 400;
    unsigned ngmax   = n; // capacity handling is not under test: no truncation possible
    Box<T> box(0, 1, BoundaryType::periodic);

    RandomCoordinates<T, HilbertKey<KeyType>> coords(n, box);
    const T* x = coords.x().data();
    const T* y = coords.y().data();
    const T* z = coords.z().data();

    /* strongly heterogeneous smoothing lengths: every 20th particle has 3x the base h.
     * All search radii stay below half the periodic box length, so the minimum-image
     * all-to-all reference is unambiguous. */
    T spacing = std::cbrt(T(1) / T(n));
    std::vector<T> h(n, 0.5 * spacing);
    for (LocalIndex i = 0; i < n; i += 20)
    {
        h[i] *= 3;
    }

    unsigned bucketSize   = 64;
    auto [csTree, counts] = computeOctree<KeyType>(coords.particleKeys(), bucketSize);
    OctreeData<KeyType, execution::Cpu> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());

    std::vector<LocalIndex> layout(nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    std::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    std::vector<Vec3<T>> centers(octree.numNodes), sizes(octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box);

    OctreeNsView<T, KeyType> nsView{octree.numLeafNodes,
                                    octree.numNodes,
                                    octree.prefixes.data(),
                                    octree.childOffsets.data(),
                                    octree.parents.data(),
                                    octree.internalToLeaf.data(),
                                    octree.leafToInternal.data(),
                                    octree.levelRange.data(),
                                    nullptr,
                                    layout.data(),
                                    centers.data(),
                                    sizes.data()};

    //! all-to-all reference at the live h; the loop's self-interaction contributes (1, x_i)
    auto reference = [&](LocalIndex i, bool symmetric)
    {
        double cnt = 1, sum = x[i];
        for (LocalIndex j = 0; j < n; ++j)
        {
            if (j == i) { continue; }
            T r2 = distanceSq<true>(x[j], y[j], z[j], x[i], y[i], z[i], box);
            T ri = 2 * h[i], rj = 2 * h[j];
            if (r2 < ri * ri || (symmetric && r2 < rj * rj))
            {
                cnt += 1;
                sum += x[j];
            }
        }
        return std::make_pair(cnt, sum);
    };

    std::vector<double> cnt(n), sum(n);
    auto runLoop = [&](bool symmetric, float extFactor, GroupView grp)
    {
        OctreeNsView<T, KeyType> view = nsView;
        view.symmetric                = symmetric;
        view.searchExtFactor          = extFactor;
        auto nbList = ijloop::CpuFullNbListNeighborhoodBuilder{ngmax}.build(execution::Cpu{}, view, box, n, grp, x, y,
                                                                            z, h.data());
        std::fill(cnt.begin(), cnt.end(), 0);
        std::fill(sum.begin(), sum.end(), 0);
        nbList.ijLoop(std::make_tuple(x), std::make_tuple(cnt.data(), sum.data()), CountInteraction{},
                      IdentityPostamble{});
    };

    auto compare = [&](LocalIndex first, LocalIndex last, bool symmetric, const char* label)
    {
        SCOPED_TRACE(label);
        for (LocalIndex i = first; i < last; ++i)
        {
            auto [refCnt, refSum] = reference(i, symmetric);
            EXPECT_EQ(refCnt, cnt[i]) << "count mismatch at particle " << i;
            EXPECT_NEAR(refSum, sum[i], 1e-9) << "checksum mismatch at particle " << i;
        }
    };

    GroupView allBodies{0, n, 0, nullptr, nullptr};

    // symmetric off: classical gather semantics, pairs within 2 * h_i only
    runLoop(false, 1.0f, allBodies);
    compare(0, n, false, "off");

    // symmetric on: pairs within 2 * max(h_i, h_j), restored through the transpose augmentation
    runLoop(true, 1.0f, allBodies);
    compare(0, n, true, "on");

    // NR-style growth: lists built with a 10% capture extension stay complete when all h grow
    // by up to that margin after the build (the interaction cutoff uses the live h)
    {
        OctreeNsView<T, KeyType> view = nsView;
        view.symmetric                = true;
        view.searchExtFactor          = 1.1f;
        auto nbList = ijloop::CpuFullNbListNeighborhoodBuilder{ngmax}.build(execution::Cpu{}, view, box, n, allBodies,
                                                                            x, y, z, h.data());
        for (auto& hi : h)
        {
            hi *= 1.05;
        }
        std::fill(cnt.begin(), cnt.end(), 0);
        std::fill(sum.begin(), sum.end(), 0);
        nbList.ijLoop(std::make_tuple(x), std::make_tuple(cnt.data(), sum.data()), CountInteraction{},
                      IdentityPostamble{});
        compare(0, n, true, "growth");
        for (auto& hi : h)
        {
            hi /= 1.05;
        }
    }

    // halo sources: only an interior range is locally owned; a large-h particle outside the
    // owned range must still interact with owned particles beyond their own search radii
    {
        LocalIndex first = n / 4, last = 3 * n / 4;
        T hSave = h[5];
        h[5]    = 3.5 * 0.5 * spacing;
        GroupView owned{first, last, 0, nullptr, nullptr};
        runLoop(true, 1.0f, owned);
        compare(first, last, true, "halo-source");
        h[5] = hSave;
    }
}

TEST(IjLoopCpu, symmetricNeighborLists)
{
    symmetricNeighborLists<uint32_t>();
    symmetricNeighborLists<uint64_t>();
}

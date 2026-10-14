/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Halo discovery tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <random>

#include "gtest/gtest.h"

#include "cstone/findneighbors.hpp"
#include "cstone/traversal/collisions.hpp"
#include "cstone/tree/cs_util.hpp"

#include "collisions_a2a.hpp"

using namespace cstone;

template<class KeyType, class T>
std::vector<uint8_t> findHalosAll2All(std::span<const KeyType> nodeKeys,
                                      std::span<const TreeNodeIndex> leaf2int,
                                      const Vec3<T>* tC,
                                      const Vec3<T>* tS,
                                      TreeNodeIndex numTargets,
                                      const Box<T>& box,
                                      KeyType exclStart,
                                      KeyType exclEnd)
{
    std::vector<uint8_t> flags(nodeKeys.size());
    auto collisions = findCollisionsAll2all(nodeKeys, tC, tS, numTargets, box);

    for (size_t i = 0; i < collisions.size(); ++i)
    {
        auto [k1, k2] = decodePlaceholderBit2K(nodeKeys[leaf2int[i]]);
        if (!containedIn(k1, k2, exclStart, exclEnd)) { continue; } // select only targets in excluded range
        for (size_t cidx : collisions[i])
        {
            auto [k1, k2] = decodePlaceholderBit2K(nodeKeys[cidx]);
            if (!containedIn(k1, k2, exclStart, exclEnd)) { flags[cidx] = 1; } // only count sources outside exclusion
        }
    }

    return flags;
}

/*! @brief Test halo discovery flags on a uniform tree, optionally with mixed-dimension boxes
 *
 * @tparam     KeyType   32-bit or 64-bit SFC key type
 * @param[in]  useMixD   if true, use a non-cubic box (0,1)x(0,0.015625)x(0,0.00390625) and a 512-leaf tree
 *
 * Compares findHalos results against an all-to-all reference for both standard uniform and
 * mixed-dimension configurations. Adjust expected surface node counts for MixD geometry.
 */
template<class KeyType>
void findHalosFlags(bool useMixD = false)
{
    using T  = double;
    auto box = useMixD ? Box<double>(0, 1, 0, 1, 0, 0.5) : Box<double>(0, 1);

    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);
    OctreeData<KeyType, execution::Cpu> octree;
    octree.resize(nNodes(tree));
    updateInternalTree<KeyType>(tree, octree.data());

    std::vector<Vec3<T>> nodeCenters(octree.numNodes), nodeSizes(octree.numNodes);
    nodeFpCenters<KeyType>(octree.prefixes, nodeCenters.data(), nodeSizes.data(), box);
    auto leaf2int = leafToInternal(octree);

    // size of one node is 0.25^3
    std::vector<double> searchRadii(octree.numLeafNodes, 0.1);

    std::vector<Vec3<T>> tC(octree.numLeafNodes), tS(octree.numLeafNodes);
    for (size_t i = 0; i < size_t(octree.numLeafNodes); ++i)
    {
        tC[i] = nodeCenters[leaf2int[i]];
        tS[i] = nodeSizes[leaf2int[i]] + Vec3<T>{searchRadii[i], searchRadii[i], searchRadii[i]};
    }

    // leaf ranges [0:a] [a:b] bipartition leaf nodes with non-zero volume
    TreeNodeIndex a = useMixD ? 16 : 32;
    TreeNodeIndex b = useMixD ? 32 : 64;

    auto od = octree.data();
    {
        std::vector<uint8_t> collisionFlags(octree.numNodes, 0);
        findHalos(od.prefixes, od.childOffsets, od.parents, nodeCenters.data(), nodeSizes.data(), tree.data(),
                  tC.data(), tS.data(), box, 0, a, collisionFlags.data());

        std::vector<uint8_t> reference = findHalosAll2All<KeyType>(octree.prefixes, leaf2int, tC.data(), tS.data(),
                                                                   octree.numLeafNodes, box, tree[0], tree[a]);

        // consistency check: the surface of the first 32 nodes with the last 32 nodes is 16 nodes (+5 internal nodes)
        EXPECT_EQ(useMixD ? 11 : 21, std::accumulate(collisionFlags.begin(), collisionFlags.end(), 0));
        EXPECT_EQ(useMixD ? 11 : 21, std::accumulate(reference.begin(), reference.end(), 0));
        EXPECT_EQ(collisionFlags, reference);
    }
    {
        std::vector<uint8_t> collisionFlags(octree.numNodes, 0);
        findHalos(od.prefixes, od.childOffsets, od.parents, nodeCenters.data(), nodeSizes.data(), tree.data(),
                  tC.data(), tS.data(), box, a, b, collisionFlags.data());

        std::vector<uint8_t> reference = findHalosAll2All<KeyType>(octree.prefixes, leaf2int, tC.data(), tS.data(),
                                                                   octree.numLeafNodes, box, tree[a], tree[b]);

        // consistency check: the surface of the first 32 nodes with the last 32 nodes is 16 nodes
        EXPECT_EQ(useMixD ? 11 : 21, std::accumulate(collisionFlags.begin(), collisionFlags.end(), 0));
        EXPECT_EQ(collisionFlags, reference);
    }
}

TEST(HaloDiscovery, findHalosFlags)
{
    findHalosFlags<unsigned>();
    findHalosFlags<uint64_t>();
    findHalosFlags<unsigned>(true);
    findHalosFlags<uint64_t>(true);
}

/*! @brief transposed halo discovery: nodes are found through their OWN interaction reach
 *
 * A remote node with a large interaction reach must be discovered as a halo even when it lies
 * beyond the reach of every local search sphere — required for symmetric pair interactions
 * within 2 * max(h_i, h_j). The plain (gather-sense) findHalos with unexpanded local target
 * boxes finds nothing at all (every target is contained in the local key range), while
 * findHalosSymmetric flags exactly the nodes whose inflated boxes overlap local leaf boxes.
 */
template<class KeyType>
static void findHalosSymmetricFlags()
{
    using T  = double;
    Box<T> box(0, 1);

    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);
    OctreeData<KeyType, execution::Cpu> octree;
    octree.resize(nNodes(tree));
    updateInternalTree<KeyType>(tree, octree.data());

    std::vector<Vec3<T>> nodeCenters(octree.numNodes), nodeSizes(octree.numNodes);
    nodeFpCenters<KeyType>(octree.prefixes, nodeCenters.data(), nodeSizes.data(), box);
    auto leaf2int = leafToInternal(octree);
    auto od       = octree.data();

    // local leaves [0:a), plain (unexpanded) local target boxes
    TreeNodeIndex a = 32;
    std::vector<Vec3<T>> tC(octree.numLeafNodes), tS(octree.numLeafNodes);
    for (size_t i = 0; i < size_t(octree.numLeafNodes); ++i)
    {
        tC[i] = nodeCenters[leaf2int[i]];
        tS[i] = nodeSizes[leaf2int[i]];
    }

    // one remote leaf in the last corner gets a large interaction reach; max-upsweep to internal nodes
    TreeNodeIndex remoteLeaf = octree.numLeafNodes - 1;
    std::vector<T> expansions(octree.numNodes, 0);
    expansions[leaf2int[remoteLeaf]] = 0.6;
    std::span<const TreeNodeIndex> levelRange{od.levelRange, size_t(maxTreeLevel<KeyType>{}) + 2};
    upsweep(levelRange, od.childOffsets, expansions.data(), MaxCombination<T>{});

    std::vector<uint8_t> flags(octree.numNodes, 0);
    findHalosSymmetric(od.prefixes, od.childOffsets, od.parents, nodeCenters.data(), nodeSizes.data(),
                       expansions.data(), octree.numNodes, tree.data(), tC.data(), tS.data(), box, 0, a, flags.data());

    // all-to-all reference: nodes inflated by their own reach against plain local leaf boxes
    std::vector<uint8_t> reference(octree.numNodes, 0);
    for (TreeNodeIndex t = 0; t < a; ++t)
    {
        for (TreeNodeIndex n = 0; n < octree.numNodes; ++n)
        {
            auto [k1, k2] = decodePlaceholderBit2K(od.prefixes[n]);
            if (containedIn(k1, k2, tree[0], tree[a])) { continue; }
            if (nodeSizes[n] == Vec3<T>{0, 0, 0}) { continue; }
            Vec3<T> inflated = nodeSizes[n] + Vec3<T>{expansions[n], expansions[n], expansions[n]};
            if (overlap(nodeCenters[n], inflated, tC[t], tS[t], box)) { reference[n] = 1; }
        }
    }
    EXPECT_EQ(flags, reference);

    // the far remote leaf is discovered through its own reach ...
    EXPECT_EQ(1, flags[leaf2int[remoteLeaf]]);

    // ... while gather-sense discovery with the same plain local boxes cannot see it
    std::vector<uint8_t> plainFlags(octree.numNodes, 0);
    findHalos(od.prefixes, od.childOffsets, od.parents, nodeCenters.data(), nodeSizes.data(), tree.data(), tC.data(),
              tS.data(), box, 0, a, plainFlags.data());
    EXPECT_EQ(0, plainFlags[leaf2int[remoteLeaf]]);

    // nodes inside the local key range are never flagged
    for (TreeNodeIndex n = 0; n < octree.numNodes; ++n)
    {
        auto [k1, k2] = decodePlaceholderBit2K(od.prefixes[n]);
        if (containedIn(k1, k2, tree[0], tree[a])) { EXPECT_EQ(0, flags[n]); }
    }

    // zero reach everywhere reduces to plain box contact, identical to the reference convention
    std::vector<T> zeroExpansions(octree.numNodes, 0);
    std::vector<uint8_t> contactFlags(octree.numNodes, 0);
    findHalosSymmetric(od.prefixes, od.childOffsets, od.parents, nodeCenters.data(), nodeSizes.data(),
                       zeroExpansions.data(), octree.numNodes, tree.data(), tC.data(), tS.data(), box, 0, a,
                       contactFlags.data());
    std::vector<uint8_t> contactReference(octree.numNodes, 0);
    for (TreeNodeIndex t = 0; t < a; ++t)
    {
        for (TreeNodeIndex n = 0; n < octree.numNodes; ++n)
        {
            auto [k1, k2] = decodePlaceholderBit2K(od.prefixes[n]);
            if (containedIn(k1, k2, tree[0], tree[a])) { continue; }
            if (nodeSizes[n] == Vec3<T>{0, 0, 0}) { continue; }
            if (overlap(nodeCenters[n], nodeSizes[n], tC[t], tS[t], box)) { contactReference[n] = 1; }
        }
    }
    EXPECT_EQ(contactFlags, contactReference);
}

TEST(HaloDiscovery, findHalosSymmetric)
{
    findHalosSymmetricFlags<unsigned>();
    findHalosSymmetricFlags<uint64_t>();
}

/*! @brief symmetric halo discovery driven from particles: leaf reach = 2 * ext * max(h) per leaf
 *
 * Replicates the leaf-reach construction of FocusedOctree::discoverHalos (symmetric branch) from
 * per-particle smoothing lengths and a particle layout, then checks the guarantee production relies
 * on at particle level: every remote particle j that reaches a local particle i within 2 * ext * h_j
 * lives in a flagged leaf, including pairs beyond i's own reach, which the gather-sense discovery
 * (local boxes inflated by the local reach only) misses.
 */
template<class KeyType>
static void findHalosSymmetricFromParticles(float ext)
{
    using T  = double;
    Box<T> box(0, 1);

    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);
    OctreeData<KeyType, execution::Cpu> octree;
    octree.resize(nNodes(tree));
    updateInternalTree<KeyType>(tree, octree.data());

    std::vector<Vec3<T>> nodeCenters(octree.numNodes), nodeSizes(octree.numNodes);
    nodeFpCenters<KeyType>(octree.prefixes, nodeCenters.data(), nodeSizes.data(), box);
    auto leaf2int = leafToInternal(octree);
    auto od       = octree.data();

    const TreeNodeIndex numLeaves = octree.numLeafNodes;
    const TreeNodeIndex a         = 32; // local leaves [0:a), remote leaves [a:numLeaves)

    // particles per leaf at consecutive indices, i.e. the SFC-sorted layout of a real domain
    constexpr LocalIndex perLeaf = 8;
    std::vector<LocalIndex> layout(numLeaves + 1);
    for (TreeNodeIndex l = 0; l <= numLeaves; ++l)
    {
        layout[l] = l * perLeaf;
    }
    const LocalIndex numParticles = layout[numLeaves];

    std::mt19937 gen(7);
    std::uniform_real_distribution<T> offset(-1.0, 1.0);
    std::vector<T> x(numParticles), y(numParticles), z(numParticles), h(numParticles, 0.01);
    std::vector<TreeNodeIndex> leafOf(numParticles);
    for (TreeNodeIndex l = 0; l < numLeaves; ++l)
    {
        const Vec3<T> c = nodeCenters[leaf2int[l]], s = nodeSizes[leaf2int[l]];
        for (LocalIndex p = layout[l]; p < layout[l + 1]; ++p)
        {
            x[p]      = c[0] + offset(gen) * s[0];
            y[p]      = c[1] + offset(gen) * s[1];
            z[p]      = c[2] + offset(gen) * s[2];
            leafOf[p] = l;
        }
    }
    // remote leaf in the last corner: large reach 2 * ext * 0.3 (inflated 0.25-cell covers the unit box)
    for (LocalIndex p = layout[numLeaves - 1], k = 0; p < layout[numLeaves]; ++p, ++k)
    {
        h[p] = 0.3 * T(k + 1) / T(perLeaf);
    }
    // another remote leaf in a different level-1 subtree with a smaller reach (non-trivial max-upsweep)
    for (LocalIndex p = layout[40]; p < layout[41]; ++p)
    {
        h[p] = 0.05;
    }
    // a local leaf with a large reach: must never flag anything by itself
    for (LocalIndex p = layout[3]; p < layout[4]; ++p)
    {
        h[p] = 0.2;
    }

    // leaf reach exactly as in FocusedOctree::discoverHalos, for all leaves (as if already peer-exchanged)
    std::vector<T> expansions(octree.numNodes, 0);
    for (TreeNodeIndex l = 0; l < numLeaves; ++l)
    {
        T hMax = 0;
        for (LocalIndex p = layout[l]; p < layout[l + 1]; ++p)
        {
            hMax = std::max(hMax, h[p]);
        }
        expansions[leaf2int[l]] = T(2 * ext) * hMax;
    }
    std::span<const TreeNodeIndex> levelRange{od.levelRange, size_t(maxTreeLevel<KeyType>{}) + 2};
    upsweep(levelRange, od.childOffsets, expansions.data(), MaxCombination<T>{});

    EXPECT_EQ(expansions[leaf2int[numLeaves - 1]], T(2 * ext) * 0.3);
    EXPECT_EQ(expansions[0], T(2 * ext) * 0.3); // root holds the global maximum

    // plain local leaf boxes as search targets
    std::vector<Vec3<T>> tC(numLeaves), tS(numLeaves);
    for (TreeNodeIndex l = 0; l < numLeaves; ++l)
    {
        tC[l] = nodeCenters[leaf2int[l]];
        tS[l] = nodeSizes[leaf2int[l]];
    }

    std::vector<uint8_t> flags(octree.numNodes, 0);
    findHalosSymmetric(od.prefixes, od.childOffsets, od.parents, nodeCenters.data(), nodeSizes.data(),
                       expansions.data(), octree.numNodes, tree.data(), tC.data(), tS.data(), box, 0, a, flags.data());

    // node-level: all-to-all reference with nodes inflated by their own reach
    std::vector<uint8_t> reference(octree.numNodes, 0);
    for (TreeNodeIndex t = 0; t < a; ++t)
    {
        for (TreeNodeIndex n = 0; n < octree.numNodes; ++n)
        {
            auto [k1, k2] = decodePlaceholderBit2K(od.prefixes[n]);
            if (containedIn(k1, k2, tree[0], tree[a])) { continue; }
            if (nodeSizes[n] == Vec3<T>{0, 0, 0}) { continue; }
            Vec3<T> inflated = nodeSizes[n] + Vec3<T>{expansions[n], expansions[n], expansions[n]};
            if (overlap(nodeCenters[n], inflated, tC[t], tS[t], box)) { reference[n] = 1; }
        }
    }
    EXPECT_EQ(flags, reference);
    EXPECT_EQ(1, flags[leaf2int[numLeaves - 1]]);
    for (TreeNodeIndex n = 0; n < octree.numNodes; ++n)
    {
        auto [k1, k2] = decodePlaceholderBit2K(od.prefixes[n]);
        if (containedIn(k1, k2, tree[0], tree[a])) { EXPECT_EQ(0, flags[n]); }
    }

    // gather-sense discovery: local boxes inflated by the LOCAL reach only (what the one-way search does)
    std::vector<Vec3<T>> tSLocal(numLeaves);
    for (TreeNodeIndex l = 0; l < numLeaves; ++l)
    {
        const T e = expansions[leaf2int[l]];
        tSLocal[l] = tS[l] + Vec3<T>{e, e, e};
    }
    std::vector<uint8_t> plainFlags(octree.numNodes, 0);
    findHalos(od.prefixes, od.childOffsets, od.parents, nodeCenters.data(), nodeSizes.data(), tree.data(), tC.data(),
              tSLocal.data(), box, 0, a, plainFlags.data());

    // particle-level: every remote j reaching a local i within 2 * ext * h_j lives in a flagged leaf
    size_t numReaching = 0, numJOnly = 0, numJOnlyMissedByPlain = 0;
    for (LocalIndex i = 0; i < layout[a]; ++i)
    {
        for (LocalIndex j = layout[a]; j < numParticles; ++j)
        {
            const T r2 = distanceSq<false>(x[i], y[i], z[i], x[j], y[j], z[j], box);
            const T ri = T(2 * ext) * h[i], rj = T(2 * ext) * h[j];
            if (r2 >= rj * rj) { continue; }

            ++numReaching;
            EXPECT_EQ(1, flags[leaf2int[leafOf[j]]]) << "remote particle " << j << " reaches local " << i;
            if (r2 >= ri * ri)
            {
                ++numJOnly;
                numJOnlyMissedByPlain += plainFlags[leaf2int[leafOf[j]]] == 0;
            }
        }
    }
    EXPECT_GT(numReaching, 0u);
    EXPECT_GT(numJOnly, 0u) << "sample has no pairs beyond the local reach";
    EXPECT_GT(numJOnlyMissedByPlain, 0u) << "gather-sense discovery already finds every reaching remote particle";
}

TEST(HaloDiscovery, findHalosSymmetricFromParticles)
{
    findHalosSymmetricFromParticles<unsigned>(1.0f);
    findHalosSymmetricFromParticles<uint64_t>(1.0f);
    findHalosSymmetricFromParticles<unsigned>(1.05f);
    findHalosSymmetricFromParticles<uint64_t>(1.05f);
}

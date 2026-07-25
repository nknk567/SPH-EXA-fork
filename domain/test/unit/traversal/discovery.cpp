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

#include "gtest/gtest.h"

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

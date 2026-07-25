/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  GPU driver for halo discovery using traversal of an octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/primitives/math.hpp"
#include "cstone/traversal/collisions_gpu.h"
#include "cstone/traversal/macs.hpp"

namespace cstone
{

template<class KeyType, class T>
__global__ void findHalosKernel(const KeyType* nodePrefixes,
                                const TreeNodeIndex* childOffsets,
                                const TreeNodeIndex* parents,
                                const Vec3<T>* nodeCenters,
                                const Vec3<T>* nodeSizes,
                                const KeyType* leaves,
                                const Vec3<T>* searchCenters,
                                const Vec3<T>* searchSizes,
                                __grid_constant__ const Box<T> box,
                                TreeNodeIndex firstNode,
                                TreeNodeIndex lastNode,
                                uint8_t* collisionFlags)
{
    TreeNodeIndex leafIdx = blockIdx.x * blockDim.x + threadIdx.x + firstNode;

    if (leafIdx < lastNode)
    {
        Vec3<T> tC         = searchCenters[leafIdx];
        Vec3<T> tS         = searchSizes[leafIdx];
        KeyType lowestKey  = leaves[firstNode];
        KeyType highestKey = leaves[lastNode];

        // A zero search size means this leaf does not have a valid MixD SFC key so it doesn't include any particles
        if (tS == Vec3<T>{0, 0, 0}) { return; }

        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (containedIn(lowestKey, highestKey, tC, tS, box)) { return; }

        // mark all colliding node indices outside [lowestKey:highestKey]
        findCollisions(nodePrefixes, childOffsets, parents, nodeCenters, nodeSizes, tC, tS, box, lowestKey, highestKey,
                       collisionFlags);
    }
}

template<class KeyType, class T>
void findHalosGpu(execution::Gpu exec,
                  const KeyType* prefixes,
                  const TreeNodeIndex* childOffsets,
                  const TreeNodeIndex* parents,
                  const Vec3<T>* nodeCenters,
                  const Vec3<T>* nodeSizes,
                  const KeyType* leaves,
                  const Vec3<T>* searchCenters,
                  const Vec3<T>* searchSizes,
                  const Box<T>& box,
                  TreeNodeIndex firstNode,
                  TreeNodeIndex lastNode,
                  uint8_t* collisionFlags)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(lastNode - firstNode, numThreads);

    if (numBlocks == 0) { return; }
    findHalosKernel<<<numBlocks, numThreads, 0, exec>>>(prefixes, childOffsets, parents, nodeCenters, nodeSizes, leaves,
                                                        searchCenters, searchSizes, box, firstNode, lastNode,
                                                        collisionFlags);
}

#define FIND_HALOS_GPU(KeyType, T)                                                                                     \
    template void findHalosGpu(execution::Gpu, const KeyType* prefixes, const TreeNodeIndex* childOffsets,             \
                               const TreeNodeIndex* parents, const Vec3<T>* nodeCenters, const Vec3<T>* nodeSizes,     \
                               const KeyType* leaves, const Vec3<T>* searchCenters, const Vec3<T>* searchSizes,        \
                               const Box<T>& box, TreeNodeIndex firstNode, TreeNodeIndex lastNode,                     \
                               uint8_t* collisionFlags)

FIND_HALOS_GPU(uint32_t, float);
FIND_HALOS_GPU(uint64_t, float);
FIND_HALOS_GPU(uint64_t, double);

template<class Th>
__global__ void leafExpansionsKernel(
    const Th* h, const LocalIndex* layout, TreeNodeIndex firstNode, TreeNodeIndex lastNode, Th scale, Th* expansions)
{
    TreeNodeIndex leafIdx = blockIdx.x * blockDim.x + threadIdx.x + firstNode;
    if (leafIdx >= lastNode) { return; }

    Th hMax = 0;
    for (LocalIndex i = layout[leafIdx]; i < layout[leafIdx + 1]; ++i)
    {
        hMax = max(hMax, h[i]);
    }
    expansions[leafIdx] = scale * hMax;
}

template<class Th>
void leafExpansionsGpu(execution::Gpu exec,
                       const Th* h,
                       const LocalIndex* layout,
                       TreeNodeIndex firstNode,
                       TreeNodeIndex lastNode,
                       Th scale,
                       Th* expansions)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(lastNode - firstNode, numThreads);
    if (numBlocks == 0) { return; }
    leafExpansionsKernel<<<numBlocks, numThreads, 0, exec>>>(h, layout, firstNode, lastNode, scale, expansions);
}

template void leafExpansionsGpu(execution::Gpu, const float*, const LocalIndex*, TreeNodeIndex, TreeNodeIndex, float,
                                float*);
template void leafExpansionsGpu(execution::Gpu, const double*, const LocalIndex*, TreeNodeIndex, TreeNodeIndex, double,
                                double*);

template<class Tc, class Th>
__global__ void
inflateNodeSizesKernel(const Vec3<Tc>* sizes, const Th* expansions, TreeNodeIndex numNodes, Vec3<Tc>* inflated)
{
    TreeNodeIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) { return; }
    const Tc e  = expansions[i];
    inflated[i] = sizes[i] + Vec3<Tc>{e, e, e};
}

template<class Tc, class Th>
void inflateNodeSizesGpu(
    execution::Gpu exec, const Vec3<Tc>* sizes, const Th* expansions, TreeNodeIndex numNodes, Vec3<Tc>* inflated)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(numNodes, numThreads);
    if (numBlocks == 0) { return; }
    inflateNodeSizesKernel<<<numBlocks, numThreads, 0, exec>>>(sizes, expansions, numNodes, inflated);
}

template void inflateNodeSizesGpu(execution::Gpu, const Vec3<float>*, const float*, TreeNodeIndex, Vec3<float>*);
template void inflateNodeSizesGpu(execution::Gpu, const Vec3<double>*, const double*, TreeNodeIndex, Vec3<double>*);
template void inflateNodeSizesGpu(execution::Gpu, const Vec3<double>*, const float*, TreeNodeIndex, Vec3<double>*);

template<class KeyType, class T>
__global__ void findHalosSymmetricKernel(const KeyType* nodePrefixes,
                                         const TreeNodeIndex* childOffsets,
                                         const TreeNodeIndex* parents,
                                         const Vec3<T>* nodeCenters,
                                         const Vec3<T>* inflatedNodeSizes,
                                         const KeyType* leaves,
                                         const Vec3<T>* searchCenters,
                                         const Vec3<T>* searchSizes,
                                         __grid_constant__ const Box<T> box,
                                         TreeNodeIndex firstNode,
                                         TreeNodeIndex lastNode,
                                         uint8_t* collisionFlags)
{
    TreeNodeIndex leafIdx = blockIdx.x * blockDim.x + threadIdx.x + firstNode;

    if (leafIdx < lastNode)
    {
        Vec3<T> tC         = searchCenters[leafIdx];
        Vec3<T> tS         = searchSizes[leafIdx];
        KeyType lowestKey  = leaves[firstNode];
        KeyType highestKey = leaves[lastNode];

        if (tS == Vec3<T>{0, 0, 0}) { return; }

        /* No containedIn early exit here: the local box being inside the assigned SFC range
         * does not preclude collisions with remote nodes inflated by their own reach. */
        findCollisions(nodePrefixes, childOffsets, parents, nodeCenters, inflatedNodeSizes, tC, tS, box, lowestKey,
                       highestKey, collisionFlags);
    }
}

template<class KeyType, class T>
void findHalosSymmetricGpu(execution::Gpu exec,
                           const KeyType* prefixes,
                           const TreeNodeIndex* childOffsets,
                           const TreeNodeIndex* parents,
                           const Vec3<T>* nodeCenters,
                           const Vec3<T>* inflatedNodeSizes,
                           const KeyType* leaves,
                           const Vec3<T>* searchCenters,
                           const Vec3<T>* searchSizes,
                           const Box<T>& box,
                           TreeNodeIndex firstNode,
                           TreeNodeIndex lastNode,
                           uint8_t* collisionFlags)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(lastNode - firstNode, numThreads);

    if (numBlocks == 0) { return; }
    findHalosSymmetricKernel<<<numBlocks, numThreads, 0, exec>>>(prefixes, childOffsets, parents, nodeCenters,
                                                                 inflatedNodeSizes, leaves, searchCenters, searchSizes,
                                                                 box, firstNode, lastNode, collisionFlags);
}

#define FIND_HALOS_SYMMETRIC_GPU(KeyType, T)                                                                           \
    template void findHalosSymmetricGpu(execution::Gpu, const KeyType* prefixes, const TreeNodeIndex* childOffsets,    \
                                        const TreeNodeIndex* parents, const Vec3<T>* nodeCenters,                      \
                                        const Vec3<T>* inflatedNodeSizes, const KeyType* leaves,                       \
                                        const Vec3<T>* searchCenters, const Vec3<T>* searchSizes, const Box<T>& box,   \
                                        TreeNodeIndex firstNode, TreeNodeIndex lastNode, uint8_t* collisionFlags)

FIND_HALOS_SYMMETRIC_GPU(uint32_t, float);
FIND_HALOS_SYMMETRIC_GPU(uint64_t, float);
FIND_HALOS_SYMMETRIC_GPU(uint64_t, double);

template<class T, class KeyType>
__global__ void markMacsGpuKernel(const KeyType* prefixes,
                                  const TreeNodeIndex* childOffsets,
                                  const TreeNodeIndex* parents,
                                  const Vec4<T>* centers,
                                  __grid_constant__ const Box<T> box,
                                  const KeyType* focusNodes,
                                  TreeNodeIndex numFocusNodes,
                                  bool limitSource,
                                  uint8_t* markings,
                                  const AxesBits axesBits)
{
    TreeNodeIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= numFocusNodes) { return; }

    KeyType focusStart = focusNodes[0];
    KeyType focusEnd   = focusNodes[numFocusNodes];

    IBox target = sfcIBox(sfcKey(focusNodes[tid]), sfcKey(focusNodes[tid + 1]), axesBits);
    if (target == IBox{}) { return; }
    IBox targetExt = IBox(target.xmin() - 1, target.xmax() + 1, target.ymin() - 1, target.ymax() + 1, target.zmin() - 1,
                          target.zmax() + 1);
    if (containedIn(focusStart, focusEnd, targetExt, axesBits)) { return; }

    auto [targetCenter, targetSize] = centerAndSize<KeyType>(target, box);
    unsigned maxLevel               = maxTreeLevel<KeyType>{};
    if (limitSource) { maxLevel = stl::max(int(treeLevel(focusNodes[tid + 1] - focusNodes[tid])) - 1, 0); }
    markMacPerBox(targetCenter, targetSize, maxLevel, prefixes, childOffsets, parents, centers, box, focusStart,
                  focusEnd, markings);
}

template<class T, class KeyType>
void markMacsGpu(execution::Gpu exec,
                 const KeyType* prefixes,
                 const TreeNodeIndex* childOffsets,
                 const TreeNodeIndex* parents,
                 const Vec4<T>* centers,
                 const Box<T>& box,
                 const KeyType* focusNodes,
                 TreeNodeIndex numFocusNodes,
                 bool limitSource,
                 uint8_t* markings)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(numFocusNodes, numThreads);

    const auto axesBits = box.getBoxDimBits(maxTreeLevel<KeyType>{});
    if (numFocusNodes)
    {
        markMacsGpuKernel<<<numBlocks, numThreads, 0, exec>>>(prefixes, childOffsets, parents, centers, box, focusNodes,
                                                              numFocusNodes, limitSource, markings, axesBits);
    }
}

#define MARK_MACS_GPU(KeyType, T)                                                                                      \
    template void markMacsGpu(execution::Gpu, const KeyType* prefixes, const TreeNodeIndex* childOffsets,              \
                              const TreeNodeIndex* parents, const Vec4<T>* centers, const Box<T>& box,                 \
                              const KeyType* focusNodes, TreeNodeIndex numFocusNodes, bool limitSource,                \
                              uint8_t* markings)

MARK_MACS_GPU(uint64_t, double);
MARK_MACS_GPU(uint64_t, float);
MARK_MACS_GPU(unsigned, float);

} // namespace cstone

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

#pragma once

#include "cstone/execution.hpp"
#include "cstone/traversal/collisions.hpp"

namespace cstone
{

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam T                     float or double
 * @param[in]  prefixes          Warren-Salmon node keys of the octree, length = numTreeNodes
 * @param[in]  childOffsets      child offsets array, length = numTreeNodes
 * @param[in]  parents           parent of each node i, stored at index (i-1)/8
 * @param[in]  nodeCenters       geometric center of each octree node
 * @param[in]  nodeSizes         geometric size of each octree node
 * @param[in]  leaves            cstone array of leaf node keys
 * @param[in]  searchCenters     effective halo search box center per octree (leaf) node, accessed [firstNode:lastNode]
 * @param[in]  searchSizes       effective halo search box size per octree (leaf) node, accessed [firstNode:lastNode]
 * @param[in]  box               coordinate bounding box
 * @param[in]  firstNode         first cstone leaf node index to consider as local
 * @param[in]  lastNode          last cstone leaf node index to consider as local
 * @param[out] collisionFlags    array of length numLeafNodes, each node that is a halo
 *                               from the perspective of [firstNode:lastNode] will be marked
 *                               with a non-zero value.
 *                               Note: does NOT reset non-colliding indices to 0, so @p collisionFlags
 *                               should be zero-initialized prior to calling this function.
 * @param[in]  exec              execution policy
 */
template<class KeyType, class T>
extern void findHalosGpu(execution::Gpu exec,
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
                         uint8_t* collisionFlags);

/*! @brief per-leaf interaction expansion: max of scale * h over the particles of each leaf
 *
 * @param[in]  h           particle smoothing lengths
 * @param[in]  layout      particle index range per leaf, indexed [firstNode:lastNode+1]
 * @param[in]  firstNode   first leaf node index
 * @param[in]  lastNode    last leaf node index
 * @param[in]  scale       scale factor, e.g. 2 * searchExtFactor for kernel support spheres
 * @param[out] expansions  per-leaf expansion, accessed [firstNode:lastNode]
 */
template<class Th>
extern void leafExpansionsGpu(execution::Gpu exec,
                              const Th* h,
                              const LocalIndex* layout,
                              TreeNodeIndex firstNode,
                              TreeNodeIndex lastNode,
                              Th scale,
                              Th* expansions);

//! @brief inflated[i] = sizes[i] + expansions[i] in each component, i in [0:numNodes]
template<class Tc, class Th>
extern void inflateNodeSizesGpu(
    execution::Gpu exec, const Vec3<Tc>* sizes, const Th* expansions, TreeNodeIndex numNodes, Vec3<Tc>* inflated);

/*! @brief transposed halo discovery: mark nodes whose own interaction reach extends into the local boxes
 *
 * Same as findHalosGpu, but with the roles of the search expansion swapped: @p nodeSizes are
 * expected to be inflated by each node's own interaction expansion, while the local search
 * boxes [firstNode:lastNode] are the plain leaf boxes. There is no early exit for boxes
 * contained in the local key range, since remote nodes reach arbitrarily far inside.
 */
template<class KeyType, class T>
extern void findHalosSymmetricGpu(execution::Gpu exec,
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
                                  uint8_t* collisionFlags);

template<class T, class KeyType>
extern void markMacsGpu(execution::Gpu exec,
                        const KeyType* prefixes,
                        const TreeNodeIndex* childOffsets,
                        const TreeNodeIndex* parents,
                        const Vec4<T>* centers,
                        const Box<T>& box,
                        const KeyType* focusNodes,
                        TreeNodeIndex numFocusNodes,
                        bool limitSource,
                        uint8_t* markings);

} // namespace cstone

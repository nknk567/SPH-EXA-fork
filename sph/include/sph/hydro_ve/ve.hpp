/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Volume element definition i-loop driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include <algorithm>
#include <vector>

#include "sph/sph_gpu.hpp"
#include "ve_kern.hpp"

namespace sph
{

template<typename Tc, class Dataset>
void computeVe(const GroupView& grp, Dataset& d, const cstone::Box<Tc>& box)
{
    if constexpr (d.useGpu) { gpu::computeVe(grp, d, box); }
    else { veIjLoop(d.neighborhood, d.K, d.xm.data(), d.wh.data(), d.kx.data()); }
}

/*! @brief compute the SPH-smoothed converged particle volume, the VE weights of the next step
 *
 * As in SPHYNX (calculate_IAD.f90/update.f90), the weights of the next step are the smoothed
 * volume estimate of the converged state, making them independent of the positions at force
 * evaluation time. @p volstd must have room for absolute particle indices up to grp.lastBody.
 */
template<typename Tc, class Dataset, class Tv>
void computeVolstd(const GroupView& grp, Dataset& d, const cstone::Box<Tc>& box, Tv* volstd)
{
    if constexpr (d.useGpu) { gpu::computeVolstd(grp, d, box, volstd); }
    else { volstdIjLoop(d.neighborhood, d.K, d.xm.data(), d.kx.data(), d.wh.data(), volstd); }
}

//! @brief assign the volume elements of the next step for locally owned particles
template<class Dataset, class Tv>
void setVolumeElements(const GroupView& grp, Dataset& d, const Tv* volstd)
{
    if constexpr (d.useGpu) { gpu::setVolumeElements(grp, d, volstd); }
    else
    {
        auto* xm = d.xm.data();
#pragma omp parallel for schedule(static)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            //! clamp the per-step change of the carried volume, see volstdGrow/ShrinkFactor
            Tv lo = xm[i] / Tv(volstdShrinkFactor);
            Tv hi = xm[i] * Tv(volstdGrowFactor);
            xm[i] = stl::min(stl::max(volstd[i], lo), hi);
        }
    }
}

/*! @brief one Newton-Raphson iteration for the smoothing length constraint rho * h^3 = eta * m
 *
 * Iterates over the fixed neighbor list with fixed volume elements xm and updates h of locally
 * owned particles in place. Uses the ay field as scratch space for the updated smoothing length
 * and the az field for the relative h change per particle (consumed by computeVeNRTail).
 * The constraint target eta = ballmassEta(ng0) depends only on the desired neighbor count.
 * @p h0 holds the smoothing lengths at the start of the step's NR iterations (filled here when
 * @p firstIteration is set); the cumulative upward h movement is capped at hNRExtFactor * h0 so
 * that the neighbor lists built before the iterations remain complete for the final h.
 *
 * @return the number of locally owned particles with a relative h change >= hNRTol; zero means
 *         the iterations are converged
 */
template<typename Tc, class Dataset, class Tv>
size_t computeVeNR(const GroupView& grp, Dataset& d, const cstone::Box<Tc>& box, Tv* h0, bool firstIteration)
{
    if constexpr (d.useGpu) { return gpu::computeVeNR(grp, d, box, h0, firstIteration); }
    else
    {
        if (firstIteration) { std::copy(d.h.data(), d.h.data() + d.x.size(), h0); }
        veNRIjLoop(d.neighborhood, d.K, d.ng0, d.xm.data(), d.m.data(), h0, d.wh.data(), d.whd.data(), d.kx.data(),
                   d.ay.data());

        using Th               = std::decay_t<decltype(d.h[0])>;
        const Th* hNew         = d.ay.data();
        Th*       h            = d.h.data();
        Th*       relDh        = d.az.data();
        size_t    numUnconverged = 0;
#pragma omp parallel for schedule(static) reduction(+ : numUnconverged)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            Th rel   = std::abs(hNew[i] - h[i]) / h[i];
            relDh[i] = rel;
            h[i]     = hNew[i];
            numUnconverged += rel >= Th(hNRTol);
        }
        return numUnconverged;
    }
}

/*! @brief finish the NR smoothing-length iterations for the unconverged residual only
 *
 * Extracts the particles whose relative h change of the last computeVeNR pass (az scratch) was
 * still >= hNRTol and iterates only those, each by direct octree traversal (veNRTraversalUpdate)
 * instead of full neighbor-list passes over all particles. Exact because the NR update of a
 * particle depends only on its own h and the fixed volume elements; converged particles keep
 * their last h instead of accumulating further sub-tolerance refinements.
 *
 * @param maxPasses           remaining iteration budget (hNRIterMax minus the passes already done)
 * @param unconvergedPerPass  appends the local unconverged count after each pass (diagnostic)
 * @return the number of passes performed
 */
template<typename Tc, class Dataset, class Tv>
unsigned computeVeNRTail(const GroupView& grp, Dataset& d, const cstone::Box<Tc>& box, const Tv* h0,
                         unsigned maxPasses, std::vector<size_t>& unconvergedPerPass)
{
    if constexpr (d.useGpu) { return gpu::computeVeNRTail(grp, d, box, h0, maxPasses, unconvergedPerPass); }
    else
    {
        using Th = std::decay_t<decltype(d.h[0])>;

        std::vector<cstone::LocalIndex> subset;
        const Th*                       relDh = d.az.data();
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            if (relDh[i] >= Th(hNRTol)) { subset.push_back(i); }
        }

        Th*      h      = d.h.data();
        unsigned passes = 0;
        while (passes < maxPasses)
        {
            ++passes;
            size_t numUnconverged = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : numUnconverged)
            for (size_t s = 0; s < subset.size(); ++s)
            {
                cstone::LocalIndex i    = subset[s];
                Th                 hNew = veNRTraversalUpdate(i, d.K, ballmassEta<Th>(d.ng0), d.treeView, box,
                                                              d.x.data(), d.y.data(), d.z.data(), h, d.xm.data(),
                                                              d.m.data(), h0, d.wh.data(), d.whd.data());
                Th                 rel  = std::abs(hNew - h[i]) / h[i];
                h[i]                    = hNew;
                numUnconverged += rel >= Th(hNRTol);
            }
            unconvergedPerPass.push_back(numUnconverged);
            if (numUnconverged == 0) { break; }
        }
        return passes;
    }
}

} // namespace sph

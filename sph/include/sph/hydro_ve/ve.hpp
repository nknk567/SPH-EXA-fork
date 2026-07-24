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
    else { std::copy(volstd + grp.firstBody, volstd + grp.lastBody, d.xm.data() + grp.firstBody); }
}

/*! @brief one Newton-Raphson iteration for the smoothing length constraint rho * h^3 = eta * m
 *
 * Iterates over the fixed neighbor list with fixed volume elements xm and updates h of locally
 * owned particles in place. Uses the ay field as scratch space for the updated smoothing length.
 * The constraint target eta = ballmassEta(ng0) depends only on the desired neighbor count.
 * @p h0 holds the smoothing lengths at the start of the step's NR iterations (filled here when
 * @p firstIteration is set); the cumulative upward h movement is capped at hNRExtFactor * h0 so
 * that the neighbor lists built before the iterations remain complete for the final h.
 *
 * @return the largest relative h change among locally owned particles, the convergence measure
 *         of the iteration
 */
template<typename Tc, class Dataset, class Tv>
auto computeVeNR(const GroupView& grp, Dataset& d, const cstone::Box<Tc>& box, Tv* h0, bool firstIteration)
{
    if constexpr (d.useGpu) { return gpu::computeVeNR(grp, d, box, h0, firstIteration); }
    else
    {
        if (firstIteration) { std::copy(d.h.data(), d.h.data() + d.x.size(), h0); }
        veNRIjLoop(d.neighborhood, d.K, d.ng0, d.xm.data(), d.m.data(), h0, d.wh.data(), d.whd.data(), d.kx.data(),
                   d.ay.data());

        using Th        = std::decay_t<decltype(d.h[0])>;
        const Th* hNew  = d.ay.data();
        Th*       h     = d.h.data();
        Th        maxDh = 0;
#pragma omp parallel for schedule(static) reduction(max : maxDh)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            maxDh = std::max(maxDh, std::abs(hNew[i] - h[i]) / h[i]);
            h[i]  = hNew[i];
        }
        return maxDh;
    }
}

} // namespace sph

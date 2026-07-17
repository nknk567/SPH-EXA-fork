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

/*! @brief initialize the per-particle Newton-Raphson target ballmass = rho * h^3
 *
 * Called with the current density estimate rho = kx * m / xm, this anchors the constraint such
 * that it is satisfiable at the current h for every particle, including free surfaces.
 */
template<class Dataset>
void ballmassFromDensity(const GroupView& grp, Dataset& d)
{
    if constexpr (d.useGpu) { gpu::ballmassFromDensity(grp, d); }
    else
    {
        const auto* kx       = d.kx.data();
        const auto* xm       = d.xm.data();
        const auto* m        = d.m.data();
        const auto* h        = d.h.data();
        auto*       ballmass = d.ballmass.data();
#pragma omp parallel for schedule(static)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            ballmass[i] = kx[i] * m[i] / xm[i] * h[i] * h[i] * h[i];
        }
    }
}

/*! @brief one Newton-Raphson iteration for the smoothing length constraint rho * h^3 = ballmass
 *
 * Iterates over the fixed neighbor list with fixed volume elements xm and updates h of locally
 * owned particles in place. Uses the ay field as scratch space for the updated smoothing length.
 */
template<typename Tc, class Dataset>
void computeVeNR(const GroupView& grp, Dataset& d, const cstone::Box<Tc>& box)
{
    if constexpr (d.useGpu) { gpu::computeVeNR(grp, d, box); }
    else
    {
        veNRIjLoop(d.neighborhood, d.K, d.xm.data(), d.m.data(), d.ballmass.data(), d.wh.data(), d.whd.data(),
                   d.kx.data(), d.ay.data());
        std::copy(d.ay.data() + grp.firstBody, d.ay.data() + grp.lastBody, d.h.data() + grp.firstBody);
    }
}

} // namespace sph

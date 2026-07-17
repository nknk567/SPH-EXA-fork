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
    else
    {
        veIjLoop(d.neighborhood, d.K, d.xm.data(), d.wh.data(), d.kx.data());
    }
}

/*! @brief update volume elements from the converged density for use in the next time-step
 *
 * As in SPHYNX (update.f90), the VE weights of the next step are m / rho with the converged
 * density rho = kx * m / xm of the current step, i.e. xm <- xm / kx. This makes the weights
 * independent of the positions at force evaluation time.
 */
template<class Dataset>
void convergedVolumeElements(const GroupView& grp, Dataset& d)
{
    if constexpr (d.useGpu) { gpu::convergedVolumeElements(grp, d); }
    else
    {
        const auto* kx = d.kx.data();
        auto*       xm = d.xm.data();
#pragma omp parallel for schedule(static)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            xm[i] /= kx[i];
        }
    }
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

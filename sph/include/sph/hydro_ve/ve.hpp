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

#include "sph/sph_gpu.hpp"
#include "ve_kern.hpp"

#include "sph/sph.hpp"

namespace sph
{

//! @brief per-pass statistics of one smoothing-length NR iteration (see computeVeNR)
struct NRStatistics
{
    //! @brief number of particles whose relative h change was >= the tolerance
    size_t n_unconverged;

    //! @brief number of particles with ballmass reset
    size_t n_ballmass_reset;
};

template<typename Tc, class Dataset>
void computeVe(Dataset& d, const cstone::Box<Tc>& box)
{
    if constexpr (d.useGpu) { gpu::computeVe(d, box); }
    else { veIjLoop(d.neighborhood, d.K, d.xm.data(), d.wh, d.kx.data()); }
}

template<class Dataset>
void initializeBallmass(const GroupView& grp, Dataset& d)
{
    if constexpr (d.useGpu) { gpu::initializeBallmass(grp, d); }
    else
    {
        using Th       = std::decay_t<decltype(d.ballmass[0])>;
        auto*       bm = d.ballmass.data();
        const auto* m  = d.m.data();
#pragma omp parallel for schedule(static)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            bm[i] = ballmassEta<Th>(d.ng0) * Th(m[i]);
        }
    }
}

template<typename Dataset>
NRStatistics computeNRStat(const cstone::GroupView& grp, Dataset& d, const typename Dataset::HydroType* h_new,
                           const float hNRTol)
{
    if constexpr (d.useGpu) { return gpu::computeNRStat(grp, d, h_new, hNRTol); }
    else
    {
        using Th                = std::decay_t<decltype(d.h[0])>;
        const Th* bm            = d.ballmass.data();
        Th*       h             = d.h.data();
        size_t    n_unconverged = 0, n_reset = 0;
#pragma omp parallel for schedule(static) reduction(+ : n_unconverged, n_reset)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            Th rel = std::abs(h_new[i] - h[i]) / h[i];
            n_unconverged += rel >= Th(hNRTol);
            n_reset += bm[i] == Th(0);
        }
        return {n_unconverged, n_reset};
    }
}

template<class Dataset, class Th>
void computeVeNR(Dataset& d, float hExtFactor, float hWallFactor, float hNRTol, float ngmin, float ngmax, const Th* h0,
                 Th* h_buf, Th* nc_buf)
{
    if constexpr (d.useGpu)
    {
        return gpu::computeVeNR(d, hExtFactor, hWallFactor, hNRTol, ngmin, ngmax, h0, h_buf, nc_buf);
    }
    else
    {
        veNRIjLoop(d.neighborhood, d.K, d.xm.data(), d.m.data(), d.ballmass.data(), d.wh, d.kx.data(), hExtFactor,
                   hWallFactor, hNRTol, ngmin, ngmax, h0, h_buf, nc_buf);
    }
}

} // namespace sph

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
 * @brief Volume definition and gradient of h architecture portable kernel
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/sph_kernels.hpp"

namespace sph
{

template<class T, class Kernel>
struct VeInteraction
{
    Kernel wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& /* r_ij */,
                              T r2) const
    {
        const auto [i, iPos, hi, xmassi] = iData;
        const auto [j, jPos, hj, xmassj] = jData;

        auto hInv = T(1) / hi;

        T dist = std::sqrt(r2);
        T vloc = dist * hInv;
        T w    = wh(vloc);

        T kxi = w * xmassj;

        return std::make_tuple(kxi);
    }
};

template<class T, class Tc>
struct VePostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, xmassi] = iData;
        auto [kxi]                       = result;

        auto hInv  = T(1) / hi;
        auto h3Inv = hInv * hInv * hInv;

        kxi *= K * h3Inv;

        return std::make_tuple(kxi);
    }
};

template<class Neighbordhood, class Tc, class T>
void veIjLoop(const Neighbordhood& neighborhood, Tc K, const T* xm, KernelVariant<T> const& wh, T* kx)
{
    std::visit(
        [&]<class Kernel>(Kernel wh)
        {
            neighborhood.ijLoop(cstone::ijloop::makeIjLoopData<Tc, T*>(
                std::make_tuple(xm), std::make_tuple(kx), VeInteraction<T, Kernel>{wh}, VePostamble<T, Tc>{K}));
        },
        wh);
}

template<class T>
constexpr T ballmassEta(unsigned ng0)
{
    return T(3) * T(ng0) / (T(32) * M_PI);
}

/*! @brief kernel sums for the Newton-Raphson iteration of the smoothing length
 *
 * Computes the sums needed to solve rho_i(h_i) * h_i^3 = ballmass_i for h_i with the generalized
 * volume elements xm_j held fixed, following SPHYNX (calculate_density.f90/calculate_hNR.f90).
 * Both sums depend only on h_i and xm_j, i.e. iterating requires neither neighbor-list rebuilds
 * nor halo exchanges. The neighbor count inside the live support 2h rides along as a third sum
 * (self included, matching the 1 + findNeighbors convention of the neighbor-count guard); it is
 * carried as the arithmetic type T so that the result tuple stays uniform (GPU warp-reduction
 * fast path). On the symmetric GPU j-side the interaction is re-evaluated with swapped roles, so
 * the predicate counts against h_j there automatically.
 */
template<class T, class Kernel>
struct VeNRInteraction
{
    Kernel wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& /* r_ij */,
                              T r2) const
    {
        const auto [i, iPos, hi, xmassi, mi, h0i] = iData;
        const auto [j, jPos, hj, xmassj, mj, h0j] = jData;

        const auto hInv = T(1) / hi;

        T dist = std::sqrt(r2);
        T vloc = dist * hInv;
        T w    = wh(vloc);
        T dw   = wh.derivative(vloc);

        T kxi  = w * xmassj;
        T dkxi = (T(3) * w + vloc * dw) * xmassj;
        // don't count the symmetric counterpart
        T count = T(r2 < T(4) * hi * hi);

        return std::make_tuple(kxi, dkxi, count);
    }
};

template<class T, class Tc>
struct VeNRPostamble
{
    Tc K;

    T* ballmass;
    // Don't reset ballmass if the neighbor count is too low due to h > hExtFactor * h0
    T hExtFactor;
    // Limit maximal positive change of h (neighbour list incomplete)
    T hWallFactor;
    //! @brief relative h change below which the iteration counts as converged
    T hNRTol;
    /*! @brief neighbor-count band for the convergence check, [ngmin, ngmax].
     * neighbors > ngmax -> ballmass reset
     * neighbors < ngmin -> ballmass reset only if h <= hExtFactor * h0
     */
    T ngmin;
    T ngmax;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, xmassi, mi, h0i] = iData;
        auto [kxi, dkxi, cnti]                    = result;

        auto hInv  = T(1) / hi;
        auto h3Inv = hInv * hInv * hInv;

        kxi *= K * h3Inv;
        T dkxdh = -K * h3Inv * hInv * dkxi;

        T          ballmassi = ballmass[i];
        const bool resetFlag = ballmassi <= T(0);
        if (resetFlag)
        {
            //! reset ballmass using density at the current h
            ballmassi   = kxi * mi / xmassi / h3Inv;
            ballmass[i] = ballmassi;
        }

        // Newton-Raphson step for g(h) = ballmass / h^3 - rho(h), rho = kx * m / xm
        T g  = ballmassi * h3Inv - kxi * mi / xmassi;
        T dg = -(T(3) * ballmassi * h3Inv * hInv + dkxdh * mi / xmassi);

        T deltah = -g / dg;
        if (!std::isfinite(deltah)) { deltah = T(0); }

        /* Omega measures the h-sensitivity of the density with fixed volume elements. It vanishes
         * when all kernel mass sits near the center or the edge of the support (isolated particle
         * in a void, or a clustered pair). There the constraint may have no root: Newton would run
         * h towards zero in always-accepted steps (observed as a blow-up trigger), so freeze h and
         * leave the adjustment to the neighbor-count management. */
        T omega = T(1) + hi * dkxdh / (T(3) * kxi);
        if (!(omega > T(0.1))) { deltah = T(0); }

        // Limit change of h per iteration to prevent divergence
        T hNew = hi + deltah;
        hNew   = stl::min(hNew, T(1.1) * hi);
        hNew   = stl::max(hNew, T(0.5) * hi);
        // Limit total change of h
        hNew = stl::min(hNew, hWallFactor * h0i);
        hNew = stl::max(hNew, T(0.5) * h0i);

        const bool converged = std::abs(hNew - hi) < hNRTol * hi;
        // Whether to trust nc < ngmin
        const bool countComplete = (hi <= hExtFactor * h0i);
        const bool ncOutsideBand = (cnti < ngmin && countComplete) || (cnti - T(1) > ngmax);
        // Set ballmass reset flag if h converged outside the neighbor band [ngmin, ngmax]
        if (!resetFlag && converged && ncOutsideBand)
        {
            // set h to the value inside [ngmin, ngmax] (found by updateSmoothingLengthIterative())
            hNew        = h0i;
            ballmass[i] = T(0);
        }

        return std::make_tuple(kxi, hNew, cnti);
    }
};

//! @brief one Newton-Raphson iteration of the smoothing length
template<class Neighbordhood, class Tc, class T, class Tm>
void veNRIjLoop(const Neighbordhood& neighborhood, Tc K, const T* xm, const Tm* m, T* ballmass,
                KernelVariant<T> const& wh, T* kx, float hExtFactor, float hWallFactor, float hNRTol, float ngmin,
                float ngmax, const T* h0, T* hNew, T* count)
{
    std::visit(
        [&]<class Kernel>(Kernel wh)
        {
            neighborhood.ijLoop(cstone::ijloop::makeIjLoopData<Tc, T*>(
                std::make_tuple(xm, m, h0), std::make_tuple(kx, hNew, count), VeNRInteraction<T, Kernel>{wh},
                VeNRPostamble<T, Tc>{K, ballmass, T(hExtFactor), T(hWallFactor), T(hNRTol), T(ngmin), T(ngmax)}));
        },
        wh);
}

} // namespace sph

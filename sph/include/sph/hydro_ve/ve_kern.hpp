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

#include "sph/table_lookup.hpp"

namespace sph
{

template<class T>
struct VeInteraction
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& /* r_ij */,
                              T r2) const
    {
        const auto [i, iPos, hi, xmassi] = iData;
        const auto [j, jPos, hj, xmassj] = jData;

        auto hInv = T(1) / hi;

        T dist = std::sqrt(r2);
        T vloc = dist * hInv;
        T w    = lt::lookup(wh, vloc);

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
void veIjLoop(const Neighbordhood& neighborhood, Tc K, const T* xm, const T* wh, T* kx)
{ neighborhood.ijLoop(std::make_tuple(xm), std::make_tuple(kx), VeInteraction{wh}, VePostamble<T, Tc>{K}); }

/*! @brief factor eta of the smoothing-length constraint rho * h^3 = eta * m
 *
 * Chosen such that a kernel support sphere of radius 2h contains ng0 particles of equal masses,
 * consistent with the initialization convention h = 0.5 * cbrt(3 * ng0 * m / (4 * pi * rho)).
 */
template<class T>
constexpr T ballmassEta(unsigned ng0)
{ return T(3) * T(ng0) / (T(32) * M_PI); }

/*! @brief kernel sums for the Newton-Raphson iteration of the smoothing length
 *
 * Computes the sums needed to solve rho_i(h_i) * h_i^3 = eta * m_i for h_i with the generalized
 * volume elements xm_j held fixed, following SPHYNX (calculate_density.f90/calculate_hNR.f90).
 * Both sums depend only on h_i and xm_j, i.e. iterating requires neither neighbor-list rebuilds
 * nor halo exchanges.
 */
template<class T>
struct VeNRInteraction
{
    const T *wh, *whd;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& /* r_ij */,
                              T r2) const
    {
        const auto [i, iPos, hi, xmassi, mi] = iData;
        const auto [j, jPos, hj, xmassj, mj] = jData;

        auto hInv = T(1) / hi;

        T dist = std::sqrt(r2);
        T vloc = dist * hInv;
        T w    = lt::lookup(wh, vloc);
        T dw   = lt::lookup(whd, vloc);

        T kxi = w * xmassj;
        //! contribution to dkx_i/dh_i, missing factor -K/h^4 is applied in the postamble
        T dkxi = (T(3) * w + vloc * dw) * xmassj;

        return std::make_tuple(kxi, dkxi);
    }
};

template<class T, class Tc>
struct VeNRPostamble
{
    Tc K;
    //! @brief constraint constant, rho * h^3 = eta * m
    Tc eta;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, xmassi, mi] = iData;
        auto [kxi, dkxi]                     = result;

        auto hInv  = T(1) / hi;
        auto h3Inv = hInv * hInv * hInv;

        kxi *= K * h3Inv;
        T dkxdh = -K * h3Inv * hInv * dkxi;

        // Newton-Raphson step for g(h) = eta * m / h^3 - rho(h), rho = kx * m / xm
        T ballmass = eta * mi;
        T g        = ballmass * h3Inv - kxi * mi / xmassi;
        T dg       = -(T(3) * ballmass * h3Inv * hInv + dkxdh * mi / xmassi);

        T deltah = -g / dg;
        if (!std::isfinite(deltah)) { deltah = T(0); }
        /* Limit steps to 20% of h. SPHYNX (calculate_hNR.f90) rejects such steps entirely, but its
         * ballmass target is re-baselined to the current h whenever neighbor counts get out of
         * bounds, whereas the fixed target eta * m can legitimately require larger adjustments,
         * e.g. at density discontinuities or free surfaces. Clamping keeps making progress there. */
        T maxStep = T(0.2) * hi;
        deltah    = deltah > maxStep ? maxStep : (deltah < -maxStep ? -maxStep : deltah);

        return std::make_tuple(kxi, hi + deltah);
    }
};

/*! @brief one Newton-Raphson iteration of the smoothing length
 *
 * The updated smoothing length is stored in @p hNew (may not alias h: h_j is read concurrently),
 * @p kx receives the volume element normalization evaluated at the old h.
 */
template<class Neighbordhood, class Tc, class T, class Tm>
void veNRIjLoop(const Neighbordhood& neighborhood, Tc K, Tc eta, const T* xm, const Tm* m, const T* wh, const T* whd,
                T* kx, T* hNew)
{
    neighborhood.ijLoop(std::make_tuple(xm, m), std::make_tuple(kx, hNew), VeNRInteraction<T>{wh, whd},
                        VeNRPostamble<T, Tc>{K, eta});
}

} // namespace sph

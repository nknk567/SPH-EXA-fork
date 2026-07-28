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

#include "cstone/primitives/stl.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
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
{
    neighborhood.ijLoop(std::make_tuple(xm), std::make_tuple(kx), VeInteraction{wh}, VePostamble<T, Tc>{K});
}

/*! @brief factor eta of the smoothing-length constraint rho * h^3 = eta * m
 *
 * Chosen such that a kernel support sphere of radius 2h contains ng0 particles of equal masses,
 * consistent with the initialization convention h = 0.5 * cbrt(3 * ng0 * m / (4 * pi * rho)).
 */
template<class T>
constexpr T ballmassEta(unsigned ng0)
{
    return T(3) * T(ng0) / (T(32) * M_PI);
}

/*! @brief SPH-smoothed volume estimate, used as the volume element weights of the next time-step
 *
 * volstd_i = sum_j V_j^2 W_ij / sum_j V_j W_ij with V = xm / kx: the Shepard-normalized SPH
 * interpolation of the converged particle volume (SPHYNX volstdprom is the unnormalized
 * variant). Smoothing suppresses particle-scale noise that the raw recursion xm <- xm / kx
 * would amplify. The normalization is essential at degenerate neighborhoods: unnormalized,
 * the interpolation of a particle whose neighbors sit near the kernel edge reduces to its
 * self-term K w0 V_i^2 / h^3 ~ 0.26 V_i at the NR root — a downward spiral that traps
 * particles at tiny h and volume with artifact velocity derivatives (observed to bind the rho
 * time step through single particles). Normalized, the isolated fixed point is exactly V_i
 * (neutral), edge particles relax toward their neighbors' volumes, and the result is bounded
 * by the largest neighbor volume, which also removes the vacuum-edge inflation engine.
 */
template<class T>
struct VolstdInteraction
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& /* r_ij */,
                              T r2) const
    {
        const auto [i, iPos, hi, xmassi, kxi] = iData;
        const auto [j, jPos, hj, xmassj, kxj] = jData;

        T dist = std::sqrt(r2);
        T vloc = dist / hi;
        T w    = lt::lookup(wh, vloc);

        T vj = xmassj / kxj;

        return std::make_tuple(vj * vj * w, vj * w);
    }
};

template<class T, class Tc>
struct VolstdPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, xmassi, kxi] = iData;
        auto [num, den]                       = result;

        //! den >= V_i * w(0) > 0 through the self contribution; fall back to the current volume
        T volstdi = den > T(0) ? num / den : xmassi / kxi;

        return std::make_tuple(volstdi);
    }
};

template<class Neighbordhood, class Tc, class T>
void volstdIjLoop(const Neighbordhood& neighborhood, Tc K, const T* xm, const T* kx, const T* wh, T* volstd)
{
    neighborhood.ijLoop(std::make_tuple(xm, kx), std::make_tuple(volstd), VolstdInteraction<T>{wh},
                        VolstdPostamble<T, Tc>{K});
}

/*! @brief kernel sums for the Newton-Raphson iteration of the smoothing length
 *
 * Computes the sums needed to solve rho_i(h_i) * h_i^3 = ballmass_i for h_i with the generalized
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
        const auto [i, iPos, hi, xmassi, mi, h0i] = iData;
        const auto [j, jPos, hj, xmassj, mj, h0j] = jData;

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
    //! @brief coefficient of the fixed constraint target, ballmass_i = ballmassEta(ng0) * m_i
    T etaBallmass;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, xmassi, mi, h0i] = iData;
        auto [kxi, dkxi]                          = result;

        const T ballmassi = etaBallmass * mi;

        auto hInv  = T(1) / hi;
        auto h3Inv = hInv * hInv * hInv;

        kxi *= K * h3Inv;
        T dkxdh = -K * h3Inv * hInv * dkxi;

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

        /* Clamp the step to [0.5, 1.5] * h: far from the root this limits the speed of approach
         * per iteration, the iterations continue until the relative change falls below hNRTol. */
        T hNew = hi + deltah;
        hNew   = stl::min(hNew, T(1.1) * hi);
        hNew   = stl::max(hNew, T(0.5) * hi);

        /* The neighbor list of this step was built with the capture radius extended by
         * hNRExtFactor around the step-start h0. Cumulative upward movement beyond that margin
         * would miss pairs inside the final support (breaking momentum/energy conservation in
         * every fast rarefaction), so cap it and let the affected particles finish converging
         * in the following steps. Downward movement always stays inside the list. */
        hNew = stl::min(hNew, T(hNRExtFactor) * h0i);

        /* Cumulative downward cap per step: where the volume elements are strongly non-uniform
         * (vacuum boundaries), the constraint can demand h far below the step-start value; the
         * per-iteration clamp alone still allows 0.5^nrIter within one step. Such a collapse
         * shrinks the neighborhood to a degenerate set within a single step (exploding kernel
         * gradients, observed as NaN forces on the second step of TDE restarts), so limit the
         * approach to the root to a factor 2 per step, mirroring the upward cap. */
        hNew = stl::max(hNew, T(0.5) * h0i);

        return std::make_tuple(kxi, hNew);
    }
};

/*! @brief one Newton-Raphson iteration of the smoothing length
 *
 * The updated smoothing length is stored in @p hNew (may not alias h: h_j is read concurrently),
 * @p kx receives the volume element normalization evaluated at the old h. The constraint target
 * ballmassEta(ng0) * m_i depends only on the desired neighbor count and the particle mass.
 * @p h0 is the smoothing length at the start of the step's NR iterations; the cumulative upward
 * movement is capped at hNRExtFactor * h0 to stay within the extended neighbor list.
 */
template<class Neighbordhood, class Tc, class T, class Tm>
void veNRIjLoop(const Neighbordhood& neighborhood, Tc K, unsigned ng0, const T* xm, const Tm* m, const T* h0,
                const T* wh, const T* whd, T* kx, T* hNew)
{
    neighborhood.ijLoop(std::make_tuple(xm, m, h0), std::make_tuple(kx, hNew), VeNRInteraction<T>{wh, whd},
                        VeNRPostamble<T, Tc>{K, ballmassEta<T>(ng0)});
}

/*! @brief one Newton-Raphson smoothing-length update for a single particle by direct octree traversal
 *
 * Functionally equivalent to one veNRIjLoop pass for particle @p i, but finds the neighbors by
 * traversing the octree (like findNeighbors) instead of consuming the prebuilt neighbor list.
 * Used for the tail of the NR iterations: once the bulk of the particles is converged, full
 * neighbor-list passes sweep all particles for the benefit of a residual O(0.01%); the symmetric
 * neighbor list cannot be restricted to a subset because each pair is stored once and scattered
 * to both endpoints. The NR update itself depends only on the particle's own smoothing length
 * and the fixed volume elements xm_j (neither h_j nor any intermediate state of the neighbors),
 * so iterating an arbitrary subset of particles is exact.
 *
 * @return the updated smoothing length of particle @p i (not committed to @p h)
 */
template<class Tc, class T, class Tm, class KeyType>
HOST_DEVICE_FUN T veNRTraversalUpdate(cstone::LocalIndex i, Tc K, T etaBallmass,
                                      const cstone::OctreeNsView<Tc, KeyType>& tree, const cstone::Box<Tc>& box,
                                      const Tc* x, const Tc* y, const Tc* z, const T* h, const T* xm, const Tm* m,
                                      const T* h0, const T* wh, const T* whd)
{
    const T                hi = h[i];
    const cstone::Vec3<Tc> particle{x[i], y[i], z[i]};
    const auto             iData = std::make_tuple(i, particle, hi, xm[i], m[i], h0[i]);

    VeNRInteraction<T> interaction{wh, whd};
    //! self contribution; the leaf sweep below skips i == j
    auto [kxsum, dkxsum] = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));

    const Tc radiusSq     = Tc(4.0) * Tc(hi) * Tc(hi);
    const Tc cellRadiusSq = radiusSq * tree.searchExtFactor * tree.searchExtFactor;

    auto pbc    = cstone::BoundaryType::periodic;
    bool anyPbc = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;
    bool usePbc = anyPbc && !cstone::insideBox(particle, {Tc(2) * hi, Tc(2) * hi, Tc(2) * hi}, box);

    auto overlapsPbc = [particle, cellRadiusSq, centers = tree.centers, sizes = tree.sizes,
                        &box](cstone::TreeNodeIndex idx)
    {
        if (sizes[idx][0] == 0 && sizes[idx][1] == 0 && sizes[idx][2] == 0) return false;
        return util::norm2(cstone::minDistance(particle, centers[idx], sizes[idx], box)) < cellRadiusSq;
    };
    auto overlaps = [particle, cellRadiusSq, centers = tree.centers, sizes = tree.sizes](cstone::TreeNodeIndex idx)
    {
        if (sizes[idx][0] == 0 && sizes[idx][1] == 0 && sizes[idx][2] == 0) return false;
        return util::norm2(cstone::minDistance(particle, centers[idx], sizes[idx])) < cellRadiusSq;
    };

    /* h_j in jData is passed as 0 instead of h[j]: the interaction does not use it, and not
     * reading it keeps concurrent tail updates of different particles free of data races. */
    auto sumBody = [&](cstone::LocalIndex j, Tc d2)
    {
        const auto jData   = std::make_tuple(j, cstone::Vec3<Tc>{x[j], y[j], z[j]}, T(0), xm[j], m[j], h0[j]);
        auto [kxc, dkxc]   = interaction(iData, jData, cstone::Vec3<Tc>{0, 0, 0}, T(d2));
        kxsum += kxc;
        dkxsum += dkxc;
    };
    auto searchBoxPbc = [&](cstone::TreeNodeIndex idx)
    {
        cstone::TreeNodeIndex leafIdx = tree.internalToLeaf[idx];
        for (cstone::LocalIndex j = tree.layout[leafIdx]; j < tree.layout[leafIdx + 1]; ++j)
        {
            if (j == i) { continue; }
            Tc d2 = cstone::distanceSq<true>(x[j], y[j], z[j], particle[0], particle[1], particle[2], box);
            if (d2 < radiusSq) { sumBody(j, d2); }
        }
    };
    auto searchBox = [&](cstone::TreeNodeIndex idx)
    {
        cstone::TreeNodeIndex leafIdx = tree.internalToLeaf[idx];
        for (cstone::LocalIndex j = tree.layout[leafIdx]; j < tree.layout[leafIdx + 1]; ++j)
        {
            if (j == i) { continue; }
            Tc d2 = cstone::distanceSq<false>(x[j], y[j], z[j], particle[0], particle[1], particle[2], box);
            if (d2 < radiusSq) { sumBody(j, d2); }
        }
    };

    if (usePbc) { cstone::singleTraversal(tree.childOffsets, tree.parents, overlapsPbc, searchBoxPbc); }
    else { cstone::singleTraversal(tree.childOffsets, tree.parents, overlaps, searchBox); }

    auto [kxi, hNew] = VeNRPostamble<T, Tc>{K, etaBallmass}(iData, std::make_tuple(kxsum, dkxsum));
    return hNew;
}

} // namespace sph

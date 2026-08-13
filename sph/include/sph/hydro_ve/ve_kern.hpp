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
 * nor halo exchanges. The neighbor count inside the live support 2h rides along as a third sum
 * (self included, matching the 1 + findNeighbors convention of the neighbor-count guard); it is
 * carried as the arithmetic type T so that the result tuple stays uniform (GPU warp-reduction
 * fast path). On the symmetric GPU j-side the interaction is re-evaluated with swapped roles, so
 * the predicate counts against h_j there automatically.
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
        //! symmetric pairs are processed out to 2 * max(h_i, h_j): count only inside own support
        T cnti = T(r2 < T(4) * hi * hi);

        return std::make_tuple(kxi, dkxi, cnti);
    }
};

template<class T, class Tc>
struct VeNRPostamble
{
    Tc K;
    /*! @brief per-particle constraint target, rho_i * h_i^3 = ballmass[i]
     *
     * Nominally ballmassEta(ng0) * m_i (see the propagator's fill). A non-positive entry is
     * the recompute signal: set by the neighbor-count band check below when a converging point
     * fell outside the band, or the zero-fill of a restart from a checkpoint without the field.
     * The target is then re-seeded to rho * h^3 at the current h (SPHYNX findneighbors.f90,
     * ballmass = promro * h^3). The signal is durable across steps: a zero left over when the
     * step's iterations end is consumed by the unconditional full sweep of the next step's
     * first NR pass, at that step's post-guard h; nothing outside the NR passes reads the field.
     * Written only for the owned particle i of the postamble: race-free.
     */
    T* ballmass;
    /*! @brief neighbor-list capture extension: pairs are enumerated out to 2 * hExtFactor * h0
     *
     * Above hi = hExtFactor * h0 the summed pair set is incomplete: kx/dkx miss only the
     * kernel tail (the missed shell starts at r/h = 2 * hExtFactor * h0 / h, where the sinc
     * kernel is orders of magnitude below its mean pair weight — and growth into vacuum misses
     * nothing at all), but the neighbor count cnti becomes a plain lower bound. Used by the
     * band check below to decide when cnti is trustworthy.
     */
    T hExtFactor;
    /*! @brief cumulative upward h cap per step, as a factor on the step-start h0
     *
     * Equal to hExtFactor by default (h never leaves the complete neighbor list). May be set
     * larger to let h migrate upward faster than the capture extension allows (SPHYNX operates
     * this way permanently: no list extension at all, NR moves h on the frozen list): the
     * truncation error this admits is the kernel tail described at hExtFactor. Keep <~ 1.3 —
     * beyond that the missed shell reaches into kernel weights that are no longer negligible
     * when the missed region is dense.
     */
    T hWallFactor;
    //! @brief relative h change below which the iteration counts as converged (hNRTol)
    T tol;
    /*! @brief neighbor-count band for the convergence check, [bandMin, bandMax]
     *
     * When a particle converges (relative h change < tol) but its neighbor count at the
     * converging point lies outside the band, h is reset to the step-start value h0 (the
     * neighbor-count guard's output, in band by construction) and the target is flagged for
     * a re-seed there (ballmass = 0). Wall-pinned particles sit at the cumulative caps with
     * a zero h change, so they are checked at the wall: by monotonicity of the count along
     * the remaining travel direction, an out-of-band wall count implies an out-of-band root
     * (below-band only while the count is complete, i.e. inside the capture radius — see the
     * gate at the check site).
     * In-band wall-pinned particles are left alone — they are legitimately migrating to a
     * new h over several steps at the per-step rate the caps allow. The check is skipped on
     * the pass that consumes a recompute signal (the target was not positive on entry), so
     * the decision is made at most once per step even where h0's own count is out of band.
     * Disabled band (bandMin = 0, bandMax = huge): plain NR without the check.
     */
    T bandMin, bandMax;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, xmassi, mi, h0i] = iData;
        auto [kxi, dkxi, cnti]                    = result;

        auto hInv  = T(1) / hi;
        auto h3Inv = hInv * hInv * hInv;

        kxi *= K * h3Inv;
        T dkxdh = -K * h3Inv * hInv * dkxi;

        T          ballmassi         = ballmass[i];
        const bool targetWasPositive = ballmassi > T(0);
        if (!targetWasPositive)
        {
            //! recompute signal: seed the target with the density at the current h
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

        /* Clamp the step to [0.5, 1.5] * h: far from the root this limits the speed of approach
         * per iteration, the iterations continue until the relative change falls below hNRTol. */
        T hNew = hi + deltah;
        hNew   = stl::min(hNew, T(1.1) * hi);
        hNew   = stl::max(hNew, T(0.5) * hi);

        /* Cumulative upward cap per step (the wall). At hWallFactor == hExtFactor movement
         * never leaves the complete neighbor list; a larger wall admits the capture-truncation
         * error described at the hExtFactor member in exchange for faster upward migration —
         * the affected particles finish converging in the following steps, each step rebuilding
         * the list around the grown h0. Downward movement always stays inside the list. */
        hNew = stl::min(hNew, hWallFactor * h0i);

        /* Cumulative downward cap per step: where the volume elements are strongly non-uniform
         * (vacuum boundaries), the constraint can demand h far below the step-start value; the
         * per-iteration clamp alone still allows 0.5^nrIter within one step. Such a collapse
         * shrinks the neighborhood to a degenerate set within a single step (exploding kernel
         * gradients, observed as NaN forces on the second step of TDE restarts), so limit the
         * approach to the root to a factor 2 per step, mirroring the upward cap. */
        hNew = stl::max(hNew, T(0.5) * h0i);

        /* Neighbor-count band check at the converging point, see the bandMin/bandMax doc.
         * Gated on convergence, so cnti (summed at hi ~ hNew) is the count at the converging
         * point; a reset makes the particle unconverged when h0 is more than tol away, so a
         * following pass or tail iteration re-seeds the target at h0 within this step.
         * Beyond the capture radius (hi > hExtFactor * h0, reachable when the wall exceeds the
         * capture extension) cnti is a lower bound: an above-band violation is then still
         * certain, but a below-band count is inconclusive — defer that decision until the
         * particle converges inside the capture radius of a later step's rebuilt list. The
         * deferral is also what lets a fast-migrating wall-pinned particle keep its stale
         * target as the engine pulling h up, instead of re-seeding at every step's wall. */
        const bool cntComplete = hi <= hExtFactor * h0i;
        if (targetWasPositive && std::abs(hNew - hi) < tol * hi &&
            ((cnti < bandMin && cntComplete) || cnti - T(1) > bandMax))
        {
            hNew        = h0i;
            ballmass[i] = T(0);
        }

        return std::make_tuple(kxi, hNew, cnti);
    }
};

/*! @brief one Newton-Raphson iteration of the smoothing length
 *
 * The updated smoothing length is stored in @p hNew (may not alias h: h_j is read concurrently),
 * @p kx receives the volume element normalization evaluated at the old h, @p cnt the neighbor
 * count inside 2h (self included) that the convergence-point band check tested. The constraint
 * target is the per-particle @p ballmass field, nominally ballmassEta(ng0) * m_i; non-positive
 * entries are recompute signals resolved (and written back) by the postamble, see VeNRPostamble.
 * @p h0 is the smoothing length at the start of the step's NR iterations; the cumulative upward
 * movement is capped at hWallFactor * h0 (>= the list extension hExtFactor, see VeNRPostamble).
 */
template<class Neighbordhood, class Tc, class T, class Tm>
void veNRIjLoop(const Neighbordhood& neighborhood, Tc K, float hExtFactor, float hWallFactor, float tol, float bandMin,
                float bandMax, const T* xm, const Tm* m, const T* h0, T* ballmass, const T* wh, const T* whd, T* kx,
                T* hNew, T* cnt)
{
    neighborhood.ijLoop(std::make_tuple(xm, m, h0), std::make_tuple(kx, hNew, cnt), VeNRInteraction<T>{wh, whd},
                        VeNRPostamble<T, Tc>{K, ballmass, T(hExtFactor), T(hWallFactor), T(tol), T(bandMin),
                                             T(bandMax)});
}

/*! @brief one Newton-Raphson smoothing-length update for a single particle by direct octree traversal
 *
 * Functionally equivalent to one veNRIjLoop pass for particle @p i, but finds the neighbors by
 * traversing the octree (like findNeighbors) instead of consuming the prebuilt neighbor list.
 * Used for the tail of the NR iterations: once the bulk of the particles is converged, full
 * neighbor-list passes sweep all particles for the benefit of a residual O(0.01%); the symmetric
 * neighbor list cannot be restricted to a subset because each pair is stored once and scattered
 * to both endpoints. The NR update itself depends only on the particle's own smoothing length
 * @p hi (passed by value, the smoothing-length array is not accessed) and the fixed volume
 * elements xm_j — neither h_j nor any intermediate state of the neighbors — so iterating an
 * arbitrary subset of particles, each freely running to its own convergence, is exact.
 *
 * @return the updated smoothing length of particle @p i (not committed); @p cnt[i] receives the
 *         neighbor count inside 2h (self included) that the convergence-point band check tested
 */
template<class Tc, class T, class Tm, class KeyType>
HOST_DEVICE_FUN T veNRTraversalUpdate(cstone::LocalIndex i, T hi, Tc K, T* ballmass, T hExtFactor, T hWallFactor,
                                      T tol, T bandMin, T bandMax, const cstone::OctreeNsView<Tc, KeyType>& tree,
                                      const cstone::Box<Tc>& box, const Tc* x, const Tc* y, const Tc* z, const T* xm,
                                      const Tm* m, const T* h0, const T* wh, const T* whd, T* cnt)
{
    const cstone::Vec3<Tc> particle{x[i], y[i], z[i]};
    const auto             iData = std::make_tuple(i, particle, hi, xm[i], m[i], h0[i]);

    VeNRInteraction<T> interaction{wh, whd};
    //! self contribution; the leaf sweep below skips i == j
    auto [kxsum, dkxsum, cntsum] = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));

    /* Enumeration is capped at the capture radius of the prebuilt lists: beyond it a live-radius
     * search would be complete on interior ranks but halo-truncated near rank boundaries, making
     * the root decomposition-dependent — and different from the ijloop passes. The cap keeps the
     * tail exactly equivalent to an ijloop pass (a no-op while hi <= hExtFactor * h0); the
     * interaction still cuts at the live 2h through the kernel support. */
    const T  hSearch      = stl::min(hi, hExtFactor * h0[i]);
    const Tc radiusSq     = Tc(4.0) * Tc(hSearch) * Tc(hSearch);
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

    //! h_j in jData is a placeholder: the interaction does not use it (gather in h_i only)
    auto sumBody = [&](cstone::LocalIndex j, Tc d2)
    {
        const auto jData       = std::make_tuple(j, cstone::Vec3<Tc>{x[j], y[j], z[j]}, T(0), xm[j], m[j], h0[j]);
        auto [kxc, dkxc, cntc] = interaction(iData, jData, cstone::Vec3<Tc>{0, 0, 0}, T(d2));
        kxsum += kxc;
        dkxsum += dkxc;
        cntsum += cntc;
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

    auto [kxi, hNew, cnti] = VeNRPostamble<T, Tc>{K, ballmass, hExtFactor, hWallFactor, tol, bandMin, bandMax}(
        iData, std::make_tuple(kxsum, dkxsum, cntsum));
    cnt[i] = cnti;
    return hNew;
}

} // namespace sph

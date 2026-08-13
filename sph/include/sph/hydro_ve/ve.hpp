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
void setVolumeElements(const GroupView& grp, Dataset& d, const Tv* volstd, float volstdGrowFactor,
                       float volstdShrinkFactor)
{
    if constexpr (d.useGpu) { gpu::setVolumeElements(grp, d, volstd, volstdGrowFactor, volstdShrinkFactor); }
    else
    {
        auto* xm = d.xm.data();
#pragma omp parallel for schedule(static)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            //! clamp the per-step change of the carried volume, see the ve-nr volstdGrow/ShrinkFactor parameters
            Tv lo = xm[i] / Tv(volstdShrinkFactor);
            Tv hi = xm[i] * Tv(volstdGrowFactor);
            xm[i] = stl::min(stl::max(volstd[i], lo), hi);
        }
    }
}

/*! @brief fill the NR constraint target with its nominal value ballmassEta(ng0) * m
 *
 * Every step when the per-particle ballmass mode is off (fixed global target, legacy behavior),
 * only on the first step of a fresh run when it is on (afterwards the field is frozen and only
 * rewritten through the recompute signal of the convergence-point neighbor-count band check,
 * see VeNRPostamble).
 */
template<class Dataset>
void fillNominalBallmass(const GroupView& grp, Dataset& d)
{
    if constexpr (d.useGpu) { gpu::fillNominalBallmass(grp, d); }
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

/*! @brief one Newton-Raphson iteration for the smoothing length constraint rho * h^3 = ballmass
 *
 * Iterates over the fixed neighbor list with fixed volume elements xm and updates h of locally
 * owned particles in place. Uses the ay field as scratch space for the updated smoothing length,
 * the az field for the relative h change per particle (consumed by computeVeNRTail) and the ax
 * field for the neighbor count inside 2h tested by the convergence-point band check.
 * The constraint target is the per-particle ballmass field, nominally ballmassEta(ng0) * m;
 * converging points whose neighbor count falls outside [bandMin, bandMax] are reset to h0 with
 * the target flagged for a re-seed there, see VeNRPostamble.
 * @p h0 holds the smoothing lengths at the start of the step's NR iterations (filled here when
 * @p firstIteration is set); the cumulative upward h movement is capped at hWallFactor * h0
 * (>= the neighbor-list extension hExtFactor; growth beyond the extension trades the capture
 * truncation of the listed sums for faster migration, see VeNRPostamble).
 *
 * @return pass statistics: the number of locally owned particles with a relative h change
 *         >= @p tol (zero means the iterations are converged), the per-iteration clamp hits
 *         and the band-check resets of this pass
 */
template<typename Tc, class Dataset, class Tv>
NRPassStats computeVeNR(const GroupView& grp, Dataset& d, const cstone::Box<Tc>& box, Tv* h0, bool firstIteration,
                        float tol, float hExtFactor, float hWallFactor, float bandMin, float bandMax)
{
    if constexpr (d.useGpu)
    {
        return gpu::computeVeNR(grp, d, box, h0, firstIteration, tol, hExtFactor, hWallFactor, bandMin, bandMax);
    }
    else
    {
        if (firstIteration) { std::copy(d.h.data(), d.h.data() + d.x.size(), h0); }
        veNRIjLoop(d.neighborhood, d.K, hExtFactor, hWallFactor, tol, bandMin, bandMax, d.xm.data(), d.m.data(), h0,
                   d.ballmass.data(), d.wh.data(), d.whd.data(), d.kx.data(), d.ay.data(), d.ax.data());

        using Th               = std::decay_t<decltype(d.h[0])>;
        const Th* hNew         = d.ay.data();
        const Th* bm           = d.ballmass.data();
        Th*       h            = d.h.data();
        Th*       relDh        = d.az.data();
        size_t    numUnconverged = 0, capUp = 0, capDown = 0, numReset = 0;
#pragma omp parallel for schedule(static) reduction(+ : numUnconverged, capUp, capDown, numReset)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            Th rel   = std::abs(hNew[i] - h[i]) / h[i];
            //! bitwise comparison against the clamp values the postamble computes from the same inputs
            capUp += hNew[i] == Th(1.1) * h[i];
            capDown += hNew[i] == Th(0.5) * h[i];
            relDh[i] = rel;
            h[i]     = hNew[i];
            numUnconverged += rel >= Th(tol);
            numReset += bm[i] == Th(0);
        }
        return {numUnconverged, capUp, capDown, numReset};
    }
}

/*! @brief count particles whose final h ended the NR iterations pinned at the cumulative walls
 *
 * @return {at hWallFactor * h0 (upper wall), at 0.5 * h0 (lower wall)}; these particles are
 *         frozen OFF their NR root (relDh = 0 at the wall counts as converged), so they do not
 *         appear in the unconverged statistics — this is the complementary view.
 */
template<class Dataset, class Tv>
std::pair<size_t, size_t> countHWallPinned(const GroupView& grp, Dataset& d, const Tv* h0, float hWallFactor)
{
    if constexpr (d.useGpu) { return gpu::countHWallPinned(grp, d, h0, hWallFactor); }
    else
    {
        using Th        = std::decay_t<decltype(d.h[0])>;
        const Th* h     = d.h.data();
        size_t    up    = 0, down = 0;
#pragma omp parallel for schedule(static) reduction(+ : up, down)
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            up += h[i] == Th(hWallFactor) * h0[i];
            down += h[i] == Th(0.5) * h0[i];
        }
        return {up, down};
    }
}

/*! @brief finish the NR smoothing-length iterations for the unconverged residual only
 *
 * Extracts the particles whose relative h change of the last computeVeNR pass (az scratch) was
 * still >= @p tol and iterates only those, each by direct octree traversal (veNRTraversalUpdate)
 * instead of full neighbor-list passes over all particles. Fused: every particle runs freely to
 * its own convergence (h kept local, committed once) instead of lockstep passes over the whole
 * subset — exact, because the NR update of a particle depends only on its own h and the fixed
 * volume elements. The per-pass unconverged counts of the lockstep formulation are recovered
 * as suffix sums over the histogram of convergence iterations. Converged particles keep their
 * last h instead of accumulating further sub-tolerance refinements.
 *
 * @param maxPasses           remaining iteration budget (hNRIterMax minus the passes already done)
 * @param unconvergedPerPass  appends the local unconverged count after each pass (diagnostic)
 * @return the number of passes performed (the largest convergence iteration among the subset)
 */
template<typename Tc, class Dataset, class Tv>
unsigned computeVeNRTail(const GroupView& grp, Dataset& d, const cstone::Box<Tc>& box, const Tv* h0,
                         unsigned maxPasses, std::vector<size_t>& unconvergedPerPass, float tol, float hExtFactor,
                         float hWallFactor, float bandMin, float bandMax, size_t& capUp, size_t& capDown,
                         size_t& numReset)
{
    if constexpr (d.useGpu)
    {
        return gpu::computeVeNRTail(grp, d, box, h0, maxPasses, unconvergedPerPass, tol, hExtFactor, hWallFactor,
                                    bandMin, bandMax, capUp, capDown, numReset);
    }
    else
    {
        using Th = std::decay_t<decltype(d.h[0])>;
        if (maxPasses == 0) { return 0; }

        std::vector<cstone::LocalIndex> subset;
        const Th*                       relDh = d.az.data();
        for (cstone::LocalIndex i = grp.firstBody; i < grp.lastBody; ++i)
        {
            if (relDh[i] >= Th(tol)) { subset.push_back(i); }
        }
        if (subset.empty()) { return 0; }

        Th*                 h  = d.h.data();
        const Th*           bm = d.ballmass.data();
        std::vector<size_t> bins(maxPasses + 2, 0);
        size_t              tailCapUp = 0, tailCapDown = 0, tailReset = 0;
#pragma omp parallel reduction(+ : tailCapUp, tailCapDown, tailReset)
        {
            std::vector<size_t> localBins(maxPasses + 2, 0);
#pragma omp for schedule(dynamic) nowait
            for (size_t s = 0; s < subset.size(); ++s)
            {
                cstone::LocalIndex i     = subset[s];
                Th                 hi    = h[i];
                unsigned           kConv = maxPasses + 1;
                bool               fired = false;
                for (unsigned it = 1; it <= maxPasses; ++it)
                {
                    Th hNew = veNRTraversalUpdate(i, hi, d.K, d.ballmass.data(), Th(hExtFactor), Th(hWallFactor),
                                                  Th(tol), Th(bandMin), Th(bandMax), d.treeView, box, d.x.data(),
                                                  d.y.data(), d.z.data(), d.xm.data(), d.m.data(), h0, d.wh.data(),
                                                  d.whd.data(), d.ax.data());
                    /* signals entering the tail were consumed by the update's re-seed branch, so a
                     * zero after an update is a band-check reset fired by this update; the once-per-
                     * step gate makes fired-detection through the array exact */
                    fired = fired || bm[i] == Th(0);
                    tailCapUp += hNew == Th(1.1) * hi;
                    tailCapDown += hNew == Th(0.5) * hi;
                    Th rel  = std::abs(hNew - hi) / hi;
                    hi      = hNew;
                    if (rel < Th(tol))
                    {
                        kConv = it;
                        break;
                    }
                }
                h[i] = hi;
                tailReset += fired;
                ++localBins[kConv];
            }
#pragma omp critical
            for (size_t k = 0; k < bins.size(); ++k)
            {
                bins[k] += localBins[k];
            }
        }
        capUp += tailCapUp;
        capDown += tailCapDown;
        numReset += tailReset;

        //! per-pass unconverged counts = suffix sums; passes performed = last needed iteration
        unsigned passes = maxPasses;
        while (passes > 1 && bins[passes] == 0 && bins[passes + 1] == 0)
        {
            --passes;
        }
        for (unsigned p = 1; p <= passes; ++p)
        {
            size_t stillUnconverged = 0;
            for (unsigned k = p + 1; k <= maxPasses + 1; ++k)
            {
                stillUnconverged += bins[k];
            }
            unconvergedPerPass.push_back(stillUnconverged);
        }
        return passes;
    }
}

} // namespace sph

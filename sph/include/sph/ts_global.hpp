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
 * @brief Min-reduction to determine global timestep
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mpi.h>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "sph/sph_gpu.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/array.hpp"

namespace sph
{

//! @brief limit time-step based on accelerations when gravity is enabled
//! Computes etaAcc * min(sqrt(h[i] / norm(a[i])))
template<class Dataset>
auto accelerationTimestep(size_t first, size_t last, const Dataset& d)
{
    using T = typename Dataset::RealType;
    if (last <= first) return std::numeric_limits<T>::infinity();

    //! @brief minimum value of all {h_i^2 / a_i^2}
    T minH2_A2 = std::numeric_limits<T>::infinity();
    if constexpr (d.useGpu)
    {
        minH2_A2 = accelerationTimestepGPU(first, last, rawPtr(d.ax), rawPtr(d.ay), rawPtr(d.az), rawPtr(d.h));
    }
    else
    {
#pragma omp parallel for reduction(min : minH2_A2)
        for (size_t i = first; i < last; ++i)
        {
            cstone::Vec3<T> A{d.ax[i], d.ay[i], d.az[i]};
            minH2_A2 = std::min(minH2_A2, d.h[i] * d.h[i] / norm2(A));
        }
    }

    return d.etaAcc * std::pow(minH2_A2, 0.25);
}

//! @brief limit time-step based on divergence of velocity, this is called in the propagator when Divv is available
template<class Dataset>
auto rhoTimestep(size_t first, size_t last, const Dataset& d)
{
    using T = std::decay_t<decltype(*d.divv.data())>;

    T maxDivv = -INFINITY;
    if constexpr (d.useGpu)
    {
        if (d.divv.empty()) { throw std::runtime_error("Divv needs to be available in rhoTimestep\n"); }
        auto minmax =
            cstone::minMax(cstone::execution::gpuDefaultStream, rawPtr(d.divv) + first, rawPtr(d.divv) + last);
        maxDivv = std::get<1>(minmax);
    }
    else
    {
        if (d.divv.empty()) { throw std::runtime_error("Divv needs to be available in rhoTimestep\n"); }

#pragma omp parallel for reduction(max : maxDivv)
        for (size_t i = first; i < last; ++i)
        {
            maxDivv = std::max(d.divv[i], maxDivv);
        }
    }
    return d.Krho / std::abs(maxDivv);
}

template<class Dataset, class... Ts>
void computeTimestep(size_t first, size_t last, Dataset& d, Ts... extraTimesteps)
{
    using T = typename Dataset::RealType;

    T minDtAcc = (d.g != 0.0) ? accelerationTimestep(first, last, d) : INFINITY;

    constexpr size_t numCandidates = 4 + sizeof...(Ts);
    util::array<T, numCandidates> candidates{minDtAcc, T(d.minDtCourant), T(d.minDtRho),
                                             T(d.maxDtIncrease * d.minDt), T(extraTimesteps)...};

    /* A NaN candidate (e.g. a NaN Courant time from a single corrupted particle) must not
     * propagate into the global time step: the growth-capped previous step is always finite, so
     * dropping NaN candidates keeps the run integrating, while the [NAN] tag in the per-step
     * print below points at the failing constraint. Infinite candidates are normal (inactive
     * constraints) and never win the min. */
    T    minDtLoc = INFINITY;
    bool hadNan   = false;
    for (size_t c = 0; c < numCandidates; ++c)
    {
        if (std::isnan(candidates[c]))
        {
            candidates[c] = INFINITY;
            hadNan        = true;
        }
        minDtLoc = std::min(minDtLoc, candidates[c]);
    }

    util::array<T, 4 + numCandidates> varsIn, varsOut;
    varsIn[0] = minDtLoc;
    varsIn[1] = 0;
    varsIn[2] = -T(d.size() - last + first);
    varsIn[3] = hadNan ? T(-1) : T(0);
    for (size_t c = 0; c < numCandidates; ++c)
    {
        varsIn[4 + c] = candidates[c];
    }
    if constexpr (d.useGpu) { varsIn[1] = -int(d.stackUsedGravity); }
    MPI_Allreduce(varsIn.data(), varsOut.data(), varsIn.size(), MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);
    T minDtGlobal = varsOut[0];
    if constexpr (d.useGpu) { d.stackUsedGravity = int(-varsOut[1]); }
    d.maxHalos = int(-varsOut[2]);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "# dt: acc=" << varsOut[4] << " courant=" << varsOut[5] << " rho=" << varsOut[6]
                  << " growth=" << varsOut[7];
        for (size_t e = 0; e < sizeof...(Ts); ++e)
        {
            std::cout << " extra" << e << "=" << varsOut[8 + e];
        }
        if (varsOut[3] < T(0)) { std::cout << " [NAN]"; }
        std::cout << std::endl;
    }

    d.ttot += minDtGlobal;

    d.minDt_m1 = d.minDt;
    d.minDt    = minDtGlobal;
}

} // namespace sph

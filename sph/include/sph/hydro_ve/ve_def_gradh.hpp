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
#include "ve_def_gradh_kern.hpp"

namespace sph
{
template<class Tc, class Dataset>
void computeVeDefGradhImpl(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<Tc>& box)
{
    const cstone::LocalIndex* neighbors      = d.neighbors.data();
    const unsigned*           neighborsCount = d.nc.data();

    const auto* x = d.x.data();
    const auto* y = d.y.data();
    const auto* z = d.z.data();
    const auto* h = d.h.data();
    const auto* m = d.m.data();

    const auto* wh  = d.wh.data();
    const auto* whd = d.whd.data();

    const auto* xm = d.xm.data();

    auto* kx    = d.kx.data();
    auto* gradh = d.gradh.data();

    const Tc K         = d.K;
    const Tc sincIndex = d.sincIndex;

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; i++)
    {
        size_t   ni        = i - startIndex;
        unsigned nc        = std::min(neighborsCount[i], d.ngmax);
        auto [kxi, gradhi] = veDefGradhJLoop(i, K, box, neighbors + d.ngmax * ni, nc, x, y, z, h, m, wh, whd, xm);

        kx[i]    = kxi;
        gradh[i] = gradhi;

#ifndef NDEBUG
        auto rhoi = kxi * m[i] / xm[i];
        if (std::isnan(rhoi))
            printf("ERROR::Density(%zu) density %f, position: (%f %f %f), h: %f\n", i, rhoi, x[i], y[i], z[i], h[i]);
#endif
    }
}

template<typename Tc, class Dataset>
void computeVeDefGradh(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<Tc>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeVeDefGradh(startIndex, endIndex, d, box);
    }
    else { computeVeDefGradhImpl(startIndex, endIndex, d, box); }
}

} // namespace sph

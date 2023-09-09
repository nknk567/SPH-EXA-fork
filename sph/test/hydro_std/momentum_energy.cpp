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
 * @brief SPH density kernel tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"

#include "sph/hydro_std/momentum_energy_kern.hpp"
#include "sph/tables.hpp"

using namespace sph;

TEST(MomentumEnergy, JLoop)
{
    using T = double;

    T sincIndex = 6.0;
    T K         = compute_3d_k(sincIndex);

    std::array<double, lt::size> wh  = lt::createWharmonicTable<double, lt::size>(sincIndex);
    std::array<double, lt::size> whd = lt::createWharmonicDerivativeTable<double, lt::size>(sincIndex);

    cstone::Box<T> box(0, 6, cstone::BoundaryType::open);

    // particle 0 has 4 neighbors
    std::vector<cstone::LocalIndex> neighbors{1, 2, 3, 4};
    unsigned                        neighborsCount = 4;

    std::vector<T> x{1.0, 1.1, 3.2, 1.3, 2.4};
    std::vector<T> y{1.1, 1.2, 1.3, 4.4, 5.5};
    std::vector<T> z{1.2, 2.3, 1.4, 1.5, 1.6};
    std::vector<T> h{5.0, 5.1, 5.2, 5.3, 5.4};
    std::vector<T> m{1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<T> rho{0.014, 0.015, 0.016, 0.017, 0.018};

    std::vector<T> vx{0.010, -0.020, 0.030, -0.040, 0.050};
    std::vector<T> vy{-0.011, 0.021, -0.031, 0.041, -0.051};
    std::vector<T> vz{0.091, -0.081, 0.071, -0.061, 0.055};

    std::vector<T> c{0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<T> p{0.2, 0.3, 0.4, 0.5, 0.6};

    std::vector<T> c11{0.21, 0.27, 0.10, 0.45, 0.46};
    std::vector<T> c12{-0.22, -0.29, -0.11, -0.44, -0.47};
    std::vector<T> c13{-0.23, -0.31, -0.12, -0.43, -0.48};
    std::vector<T> c22{0.24, 0.32, 0.13, 0.42, 0.49};
    std::vector<T> c23{-0.25, -0.33, -0.14, -0.41, -0.50};
    std::vector<T> c33{0.26, 0.34, 0.15, 0.40, 0.51};

    /* distances of particle zero to particle j
     *
     * j = 1   1.10905
     * j = 2   2.21811
     * j = 3   3.32716
     * j = 4   4.63465
     */

    // fill with invalid initial value to make sure that the kernel overwrites it instead of add to it
    T du         = -1;
    T grad_Px    = -1;
    T grad_Py    = -1;
    T grad_Pz    = -1;
    T maxvsignal = -1;

    // compute gradient for for particle 0
    momentumAndEnergyJLoop(0, K, box, neighbors.data(), neighborsCount, x.data(), y.data(), z.data(), vx.data(),
                           vy.data(), vz.data(), h.data(), m.data(), rho.data(), p.data(), c.data(), c11.data(),
                           c12.data(), c13.data(), c22.data(), c23.data(), c33.data(), wh.data(), whd.data(), &grad_Px,
                           &grad_Py, &grad_Pz, &du, &maxvsignal);

    EXPECT_NEAR(grad_Px, 10.483775058458949, 1.3e-7);
    EXPECT_NEAR(grad_Py, 0.090944108814885433, 1.4e-7);
    EXPECT_NEAR(grad_Pz, 10.561586200512671, 2.15e-7);
    EXPECT_NEAR(du, -0.26313730792760737, 1e-8);
    EXPECT_NEAR(maxvsignal, 1.4112466828564341, 1e-10);
}

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
 * @brief Sedov blast simulation data initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"

#include "io/file_utils.hpp"
#include "isim_init.hpp"
#include "sedov_constants.hpp"
#include "grid.hpp"

namespace sphexa
{

template<class Dataset>
void initSedovFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using T = typename Dataset::RealType;

    int    ng0         = 100;
    double r           = constants.at("r1");
    double totalVolume = std::pow(2 * r, 3);
    double hInit       = std::cbrt(3.0 / (4 * M_PI) * ng0 * totalVolume / d.n) * 0.5;

    double mPart  = constants.at("mTotal") / d.n;
    double width  = constants.at("width");
    double width2 = width * width;

    double firstTimeStep = constants.at("firstTimeStep");

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.h.begin(), d.h.end(), hInit);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), 10.0);
    std::fill(d.dt.begin(), d.dt.end(), firstTimeStep);
    std::fill(d.dt_m1.begin(), d.dt_m1.end(), firstTimeStep);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
    d.minDt = firstTimeStep;

    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.count; i++)
    {
        T xi = d.x[i];
        T yi = d.y[i];
        T zi = d.z[i];
        T r2 = xi * xi + yi * yi + zi * zi;

        d.u[i] = constants.at("ener0") * exp(-(r2 / width2)) + constants.at("u0");

        d.x_m1[i] = xi - d.vx[i] * firstTimeStep;
        d.y_m1[i] = yi - d.vy[i] * firstTimeStep;
        d.z_m1[i] = zi - d.vz[i] * firstTimeStep;
    }
}

template<class Dataset>
class SedovGrid : public ISimInitializer<Dataset>
{
    std::map<std::string, double> constants_;

public:
    SedovGrid() { constants_ = sedovConstants(); }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, Dataset& d) const override
    {
        using T = typename Dataset::RealType;
        d.n     = d.side * d.side * d.side;

        auto [first, last] = partitionRange(d.n, rank, numRanks);
        d.count            = last - first;

        resize(d, d.count);

        if (rank == 0)
        {
            std::cout << "Approx: " << d.count * (d.data().size() * 64.) / (8. * 1000. * 1000. * 1000.)
                      << "GB allocated on rank 0." << std::endl;
        }

        T r = constants_.at("r1");
        regularGrid(r, d.side, first, last, d.x, d.y, d.z);
        initSedovFields(d, constants_);

        T halfStep = r / d.side;
        return cstone::Box<T>(-r - halfStep, r - halfStep, true);
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

template<class Dataset>
class SedovGlass : public ISimInitializer<Dataset>
{
    std::string                   glassBlock;
    std::map<std::string, double> constants_;

public:
    SedovGlass(std::string initBlock)
        : glassBlock(initBlock)
    {
        constants_ = sedovConstants();
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, Dataset& d) const override
    {
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        T r           = constants_.at("r1");

        size_t blockSize = 125000lu;
        d.n              = d.side * d.side * d.side * blockSize;

        cstone::Box<T> globalBox(-r, r, true);
        int            multiplicity = d.side;

        auto [keyStart, keyEnd] = partitionRange(cstone::nodeRange<KeyType>(0), rank, numRanks);

        // read the template block
        std::vector<T> xBlock(blockSize), yBlock(blockSize), zBlock(blockSize);
        fileutils::readAscii<T>(glassBlock, d.n, {xBlock.data(), yBlock.data(), zBlock.data()});

        assembleCube<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        d.count = d.x.size();
        resize(d, d.x.size());

        initSedovFields(d, constants_);

        return globalBox;
    }

    const std::map<std::string, double>& constants() const override { return constants_; }
};

} // namespace sphexa

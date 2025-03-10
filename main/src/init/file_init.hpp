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
 * @brief Simulation data initialization from an HDF5 file
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cmath>
#include <map>
#include <numbers>

#include "cstone/sfc/box.hpp"
#include "isim_init.hpp"
#include "../../../physics/Disk/include/star_data.hpp"

namespace sphexa
{

template<class Dataset>
void restoreDataset(IFileReader* reader, Dataset& d)
{
    d.loadOrStoreAttributes(reader);
    d.resize(reader->localNumParticles());

    auto fieldPointers = d.data();
    for (size_t i = 0; i < fieldPointers.size(); ++i)
    {
        if (d.isConserved(i))
        {
            if (reader->rank() == 0) { std::cout << "restoring " << d.fieldNames[i] << std::endl; }
            std::visit([reader, key = d.fieldNames[i]](auto field)
                       { reader->readField(Dataset::prefix + key, field->data()); },
                       fieldPointers[i]);
        }
    }
}

template<class SimulationData>
auto restoreData(IFileReader* reader, SimulationData& simData)
{
    using T = typename SimulationData::RealType;

    cstone::Box<T> box(0, 1);
    box.loadOrStore(reader);

    restoreDataset(reader, simData.hydro);
    restoreDataset(reader, simData.chem);
    simData.star.loadOrStoreAttributes(reader);

    return box;
}

template<class Dataset>
class FileInit : public ISimInitializer<Dataset>
{
    InitSettings settings_;
    std::string  h5_fname;
    int          initStep = -1;

public:
    explicit FileInit(const std::string& fname, int initStep_, IFileReader* reader)
        : h5_fname(fname)
        , initStep(initStep_)
    {
        // Read file attributes and put them in settings_ such that they propagate to the new output after a restart
        readFileAttributes(settings_, h5_fname, reader, false);
    }

    cstone::Box<typename Dataset::RealType> init(int /*rank*/, int numRanks, size_t /*n*/, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        reader->setStep(h5_fname, initStep, FileMode::collective);
        auto box = restoreData(reader, simData);
        reader->closeStep();
        return box;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

InitSettings tdeOrbitConstants()
{
    InitSettings ret{{"beta_impact", 1.0}, {"r0_per_periapsis", 5.0}, {"mTotal", 1.0}, {"r", 1.0}, {"star::m", 1.0}};
    return ret;
}

/*! @brief Compute the position and the velocity of an object in a parabolic orbit.
 * The function assumes that the orbital plane is the xy-plane and the periapsis is at y = 0.
 * @param m_b Mass of the central object (black hole)
 * @param r_periapsis Periapsis distance of the orbit
 * @param r0_per_periapsis Initial distance from the central object measured in periapsis distances
 * @param G Gravitational constant
 * */
auto computeParabolicOrbit(double m_b, double r_periapsis, double r0_per_periapsis, double G)
{
    //! @brief 1 + cos(theta0); while theta is the angle of the object in the orbital (xy) plane, measured from
    //! periapsis counter-clock wise
    const double theta0_factor = 2. / r0_per_periapsis;

    const double r0 = r0_per_periapsis * r_periapsis;

    // I choose the negative solution as a starting point; so the object moves from negative to positive theta.
    const double theta0 = -std::acos(theta0_factor - 1.);

    // Periapsis is chosen to be at y = 0.
    const double x0 = r0 * std::cos(theta0);
    const double y0 = r0 * std::sin(theta0);

    //! @brief The angle of the velocity measured from the orbiter. 0 if in the tangential direction (circular orbit),
    //! counter-clock wise
    const double phi0 = -theta0 / 2.;
    //! @brief The angle of the velocity measured from the central object.
    const double phi0_centre = theta0 + std::numbers::pi_v<double> / 2. + phi0;

    const double v = std::sqrt(G * m_b / r_periapsis * theta0_factor);

    const double vx = v * std::cos(phi0_centre);
    const double vy = v * std::sin(phi0_centre);

    return std::tuple{cstone::Vec3<double>{x0, y0, 0.}, cstone::Vec3<double>{vx, vy, 0.}};
}
void printMap(const auto& map)
{
    for (const auto& elem : map)
    {
        std::cout << elem.first << " " << elem.second << "\n";
    }
}
template<typename Dataset>
class TDEOrbitInit : public ISimInitializer<Dataset>
{
    InitSettings settings_;
    std::string  h5_fname;
    int          initStep = -1;

public:
    explicit TDEOrbitInit(const std::string& filename, int initStep, IFileReader* reader)
        : h5_fname(filename)
        , initStep(initStep)
    {
        BuiltinReader extractor(settings_);
        Dataset       simData;
        simData.hydro.loadOrStoreAttributes(&extractor);
        simData.star.loadOrStoreAttributes(&extractor);

        for (const auto& kv : tdeOrbitConstants())
        {
            settings_[kv.first] = kv.second;
        }

        readFileAttributes(settings_, filename, reader, true);
        printMap(settings_);
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t n, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        BuiltinWriter attributeSetter(settings_);
        simData.hydro.loadOrStoreAttributes(&attributeSetter);
        simData.star.loadOrStoreAttributes(&attributeSetter);

        reader->setStep(h5_fname, initStep, FileMode::collective);
        auto box = restoreData(reader, simData);
        reader->closeStep();

        simData.hydro.relaxationTimescale = 0.;
        // place the center of mass on a parabolic orbit.
        // Compute the tidal radius
        const double mTotal = settings_.at("mTotal");
        const double r      = settings_.at("r");
        const double m_b    = simData.star.m;

        const double r_tidal          = r * std::pow(m_b / mTotal, 1. / 3.);
        const double r_periapsis      = r_tidal / settings_.at("beta_impact");
        const double r0_per_periapsis = settings_.at("r0_per_periapsis");

        const auto [X, V] = computeParabolicOrbit(m_b, r_periapsis, r0_per_periapsis, simData.hydro.g);
        printf("x: %lf, %lf, %lf\n", X[0], X[1], X[2]);
        printf("v: %lf, %lf, %lf\n", V[0], V[1], V[2]);
        // Add X and V to every particle
        auto& d = simData.hydro;
#pragma omp parallel for
        for (size_t i = 0; i < d.x.size(); i++)
        {
            d.x[i] += X[0];
            d.y[i] += X[1];
            d.z[i] += X[2];
            d.vx[i] += V[0];
            d.vy[i] += V[1];
            d.vz[i] += V[2];
        }
        return box;
    }
};

template<class Dataset>
class FileSplitInit : public ISimInitializer<Dataset>
{
    InitSettings settings_;
    std::string  h5_fname;
    int          numSplits;

public:
    explicit FileSplitInit(const std::string& fname, int numSplits_, IFileReader* reader)
        : h5_fname(fname)
        , numSplits(numSplits_)
    {
        if (numSplits < 1)
        {
            throw std::runtime_error("Number of particle splits must be a positive integer. Provided value: " +
                                     std::to_string(numSplits));
        }
        // Read file attributes and put them in constants_ such that they propagate to the new output after a restart
        readFileAttributes(settings_, h5_fname, reader, false);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int, size_t, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        reader->setStep(h5_fname, -1, FileMode::collective);

        size_t numParticlesInFile = reader->localNumParticles();
        size_t numParticlesSplit  = numParticlesInFile * numSplits;

        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;
        cstone::Box<T> box(0, 1);
        box.loadOrStore(reader);

        auto& d = simData.hydro;
        d.loadOrStoreAttributes(reader);

        d.numParticlesGlobal = reader->globalNumParticles() * numSplits;
        d.iteration          = 1;
        d.ttot               = 0.0;
        d.minDt /= (100 * numSplits);
        d.minDt_m1 /= (100 * numSplits);

        d.x.resize(numParticlesSplit);
        d.y.resize(numParticlesSplit);
        d.z.resize(numParticlesSplit);
        d.h.resize(numParticlesSplit);

        std::vector<cstone::LocalIndex> sfcOrder(numParticlesInFile);
        {
            std::vector<T> x0(numParticlesInFile), y0(numParticlesInFile), z0(numParticlesInFile),
                tmp(numParticlesInFile);
            reader->readField("x", x0.data());
            reader->readField("y", y0.data());
            reader->readField("z", z0.data());

            std::vector<KeyType> keys(numParticlesInFile);
            cstone::computeSfcKeys(x0.data(), y0.data(), z0.data(), cstone::sfcKindPointer(keys.data()),
                                   numParticlesInFile, box);
            std::iota(sfcOrder.begin(), sfcOrder.end(), 0);
            cstone::sort_by_key(keys.begin(), keys.end(), sfcOrder.begin());

            auto gatherSwap = [&tmp](auto& v, auto& order)
            {
                cstone::gather<cstone::LocalIndex>(order, v.data(), tmp.data());
                swap(v, tmp);
            };
            gatherSwap(x0, sfcOrder);
            gatherSwap(y0, sfcOrder);
            gatherSwap(z0, sfcOrder);

#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < numParticlesInFile; ++i)
            {
                size_t sIdx = numSplits * i;

                d.x[sIdx] = x0[i];
                d.y[sIdx] = y0[i];
                d.z[sIdx] = z0[i];

                bool isLast   = (i == numParticlesInFile - 1);
                long keyDelta = (isLast ? -(keys[i] - keys[i - 1]) : keys[i + 1] - keys[i]) / (numSplits + isLast);

                for (size_t j = 1; j < numSplits; ++j)
                {
                    auto [ixj, iyj, izj] = cstone::decodeSfc(cstone::sfcKey(keys[i] + j * keyDelta));

                    d.x[sIdx + j] = box.xmin() + (ixj * box.lx()) / cstone::maxCoord<KeyType>{};
                    d.y[sIdx + j] = box.ymin() + (iyj * box.ly()) / cstone::maxCoord<KeyType>{};
                    d.z[sIdx + j] = box.zmin() + (izj * box.lz()) / cstone::maxCoord<KeyType>{};
                }
            }
        }

        auto replicateField = [&sfcOrder, numParticlesInFile, numParticlesSplit,
                               this](IFileReader* reader, const std::string& key, auto& dest, T scale)
        {
            std::vector<T> src(numParticlesInFile), tmp(numParticlesInFile);
            reader->readField(key, src.data());
            cstone::gather<cstone::LocalIndex>(sfcOrder, src.data(), tmp.data());
            swap(src, tmp);

            dest.resize(numParticlesSplit);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < numParticlesInFile; ++i)
            {
                size_t sIdx = numSplits * i;
                std::fill(dest.data() + sIdx, dest.data() + sIdx + numSplits, src[i] * scale);
            }
        };

        d.resize(numParticlesSplit);
        replicateField(reader, "m", d.m, T(1) / numSplits);
        replicateField(reader, "h", d.h, T(1) / std::cbrt(numSplits));
        replicateField(reader, "vx", d.vx, T(1));
        replicateField(reader, "vy", d.vy, T(1));
        replicateField(reader, "vz", d.vz, T(1));
        replicateField(reader, "temp", d.temp, T(1));

        std::fill(d.du_m1.begin(), d.du_m1.end(), 0);
        std::fill(d.rung.begin(), d.rung.end(), 0);
        std::transform(d.vx.begin(), d.vx.end(), d.x_m1.begin(), [dt = d.minDt](auto v_) { return v_ * dt; });
        std::transform(d.vy.begin(), d.vy.end(), d.y_m1.begin(), [dt = d.minDt](auto v_) { return v_ * dt; });
        std::transform(d.vz.begin(), d.vz.end(), d.z_m1.begin(), [dt = d.minDt](auto v_) { return v_ * dt; });

        if (d.isAllocated("alpha"))
        {
            try
            {
                replicateField(reader, "alpha", d.alpha, T(1));
            }
            catch (std::runtime_error&)
            {
                std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
            }
        }

        reader->closeStep();

        return box;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa

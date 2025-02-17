//
// Created by Noah Kubli on 15.02.2025.
//

#pragma once

#include <map>

#include "cstone/sfc/box.hpp"
#include "cstone/tree/continuum.hpp"
#include "sph/eos.hpp"

#include "isim_init.hpp"
#include "early_sync.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

std::map<std::string, double> polytropeConstants()
{
    return {{"gravConstant", 1.}, {"r", 1.},          {"mTotal", 1.}, {"polytropic_exponent", 5. / 3.},
            {"minDt", 1e-4},      {"minDt_m1", 1e-4}, {"mui", 10},    {"ng0", 100},
            {"ngmax", 150}};
}
//
// template<class Dataset>
// void initEvrardFields(Dataset& d, const std::map<std::string, double>& constants)
//{
//    using T = typename Dataset::RealType;
//
//    double mPart = constants.at("mTotal") / d.numParticlesGlobal;
//
//    std::fill(d.m.begin(), d.m.end(), mPart);
//    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
//    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
//    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);
//
//    std::fill(d.vx.begin(), d.vx.end(), 0.0);
//    std::fill(d.vy.begin(), d.vy.end(), 0.0);
//    std::fill(d.vz.begin(), d.vz.end(), 0.0);
//
//    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.0);
//    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.0);
//    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.0);
//
//    generateParticleIDs(d.id);
//
//    auto cv    = sph::idealGasCv(d.muiConst, d.gamma);
//    auto temp0 = constants.at("u0") / cv;
//    std::fill(d.temp.begin(), d.temp.end(), temp0);
//
//    T totalVolume = 4 * M_PI / 3 * std::pow(constants.at("r"), 3);
//    // before the contraction with sqrt(r), the sphere has a constant particle concentration of Ntot / Vtot
//    // after shifting particles towards the center by factor sqrt(r), the local concentration becomes
//    // c(r) = 2/3 * 1/r * Ntot / Vtot
//    T c0 = 2. / 3. * d.numParticlesGlobal / totalVolume;
//
// #pragma omp parallel for schedule(static)
//    for (size_t i = 0; i < d.x.size(); i++)
//    {
//        T radius        = std::sqrt((d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]));
//        T concentration = c0 / radius;
//        d.h[i]          = std::cbrt(3 / (4 * M_PI) * d.ng0 / concentration) * 0.5;
//    }
//}
//
// template<class Vector>
// void contractRhoProfile(Vector& x, Vector& y, Vector& z)
//{
// #pragma omp parallel for schedule(static)
//    for (size_t i = 0; i < x.size(); i++)
//    {
//        auto radius0 = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
//
//        // multiply coordinates by sqrt(r) to generate a density profile ~ 1/r
//        auto contraction = std::sqrt(radius0);
//        x[i] *= contraction;
//        y[i] *= contraction;
//        z[i] *= contraction;
//    }
//}
//
////! @brief Estimate SFC partition of the Evrard sphere based on approximate continuum particle counts
// template<class KeyType, class T>
// std::tuple<KeyType, KeyType> estimateEvrardSfcPartition(size_t cbrtNumPart, const cstone::Box<T>& box, int rank,
//                                                         int numRanks)
//{
//     size_t numParticlesGlobal = 0.523 * cbrtNumPart * cbrtNumPart * cbrtNumPart;
//     T      r                  = box.xmax();
//
//     double   eps        = 2.0 * r / (1u << cstone::maxTreeLevel<KeyType>{});
//     unsigned bucketSize = numParticlesGlobal / (100 * numRanks);
//
//     auto oneOverR = [numParticlesGlobal, r, eps](T x, T y, T z)
//     {
//         T radius = std::max(std::sqrt(norm2(cstone::Vec3<T>{x, y, z})), eps);
//         if (radius > r) { return 0.0; }
//         else { return T(numParticlesGlobal) / (2 * M_PI * radius); }
//     };
//
//     auto [tree, counts] = cstone::computeContinuumCsarray<KeyType>(oneOverR, box, bucketSize);
//     auto a              = cstone::makeSfcAssignment(numRanks, counts, tree.data());
//
//     return {a[rank], a[rank + 1]};
// }

template<class Dataset>
class Polytrope : public ISimInitializer<Dataset>
{
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    explicit Polytrope(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        Dataset d;
        settings_ = buildSettings(d, polytropeConstants(), settingsFile, reader);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int               multi1D      = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> multiplicity = {multi1D, multi1D, multi1D};

        T              r = settings_.at("r");
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::open);

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);
        cutSphere(r, d.x, d.y, d.z);

        size_t numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

//        double polytropic_n, double total_mass, double radial_size, double G
        contractRhoProfileToPolytrope(d.x, d.y, d.z, n_polytrope, total_mass, radial_size, G);
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);

        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        //        initEvrardFields(d, settings_);

        return globalBox;
    }

    void resetConstants(InitSettings newSettings) { settings_ = std::move(newSettings); }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};
} // namespace sphexa

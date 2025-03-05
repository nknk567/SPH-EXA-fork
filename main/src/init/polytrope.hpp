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

#include "polytrope/lane_emden.hpp"
#include "polytrope/interpolator.hpp"
#include "polytrope/integrate.hpp"

namespace sphexa
{

std::map<std::string, double> polytropeConstants()
{
    constexpr double r            = 4.72108762739756E-01;
    constexpr double mTotal       = 1e-6;
    constexpr double gravConstant = 1.0;
    const double     t_relax      = std::sqrt(r * r * r / (gravConstant * mTotal)) / 3.;

    return {{"gravConstant", gravConstant}, // {"r", 0.47},
            {"r", r},
            {"mTotal", mTotal},
            {"polytropic_exponent", 5. / 3.},
            {"minDt", 1e-4},
            {"minDt_m1", 1e-4},
            {"mui", 10},
            {"ng0", 100},
            {"ngmax", 150},
            {"eosChoice", sph::EosType::polytropic},
            {"relaxationTimescale", t_relax}};
}

template<class Dataset>
void initPolytropeFields(Dataset& d, const std::map<std::string, double>& constants)
{
    using T = typename Dataset::RealType;

    double mPart = constants.at("mTotal") / d.numParticlesGlobal;

    std::fill(d.m.begin(), d.m.end(), mPart);
    std::fill(d.du_m1.begin(), d.du_m1.end(), 0.0);
    std::fill(d.mui.begin(), d.mui.end(), d.muiConst);
    std::fill(d.alpha.begin(), d.alpha.end(), d.alphamin);

    std::fill(d.vx.begin(), d.vx.end(), 0.0);
    std::fill(d.vy.begin(), d.vy.end(), 0.0);
    std::fill(d.vz.begin(), d.vz.end(), 0.0);

    std::fill(d.x_m1.begin(), d.x_m1.end(), 0.0);
    std::fill(d.y_m1.begin(), d.y_m1.end(), 0.0);
    std::fill(d.z_m1.begin(), d.z_m1.end(), 0.0);

    generateParticleIDs(d.id);

    //    auto cv    = sph::idealGasCv(d.muiConst, d.gamma);
    //    auto temp0 = constants.at("u0") / cv;
    //    std::fill(d.temp.begin(), d.temp.end(), temp0);

    //    T totalVolume = 4 * M_PI / 3 * std::pow(constants.at("r"), 3);
    // before the contraction with sqrt(r), the sphere has a constant particle concentration of Ntot / Vtot
    // after shifting particles towards the center by factor sqrt(r), the local concentration becomes
    // c(r) = 2/3 * 1/r * Ntot / Vtot
    //    T c0 = 2. / 3. * d.numParticlesGlobal / totalVolume;
    //
    // #pragma omp parallel for schedule(static)
    //    for (size_t i = 0; i < d.x.size(); i++)
    //    {
    //        T radius        = std::sqrt((d.x[i] * d.x[i]) + (d.y[i] * d.y[i]) + (d.z[i] * d.z[i]));
    //        T concentration = c0 / radius;
    //        d.h[i]          = std::cbrt(3 / (4 * M_PI) * d.ng0 / concentration) * 0.5;
    //    }
}

// SET THE MINIMUM STEP SIZE!
auto computeDensityProfile(double polytropic_n, double total_mass, double radial_size, double G)
{
    const double G_code = 1.0;

    LaneEmdenAsymptoticStart asympt{polytropic_n};
    // set step size here
    auto [xi, theta_phi] = integrate_to_zero(LaneEmden{polytropic_n}, asympt(1., 1e-8), 1e-8, 20.0, 1e-2, 1e-12);

    const double xi_1        = xi.back();
    const double dtheta_xi_1 = -theta_phi.back()[1] / (xi.back() * xi.back());

    const double rho_c = get_rho_c(xi_1, dtheta_xi_1, radial_size, total_mass);
    const double K     = get_K(polytropic_n, xi_1, dtheta_xi_1, radial_size, rho_c, G_code);

    std::vector<double> density(theta_phi.size());

    auto theta_phi_to_rho = [rho_c, polytropic_n](const auto& theta_phi)
    { return rho_c * std::pow(theta_phi[0], polytropic_n); };

    std::transform(theta_phi.begin(), theta_phi.end(), density.begin(), theta_phi_to_rho);

    std::vector<double> enclosed_mass(theta_phi.size());
    auto                value_to_encl_m = [rho_c, K, polytropic_n, G_code](/*const auto &xi, */ const auto& theta_phi)
    {
        const double phi = theta_phi[1]; // phi = -xi^2 * dtheta / dxi
        return get_enclosed_mass(polytropic_n, phi, K, rho_c, G_code);
    };
    std::transform(theta_phi.begin(), theta_phi.end(), enclosed_mass.begin(), value_to_encl_m);

    std::vector<double> radius(theta_phi.size());
    const auto          alpha_c = alpha(polytropic_n, rho_c, K, G_code);

    auto xi_to_r = [alpha_c](const auto& xi) { return alpha_c * xi; };

    std::transform(xi.begin(), xi.end(), radius.begin(), xi_to_r);

    return std::tuple{std::move(radius), std::move(density), std::move(enclosed_mass), K};
}

auto getInterpolators(double polytropic_exponent, double total_mass, double radius, double G)
{
    const auto [r, rho, encl_mass, K] = computeDensityProfile(polytropic_exponent, total_mass, radius, G);
    LinearInterpolator rho_interp{r, rho};
    LinearInterpolator m_interp{r, encl_mass};

    // if two subsequent values in encl_mass are the same, it is not sorted
    //    std::vector<size_t> indices(r.size());
    //    std::iota(indices.begin(), indices.end(), size_t(0));
    //    std::sort(indices.begin(), indices.end(), [&encl_mass](size_t i, size_t j) { return encl_mass[i] <
    //    encl_mass[j]; });
    //
    //    std::vector<double> r_sorted(r.size());
    //    std::transform(indices.begin(), indices.end(), r_sorted.begin(), [&r](size_t i) { return r[i]; });
    //    std::vector<double> encl_mass_sorted(r.size());
    //    std::transform(indices.begin(), indices.end(), encl_mass_sorted.begin(),
    //                   [&encl_mass](size_t i) { return encl_mass[i]; });
    for (size_t i = 0; i < 100; i++)
    {
        const double r = i * (0.5) / 100.;
        //        printf("r: %lf\trho: %g\t\tencl mass: %g\n", r, rho_interp(r), m_interp(r));
        printf("%lf\t%g\t%g\n", r, rho_interp(r), m_interp(r));
    }
    LinearInterpolator M_inv_interp{encl_mass, r};

    return std::make_tuple(std::move(rho_interp), std::move(M_inv_interp), K);
}

template<class Vector, typename HType>
void contractRhoProfileToPolytrope(Vector& x, Vector& y, Vector& z, HType& h, double rho_original, double total_mass,
                                   double radius, auto rho_interp, auto M_inv_interp)
{
    const size_t n_part = x.size();

    // The radius in the original distribution that corresponds to the outer edge of the star after stretching.
    //    const double r_original = std::cbrt(3. * total_mass / (4. * M_PI));

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); i++)
    {
        const auto old_radius = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        const auto new_radius = M_inv_interp(4. * M_PI / 3. * old_radius * old_radius * old_radius * rho_original);
        // multiply coordinates by sqrt(r) to generate a density profile ~ 1/r
        //        printf("old radius: %lf\tnew radius: %lf\n", old_radius, new_radius);
        const auto contraction = new_radius / old_radius;
        //        auto contraction = std::sqrt(radius0);
        x[i] *= contraction;
        y[i] *= contraction;
        z[i] *= contraction;

        const double rho = rho_interp(new_radius);
        const size_t ng0 = 100;
        //        h[i]             = 0.5 * std::cbrt(100) * std::cbrt(total_mass / n_part / rho);
        const double m_part = total_mass / x.size();
        h[i]                = 0.5 * std::cbrt(3. * ng0 * m_part / (4. * M_PI * rho));
    }
}

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

        const double n_polytropic = 1. / (settings_.at("polytropic_exponent") - 1.);
        auto [rho_interp, M_inv_interp, K] =
            getInterpolators(n_polytropic, settings_.at("mTotal"), settings_.at("r"), settings_.at("gravConstant"));
        settings_["polytropic_const"] = K;
        //        const auto r_original = std::cbrt(3. / (4. * M_PI) * M_inv_interp.y_values.back());
        printf("rmax interpolator: %lf\n", M_inv_interp.y_values.back());

        T r_original = settings_.at("r");
        //        const T        r_orig = 1.0;
        cstone::Box<T> globalBox(-r_original, r_original, cstone::BoundaryType::open);

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, d.x, d.y, d.z);

        cutSphere(r_original, d.x, d.y, d.z);

        d.h.resize(d.x.size());

        size_t numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);

        const double rho_original =
            settings_.at("mTotal") / (4. / 3. * M_PI * settings_.at("r") * settings_.at("r") * settings_.at("r"));

        contractRhoProfileToPolytrope(d.x, d.y, d.z, d.h, rho_original, settings_.at("mTotal"), settings_.at("r"),
                                      rho_interp, M_inv_interp);
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);

        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initPolytropeFields(d, settings_);

        return globalBox;
    }

    void resetConstants(InitSettings newSettings) { settings_ = std::move(newSettings); }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};
} // namespace sphexa

/*! @file
 * @brief A Propagator class for disks around a central star with generalized volume elements
 *
 * @author Noah Kubli
 */

#pragma once

#include <cstdio>

#include "io/arg_parser.hpp"
#include "ipropagator.hpp"
#include "ve_hydro_nr.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "accretion.hpp"
#include "beta_cooling.hpp"
#include "central_force.hpp"
#include "exchange_star_position.hpp"
#include "star_data.hpp"

namespace sphexa
{

using namespace sph;

/*! @brief VE propagator with disk physics around a central star
 *
 * Inherits the full ve-nr force computation (Newton-Raphson smoothing-length iterations and
 * symmetric interactions). The energy variable is the internal energy "u" instead of "temp",
 * as required by the disk physics functions.
 */
template<bool avClean, class DomainType, class DataType>
class VeDiskProp : public HydroVeNRProp<avClean, DomainType, DataType, "u">
{
protected:
    using Base = HydroVeNRProp<avClean, DomainType, DataType, "u">;
    using Base::groups_;
    using Base::timer;
    using Base::volstd_;

    using T = typename DataType::RealType;

    disk::StarData star;

public:
    VeDiskProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank, settings)
    {
        BuiltinWriter attributeWriter(settings);
        star.loadOrStoreAttributes(&attributeWriter);
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        Base::load(initCond, reader);

        const std::string path = removeModifiers(initCond);
        if (std::filesystem::exists(path))
        {
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);
            star.loadOrStoreAttributes(reader);
            reader->closeStep();
        }
    }

    void save(IFileWriter* writer) override
    {
        Base::save(writer);
        star.loadOrStoreAttributes(writer);
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        Base::computeForces(domain, simData);

        auto&        d     = simData.hydro;
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();

        disk::computeCentralForce(first, last, d, star);
        timer.step("computeCentralForce");

        Base::printNonFinite("disk", d, first, last); // NaN-localizer diagnostic, safe to comment out
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();
        auto&        d     = simData.hydro;

        disk::duTimestep(first, last, d, star);
        timer.step("duTimestep");

        computeTimestep(first, last, d, star.t_du, d.etaAcc * star.t_star);
        timer.step("Timestep");

        computePositions(groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        /* h is converged towards rho * h^3 = eta * m during the force computation; nudging h
         * towards the neighbor count target here would displace it from the converged solution
         * every step. Unresolvable particles are still flagged. */
        bool haveUnconvergedParticles = updateSmoothingLength(groups_.view(), d, /*adjustH*/ false);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }

        if (Base::nrParams_.xmSource == 0)
        {
            /* volume elements of the next step, the smoothed converged volume of this step
             * (SPHYNX-style); placed after the checkpoint output so that restarts see the
             * weights that belong to the dumped positions */
            setVolumeElements(groups_.view(), d, cstone::rawPtr(volstd_), Base::nrParams_.volstdGrowFactor,
                              Base::nrParams_.volstdShrinkFactor);
        }
        timer.step("UpdateQuantities");

        disk::computeAndExchangeStarPosition(star, d.minDt, d.minDt_m1);
        timer.step("computeAndExchangeStarPosition");

        disk::computeAccretionCondition(first, last, d, star);
        timer.step("computeAccretionCondition");

        disk::exchangeAndAccreteOnStar(star, d.minDt_m1, Base::rank_);
        timer.step("exchangeAndAccreteOnStar");

        if (Base::rank_ == 0)
        {
            std::printf("star position: %lf\t%lf\t%lf\n", star.position[0], star.position[1], star.position[2]);
            std::printf("star mass: %lf\n", star.m);
            std::printf("additional pot. erg.: %lf\n", star.potential);
        }
    }
};

} // namespace sphexa

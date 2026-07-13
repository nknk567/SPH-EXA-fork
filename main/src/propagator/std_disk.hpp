
/*! @file
 * @brief A Propagator class for Protoplanetary disk
 *
 * @author Noah Kubli
 */

#pragma once

#include <cstdio>

#include "io/arg_parser.hpp"
#include "ipropagator.hpp"
#include "std_hydro.hpp"
#include "std_hydro_bdt.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph.hpp"

#include "accretion.hpp"
#include "beta_cooling.hpp"
#include "central_force.hpp"
#include "exchange_star_position.hpp"
#include "relaxation.hpp"
#include "star_data.hpp"

namespace sphexa
{

using namespace sph;

template<class DomainType, class DataType>
class HydroPropRelax : public HydroProp<DomainType, DataType>
{
    using Base = HydroProp<DomainType, DataType>;
    struct Params
    {
        double relaxationTimescale{0};
        int    asynchronous_relaxation{1};

        template<class Archive>
        void loadOrStoreAttributes(Archive* ar)
        {
            ar->stepAttribute("relaxationTimescale", &relaxationTimescale, 1);
            ar->stepAttribute("asynchronous_relaxation", &asynchronous_relaxation, 1);
        }
    };
    Params params_;

public:
    HydroPropRelax(std::ostream& output, size_t rank, const InitSettings& settings)
        : HydroProp<DomainType, DataType>(output, rank)
    {
        BuiltinWriter attributeWriter(settings);
        params_.loadOrStoreAttributes(&attributeWriter);
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        HydroProp<DomainType, DataType>::computeForces(domain, simData);
        if (!params_.asynchronous_relaxation)
        {
            relaxSystem(domain.startIndex(), domain.endIndex(), simData.hydro, params_.relaxationTimescale);
        }
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        if (!params_.asynchronous_relaxation) { Base::integrate(domain, simData); }
        else { asynchronous_relaxation(domain, simData); }
    }

    void asynchronous_relaxation(DomainType& domain, DataType& simData)
    {
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();
        auto&        d     = simData.hydro;

        computeTimestep(first, last, d);
        Base::timer.step("Timestep");

        disk::moveToLocalMinimum(first, last, d, domain.box());

        bool haveUnconvergedParticles = updateSmoothingLength(Base::groups_.view(), d);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
        }
        Base::timer.step("UpdateQuantities");
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        Base::load(initCond, reader);
        const std::string path = removeModifiers(initCond);
        if (std::filesystem::exists(path))
        {
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);
            params_.loadOrStoreAttributes(reader);
            reader->closeStep();
        }
    }

    void save(IFileWriter* writer) override
    {
        Base::save(writer);
        params_.loadOrStoreAttributes(writer);
    }
};

template<class DomainType, class DataType>
class DiskProp : public HydroProp<DomainType, DataType>
{
protected:
    using Base = HydroProp<DomainType, DataType>;
    using Base::timer;

    using T = typename DataType::RealType;

    disk::StarData star;

public:
    DiskProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank)
    {
        BuiltinWriter attributeWriter(settings);
        star.loadOrStoreAttributes(&attributeWriter);
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        const std::string path = removeModifiers(initCond);
        if (std::filesystem::exists(path))
        {
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);
            star.loadOrStoreAttributes(reader);
            reader->closeStep();
        }
    }

    void save(IFileWriter* writer) override { star.loadOrStoreAttributes(writer); }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        Base::computeForces(domain, simData);

        auto&        d     = simData.hydro;
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();

        disk::betaCooling(first, last, d, star);
        timer.step("betaCooling");

        disk::computeCentralForce(Base::groups_.view(), d, star);
        timer.step("computeCentralForce");
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

        computePositions(Base::groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});

        bool haveUnconvergedParticles = updateSmoothingLength(Base::groups_.view(), d);
        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        {
            throw std::runtime_error("Neighbor search did not converge\n");
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

template<class DomainType, class DataType>
class DiskBdtProp : public HydroBdtProp<DomainType, DataType>
{
protected:
    using Base = HydroBdtProp<DomainType, DataType>;
    using Base::timer;

    using T = typename DataType::RealType;

    disk::StarData star;

public:
    DiskBdtProp(std::ostream& output, size_t rank, const InitSettings& settings)
        : Base(output, rank, settings)
    {
        BuiltinWriter attributeWriter(settings);
        star.loadOrStoreAttributes(&attributeWriter);
    }

    void load(const std::string& initCond, IFileReader* reader) override
    {
        const std::string path = removeModifiers(initCond);
        if (std::filesystem::exists(path))
        {
            int snapshotIndex = numberAfterSign(initCond, ":");
            reader->setStep(path, snapshotIndex, FileMode::independent);
            star.loadOrStoreAttributes(reader);
            reader->closeStep();
        }
    }

    void save(IFileWriter* writer) override { star.loadOrStoreAttributes(writer); }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        Base::computeForces(domain, simData);

        auto&        d     = simData.hydro;
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();

        //        disk::betaCooling(first, last, d, star);
        //        timer.step("betaCooling");

        // adapt computeCentralForce
        /*
         * for (size_t rung = 1; rung <= highestRung; i++)
         * {
         *     auto rung_group = makeSlicedView(tsGroups_.view(), timestep_.rungRanges[0], timestep_.rungRanges[rung]);
         *     disk::computeCentralForce(rung_group, groupDt_, rung, d, star);
         * }
         */
        disk::computeCentralForceBdt(Base::groups_.view(), Base::activeRungs_, Base::groupDt_, d, star);
        //        timer.step("computeCentralForce");
    }

    void integrate(DomainType& domain, DataType& simData) override
    {
        const size_t first = domain.startIndex();
        const size_t last  = domain.endIndex();
        auto&        d     = simData.hydro;

        // Adapt duTimestep like groupAccTimestep()
        //        disk::duTimestep(activeRungs_, groupDt_, d, star);
        timer.step("duTimestep");

        //        computeTimestep(first, last, d, star.t_du, d.etaAcc * star.t_star);
        //        timer.step("Timestep");

        Base::integrate(domain, simData);

        //        computePositions(Base::groups_.view(), d, domain.box(), d.minDt, {float(d.minDt_m1)});
        //
        //        bool haveUnconvergedParticles = updateSmoothingLength(Base::groups_.view(), d);
        //        if (haveUnconvergedParticles && not d.removeUnconvergedParticles)
        //        {
        //            throw std::runtime_error("Neighbor search did not converge\n");
        //        }
        //        timer.step("UpdateQuantities");
        //
        // The star runs on the smallest timestep.
        //        const int highestRung = Base::activeRung(Base::timestep_.substep, Base::timestep_.numRungs);
        const bool useRung = (Base::timestep_.substep == 0);
        const auto dt_m1   = useRung ? Base::prevTimestep_.dt_m1 : Base::timestep_.dt_m1;
        disk::computeAndExchangeStarPosition(star, Base::timestep_.nextDt, d.minDt_m1);
        //        disk::computeAndExchangeStarPosition(star, d.minDt, d.minDt_m1);
        //        timer.step("computeAndExchangeStarPosition");
        //
        // Accrete all particles that pass through the boundary. Particles that are already to be removed don't count in
        // the star's momentum.
        if (Base::activeRung(Base::timestep_.substep, Base::timestep_.numRungs) == 0)
        {
            disk::computeAccretionCondition(first, last, d, star);
            disk::exchangeAndAccreteOnStar(star, d.minDt_m1, Base::rank_);
            timer.step("exchangeAndAccreteOnStar");
        }
        //        disk::computeAccretionCondition(first, last, d, star);
        //        timer.step("computeAccretionCondition");
        //

        if (Base::rank_ == 0)
        {
            std::printf("star position: %lf\t%lf\t%lf\n", star.position[0], star.position[1], star.position[2]);
            std::printf("star mass: %lf\n", star.m);
            std::printf("additional pot. erg.: %lf\n", star.potential);
        }
    }
};

} // namespace sphexa
